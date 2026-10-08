"""src/services/vector_engine.py — Vector Store Service với Qdrant Hybrid Search (Dense + Sparse BM25).

Kiến trúc Dual-Backend (quyết định chọn theo Plan.md §3):
  - Dense Embedding : sentence-transformers → BAAI/bge-m3
      * 1024 dims, hỗ trợ đa ngữ tiếng Việt natively
      * fastembed KHÔNG có bge-m3 → phải dùng HuggingFace sentence-transformers
  - Sparse Embedding: fastembed → Qdrant/bm25
      * Tokenisation BM25 nhẹ (10 MB), không cần GPU
      * Kết hợp với Dense qua Reciprocal Rank Fusion (RRF) trong Qdrant
"""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.models.block import JSONBlock, SemanticChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — graceful fallback nếu thư viện chưa cài
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers chưa cài — Dense embedding sẽ dùng vector zero.\n"
        "Cài: pip install sentence-transformers"
    )

try:
    from fastembed import SparseTextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    logger.warning(
        "fastembed chưa cài — Sparse BM25 embedding sẽ bị bỏ qua.\n"
        "Cài: pip install fastembed"
    )


class VectorEngineService:
    """Quản lý lưu trữ và tìm kiếm Hybrid (Dense bge-m3 + Sparse BM25) trên Qdrant.

    Dense backend : sentence-transformers ``BAAI/bge-m3`` (1024 dims, đa ngữ tiếng Việt).
    Sparse backend: fastembed ``Qdrant/bm25`` (BM25 tokenisation, 10 MB).
    Fusion        : Reciprocal Rank Fusion (RRF) natively trong Qdrant query engine.
    """

    DEFAULT_COLLECTION = "financial_blocks"
    DENSE_MODEL = "BAAI/bge-m3"       # 1024 dims, multilingual (Vietnamese-native)
    SPARSE_MODEL = "Qdrant/bm25"      # BM25 lexical tokenisation
    DENSE_DIM = 1024                  # Cố định cho bge-m3

    def __init__(
        self,
        url: str | None = None,
        path: str | None = None,
        dense_model_name: str = DENSE_MODEL,
        sparse_model_name: str = SPARSE_MODEL,
        device: str = "cpu",
    ) -> None:
        """Khởi tạo VectorEngineService.

        Args:
            url: URL Qdrant server (ví dụ: ``http://localhost:6333``).
            path: Đường dẫn lưu Qdrant local (ví dụ: ``data/qdrant_storage``).
            dense_model_name: Model sentence-transformers cho dense embedding.
            sparse_model_name: Model fastembed cho sparse BM25.
            device: Thiết bị tính toán cho sentence-transformers (``"cpu"`` hoặc ``"cuda"``).
        """
        if path:
            self.client = QdrantClient(path=path)
        elif url and url != ":memory:":
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(location=":memory:")

        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.device = device

        self._dense_model: Any = None   # SentenceTransformer, lazy-loaded
        self._sparse_model: Any = None  # SparseTextEmbedding, lazy-loaded

    # ------------------------------------------------------------------
    # Lazy-loaded model properties
    # ------------------------------------------------------------------
    @property
    def dense_model(self) -> Any:
        """Lazy-load SentenceTransformer model (chỉ tải lần đầu khi cần)."""
        if self._dense_model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                return None
            logger.info("Đang tải Dense Embedding model: %s (device=%s) …", self.dense_model_name, self.device)
            self._dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
            logger.info("✅ Dense model sẵn sàng: %s", self.dense_model_name)
        return self._dense_model

    @property
    def sparse_model(self) -> Any:
        """Lazy-load SparseTextEmbedding model (BM25)."""
        if self._sparse_model is None:
            if not HAS_FASTEMBED:
                return None
            logger.info("Đang tải Sparse BM25 model: %s …", self.sparse_model_name)
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
            logger.info("✅ Sparse BM25 model sẵn sàng.")
        return self._sparse_model

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def create_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        recreate: bool = False,
    ) -> None:
        """Khởi tạo Qdrant collection với Dense (1024) + Sparse vector config."""
        exists = self.client.collection_exists(collection_name)
        if exists and recreate:
            logger.info("Xoá collection cũ: %s", collection_name)
            self.client.delete_collection(collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
            logger.info("Đã tạo collection: %s (dense=%d dims + sparse BM25)", collection_name, self.DENSE_DIM)

            # Payload indexes để filter cực nhanh theo company/year/page
            for field_name, field_type in [
                ("company", models.PayloadSchemaType.KEYWORD),
                ("year", models.PayloadSchemaType.INTEGER),
                ("page", models.PayloadSchemaType.INTEGER),
                ("is_note", models.PayloadSchemaType.BOOL),
                ("block_type", models.PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Sinh dense vectors từ bge-m3 (1024 dims).

        Fallback: trả về zero vectors nếu model chưa cài.
        """
        if self.dense_model is None:
            logger.warning("Dense model không khả dụng — trả về zero vectors.")
            return [[0.0] * self.DENSE_DIM for _ in texts]

        # sentence-transformers trả về numpy array → chuyển sang list[list[float]]
        embeddings = self.dense_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # Cosine similarity chuẩn hoá L2
        )
        return embeddings.tolist()

    def embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        """Sinh sparse BM25 vectors.

        Fallback: trả về sparse vector rỗng nếu fastembed chưa cài.
        """
        if self.sparse_model is None:
            return [models.SparseVector(indices=[], values=[]) for _ in texts]

        results: list[models.SparseVector] = []
        for res in self.sparse_model.embed(texts):
            results.append(
                models.SparseVector(
                    indices=res.indices.tolist(),
                    values=res.values.tolist(),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_chunks(
        self,
        chunks: list[SemanticChunk | JSONBlock],
        collection_name: str = DEFAULT_COLLECTION,
        batch_size: int = 16,          # Giảm xuống 16 vì bge-m3 nặng hơn
        recreate: bool = False,
    ) -> int:
        """Nạp danh sách SemanticChunks (hoặc JSONBlocks) vào Qdrant Hybrid Collection.

        Args:
            chunks: Danh sách chunks cần index.
            collection_name: Tên collection Qdrant.
            batch_size: Số chunks xử lý mỗi lần (giảm nếu OOM).
            recreate: Nếu True → xoá và tạo lại collection từ đầu.

        Returns:
            Tổng số points đã index thành công.
        """
        if not chunks:
            return 0

        self.create_collection(collection_name=collection_name, recreate=recreate)

        total_ingested = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
            batch = chunks[i : i + batch_size]
            contents = [item.content for item in batch]

            logger.info(
                "Embedding batch %d/%d (%d chunks) …",
                batch_idx + 1,
                total_batches,
                len(batch),
            )
            dense_vectors = self.embed_dense(contents)
            sparse_vectors = self.embed_sparse(contents)

            points: list[models.PointStruct] = []
            for item, d_vec, s_vec in zip(batch, dense_vectors, sparse_vectors):
                item_id = getattr(item, "chunk_id", getattr(item, "block_id", str(uuid.uuid4())))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item_id))

                payload = self._build_payload(item)
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={"dense": d_vec, "sparse": s_vec},
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)
            total_ingested += len(points)

        logger.info("✅ Đã index %d points vào collection '%s'.", total_ingested, collection_name)
        return total_ingested

    # Alias tương thích
    def ingest_blocks(self, blocks: list[JSONBlock], **kwargs) -> int:
        return self.ingest_chunks(blocks, **kwargs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        is_note: bool | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Tìm kiếm kết hợp Hybrid (Dense bge-m3 + Sparse BM25) với RRF Fusion.

        Args:
            query: Câu truy vấn tiếng Việt hoặc tiếng Anh.
            company: Lọc theo mã cổ phiếu (None = không lọc).
            year: Lọc theo năm tài chính (None = không lọc).
            is_note: Lọc theo loại block (True = thuyết minh, None = tất cả).
            collection_name: Tên collection Qdrant.
            limit: Số kết quả tối đa trả về.

        Returns:
            Danh sách dict payload Qdrant kèm ``score`` RRF.
        """
        dense_vec = self.embed_dense([query])[0]
        sparse_vec = self.embed_sparse([query])[0]

        # Metadata filter
        must_conditions: list[models.FieldCondition] = []
        if company:
            must_conditions.append(
                models.FieldCondition(key="company", match=models.MatchValue(value=company.upper()))
            )
        if year:
            must_conditions.append(
                models.FieldCondition(key="year", match=models.MatchValue(value=year))
            )
        if is_note is not None:
            must_conditions.append(
                models.FieldCondition(key="is_note", match=models.MatchValue(value=is_note))
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Prefetch song song Dense + Sparse → RRF fusion
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                filter=query_filter,
                limit=limit * 3,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                filter=query_filter,
                limit=limit * 3,
            ),
        ]

        results = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        hits: list[dict[str, Any]] = []
        for point in results.points:
            p_dict = dict(point.payload or {})
            p_dict["score"] = point.score
            hits.append(p_dict)

        return hits

    # Alias tương thích
    def search(self, query: str, **kwargs) -> list[dict[str, Any]]:
        return self.hybrid_search(query, **kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_payload(self, item: SemanticChunk | JSONBlock) -> dict[str, Any]:
        """Chuẩn bị dict payload bảo toàn toạ độ BBox và metadata đầy đủ."""
        if isinstance(item, SemanticChunk):
            return {
                "chunk_id": item.chunk_id,
                "content": item.content,
                "raw_content": item.raw_content,
                "page": item.page,
                "bbox": item.bbox,
                "block_type": item.block_type,
                "source_block_ids": item.source_block_ids,
                "company": item.company,
                "year": item.year,
                "is_note": item.is_note,
            }
        # JSONBlock
        return {
            "chunk_id": item.block_id,
            "content": item.content,
            "raw_content": item.content,
            "page": item.page,
            "bbox": item.bbox,
            "block_type": item.block_type,
            "source_block_ids": [item.block_id],
            "company": item.metadata.company,
            "year": item.metadata.year,
            "is_note": item.metadata.is_note,
        }
