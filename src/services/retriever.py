"""src/services/retriever.py — Semantic Hybrid Retriever với BGE Reranker (Phase 2.2).

Pipeline truy xuất 2 bước:
  1. Hybrid Search (Dense + Sparse BM25) trên Qdrant → lấy Top-K * 3 candidates.
  2. BGE-Reranker-Large cross-encoder → tái chấm điểm và lấy Top-K cuối cùng.

Thiết kế Lean: Retriever là Fat Service thuần, không phụ thuộc LangGraph hay bất kỳ
agentic framework nào. LangGraph node chỉ gọi `retriever.retrieve(query, ...)`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.models.citation import CitationWithBBox
from src.services.vector_engine import VectorEngineService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import reranker để không crash khi fastembed chưa cài
# ---------------------------------------------------------------------------
try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    logger.warning(
        "fastembed TextCrossEncoder không khả dụng — Reranker sẽ bị bỏ qua. "
        "Cài: pip install fastembed"
    )


# ---------------------------------------------------------------------------
# Data class kết quả truy xuất
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """Đơn vị kết quả trả về từ Retriever sau khi đã qua Reranker."""

    chunk_id: str
    content: str
    raw_content: str
    page: int
    bbox: list[float]                    # [ymin, xmin, ymax, xmax]
    block_type: str                      # "text" | "table"
    source_block_ids: list[str]
    company: str
    year: int
    is_note: bool
    hybrid_score: float = 0.0           # Điểm RRF từ Qdrant
    rerank_score: float | None = None   # Điểm cross-encoder reranker (None nếu không có)
    final_score: float = 0.0            # Điểm cuối dùng để sắp xếp

    def to_citation(self, citation_id: str, snippet_len: int = 300) -> CitationWithBBox:
        """Chuyển đổi thành CitationWithBBox cho Visual Grounding."""
        source_type = "note" if self.is_note else "statement"
        snippet = self.raw_content[:snippet_len].rstrip()
        if len(self.raw_content) > snippet_len:
            snippet += "…"
        return CitationWithBBox(
            citation_id=citation_id,
            block_id=self.source_block_ids[0] if self.source_block_ids else self.chunk_id,
            source_type=source_type,
            page=self.page,
            bbox=self.bbox,
            snippet=snippet,
            company=self.company,
            year=self.year,
            confidence=round(self.final_score, 4),
        )


# ---------------------------------------------------------------------------
# Retriever Service
# ---------------------------------------------------------------------------
class HybridRetriever:
    """2-stage Retriever: Qdrant Hybrid Search → BGE Reranker.

    Args:
        vec_svc: VectorEngineService đã khởi tạo sẵn.
        reranker_model: Tên model reranker của fastembed
            (mặc định: ``BAAI/bge-reranker-base``).
            Dùng ``BAAI/bge-reranker-large`` nếu máy đủ RAM.
        top_k: Số kết quả cuối cần trả về sau rerank.
        prefetch_multiplier: Số nhân để lấy candidates từ Qdrant trước rerank
            (ví dụ: top_k=5, multiplier=4 → prefetch 20 candidates).
        use_reranker: Bật/tắt bước rerank. Tắt thì dùng điểm Qdrant RRF trực tiếp.
    """

    DEFAULT_RERANKER = "BAAI/bge-reranker-base"

    def __init__(
        self,
        vec_svc: VectorEngineService,
        reranker_model: str = DEFAULT_RERANKER,
        top_k: int = 5,
        prefetch_multiplier: int = 4,
        use_reranker: bool = True,
    ) -> None:
        self.vec_svc = vec_svc
        self.top_k = top_k
        self.prefetch_multiplier = prefetch_multiplier
        self.use_reranker = use_reranker and HAS_RERANKER
        self._reranker: Any = None
        self._reranker_model = reranker_model

        if use_reranker and not HAS_RERANKER:
            logger.warning("use_reranker=True nhưng fastembed không cài → tự động tắt reranker.")

    @property
    def reranker(self) -> Any:
        """Lazy-load reranker model (chỉ tải lần đầu khi cần)."""
        if self._reranker is None and self.use_reranker:
            logger.info("Đang tải BGE Reranker: %s …", self._reranker_model)
            self._reranker = TextCrossEncoder(model_name=self._reranker_model)
            logger.info("✅ Reranker đã sẵn sàng.")
        return self._reranker

    # ------------------------------------------------------------------
    # Core retrieve method
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        is_note: bool | None = None,
        collection_name: str = VectorEngineService.DEFAULT_COLLECTION,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Truy xuất ngữ cảnh liên quan nhất cho một câu truy vấn.

        Bước 1 — Hybrid Search:
            Lấy ``top_k * prefetch_multiplier`` candidates từ Qdrant (Dense + Sparse RRF).

        Bước 2 — Rerank:
            Nếu ``use_reranker`` và fastembed khả dụng: chạy cross-encoder BGE Reranker
            để tái chấm điểm rồi sắp xếp lại.

        Args:
            query: Câu hỏi hoặc đoạn truy vấn.
            company: Lọc theo mã cổ phiếu (None = không lọc).
            year: Lọc theo năm tài chính (None = không lọc).
            is_note: Lọc theo loại block thuyết minh (None = cả 2 loại).
            collection_name: Tên collection Qdrant.
            top_k: Override số kết quả trả về (None = dùng self.top_k).

        Returns:
            Danh sách ``RetrievalResult`` đã sắp xếp theo ``final_score`` giảm dần.
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        prefetch_limit = effective_top_k * self.prefetch_multiplier

        # --- Stage 1: Hybrid Search ---
        raw_hits: list[dict[str, Any]] = self.vec_svc.hybrid_search(
            query=query,
            company=company,
            year=year,
            is_note=is_note,
            collection_name=collection_name,
            limit=prefetch_limit,
        )

        if not raw_hits:
            return []

        candidates = [self._hit_to_result(h) for h in raw_hits]

        # --- Stage 2: Rerank ---
        if self.use_reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates)

        return candidates[:effective_top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hit_to_result(self, hit: dict[str, Any]) -> RetrievalResult:
        """Chuyển đổi dict payload Qdrant thành RetrievalResult."""
        score = hit.get("score", 0.0)
        return RetrievalResult(
            chunk_id=hit.get("chunk_id", ""),
            content=hit.get("content", ""),
            raw_content=hit.get("raw_content", hit.get("content", "")),
            page=int(hit.get("page", 0)),
            bbox=hit.get("bbox", [0.0, 0.0, 1.0, 1.0]),
            block_type=hit.get("block_type", "text"),
            source_block_ids=hit.get("source_block_ids", []),
            company=hit.get("company", ""),
            year=int(hit.get("year", 0)),
            is_note=bool(hit.get("is_note", False)),
            hybrid_score=score,
            rerank_score=None,
            final_score=score,
        )

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Tái chấm điểm candidates bằng BGE Cross-Encoder Reranker."""
        try:
            passages = [c.raw_content for c in candidates]
            scores = list(self.reranker.rerank(query, passages))

            for i, (cand, score_obj) in enumerate(zip(candidates, scores)):
                # fastembed trả về object có attribute .score hoặc float
                rerank_val = score_obj.score if hasattr(score_obj, "score") else float(score_obj)
                cand.rerank_score = round(rerank_val, 6)
                cand.final_score = rerank_val

            candidates.sort(key=lambda c: c.final_score, reverse=True)
            logger.debug("Rerank hoàn tất: %d candidates.", len(candidates))

        except Exception as exc:
            logger.warning("Reranker gặp lỗi (%s) — fallback dùng hybrid_score.", exc)
            for c in candidates:
                c.final_score = c.hybrid_score
            candidates.sort(key=lambda c: c.final_score, reverse=True)

        return candidates

    def retrieve_with_citations(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        is_note: bool | None = None,
        top_k: int | None = None,
        snippet_len: int = 300,
    ) -> tuple[list[RetrievalResult], list[CitationWithBBox]]:
        """Wrapper tiện lợi: truy xuất và tự động tạo danh sách Citation kèm BBox.

        Returns:
            Tuple (results, citations) — citations đã được đánh số cite_1, cite_2, …
        """
        results = self.retrieve(
            query=query,
            company=company,
            year=year,
            is_note=is_note,
            top_k=top_k,
        )
        citations = [
            r.to_citation(citation_id=f"cite_{i + 1}", snippet_len=snippet_len)
            for i, r in enumerate(results)
        ]
        return results, citations
