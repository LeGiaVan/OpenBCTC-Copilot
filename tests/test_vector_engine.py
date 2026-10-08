"""tests/test_vector_engine.py — Kiểm tra Hybrid Search (Dense bge-m3 + Sparse BM25) trên Qdrant.

Chiến lược test:
  - Mock `SentenceTransformer.encode()` trả về numpy array zeros (1024 dims) → không cần tải model thật.
  - Mock `SparseTextEmbedding.embed()` trả về sparse vector giả → không cần fastembed download.
  - Dùng Qdrant in-memory → không cần server.
  - 2 test integration với data thật (mark slow) có thể skip khi CI không có model.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.parsers.json_block_loader import JSONBlockLoader
from src.services.chunker import LayoutAwareChunker
from src.services.vector_engine import VectorEngineService


# ---------------------------------------------------------------------------
# Helpers: tạo fake numpy embeddings đúng chuẩn của sentence-transformers
# ---------------------------------------------------------------------------
DENSE_DIM = VectorEngineService.DENSE_DIM  # 1024


def _make_fake_dense(texts: list[str], **kwargs) -> np.ndarray:
    """Trả về numpy array zeros đúng shape mà sentence-transformers.encode() sẽ trả về."""
    n = len(texts)
    arr = np.zeros((n, DENSE_DIM), dtype=np.float32)
    # Thêm chút nhiễu để điểm cosine không hoàn toàn bằng nhau
    for i in range(n):
        arr[i, i % DENSE_DIM] = float(i + 1) * 0.01
    return arr


def _make_sparse_result():
    """Trả về fake SparseEmbedding object với indices/values giống fastembed output."""
    res = MagicMock()
    res.indices = np.array([0, 5, 100], dtype=np.int32)
    res.values = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    return res


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_dense_model():
    """Mock SentenceTransformer — không tải model thật."""
    m = MagicMock()
    m.encode.side_effect = _make_fake_dense
    return m


@pytest.fixture
def mock_sparse_model():
    """Mock SparseTextEmbedding — không download fastembed."""
    m = MagicMock()
    m.embed.side_effect = lambda texts: [_make_sparse_result() for _ in texts]
    return m


@pytest.fixture
def vector_service(mock_dense_model, mock_sparse_model):
    """VectorEngineService Qdrant in-memory với model đã được mock."""
    svc = VectorEngineService(url=":memory:")
    svc._dense_model = mock_dense_model    # Inject mock thay vì lazy-load thật
    svc._sparse_model = mock_sparse_model
    return svc


# ---------------------------------------------------------------------------
# Tests: embed_dense
# ---------------------------------------------------------------------------
class TestEmbedDense:
    def test_returns_correct_shape(self, vector_service):
        texts = ["Doanh thu thuần 2025", "Lợi nhuận sau thuế"]
        vecs = vector_service.embed_dense(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == DENSE_DIM  # bge-m3 = 1024 dims

    def test_returns_list_of_floats(self, vector_service):
        vecs = vector_service.embed_dense(["Test"])
        assert all(isinstance(v, float) for v in vecs[0])

    def test_fallback_zero_when_no_model(self):
        svc = VectorEngineService(url=":memory:")
        svc._dense_model = None  # Giả lập chưa cài sentence-transformers
        with patch("src.services.vector_engine.HAS_SENTENCE_TRANSFORMERS", False):
            vecs = svc.embed_dense(["Test"])
        assert vecs == [[0.0] * DENSE_DIM]


# ---------------------------------------------------------------------------
# Tests: embed_sparse
# ---------------------------------------------------------------------------
class TestEmbedSparse:
    def test_returns_sparse_vectors(self, vector_service):
        from qdrant_client.http import models as qmodels
        svecs = vector_service.embed_sparse(["Báo cáo tài chính"])
        assert len(svecs) == 1
        assert isinstance(svecs[0], qmodels.SparseVector)
        assert len(svecs[0].indices) == 3  # Từ mock

    def test_fallback_empty_when_no_model(self):
        from qdrant_client.http import models as qmodels
        svc = VectorEngineService(url=":memory:")
        svc._sparse_model = None
        with patch("src.services.vector_engine.HAS_FASTEMBED", False):
            svecs = svc.embed_sparse(["Test"])
        assert svecs[0].indices == []
        assert svecs[0].values == []


# ---------------------------------------------------------------------------
# Tests: create_collection
# ---------------------------------------------------------------------------
class TestCreateCollection:
    def test_creates_collection_with_dense_and_sparse(self, vector_service):
        vector_service.create_collection("test_coll")
        assert vector_service.client.collection_exists("test_coll")

    def test_recreate_deletes_and_recreates(self, vector_service):
        vector_service.create_collection("test_coll")
        vector_service.create_collection("test_coll", recreate=True)
        assert vector_service.client.collection_exists("test_coll")

    def test_idempotent_create(self, vector_service):
        vector_service.create_collection("test_coll")
        vector_service.create_collection("test_coll")  # Không nên crash
        assert vector_service.client.collection_exists("test_coll")


# ---------------------------------------------------------------------------
# Tests: ingest_chunks
# ---------------------------------------------------------------------------
class TestIngestChunks:
    def _make_chunk(self, idx: int):
        from src.models.block import SemanticChunk
        return SemanticChunk(
            chunk_id=f"p{idx}_c1_text",
            content=f"[VNM 2025 | Trang {idx}]\nDoanh thu thuần đạt {idx * 100} tỷ đồng.",
            raw_content=f"Doanh thu thuần đạt {idx * 100} tỷ đồng.",
            page=idx,
            bbox=[0.1, 0.05, 0.5 + idx * 0.01, 0.9],
            block_type="text",
            source_block_ids=[f"p{idx}_txt_1"],
            company="VNM",
            year=2025,
            is_note=False,
        )

    def test_ingest_returns_correct_count(self, vector_service):
        chunks = [self._make_chunk(i) for i in range(1, 4)]
        count = vector_service.ingest_chunks(chunks)
        assert count == 3

    def test_ingest_empty_returns_zero(self, vector_service):
        assert vector_service.ingest_chunks([]) == 0

    def test_ingest_preserves_payload(self, vector_service):
        chunk = self._make_chunk(5)
        vector_service.ingest_chunks([chunk])
        hits = vector_service.hybrid_search("Doanh thu", company="VNM", year=2025, limit=1)
        assert len(hits) == 1
        assert hits[0]["company"] == "VNM"
        assert hits[0]["page"] == 5
        assert hits[0]["bbox"] == chunk.bbox

    def test_ingest_with_recreate(self, vector_service):
        chunks = [self._make_chunk(1)]
        vector_service.ingest_chunks(chunks)
        # Ingest lại với recreate=True → collection được tạo lại
        count = vector_service.ingest_chunks(chunks, recreate=True)
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: hybrid_search filters
# ---------------------------------------------------------------------------
class TestHybridSearch:
    def _ingest_test_data(self, vector_service):
        from src.models.block import SemanticChunk
        chunks = [
            SemanticChunk(
                chunk_id="vnm_p12_text",
                content="[VNM 2025 | Trang 12]\nDoanh thu thuần hợp nhất.",
                raw_content="Doanh thu thuần hợp nhất.",
                page=12, bbox=[0.1, 0.1, 0.5, 0.9],
                block_type="text", source_block_ids=["p12_txt_1"],
                company="VNM", year=2025, is_note=False,
            ),
            SemanticChunk(
                chunk_id="vnm_p24_note",
                content="[VNM 2025 | Trang 24]\nThuyết minh 5: Phải thu ngắn hạn.",
                raw_content="Thuyết minh 5: Phải thu ngắn hạn.",
                page=24, bbox=[0.2, 0.1, 0.6, 0.9],
                block_type="text", source_block_ids=["p24_txt_5"],
                company="VNM", year=2025, is_note=True,
            ),
        ]
        vector_service.ingest_chunks(chunks)
        return chunks

    def test_search_returns_hits(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Doanh thu", limit=5)
        assert len(hits) > 0

    def test_score_in_hits(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Tài chính")
        for h in hits:
            assert "score" in h

    def test_hits_have_bbox(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Tài chính", limit=5)
        for h in hits:
            assert "bbox" in h
            assert len(h["bbox"]) == 4

    def test_company_filter(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Tài chính", company="VNM", limit=5)
        for h in hits:
            assert h["company"] == "VNM"

    def test_is_note_filter_true(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Thuyết minh", company="VNM", is_note=True, limit=5)
        for h in hits:
            assert h["is_note"] is True

    def test_is_note_filter_false(self, vector_service):
        self._ingest_test_data(vector_service)
        hits = vector_service.hybrid_search("Doanh thu", company="VNM", is_note=False, limit=5)
        for h in hits:
            assert h["is_note"] is False


# ---------------------------------------------------------------------------
# Integration tests với data thật (cần sentence-transformers + fastembed cài)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_integration_real_model_ingest_and_search():
    """Test tích hợp thật: tải bge-m3 + BM25 và nạp data thực tế VNM 2025.

    Chỉ chạy khi có --runslow flag hoặc CI có model cache.
    Bỏ qua nếu chưa cài sentence-transformers hoặc không có data.
    """
    pytest.importorskip("sentence_transformers", reason="sentence-transformers chưa cài")
    sample_file = Path("data/VNM_2025/cache/notes/page_13.json")
    if not sample_file.exists():
        pytest.skip("Chưa có file data/VNM_2025/cache/notes/page_13.json")

    blocks = JSONBlockLoader.load_from_file(sample_file)
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_blocks(blocks)
    assert len(chunks) > 0

    svc = VectorEngineService(url=":memory:")
    count = svc.ingest_chunks(chunks)
    assert count == len(chunks)

    hits = svc.hybrid_search(
        query="thuyết minh báo cáo tài chính riêng",
        company="VNM", year=2025, limit=3,
    )
    assert len(hits) > 0
    assert hits[0]["company"] == "VNM"
    assert "bbox" in hits[0]
    assert len(hits[0]["bbox"]) == 4
