"""tests/test_retriever.py — Unit tests cho HybridRetriever (Phase 2.2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.models.citation import CitationWithBBox
from src.services.retriever import HybridRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_HIT = {
    "chunk_id": "p12_c1_text",
    "content": "[VNM 2025 | Trang 12]\nDoanh thu thuần hợp nhất đạt 14.352 tỷ đồng...",
    "raw_content": "Doanh thu thuần hợp nhất đạt 14.352 tỷ đồng, tăng 8.2% so với cùng kỳ.",
    "page": 12,
    "bbox": [0.12, 0.10, 0.45, 0.85],
    "block_type": "text",
    "source_block_ids": ["p12_mineru_txt_2"],
    "company": "VNM",
    "year": 2025,
    "is_note": False,
    "score": 0.87,
}

SAMPLE_HIT_NOTE = {
    "chunk_id": "p24_c3_text",
    "content": "[VNM 2025 | Trang 24]\nThuyết minh 5: Các khoản phải thu...",
    "raw_content": "Thuyết minh 5: Các khoản phải thu ngắn hạn gồm...",
    "page": 24,
    "bbox": [0.20, 0.10, 0.70, 0.90],
    "block_type": "text",
    "source_block_ids": ["p24_mineru_txt_5"],
    "company": "VNM",
    "year": 2025,
    "is_note": True,
    "score": 0.72,
}


@pytest.fixture
def mock_vec_svc():
    """Mock VectorEngineService trả về 2 hit mẫu."""
    svc = MagicMock()
    svc.DEFAULT_COLLECTION = "financial_blocks"
    svc.hybrid_search.return_value = [SAMPLE_HIT, SAMPLE_HIT_NOTE]
    return svc


@pytest.fixture
def retriever(mock_vec_svc):
    """HybridRetriever không dùng reranker (use_reranker=False) để test nhanh."""
    return HybridRetriever(vec_svc=mock_vec_svc, use_reranker=False, top_k=5)


# ---------------------------------------------------------------------------
# Tests: _hit_to_result
# ---------------------------------------------------------------------------
class TestHitToResult:
    def test_basic_conversion(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT)
        assert result.chunk_id == "p12_c1_text"
        assert result.page == 12
        assert result.company == "VNM"
        assert result.year == 2025
        assert result.is_note is False
        assert result.hybrid_score == pytest.approx(0.87)
        assert result.final_score == pytest.approx(0.87)
        assert result.rerank_score is None

    def test_note_conversion(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT_NOTE)
        assert result.is_note is True
        assert result.page == 24
        assert result.bbox == [0.20, 0.10, 0.70, 0.90]

    def test_missing_fields_fallback(self, retriever):
        hit = {"score": 0.5}  # Thiếu hầu hết field
        result = retriever._hit_to_result(hit)
        assert result.chunk_id == ""
        assert result.page == 0
        assert result.bbox == [0.0, 0.0, 1.0, 1.0]  # Default bbox


# ---------------------------------------------------------------------------
# Tests: retrieve (no reranker)
# ---------------------------------------------------------------------------
class TestRetrieve:
    def test_returns_results(self, retriever, mock_vec_svc):
        results = retriever.retrieve(query="Doanh thu thuần 2025", company="VNM", year=2025)
        assert len(results) == 2
        mock_vec_svc.hybrid_search.assert_called_once()

    def test_respects_top_k_override(self, retriever, mock_vec_svc):
        results = retriever.retrieve(query="Test", top_k=1)
        assert len(results) == 1  # Giới hạn top_k=1

    def test_empty_hits_returns_empty(self, retriever, mock_vec_svc):
        mock_vec_svc.hybrid_search.return_value = []
        results = retriever.retrieve(query="Test trống")
        assert results == []

    def test_passes_filters_to_hybrid_search(self, retriever, mock_vec_svc):
        retriever.retrieve(query="Test", company="VNM", year=2025, is_note=True)
        call_kwargs = mock_vec_svc.hybrid_search.call_args.kwargs
        assert call_kwargs["company"] == "VNM"
        assert call_kwargs["year"] == 2025
        assert call_kwargs["is_note"] is True

    def test_results_sorted_by_score_descending(self, retriever, mock_vec_svc):
        # SAMPLE_HIT có score 0.87 > SAMPLE_HIT_NOTE score 0.72
        results = retriever.retrieve(query="Test")
        assert results[0].hybrid_score >= results[1].hybrid_score


# ---------------------------------------------------------------------------
# Tests: retrieve_with_citations
# ---------------------------------------------------------------------------
class TestRetrieveWithCitations:
    def test_returns_tuple(self, retriever):
        results, citations = retriever.retrieve_with_citations("Doanh thu")
        assert isinstance(results, list)
        assert isinstance(citations, list)

    def test_citations_count_matches_results(self, retriever):
        results, citations = retriever.retrieve_with_citations("Doanh thu")
        assert len(citations) == len(results)

    def test_citation_ids_sequential(self, retriever):
        _, citations = retriever.retrieve_with_citations("Doanh thu")
        for i, cit in enumerate(citations, start=1):
            assert cit.citation_id == f"cite_{i}"

    def test_citation_is_citationwithbbox(self, retriever):
        _, citations = retriever.retrieve_with_citations("Doanh thu")
        for cit in citations:
            assert isinstance(cit, CitationWithBBox)

    def test_note_citation_source_type(self, retriever):
        _, citations = retriever.retrieve_with_citations("Thuyết minh")
        note_citations = [c for c in citations if c.source_type == "note"]
        statement_citations = [c for c in citations if c.source_type == "statement"]
        assert len(note_citations) == 1  # Từ SAMPLE_HIT_NOTE
        assert len(statement_citations) == 1  # Từ SAMPLE_HIT


# ---------------------------------------------------------------------------
# Tests: to_citation on RetrievalResult
# ---------------------------------------------------------------------------
class TestRetrievalResultToCitation:
    def test_snippet_truncation(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT)
        citation = result.to_citation("cite_1", snippet_len=10)
        assert len(citation.snippet) <= 13  # 10 chars + "…"
        assert citation.snippet.endswith("…")

    def test_full_snippet_no_ellipsis(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT_NOTE)
        citation = result.to_citation("cite_2", snippet_len=9999)
        assert not citation.snippet.endswith("…")

    def test_citation_page_and_bbox(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT)
        citation = result.to_citation("cite_1")
        assert citation.page == 12
        assert citation.bbox == [0.12, 0.10, 0.45, 0.85]

    def test_block_id_from_source_block_ids(self, retriever):
        result = retriever._hit_to_result(SAMPLE_HIT)
        citation = result.to_citation("cite_1")
        assert citation.block_id == "p12_mineru_txt_2"


# ---------------------------------------------------------------------------
# Tests: reranker fallback
# ---------------------------------------------------------------------------
class TestRerankerFallback:
    def test_reranker_disabled_uses_hybrid_score(self, mock_vec_svc):
        retriever = HybridRetriever(mock_vec_svc, use_reranker=False, top_k=5)
        results = retriever.retrieve("Test")
        for r in results:
            assert r.final_score == r.hybrid_score
            assert r.rerank_score is None

    def test_reranker_error_falls_back_gracefully(self, mock_vec_svc):
        """Nếu reranker raise exception → fallback về hybrid_score không crash."""
        with patch("src.services.retriever.HAS_RERANKER", True):
            retriever = HybridRetriever(mock_vec_svc, use_reranker=True, top_k=5)
            mock_reranker = MagicMock()
            mock_reranker.rerank.side_effect = RuntimeError("Model not loaded")
            retriever._reranker = mock_reranker

            results = retriever.retrieve("Test")
            assert len(results) > 0
            for r in results:
                assert r.final_score == r.hybrid_score  # Fallback về hybrid score
