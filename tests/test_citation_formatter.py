"""tests/test_citation_formatter.py — Unit tests cho CitationFormatter (Phase 2.3)."""

from __future__ import annotations

import pytest
from src.models.citation import CitationWithBBox
from src.services.citation_formatter import CitationFormatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_citation(
    citation_id: str = "cite_1",
    page: int = 12,
    bbox: list[float] | None = None,
    source_type: str = "note",
    snippet: str = "Doanh thu thuần năm 2025 đạt 14.352 tỷ đồng.",
    confidence: float = 0.95,
    company: str = "VNM",
    year: int = 2025,
) -> CitationWithBBox:
    return CitationWithBBox(
        citation_id=citation_id,
        block_id=f"p{page}_txt_1",
        source_type=source_type,
        page=page,
        bbox=bbox if bbox is not None else [0.10, 0.05, 0.50, 0.90],
        snippet=snippet,
        company=company,
        year=year,
        confidence=confidence,
    )


@pytest.fixture
def formatter():
    return CitationFormatter()


@pytest.fixture
def valid_citations():
    return [
        make_citation("cite_1", page=12, source_type="statement"),
        make_citation("cite_2", page=24, source_type="note"),
    ]


# ---------------------------------------------------------------------------
# Tests: validate_citation
# ---------------------------------------------------------------------------
class TestValidateCitation:
    def test_valid_citation_passes(self, formatter):
        c = make_citation(bbox=[0.10, 0.05, 0.50, 0.90])
        assert formatter.validate_citation(c) is True

    def test_invalid_bbox_length(self, formatter):
        c = make_citation(bbox=[0.1, 0.2, 0.5])  # Chỉ có 3 phần tử
        assert formatter.validate_citation(c) is False

    def test_bbox_out_of_range(self, formatter):
        c = make_citation(bbox=[-0.1, 0.0, 1.5, 1.0])  # ymin < 0, ymax > 1
        assert formatter.validate_citation(c) is False

    def test_bbox_zero_height(self, formatter):
        c = make_citation(bbox=[0.5, 0.1, 0.5, 0.9])  # ymin == ymax → area = 0
        assert formatter.validate_citation(c) is False

    def test_bbox_inverted(self, formatter):
        c = make_citation(bbox=[0.9, 0.1, 0.1, 0.9])  # ymin > ymax
        assert formatter.validate_citation(c) is False

    def test_page_zero_invalid(self, formatter):
        c = make_citation(page=0)
        assert formatter.validate_citation(c) is False

    def test_page_negative_invalid(self, formatter):
        c = make_citation(page=-5)
        assert formatter.validate_citation(c) is False


# ---------------------------------------------------------------------------
# Tests: filter_valid_citations
# ---------------------------------------------------------------------------
class TestFilterValidCitations:
    def test_filters_invalid_bbox(self, formatter):
        bad = make_citation("cite_bad", bbox=[0.0, 0.0, 0.0, 0.0])
        good = make_citation("cite_good", bbox=[0.1, 0.1, 0.5, 0.9])
        result = formatter.filter_valid_citations([bad, good])
        assert len(result) == 1
        assert result[0].citation_id == "cite_good"

    def test_all_valid_passes(self, formatter, valid_citations):
        result = formatter.filter_valid_citations(valid_citations)
        assert len(result) == 2

    def test_all_invalid_returns_empty(self, formatter):
        bad_citations = [make_citation(bbox=[1.0, 1.0, 0.0, 0.0])]  # inverted
        result = formatter.filter_valid_citations(bad_citations)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: inject_citation_tags
# ---------------------------------------------------------------------------
class TestInjectCitationTags:
    def test_no_auto_inject_when_tags_present(self, formatter, valid_citations):
        answer = "ROE đạt 25% [[cite_1]] và margin 18% [[cite_2]]."
        result = formatter.inject_citation_tags(answer, valid_citations, auto_inject=True)
        # Đã có tag → không inject thêm
        assert "---" not in result
        assert result == answer

    def test_auto_inject_when_no_tags(self, formatter, valid_citations):
        answer = "ROE đạt 25% và biên lợi nhuận gộp 18%."
        result = formatter.inject_citation_tags(answer, valid_citations, auto_inject=True)
        assert "Nguồn trích dẫn" in result
        assert "[[cite_1]]" in result
        assert "[[cite_2]]" in result

    def test_no_inject_when_disabled(self, formatter, valid_citations):
        answer = "ROE đạt 25%."
        result = formatter.inject_citation_tags(answer, valid_citations, auto_inject=False)
        assert result == answer  # Không thay đổi

    def test_empty_citations_returns_original(self, formatter):
        answer = "Không có trích dẫn."
        result = formatter.inject_citation_tags(answer, [], auto_inject=True)
        assert result == answer


# ---------------------------------------------------------------------------
# Tests: extract_cited_ids
# ---------------------------------------------------------------------------
class TestExtractCitedIds:
    def test_double_bracket(self, formatter):
        answer = "Doanh thu tăng [[cite_1]] và chi phí giảm [[cite_3]]."
        ids = formatter.extract_cited_ids(answer)
        assert ids == [1, 3]

    def test_single_bracket(self, formatter):
        answer = "ROE = 25% [cite_2]."
        ids = formatter.extract_cited_ids(answer)
        assert ids == [2]

    def test_no_tags(self, formatter):
        ids = formatter.extract_cited_ids("Không có cite nào.")
        assert ids == []

    def test_mixed_case(self, formatter):
        ids = formatter.extract_cited_ids("Tham chiếu [[CITE_1]].")
        assert ids == [1]


# ---------------------------------------------------------------------------
# Tests: to_api_payload
# ---------------------------------------------------------------------------
class TestToApiPayload:
    def test_returns_required_keys(self, formatter, valid_citations):
        payload = formatter.to_api_payload("Câu trả lời.", valid_citations)
        assert "answer" in payload
        assert "citations" in payload
        assert "citation_count" in payload
        assert "has_grounding" in payload

    def test_citation_count_correct(self, formatter, valid_citations):
        payload = formatter.to_api_payload("Test.", valid_citations)
        assert payload["citation_count"] == 2
        assert payload["has_grounding"] is True

    def test_empty_citations_no_grounding(self, formatter):
        payload = formatter.to_api_payload("Test.", [])
        assert payload["citation_count"] == 0
        assert payload["has_grounding"] is False

    def test_filters_invalid_by_default(self, formatter):
        bad = make_citation("cite_bad", bbox=[0.0, 0.0, 0.0, 0.0])
        good = make_citation("cite_good")
        payload = formatter.to_api_payload("Test.", [bad, good], filter_invalid=True)
        assert payload["citation_count"] == 1

    def test_citation_payload_structure(self, formatter, valid_citations):
        payload = formatter.to_api_payload("Test.", valid_citations)
        cite = payload["citations"][0]
        required_keys = {"citation_id", "block_id", "source_type", "page", "bbox", "snippet", "company", "year", "confidence"}
        assert required_keys.issubset(cite.keys())


# ---------------------------------------------------------------------------
# Tests: format_sql_context (snapshot)
# ---------------------------------------------------------------------------
class TestFormatSqlContext:
    def test_snapshot_format(self, formatter):
        snapshot = {
            "company": "VNM",
            "year": 2025,
            "key_facts": {"NET_REVENUE": 14352000000000.0, "NET_INCOME": 2100000000000.0},
            "unit": {"NET_REVENUE": "VND", "NET_INCOME": "VND"},
            "ratios": {"roe": 0.2512, "current_ratio": 1.85},
            "verification": {"is_balanced": 1, "total_checks": 17, "passed_checks": 17, "failed_checks": 0},
        }
        result = formatter.format_sql_context(snapshot, query_type="snapshot")
        assert "VNM" in result
        assert "2025" in result
        assert "NET_REVENUE" in result
        assert "roe" in result
        assert "ĐÃ CÂN ĐỐI" in result

    def test_single_fact_format(self, formatter):
        fact = {"concept": "NET_REVENUE", "val_current": 14352000000000.0, "unit": "VND"}
        result = formatter.format_sql_context(fact, query_type="fact")
        assert "NET_REVENUE" in result
        assert "VND" in result
