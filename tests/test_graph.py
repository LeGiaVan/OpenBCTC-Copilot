"""tests/test_graph.py — Integration tests cho LangGraph Copilot (Phase 3.5).

Chiến lược:
  - Mock LLM (không gọi API thật).
  - Mock VectorEngineService (không cần Qdrant data).
  - Dùng SQLiteFactService thật với data VNM_2025 thật.
  - Kiểm tra luồng end-to-end: query → state → final_answer + citations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.copilot.graph import build_copilot_graph, run_query
from src.agents.copilot.state import FactCheckStatus, QueryIntent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DB_PATH = Path("data/VNM_2025/benchmark_VNM_2025.db")


@pytest.fixture
def sql_svc():
    """SQLiteFactService thật với data VNM_2025."""
    if not DB_PATH.exists():
        pytest.skip("Chưa có file benchmark_VNM_2025.db")
    from src.services.sql_engine import SQLiteFactService
    return SQLiteFactService(db_path=DB_PATH)


@pytest.fixture
def mock_retriever():
    """HybridRetriever mock — trả về kết quả giả."""
    retriever = MagicMock()
    mock_result = MagicMock()
    mock_result.page = 12
    mock_result.bbox = [0.10, 0.05, 0.50, 0.90]
    mock_result.raw_content = "Doanh thu thuần hợp nhất đạt 14.352 tỷ đồng."
    mock_result.content = "[VNM 2025 | Trang 12]\nDoanh thu thuần hợp nhất đạt 14.352 tỷ đồng."
    mock_result.is_note = False
    mock_result.company = "VNM"
    mock_result.year = 2025
    mock_result.__dict__ = {
        "chunk_id": "p12_c1_text",
        "content": mock_result.content,
        "raw_content": mock_result.raw_content,
        "page": 12,
        "bbox": [0.10, 0.05, 0.50, 0.90],
        "block_type": "text",
        "source_block_ids": ["p12_txt_1"],
        "company": "VNM",
        "year": 2025,
        "is_note": False,
        "hybrid_score": 0.87,
        "rerank_score": None,
        "final_score": 0.87,
    }

    from src.models.citation import CitationWithBBox
    mock_citation = CitationWithBBox(
        citation_id="cite_1",
        block_id="p12_txt_1",
        source_type="statement",
        page=12,
        bbox=[0.10, 0.05, 0.50, 0.90],
        snippet="Doanh thu thuần hợp nhất đạt 14.352 tỷ đồng.",
        company="VNM",
        year=2025,
        confidence=0.87,
    )
    retriever.retrieve_with_citations.return_value = ([mock_result], [mock_citation])
    return retriever


@pytest.fixture
def mock_llm():
    """Mock LLM trả về câu trả lời giả định."""
    llm = MagicMock()
    response = MagicMock()
    response.content = (
        "Doanh thu thuần hợp nhất của VNM năm 2025 đạt **14.352 tỷ đồng** [[cite_1]], "
        "tăng 8,2% so với cùng kỳ năm trước."
    )
    llm.invoke.return_value = response
    return response, llm


@pytest.fixture
def compiled_graph(sql_svc, mock_retriever, mock_llm):
    """Compiled LangGraph với SQL thật + mock retriever + mock LLM."""
    _, llm = mock_llm
    return build_copilot_graph(
        sql_svc=sql_svc,
        retriever=mock_retriever,
        llm=llm,
        use_memory_saver=True,
        langfuse_enabled=False,
    )


# ---------------------------------------------------------------------------
# Tests: Graph Build
# ---------------------------------------------------------------------------
class TestGraphBuild:
    def test_graph_compiles(self, compiled_graph):
        assert compiled_graph is not None

    def test_graph_has_nodes(self, compiled_graph):
        # LangGraph compiled graph có get_graph()
        graph_repr = compiled_graph.get_graph()
        node_names = list(graph_repr.nodes.keys())
        for expected in ["router", "sql_node", "vector_node", "hybrid_node", "synthesize", "fact_verify", "citation_verify"]:
            assert expected in node_names, f"Node '{expected}' missing!"


# ---------------------------------------------------------------------------
# Tests: run_query end-to-end
# ---------------------------------------------------------------------------
class TestRunQuery:
    def test_numeric_query_returns_answer(self, compiled_graph, mock_llm):
        _, llm = mock_llm
        result = run_query(
            compiled_graph,
            query="Tổng tài sản VNM năm 2025 là bao nhiêu?",
            company="VNM",
            year=2025,
            langfuse_enabled=False,
        )
        assert isinstance(result["final_answer"], str)
        assert len(result["final_answer"]) > 0

    def test_result_has_required_keys(self, compiled_graph):
        result = run_query(
            compiled_graph,
            query="ROE của VNM 2025?",
            company="VNM",
            year=2025,
            langfuse_enabled=False,
        )
        for key in ["final_answer", "citations", "intent", "fact_check_status"]:
            assert key in result, f"Key '{key}' missing từ result"

    def test_intent_is_numeric_for_ratio_question(self, compiled_graph):
        result = run_query(
            compiled_graph,
            query="ROE của VNM năm 2025?",
            company="VNM",
            year=2025,
            langfuse_enabled=False,
        )
        assert result["intent"] == QueryIntent.NUMERIC_FACT

    def test_intent_is_deep_for_analysis_question(self, compiled_graph):
        result = run_query(
            compiled_graph,
            query="Tại sao ROE của VNM giảm trong năm 2025? Phân tích nguyên nhân.",
            company="VNM",
            year=2025,
            langfuse_enabled=False,
        )
        assert result["intent"] == QueryIntent.DEEP_ANALYSIS

    def test_note_query_routes_to_vector(self, compiled_graph, mock_retriever):
        result = run_query(
            compiled_graph,
            query="Thuyết minh số 5 về khoản phải thu của VNM? Chính sách kế toán gồm những gì?",
            company="VNM",
            year=2025,
            langfuse_enabled=False,
        )
        # Vector retriever được gọi ít nhất 1 lần (từ vector_node hoặc hybrid_node)
        assert mock_retriever.retrieve_with_citations.call_count >= 1

    def test_multi_turn_same_thread(self, compiled_graph):
        """Multi-turn: 2 câu hỏi cùng thread_id không crash."""
        run_query(compiled_graph, query="Tổng tài sản VNM 2025?", company="VNM", year=2025, thread_id="test-123", langfuse_enabled=False)
        result2 = run_query(compiled_graph, query="ROE VNM 2025?", company="VNM", year=2025, thread_id="test-123", langfuse_enabled=False)
        assert isinstance(result2["final_answer"], str)


# ---------------------------------------------------------------------------
# Tests: Fact Verifier (Phase 3.3)
# ---------------------------------------------------------------------------
class TestFactVerifier:
    def test_no_violations_when_draft_empty(self, sql_svc):
        from src.agents.copilot.nodes import fact_verify_node
        from src.agents.copilot.state import FactCheckStatus

        state = {
            "draft_answer": "",
            "sql_result": {},
            "fact_check_violations": [],
            "fact_check_status": None,
            "correction_attempts": 0,
            "query": "", "messages": [], "thread_id": "t",
            "company": "VNM", "year": 2025, "intent": None,
            "sql_context": "", "vector_results": [], "vector_context": "",
            "citations": [], "final_answer": "",
        }
        patch = fact_verify_node(state, sql_svc=sql_svc)
        assert patch["fact_check_status"] == FactCheckStatus.SKIPPED

    def test_should_correct_triggers_when_failed(self):
        from src.agents.copilot.nodes import should_correct
        from src.agents.copilot.state import FactCheckStatus

        state = {
            "fact_check_status": FactCheckStatus.FAILED,
            "correction_attempts": 0,
            "query": "", "messages": [], "thread_id": "t",
            "company": None, "year": None, "intent": None,
            "sql_result": None, "sql_context": "", "vector_results": [],
            "vector_context": "", "draft_answer": "", "fact_check_violations": [],
            "citations": [], "final_answer": "",
        }
        assert should_correct(state) == "synthesize"

    def test_should_not_correct_when_max_attempts_reached(self):
        from src.agents.copilot.nodes import should_correct
        from src.agents.copilot.state import FactCheckStatus

        state = {
            "fact_check_status": FactCheckStatus.FAILED,
            "correction_attempts": 2,  # Đã đạt giới hạn
            "query": "", "messages": [], "thread_id": "t",
            "company": None, "year": None, "intent": None,
            "sql_result": None, "sql_context": "", "vector_results": [],
            "vector_context": "", "draft_answer": "", "fact_check_violations": [],
            "citations": [], "final_answer": "",
        }
        assert should_correct(state) == "citation_verify"

    def test_should_proceed_when_passed(self):
        from src.agents.copilot.nodes import should_correct
        from src.agents.copilot.state import FactCheckStatus

        state = {
            "fact_check_status": FactCheckStatus.PASSED,
            "correction_attempts": 0,
            "query": "", "messages": [], "thread_id": "t",
            "company": None, "year": None, "intent": None,
            "sql_result": None, "sql_context": "", "vector_results": [],
            "vector_context": "", "draft_answer": "", "fact_check_violations": [],
            "citations": [], "final_answer": "",
        }
        assert should_correct(state) == "citation_verify"


# ---------------------------------------------------------------------------
# Tests: Citation Verify Node (Phase 3.4)
# ---------------------------------------------------------------------------
class TestCitationVerifyNode:
    def _base_state(self, draft="Test answer.", citations=None):
        return {
            "draft_answer": draft,
            "citations": citations or [],
            "query": "", "messages": [], "thread_id": "t",
            "company": "VNM", "year": 2025, "intent": None,
            "sql_result": None, "sql_context": "", "vector_results": [],
            "vector_context": "", "fact_check_status": None,
            "fact_check_violations": [], "correction_attempts": 0,
            "final_answer": "",
        }

    def test_empty_citations_still_produces_final_answer(self):
        from src.agents.copilot.nodes import citation_verify_node
        state = self._base_state(draft="Doanh thu đạt 100 tỷ.")
        patch = citation_verify_node(state)
        assert patch["final_answer"] == "Doanh thu đạt 100 tỷ."  # Không có citation → không thay đổi

    def test_valid_citation_kept(self):
        from src.agents.copilot.nodes import citation_verify_node
        citations = [{
            "citation_id": "cite_1",
            "block_id": "p12_txt_1",
            "source_type": "statement",
            "page": 12,
            "bbox": [0.1, 0.1, 0.5, 0.9],
            "snippet": "Doanh thu 14.352 tỷ.",
            "company": "VNM",
            "year": 2025,
            "confidence": 0.9,
        }]
        state = self._base_state(citations=citations)
        patch = citation_verify_node(state)
        assert len(patch["citations"]) == 1

    def test_invalid_bbox_citation_filtered(self):
        from src.agents.copilot.nodes import citation_verify_node
        citations = [{
            "citation_id": "cite_bad",
            "block_id": "p0_txt_1",
            "source_type": "note",
            "page": 0,  # Invalid page
            "bbox": [0.0, 0.0, 0.0, 0.0],  # Zero area
            "snippet": "...",
            "company": "VNM",
            "year": 2025,
            "confidence": 0.5,
        }]
        state = self._base_state(citations=citations)
        patch = citation_verify_node(state)
        assert len(patch["citations"]) == 0
