"""tests/test_router.py — Unit tests cho Query Intent Classifier & Router (Phase 3.1)."""

from __future__ import annotations

import pytest
from src.agents.copilot.router import classify_query, router_node, route_by_intent
from src.agents.copilot.state import QueryIntent


# ---------------------------------------------------------------------------
# Tests: classify_query — NUMERIC_FACT
# ---------------------------------------------------------------------------
class TestNumericFact:
    def test_asks_total_assets(self):
        r = classify_query("Tổng tài sản của VNM năm 2025 là bao nhiêu?")
        assert r.intent == QueryIntent.NUMERIC_FACT

    def test_asks_roe(self):
        r = classify_query("ROE của VNM năm 2025 là bao nhiêu?")
        assert r.intent == QueryIntent.NUMERIC_FACT

    def test_asks_net_revenue(self):
        r = classify_query("Doanh thu thuần năm 2025 đạt bao nhiêu tỷ?")
        assert r.intent == QueryIntent.NUMERIC_FACT

    def test_asks_current_ratio(self):
        r = classify_query("Thanh toán hiện hành của VNM là bao nhiêu?")
        assert r.intent == QueryIntent.NUMERIC_FACT

    def test_asks_ratio_concept(self):
        r = classify_query("Biên lợi nhuận gộp năm 2025?")
        assert r.intent == QueryIntent.NUMERIC_FACT

    def test_asks_cash_flow(self):
        r = classify_query("Lưu chuyển tiền từ hoạt động kinh doanh năm 2025 là bao nhiêu?")
        assert r.intent == QueryIntent.NUMERIC_FACT


# ---------------------------------------------------------------------------
# Tests: classify_query — NOTE_EXPLANATION
# ---------------------------------------------------------------------------
class TestNoteExplanation:
    def test_asks_accounting_policy(self):
        r = classify_query("Chính sách kế toán hàng tồn kho của VNM như thế nào?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION

    def test_asks_note_detail(self):
        r = classify_query("Thuyết minh số 5 về khoản phải thu ngắn hạn gồm những gì?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION

    def test_asks_auditor_opinion(self):
        r = classify_query("Ý kiến kiểm toán viên về BCTC của VNM 2025 là gì?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION

    def test_asks_related_party(self):
        r = classify_query("Giao dịch bên liên quan của VNM trong năm 2025?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION

    def test_asks_depreciation_method(self):
        r = classify_query("Phương pháp khấu hao tài sản cố định hữu hình là gì?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION

    def test_asks_inventory_breakdown(self):
        r = classify_query("Cơ cấu hàng tồn kho của VNM bao gồm những gì?")
        assert r.intent == QueryIntent.NOTE_EXPLANATION


# ---------------------------------------------------------------------------
# Tests: classify_query — DEEP_ANALYSIS
# ---------------------------------------------------------------------------
class TestDeepAnalysis:
    def test_asks_why_cost_increased(self):
        r = classify_query("Tại sao chi phí tài chính của VNM tăng mạnh trong năm 2025?")
        assert r.intent == QueryIntent.DEEP_ANALYSIS

    def test_asks_analysis(self):
        r = classify_query("Phân tích hiệu quả hoạt động kinh doanh của VNM năm 2025.")
        assert r.intent == QueryIntent.DEEP_ANALYSIS

    def test_asks_comparison(self):
        r = classify_query("So sánh biên lợi nhuận của VNM qua 3 năm gần nhất.")
        assert r.intent == QueryIntent.DEEP_ANALYSIS

    def test_asks_cause(self):
        r = classify_query("Nguyên nhân nào khiến ROE của VNM giảm trong năm 2025?")
        assert r.intent == QueryIntent.DEEP_ANALYSIS

    def test_asks_trend(self):
        r = classify_query("Xu hướng biến động của nợ vay VNM qua các năm?")
        assert r.intent == QueryIntent.DEEP_ANALYSIS


# ---------------------------------------------------------------------------
# Tests: metadata extraction
# ---------------------------------------------------------------------------
class TestMetadataExtraction:
    def test_extracts_year(self):
        r = classify_query("Doanh thu VNM năm 2025?")
        assert r.year == 2025

    def test_extracts_company(self):
        r = classify_query("Tổng tài sản HPG năm 2024?")
        assert r.company == "HPG"

    def test_default_company_fallback(self):
        r = classify_query("Doanh thu năm 2025?", default_company="VNM")
        assert r.company == "VNM"

    def test_excludes_vnd_false_positive(self):
        r = classify_query("Doanh thu tính bằng VND là bao nhiêu?", default_company="VNM")
        # VND không phải mã CK → dùng default
        assert r.company == "VNM"

    def test_no_year_returns_none_without_default(self):
        r = classify_query("Tổng tài sản là bao nhiêu?")
        assert r.year is None

    def test_confidence_is_float(self):
        r = classify_query("ROE VNM 2025?")
        assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# Tests: router_node (LangGraph node function)
# ---------------------------------------------------------------------------
class TestRouterNode:
    def _make_state(self, query: str, company=None, year=None):
        return {
            "query": query,
            "company": company,
            "year": year,
            "messages": [],
            "thread_id": "test",
            "intent": None,
            "sql_result": None,
            "sql_context": "",
            "vector_results": [],
            "vector_context": "",
            "draft_answer": "",
            "fact_check_status": None,
            "fact_check_violations": [],
            "correction_attempts": 0,
            "citations": [],
            "final_answer": "",
        }

    def test_router_node_sets_intent(self):
        state = self._make_state("Tổng tài sản VNM 2025 là bao nhiêu?")
        patch = router_node(state)
        assert "intent" in patch
        assert isinstance(patch["intent"], QueryIntent)

    def test_router_node_numeric_fact(self):
        state = self._make_state("ROE của VNM năm 2025?")
        patch = router_node(state)
        assert patch["intent"] == QueryIntent.NUMERIC_FACT

    def test_router_node_deep_analysis(self):
        state = self._make_state("Tại sao chi phí VNM tăng?")
        patch = router_node(state)
        assert patch["intent"] == QueryIntent.DEEP_ANALYSIS

    def test_router_node_updates_company_and_year(self):
        state = self._make_state("Doanh thu HPG năm 2024?")
        patch = router_node(state)
        assert patch.get("company") == "HPG"
        assert patch.get("year") == 2024


# ---------------------------------------------------------------------------
# Tests: route_by_intent (Conditional Edge)
# ---------------------------------------------------------------------------
class TestRouteByIntent:
    def _state_with_intent(self, intent):
        return {"intent": intent, "query": "", "messages": [], "thread_id": "test",
                "company": None, "year": None, "sql_result": None, "sql_context": "",
                "vector_results": [], "vector_context": "", "draft_answer": "",
                "fact_check_status": None, "fact_check_violations": [],
                "correction_attempts": 0, "citations": [], "final_answer": ""}

    def test_numeric_routes_to_sql(self):
        state = self._state_with_intent(QueryIntent.NUMERIC_FACT)
        assert route_by_intent(state) == "sql_node"

    def test_note_routes_to_vector(self):
        state = self._state_with_intent(QueryIntent.NOTE_EXPLANATION)
        assert route_by_intent(state) == "vector_node"

    def test_deep_routes_to_hybrid(self):
        state = self._state_with_intent(QueryIntent.DEEP_ANALYSIS)
        assert route_by_intent(state) == "hybrid_node"

    def test_clarify_routes_to_clarify_node(self):
        state = self._state_with_intent(QueryIntent.CLARIFY)
        assert route_by_intent(state) == "clarify_node"


# ---------------------------------------------------------------------------
# Tests: classify_query_llm & clarify_node
# ---------------------------------------------------------------------------
class TestLLMRouterAndClarifyNode:
    def test_classify_query_llm_note_explanation(self):
        from unittest.mock import MagicMock
        from src.agents.copilot.router import classify_query_llm

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            '{"intent": "NOTE_EXPLANATION", "reasoning": "Khoản mục con trong Thuyết minh", "clarification_prompt": null}'
        )

        result = classify_query_llm(
            "Nguyên giá trong bất động sản đầu tư nắm giữ cho thuê ngày 31/12 là bao nhiêu",
            llm=mock_llm,
            default_company="VNM",
            default_year=2025,
        )
        assert result.intent == QueryIntent.NOTE_EXPLANATION
        assert result.clarification_prompt is None
        assert result.company == "VNM"
        assert result.year == 2025

    def test_classify_query_llm_clarify_trigger(self):
        from unittest.mock import MagicMock
        from src.agents.copilot.router import classify_query_llm

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            '{"intent": "CLARIFY", "reasoning": "Câu hỏi mơ hồ", "clarification_prompt": "Bạn muốn xem khoản mục cụ thể nào?"}'
        )

        result = classify_query_llm(
            "Cho tôi xem cái đó đi",
            llm=mock_llm,
        )
        assert result.intent == QueryIntent.CLARIFY
        assert result.clarification_prompt == "Bạn muốn xem khoản mục cụ thể nào?"

    def test_clarify_node_returns_prompt(self):
        from src.agents.copilot.nodes import clarify_node
        from src.agents.copilot.state import FactCheckStatus

        state = {
            "query": "cái đó",
            "clarification_prompt": "Bạn muốn hỏi khoản mục nào cụ thể?",
            "company": "VNM",
            "year": 2025,
        }
        res = clarify_node(state)
        assert res["final_answer"] == "Bạn muốn hỏi khoản mục nào cụ thể?"
        assert res["fact_check_status"] == FactCheckStatus.SKIPPED
