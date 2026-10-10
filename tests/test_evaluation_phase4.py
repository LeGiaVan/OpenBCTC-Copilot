"""tests/test_evaluation_phase4.py — Unit & Integration tests cho Phase 4.

Kiểm tra:
  1. Golden Financial Dataset: Đầy đủ 50 câu hỏi, đúng 3 nhóm category, schema hợp lệ.
  2. FinancialEvaluator: Độ chính xác của SQL Accuracy, Citation Precision, Faithfulness, Relevancy.
  3. Feedback API: Gửi Upvote/Downvote và kiểm tra thống kê qua /api/v1/feedback.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.observability.evaluator import FinancialEvaluator
from src.observability.langfuse_client import is_langfuse_configured


@pytest.fixture
def client():
    return TestClient(app)


def test_golden_dataset_structure():
    """Kiểm tra Golden Dataset đủ 50 câu hỏi và đúng phân bổ."""
    dataset_path = Path("data/golden_dataset.json")
    assert dataset_path.exists(), "Không tìm thấy data/golden_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 50, f"Golden Dataset phải có đúng 50 câu hỏi, hiện có: {len(data)}"

    categories = [item["category"] for item in data]
    assert categories.count("NUMERIC_FACT") == 20
    assert categories.count("NOTE_EXPLANATION") == 20
    assert categories.count("DEEP_ANALYSIS") == 10

    for item in data:
        assert "id" in item
        assert "query" in item
        assert "expected_intent" in item
        assert "expected_pages" in item or "expected_values" in item


def test_evaluator_sql_accuracy():
    """Kiểm tra logic đánh giá SQL Execution Accuracy."""
    evaluator = FinancialEvaluator()

    # Case 1: Khớp số liệu chính xác
    answer = "Tổng tài sản của Vinamilk năm 2025 là 45.952.496.972.636 VND (khoảng 45.952 tỷ đồng)."
    expected = {"TOTAL_ASSETS": 45952496972636.0}
    score, det = evaluator.evaluate_sql_accuracy(answer, expected)
    assert score == 1.0
    assert det["matched_count"] == 1

    # Case 2: Số liệu sai lệch hoàn toàn
    wrong_answer = "Tổng tài sản của Vinamilk năm 2025 là 10.000.000.000 VND (10 tỷ đồng)."
    score_wrong, det_wrong = evaluator.evaluate_sql_accuracy(wrong_answer, expected)
    assert score_wrong == 0.0
    assert len(det_wrong["unmatched_concepts"]) == 1


def test_evaluator_citation_precision():
    """Kiểm tra logic đánh giá Citation Precision."""
    evaluator = FinancialEvaluator()

    # Case 1: Trích dẫn đúng trang
    citations = [
        {"citation_id": "cite_1", "page": 8, "source_type": "statement"},
        {"citation_id": "cite_2", "page": 9, "source_type": "statement"},
    ]
    expected_pages = [8, 9]
    score, det = evaluator.evaluate_citation_precision(citations, expected_pages)
    assert score == 1.0

    # Case 2: Trích dẫn nhầm trang
    wrong_citations = [
        {"citation_id": "cite_1", "page": 53, "source_type": "notes"},
    ]
    score_wrong, det_wrong = evaluator.evaluate_citation_precision(wrong_citations, [10])
    assert score_wrong == 0.0


def test_evaluator_relevancy():
    """Kiểm tra logic đánh giá Relevancy từ khóa."""
    evaluator = FinancialEvaluator()

    answer = "Doanh thu thuần về bán hàng năm 2025 của Vinamilk đạt 52.991 tỷ đồng."
    keywords = ["doanh thu thuần", "52.991", "đồng"]
    score, det = evaluator.evaluate_relevancy("Doanh thu bao nhiêu?", answer, keywords)
    assert score == 1.0


def test_feedback_api_lifecycle(client):
    """Kiểm tra toàn bộ luồng gửi và đọc thống kê Human Feedback API."""
    # 1. Gửi Upvote
    upvote_payload = {
        "query": "Doanh thu thuần năm 2025 là bao nhiêu?",
        "answer": "Doanh thu thuần là 52.991 tỷ đồng [[cite_1]].",
        "feedback_type": "UPVOTE",
        "rating": 5,
        "comment": "Số liệu chuẩn xác, trích dẫn đúng trang 10",
        "thread_id": "test_session_fb_1",
    }
    res_up = client.post("/api/v1/feedback", json=upvote_payload)
    assert res_up.status_code == 200
    data_up = res_up.json()
    assert data_up["status"] == "SUCCESS"
    assert "feedback_id" in data_up

    # 2. Gửi Downvote
    downvote_payload = {
        "query": "Thù lao HĐQT năm 2025?",
        "answer": "Không rõ thông tin.",
        "feedback_type": "DOWNVOTE",
        "rating": 2,
        "category": "MO_HO",
        "comment": "Cần trích dẫn thuyết minh trang 53",
        "thread_id": "test_session_fb_1",
    }
    res_down = client.post("/api/v1/feedback", json=downvote_payload)
    assert res_down.status_code == 200

    # 3. Lấy Stats
    res_stats = client.get("/api/v1/feedback/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert stats["total"] >= 2
    assert stats["upvotes"] >= 1
    assert stats["downvotes"] >= 1
    assert 0.0 <= stats["satisfaction_rate"] <= 1.0
