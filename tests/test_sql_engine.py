"""tests/test_sql_engine.py — Kiểm tra SQLiteFactService trên database thật VNM 2025."""

from pathlib import Path
import pytest

from src.services.sql_engine import SQLiteFactService


@pytest.fixture
def sql_service():
    db_path = Path("data/VNM_2025/benchmark_VNM_2025.db")
    if not db_path.exists():
        pytest.skip("Chưa có database thật: data/VNM_2025/benchmark_VNM_2025.db")
    return SQLiteFactService(db_path=db_path)


def test_get_core_fact(sql_service):
    """Kiểm tra tra cứu Tổng tài sản (CURRENT_ASSETS, mã 100) của VNM năm 2025."""
    fact = sql_service.get_fact(company="VNM", year=2025, concept="CURRENT_ASSETS")
    assert fact is not None
    assert fact.company == "VNM"
    assert fact.year == 2025
    assert fact.value == 27309234148199.0
    assert fact.standard_code == "100"


def test_get_fact_by_code(sql_service):
    """Kiểm tra tra cứu theo mã chuẩn TT200 (mã 110: Tiền và tương đương tiền)."""
    fact = sql_service.get_fact_by_code(company="VNM", year=2025, standard_code="110")
    assert fact is not None
    assert fact.concept == "CASH_AND_EQUIVALENTS"
    assert fact.value == 1047628845195.0


def test_get_financial_ratios(sql_service):
    """Kiểm tra tra cứu các chỉ số tài chính của VNM 2025."""
    ratios = sql_service.get_ratios(company="VNM", year=2025)
    assert len(ratios) == 13

    # Kiểm tra chỉ số cụ thể: current_ratio
    cr = sql_service.get_ratio_by_name(company="VNM", year=2025, ratio_name="current_ratio")
    assert cr is not None
    assert cr.value == 1.6365
    assert cr.ratio_category == "liquidity"


def test_compare_facts_growth(sql_service):
    """Kiểm tra tính năng so sánh biến động chỉ tiêu giữa 2 kỳ."""
    comp = sql_service.compare_facts(
        company="VNM",
        concept="CURRENT_ASSETS",
        year_current=2025,
        year_previous=2024,
    )
    assert comp["concept"] == "CURRENT_ASSETS"
    assert comp["val_current"] == 27309234148199.0
    assert comp["val_previous"] == 29011829291350.0
    assert comp["diff_absolute"] < 0  # Tài sản ngắn hạn 2025 giảm so với 2024
    assert comp["diff_percentage"] is not None


def test_verification_status(sql_service):
    """Kiểm tra trạng thái cân đối kế toán Anti-GIGO 17 đẳng thức."""
    report = sql_service.get_verification_report(company="VNM", year=2025)
    assert report is not None
    assert report["is_balanced"] == 1
    assert report["total_checks"] == 17
    assert report["passed_checks"] is not None
