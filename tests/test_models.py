"""tests/test_models.py — Kiểm tra Pydantic validation của các Data Models."""

import pytest
from pydantic import ValidationError

from src.models import (
    BlockMetadata,
    CitationWithBBox,
    FinancialFactDTO,
    FinancialRatioDTO,
    JSONBlock,
    RatioCategory,
    StatementType,
)


def test_json_block_valid():
    """Kiểm tra khởi tạo JSONBlock với dữ liệu mẫu từ OpenBCTC."""
    data = {
        "block_id": "p12_mineru_txt_2",
        "block_type": "text",
        "page": 12,
        "content": "Các thuyết minh này là một bộ phận hợp thành và cần được đọc đồng thời...",
        "bbox": [0.142, 0.098, 0.915, 0.133],
        "source": "local_ocr",
        "metadata": {
            "company": "HPG",
            "year": 2025,
            "engine": "mineru_vietocr",
            "is_note": True,
        },
    }
    block = JSONBlock.model_validate(data)
    assert block.block_id == "p12_mineru_txt_2"
    assert block.page == 12
    assert block.metadata.company == "HPG"
    assert block.metadata.is_note is True
    assert len(block.bbox) == 4


def test_json_block_invalid_bbox_length():
    """Kiểm tra bbox thiếu toạ độ sẽ bị báo lỗi."""
    data = {
        "block_id": "bad_block",
        "page": 1,
        "content": "test",
        "bbox": [0.1, 0.2],  # Thiếu 2 toạ độ
        "metadata": {"company": "HPG", "year": 2025},
    }
    with pytest.raises(ValidationError):
        JSONBlock.model_validate(data)


def test_citation_with_bbox():
    """Kiểm tra schema CitationWithBBox."""
    citation = CitationWithBBox(
        citation_id="cite_1",
        block_id="p12_mineru_txt_2",
        page=12,
        bbox=[0.142, 0.098, 0.915, 0.133],
        snippet="Các thuyết minh này là một bộ phận hợp thành...",
        company="HPG",
        year=2025,
    )
    assert citation.citation_id == "cite_1"
    assert citation.page == 12
    assert citation.confidence == 1.0


def test_financial_fact_and_ratio():
    """Kiểm tra schema FinancialFactDTO và FinancialRatioDTO."""
    fact = FinancialFactDTO(
        company="HPG",
        year=2025,
        statement_type=StatementType.BALANCE_SHEET,
        concept="TOTAL_ASSETS",
        code="270",
        value=1383355031957.0,
        period="current",
    )
    assert fact.concept == "TOTAL_ASSETS"
    assert fact.value > 0

    ratio = FinancialRatioDTO(
        company="HPG",
        year=2025,
        category=RatioCategory.PROFITABILITY,
        ratio_code="ROE",
        ratio_name="Tỷ suất sinh lời trên vốn chủ sở hữu (ROE)",
        value=15.8,
        unit="%",
    )
    assert ratio.category == RatioCategory.PROFITABILITY
    assert ratio.value == 15.8
