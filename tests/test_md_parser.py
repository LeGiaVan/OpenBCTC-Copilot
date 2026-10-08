"""tests/test_md_parser.py — Kiểm tra MarkdownHierarchyParser trên file Markdown thật VNM 2025."""

from pathlib import Path
import pytest

from ingestion.parsers.md_hierarchy_parser import MarkdownHierarchyParser


def test_parse_vnm_markdown_toc():
    """Kiểm tra trích xuất cây mục lục từ file Markdown báo cáo tài chính thật."""
    md_path = Path("data/VNM_2025/VNM_2025_financial_report_final.md")
    if not md_path.exists():
        pytest.skip("Chưa có file data/VNM_2025/VNM_2025_financial_report_final.md")

    toc = MarkdownHierarchyParser.parse_file(md_path)
    assert len(toc) > 0

    # Kiểm tra các heading chính có xuất hiện
    titles = [item.title for item in toc]
    assert any("BẢNG CÂN ĐỐI KẾ TOÁN" in t for t in titles)

    # Kiểm tra nhận diện số trang
    balance_sheet = next(item for item in toc if "BẢNG CÂN ĐỐI KẾ TOÁN" in item.title)
    assert balance_sheet.page_start == 7
    assert balance_sheet.page_end == 9

    # Kiểm tra hàm tìm section theo page
    section_p8 = MarkdownHierarchyParser.find_section_for_page(toc, page=8)
    assert section_p8 is not None
    assert section_p8.page_start <= 8
