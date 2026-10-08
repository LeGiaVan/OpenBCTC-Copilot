"""tests/test_json_loader.py — Kiểm tra JSONBlockLoader trên dữ liệu thật VNM 2025."""

from pathlib import Path
import pytest

from ingestion.parsers.json_block_loader import JSONBlockLoader


def test_load_single_page_real_data():
    """Kiểm tra đọc file thật page_13.json của VNM 2025."""
    sample_file = Path("data/VNM_2025/cache/notes/page_13.json")
    if not sample_file.exists():
        pytest.skip("Chưa có file data/VNM_2025/cache/notes/page_13.json")

    blocks = JSONBlockLoader.load_from_file(sample_file)
    assert len(blocks) > 0

    first_block = blocks[0]
    assert first_block.page == 13
    assert first_block.metadata.company == "VNM"
    assert first_block.metadata.year == 2025
    assert first_block.metadata.is_note is True
    assert len(first_block.bbox) == 4
    # BBox phải nằm trong khoảng hợp lệ
    assert all(0.0 <= coord <= 1.05 for coord in first_block.bbox)


def test_load_all_notes_real_dir():
    """Kiểm tra quét toàn bộ thư mục 42 trang thuyết minh VNM 2025."""
    notes_dir = Path("data/VNM_2025/cache/notes")
    if not notes_dir.exists():
        pytest.skip("Chưa có thư mục data/VNM_2025/cache/notes")

    all_blocks = JSONBlockLoader.load_from_dir(notes_dir)
    assert len(all_blocks) > 100  # 42 trang sẽ có hàng trăm blocks
    # Kiểm tra tính toàn vẹn của các blocks
    pages = {b.page for b in all_blocks}
    assert 13 in pages
    assert 54 in pages
