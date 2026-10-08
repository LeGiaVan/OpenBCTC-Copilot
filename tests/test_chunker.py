"""tests/test_chunker.py — Kiểm tra LayoutAwareChunker trên dữ liệu thật VNM 2025."""

from pathlib import Path
import pytest

from ingestion.parsers.json_block_loader import JSONBlockLoader
from src.services.chunker import LayoutAwareChunker


@pytest.fixture
def real_blocks():
    notes_dir = Path("data/VNM_2025/cache/notes")
    if not notes_dir.exists():
        pytest.skip("Chưa có data/VNM_2025/cache/notes")
    return JSONBlockLoader.load_from_dir(notes_dir)


def test_chunking_preserves_structure_and_merges(real_blocks):
    """Kiểm tra gom 438 blocks thành các Semantic Chunks giàu ngữ nghĩa."""
    chunker = LayoutAwareChunker(min_chars=250, max_chars=800)
    chunks = chunker.chunk_blocks(real_blocks)

    assert len(chunks) > 0
    # Số chunk phải ít hơn số block do đã gộp các dòng ngắn và lọc rác
    assert len(chunks) < len(real_blocks)

    # Độ dài trung bình của chunk phải đạt mức tối ưu cho RAG (~500 ký tự)
    avg_len = sum(len(c.content) for c in chunks) / len(chunks)
    assert avg_len >= 300

    # Kiểm tra tính hợp lệ của Enclosing BBox
    for c in chunks:
        assert len(c.bbox) == 4
        assert 0.0 <= c.bbox[0] <= 1.05  # ymin
        assert 0.0 <= c.bbox[1] <= 1.05  # xmin
        assert c.bbox[0] <= c.bbox[2]    # ymin <= ymax
        assert c.bbox[1] <= c.bbox[3]    # xmin <= xmax
        assert len(c.source_block_ids) >= 1


def test_table_chunks_are_standalone(real_blocks):
    """Bảng biểu (table) không được gộp lẫn với văn bản thường."""
    chunker = LayoutAwareChunker()
    chunks = chunker.chunk_blocks(real_blocks)

    table_chunks = [c for c in chunks if c.block_type == "table"]
    for tc in table_chunks:
        # Mỗi table chunk chỉ đến từ 1 table block gốc
        assert len(tc.source_block_ids) == 1
