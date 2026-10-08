"""ingestion/parsers/md_hierarchy_parser.py — Trích xuất cấu trúc mục lục phân cấp (TOC Tree) từ Markdown BCTC."""

import re
from pathlib import Path
from pydantic import BaseModel, Field


class TOCItem(BaseModel):
    """Một mục trong cây phân cấp mục lục Markdown."""
    title: str = Field(description="Tiêu đề của heading")
    level: int = Field(description="Cấp độ heading (1 = #, 2 = ##, 3 = ###)")
    line_number: int = Field(description="Dòng xuất hiện trong file Markdown (1-indexed)")
    page_start: int | None = Field(default=None, description="Số trang bắt đầu nếu phát hiện được")
    page_end: int | None = Field(default=None, description="Số trang kết thúc nếu phát hiện được")


class MarkdownHierarchyParser:
    """Parser phân tích cú pháp Markdown để trích xuất cây mục lục và định vị phân đoạn tài liệu."""

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$")
    PAGE_REGEX = re.compile(r"\*\(Trang\s+(\d+)(?:[–-](\d+))?\)\*")

    @classmethod
    def parse_file(cls, file_path: Path | str) -> list[TOCItem]:
        """Phân tích toàn văn file Markdown và trả về danh sách các mục TOC."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file markdown: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        toc_items: list[TOCItem] = []

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if m := cls.HEADING_REGEX.match(stripped):
                level = len(m.group(1))
                title = m.group(2).strip()

                # Kiểm tra 1-2 dòng tiếp theo xem có tag trang không (ví dụ: *(Trang 7–9)*)
                p_start, p_end = None, None
                for lookahead in range(1, 3):
                    if idx + lookahead <= len(lines):
                        next_line = lines[idx + lookahead - 1].strip()
                        if pm := cls.PAGE_REGEX.search(next_line):
                            p_start = int(pm.group(1))
                            p_end = int(pm.group(2)) if pm.group(2) else p_start
                            break

                toc_items.append(
                    TOCItem(
                        title=title,
                        level=level,
                        line_number=idx,
                        page_start=p_start,
                        page_end=p_end,
                    )
                )

        return toc_items

    @classmethod
    def find_section_for_page(cls, toc: list[TOCItem], page: int) -> TOCItem | None:
        """Tìm mục lục gần nhất bao trùm hoặc bắt đầu trước số trang chỉ định."""
        candidates = [item for item in toc if item.page_start is not None and item.page_start <= page]
        if not candidates:
            return None
        # Lấy mục có page_start lớn nhất <= page
        return max(candidates, key=lambda x: (x.page_start or 0, x.level))
