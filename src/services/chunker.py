"""src/services/chunker.py — Layout-Aware Semantic Chunker bảo toàn Bounding Box cho PDF Highlight."""

import re
from src.models.block import JSONBlock, SemanticChunk


class LayoutAwareChunker:
    """Bộ chia đoạn ngữ nghĩa thông minh:
    - Loại bỏ boilerplate vụn vặt (< 25 ký tự)
    - Gom các block văn bản liên tiếp cùng trang thành chunk đạt 300 - 800 ký tự
    - Giữ nguyên bảng biểu thành các chunk độc lập
    - Tự động tính Enclosing Bounding Box [ymin, xmin, ymax, xmax] cho toàn bộ đoạn gộp
    """

    BOILERPLATE_PATTERNS = [
        re.compile(r"^mẫu\s+b\s*0[1-9]", re.IGNORECASE),
        re.compile(r"^trang\s+\d+", re.IGNORECASE),
        re.compile(r"^các thuyết minh này là bộ phận hợp thành", re.IGNORECASE),
        re.compile(r"^\(tiếp theo\)$", re.IGNORECASE),
    ]

    HEADING_PREFIX_PATTERNS = [
        re.compile(r"^(#{1,6})\s+"),
        re.compile(r"^[IVXLCDM]+\.\s+"),          # I. II. III.
        re.compile(r"^\d+\.\s+[A-ZÀ-Ỹ]"),          # 1. 2. 3.
        re.compile(r"^thuyết minh\s+\d+", re.IGNORECASE),
    ]

    def __init__(
        self,
        min_chars: int = 250,
        max_chars: int = 800,
        enrich_context: bool = True,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.enrich_context = enrich_context

    def _is_boilerplate(self, content: str) -> bool:
        """Kiểm tra xem block có phải rác/tiêu đề mẫu biểu vụn vặt không."""
        c = content.strip().lower()
        if len(c) < 15:
            return True
        return any(p.search(c) for p in self.BOILERPLATE_PATTERNS)

    def _starts_new_section(self, content: str) -> bool:
        """Kiểm tra văn bản có bắt đầu một mục mới không."""
        c = content.strip()
        return any(p.match(c) for p in self.HEADING_PREFIX_PATTERNS)

    def _create_chunk(
        self,
        group: list[JSONBlock],
        chunk_idx: int,
    ) -> SemanticChunk:
        """Tính hộp bao toạ độ (Enclosing BBox) và tạo SemanticChunk."""
        first_b = group[0]
        raw_text = "\n\n".join(b.content.strip() for b in group if b.content.strip())

        # Tính hộp bao ngoài (Enclosing Union Bounding Box)
        ymin = min(b.bbox[0] for b in group)
        xmin = min(b.bbox[1] for b in group)
        ymax = max(b.bbox[2] for b in group)
        xmax = max(b.bbox[3] for b in group)
        enclosing_bbox = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]

        # Context header
        if self.enrich_context:
            header = f"[{first_b.metadata.company} {first_b.metadata.year} | Trang {first_b.page}]\n"
            content = header + raw_text
        else:
            content = raw_text

        chunk_id = f"p{first_b.page}_c{chunk_idx}_{first_b.block_type}"

        return SemanticChunk(
            chunk_id=chunk_id,
            content=content,
            raw_content=raw_text,
            page=first_b.page,
            bbox=enclosing_bbox,
            block_type=first_b.block_type,
            source_block_ids=[b.block_id for b in group],
            company=first_b.metadata.company,
            year=first_b.metadata.year,
            is_note=any(b.metadata.is_note for b in group),
        )

    def chunk_blocks(self, blocks: list[JSONBlock]) -> list[SemanticChunk]:
        """Chia và gom cụm toàn bộ danh sách blocks theo trang."""
        if not blocks:
            return []

        # 1. Nhóm theo số trang (bảo toàn thứ tự)
        pages_dict: dict[int, list[JSONBlock]] = {}
        for b in blocks:
            pages_dict.setdefault(b.page, []).append(b)

        chunks: list[SemanticChunk] = []

        for page in sorted(pages_dict.keys()):
            page_blocks = pages_dict[page]
            text_accumulator: list[JSONBlock] = []
            chunk_idx = 0

            def flush_text_acc() -> None:
                nonlocal chunk_idx
                if text_accumulator:
                    chunk_idx += 1
                    chunks.append(self._create_chunk(text_accumulator, chunk_idx))
                    text_accumulator.clear()

            for block in page_blocks:
                content = block.content.strip()

                # Bảng biểu luôn giữ nguyên thành chunk độc lập
                if block.block_type == "table":
                    flush_text_acc()
                    chunk_idx += 1
                    chunks.append(self._create_chunk([block], chunk_idx))
                    continue

                # Lọc bỏ rác vụn vặt (nhưng nếu là heading của text thì giữ)
                if self._is_boilerplate(content) and not self._starts_new_section(content):
                    continue

                # Nếu gặp tiêu đề mục mới và accumulator đã có đủ ngữ cảnh tối thiểu -> flush
                current_len = sum(len(b.content) for b in text_accumulator)
                if current_len >= self.min_chars and self._starts_new_section(content):
                    flush_text_acc()

                text_accumulator.append(block)

                # Nếu độ dài vượt ngưỡng tối đa -> flush
                if sum(len(b.content) for b in text_accumulator) >= self.max_chars:
                    flush_text_acc()

            flush_text_acc()

        return chunks
