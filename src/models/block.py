"""src/models/block.py — Pydantic schema cho JSON Block và SemanticChunk kế thừa từ OpenBCTC."""

from typing import Any
from pydantic import BaseModel, Field, field_validator


class BlockMetadata(BaseModel):
    """Metadata gắn liền với từng khối block từ pipeline OCR & post-process của OpenBCTC."""
    company: str = Field(description="Mã cổ phiếu / Tên doanh nghiệp (ví dụ: HPG, VNM)")
    year: int = Field(description="Năm tài chính của báo cáo (ví dụ: 2025)")
    engine: str = Field(default="mineru_vietocr", description="Công cụ OCR trích xuất")
    is_note: bool = Field(default=False, description="Đánh dấu block thuộc phần Thuyết minh BCTC")
    note_number: str | None = Field(default=None, description="Mục thuyết minh nếu nhận diện được (ví dụ: '5', '12.1')")
    extra: dict[str, Any] = Field(default_factory=dict, description="Metadata mở rộng khác")


class JSONBlock(BaseModel):
    """Khối nội dung văn bản hoặc bảng biểu đơn lẻ được trích xuất từ PDF kèm toạ độ BBox."""
    block_id: str = Field(description="Định danh duy nhất của block (ví dụ: 'p12_mineru_txt_2')")
    block_type: str = Field(default="text", description="Loại block ('text', 'table', 'title')")
    page: int = Field(description="Số trang trong tài liệu PDF gốc (1-indexed)")
    content: str = Field(description="Nội dung văn bản hoặc biểu diễn bảng dạng Markdown của block")
    bbox: list[float] = Field(
        description="Toạ độ bounding box [ymin, xmin, ymax, xmax] chuẩn hoá từ 0.0 đến 1.0"
    )
    source: str = Field(default="local_ocr", description="Nguồn trích xuất (ví dụ: 'local_ocr', 'pdfplumber')")
    metadata: BlockMetadata = Field(description="Metadata chi tiết về doanh nghiệp và năm")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(f"BBox phải có đúng 4 phần tử [ymin, xmin, ymax, xmax], nhận được: {len(v)}")
        for coord in v:
            if not (0.0 <= coord <= 1.05):  # cho phép dung sai nhỏ do rounding OCR
                raise ValueError(f"Toạ độ BBox phải nằm trong khoảng [0.0, 1.0], nhận được: {coord}")
        return v


class SemanticChunk(BaseModel):
    """Chunk ngữ nghĩa sau khi gom các block liên tiếp, có Enclosing BBox phục vụ Visual Grounding."""
    chunk_id: str = Field(description="Định danh duy nhất của chunk")
    content: str = Field(description="Nội dung văn bản (có thể đã kèm Context Header)")
    raw_content: str = Field(description="Nội dung văn bản gốc chưa thêm header")
    page: int = Field(description="Số trang trong tài liệu PDF gốc")
    bbox: list[float] = Field(description="Hộp bao ngoài [ymin, xmin, ymax, xmax] bao trọn các block cấu thành")
    block_type: str = Field(default="text", description="Loại chunk ('text' hoặc 'table')")
    source_block_ids: list[str] = Field(default_factory=list, description="Danh sách các block_id cấu thành")
    company: str = Field(description="Mã công ty")
    year: int = Field(description="Năm báo cáo")
    is_note: bool = Field(default=False, description="Đánh dấu thuộc Thuyết minh BCTC")
