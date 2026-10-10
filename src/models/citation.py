"""src/models/citation.py — Pydantic schema cho trích dẫn nguồn kèm Bounding Box (Visual Grounding)."""

from pydantic import BaseModel, Field


class CitationWithBBox(BaseModel):
    """Thông tin trích dẫn phục vụ hiển thị highlight trực quan (Visual Grounding) trên PDF Viewer."""
    citation_id: str = Field(description="Mã định danh trích dẫn gắn trong câu trả lời (ví dụ: 'cite_1')")
    block_id: str = Field(description="Định danh block gốc (ví dụ: 'p12_mineru_txt_2')")
    source_type: str = Field(default="note", description="Loại nguồn ('note' | 'statement' | 'report')")
    page: int = Field(description="Số trang PDF chứa bằng chứng trích dẫn")
    bbox: list[float] | None = Field(default=None, description="Toạ độ [xmin, ymin, xmax, ymax] của đoạn văn bản trên trang PDF (None nếu là 3 bảng BCTC cốt lõi)")
    snippet: str = Field(description="Trích đoạn văn bản ngắn chứng minh cho luận điểm")
    company: str | None = Field(default=None, description="Mã công ty liên quan")
    year: int | None = Field(default=None, description="Năm tài chính liên quan")
    confidence: float = Field(default=1.0, description="Đểm số/Độ tin cậy của trích dẫn (Logit score)")
