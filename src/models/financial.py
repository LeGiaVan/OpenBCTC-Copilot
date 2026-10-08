"""src/models/financial.py — Pydantic schema cho Fact tài chính và Chỉ số tài chính từ SQLite."""

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class StatementType(str, Enum):
    """3 Báo cáo tài chính cốt lõi theo Thông tư 200/2014/TT-BTC."""
    BALANCE_SHEET = "BALANCE_SHEET"           # B01-DN: Bảng Cân đối kế toán
    INCOME_STATEMENT = "INCOME_STATEMENT"     # B02-DN: Báo cáo Kết quả hoạt động kinh doanh
    CASH_FLOW = "CASH_FLOW"                   # B03-DN: Báo cáo Lưu chuyển tiền tệ
    CONSOLIDATED = "CONSOLIDATED"             # Báo cáo hợp nhất


class RatioCategory(str, Enum):
    """Phân nhóm 13 chỉ số tài chính chuẩn."""
    LIQUIDITY = "liquidity"                   # Khả năng thanh toán (Hiện hành, Nhanh, Tiền mặt)
    SOLVENCY = "solvency"                     # Đòn bẩy tài chính (Nợ/VCSH, Nợ/Tổng tài sản, Khả năng trả lãi)
    PROFITABILITY = "profitability"           # Khả năng sinh lời (Biên gộp, Biên ròng, ROA, ROE)
    EFFICIENCY = "efficiency"                 # Hiệu quả hoạt động (Vòng quay tồn kho, Vòng quay tài sản)


class FinancialFactDTO(BaseModel):
    """DTO đại diện cho 1 bản ghi số liệu tài chính đã chuẩn hoá và kiểm toán trong SQLite."""
    model_config = ConfigDict(populate_by_name=True)

    id: str | int | None = None
    prov_id: str | None = None
    company: str = Field(description="Mã cổ phiếu (ví dụ: 'VNM', 'HPG')")
    year: int = Field(description="Năm báo cáo tài chính (ví dụ: 2025)")
    period: str = Field(default="current", description="Kỳ số liệu ('2025', '2024', hoặc 'current')")
    period_type: str | None = Field(default="ANNUAL", description="Loại kỳ (ANNUAL, QUARTER)")
    concept: str = Field(description="Canonical Concept TT200 (ví dụ: 'TOTAL_ASSETS', 'NET_REVENUE')")
    standard_code: str = Field(default="", alias="code", description="Mã số chỉ tiêu theo TT200 (ví dụ: '270', '10')")
    raw_label: str | None = Field(default=None, description="Tên khoản mục gốc từ bảng OCR")
    value: float = Field(description="Giá trị số liệu tài chính đã bóc tách")
    unit: str = Field(default="VND", description="Đơn vị tiền tệ")
    page: int | None = Field(default=None, description="Trang xuất hiện chỉ tiêu")
    table_id: str | None = Field(default=None, description="Mã bảng bóc tách")
    source: str | None = Field(default=None, description="Nguồn trích xuất")
    confidence: float | None = Field(default=1.0, description="Độ tin cậy của fact")
    verification_status: str | None = Field(default=None, description="Trạng thái kiểm toán (PASSED / UNVERIFIED)")
    verification_detail: str | None = Field(default=None, description="Chi tiết kiểm toán đẳng thức")
    statement_type: StatementType | None = Field(default=None, description="Loại báo cáo")

    @property
    def code(self) -> str:
        return self.standard_code


class FinancialRatioDTO(BaseModel):
    """DTO đại diện cho 1 chỉ số tài chính tính toán chuẩn."""
    model_config = ConfigDict(populate_by_name=True)

    id: str | int | None = None
    company: str = Field(description="Mã cổ phiếu")
    year: int = Field(description="Năm tài chính")
    ratio_name: str = Field(default="", alias="ratio_code", description="Tên định danh chỉ số (ví dụ: 'current_ratio', 'roe')")
    ratio_category: str = Field(default="general", alias="category", description="Nhóm chỉ số (liquidity, solvency, profitability, efficiency)")
    value: float = Field(description="Giá trị chỉ số đã tính toán")
    formula: str | None = Field(default=None, description="Công thức tính toán")
    input_prov_ids: str | None = Field(default=None, description="ID nguồn các facts đầu vào")
    is_deterministic: bool = Field(default=True, description="Chỉ số tính bằng Python xác định 100%")
    unit: str = Field(default="ratio", description="Đơn vị tính (lần, %)")

    @property
    def category(self) -> str:
        return self.ratio_category

    @property
    def ratio_code(self) -> str:
        return self.ratio_name
