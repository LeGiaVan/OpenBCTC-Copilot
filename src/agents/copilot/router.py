"""src/agents/copilot/router.py — Query Intent Classifier & Router (Phase 3.1).

Thiết kế Thin Router (Pure Function):
  - Không gọi LLM để classify → dùng keyword matching + regex thuần Python.
  - Nhanh, deterministic, không tốn token, dễ test.
  - Nếu muốn nâng lên LLM-based classification sau này, chỉ cần thay hàm classify().

Logic phân loại (theo thứ tự ưu tiên):
  1. NUMERIC_FACT   : câu hỏi có keyword số liệu cụ thể / chỉ số tài chính.
  2. NOTE_EXPLANATION: câu hỏi về chính sách, thuyết minh, kiểm toán, phương pháp.
  3. DEEP_ANALYSIS  : câu hỏi "tại sao", "nguyên nhân", "phân tích", "so sánh" → fallback.
"""

from __future__ import annotations

import re
import logging
from typing import NamedTuple

from src.agents.copilot.state import CopilotState, QueryIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword patterns (tiếng Việt + tiếng Anh)
# ---------------------------------------------------------------------------
_NUMERIC_PATTERNS = [
    # Hỏi giá trị / số liệu cụ thể
    re.compile(r"\b(bao nhiêu|giá trị|số liệu|chỉ tiêu|mã số|mã chỉ tiêu)\b", re.I),
    re.compile(r"\b(tổng tài sản|vốn chủ sở hữu|doanh thu|lợi nhuận|chi phí|nợ phải trả)\b", re.I),
    re.compile(r"\b(lưu chuyển tiền|tiền và tương đương tiền|hàng tồn kho|phải thu)\b", re.I),
    # Chỉ số tài chính (ratios)
    re.compile(r"\b(roe|roa|ros|current ratio|thanh toán (hiện hành|nhanh|tiền mặt))\b", re.I),
    re.compile(r"\b(biên lợi nhuận (gộp|ròng|hoạt động)|nợ[/ ](vcsh|tổng tài sản))\b", re.I),
    re.compile(r"\b(vòng quay (tồn kho|tài sản|phải thu)|ebitda|eps|p/e)\b", re.I),
    # Hỏi năm/kỳ cụ thể về 1 con số
    re.compile(r"\b(năm \d{4}|quý [1-4]|6 tháng|nửa năm)\b.*\b(là|bằng|đạt|tăng|giảm)\b", re.I),
    re.compile(r"\b(tổng cộng|cộng cuối kỳ|số dư cuối kỳ|số dư đầu kỳ)\b", re.I),
    # Code chỉ tiêu TT200
    re.compile(r"\bchỉ tiêu\s+(mã\s+)?\d{2,3}\b", re.I),
]

_NOTE_PATTERNS = [
    # Thuyết minh / Notes
    re.compile(r"\b(thuyết minh|thuyết minh số|note \d+|ghi chú)\b", re.I),
    # Chính sách kế toán
    re.compile(r"\b(chính sách kế toán|phương pháp (khấu hao|tính giá|ghi nhận))\b", re.I),
    re.compile(r"\b(cơ sở lập báo cáo|nguyên tắc kế toán|chuẩn mực kế toán)\b", re.I),
    # Kiểm toán / Kiểm soát nội bộ
    re.compile(r"\b(kiểm toán viên|ý kiến kiểm toán|báo cáo kiểm toán|độc lập)\b", re.I),
    # Cơ cấu chi tiết
    re.compile(r"\b(cơ cấu|chi tiết|bảng kê|danh mục|liệt kê|bao gồm)\b", re.I),
    re.compile(r"\b(hàng tồn kho gồm|phân loại|chia ra|trong đó)\b", re.I),
    # Rủi ro / Cam kết
    re.compile(r"\b(rủi ro (tín dụng|lãi suất|thanh khoản|ngoại hối)|cam kết|bảo lãnh)\b", re.I),
    # Công ty con / Liên kết
    re.compile(r"\b(công ty con|công ty liên (kết|doanh)|giao dịch (bên liên quan|nội bộ))\b", re.I),
]

_DEEP_ANALYSIS_PATTERNS = [
    re.compile(r"\b(tại sao|vì sao|nguyên nhân|lý do|giải thích)\b", re.I),
    re.compile(r"\b(phân tích|đánh giá|nhận xét|bình luận|xu hướng)\b", re.I),
    re.compile(r"\b(so sánh|tương quan|biến động|thay đổi|ảnh hưởng|tác động)\b", re.I),
    re.compile(r"\b(rủi ro|triển vọng|dự báo|kế hoạch|chiến lược)\b", re.I),
    re.compile(r"\b(hiệu quả|năng lực|chất lượng|bền vững|cạnh tranh)\b", re.I),
]

# Regex trích xuất công ty & năm từ câu hỏi
_COMPANY_RE = re.compile(
    r"\b([A-Z]{2,5})\b",  # Ký tự hoa 2-5 ký tự: VNM, HPG, FPT, MWG...
)
_YEAR_RE = re.compile(r"\b(202[0-9]|201[5-9])\b")


# ---------------------------------------------------------------------------
# NamedTuple kết quả classify
# ---------------------------------------------------------------------------
class ClassifyResult(NamedTuple):
    intent: QueryIntent
    company: str | None
    year: int | None
    confidence: float    # 0.0 → 1.0, dùng cho logging/observability


# ---------------------------------------------------------------------------
# Core classify function
# ---------------------------------------------------------------------------
def classify_query(query: str, default_company: str | None = None, default_year: int | None = None) -> ClassifyResult:
    """Phân loại câu hỏi tài chính theo 3 nhóm (Pure Function, không gọi LLM).

    Args:
        query: Câu hỏi của người dùng.
        default_company: Công ty mặc định nếu không tìm thấy trong câu hỏi.
        default_year: Năm mặc định nếu không tìm thấy trong câu hỏi.

    Returns:
        ClassifyResult(intent, company, year, confidence)
    """
    q = query.strip()

    # --- Trích xuất metadata ---
    year_match = _YEAR_RE.search(q)
    year = int(year_match.group(1)) if year_match else default_year

    # Tìm mã cổ phiếu: ưu tiên từ các keyword rõ ràng như "VNM 2025", "HPG:"
    company_match = _COMPANY_RE.search(q)
    company = company_match.group(1) if company_match else default_company
    # Loại bỏ false positive: từ viết tắt thông dụng không phải mã CK
    _NOT_TICKER = {"VND", "ROE", "ROA", "ROS", "EPS", "GDP", "USD", "EUR", "CAGR"}
    if company in _NOT_TICKER:
        company = default_company

    # --- Đếm điểm cho từng intent ---
    numeric_hits = sum(1 for p in _NUMERIC_PATTERNS if p.search(q))
    note_hits = sum(1 for p in _NOTE_PATTERNS if p.search(q))
    deep_hits = sum(1 for p in _DEEP_ANALYSIS_PATTERNS if p.search(q))

    logger.debug(
        "Classify '%s' → numeric=%d, note=%d, deep=%d",
        q[:60], numeric_hits, note_hits, deep_hits
    )

    total = numeric_hits + note_hits + deep_hits or 1

    # Áp dụng boost: NOTE và DEEP có ưu tiên cao hơn khi có signal đặc trưng mạnh
    # (tránh NUMERIC override do câu chứa tên khoản mục như 'hàng tồn kho', 'chi phí')
    note_score = note_hits * 2   # NOTE patterns thường đặc trưng hơn NUMERIC
    deep_score = deep_hits * 3   # DEEP patterns ("tại sao", "phân tích") rất đặc trưng
    numeric_score = numeric_hits

    if deep_score >= note_score and deep_score >= numeric_score and deep_hits > 0:
        return ClassifyResult(
            intent=QueryIntent.DEEP_ANALYSIS,
            company=company,
            year=year,
            confidence=round(deep_hits / total, 2),
        )
    if note_score >= numeric_score and note_hits > 0:
        return ClassifyResult(
            intent=QueryIntent.NOTE_EXPLANATION,
            company=company,
            year=year,
            confidence=round(note_hits / total, 2),
        )
    if numeric_hits > 0:
        return ClassifyResult(
            intent=QueryIntent.NUMERIC_FACT,
            company=company,
            year=year,
            confidence=round(numeric_hits / total, 2),
        )
    # Fallback: DEEP_ANALYSIS
    return ClassifyResult(
        intent=QueryIntent.DEEP_ANALYSIS,
        company=company,
        year=year,
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# LangGraph Node function (Thin Node)
# ---------------------------------------------------------------------------
def router_node(state: CopilotState) -> dict:
    """LangGraph Node: phân loại intent và cập nhật state.

    Là Pure Function: không có side effect, không gọi service nào.
    """
    query = state["query"]
    result = classify_query(
        query=query,
        default_company=state.get("company"),
        default_year=state.get("year"),
    )

    logger.info(
        "Intent classified: %s (confidence=%.2f) — company=%s, year=%s",
        result.intent,
        result.confidence,
        result.company,
        result.year,
    )

    return {
        "intent": result.intent,
        "company": result.company,
        "year": result.year,
    }


def route_by_intent(state: CopilotState) -> str:
    """Conditional Edge function: trả về tên node tiếp theo dựa vào intent.

    Được dùng trong ``graph.add_conditional_edges()``.
    """
    intent = state.get("intent")
    if intent == QueryIntent.NUMERIC_FACT:
        return "sql_node"
    if intent == QueryIntent.NOTE_EXPLANATION:
        return "vector_node"
    return "hybrid_node"  # DEEP_ANALYSIS
