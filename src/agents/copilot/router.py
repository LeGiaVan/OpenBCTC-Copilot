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
from typing import NamedTuple, Any

from src.agents.copilot.state import CopilotState, QueryIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword patterns (tiếng Việt + tiếng Anh)
# ---------------------------------------------------------------------------
"""
Mục tiêu cốt lõi của danh sách này là nhận diện xem câu hỏi có thuộc nhóm QueryIntent.NUMERIC_FACT (truy vấn số liệu kế toán xác định / chỉ số tài chính) hay không, để điều hướng LangGraph sang sql_node lấy dữ liệu từ SQLite DB.
"""
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
    # Chỉ số tài chính & hệ cơ sở dữ liệu SQL
    re.compile(r"\b(chỉ số|chỉ số tài chính|các chỉ số|tất cả chỉ số|hệ cơ sở dữ liệu|cơ sở dữ liệu|database|sql)\b", re.I),
    re.compile(r"\b(bảng cân đối kế toán|kết quả hoạt động kinh doanh|kết quả kinh doanh|lưu chuyển tiền tệ|báo cáo tài chính cốt lõi)\b", re.I),
    re.compile(r"\b(trình bày|liệt kê|cho biết|cho xem|xem|tổng hợp).*(chỉ số|chỉ tiêu|số liệu|tài chính)\b", re.I),
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
    # Cơ cấu tổ chức & Lãnh đạo, ban điều hành
    re.compile(r"\b(hội đồng quản trị|hđqt|ban giám đốc|ban kiểm soát|thành viên|chủ tịch|tổng giám đốc|lãnh đạo|người đại diện|ai là|là ai)\b", re.I),
    # Trụ sở, địa chỉ, cơ sở hoạt động
    re.compile(r"\b(trụ sở|địa chỉ|ở đâu|tại đâu|thông tin chung|nhà máy|chi nhánh|đơn vị trực thuộc)\b", re.I),
    # Cơ cấu chi tiết & danh mục thuyết minh
    re.compile(r"\b(cơ cấu|thành phần|phân bổ|bao gồm những gì|chi tiết thuyết minh|bảng kê)\b", re.I),
    re.compile(r"\b(hàng tồn kho gồm|phân loại chi tiết|chia ra chi tiết)\b", re.I),
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
    clarification_prompt: str | None = None
    standalone_query: str | None = None



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

    # Nếu câu hỏi nói rõ về chỉ số tài chính, 3 BCTC cốt lõi hoặc hệ CSDL SQL
    # và không hỏi sâu "tại sao" / "phân tích" hoặc hỏi riêng về Thuyết minh -> ưu tiên NUMERIC_FACT
    is_explicit_financial_metric = bool(re.search(
        r"\b(chỉ số|chỉ số tài chính|các chỉ số|tất cả chỉ số|chỉ tiêu tài chính|số liệu tài chính|"
        r"hệ cơ sở dữ liệu|cơ sở dữ liệu|database|sql|báo cáo tài chính cốt lõi|"
        r"bảng cân đối kế toán|kết quả hoạt động kinh doanh|kết quả kinh doanh|lưu chuyển tiền tệ)\b",
        q,
        re.I,
    ))
    if is_explicit_financial_metric and deep_hits == 0 and not re.search(r"\b(thuyết minh|chính sách kế toán|kiểm toán)\b", q, re.I):
        numeric_score += 10

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
# LLM-based Semantic Router & Clarification
# ---------------------------------------------------------------------------
_ROUTER_SYSTEM_PROMPT = """Bạn là chuyên gia định tuyến truy vấn tài chính (Semantic Financial Query Router) am hiểu sâu Chuẩn mực Kế toán Việt Nam (VAS) và Thông tư 200/2014/TT-BTC.

Nhiệm vụ của bạn là phân tích toàn diện ngữ nghĩa câu hỏi của người dùng và phân loại vào CHÍNH XÁC một trong các nhóm Intent sau:

1. "NUMERIC_FACT":
   - Chỉ khi câu hỏi tra cứu các CHỈ TIÊU LỚN TỔNG HỢP từ 3 Báo cáo Tài chính cốt lõi (Bảng Cân đối kế toán, Báo cáo Kết quả HĐKD, Báo cáo Lưu chuyển tiền tệ).
   - Ví dụ: Tổng tài sản, Doanh thu thuần, Lợi nhuận gộp, Lợi nhuận sau thuế, Vốn chủ sở hữu, Tổng nợ phải trả, Tiền và tương đương tiền, 13 chỉ số tài chính (ROE, ROA, ROS, Current ratio, Tỷ số nợ...).
   - Hỏi danh sách tất cả các chỉ số tài chính trong hệ cơ sở dữ liệu SQL.

2. "NOTE_EXPLANATION":
   - Khi câu hỏi tra cứu các KHOẢN MỤC PHÂN RÃ CHI TIẾT, BẢNG BIỂU CON, CHÍNH SÁCH KẾ TOÁN hoặc THUYẾT MINH CHUYÊN SÂU (kể cả khi câu hỏi có từ "bao nhiêu", "giá trị", "ngày 31/12").
   - Ví dụ:
     + "Nguyên giá trong bất động sản đầu tư nắm giữ cho thuê ngày 31/12 là bao nhiêu" -> Đây là khoản mục phân rã chi tiết trong Thuyết minh Bất động sản đầu tư, Bảng CĐKT chỉ ghi tổng Bất động sản đầu tư.
     + "Thù lao của các thành viên HĐQT là bao nhiêu".
     + "Chi tiết các khoản vay ngân hàng / danh sách ngân hàng cho vay".
     + "Chi phí xây dựng cơ bản dở dang gồm những dự án nào".
     + "Phương pháp khấu hao tài sản cố định", "Chính sách kế toán hàng tồn kho".
     + "Giao dịch với các bên liên quan".

3. "DEEP_ANALYSIS":
   - Khi câu hỏi yêu cầu phân tích tổng hợp, giải thích nguyên nhân biến động, so sánh xu hướng ("Tại sao chi phí tăng", "Phân tích khả năng sinh lời", "Xu hướng nợ vay").

4. "CLARIFY":
   - Chỉ khi câu hỏi THỰC SỰ QUÁ MƠ HỒ, TỐI NGHĨA hoặc KHÔNG THỂ XÁC ĐỊNH ĐƯỢC KHOẢN MỤC NÀO (Ví dụ: "Cho tôi xem số đó đi", "Cái kia bao nhiêu", "Tính toán thế nào", "Dữ liệu đó ra sao").
   - NGUYÊN TẮC BẮT BUỘC (QUAN TRỌNG NHẤT):
     + Hiện tại dự án chỉ lấy context từ một tài liệu PDF cố định (1 doanh nghiệp trong 1 năm nhất định, ví dụ Vinamilk VNM 2025).
     + TUYỆT ĐỐI KHÔNG hỏi người dùng xác nhận hay yêu cầu cung cấp tên công ty hoặc năm.
     + Kể cả khi câu hỏi KHÔNG có tên công ty hay năm nhưng ĐÃ CÓ khoản mục/nội dung rõ ràng (ví dụ: "doanh thu thuần là bao nhiêu", "nguyên giá bất động sản đầu tư là bao nhiêu"), HÃY PHÂN LOẠI NGAY vào NUMERIC_FACT hoặc NOTE_EXPLANATION, KHÔNG ĐƯỢC CHỌN CLARIFY.
     + CHỈ CHỌN CLARIFY khi hoàn toàn không biết người dùng đang muốn hỏi khoản mục tài chính hay thông tin nào.

5. XỬ LÝ HỘI THOẠI ĐA LƯỢT (MULTI-TURN MEMORY):
   - Nếu có "Lịch sử hội thoại trước đó", câu hỏi hiện tại có thể là câu trả lời ngắn hoặc câu hỏi tiếp nối cho câu hỏi/trả lời trước đó của Trợ lý hoặc Người dùng (ví dụ: "toàn bộ", "năm 2024 thì sao", "vậy còn chi phí", "thế còn tài sản", "cho tôi xem chi tiết").
   - Bạn PHẢI kết hợp ngữ cảnh lịch sử hội thoại để hiểu đúng trọn vẹn mục đích của người dùng và khôi phục thành "standalone_query" (câu hỏi độc lập hoàn chỉnh).
   - Ví dụ:
     + Lịch sử:
       Người dùng: "Báo cáo kết quả hoạt động kinh doanh"
       Trợ lý: "Bạn muốn biết thông tin cụ thể nào trong báo cáo kết quả hoạt động kinh doanh? Ví dụ: doanh thu thuần, lợi nhuận gộp..."
       Câu hỏi hiện tại: "toàn bộ"
       => standalone_query: "Toàn bộ báo cáo kết quả hoạt động kinh doanh"
       => intent: "NUMERIC_FACT" (vì đây là BCTC cốt lõi). TUYỆT ĐỐI KHÔNG chọn CLARIFY!
     + Lịch sử:
       Người dùng: "Doanh thu năm 2025 là bao nhiêu?"
       Trợ lý: "Doanh thu thuần năm 2025 là 45.886 tỷ đồng."
       Câu hỏi hiện tại: "thế còn năm 2024"
       => standalone_query: "Doanh thu thuần năm 2024 là bao nhiêu"
       => intent: "NUMERIC_FACT".
   - Nếu không có lịch sử hội thoại hoặc câu hỏi đã độc lập: "standalone_query" giữ nguyên câu hỏi của người dùng.

BẮT BUỘC chỉ trả về định dạng JSON thuần túy (không bọc text giải thích bên ngoài):
{
  "intent": "NUMERIC_FACT" | "NOTE_EXPLANATION" | "DEEP_ANALYSIS" | "CLARIFY",
  "standalone_query": "Câu hỏi độc lập hoàn chỉnh đã khôi phục đầy đủ ngữ nghĩa",
  "reasoning": "Giải thích ngắn gọn lý do phân loại",
  "clarification_prompt": "Câu hỏi làm rõ gửi người dùng nếu intent là CLARIFY (hoặc null nếu câu hỏi đã rõ)"
}"""


def classify_query_llm(
    query: str,
    llm: Any,
    default_company: str | None = None,
    default_year: int | None = None,
    history: list[Any] | None = None,
) -> ClassifyResult:
    """Phân loại ngữ nghĩa câu hỏi tài chính chuyên sâu bằng LLM (Semantic Router).
    
    Có khả năng nhận diện các khoản mục chi tiết thuộc Thuyết minh (Notes),
    hiểu ngữ cảnh lịch sử hội thoại đa lượt (multi-turn),
    và kích hoạt cơ chế Làm rõ (Clarify) khi câu hỏi mơ hồ.
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage

    # Dự án cố định theo context tài liệu hiện tại, trích xuất nếu có hoặc dùng default
    company_match = _COMPANY_RE.search(query)
    company = company_match.group(1) if company_match else default_company
    _NOT_TICKER = {"VND", "ROE", "ROA", "ROS", "EPS", "GDP", "USD", "EUR", "CAGR"}
    if company in _NOT_TICKER:
        company = default_company

    year_match = _YEAR_RE.search(query)
    year = int(year_match.group(1)) if year_match else default_year

    # Xây dựng ngữ cảnh hội thoại trước đó nếu có
    history_context = ""
    if history and len(history) > 1:
        # Lấy tối đa 4 tin nhắn trước tin nhắn hiện tại (tin nhắn cuối cùng là query hiện tại)
        prior_messages = history[:-1][-4:]
        if prior_messages:
            lines = []
            for m in prior_messages:
                role = "Người dùng" if isinstance(m, HumanMessage) else "Trợ lý"
                content = getattr(m, "content", str(m))
                lines.append(f"{role}: {content[:300]}")
            history_context = "Lịch sử hội thoại trước đó:\n" + "\n".join(lines) + "\n\n"

    try:
        user_prompt = f"{history_context}Câu hỏi hiện tại của người dùng:\n\"{query}\""
        messages = [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\n", "", clean_content)
            clean_content = re.sub(r"\n```$", "", clean_content)

        data = json.loads(clean_content)
        raw_intent = data.get("intent", "").upper()
        reasoning = data.get("reasoning", "")
        clarification_prompt = data.get("clarification_prompt")
        standalone_query = data.get("standalone_query") or query

        intent_map = {
            "NUMERIC_FACT": QueryIntent.NUMERIC_FACT,
            "NOTE_EXPLANATION": QueryIntent.NOTE_EXPLANATION,
            "DEEP_ANALYSIS": QueryIntent.DEEP_ANALYSIS,
            "CLARIFY": QueryIntent.CLARIFY,
        }
        intent = intent_map.get(raw_intent, QueryIntent.DEEP_ANALYSIS)
        logger.info(
            "classify_query_llm: intent=%s, standalone_query='%s', reasoning='%s', clarify=%s",
            intent.value,
            standalone_query[:60],
            reasoning[:60],
            bool(clarification_prompt),
        )

        return ClassifyResult(
            intent=intent,
            company=company,
            year=year,
            confidence=0.95,
            clarification_prompt=clarification_prompt,
            standalone_query=standalone_query,
        )
    except Exception as exc:
        logger.warning("classify_query_llm lỗi (%s) -> fallback về heuristic regex", exc)
        return classify_query(query, default_company=company, default_year=year, history=history)


# ---------------------------------------------------------------------------
# LangGraph Node function
# ---------------------------------------------------------------------------
def router_node(state: CopilotState, *, llm: Any = None) -> dict:
    """LangGraph Node: phân loại intent và cập nhật state.

    Nếu có LLM: Sử dụng Semantic LLM Router hiểu trọn vẹn ngữ nghĩa và kích hoạt Clarification Loop.
    Nếu không có LLM hoặc timeout: Fallback tự động về Heuristic Regex.
    """
    query = state["query"]
    messages = state.get("messages", [])
    default_company = state.get("company")
    default_year = state.get("year")

    if llm is not None:
        result = classify_query_llm(
            query=query,
            llm=llm,
            default_company=default_company,
            default_year=default_year,
            history=messages,
        )
    else:
        result = classify_query(
            query=query,
            default_company=default_company,
            default_year=default_year,
            history=messages,
        )

    clarification_prompt = getattr(result, "clarification_prompt", None)
    standalone_query = getattr(result, "standalone_query", None) or query

    logger.info(
        "Intent classified: %s (confidence=%.2f) — company=%s, year=%s, standalone='%s', clarify=%s",
        result.intent,
        result.confidence,
        result.company,
        result.year,
        standalone_query[:60],
        bool(clarification_prompt),
    )

    return {
        "intent": result.intent,
        "company": result.company,
        "year": result.year,
        "query": standalone_query,
        "clarification_prompt": clarification_prompt,
    }


def route_by_intent(state: CopilotState) -> str:
    """Conditional Edge function: trả về tên node tiếp theo dựa vào intent.

    Được dùng trong ``graph.add_conditional_edges()``.
    """
    intent = state.get("intent")
    if intent == QueryIntent.CLARIFY:
        return "clarify_node"
    if intent == QueryIntent.NUMERIC_FACT:
        return "sql_node"
    if intent == QueryIntent.NOTE_EXPLANATION:
        return "vector_node"
    return "hybrid_node"  # DEEP_ANALYSIS
