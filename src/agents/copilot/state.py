"""src/agents/copilot/state.py — LangGraph State Schema cho Copilot (Phase 3).

Thiết kế theo Lean Agent: State chỉ lưu dữ liệu trung gian cần thiết,
không nhúng business logic. Toàn bộ xử lý nằm trong Fat Services.

Vòng đời của 1 turn:
  HumanMessage → classify_intent → [sql_node | vector_node | hybrid_node]
               → synthesize → fact_verify → citation_verify → END
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class QueryIntent(str, Enum):
    """3 nhóm phân loại câu hỏi tài chính."""
    NUMERIC_FACT = "NUMERIC_FACT"
    """Hỏi số liệu đơn lẻ / chỉ số → chuyển thẳng SQL Tool."""
    NOTE_EXPLANATION = "NOTE_EXPLANATION"
    """Hỏi chính sách kế toán, thuyết minh, cơ cấu chi tiết → Qdrant Retriever."""
    DEEP_ANALYSIS = "DEEP_ANALYSIS"
    """Hỏi nguyên nhân, tương quan, phân tích tổng hợp → SQL + Vector + LLM Synthesis."""


class FactCheckStatus(str, Enum):
    """Trạng thái kiểm toán số liệu trong câu trả lời LLM."""
    PENDING = "PENDING"       # Chưa kiểm tra
    PASSED = "PASSED"         # Tất cả số khớp với DB
    FAILED = "FAILED"         # Phát hiện số không khớp → kích hoạt correction loop
    SKIPPED = "SKIPPED"       # Câu trả lời không chứa số liệu cụ thể


class CopilotState(TypedDict):
    """LangGraph State cho Copilot một lượt truy vấn tài chính.

    Quy ước đặt tên:
      - ``messages``: lịch sử hội thoại, dùng Annotated[list, add_messages] để tự merge.
      - ``*_context``: chuỗi ngữ cảnh sẽ chèn vào System Prompt.
      - ``*_result``: kết quả thô từ các Fat Services.
      - ``fact_check_*``: dữ liệu liên quan đến vòng kiểm toán số liệu.
    """

    # ------------------------------------------------------------------
    # Hội thoại
    # ------------------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    """Lịch sử tin nhắn (HumanMessage + AIMessage), tự động append qua add_messages."""

    thread_id: str
    """ID phiên hội thoại để checkpointing multi-turn."""

    # ------------------------------------------------------------------
    # Trích xuất từ câu hỏi
    # ------------------------------------------------------------------
    query: str
    """Câu hỏi thô của người dùng (turn hiện tại)."""

    company: str | None
    """Mã cổ phiếu trích xuất từ câu hỏi (ví dụ: 'VNM')."""

    year: int | None
    """Năm tài chính trích xuất từ câu hỏi (ví dụ: 2025)."""

    # ------------------------------------------------------------------
    # Phân loại & Định tuyến
    # ------------------------------------------------------------------
    intent: QueryIntent | None
    """Nhãn phân loại câu hỏi: NUMERIC_FACT | NOTE_EXPLANATION | DEEP_ANALYSIS."""

    # ------------------------------------------------------------------
    # Kết quả từ Fat Services
    # ------------------------------------------------------------------
    sql_result: dict[str, Any] | None
    """Kết quả thô từ SQLiteFactService (get_fact, compare_periods, get_ratios...)."""

    sql_context: str
    """Ngữ cảnh SQL đã định dạng để chèn vào System Prompt."""

    vector_results: list[dict[str, Any]]
    """Danh sách RetrievalResult thô từ HybridRetriever."""

    vector_context: str
    """Ngữ cảnh vector đã định dạng để chèn vào System Prompt."""

    # ------------------------------------------------------------------
    # Đầu ra LLM & Kiểm toán
    # ------------------------------------------------------------------
    draft_answer: str
    """Câu trả lời nháp từ LLM (chưa qua fact-check)."""

    fact_check_status: FactCheckStatus
    """Trạng thái kiểm toán số liệu."""

    fact_check_violations: list[dict[str, Any]]
    """Danh sách vi phạm: {'number_in_answer': ..., 'db_value': ..., 'concept': ...}."""

    correction_attempts: int
    """Số lần đã thử correction loop (giới hạn tối đa 2 lần)."""

    # ------------------------------------------------------------------
    # Visual Grounding
    # ------------------------------------------------------------------
    citations: list[dict[str, Any]]
    """Danh sách citation đã validate kèm bbox, page, snippet."""

    final_answer: str
    """Câu trả lời cuối đã gắn citation tags và sẵn sàng trả về API."""
