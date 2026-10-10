"""src/observability/langfuse_client.py — Langfuse Telemetry, Tracing & Scoring Client.

Cung cấp các tiện ích tích hợp sâu với Langfuse:
  - CallbackHandler cho LangGraph để trace tự động từng Node, LLM Call.
  - Client ghi nhận điểm số đánh giá (Evaluation Scores: Faithfulness, SQL Accuracy, Citation Precision).
  - Client tiếp nhận Human Feedback (Upvote / Downvote) từ chuyên viên kế toán.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import settings

logger = logging.getLogger(__name__)

_langfuse_instance = None
_client_initialized = False


def is_langfuse_configured() -> bool:
    """Kiểm tra xem Langfuse đã được cấu hình với key hợp lệ hay chưa."""
    sec = settings.LANGFUSE_SECRET_KEY
    pub = settings.LANGFUSE_PUBLIC_KEY
    if not sec or not pub:
        return False
    if sec.startswith("sk-lf-...") or pub.startswith("pk-lf-..."):
        return False
    return True


def get_langfuse_client() -> Any | None:
    """Khởi tạo và trả về Langfuse client singleton."""
    global _langfuse_instance, _client_initialized
    if _client_initialized:
        return _langfuse_instance

    _client_initialized = True
    if not is_langfuse_configured():
        logger.info("Langfuse chưa cấu hình credentials thực tế — hoạt động ở chế độ offline.")
        _langfuse_instance = None
        return None

    try:
        from langfuse import Langfuse
        _langfuse_instance = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info("✅ Langfuse client đã khởi tạo thành công (host=%s).", settings.LANGFUSE_HOST)
        return _langfuse_instance
    except Exception as exc:
        logger.warning("Không thể khởi tạo Langfuse client: %s", exc)
        _langfuse_instance = None
        return None


def get_langfuse_callback() -> Any | None:
    """Trả về CallbackHandler cho LangGraph / LangChain."""
    if not is_langfuse_configured():
        return None
    try:
        from langfuse.callback import CallbackHandler
        handler = CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        return handler
    except Exception as exc:
        logger.warning("Không thể khởi tạo Langfuse CallbackHandler: %s", exc)
        return None


def log_score(
    trace_id: str | None,
    name: str,
    value: float,
    comment: str | None = None,
    observation_id: str | None = None,
) -> bool:
    """Ghi nhận điểm số (Evaluation Score hoặc User Feedback) vào trace trên Langfuse.

    Args:
        trace_id: ID của trace trên Langfuse (hoặc thread_id).
        name: Tên chỉ số (ví dụ: 'user_feedback', 'sql_accuracy', 'faithfulness', 'citation_precision').
        value: Giá trị số (ví dụ: 1.0 cho upvote/pass, 0.0 cho downvote/fail).
        comment: Ghi chú, giải thích hoặc phản hồi chi tiết.
        observation_id: ID của observation cụ thể nếu có.

    Returns:
        True nếu ghi thành công, False nếu Langfuse offline hoặc lỗi.
    """
    if not trace_id:
        logger.debug("log_score: trace_id rỗng, bỏ qua.")
        return False

    client = get_langfuse_client()
    if not client:
        logger.debug("Langfuse client offline — điểm '%s': %.2f lưu cục bộ.", name, value)
        return False

    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            observation_id=observation_id,
        )
        client.flush()
        logger.info("✅ Đã gửi điểm Langfuse: trace_id=%s, %s=%.2f", trace_id, name, value)
        return True
    except Exception as exc:
        logger.warning("Gửi điểm lên Langfuse thất bại: %s", exc)
        return False
