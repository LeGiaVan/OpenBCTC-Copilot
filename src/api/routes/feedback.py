"""src/api/routes/feedback.py — Human-in-the-Loop Feedback Collector (Phase 4.3).

Endpoint tiếp nhận phản hồi đánh giá (Upvote / Downvote, rating, comment) từ người dùng & chuyên viên:
  - Ghi nhận vào file log kiểm toán `logs/user_feedback.jsonl`.
  - Tự động đồng bộ điểm số phản hồi (Score) lên Langfuse traces để tinh chỉnh Prompt.
  - Cung cấp API thống kê tỷ lệ hài lòng (Satisfaction Rate) phục vụ Dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.observability.langfuse_client import log_score

logger = logging.getLogger("Copilot.Feedback")
router = APIRouter(prefix="/feedback", tags=["Feedback"])

FEEDBACK_LOG_PATH = Path("logs/user_feedback.jsonl")


class FeedbackRequest(BaseModel):
    """Payload gửi phản hồi từ giao diện người dùng."""
    query: str = Field(description="Câu hỏi người dùng đã gửi")
    answer: str = Field(description="Câu trả lời của AI tương ứng")
    feedback_type: Literal["UPVOTE", "DOWNVOTE"] = Field(description="Loại đánh giá: UPVOTE hoặc DOWNVOTE")
    rating: int | None = Field(default=None, ge=1, le=5, description="Điểm đánh giá từ 1 đến 5 sao (tuỳ chọn)")
    comment: str | None = Field(default=None, description="Ý kiến đóng góp / lý do không hài lòng")
    category: str | None = Field(default=None, description="Chủ đề đánh giá (ví dụ: 'SAI_SO_LIEU', 'SAI_CITE', 'MO_HO')")
    thread_id: str | None = Field(default=None, description="ID phiên hội thoại")
    message_id: str | None = Field(default=None, description="ID định danh tin nhắn trong UI")
    trace_id: str | None = Field(default=None, description="ID trace trên Langfuse nếu có")
    citations: list[dict[str, Any]] | None = Field(default=None, description="Danh sách trích dẫn trong câu trả lời")


@router.post("")
async def submit_feedback(request: Request, payload: FeedbackRequest):
    """Tiếp nhận phản hồi Upvote / Downvote và lưu vết kiểm toán."""
    feedback_id = f"fb_{int(time.time() * 1000)}"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    score_val = 1.0 if payload.feedback_type == "UPVOTE" else 0.0

    record = {
        "feedback_id": feedback_id,
        "timestamp": timestamp,
        "feedback_type": payload.feedback_type,
        "score": score_val,
        "rating": payload.rating,
        "comment": payload.comment,
        "category": payload.category,
        "query": payload.query,
        "answer_snippet": payload.answer[:300] if payload.answer else "",
        "thread_id": payload.thread_id,
        "message_id": payload.message_id,
        "trace_id": payload.trace_id,
        "citations_count": len(payload.citations or []),
    }

    # 1. Lưu cục bộ vào logs/user_feedback.jsonl
    try:
        FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Lỗi khi ghi feedback ra file: %s", exc)

    # 2. Đồng bộ điểm số lên Langfuse
    target_trace = payload.trace_id or payload.thread_id
    if target_trace:
        log_score(
            trace_id=target_trace,
            name="user_feedback",
            value=score_val,
            comment=f"{payload.feedback_type}: {payload.comment or 'Không có ghi chú'}",
        )

    logger.info("✓ Đã lưu feedback %s (%s) cho câu hỏi: '%s'", feedback_id, payload.feedback_type, payload.query[:50])

    return {
        "status": "SUCCESS",
        "message": "Cảm ơn bạn đã đóng góp phản hồi giúp hoàn thiện hệ thống!",
        "feedback_id": feedback_id,
    }


@router.get("/stats")
async def get_feedback_stats():
    """Thống kê tổng quan về phản hồi người dùng."""
    if not FEEDBACK_LOG_PATH.exists():
        return {
            "total": 0,
            "upvotes": 0,
            "downvotes": 0,
            "satisfaction_rate": 1.0,
            "recent": [],
        }

    total = 0
    upvotes = 0
    downvotes = 0
    recent = []

    try:
        with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                total += 1
                if data.get("feedback_type") == "UPVOTE":
                    upvotes += 1
                elif data.get("feedback_type") == "DOWNVOTE":
                    downvotes += 1
                recent.append(data)
    except Exception as exc:
        logger.warning("Lỗi đọc feedback stats: %s", exc)

    rate = (upvotes / total) if total > 0 else 1.0
    return {
        "total": total,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "satisfaction_rate": round(rate, 4),
        "recent": recent[-10:],
    }
