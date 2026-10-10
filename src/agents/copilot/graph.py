"""src/agents/copilot/graph.py — LangGraph StateGraph Builder (Phase 3.5).

Ghép nối tất cả Nodes và Edges thành đồ thị trạng thái hoàn chỉnh:

  START
    │
    ▼
  router_node  ──(conditional edge: route_by_intent)──►  sql_node
                                                      ►  vector_node
                                                      ►  hybrid_node
    │ (tất cả merge về)
    ▼
  synthesize_node
    │
    ▼
  fact_verify_node  ──(conditional edge: should_correct)──►  synthesize_node (loop)
                                                         ►  citation_verify_node
    │
    ▼
  citation_verify_node
    │
    ▼
  END

Checkpointing: MemorySaver (in-memory) hoặc SqliteSaver (persistent) theo thread_id.
Langfuse: nếu LANGFUSE_SECRET_KEY tồn tại trong env → tự động attach callback.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.copilot.nodes import (
    citation_verify_node,
    clarify_node,
    fact_verify_node,
    hybrid_node,
    should_correct,
    sql_node,
    synthesize_node,
    vector_node,
)
from src.agents.copilot.router import route_by_intent, router_node
from src.agents.copilot.state import CopilotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Langfuse callback (lazy, optional)
# ---------------------------------------------------------------------------
def _get_langfuse_callbacks() -> list[Any]:
    """Trả về Langfuse callback handler nếu cấu hình env đầy đủ."""
    from src.core.config import settings

    if not settings.LANGFUSE_SECRET_KEY or settings.LANGFUSE_SECRET_KEY.startswith("sk-lf-..."):
        return []
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
        handler = LangfuseCallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        logger.info("✅ Langfuse callback đã gắn.")
        return [handler]
    except ImportError:
        logger.warning("langfuse chưa cài — bỏ qua tracing.")
        return []
    except Exception as exc:
        logger.warning("Langfuse init lỗi (%s) — bỏ qua tracing.", exc)
        return []


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_copilot_graph(
    sql_svc: Any,
    retriever: Any,
    llm: Any,
    *,
    use_memory_saver: bool = True,
    langfuse_enabled: bool = True,
) -> Any:
    """Xây dựng và compile LangGraph StateGraph cho Financial Copilot.

    Args:
        sql_svc: SQLiteFactService instance đã khởi tạo.
        retriever: HybridRetriever instance đã khởi tạo.
        llm: LangChain Chat LLM (ChatOpenAI / ChatAnthropic / v.v.).
        use_memory_saver: Dùng MemorySaver để giữ state multi-turn.
        langfuse_enabled: Bật Langfuse tracing nếu env key có.

    Returns:
        compiled_graph: LangGraph CompiledGraph sẵn sàng invoke.
    """
    # --- Bind services vào nodes (Partial Application) ---
    # Đây là cách Lean Architecture inject dependency mà không dùng global state
    _router_node = functools.partial(router_node, llm=llm)
    _sql_node = functools.partial(sql_node, sql_svc=sql_svc, retriever=retriever)
    _vector_node = functools.partial(vector_node, retriever=retriever)
    _hybrid_node = functools.partial(hybrid_node, sql_svc=sql_svc, retriever=retriever)
    _synthesize_node = functools.partial(synthesize_node, llm=llm)
    _fact_verify_node = functools.partial(fact_verify_node, sql_svc=sql_svc)

    # --- Xây StateGraph ---
    builder = StateGraph(CopilotState)

    # Đăng ký tất cả nodes
    builder.add_node("router", _router_node)
    builder.add_node("clarify_node", clarify_node)
    builder.add_node("sql_node", _sql_node)
    builder.add_node("vector_node", _vector_node)
    builder.add_node("hybrid_node", _hybrid_node)
    builder.add_node("synthesize", _synthesize_node)
    builder.add_node("fact_verify", _fact_verify_node)
    builder.add_node("citation_verify", citation_verify_node)

    # --- Edges ---
    # START → router
    builder.add_edge(START, "router")

    # router → [clarify_node | sql_node | vector_node | hybrid_node] (Conditional Edge)
    builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "clarify_node": "clarify_node",
            "sql_node": "sql_node",
            "vector_node": "vector_node",
            "hybrid_node": "hybrid_node",
        },
    )

    # clarify_node → END (Phản hồi làm rõ câu hỏi trực tiếp, không qua retrieval/synthesis)
    builder.add_edge("clarify_node", END)

    # Tất cả retrieval nodes → synthesize
    builder.add_edge("sql_node", "synthesize")
    builder.add_edge("vector_node", "synthesize")
    builder.add_edge("hybrid_node", "synthesize")

    # synthesize → fact_verify
    builder.add_edge("synthesize", "fact_verify")

    # fact_verify → [synthesize (correction loop) | citation_verify] (Conditional Edge)
    builder.add_conditional_edges(
        "fact_verify",
        should_correct,
        {
            "synthesize": "synthesize",
            "citation_verify": "citation_verify",
        },
    )

    # citation_verify → END
    builder.add_edge("citation_verify", END)

    # --- Checkpointing ---
    checkpointer = MemorySaver() if use_memory_saver else None

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info("✅ Copilot LangGraph compiled thành công (checkpointer=%s).", type(checkpointer).__name__ if checkpointer else "None")
    return compiled


# ---------------------------------------------------------------------------
# Convenience: run_query helper
# ---------------------------------------------------------------------------
def run_query(
    graph: Any,
    query: str,
    *,
    thread_id: str = "default",
    company: str | None = None,
    year: int | None = None,
    langfuse_enabled: bool = True,
) -> dict[str, Any]:
    """Chạy một câu hỏi qua Copilot graph và trả về kết quả cuối cùng.

    Args:
        graph: Compiled LangGraph từ ``build_copilot_graph()``.
        query: Câu hỏi tài chính của người dùng.
        thread_id: ID phiên để checkpointing multi-turn.
        company: Mã cổ phiếu mặc định (nếu không trích được từ query).
        year: Năm tài chính mặc định.
        langfuse_enabled: Gắn Langfuse callback hay không.

    Returns:
        Dict chứa ``final_answer``, ``citations``, ``intent``, ``fact_check_status``.
    """
    from langchain_core.messages import HumanMessage

    callbacks = _get_langfuse_callbacks() if langfuse_enabled else []
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
    }

    initial_state: CopilotState = {
        "messages": [HumanMessage(content=query)],
        "thread_id": thread_id,
        "query": query,
        "company": company,
        "year": year,
        # Khởi tạo giá trị mặc định cho các field bắt buộc
        "intent": None,
        "clarification_prompt": None,
        "sql_result": None,
        "sql_context": "",
        "vector_results": [],
        "vector_context": "",
        "draft_answer": "",
        "fact_check_status": None,
        "fact_check_violations": [],
        "correction_attempts": 0,
        "citations": [],
        "final_answer": "",
    }

    logger.info("run_query: thread_id=%s, query='%s'", thread_id, query[:80])

    final_state = graph.invoke(initial_state, config=config)

    return {
        "final_answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "intent": final_state.get("intent"),
        "fact_check_status": final_state.get("fact_check_status"),
        "fact_check_violations": final_state.get("fact_check_violations", []),
        "company": final_state.get("company"),
        "year": final_state.get("year"),
        "sql_context": final_state.get("sql_context", ""),
        "vector_context": final_state.get("vector_context", ""),
        "sql_result": final_state.get("sql_result"),
    }
