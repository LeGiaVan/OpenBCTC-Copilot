"""src/agents/copilot/nodes.py — LangGraph Nodes cho Copilot (Phase 3.2-3.4).

Thiết kế Lean: mỗi Node là hàm thuần nhận state → trả dict patch.
Business logic nặng nằm hoàn toàn trong Fat Services.

Nodes:
  - sql_node          : Gọi SQLiteFactService → điền sql_result, sql_context
  - vector_node       : Gọi HybridRetriever → điền vector_results, vector_context, citations
  - hybrid_node       : Gọi cả SQL + Vector song song → điền cả hai
  - synthesize_node   : Gọi LLM → sinh draft_answer từ context
  - fact_verify_node  : Scan số trong draft_answer → đối chiếu SQL DB (Phase 3.3)
  - citation_verify_node: Validate citations, gắn tags → sinh final_answer (Phase 3.4)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from src.agents.copilot.state import CopilotState, FactCheckStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex trích xuất số tài chính trong câu trả lời LLM
# ---------------------------------------------------------------------------
_FINANCIAL_NUM_RE = re.compile(
    r"(?<!\w)"                          # Không có ký tự trước (tránh false positive)
    r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)"  # Số có dấu phẩy/chấm phân cách nghìn
    r"\s*(?:tỷ|triệu|nghìn|đồng|VND|%)?"
    r"(?!\w)",
    re.IGNORECASE,
)

_MAX_CORRECTION_ATTEMPTS = 2


# ===========================================================================
# Phase 3.2 — Fat Service Caller Nodes
# ===========================================================================
def sql_node(state: CopilotState, *, sql_svc: Any) -> dict:
    """Node: Gọi SQLiteFactService và format ngữ cảnh SQL.

    Args:
        state: LangGraph state hiện tại.
        sql_svc: SQLiteFactService instance (inject từ graph builder).

    Returns:
        Patch dict cập nhật sql_result và sql_context.
    """
    from src.services.citation_formatter import CitationFormatter

    company = state.get("company") or "VNM"
    year = state.get("year") or 2025
    query = state["query"]
    formatter = CitationFormatter()

    logger.info("sql_node: company=%s, year=%s, query='%s'", company, year, query[:60])

    try:
        # Lấy snapshot tổng quan — đủ cho hầu hết câu hỏi NUMERIC_FACT
        snapshot = sql_svc.get_summary_snapshot(company=company, year=year)
        sql_context = formatter.format_sql_context(snapshot, query_type="snapshot")
        logger.info("sql_node: snapshot OK — %d facts, %d ratios", len(snapshot.get("key_facts", {})), len(snapshot.get("ratios", {})))
        return {
            "sql_result": snapshot,
            "sql_context": sql_context,
        }
    except Exception as exc:
        logger.error("sql_node lỗi: %s", exc)
        return {
            "sql_result": {"error": str(exc)},
            "sql_context": f"⚠️ Không thể truy vấn SQL: {exc}",
        }


def vector_node(state: CopilotState, *, retriever: Any) -> dict:
    """Node: Gọi HybridRetriever và format ngữ cảnh vector.

    Args:
        state: LangGraph state hiện tại.
        retriever: HybridRetriever instance (inject từ graph builder).

    Returns:
        Patch dict cập nhật vector_results, vector_context, citations.
    """
    from src.services.citation_formatter import CitationFormatter

    company = state.get("company")
    year = state.get("year")
    query = state["query"]
    formatter = CitationFormatter()

    logger.info("vector_node: company=%s, year=%s, query='%s'", company, year, query[:60])

    try:
        results, raw_citations = retriever.retrieve_with_citations(
            query=query,
            company=company,
            year=year,
            top_k=5,
        )

        vector_context = formatter.build_context_block(results, max_tokens_estimate=2500)
        valid_citations = formatter.filter_valid_citations(raw_citations)
        citations_payload = [
            {
                "citation_id": c.citation_id,
                "block_id": c.block_id,
                "source_type": c.source_type,
                "page": c.page,
                "bbox": c.bbox,
                "snippet": c.snippet,
                "company": c.company,
                "year": c.year,
                "confidence": c.confidence,
            }
            for c in valid_citations
        ]

        logger.info("vector_node: %d results → %d valid citations", len(results), len(citations_payload))
        return {
            "vector_results": [r.__dict__ if hasattr(r, "__dict__") else r for r in results],
            "vector_context": vector_context,
            "citations": citations_payload,
        }
    except Exception as exc:
        logger.error("vector_node lỗi: %s", exc)
        return {
            "vector_results": [],
            "vector_context": f"⚠️ Không thể truy vấn Vector DB: {exc}",
            "citations": [],
        }


def hybrid_node(state: CopilotState, *, sql_svc: Any, retriever: Any) -> dict:
    """Node: Gọi SQL + Vector song song cho câu hỏi DEEP_ANALYSIS.

    Thứ tự: SQL trước (số liệu định lượng nền) → Vector sau (giải trình ngữ cảnh).
    """
    sql_patch = sql_node(state, sql_svc=sql_svc)
    vector_patch = vector_node(state, retriever=retriever)
    return {**sql_patch, **vector_patch}


# ===========================================================================
# Phase 3.2 — Synthesis Node (LLM Call)
# ===========================================================================
_SYSTEM_PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích tài chính doanh nghiệp chuyên sâu, \
am hiểu sâu Thông tư 200/2014/TT-BTC và chuẩn mực kế toán Việt Nam (VAS).

**Nguyên tắc bắt buộc:**
1. Chỉ sử dụng số liệu được cung cấp trong phần ngữ cảnh bên dưới.
2. KHÔNG bịa đặt, KHÔNG ngoại suy số liệu không có trong ngữ cảnh.
3. Mỗi khi trích dẫn thông tin từ báo cáo, gắn tag [[cite_N]] tương ứng.
4. Nếu không tìm thấy thông tin để trả lời, hãy nói rõ "Không có dữ liệu trong báo cáo."
5. Định dạng số theo chuẩn Việt Nam: 14.352 tỷ đồng (không dùng triệu USD).

{sql_context}

{vector_context}"""


def synthesize_node(state: CopilotState, *, llm: Any) -> dict:
    """Node: Gọi LLM tổng hợp câu trả lời từ context SQL + Vector.

    Args:
        state: LangGraph state.
        llm: LangChain LLM instance (ChatOpenAI / ChatAnthropic / v.v.).

    Returns:
        Patch dict cập nhật draft_answer.
    """
    sql_context = state.get("sql_context", "")
    vector_context = state.get("vector_context", "")

    # Nếu cả hai rỗng → không có ngữ cảnh để trả lời
    if not sql_context and not vector_context:
        return {
            "draft_answer": "Không tìm thấy dữ liệu liên quan trong báo cáo tài chính.",
            "fact_check_status": FactCheckStatus.SKIPPED,
        }

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        sql_context=f"## 📊 Dữ liệu tài chính từ SQL\n{sql_context}" if sql_context else "",
        vector_context=vector_context if vector_context else "",
    ).strip()

    # Lấy câu hỏi từ messages (HumanMessage cuối cùng)
    query = state["query"]

    correction_context = ""
    if state.get("fact_check_status") == FactCheckStatus.FAILED:
        violations = state.get("fact_check_violations", [])
        violation_strs = [
            f"  - Số '{v.get('number_in_answer')}' trong câu trả lời ≠ giá trị DB '{v.get('db_value')}' (concept: {v.get('concept', 'N/A')})"
            for v in violations
        ]
        correction_context = (
            "\n\n⚠️ **YÊU CẦU SỬA LẠI:** Câu trả lời trước có các số liệu không chính xác:\n"
            + "\n".join(violation_strs)
            + "\nHãy sinh lại câu trả lời CHỈ dùng số liệu đúng từ ngữ cảnh."
        )

    messages = [
        SystemMessage(content=system_prompt + correction_context),
        HumanMessage(content=query),
    ]

    logger.info("synthesize_node: gọi LLM (attempt=%d)", state.get("correction_attempts", 0))

    try:
        response = llm.invoke(messages)
        draft = response.content if hasattr(response, "content") else str(response)
        logger.info("synthesize_node: nhận được %d ký tự.", len(draft))
        return {"draft_answer": draft}
    except Exception as exc:
        logger.error("synthesize_node lỗi LLM: %s", exc)
        return {
            "draft_answer": f"Lỗi khi sinh câu trả lời: {exc}",
            "fact_check_status": FactCheckStatus.SKIPPED,
        }


# ===========================================================================
# Phase 3.3 — Fact-Consistency Verifier & Correction Loop
# ===========================================================================
def fact_verify_node(state: CopilotState, *, sql_svc: Any) -> dict:
    """Node: Kiểm toán số liệu trong draft_answer đối chiếu với SQL DB.

    Thuật toán:
      1. Regex quét tất cả số tài chính trong draft_answer.
      2. Với mỗi số, tìm fact gần nhất trong snapshot SQL.
      3. Nếu số lệch > 1% → ghi vào violations.
      4. Cập nhật fact_check_status: PASSED / FAILED / SKIPPED.

    Note: Đây là heuristic đơn giản. Không phải 100% chính xác do ambiguity
    (cùng số có thể xuất hiện ở nhiều context). Mục đích là bắt hallucination thô.
    """
    draft = state.get("draft_answer", "")
    sql_result = state.get("sql_result") or {}

    if not draft or "Lỗi khi sinh" in draft:
        return {"fact_check_status": FactCheckStatus.SKIPPED, "fact_check_violations": []}

    # Lấy bảng số liệu chuẩn từ snapshot
    key_facts: dict[str, float] = {}
    if "key_facts" in sql_result:
        for concept, val in sql_result["key_facts"].items():
            if val is not None:
                key_facts[concept] = float(val)
    if "ratios" in sql_result:
        for name, val in sql_result["ratios"].items():
            if val is not None:
                key_facts[f"ratio:{name}"] = float(val)

    if not key_facts:
        # Không có dữ liệu SQL để so sánh → skip
        return {"fact_check_status": FactCheckStatus.SKIPPED, "fact_check_violations": []}

    # Tìm các số trong câu trả lời
    raw_numbers = _FINANCIAL_NUM_RE.findall(draft)
    if not raw_numbers:
        return {"fact_check_status": FactCheckStatus.SKIPPED, "fact_check_violations": []}

    # Chuẩn hoá: bỏ dấu phẩy/chấm phân cách nghìn → float
    parsed_numbers: list[float] = []
    for raw in raw_numbers:
        try:
            # Chuẩn hoá số: VN dùng dấu chấm phân nghìn, phẩy thập phân
            cleaned = raw.replace(".", "").replace(",", ".")
            parsed_numbers.append(float(cleaned))
        except ValueError:
            pass

    violations: list[dict[str, Any]] = []

    for num_in_answer in parsed_numbers:
        # Tìm fact gần nhất (tolerance 1%)
        for concept, db_val in key_facts.items():
            if db_val == 0:
                continue
            # So sánh ở đơn vị tỷ (nhiều BCTC VN đơn vị triệu đồng → tỷ = x1000)
            # Thử khớp cả ở các scale: x1, x1e6, x1e9
            for scale in [1, 1e6, 1e9]:
                scaled_db = db_val / scale
                if scaled_db == 0:
                    continue
                diff_pct = abs((num_in_answer - scaled_db) / scaled_db)
                if diff_pct < 0.01:  # Khớp trong 1% → OK
                    break
            else:
                # Không khớp ở bất kỳ scale nào với concept này → tiếp tục
                continue
            # Nếu khớp → break vòng concept → num này OK
            break
        else:
            # num_in_answer không khớp với BẤT KỲ fact nào trong DB
            # Chỉ vi phạm nếu số đủ lớn (> 100 triệu = 100) để tránh false positive với %, năm, trang
            if num_in_answer > 100:
                violations.append({
                    "number_in_answer": num_in_answer,
                    "db_value": None,
                    "concept": "UNKNOWN",
                })

    if violations:
        logger.warning("fact_verify_node: %d violations phát hiện.", len(violations))
        return {
            "fact_check_status": FactCheckStatus.FAILED,
            "fact_check_violations": violations,
            "correction_attempts": state.get("correction_attempts", 0) + 1,
        }

    logger.info("fact_verify_node: PASSED — không phát hiện số liệu sai.")
    return {
        "fact_check_status": FactCheckStatus.PASSED,
        "fact_check_violations": [],
    }


def should_correct(state: CopilotState) -> str:
    """Conditional Edge: có cần correction loop không?

    Returns:
        'synthesize' nếu cần sinh lại (FAILED + còn attempt).
        'citation_verify' nếu đã OK hoặc hết lần thử.
    """
    status = state.get("fact_check_status")
    attempts = state.get("correction_attempts", 0)

    if status == FactCheckStatus.FAILED and attempts < _MAX_CORRECTION_ATTEMPTS:
        logger.warning("Correction loop kích hoạt (attempt %d/%d).", attempts, _MAX_CORRECTION_ATTEMPTS)
        return "synthesize"

    return "citation_verify"


# ===========================================================================
# Phase 3.4 — Citation Verifier Node
# ===========================================================================
def citation_verify_node(state: CopilotState) -> dict:
    """Node: Validate citations, inject tags vào draft_answer → sinh final_answer.

    Không phụ thuộc service nào — dùng CitationFormatter pure class.
    """
    from src.models.citation import CitationWithBBox
    from src.services.citation_formatter import CitationFormatter

    draft = state.get("draft_answer", "")
    raw_citations_dicts = state.get("citations", [])
    formatter = CitationFormatter()

    # Chuyển dict → CitationWithBBox object
    citation_objects = []
    for c in raw_citations_dicts:
        try:
            citation_objects.append(CitationWithBBox(**c))
        except Exception as e:
            logger.warning("citation_verify: bỏ qua citation không hợp lệ: %s", e)

    # Validate và filter
    valid_citations = formatter.filter_valid_citations(citation_objects)

    # Inject tags và tạo final answer
    final_answer = formatter.inject_citation_tags(draft, valid_citations, auto_inject=True)

    # Cập nhật lại citations dict (chỉ giữ valid)
    final_citations = [
        {
            "citation_id": c.citation_id,
            "block_id": c.block_id,
            "source_type": c.source_type,
            "page": c.page,
            "bbox": c.bbox,
            "snippet": c.snippet,
            "company": c.company,
            "year": c.year,
            "confidence": c.confidence,
        }
        for c in valid_citations
    ]

    logger.info(
        "citation_verify_node: %d valid citations, final_answer=%d ký tự.",
        len(final_citations),
        len(final_answer),
    )
    return {
        "final_answer": final_answer,
        "citations": final_citations,
    }
