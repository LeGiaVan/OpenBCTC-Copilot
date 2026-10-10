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
    r"(?<!\w)"                          # Không có ký tự trước
    r"(\d{1,3}(?:[.,\s\u202f\u00a0]\d{3})*(?:[.,]\d+)?)"  # Cụm 1: Số lượng
    r"\s*(tỷ|triệu|nghìn|đồng|vnd|%)?"      # Cụm 2: Đơn vị (Optional)
    r"(?!\w)",
    re.IGNORECASE,
)

_MAX_CORRECTION_ATTEMPTS = 2


def _get_statement_name_by_page(page: int) -> str:
    """Trả về tên 3 Báo cáo tài chính cốt lõi dựa theo số trang PDF (Thông tư 200/2014/TT-BTC)."""
    if page in (7, 8, 9):
        return "Báo cáo tình hình tài chính (Bảng Cân đối kế toán)"
    elif page == 10:
        return "Báo cáo kết quả hoạt động kinh doanh"
    elif page in (11, 12):
        return "Báo cáo lưu chuyển tiền tệ"
    elif page <= 6:
        return "Báo cáo tài chính"
    else:
        return "Thuyết minh BCTC"


# ===========================================================================
# Phase 3.2 — Fat Service Caller Nodes
# ===========================================================================
def sql_node(state: CopilotState, *, sql_svc: Any, retriever: Any = None) -> dict:
    """Node: Gọi SQLiteFactService và format ngữ cảnh SQL kèm Visual Citations.
    Nếu khoản mục hỏi là chi tiết chuyên sâu thuộc Thuyết minh -> Tự động Fallback sang Vector Search.

    Args:
        state: LangGraph state hiện tại.
        sql_svc: SQLiteFactService instance (inject từ graph builder).
        retriever: HybridRetriever instance (optional, dùng cho fallback).

    Returns:
        Patch dict cập nhật sql_result, sql_context và citations.
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

        # Trích xuất facts liên quan đến query để tạo citations kèm số trang
        all_facts = []
        if hasattr(sql_svc, "get_all_facts"):
            try:
                all_facts = sql_svc.get_all_facts(company=company, year=year, period=str(year))
                if not all_facts:
                    all_facts = sql_svc.get_all_facts(company=company, year=year, period="current")
                if not all_facts:
                    all_facts = sql_svc.get_all_facts(company=company, year=year)
            except Exception as e:
                logger.warning("sql_node: không thể lấy all_facts: %s", e)

        q_lower = query.lower()
        CONCEPT_KEYWORD_MAP = {
            "lợi nhuận gộp": ["GROSS_PROFIT"],
            "doanh thu thuần": ["NET_REVENUE"],
            "doanh thu bán hàng": ["GROSS_REVENUE"],
            "doanh thu": ["NET_REVENUE", "GROSS_REVENUE", "FINANCIAL_INCOME"],
            "giá vốn": ["COGS"],
            "tổng tài sản": ["TOTAL_ASSETS"],
            "tài sản ngắn hạn": ["CURRENT_ASSETS"],
            "tài sản dài hạn": ["NON_CURRENT_ASSETS"],
            "vốn chủ sở hữu": ["TOTAL_EQUITY", "EQUITY", "OWNERS_EQUITY_TOTAL"],
            "vốn chủ": ["TOTAL_EQUITY", "EQUITY"],
            "nợ phải trả": ["TOTAL_LIABILITIES", "LIABILITIES"],
            "nợ ngắn hạn": ["CURRENT_LIABILITIES"],
            "nợ dài hạn": ["NON_CURRENT_LIABILITIES"],
            "tiền và tương đương tiền": ["CASH_AND_EQUIVALENTS"],
            "tiền": ["CASH_AND_EQUIVALENTS", "CASH"],
            "đầu tư tài chính ngắn hạn": ["SHORT_TERM_INVESTMENTS", "HELD_TO_MATURITY_INVESTMENTS_SHORT"],
            "đầu tư ngắn hạn": ["SHORT_TERM_INVESTMENTS"],
            "phải thu ngắn hạn": ["SHORT_TERM_RECEIVABLES"],
            "phải thu khách hàng": ["SHORT_TERM_TRADE_RECEIVABLES"],
            "hàng tồn kho": ["INVENTORIES"],
            "lợi nhuận sau thuế": ["NET_INCOME", "NET_PROFIT"],
            "lợi nhuận trước thuế": ["PROFIT_BEFORE_TAX"],
            "lợi nhuận thuần": ["OPERATING_PROFIT"],
            "lợi nhuận hoạt động": ["OPERATING_PROFIT"],
            "lợi nhuận": ["GROSS_PROFIT", "NET_PROFIT", "NET_INCOME", "OPERATING_PROFIT"],
            "chi phí tài chính": ["FINANCIAL_EXPENSES"],
            "chi phí bán hàng": ["SELLING_EXPENSES"],
            "kết quả hoạt động kinh doanh": ["GROSS_REVENUE", "NET_REVENUE", "COGS", "GROSS_PROFIT", "FINANCIAL_INCOME", "FINANCIAL_EXPENSES", "SELLING_EXPENSES", "ADMIN_EXPENSES", "OPERATING_PROFIT", "PROFIT_BEFORE_TAX", "NET_INCOME"],
            "kết quả kinh doanh": ["GROSS_REVENUE", "NET_REVENUE", "COGS", "GROSS_PROFIT", "FINANCIAL_INCOME", "FINANCIAL_EXPENSES", "SELLING_EXPENSES", "ADMIN_EXPENSES", "OPERATING_PROFIT", "PROFIT_BEFORE_TAX", "NET_INCOME"],
            "bảng cân đối kế toán": ["TOTAL_ASSETS", "CURRENT_ASSETS", "NON_CURRENT_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY", "CASH_AND_EQUIVALENTS"],
            "cân đối kế toán": ["TOTAL_ASSETS", "CURRENT_ASSETS", "NON_CURRENT_ASSETS", "TOTAL_LIABILITIES", "TOTAL_EQUITY", "CASH_AND_EQUIVALENTS"],
            "lưu chuyển tiền tệ": ["CF_NET_OPERATING", "CASH_FROM_OPERATIONS", "CF_NET_INVESTING", "CF_NET_FINANCING", "CF_NET_CHANGE", "CF_ENDING_CASH"],
            "lưu chuyển tiền": ["CASH_FROM_OPERATIONS", "CF_NET_OPERATING", "CF_NET_INVESTING", "CF_NET_FINANCING"],
            "dòng tiền": ["CASH_FROM_OPERATIONS", "CF_NET_OPERATING"],
            "khấu hao": ["DEPRECIATION", "CF_DEPRECIATION"],
        }

        # Nhận diện nếu câu hỏi hỏi tổng quan toàn bộ chỉ số / hệ CSDL
        is_all_metrics = any(
            w in q_lower
            for w in [
                "tất cả", "toàn bộ", "các chỉ số", "danh sách",
                "hệ cơ sở dữ liệu", "cơ sở dữ liệu", "database", "sql",
                "tổng quan", "báo cáo tài chính cốt lõi", "bảng báo cáo",
                "kết quả kinh doanh", "báo cáo kết quả", "cân đối kế toán"
            ]
        )

        selected_facts = []
        seen_concepts = set()

        CORE_METRIC_ORDER = [
            # 1. Bảng cân đối kế toán (Trang 7-9)
            "CURRENT_ASSETS",
            "CASH_AND_EQUIVALENTS",
            "SHORT_TERM_INVESTMENTS",
            "SHORT_TERM_RECEIVABLES",
            "INVENTORIES",
            "NON_CURRENT_ASSETS",
            "TOTAL_ASSETS",
            "CURRENT_LIABILITIES",
            "NON_CURRENT_LIABILITIES",
            "LIABILITIES",
            "TOTAL_LIABILITIES",
            "EQUITY",
            "TOTAL_EQUITY",
            # 2. Báo cáo kết quả hoạt động kinh doanh (Trang 10)
            "GROSS_REVENUE",
            "NET_REVENUE",
            "COGS",
            "GROSS_PROFIT",
            "FINANCIAL_INCOME",
            "FINANCIAL_EXPENSES",
            "SELLING_EXPENSES",
            "ADMIN_EXPENSES",
            "OPERATING_PROFIT",
            "PROFIT_BEFORE_TAX",
            "NET_PROFIT",
            "NET_INCOME",
            # 3. Báo cáo lưu chuyển tiền tệ (Trang 11-12)
            "CF_NET_OPERATING",
            "CASH_FROM_OPERATIONS",
            "CF_NET_INVESTING",
            "CF_NET_FINANCING",
            "CF_NET_CHANGE",
            "CF_ENDING_CASH",
        ]

        INCOME_STATEMENT_METRICS = [
            "GROSS_REVENUE", "NET_REVENUE", "COGS", "GROSS_PROFIT",
            "FINANCIAL_INCOME", "FINANCIAL_EXPENSES", "SELLING_EXPENSES",
            "ADMIN_EXPENSES", "OPERATING_PROFIT", "PROFIT_BEFORE_TAX",
            "NET_PROFIT", "NET_INCOME"
        ]
        BALANCE_SHEET_METRICS = [
            "CURRENT_ASSETS", "CASH_AND_EQUIVALENTS", "SHORT_TERM_INVESTMENTS",
            "SHORT_TERM_RECEIVABLES", "INVENTORIES", "NON_CURRENT_ASSETS",
            "TOTAL_ASSETS", "CURRENT_LIABILITIES", "NON_CURRENT_LIABILITIES",
            "LIABILITIES", "TOTAL_LIABILITIES", "EQUITY", "TOTAL_EQUITY"
        ]
        CASH_FLOW_METRICS = [
            "CF_NET_OPERATING", "CASH_FROM_OPERATIONS", "CF_NET_INVESTING",
            "CF_NET_FINANCING", "CF_NET_CHANGE", "CF_ENDING_CASH"
        ]

        if any(w in q_lower for w in ["kết quả hoạt động kinh doanh", "kết quả kinh doanh"]) and all_facts:
            target_metric_list = INCOME_STATEMENT_METRICS
            for c in target_metric_list:
                for f in all_facts:
                    if f.concept == c and f.concept not in seen_concepts:
                        seen_concepts.add(f.concept)
                        selected_facts.append(f)
                        break
        elif any(w in q_lower for w in ["bảng cân đối kế toán", "cân đối kế toán"]) and all_facts:
            target_metric_list = BALANCE_SHEET_METRICS
            for c in target_metric_list:
                for f in all_facts:
                    if f.concept == c and f.concept not in seen_concepts:
                        seen_concepts.add(f.concept)
                        selected_facts.append(f)
                        break
        elif any(w in q_lower for w in ["lưu chuyển tiền tệ", "lưu chuyển tiền", "dòng tiền"]) and all_facts:
            target_metric_list = CASH_FLOW_METRICS
            for c in target_metric_list:
                for f in all_facts:
                    if f.concept == c and f.concept not in seen_concepts:
                        seen_concepts.add(f.concept)
                        selected_facts.append(f)
                        break
        elif is_all_metrics and all_facts:
            for c in CORE_METRIC_ORDER:
                for f in all_facts:
                    if f.concept == c and f.concept not in seen_concepts:
                        seen_concepts.add(f.concept)
                        selected_facts.append(f)
                        break
        else:
            target_concepts = []
            for phrase, concepts in CONCEPT_KEYWORD_MAP.items():
                if phrase in q_lower:
                    target_concepts.extend(concepts)

            if target_concepts and all_facts:
                for c in target_concepts:
                    for f in all_facts:
                        if f.concept == c and f.concept not in seen_concepts:
                            seen_concepts.add(f.concept)
                            selected_facts.append(f)
                            break

            # Nếu là câu hỏi số liệu nhưng chưa tìm thấy theo keyword map, tìm theo từ khoá trong raw_label
            FINANCIAL_METRIC_KEYWORDS = [
                "bao nhiêu", "giá trị", "số liệu", "chỉ số", "chỉ tiêu", "mã số",
                "doanh thu", "lợi nhuận", "tổng tài sản", "tài sản", "vốn chủ",
                "nợ phải trả", "nợ ngắn hạn", "nợ dài hạn", "chi phí", "dòng tiền",
                "lưu chuyển tiền", "tiền mặt", "hàng tồn kho", "roe", "roa", "ros",
                "ebitda", "biên lợi nhuận", "tăng trưởng", "triệu đồng", "tỷ đồng",
                "nghìn đồng", "vnd", "usd", "%", "cuối kỳ", "đầu kỳ",
            ]
            is_metric_q = any(k in q_lower for k in FINANCIAL_METRIC_KEYWORDS)

            if is_metric_q and not selected_facts and all_facts:
                words = [w for w in re.split(r"\s+", q_lower) if len(w) >= 3 and w not in {"năm", "của", "cho", "bao", "nhiêu", "vnm", "vinamilk", "hpg", "fpt"}]
                if words:
                    for f in all_facts:
                        label = (f.raw_label or "").lower()
                        if any(w in label for w in words) and f.concept not in seen_concepts:
                            seen_concepts.add(f.concept)
                            selected_facts.append(f)
            # Kiểm tra xem câu hỏi có phải về khoản mục con/chi tiết của Thuyết minh không
            is_detail_subitem = any(
                k in q_lower for k in [
                    "nắm giữ", "cho thuê", "chờ tăng giá", "thù lao", "lương thưởng",
                    "ngân hàng", "chi tiết", "dự phòng", "xây dựng dở dang", "bên liên quan"
                ]
            )

            # Nếu hỏi khoản mục con hoặc không tìm thấy fact khớp trong SQL:
            # Tự động kích hoạt Fallback sang Vector Search Thuyết minh (Notes)
            if (not selected_facts or is_detail_subitem) and retriever is not None and not is_all_metrics:
                logger.info("sql_node: tự động Fallback sang Vector Search Thuyết minh cho query: '%s'", query[:60])
                try:
                    results, raw_citations = retriever.retrieve_with_citations(
                        query=query,
                        company=company,
                        year=year,
                        top_k=5,
                    )
                    if raw_citations:
                        vector_context = formatter.build_context_block(results, max_tokens_estimate=2500)
                        valid_citations = formatter.filter_valid_citations(raw_citations)
                        fallback_citations = [
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
                        explain_header = (
                            f"## 📄 Số liệu bóc tách từ Thuyết minh Báo cáo Tài chính ({company} {year})\n"
                            f"*(Lưu ý: Khoản mục này là số liệu phân rã chi tiết chuyên sâu thuộc Thuyết minh BCTC, "
                            f"không nằm trong các dòng tổng hợp của 3 Báo cáo cốt lõi. Hãy trả lời trực tiếp dựa trên Thuyết minh và gắn tag trích dẫn [cite_N] tương ứng)*\n\n"
                        )
                        return {
                            "sql_result": snapshot,
                            "sql_context": explain_header,
                            "vector_context": vector_context,
                            "citations": fallback_citations,
                        }
                except Exception as e_ret:
                    logger.warning("sql_node fallback retriever lỗi: %s", e_ret)

            # Fallback về các chỉ tiêu cốt lõi nếu chỉ khi là câu hỏi tổng quát
            if is_metric_q and not selected_facts and all_facts and is_all_metrics:
                for core in ["GROSS_PROFIT", "NET_REVENUE", "NET_PROFIT", "NET_INCOME", "TOTAL_ASSETS", "EQUITY", "TOTAL_EQUITY"]:
                    for f in all_facts:
                        if f.concept == core and f.concept not in seen_concepts:
                            seen_concepts.add(f.concept)
                            selected_facts.append(f)
                            break

        citations_payload = []
        sql_context_lines = [
            f"## 📊 Dữ liệu tài chính từ 3 Báo cáo Tài chính cốt lõi (SQL Database: {company} {year})\n"
        ]

        if selected_facts:
            sql_context_lines.append("### Danh sách chỉ tiêu tài chính từ 3 BCTC gốc (kèm tag trích dẫn nguồn):")
            for i, f in enumerate(selected_facts, start=1):
                cite_tag = f"cite_{i}"
                fact_page = f.page if (f.page and f.page >= 1) else 7
                stmt_name = _get_statement_name_by_page(fact_page)
                label = f.raw_label or f.concept
                unit = f.unit or "VND"

                sql_context_lines.append(
                    f"- **{label} ({f.concept})**: {f.value:,.0f} {unit} [{cite_tag}] (Nguồn: {stmt_name} — Trang {fact_page}, Mã số: {f.standard_code or 'N/A'})"
                )

                citations_payload.append({
                    "citation_id": cite_tag,
                    "block_id": f.table_id or f"p{fact_page}_b3",
                    "source_type": "statement",
                    "page": fact_page,
                    "bbox": None,  # 3 Bảng BCTC cốt lõi OCR lưu theo số trang, không có bbox
                    "snippet": f"{stmt_name} ({company} {year}) — {label}: {f.value:,.0f} {unit} (Trang {fact_page})",
                    "company": company,
                    "year": year,
                    "confidence": 1.0,
                })

        ratios = snapshot.get("ratios", {})
        if ratios:
            sql_context_lines.append("\n### 13 Chỉ số tài chính chuẩn (Tính toán từ hệ CSDL theo công thức):")
            for r_name, r_val in ratios.items():
                sql_context_lines.append(f"- **{r_name}**: {r_val}")
            sql_context_lines.append(
                "\n*(Lưu ý: 13 chỉ số tài chính chuẩn là giá trị tính toán tự động từ hệ CSDL, KHÔNG gắn tag citation của Thuyết minh)*"
            )

        verif = snapshot.get("verification", {})
        if verif:
            balanced = "✅ ĐÃ CÂN ĐỐI" if verif.get("is_balanced") == 1 else "❌ CHƯA CÂN ĐỐI"
            sql_context_lines.append(f"\n### Kiểm toán số học Anti-GIGO: {balanced}")
            sql_context_lines.append(
                f"- Tổng kiểm tra: {verif.get('total_checks')} | Đạt: {verif.get('passed_checks')} | Lỗi: {verif.get('failed_checks')}"
            )

        sql_context = "\n".join(sql_context_lines)

        logger.info(
            "sql_node: snapshot OK — %d facts, %d ratios, %d citations",
            len(selected_facts),
            len(ratios),
            len(citations_payload),
        )
        return {
            "sql_result": snapshot,
            "sql_context": sql_context,
            "citations": citations_payload,
        }
    except Exception as exc:
        logger.error("sql_node lỗi: %s", exc)
        return {
            "sql_result": {"error": str(exc)},
            "sql_context": f"⚠️ Không thể truy vấn SQL: {exc}",
            "citations": [],
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
            "raw_vector_results": results,
            "vector_context": vector_context,
            "citations": citations_payload,
        }
    except Exception as exc:
        logger.error("vector_node lỗi: %s", exc)
        return {
            "vector_results": [],
            "raw_vector_results": [],
            "vector_context": f"⚠️ Không thể truy vấn Vector DB: {exc}",
            "citations": [],
        }


def hybrid_node(state: CopilotState, *, sql_svc: Any, retriever: Any) -> dict:
    """Node: Gọi SQL + Vector song song cho câu hỏi DEEP_ANALYSIS.

    Thứ tự: SQL trước (số liệu định lượng nền) → Vector sau (giải trình ngữ cảnh).
    """
    from src.services.citation_formatter import CitationFormatter
    formatter = CitationFormatter()

    sql_patch = sql_node(state, sql_svc=sql_svc)
    vector_patch = vector_node(state, retriever=retriever)

    # Ghép citations an toàn giữa SQL và Vector
    sql_citations = sql_patch.get("citations", [])
    vec_citations = vector_patch.get("citations", [])
    raw_vec_results = vector_patch.get("raw_vector_results", [])
    offset = len(sql_citations)

    merged_citations = list(sql_citations)
    if offset == 0:
        merged_citations.extend(vec_citations)
        final_vector_context = vector_patch.get("vector_context", "")
    else:
        # Re-build vector_context bắt đầu từ start_index = offset + 1 để header cite khớp 100% với merged_citations!
        if raw_vec_results:
            final_vector_context = formatter.build_context_block(
                raw_vec_results,
                max_tokens_estimate=2500,
                start_index=offset + 1,
            )
        else:
            final_vector_context = vector_patch.get("vector_context", "")

        for i, c in enumerate(vec_citations, start=offset + 1):
            c_copy = dict(c)
            c_copy["citation_id"] = f"cite_{i}"
            merged_citations.append(c_copy)

    return {
        **sql_patch,
        **vector_patch,
        "vector_context": final_vector_context,
        "citations": merged_citations,
    }


# ===========================================================================
# Phase 3.2 — Synthesis Node (LLM Call)
# ===========================================================================
_SYSTEM_PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích tài chính doanh nghiệp chuyên sâu, \
am hiểu sâu Thông tư 200/2014/TT-BTC và chuẩn mực kế toán Việt Nam (VAS).

**Nguyên tắc bắt buộc:**
1. Chỉ sử dụng số liệu và dữ kiện được cung cấp trong phần ngữ cảnh bên dưới.
2. KHÔNG bịa đặt, KHÔNG ngoại suy số liệu hoặc thông tin không có trong ngữ cảnh.
3. **Quy tắc trích dẫn nguồn chuẩn xác (TUYỆT ĐỐI TRÁNH CITE NHẦM):**
   - **Số liệu thuộc 3 Báo cáo Tài chính cốt lõi** (Bảng Cân đối kế toán Trang 7-9, Báo cáo KQKD Trang 10, Báo cáo LCTT Trang 11-12):
     Được cung cấp trong phần "Dữ liệu tài chính từ 3 Báo cáo Tài chính cốt lõi (SQL Database)".
     Mỗi chỉ tiêu đã có sẵn tag trích dẫn `[cite_N]` kèm theo (ví dụ: `[cite_1]`, `[cite_2]`).
     BẮT BUỘC chỉ sử dụng đúng tag trích dẫn được cấp kèm theo chỉ tiêu đó trong câu trả lời.
   - **Thuyết minh BCTC (Notes, Trang 13 trở đi)**:
     Được cung cấp trong phần "Thuyết minh báo cáo tài chính" (Vector Context).
     CHỈ gắn tag trích dẫn của phần Thuyết minh khi giải thích, phân tích các khoản mục chi tiết thuộc Thuyết minh đó.
     TUYỆT ĐỐI KHÔNG dùng tag của Thuyết minh (như Trang 53 - Thù lao HĐQT, Trang 25, Trang 15...) để gán cho các chỉ số tài chính cốt lõi trong SQL!
   - **13 Chỉ số tài chính chuẩn (Financial Ratios: Current ratio, ROA, ROE, ROS, Tỷ số nợ...)**:
     Đây là các chỉ số tính toán tự động theo công thức chuẩn từ hệ CSDL SQL, KHÔNG gán tag cite của Thuyết minh.
4. Nếu không tìm thấy thông tin để trả lời, hãy nói rõ: "Không tìm thấy dữ liệu liên quan trong báo cáo tài chính." và hướng dẫn người dùng gõ lại hoặc làm rõ khoản mục. TUYỆT ĐỐI KHÔNG tự bịa số liệu và TUYỆT ĐỐI KHÔNG gắn bất kỳ tag [cite_N] nào khi không tìm thấy dữ liệu.
5. Định dạng số theo chuẩn Việt Nam: 14.352 tỷ đồng (hoặc 14.352.000.000 VND; không dùng triệu USD).
6. **Định dạng bảng Markdown:** Khi trình bày bảng biểu (Table), BẮT BUỘC mỗi hàng phải nằm trên MỘT DÒNG RIÊNG BIỆT (dùng ký tự xuống dòng `\\n`). TUYỆT ĐỐI KHÔNG viết các hàng bảng dính liền nhau bằng dấu `||` trên cùng một dòng đơn.
7. **Không tự tạo danh mục nguồn ở cuối:** TUYỆT ĐỐI KHÔNG tự tạo danh sách nguồn trích dẫn ở cuối câu trả lời (như "--- Nguồn trích dẫn:" hoặc liệt kê "Block: ..."). Hệ thống giao diện sẽ tự động hiển thị mục nguồn. Chỉ cần chèn tag [cite_N] trực tiếp trong câu văn.

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

    raw_history = state.get("messages", [])
    conversation_history: list[Any] = []
    if len(raw_history) > 1:
        # Lấy tối đa 4 tin nhắn trước tin nhắn hiện tại để duy trì ngữ cảnh đa lượt
        conversation_history = list(raw_history[:-1][-4:])

    messages = [
        SystemMessage(content=system_prompt + correction_context),
        *conversation_history,
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

    company = state.get("company") or "VNM"
    year = state.get("year") or 2025

    # Lấy bảng số liệu chuẩn từ SQL DB (toàn bộ facts của doanh nghiệp + ratios + snapshot)
    key_facts: dict[str, float] = {}
    if hasattr(sql_svc, "get_all_facts"):
        try:
            facts_list = sql_svc.get_all_facts(company=company, year=year)
            for f in facts_list:
                if f.value is not None:
                    key_facts[f.concept] = float(f.value)
        except Exception as e:
            logger.warning("fact_verify_node: không thể lấy all_facts: %s", e)

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

    # Tìm các số trong câu trả lời (List of Tuples)
    raw_matches = _FINANCIAL_NUM_RE.findall(draft)
    if not raw_matches:
        return {"fact_check_status": FactCheckStatus.SKIPPED, "fact_check_violations": []}

    violations: list[dict[str, Any]] = []

    for match in raw_matches:
        raw_num, unit = match
        raw_clean = re.sub(r'[\s\u202f\u00a0]', '', raw_num.strip())
        unit = unit.strip().lower() if unit else ""
        
        # 1. Bỏ qua các số dễ bị false positive (năm, số trang, mã số chỉ tiêu TT200)
        # VD: 2025, 2024, 7, 10, 100, 270, 300, 400... nếu không có chữ "tỷ", "triệu", "%" đi kèm
        clean_digits = re.sub(r'\D', '', raw_clean)
        if not unit and len(clean_digits) <= 4:
            continue

        # 2. Sinh ra nhiều biến thể (variants) dịch số học do sự nhập nhằng giữa dấu '.' và ','
        # VD: "4,567" -> 4567.0 (VN) hoặc 4.567 (English)
        variants = []
        try: variants.append(abs(float(re.sub(r'[,.]', '', raw_clean))))  # Bỏ hết dấu
        except: pass
        try: variants.append(abs(float(raw_clean.replace('.', '').replace(',', '.')))) # Hệ VN
        except: pass
        try: variants.append(abs(float(raw_clean.replace(',', '')))) # Hệ Anh
        except: pass
        
        variants = list(set(variants))
        if not variants:
            continue

        # 3. Quét kiểm tra xem có BẤT KỲ variant nào khớp với DB không
        matched_any_variant = False
        for num_variant in variants:
            for concept, db_val in key_facts.items():
                if db_val == 0:
                    continue
                abs_db = abs(db_val)
                # Thử khớp ở các scale phổ biến: x1, x1e3, x1e6, x1e9, và x1e-2 (đối với tỷ lệ %)
                for scale in [1, 1e3, 1e6, 1e9, 1e-2]:
                    scaled_db = abs_db / scale
                    if scaled_db == 0:
                        continue
                    diff_pct = abs(num_variant - scaled_db) / scaled_db
                    if diff_pct < 0.01:  # Khớp trong dung sai 1% → Quá chuẩn
                        matched_any_variant = True
                        break
                if matched_any_variant:
                    break
            if matched_any_variant:
                break
                
        # 4. Nếu không khớp với BẤT KỲ fact nào trong DB ở MỌI biến thể
        if not matched_any_variant:
            best_repr = max(variants)
            # Chỉ báo vi phạm nếu số đủ lớn (> 10) hoặc có đơn vị rõ ràng (tránh bắt nhầm ngày tháng)
            if best_repr > 10 or unit:
                violations.append({
                    "number_in_answer": best_repr,
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

    # Chuẩn hoá mọi biến thể tag [cite 1], [cite_1] về dạng chuẩn [[cite_N]]
    normalized_draft = formatter.normalize_citation_tags(draft)

    # Nếu câu trả lời là thông báo không tìm thấy dữ liệu:
    # Tuyệt đối KHÔNG inject citations và xóa toàn bộ citations rác, đồng thời cung cấp hướng dẫn tương tác
    if formatter.is_no_data_answer(normalized_draft):
        logger.info("citation_verify_node: Câu trả lời không có dữ liệu -> Xóa toàn bộ citations và tạo hướng dẫn tương tác.")
        interactive_msg = (
            "⚠️ **Không tìm thấy dữ liệu liên quan trong Báo cáo tài chính.**\n\n"
            "Có thể do câu hỏi chưa có dấu tiếng Việt, viết tắt hoặc tên khoản mục chưa trùng khớp với cách trình bày trong báo cáo.\n\n"
            "💡 **Gợi ý tra cứu:**\n"
            "- **Thử gõ lại có dấu tiếng Việt đầy đủ:** Ví dụ *\"Báo cáo kết quả hoạt động kinh doanh\"*, *\"Doanh thu thuần\"*.\n"
            "- **Các chỉ tiêu cốt lõi:** *Doanh thu thuần*, *Lợi nhuận gộp*, *Lợi nhuận sau thuế*, *Tổng tài sản*, *Vốn chủ sở hữu*, *Nợ phải trả*.\n"
            "- **Các chỉ số tài chính:** *ROE*, *ROA*, *Current ratio*, *Biên lợi nhuận gộp*.\n"
            "- **Thuyết minh chuyên sâu:** *Thù lao HĐQT*, *Bất động sản đầu tư*, *Chi phí xây dựng dở dang*, *Vay ngân hàng*.\n\n"
            "👇 *Bạn có thể gõ lại câu hỏi cụ thể hơn hoặc bấm vào các nút gợi ý bên dưới để tra cứu nhanh nhé!*"
        )
        return {
            "final_answer": interactive_msg,
            "messages": [AIMessage(content=interactive_msg)],
            "citations": [],
        }

    # Validate và filter
    valid_citations = formatter.filter_valid_citations(citation_objects)

    # Inject tags và tạo final answer
    final_answer = formatter.inject_citation_tags(normalized_draft, valid_citations, auto_inject=True)

    # Lọc nghiêm ngặt: chỉ giữ lại những citation THỰC SỰ được dùng trong final_answer
    cited_ids = formatter.extract_cited_ids(final_answer)
    if cited_ids:
        cited_tags = {f"cite_{i}" for i in cited_ids}
        used_citations = [c for c in valid_citations if c.citation_id in cited_tags]
        valid_citations = used_citations
    else:
        # Nếu final_answer hoàn toàn không dùng citation nào -> citations phải rỗng
        valid_citations = []

    # Cập nhật lại citations dict (chỉ giữ valid và actually cited)
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
        "citation_verify_node: %d valid citations (sau khi lọc theo câu trả lời), final_answer=%d ký tự.",
        len(final_citations),
        len(final_answer),
    )
    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "citations": final_citations,
    }


# ===========================================================================
# Clarification Node (Human-in-the-loop / Confirmation)
# ===========================================================================
def clarify_node(state: CopilotState) -> dict:
    """Node: Phản hồi yêu cầu làm rõ câu hỏi khi người dùng hỏi mơ hồ/chưa rõ ràng."""
    prompt = state.get("clarification_prompt") or (
        "Câu hỏi của bạn chưa nêu rõ khoản mục tài chính cụ thể cần tra cứu. "
        "Bạn vui lòng cho biết rõ hơn nội dung bạn muốn xem nhé "
        "(Ví dụ: Doanh thu thuần, Tổng tài sản, Nguyên giá bất động sản đầu tư nắm giữ cho thuê, Thù lao HĐQT...)?"
    )
    return {
        "final_answer": prompt,
        "draft_answer": prompt,
        "messages": [AIMessage(content=prompt)],
        "citations": [],
        "fact_check_status": FactCheckStatus.SKIPPED,
    }
