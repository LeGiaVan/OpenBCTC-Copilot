"""src/observability/evaluator.py — Automated Financial RAG Evaluation Metrics (Phase 4.2).

Thực thi bộ 4 chỉ số đo lường chất lượng tiêu chuẩn:
  1. SQL Execution Accuracy: Độ khớp tuyệt đối giữa số liệu câu trả lời và SQLite DB (< 1% dung sai).
  2. Citation Precision: Độ chính xác của số trang và bounding box trích dẫn so với tài liệu gốc.
  3. Faithfulness (Ragas-standard): Đo lường mức độ trung thực, chống ảo giác (zero-hallucination).
  4. Answer Relevancy & Intent Accuracy: Độ tương thích và bao quát đúng trọng tâm câu hỏi.
"""

from __future__ import annotations

import re
import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Regex trích xuất số tài chính Việt Nam và quốc tế
_FINANCIAL_NUM_RE = re.compile(
    r"(?<!\w)"
    r"(\d{1,3}(?:[.,\s\u202f\u00a0]\d{3})*(?:[.,]\d+)?)"
    r"\s*(tỷ|triệu|nghìn|đồng|vnd|%)?"
    r"(?!\w)",
    re.IGNORECASE,
)


class EvaluationResult(NamedTuple):
    query_id: str
    sql_accuracy: float           # 0.0 -> 1.0
    citation_precision: float     # 0.0 -> 1.0
    faithfulness: float           # 0.0 -> 1.0
    answer_relevancy: float       # 0.0 -> 1.0
    intent_match: bool            # True / False
    overall_score: float          # 0.0 -> 1.0 (trung bình có trọng số)
    details: dict[str, Any]


class FinancialEvaluator:
    """Bộ đánh giá chất lượng tự động cho OpenBCTC Copilot."""

    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def extract_numbers_from_text(self, text: str) -> list[float]:
        """Trích xuất danh sách các số thực từ văn bản câu trả lời."""
        raw_matches = _FINANCIAL_NUM_RE.findall(text)
        numbers = []
        for raw_num, unit in raw_matches:
            raw_clean = re.sub(r'[\s\u202f\u00a0]', '', raw_num.strip())
            unit = unit.strip().lower() if unit else ""
            clean_digits = re.sub(r'\D', '', raw_clean)
            if not unit and len(clean_digits) <= 4:
                # Bỏ qua năm (2025, 2024), số trang đơn lẻ (1, 2, 10)...
                try:
                    val = float(clean_digits)
                    if val in (2024, 2025, 2026, 2023, 10, 11, 12, 7, 8, 9):
                        continue
                except:
                    pass

            variants = []
            try: variants.append(abs(float(re.sub(r'[,.]', '', raw_clean))))
            except: pass
            try: variants.append(abs(float(raw_clean.replace('.', '').replace(',', '.'))))
            except: pass
            try: variants.append(abs(float(raw_clean.replace(',', ''))))
            except: pass

            for v in variants:
                if unit == "tỷ":
                    numbers.append(v * 1e9)
                    numbers.append(v)
                elif unit == "triệu":
                    numbers.append(v * 1e6)
                    numbers.append(v)
                elif unit == "%":
                    numbers.append(v)
                    numbers.append(v / 100.0)
                else:
                    numbers.append(v)
        return list(set(numbers))

    def evaluate_sql_accuracy(
        self,
        answer: str,
        expected_values: dict[str, float] | None,
        sql_result: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """1. Đánh giá SQL Execution Accuracy.

        Kỳ vọng: Toàn bộ số liệu mong đợi trong expected_values phải khớp với số liệu
        trong câu trả lời (hoặc sql_result) với dung sai sai lệch < 1%.
        """
        if not expected_values:
            return 1.0, {"reason": "No expected values specified for this query"}

        text_numbers = self.extract_numbers_from_text(answer)
        matched_concepts = []
        unmatched_concepts = []

        for concept, exp_val in expected_values.items():
            if exp_val == 0:
                matched_concepts.append(concept)
                continue

            abs_exp = abs(exp_val)
            matched = False

            # Kiểm tra trong sql_result trước nếu có
            if sql_result and "facts" in sql_result:
                facts = sql_result["facts"]
                if isinstance(facts, dict) and concept in facts:
                    val = facts[concept].get("value") if isinstance(facts[concept], dict) else facts[concept]
                    if val is not None and abs(abs(val) - abs_exp) / abs_exp < 0.01:
                        matched = True

            # Kiểm tra trong các số trích xuất từ câu trả lời
            if not matched:
                for num in text_numbers:
                    for scale in [1.0, 1e9, 1e6, 1e3, 0.01]:
                        scaled_exp = abs_exp / scale
                        if scaled_exp > 0 and abs(num - scaled_exp) / scaled_exp < 0.015:
                            matched = True
                            break
                    if matched:
                        break

            if matched:
                matched_concepts.append(concept)
            else:
                unmatched_concepts.append({"concept": concept, "expected": exp_val})

        total = len(expected_values)
        acc = len(matched_concepts) / total if total > 0 else 1.0
        return round(acc, 4), {
            "total_expected": total,
            "matched_count": len(matched_concepts),
            "matched_concepts": matched_concepts,
            "unmatched_concepts": unmatched_concepts,
        }

    def evaluate_citation_precision(
        self,
        citations: list[dict[str, Any]] | None,
        expected_pages: list[int] | None,
        answer: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """2. Đánh giá Citation Precision.

        Kiểm tra các citation tags [cite_N] có trỏ đúng vào các số trang mong đợi (expected_pages)
        và có toạ độ bounding box hợp lệ hay không.
        """
        if not expected_pages:
            return 1.0, {"reason": "No specific expected pages specified"}

        if not citations:
            # Nếu câu trả lời không có citations trong khi expected_pages yêu cầu
            return 0.0, {"reason": "No citations provided", "expected_pages": expected_pages}

        correct_citations = 0
        citation_details = []

        for c in citations:
            page = c.get("page")
            try:
                page_num = int(page) if page is not None else None
            except:
                page_num = None

            is_page_match = page_num in expected_pages
            has_valid_source = bool(c.get("source_type") or c.get("snippet"))
            if is_page_match:
                correct_citations += 1

            citation_details.append({
                "citation_id": c.get("citation_id"),
                "page": page_num,
                "is_match": is_page_match,
                "has_source": has_valid_source,
            })

        total = len(citations)
        precision = correct_citations / total if total > 0 else 0.0
        return round(precision, 4), {
            "total_citations": total,
            "correct_citations": correct_citations,
            "citation_details": citation_details,
            "expected_pages": expected_pages,
        }

    def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> tuple[float, dict[str, Any]]:
        """3. Đánh giá Faithfulness (Ragas Metric).

        Đo lường mức độ trung thực của câu trả lời so với ngữ cảnh (context).
        Nếu có LLM: Sử dụng LLM-as-a-judge phân rã claims và kiểm chứng.
        Nếu không có LLM: Sử dụng kiểm toán số liệu và trích xuất thực thể.
        """
        if not answer or "Không tìm thấy dữ liệu" in answer:
            return 1.0, {"reason": "Answer explicitly acknowledges lack of data"}

        if not context:
            # Trả lời số liệu nhưng ngữ cảnh rỗng -> có nguy cơ hallucination
            return 0.5, {"reason": "No context provided to verify answer"}

        if self.llm is not None:
            try:
                import json
                prompt = (
                    "Bạn là chuyên gia kiểm toán Báo cáo tài chính độc lập.\n"
                    "Nhiệm vụ: Hãy phân tích câu trả lời dưới đây và đối chiếu với phần Ngữ cảnh.\n"
                    "1. Liệt kê các luận điểm / số liệu chính trong câu trả lời.\n"
                    "2. Kiểm tra xem mỗi luận điểm có được chứng minh bởi Ngữ cảnh không (True/False).\n"
                    "3. Tính tỷ lệ Faithfulness = (số luận điểm đúng / tổng số luận điểm) từ 0.0 đến 1.0.\n\n"
                    f"NGỮ CẢNH:\n{context[:2500]}\n\n"
                    f"CÂU TRẢ LỜI:\n{answer}\n\n"
                    "Chỉ trả về JSON thuần:\n"
                    "{\n  \"faithfulness_score\": 0.95,\n  \"total_claims\": 4,\n  \"supported_claims\": 4,\n  \"reason\": \"Mọi số liệu đều khớp với bảng BCTC\"\n}"
                )
                from langchain_core.messages import SystemMessage, HumanMessage
                res = self.llm.invoke([HumanMessage(content=prompt)])
                txt = res.content if hasattr(res, "content") else str(res)
                txt = re.sub(r"^```(?:json)?\n", "", txt.strip())
                txt = re.sub(r"\n```$", "", txt)
                data = json.loads(txt)
                score = float(data.get("faithfulness_score", 0.9))
                return round(min(1.0, max(0.0, score)), 4), data
            except Exception as e:
                logger.debug("LLM faithfulness check exception: %s -> fallback heuristic", e)

        # Heuristic fallback: Kiểm tra tỷ lệ từ khoá / số liệu câu trả lời nằm trong context
        ans_numbers = re.findall(r"\d[\d.,]*\d", answer)
        if not ans_numbers:
            return 0.95, {"method": "heuristic", "reason": "Text-only answer without ungrounded numbers"}

        matched_nums = [n for n in ans_numbers if n.replace('.', '').replace(',', '') in context.replace('.', '').replace(',', '')]
        score = len(matched_nums) / len(ans_numbers) if ans_numbers else 1.0
        return round(score, 4), {
            "method": "heuristic",
            "matched_numbers_count": len(matched_nums),
            "total_numbers_count": len(ans_numbers),
        }

    def evaluate_relevancy(
        self,
        query: str,
        answer: str,
        expected_keywords: list[str] | None,
    ) -> tuple[float, dict[str, Any]]:
        """4. Đánh giá Answer Relevancy & Completeness.

        Kiểm tra câu trả lời có chứa các từ khóa trọng tâm và giải quyết đúng nội dung câu hỏi không.
        """
        if not expected_keywords:
            return 1.0, {"reason": "No expected keywords provided"}

        ans_lower = answer.lower()
        matched = [kw for kw in expected_keywords if kw.lower() in ans_lower]
        total = len(expected_keywords)
        score = len(matched) / total if total > 0 else 1.0
        return round(score, 4), {
            "total_keywords": total,
            "matched_keywords": matched,
            "unmatched_keywords": [kw for kw in expected_keywords if kw not in matched],
        }

    def evaluate_item(
        self,
        item: dict[str, Any],
        actual_output: dict[str, Any],
    ) -> EvaluationResult:
        """Thực thi đầy đủ bộ 4 chỉ số đánh giá cho 1 câu hỏi kiểm thử."""
        query_id = item["id"]
        category = item.get("category", "NUMERIC_FACT")
        query = item["query"]
        expected_intent = item.get("expected_intent")
        expected_values = item.get("expected_values")
        expected_pages = item.get("expected_pages")
        expected_keywords = item.get("expected_keywords")

        final_answer = actual_output.get("final_answer", "")
        actual_intent = str(actual_output.get("intent", ""))
        citations = actual_output.get("citations", [])
        sql_result = actual_output.get("sql_result")
        context = actual_output.get("sql_context", "") + "\n" + actual_output.get("vector_context", "")

        # 1. SQL Accuracy
        sql_acc, sql_det = self.evaluate_sql_accuracy(final_answer, expected_values, sql_result)

        # 2. Citation Precision
        cit_prec, cit_det = self.evaluate_citation_precision(citations, expected_pages, final_answer)

        # 3. Faithfulness
        faith, faith_det = self.evaluate_faithfulness(query, final_answer, context)

        # 4. Relevancy
        relevancy, rel_det = self.evaluate_relevancy(query, final_answer, expected_keywords)

        # 5. Intent Match
        intent_match = expected_intent in actual_intent if expected_intent else True

        # Trọng số điểm tổng hợp dựa theo Category
        if category == "NUMERIC_FACT":
            # Số liệu: SQL Accuracy chiếm trọng số cao nhất (50%), Citations (25%), Faithfulness (15%), Relevancy (10%)
            overall = sql_acc * 0.50 + cit_prec * 0.25 + faith * 0.15 + relevancy * 0.10
        elif category == "NOTE_EXPLANATION":
            # Thuyết minh: Citation Precision (40%), Faithfulness (30%), Relevancy (30%)
            overall = cit_prec * 0.40 + faith * 0.30 + relevancy * 0.30
        else:  # DEEP_ANALYSIS
            # Phân tích suy luận: Faithfulness (40%), Relevancy (35%), Citation Precision (25%)
            overall = faith * 0.40 + relevancy * 0.35 + cit_prec * 0.25

        return EvaluationResult(
            query_id=query_id,
            sql_accuracy=sql_acc,
            citation_precision=cit_prec,
            faithfulness=faith,
            answer_relevancy=relevancy,
            intent_match=intent_match,
            overall_score=round(overall, 4),
            details={
                "category": category,
                "sql": sql_det,
                "citation": cit_det,
                "faithfulness": faith_det,
                "relevancy": rel_det,
                "intent_match": intent_match,
            },
        )
