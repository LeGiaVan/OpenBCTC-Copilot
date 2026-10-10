"""src/services/citation_formatter.py — Visual Citation Assembler & Answer Formatter (Phase 2.3).

Nhiệm vụ:
  1. Nhận danh sách CitationWithBBox từ Retriever.
  2. Chèn tags [[cite_N]] vào câu trả lời của LLM (nếu LLM chưa tự chèn).
  3. Định dạng payload Citation chuẩn cho API response và PDF.js highlight.
  4. Kiểm tra hợp lệ BBox và số trang để tránh trả về toạ độ rác.

Thiết kế Fat Service thuần Python — không phụ thuộc LangGraph.
"""

from __future__ import annotations

import re
import logging
from typing import Any

from src.models.citation import CitationWithBBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Nhận diện tag [[cite_N]] hoặc [cite_N] hoặc 【cite_N】 trong câu trả lời LLM
_CITE_TAG_RE = re.compile(r"[\[【]{1,2}cite[ _:-]?(\d+)[\]】]{1,2}", re.IGNORECASE)

# Nhận diện số tài chính dạng tiền: 1.234.567 hoặc 1,234,567 hoặc 12,345.67
_FINANCIAL_NUMBER_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(tỷ|triệu|nghìn|đồng|VND|%)?\b",
    re.IGNORECASE,
)


class CitationFormatter:
    """Assembler trích dẫn nguồn kèm Bounding Box cho Visual Grounding.

    Usage:
        formatter = CitationFormatter()
        formatted = formatter.format_answer(llm_answer, citations)
        api_payload = formatter.to_api_payload(llm_answer, citations)
    """

    # Bbox hợp lệ: toạ độ normalised [0, 1] và có diện tích tối thiểu
    MIN_BBOX_AREA = 1e-4

    def validate_citation(self, citation: CitationWithBBox) -> bool:
        """Kiểm tra hợp lệ một citation: trang phải >= 1; BBox nếu có phải hợp lệ."""
        if citation.page < 1:
            logger.warning("Citation %s có số trang không hợp lệ: %d", citation.citation_id, citation.page)
            return False

        # Các chỉ số tài chính từ 3 bảng BCTC cốt lõi (SQL / OCR) không có bbox, chỉ có số trang
        if citation.bbox is None:
            return True

        bbox = citation.bbox
        if len(bbox) != 4:
            logger.warning("Citation %s có BBox không hợp lệ (cần 4 giá trị): %s", citation.citation_id, bbox)
            return False
        ymin, xmin, ymax, xmax = bbox
        if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
            logger.warning(
                "Citation %s có BBox ngoài dải [0,1]: %s",
                citation.citation_id, bbox,
            )
            return False
        area = (ymax - ymin) * (xmax - xmin)
        if area < self.MIN_BBOX_AREA:
            logger.warning("Citation %s có BBox quá nhỏ (area=%.6f): %s", citation.citation_id, area, bbox)
            return False
        return True

    def filter_valid_citations(
        self, citations: list[CitationWithBBox]
    ) -> list[CitationWithBBox]:
        """Lọc bỏ các citation có BBox/page không hợp lệ."""
        valid = [c for c in citations if self.validate_citation(c)]
        if len(valid) < len(citations):
            logger.info(
                "Đã lọc %d citation không hợp lệ (còn lại %d valid).",
                len(citations) - len(valid),
                len(valid),
            )
        return valid

    @staticmethod
    def is_no_data_answer(text: str) -> bool:
        """Kiểm tra xem câu trả lời có phải là thông báo không tìm thấy dữ liệu hay không."""
        if not text or not text.strip():
            return True
        clean = text.lower().strip()
        no_data_phrases = [
            "không tìm thấy dữ liệu",
            "không tìm thấy thông tin",
            "không có dữ liệu",
            "không có thông tin",
            "chưa tìm thấy dữ liệu",
            "chưa tìm thấy thông tin",
            "không ghi nhận dữ liệu",
            "không ghi nhận thông tin",
            "không thể tìm thấy",
            "chưa có thông tin",
            "chưa có dữ liệu",
        ]
        return any(phrase in clean for phrase in no_data_phrases)

    def inject_citation_tags(
        self,
        answer: str,
        citations: list[CitationWithBBox],
        auto_inject: bool = True,
    ) -> str:
        """Chèn tag [[cite_N]] vào cuối câu trả lời nếu LLM chưa tự thêm.

        Args:
            answer: Câu trả lời thô từ LLM.
            citations: Danh sách citation đã có.
            auto_inject: Nếu True và không có tag nào → tự thêm reference list vào cuối.

        Returns:
            Câu trả lời đã được gắn citation tags.
        """
        if not citations:
            return answer

        # Tuyệt đối không auto-inject citation rác khi câu trả lời thông báo không tìm thấy dữ liệu
        if self.is_no_data_answer(answer):
            return answer

        existing_tags = _CITE_TAG_RE.findall(answer)
        if existing_tags or not auto_inject:
            return answer

        # Tự thêm reference list vào cuối câu trả lời nếu LLM hoàn toàn chưa chèn tag
        ref_lines = ["\n\n---\n**Nguồn trích dẫn:**"]
        for c in citations[:3]:
            source_label = "Thuyết minh" if c.source_type == "note" else "Báo cáo tài chính"
            page_str = f"— Trang {c.page}" if c.page and c.page > 0 else ""
            ref_lines.append(f"- [[{c.citation_id}]] {source_label} {page_str}".rstrip())

        return answer + "\n".join(ref_lines)

    def extract_cited_ids(self, answer: str) -> list[int]:
        """Trích xuất danh sách số thứ tự citation đã được gắn vào câu trả lời."""
        return [int(m) for m in _CITE_TAG_RE.findall(answer)]

    def normalize_citation_tags(self, answer: str) -> str:
        """Chuẩn hoá mọi biến thể tag [cite 1], [cite_1], [[cite 1]] về dạng chuẩn [[cite_N]]."""
        return _CITE_TAG_RE.sub(r"[[cite_\1]]", answer)

    def to_api_payload(
        self,
        answer: str,
        citations: list[CitationWithBBox],
        *,
        auto_inject: bool = True,
        filter_invalid: bool = True,
    ) -> dict[str, Any]:
        """Tạo payload chuẩn cho FastAPI response endpoint `/api/v1/chat`.

        Schema:
            {
                "answer": str,              # Câu trả lời đã có citation tags
                "citations": [              # Danh sách citation để PDF.js highlight
                    {
                        "citation_id": "cite_1",
                        "block_id": "p12_mineru_txt_2",
                        "source_type": "note" | "statement",
                        "page": 12,
                        "bbox": [ymin, xmin, ymax, xmax],
                        "snippet": "...",
                        "company": "VNM",
                        "year": 2025,
                        "confidence": 0.96
                    },
                    ...
                ],
                "citation_count": int,
                "has_grounding": bool       # True nếu có ít nhất 1 citation hợp lệ
            }
        """
        if filter_invalid:
            citations = self.filter_valid_citations(citations)

        # Nếu là câu trả lời không có dữ liệu -> tuyệt đối không trả về citations và không grounding
        if self.is_no_data_answer(answer):
            return {
                "answer": answer,
                "citations": [],
                "citation_count": 0,
                "has_grounding": False,
            }

        answer_with_tags = self.inject_citation_tags(answer, citations, auto_inject=auto_inject)

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
            for c in citations
        ]

        return {
            "answer": answer_with_tags,
            "citations": citations_payload,
            "citation_count": len(citations_payload),
            "has_grounding": len(citations_payload) > 0,
        }

    def build_context_block(
        self,
        results: list[Any],  # list[RetrievalResult]
        max_tokens_estimate: int = 3000,
        chars_per_token: float = 3.0,
        start_index: int = 1,
    ) -> str:
        """Xây dựng Context Block cho System Prompt từ danh sách RetrievalResult.

        Gắn header [cite_N] trước mỗi đoạn ngữ cảnh để LLM có thể tham chiếu.
        Giới hạn tổng độ dài ≈ max_tokens_estimate * chars_per_token ký tự.

        Args:
            results: Danh sách RetrievalResult từ HybridRetriever.
            max_tokens_estimate: Số tokens tối đa cho phần context (mặc định 3000).
            chars_per_token: Ước tính ký tự/token cho tiếng Việt (mặc định 3.0).
            start_index: Số thứ tự bắt đầu cho citation tag (mặc định 1).

        Returns:
            Chuỗi văn bản context để chèn vào System Prompt.
        """
        max_chars = int(max_tokens_estimate * chars_per_token)
        lines: list[str] = ["## 📄 Ngữ Cảnh Truy Xuất Từ Báo Cáo Tài Chính\n"]
        total_chars = 0

        for i, result in enumerate(results, start=start_index):
            cite_tag = f"cite_{i}"
            source_type = "Thuyết minh" if getattr(result, "is_note", False) else "Báo cáo tài chính"
            header = f"### [{cite_tag}] {source_type} — Trang {result.page}"
            body = getattr(result, "raw_content", getattr(result, "content", ""))

            block_str = f"{header}\n{body}\n"
            if total_chars + len(block_str) > max_chars:
                logger.debug(
                    "Context block cắt ngắn tại cite_%d (đã đạt %d chars).", i, total_chars
                )
                break
            lines.append(block_str)
            total_chars += len(block_str)

        return "\n".join(lines)

    def format_sql_context(
        self,
        sql_result: dict[str, Any],
        query_type: str = "fact",
    ) -> str:
        """Định dạng kết quả SQL Fact Engine thành chuỗi ngữ cảnh cho LLM.

        Args:
            sql_result: Dict trả về từ SQLiteFactService (get_fact, compare_periods, v.v.).
            query_type: Loại truy vấn ('fact', 'ratio', 'compare', 'snapshot').

        Returns:
            Chuỗi văn bản mô tả số liệu để chèn vào System Prompt.
        """
        if query_type == "snapshot":
            return self._format_snapshot(sql_result)
        if query_type == "compare":
            return self._format_compare(sql_result)
        # Default: single fact or ratio
        return self._format_single(sql_result)

    # ------------------------------------------------------------------
    # Private formatting helpers
    # ------------------------------------------------------------------
    def _format_single(self, result: dict[str, Any]) -> str:
        """Định dạng 1 fact hoặc ratio đơn lẻ."""
        if "error" in result:
            return f"⚠️ Không tìm thấy dữ liệu: {result['error']}"
        concept = result.get("concept") or result.get("ratio_name", "N/A")
        val = result.get("val_current") or result.get("value")
        unit = result.get("unit", "VND")
        return f"**{concept}**: {val:,.0f} {unit}" if isinstance(val, (int, float)) else f"**{concept}**: {val}"

    def _format_compare(self, results: list[dict[str, Any]] | dict[str, Any]) -> str:
        """Định dạng bảng so sánh nhiều kỳ."""
        if isinstance(results, dict):
            results = [results]
        lines = ["| Chỉ tiêu | " + " | ".join([]) + " |"]
        for row in results:
            concept = row.get("concept", "")
            unit = row.get("unit", "VND")
            values = row.get("values", {})
            cells = []
            for yr, v in sorted(values.items()):
                val = v.get("value")
                pct = v.get("pct_change_yoy")
                if val is not None:
                    cell = f"{val:,.0f}"
                    if pct is not None:
                        arrow = "▲" if pct >= 0 else "▼"
                        cell += f" ({arrow}{abs(pct):.1f}%)"
                    cells.append(cell)
                else:
                    cells.append("N/A")
            lines.append(f"| {concept} ({unit}) | {' | '.join(cells)} |")
        return "\n".join(lines)

    def _format_snapshot(self, snapshot: dict[str, Any]) -> str:
        """Định dạng snapshot tổng quan tài chính."""
        company = snapshot.get("company", "")
        year = snapshot.get("year", "")
        facts = snapshot.get("key_facts", {})
        ratios = snapshot.get("ratios", {})
        verif = snapshot.get("verification", {})

        lines = [f"## 📊 Tổng quan tài chính {company} ({year})\n"]

        if facts:
            lines.append("### Chỉ tiêu cốt lõi")
            for concept, val in facts.items():
                unit_map = snapshot.get("unit", {})
                unit = unit_map.get(concept, "VND")
                if val is not None:
                    lines.append(f"- **{concept}**: {val:,.0f} {unit}")
                else:
                    lines.append(f"- **{concept}**: Không có dữ liệu")

        if ratios:
            lines.append("\n### 13 Chỉ số tài chính chuẩn")
            for name, val in ratios.items():
                lines.append(f"- **{name}**: {val}")

        if verif:
            balanced = "✅ ĐÃ CÂN ĐỐI" if verif.get("is_balanced") == 1 else "❌ CHƯA CÂN ĐỐI"
            lines.append(f"\n### Kiểm toán số học Anti-GIGO: {balanced}")
            lines.append(
                f"- Tổng kiểm tra: {verif.get('total_checks')} "
                f"| Đạt: {verif.get('passed_checks')} "
                f"| Lỗi: {verif.get('failed_checks')}"
            )

        return "\n".join(lines)
