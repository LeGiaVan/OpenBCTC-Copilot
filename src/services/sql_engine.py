"""src/services/sql_engine.py — Service truy vấn số liệu tài chính và chỉ số chuẩn từ SQLite."""

import sqlite3
from pathlib import Path
from typing import Any

from src.models.financial import FinancialFactDTO, FinancialRatioDTO


class SQLiteFactService:
    """Service thực hiện các truy vấn xác định (Deterministic SQL queries) trên SQLite facts database."""

    def __init__(self, db_path: Path | str = "data/VNM_2025/benchmark_VNM_2025.db") -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file SQLite database: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_fact(
        self,
        company: str,
        year: int,
        concept: str,
        period: str | None = None,
    ) -> FinancialFactDTO | None:
        """Tra cứu một chỉ tiêu tài chính duy nhất.
        
        Args:
            company: Mã cổ phiếu (ví dụ: 'VNM')
            year: Năm tài chính (ví dụ: 2025)
            concept: Tên chuẩn TT200 (ví dụ: 'TOTAL_ASSETS', 'NET_REVENUE')
            period: Kỳ báo cáo (ví dụ: '2025' hoặc '2024'; nếu None sẽ ưu tiên kỳ của năm đó)
        """
        target_period = period if period is not None else str(year)

        query = """
            SELECT * FROM financial_facts
            WHERE company = ? AND year = ? AND concept = ? AND period = ?
            LIMIT 1
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (company.upper(), year, concept.upper(), target_period))
            row = cursor.fetchone()

            if not row and period is None:
                # Fallback thử tìm period = 'current'
                cursor.execute(
                    """
                    SELECT * FROM financial_facts
                    WHERE company = ? AND year = ? AND concept = ? AND period = 'current'
                    LIMIT 1
                    """,
                    (company.upper(), year, concept.upper()),
                )
                row = cursor.fetchone()

            if not row:
                return None

            return FinancialFactDTO.model_validate(dict(row))

    def get_fact_by_code(
        self,
        company: str,
        year: int,
        standard_code: str,
        period: str | None = None,
    ) -> FinancialFactDTO | None:
        """Tra cứu chỉ tiêu tài chính theo mã số chuẩn TT200 (ví dụ: '270', '10', '60')."""
        target_period = period if period is not None else str(year)
        query = """
            SELECT * FROM financial_facts
            WHERE company = ? AND year = ? AND standard_code = ? AND period = ?
            LIMIT 1
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (company.upper(), year, str(standard_code), target_period))
            row = cursor.fetchone()
            if not row:
                return None
            return FinancialFactDTO.model_validate(dict(row))

    def get_all_facts(
        self,
        company: str,
        year: int,
        period: str | None = None,
    ) -> list[FinancialFactDTO]:
        """Lấy tất cả các facts của một doanh nghiệp trong năm/kỳ."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if period is not None:
                cursor.execute(
                    "SELECT * FROM financial_facts WHERE company = ? AND year = ? AND period = ?",
                    (company.upper(), year, period),
                )
            else:
                cursor.execute(
                    "SELECT * FROM financial_facts WHERE company = ? AND year = ?",
                    (company.upper(), year),
                )
            rows = cursor.fetchall()
            return [FinancialFactDTO.model_validate(dict(r)) for r in rows]

    def get_ratios(self, company: str, year: int) -> list[FinancialRatioDTO]:
        """Lấy 13 chỉ số tài chính chuẩn hoá của doanh nghiệp."""
        query = """
            SELECT * FROM financial_ratios
            WHERE company = ? AND year = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (company.upper(), year))
            rows = cursor.fetchall()
            return [FinancialRatioDTO.model_validate(dict(r)) for r in rows]

    def get_ratio_by_name(
        self,
        company: str,
        year: int,
        ratio_name: str,
    ) -> FinancialRatioDTO | None:
        """Tra cứu 1 chỉ số tài chính cụ thể (ví dụ: 'current_ratio', 'roe')."""
        query = """
            SELECT * FROM financial_ratios
            WHERE company = ? AND year = ? AND LOWER(ratio_name) = ?
            LIMIT 1
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (company.upper(), year, ratio_name.lower()))
            row = cursor.fetchone()
            if not row:
                return None
            return FinancialRatioDTO.model_validate(dict(row))

    def compare_facts(
        self,
        company: str,
        concept: str,
        year_current: int,
        year_previous: int,
    ) -> dict[str, Any]:
        """So sánh biến động của một chỉ tiêu qua 2 kỳ (Tuyệt đối & Tỷ lệ %)."""
        current_fact = self.get_fact(company, year_current, concept, period=str(year_current))
        prev_fact = self.get_fact(company, year_current, concept, period=str(year_previous))

        # Nếu không có trong cùng record năm hiện tại, tìm qua năm cũ
        if not prev_fact:
            prev_fact = self.get_fact(company, year_previous, concept, period=str(year_previous))

        if not current_fact or not prev_fact:
            return {
                "concept": concept,
                "error": f"Không tìm đủ dữ liệu 2 kỳ cho concept {concept}",
                "current_value": current_fact.value if current_fact else None,
                "previous_value": prev_fact.value if prev_fact else None,
            }

        val_curr = current_fact.value
        val_prev = prev_fact.value
        abs_diff = val_curr - val_prev
        pct_diff = (abs_diff / abs(val_prev) * 100.0) if val_prev != 0 else None

        return {
            "concept": concept,
            "unit": current_fact.unit,
            "year_current": year_current,
            "val_current": val_curr,
            "year_previous": year_previous,
            "val_previous": val_prev,
            "diff_absolute": abs_diff,
            "diff_percentage": round(pct_diff, 2) if pct_diff is not None else None,
        }

    def compare_periods(
        self,
        company: str,
        concepts: list[str],
        years: list[int],
    ) -> list[dict[str, Any]]:
        """So sánh chuỗi thời gian nhiều concepts × nhiều kỳ (Time-series matrix).

        Args:
            company: Mã cổ phiếu.
            concepts: Danh sách concept TT200 muốn so sánh (ví dụ: ['NET_REVENUE', 'GROSS_PROFIT']).
            years: Danh sách các năm muốn so sánh (ví dụ: [2024, 2025]).

        Returns:
            Danh sách dict với mỗi phần tử là 1 dòng comparison theo concept.
        """
        if not concepts or not years:
            return []

        results: list[dict[str, Any]] = []
        for concept in concepts:
            row: dict[str, Any] = {"concept": concept, "company": company.upper(), "values": {}}
            prev_value: float | None = None
            unit: str = "VND"

            for yr in sorted(years):
                fact = self.get_fact(company, yr, concept, period=str(yr))
                # Nếu không tìm thấy trong fact của năm đó, thử period khác trong cùng năm
                if not fact:
                    fact = self.get_fact(company, yr, concept, period="current")
                if fact:
                    val = fact.value
                    unit = fact.unit
                    pct_change = None
                    if prev_value is not None and prev_value != 0:
                        pct_change = round((val - prev_value) / abs(prev_value) * 100.0, 2)
                    row["values"][str(yr)] = {
                        "value": val,
                        "pct_change_yoy": pct_change,
                    }
                    prev_value = val
                else:
                    row["values"][str(yr)] = {"value": None, "pct_change_yoy": None}

            row["unit"] = unit
            results.append(row)

        return results

    def get_summary_snapshot(
        self,
        company: str,
        year: int,
    ) -> dict[str, Any]:
        """Lấy bức tranh tổng quan tài chính: key facts + 13 ratios + verification status.

        Dùng cho node Summary trong LangGraph khi người dùng hỏi tổng quan.
        """
        key_concepts = [
            "TOTAL_ASSETS",
            "TOTAL_EQUITY",
            "TOTAL_LIABILITIES",
            "NET_REVENUE",
            "GROSS_PROFIT",
            "OPERATING_PROFIT",
            "NET_INCOME",
            "CASH_FROM_OPERATIONS",
        ]

        facts_snapshot: dict[str, float | None] = {}
        unit_map: dict[str, str] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for concept in key_concepts:
                cursor.execute(
                    "SELECT value, unit FROM financial_facts WHERE company=? AND year=? AND concept=? LIMIT 1",
                    (company.upper(), year, concept.upper()),
                )
                row = cursor.fetchone()
                facts_snapshot[concept] = row["value"] if row else None
                if row:
                    unit_map[concept] = row["unit"]

        ratios = self.get_ratios(company, year)
        ratios_snapshot = {r.ratio_name: round(r.value, 4) for r in ratios}

        verif = self.get_verification_report(company, year)

        return {
            "company": company.upper(),
            "year": year,
            "key_facts": facts_snapshot,
            "unit": unit_map,
            "ratios": ratios_snapshot,
            "verification": verif or {},
        }

    def get_verification_report(self, company: str, year: int) -> dict[str, Any] | None:
        """Lấy báo cáo kiểm toán số học 17 đẳng thức kế toán Anti-GIGO."""
        query = """
            SELECT is_balanced, total_checks, passed_checks, failed_checks
            FROM financial_statements
            WHERE company = ? AND year = ?
            LIMIT 1
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (company.upper(), year))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)
