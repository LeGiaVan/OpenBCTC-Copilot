"""src/services/fact_manager.py — Dynamic SQLite Fact Resolver theo Company & Year."""

import logging
import os
from pathlib import Path
from typing import Any

from src.models.financial import FinancialFactDTO, FinancialRatioDTO
from src.services.sql_engine import SQLiteFactService

logger = logging.getLogger("Copilot.FactManager")


class FactServiceManager:
    """Quản lý kết nối SQLite Fact DB động theo Company và Year (Nhiệm Vụ 2)."""

    _instances: dict[tuple[str, int], SQLiteFactService] = {}

    @classmethod
    def get_service(cls, company: str, year: int, base_dir: str = "/app/data") -> SQLiteFactService:
        comp = company.upper()
        cache_key = (comp, year)
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # 1. Thử tìm trong shared volume / mount cục bộ
        candidates = [
            Path(base_dir) / f"{comp}_{year}" / f"benchmark_{comp}_{year}.db",
            Path(base_dir) / f"{comp}_{year}" / f"benchmark_{company.lower()}_{year}.db",
            Path("/app/outputs") / f"{comp}_{year}" / f"benchmark_{comp}_{year}.db",
            Path("/app/outputs") / f"{comp}_{year}" / f"benchmark_{company.lower()}_{year}.db",
            Path("../OpenBCTC/outputs") / f"{comp}_{year}" / f"benchmark_{comp}_{year}.db",
            Path("../OpenBCTC/outputs") / f"{comp}_{year}" / f"benchmark_{company.lower()}_{year}.db",
            Path(f"data/{comp}_{year}/benchmark_{comp}_{year}.db"),
            Path(f"data/{comp}_{year}/benchmark_{company.lower()}_{year}.db"),
        ]
        for p in candidates:
            if p.exists():
                logger.info("✓ Tìm thấy SQLite Fact DB tại cục bộ: %s", p)
                svc = SQLiteFactService(db_path=p)
                cls._instances[cache_key] = svc
                return svc

        # 2. Fallback: Tải từ MongoDB GridFS về thư mục đệm cache_db/
        cache_dir = Path("/app/cache_db") if Path("/app").exists() else Path("data/cache_db")
        cache_path = cache_dir / f"benchmark_{company.lower()}_{year}.db"
        if not cache_path.exists():
            try:
                from src.services.mongo_service import MongoGridFSService
                mongo = MongoGridFSService(uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                grid_out = mongo.sync_fs.find_one({"filename": f"benchmark_{company.lower()}_{year}.db"})
                if grid_out:
                    with open(cache_path, "wb") as f:
                        f.write(grid_out.read())
                    logger.info("✓ Đã tải SQLite Fact DB từ MongoDB GridFS về: %s", cache_path)
            except Exception as e:
                logger.warning("Không thể tải SQLite DB từ GridFS (%s): %s", company, e)

        if cache_path.exists():
            svc = SQLiteFactService(db_path=cache_path)
            cls._instances[cache_key] = svc
            return svc

        # 3. Fallback mặc định về file mẫu có sẵn nếu không tìm thấy
        for fallback_path in [
            Path("data/VNM_2025/benchmark_VNM_2025.db"),
            Path("/app/data/VNM_2025/benchmark_VNM_2025.db"),
        ]:
            if fallback_path.exists():
                logger.warning("⚠️ Không tìm thấy DB cho %s_%s, fallback về mẫu: %s", comp, year, fallback_path)
                svc = SQLiteFactService(db_path=fallback_path)
                cls._instances[cache_key] = svc
                return svc

        raise FileNotFoundError(f"Không tìm thấy SQLite Fact DB cho {comp}_{year}")


class DynamicFactService:
    """Proxy đa doanh nghiệp tương thích với SQLiteFactService cho LangGraph nodes."""

    def __init__(self, base_dir: str = "/app/data"):
        self.base_dir = base_dir

    def _resolve(self, company: str, year: int) -> SQLiteFactService:
        return FactServiceManager.get_service(company, year, base_dir=self.base_dir)

    def get_fact(
        self, company: str, year: int, concept: str, period: str | None = None
    ) -> FinancialFactDTO | None:
        return self._resolve(company, year).get_fact(company, year, concept, period=period)

    def get_fact_by_code(
        self, company: str, year: int, standard_code: str, period: str | None = None
    ) -> FinancialFactDTO | None:
        return self._resolve(company, year).get_fact_by_code(company, year, standard_code, period=period)

    def get_all_facts(
        self, company: str, year: int, period: str | None = None
    ) -> list[FinancialFactDTO]:
        return self._resolve(company, year).get_all_facts(company, year, period=period)

    def get_ratios(self, company: str, year: int) -> list[FinancialRatioDTO]:
        return self._resolve(company, year).get_ratios(company, year)

    def get_ratio_by_name(self, company: str, year: int, ratio_name: str) -> FinancialRatioDTO | None:
        return self._resolve(company, year).get_ratio_by_name(company, year, ratio_name)

    def compare_facts(
        self, company: str, concept: str, year_current: int, year_previous: int
    ) -> dict[str, Any]:
        return self._resolve(company, year_current).compare_facts(company, concept, year_current, year_previous)

    def compare_periods(
        self, company: str, concepts: list[str], years: list[int]
    ) -> list[dict[str, Any]]:
        year = years[-1] if years else 2025
        return self._resolve(company, year).compare_periods(company, concepts, years)

    def get_summary_snapshot(self, company: str, year: int) -> dict[str, Any]:
        return self._resolve(company, year).get_summary_snapshot(company, year)

    def get_verification_report(self, company: str, year: int) -> dict[str, Any] | None:
        return self._resolve(company, year).get_verification_report(company, year)
