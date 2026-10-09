"""src/api/routes/ingest.py — Webhook Ingestion API nhận dữ liệu từ OpenBCTC."""

import logging
import os
from pathlib import Path
from typing import Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.models.block import JSONBlock, BlockMetadata
from src.services.chunker import LayoutAwareChunker
from src.services.vector_engine import VectorEngineService
from ingestion.parsers.json_block_loader import JSONBlockLoader

logger = logging.getLogger("Copilot.Ingest")
router = APIRouter(tags=["Ingestion"])


class IngestRequest(BaseModel):
    company: str
    year: int
    source: str | None = "openbctc_syncer"


def execute_ingest(company: str, year: int) -> int:
    """Quy trình Ingestion: Đọc blocks -> Chunking -> Nạp Qdrant."""
    comp_upper = company.upper()
    logger.info("Bắt đầu Ingestion tự động cho [%s - %s]...", comp_upper, year)

    blocks: list[JSONBlock] = []

    # 1. Thử lấy từ MongoDB Collection 'document_blocks'
    try:
        from src.services.mongo_service import MongoGridFSService
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        mongo_db = os.getenv("MONGO_DB", "openbctc")
        mongo = MongoGridFSService(uri=mongo_uri, db_name=mongo_db)
        getter = getattr(mongo, "get_document_blocks_sync", getattr(mongo, "get_blocks_sync", None))
        raw_blocks = getter(comp_upper, year) if getter else []
        if raw_blocks:
            for b in raw_blocks:
                b.pop("_id", None)
                if "metadata" not in b or not isinstance(b["metadata"], dict):
                    b["metadata"] = {
                        "company": comp_upper,
                        "year": year,
                        "engine": "openbctc",
                        "is_note": True,
                    }
                try:
                    blocks.append(JSONBlock.model_validate(b))
                except Exception as ex_val:
                    logger.debug("Bỏ qua block lỗi: %s", ex_val)
            logger.info("Đã nạp %d blocks từ MongoDB 'document_blocks'", len(blocks))
    except Exception as ex_mongo:
        logger.warning("Không thể đọc từ MongoDB (%s). Chuyển sang quét tệp cục bộ...", ex_mongo)

    # 2. Fallback: Quét tệp cục bộ nếu MongoDB chưa có blocks
    if not blocks:
        candidates = [
            Path(f"data/{comp_upper}_{year}/cache/notes"),
            Path(f"/app/data/{comp_upper}_{year}/cache/notes"),
            Path(f"/app/data/openbctc_outputs/{comp_upper}_{year}/cache/notes"),
            Path(f"../OpenBCTC/outputs/{comp_upper}_{year}/cache/notes"),
        ]
        for cdir in candidates:
            if cdir.exists():
                try:
                    blocks = JSONBlockLoader.load_from_dir(cdir)
                    logger.info("Đã nạp %d blocks từ thư mục local: %s", len(blocks), cdir)
                    break
                except Exception as ex_dir:
                    logger.warning("Lỗi đọc local blocks từ %s: %s", cdir, ex_dir)

    if not blocks:
        logger.error("Không tìm thấy blocks nào cho [%s - %s] để nạp vào Qdrant!", comp_upper, year)
        return 0

    # 3. Layout-Aware Semantic Chunking
    chunker = LayoutAwareChunker(min_chars=250, max_chars=800)
    chunks = chunker.chunk_blocks(blocks)
    logger.info("Đã gom thành %d semantic chunks tối ưu", len(chunks))

    # 4. Nạp vào Qdrant (Ưu tiên QDRANT_URL container, fallback sang local storage)
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        vec_svc = VectorEngineService(url=qdrant_url)
    else:
        vec_svc = VectorEngineService(path="data/qdrant_storage")

    indexed_count = vec_svc.ingest_chunks(chunks)
    logger.info("ĐÃ NẠP THÀNH CÔNG %d chunks vào Qdrant cho [%s - %s]!", indexed_count, comp_upper, year)
    return indexed_count


@router.post("/ingest")
async def ingest_webhook(payload: IngestRequest, bg_tasks: BackgroundTasks):
    """Webhook nhận tín hiệu hoàn tất OCR từ OpenBCTC để tự động index Qdrant."""
    comp = payload.company.upper()
    yr = payload.year

    logger.info("Tiếp nhận Webhook Ingest cho [%s - %s] từ %s", comp, yr, payload.source)
    bg_tasks.add_task(execute_ingest, comp, yr)

    return {
        "status": "ACCEPTED",
        "message": f"Đã tiếp nhận yêu cầu đồng bộ cho {comp} {yr}. Quá trình vector hóa đang chạy nền.",
        "company": comp,
        "year": yr,
    }
