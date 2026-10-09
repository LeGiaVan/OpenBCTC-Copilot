import os
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PDF"])

# Dùng chung biến môi trường hoặc cấu hình (Mặc định Mongo)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "openbctc")

def get_mongo_service():
    try:
        from src.services.mongo_service import MongoGridFSService
        # Kiểm tra kết nối nhẹ
        return MongoGridFSService(uri=MONGO_URI, db_name=MONGO_DB)
    except Exception as e:
        logger.warning(f"Không thể khởi tạo MongoGridFSService: {e}")
        return None

def _parse_bbox(bbox: str | None) -> list[float]:
    if bbox is None:
        raise HTTPException(status_code=400, detail="bbox is required")

    try:
        parts = [float(part.strip()) for part in bbox.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="bbox must be a comma-separated list of 4 floats") from exc

    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must contain exactly 4 coordinates [ymin, xmin, ymax, xmax]")

    if any(not 0.0 <= value <= 1.0 for value in parts):
        raise HTTPException(status_code=400, detail="bbox coordinates must be normalized to the range [0, 1]")

    return parts


@router.get("/pdf/{company}/{year}")
async def get_pdf(company: str, year: int):
    """Lấy file PDF của báo cáo tài chính qua MongoDB GridFS (Fallback: Local Folder)."""
    
    company_lower = company.lower()
    company_upper = company.upper()
    file_name = f"{company_upper}_{year}.pdf"

    # 1. Thử đọc từ MongoDB GridFS
    mongo = get_mongo_service()
    if mongo:
        try:
            stream = await mongo.get_pdf_stream_async(company_lower, year)
            if stream:
                async def generate_chunks():
                    while True:
                        chunk = await stream.read(1024 * 1024) # 1MB chunk
                        if not chunk:
                            break
                        yield chunk
                
                logger.info("Serving PDF from MongoDB GridFS cho %s_%s", company, year)
                return StreamingResponse(
                    generate_chunks(), 
                    media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{file_name}"', "Cache-Control": "no-cache, no-store, must-revalidate"}
                )
        except Exception as e:
            logger.warning(f"Không thể kết nối hoặc đọc từ MongoDB ({e}). Chuyển sang đọc Local...")

    # 2. Fallback: Đọc từ thư mục Local (data/)
    local_path = f"data/{company_upper}_{year}/{file_name}"
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="PDF not found in Mongo or Local Storage")
        
    logger.info("Serving PDF từ Local data/ cho %s_%s", company, year)
    return FileResponse(
        local_path, 
        media_type="application/pdf",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@router.get("/pdf/bbox")
async def get_pdf_bbox(
    company: str = Query(default="VNM", min_length=1),
    year: int = Query(default=2025, ge=1900, le=2100),
    page: int = Query(default=1, ge=1),
    bbox: str | None = Query(default=None, description="Comma-separated normalized bbox: ymin,xmin,ymax,xmax"),
):
    parsed_bbox = _parse_bbox(bbox)
    pdf_path = f"data/{company.upper()}_{year}/{company.upper()}_{year}.pdf"

    return {
        "company": company.upper(),
        "year": year,
        "page": page,
        "bbox": parsed_bbox,
        "pdf_path": pdf_path,
        "status": "ready",
    }


@router.get("/documents")
async def list_documents():
    """Lấy danh sách Báo cáo từ MongoDB (nếu có) hoặc Local data/"""
    docs = []
    
    # 1. Chế độ Cloud (MongoDB GridFS)
    mongo = get_mongo_service()
    if mongo:
        try:
            # Tìm danh sách file PDF
            cursor = mongo.async_fs.find({"filename": {"$regex": r"\.pdf$"}})
            async for grid_out in cursor:
                filename = grid_out["filename"] if isinstance(grid_out, dict) else getattr(grid_out, "filename", "")
                # VD: vnm_2025.pdf
                base = filename.split(".")[0]
                if "_" in base:
                    c, y = base.split("_")
                    if y.isdigit():
                        docs.append({
                            "id": f"{c.upper()}_{y}",
                            "company": c.upper(),
                            "year": int(y),
                            "name": f"Báo cáo {c.upper()} năm {y} (MongoDB)"
                        })
            if docs:
                return {"documents": docs}
        except Exception as e:
            logger.warning(f"Lỗi đọc list MongoDB: {e}")
            pass

    # 2. Chế độ Local (Thư mục data/)
    data_dir = "data"
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and "_" in item:
                parts = item.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    docs.append({
                        "id": item,
                        "company": parts[0].upper(),
                        "year": int(parts[1]),
                        "name": f"Báo cáo tài chính {parts[0].upper()} năm {parts[1]}"
                    })
    return {"documents": docs}
