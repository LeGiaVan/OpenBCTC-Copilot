import os
import io
import json
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from pymongo import MongoClient
import gridfs

class MongoGridFSService:
    """Service thao tác với MongoDB GridFS (Lưu PDF, JSON, MD).
    
    Cung cấp cả hàm Sync (cho Tool migrate) và Async (cho FastAPI endpoints).
    """
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "openbctc"):
        self.uri = uri
        self.db_name = db_name
        
        # Async client (for FastAPI)
        self.async_client = AsyncIOMotorClient(uri)
        self.async_db = self.async_client[db_name]
        self.async_fs = AsyncIOMotorGridFSBucket(self.async_db)
        self.async_json_collection = self.async_db["documents_json"]
        self.async_document_blocks = self.async_db["document_blocks"]
        self.async_ocr_benchmarks = self.async_db["ocr_benchmarks"]

        # Sync client (for scripts/migrations)
        self.sync_client = MongoClient(uri)
        self.sync_db = self.sync_client[db_name]
        self.sync_fs = gridfs.GridFS(self.sync_db)
        self.sync_json_collection = self.sync_db["documents_json"]
        self.sync_document_blocks = self.sync_db["document_blocks"]
        self.sync_ocr_benchmarks = self.sync_db["ocr_benchmarks"]

    # -----------------------------------------------------------------------
    # SYNC METHODS (For Migrate Script)
    # -----------------------------------------------------------------------
    def upload_file_sync(self, file_path: str, filename: str, metadata: dict = None) -> str:
        """Upload file (PDF, MD) vào GridFS bằng Sync mode."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
            
        with open(file_path, "rb") as f:
            file_id = self.sync_fs.put(f, filename=filename, metadata=metadata)
        return str(file_id)

    def upload_json_sync(self, json_data: dict, company: str, year: int) -> str:
        """Lưu JSON trực tiếp vào Collection (không cần GridFS vì JSON query được)."""
        doc = {
            "company": company.upper(),
            "year": year,
            "data": json_data
        }
        result = self.sync_json_collection.update_one(
            {"company": company.upper(), "year": year},
            {"$set": doc},
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else "updated"

    def save_document_blocks_sync(self, company: str, year: int, blocks: list[dict]) -> int:
        """Lưu danh sách blocks có toạ độ (bbox, page) vào collection document_blocks."""
        comp = company.upper()
        # Xóa các blocks cũ của company & year để tránh trùng lặp
        self.sync_document_blocks.delete_many({"company": comp, "year": year})
        docs = []
        for b in blocks:
            item = dict(b)
            item["company"] = comp
            item["year"] = year
            docs.append(item)
        if docs:
            self.sync_document_blocks.insert_many(docs)
        return len(docs)

    def get_document_blocks_sync(self, company: str, year: int) -> list[dict]:
        """Đọc danh sách blocks có toạ độ từ collection document_blocks."""
        cursor = self.sync_document_blocks.find({"company": company.upper(), "year": year}, {"_id": 0})
        return list(cursor)

    def save_ocr_benchmark_sync(self, company: str, year: int, metrics: dict) -> str:
        """Lưu báo cáo kiểm toán 17 đẳng thức Anti-GIGO vào collection ocr_benchmarks."""
        doc = {
            "company": company.upper(),
            "year": year,
            "metrics": metrics
        }
        result = self.sync_ocr_benchmarks.update_one(
            {"company": company.upper(), "year": year},
            {"$set": doc},
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else "updated"

    def clear_db_sync(self):
        """Xoá sạch dữ liệu cũ."""
        self.sync_client.drop_database(self.db_name)

    # -----------------------------------------------------------------------
    # ASYNC METHODS (For FastAPI)
    # -----------------------------------------------------------------------
    async def get_pdf_stream_async(self, company: str, year: int):
        """Lấy luồng dữ liệu PDF từ GridFS."""
        filename = f"{company.lower()}_{year}.pdf"
        cursor = self.async_fs.find({"filename": filename})
        docs = await cursor.to_list(length=1)
        if not docs:
            # Fallback chữ hoa nếu có
            cursor = self.async_fs.find({"filename": f"{company.upper()}_{year}.pdf"})
            docs = await cursor.to_list(length=1)
        if not docs:
            return None
        
        file_id = docs[0]["_id"] if isinstance(docs[0], dict) else getattr(docs[0], "_id")
        grid_out = await self.async_fs.open_download_stream(file_id)
        return grid_out

    async def get_json_async(self, company: str, year: int) -> Optional[Dict[str, Any]]:
        """Lấy dữ liệu JSON từ Collection."""
        doc = await self.async_json_collection.find_one({"company": company.upper(), "year": year})
        if doc:
            return doc.get("data")
        return None

    async def get_document_blocks_async(self, company: str, year: int) -> list[dict]:
        """Lấy danh sách blocks có toạ độ từ collection document_blocks."""
        cursor = self.async_document_blocks.find({"company": company.upper(), "year": year}, {"_id": 0})
        return await cursor.to_list(length=10000)

    async def get_ocr_benchmark_async(self, company: str, year: int) -> Optional[Dict[str, Any]]:
        """Lấy benchmark metrics từ collection ocr_benchmarks."""
        doc = await self.async_ocr_benchmarks.find_one({"company": company.upper(), "year": year})
        if doc:
            return doc.get("metrics")
        return None
