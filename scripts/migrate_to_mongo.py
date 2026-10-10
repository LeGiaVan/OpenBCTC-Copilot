import os
import glob
import json
import sys

from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.services.mongo_service import MongoGridFSService

def migrate():
    print("🚀 Bắt đầu Migrate toàn bộ dữ liệu từ ổ cứng (data/) vào bụng MongoDB...")
    
    # Kết nối MongoDB (sẽ dùng Sync client)
    # Lấy từ env MONGO_URI (nếu chạy trong Docker thì là mongodb://mongodb:27017)
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db = os.getenv("MONGO_DB", "openbctc")
    print(f"🔌 Kết nối tới MongoDB: {mongo_uri} (Database: {mongo_db})")
    mongo = MongoGridFSService(uri=mongo_uri, db_name=mongo_db)
    
    print("🧹 Dọn dẹp dữ liệu cũ trong MongoDB...")
    mongo.clear_db_sync()

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục {data_dir}. Hủy Migrate.")
        return

    # Quét tất cả thư mục con (VD: VNM_2025)
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"\n📂 Đang xử lý bộ hồ sơ: {folder_name}")
        
        # Parse company, year từ tên folder (VNM_2025)
        try:
            company, year_str = folder_name.split("_")
            year = int(year_str)
        except ValueError:
            company = folder_name
            year = 2025 # Default

        # 1. Upload PDF vào GridFS
        pdf_path = os.path.join(folder_path, f"{folder_name}.pdf")
        if os.path.exists(pdf_path):
            file_id = mongo.upload_file_sync(
                file_path=pdf_path, 
                filename=f"{company.lower()}_{year}.pdf", 
                metadata={"company": company, "year": year, "type": "pdf"}
            )
            print(f"  ✅ Đã nuốt PDF vào GridFS (ID: {file_id})")

        # 2. Upload Markdown vào GridFS
        md_candidates = [
            os.path.join(folder_path, f"{folder_name}_financial_report_final.md"),
            os.path.join(folder_path, f"{folder_name}_financial_report.md"),
            os.path.join(folder_path, f"{folder_name}_elements.md"),
        ]
        for md_path in md_candidates:
            if os.path.exists(md_path):
                file_id = mongo.upload_file_sync(
                    file_path=md_path, 
                    filename=f"{company.lower()}_{year}_final.md", 
                    metadata={"company": company, "year": year, "type": "md"}
                )
                print(f"  ✅ Đã nuốt Markdown ({os.path.basename(md_path)}) vào GridFS (ID: {file_id})")
                break

        # 3. Upload JSON vào Document Collection
        json_path = os.path.join(folder_path, f"{folder_name}_layout.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            doc_id = mongo.upload_json_sync(json_data, company=company, year=year)
            print(f"  ✅ Đã nuốt JSON vào Collection (ID: {doc_id})")

        # 3. Upload SQLite DB vào GridFS để FactServiceManager có thể fallback tải về
        db_candidates = [
            os.path.join(folder_path, f"benchmark_{folder_name}.db"),
            os.path.join(folder_path, f"benchmark_{company.lower()}_{year}.db"),
        ]
        for db_file in db_candidates:
            if os.path.exists(db_file):
                file_id = mongo.upload_file_sync(
                    file_path=db_file,
                    filename=f"benchmark_{company.lower()}_{year}.db",
                    metadata={"company": company, "year": year, "type": "sqlite"}
                )
                print(f"  ✅ Đã nuốt SQLite DB ({os.path.basename(db_file)}) vào GridFS (ID: {file_id})")
                break

        # 4. Upload Benchmark Metrics vào Collection 'ocr_benchmarks'
        metrics_path = os.path.join(folder_path, f"{folder_name}_ocr_benchmark_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            mongo.save_ocr_benchmark_sync(company=company, year=year, metrics=metrics_data)
            print(f"  ✅ Đã nuốt Benchmark Metrics vào Collection 'ocr_benchmarks'")

        # 5. Upload các blocks có toạ độ vào chuẩn collection 'document_blocks'
        notes_dir = os.path.join(folder_path, "cache", "notes")
        if os.path.exists(notes_dir):
            all_blocks = []
            for note_file in glob.glob(os.path.join(notes_dir, "*.json")):
                with open(note_file, "r", encoding="utf-8") as f:
                    page_blocks = json.load(f)
                if isinstance(page_blocks, list):
                    all_blocks.extend(page_blocks)
                elif isinstance(page_blocks, dict):
                    all_blocks.append(page_blocks)
            if all_blocks:
                count = mongo.save_document_blocks_sync(company=company, year=year, blocks=all_blocks)
                print(f"  ✅ Đã nuốt {count} blocks toạ độ vào chuẩn Collection 'document_blocks'")

    print("\n🎉 Hoàn tất Migrate! Toàn bộ file đã nằm trong MongoDB.")

if __name__ == "__main__":
    migrate()
