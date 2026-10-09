# 🚀 Hướng Dẫn Tích Hợp Dự Án OpenBCTC & OpenBCTC Copilot

Việc chuyển đổi kiến trúc sang **Docker & MongoDB GridFS** ở dự án `OpenBCTC Copilot` mang lại lợi ích khổng lồ về mặt tốc độ truy xuất và khả năng mở rộng. Để dự án tiền nhiệm `OpenBCTC` (chịu trách nhiệm OCR và bóc tách dữ liệu) giao tiếp mượt mà với Copilot, chúng ta cần cấu trúc lại luồng (Pipeline) một chút.

Dưới đây là hướng dẫn để kết nối 2 hệ thống này thành một thể thống nhất.

---

## 1. Kiến Trúc Mới (The New Pipeline)

- **Lưu trữ tĩnh (Cho người dùng cơ bản):** `OpenBCTC (OCR)` ➡️ Vẫn giữ nguyên cơ chế lưu file vào thư mục local `data/` để bất kỳ ai cũng có thể đọc trực tiếp PDF, MD, JSON.
- **Lưu trữ động (Cho người dùng Copilot):** `OpenBCTC (OCR)` ➡️ Bơm thêm một bản sao dữ liệu vào **MongoDB (GridFS)** qua cổng `27017` ➡️ `Copilot` (chạy trong Docker) tự động thấy và phục vụ ngay lập tức.

Ưu điểm: Vừa giữ được sự đơn giản "mì ăn liền" của dự án gốc, vừa đáp ứng được tốc độ stream PDF và khả năng tìm kiếm nâng cao của dự án Copilot.

---

## 2. Các Bước Cấu Trúc Lại OpenBCTC (OCR Engine)

Khi pipeline OCR của OpenBCTC chạy xong (đã tạo ra PDF, MD, và JSON), thay vì chỉ lưu ra thư mục `data/` ở ổ cứng, OpenBCTC cần "bơm" (Push) dữ liệu này vào MongoDB của Copilot.

### Bước 2.1: Bổ sung thư viện
Trên môi trường của dự án OpenBCTC, cài đặt thư viện kết nối MongoDB:
```bash
pip install pymongo motor
```

### Bước 2.2: Tích hợp Mongo Uploader vào cuối Pipeline OCR
Bạn hãy copy file `mongo_uploader.py` này vào dự án `OpenBCTC` và gọi nó ở cuối hàm chạy OCR.

**Tạo file `mongo_uploader.py` trong dự án OpenBCTC:**

```python
import os
import json
from pymongo import MongoClient
import gridfs

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "openbctc")

def push_to_copilot(company: str, year: int, pdf_path: str, md_path: str, json_path: str):
    """
    Hàm này được gọi sau khi OCR hoàn tất. 
    Nhiệm vụ: Đẩy file trực tiếp vào MongoDB của Copilot.
    """
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    fs = gridfs.GridFS(db)
    collection = db["documents_json"]

    company_lower = company.lower()

    # 1. Nuốt file PDF vào GridFS
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            fs.put(f, filename=f"{company_lower}_{year}.pdf", metadata={"company": company, "year": year, "type": "pdf"})
        print(f"[Upload] Đã đẩy {pdf_path} lên GridFS.")

    # 2. Nuốt file Markdown vào GridFS
    if os.path.exists(md_path):
        with open(md_path, "rb") as f:
            fs.put(f, filename=f"{company_lower}_{year}_elements.md", metadata={"company": company, "year": year, "type": "md"})
        print(f"[Upload] Đã đẩy {md_path} lên GridFS.")

    # 3. Nuốt JSON vào Document DB
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        collection.update_one(
            {"company": company, "year": year},
            {"$set": {"company": company, "year": year, "data": json_data}},
            upsert=True
        )
        print(f"[Upload] Đã đẩy {json_path} lên Collection.")

    print(f"🎉 Hoàn tất đưa dữ liệu {company} {year} vào Copilot!")
```

### Bước 2.3: Gọi hàm Push khi OCR xong
Trong file chạy chính của `OpenBCTC`, thêm vài dòng logic sau:
```python
from mongo_uploader import push_to_copilot

def process_pipeline():
    # ... code OCR ...
    # ... lưu file tạm ra ổ cứng ...
    
    # KHI XONG TẤT CẢ: Bơm thẳng vào Copilot
    push_to_copilot(
        company="VNM",
        year=2025,
        pdf_path="temp_data/VNM_2025/VNM_2025.pdf",
        md_path="temp_data/VNM_2025/VNM_2025_elements.md",
        json_path="temp_data/VNM_2025/VNM_2025_layout.json"
    )
```

---

## 3. Quy Trình Vận Hành Thống Nhất

1. Bật Docker của `OpenBCTC Copilot`: `docker compose up -d`. Lúc này MongoDB (`27017`) và Qdrant (`6333`) đã sẵn sàng lắng nghe.
2. Tại máy chạy `OpenBCTC`, cứ thả file PDF đầu vào và chạy lệnh OCR.
3. OCR chạy xong, nó tự động Push toàn bộ kết quả chui tọt vào bụng MongoDB thông qua cổng `27017`.
4. Người dùng mở `OpenBCTC Copilot` lên chat và phân tích dữ liệu ngay lập tức mà không cần copy/paste thư mục `data/` như thủ công lúc trước.

🎯 **Mọi thứ trở thành một vòng tuần hoàn khép kín và tự động 100%!**
