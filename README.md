# OpenBCTC Copilot 🤖💼

OpenBCTC Copilot là một trợ lý ảo phân tích Báo cáo tài chính doanh nghiệp được xây dựng trên kiến trúc **Dual-Engine RAG (SQL + Vector)** kết hợp với mô hình luồng tác vụ đa tác tử **LangGraph**.

Dự án này đặc biệt giải quyết hai bài toán khó nhất trong việc phân tích tài liệu tài chính phức tạp:
1. **Truy xuất dữ liệu định lượng chính xác tuyệt đối:** Sử dụng SQL Engine để đọc dữ liệu dạng Bảng biểu (Tables) mà không bị "ảo giác" (hallucinations).
2. **Visual Grounding - Định vị nguồn tài liệu:** Giao diện Web hiển thị file PDF gốc và cho phép highlight (BBox) chính xác vị trí chứa số liệu được Bot trích dẫn.

---

## 🏗 Kiến trúc Hệ thống

Hệ thống được thiết kế hoàn toàn tách biệt giữa Backend (API) và Frontend (UI) để dễ dàng kiểm thử và mở rộng.

* **Backend (FastAPI):**
  * **API Server:** Chịu trách nhiệm nhận yêu cầu, xử lý streaming (SSE), và phục vụ các file PDF.
  * **CORS:** Đã được mở hoàn toàn (`allow_origins=["*"]`) để frontend ở domain bất kỳ có thể gọi API.
  * **AI Orchestrator (LangGraph):** Điều phối luồng suy nghĩ của Agent, tự động đánh giá (Router) nên dùng **SQL Engine** (cho các câu hỏi cần sự chính xác dạng số liệu) hay **Vector Engine** (cho các câu hỏi về chính sách, chữ, rủi ro).
  * **Vector Database:** `Qdrant` (Local Mode).
  * **SQL Database:** `SQLite` lưu trữ bảng dữ liệu BCTC.

* **Frontend (Vanilla HTML/JS):**
  * Tách rời độc lập hoàn toàn ở thư mục `frontend/`.
  * Tích hợp **PDF.js** để render file PDF thực tế và tự động tính toán toạ độ (BBox) để vẽ khung highlight.
  * Xử lý giao thức Server-Sent Events (SSE) để tạo hiệu ứng streaming token như ChatGPT, kèm parsing Markdown và Regex thay thế thẻ trích dẫn `[[cite_...]]` thành nút bấm.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cấu hình Môi trường
Tạo file `.env` tại thư mục gốc của dự án và điền các API Key (có thể sử dụng Groq, OpenRouter hoặc OpenAI):
```env
# Lựa chọn LLM Provider
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# URL (Mặc định tự động nhận diện nếu để trống)
LLM_BASE_URL=
```

### 2. Khởi chạy Backend (API Server)
Mở Terminal, kích hoạt môi trường ảo (Virtual Environment) và chạy lệnh sau:
```bash
uvicorn src.api.main:app --reload --port 8000
```
*Backend sẽ chạy tại `http://127.0.0.1:8000` và sẵn sàng cung cấp dữ liệu qua API.*

### 3. Khởi chạy Frontend Độc lập
Để quan sát chi tiết cách hệ thống hoạt động (ví dụ: dùng tab Network F12 xem luồng SSE), bạn nên chạy thư mục `frontend` tách biệt khỏi Backend.

1. Mở VSCode.
2. Cài đặt Extension **Live Server**.
3. Chuột phải vào file `frontend/index.html` và chọn **"Open with Live Server"**.
4. Giao diện Frontend sẽ chạy trên một cổng riêng (ví dụ: `http://127.0.0.1:5500`) và tự động giao tiếp chéo (CORS) với Backend ở cổng `8000`.

---

## 🔍 Chức năng chính
* **Truy vấn Streaming (Chat):** Hiển thị câu trả lời ngay lập tức từng chữ một.
* **Smart Routing:** LLM tự động phán đoán Ý định người dùng (NUMERIC_FACT, DEEP_ANALYSIS) để gọi đúng Engine.
* **Auto BBox Highlight:** Khi người dùng click vào thẻ trích dẫn `[cite_1]`, trình duyệt sẽ lật tới đúng trang PDF tương ứng và nháy sáng một khung Đỏ bao quanh đoạn văn / bảng biểu chứa dữ liệu đó. (Visual Grounding).
* **Quản lý Cache Cache-Buster:** Tự động loại bỏ cache của trình duyệt để đảm bảo PDF và nội dung luôn tải bản mới nhất.

## 🛠 Troubleshooting
Nếu Web UI gặp lỗi hiển thị ô vuông khi xem file BCTC PDF tiếng Việt, vui lòng sử dụng phím **Ctrl + F5** (Hard Reload) để trình duyệt xoá sạch cache và tải lại thư viện CMap/Font tiếng Việt của PDF.js.
