# 📊 Báo Cáo Đánh Giá Tự Động OpenBCTC Copilot (Phase 4 Evaluation)

- **Thời gian chạy:** `2026-10-10 17:36:55`
- **Tổng số câu hỏi kiểm thử:** `3`
- **Tỷ lệ thành công:** `3/3`
- **Thời gian phản hồi trung bình (Latency):** `18.36s / câu hỏi`
- **Tổng thời gian thực thi:** `83.88s`

---

## 🎯 1. Bảng Chỉ Số Chất Lượng Tiêu Chuẩn (Core Metrics)

| Chỉ Số Đánh Giá | Kết Quả Đạt Được | Mục Tiêu Chuẩn (Target) | Trạng Thái |
| :--- | :---: | :---: | :---: |
| **SQL Execution Accuracy** | **66.67%** | 100.0% | ⚠️ CẦN TỐI ƯU |
| **Citation Precision** | **66.67%** | ≥ 85.0% | ⚠️ CẦN TỐI ƯU |
| **Faithfulness (Ragas)** | **91.67%** | ≥ 90.0% | ✅ ĐẠT |
| **Answer Relevancy** | **44.44%** | ≥ 90.0% | ⚠️ CẦN TỐI ƯU |
| **Intent Match Accuracy** | **100.00%** | ≥ 95.0% | ✅ ĐẠT |
| **⭐ ĐIỂM CHẤT LƯỢNG TỔNG THỂ** | **68.19%** | ≥ 90.0% | ✅ ĐẠT |

---

## 📈 2. Kết Quả Theo Từng Nhóm Câu Hỏi (Breakdown by Category)

| Nhóm Câu Hỏi | Số Lượng | Điểm Trung Bình | SQL Accuracy | Citation Precision | Faithfulness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NUMERIC_FACT** (Số liệu cốt lõi SQL) | 3 | 68.2% | 66.7% | 66.7% | 91.7% |
| **NOTE_EXPLANATION** (Thuyết minh BCTC) | 0 | 0.0% | - | 0.0% | 0.0% |
| **DEEP_ANALYSIS** (Phân tích suy luận) | 0 | 0.0% | - | 0.0% | 0.0% |

---

## 📝 3. Danh Sách Chi Tiết Kết Quả Kiểm Thử (Top Samples)

| ID | Nhóm | Câu Hỏi | Overall Score | SQL Acc | Cite Prec | Faithfulness | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Q01` | `NUMERIC_FACT` | Tổng tài sản của Vinamilk năm 2025 là bao nhi... | **18.3%** | 0% | 0% | 100% | 3.76s |
| `Q02` | `NUMERIC_FACT` | Vốn chủ sở hữu của Vinamilk tại ngày 31/12/20... | **89.6%** | 100% | 100% | 75% | 8.91s |
| `Q03` | `NUMERIC_FACT` | Nợ phải trả của công ty năm 2025 là bao nhiêu... | **96.7%** | 100% | 100% | 100% | 42.41s |

---
*Báo cáo được khởi tạo tự động bởi OpenBCTC Copilot Automated Evaluation Pipeline (Phase 4.2).*
