"""ingestion/run_ingest.py — CLI Runner nạp dữ liệu hoàn chỉnh từ OpenBCTC vào Copilot."""

import argparse
import sys
from pathlib import Path

# Cấu hình UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ingestion.parsers.json_block_loader import JSONBlockLoader
from ingestion.parsers.md_hierarchy_parser import MarkdownHierarchyParser
from src.services.chunker import LayoutAwareChunker
from src.services.sql_engine import SQLiteFactService
from src.services.vector_engine import VectorEngineService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion Pipeline: Nạp dữ liệu BCTC vào OpenBCTC Copilot")
    parser.add_argument(
        "--notes-dir",
        type=str,
        default="data/VNM_2025/cache/notes",
        help="Đường dẫn thư mục chứa các file JSON blocks thuyết minh",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/VNM_2025/benchmark_VNM_2025.db",
        help="Đường dẫn file SQLite database chứa facts & ratios",
    )
    parser.add_argument(
        "--md-path",
        type=str,
        default="data/VNM_2025/VNM_2025_financial_report_final.md",
        help="Đường dẫn file Markdown báo cáo tài chính toàn văn",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="data/qdrant_storage",
        help="Đường dẫn lưu trữ Qdrant local hoặc ':memory:'",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="URL kết nối tới Qdrant server (ví dụ: http://localhost:6333)",
    )
    parser.add_argument("--company", type=str, default="VNM", help="Mã cổ phiếu")
    parser.add_argument("--year", type=int, default=2025, help="Năm tài chính")

    args = parser.parse_args()

    print(f"\n🚀 BẮT ĐẦU INGESTION PIPELINE CHO DOANH NGHIỆP: {args.company} ({args.year})")
    print("=" * 70)

    # 1. Kiểm tra & đọc SQLite Fact Database
    print(f"📊 [1/5] Kết nối SQLite Database: {args.db_path}...")
    sql_svc = SQLiteFactService(db_path=args.db_path)
    all_facts = sql_svc.get_all_facts(company=args.company, year=args.year)
    ratios = sql_svc.get_ratios(company=args.company, year=args.year)
    verif = sql_svc.get_verification_report(company=args.company, year=args.year)

    print(f"   -> Tìm thấy {len(all_facts)} chỉ tiêu tài chính (Facts)")
    print(f"   -> Tìm thấy {len(ratios)} chỉ số tài chính tính toán sẵn (Ratios)")
    if verif:
        balanced_str = "✅ ĐÃ CÂN ĐỐI" if verif.get("is_balanced") == 1 else "❌ CHƯA CÂN ĐỐI"
        print(f"   -> Kiểm toán số học Anti-GIGO: {balanced_str} ({verif.get('total_checks')} đẳng thức)")

    # 2. Đọc & phân tích cấu trúc Markdown
    print(f"\n📑 [2/5] Phân tích cấu trúc Markdown: {args.md_path}...")
    toc_items = MarkdownHierarchyParser.parse_file(args.md_path)
    print(f"   -> Trích xuất thành công {len(toc_items)} mục phân cấp Heading (TOC Tree)")

    # 3. Nạp danh sách JSON Blocks có Bounding Box
    print(f"\n📦 [3/5] Quét các file JSON Blocks từ: {args.notes_dir}...")
    blocks = JSONBlockLoader.load_from_dir(args.notes_dir)
    print(f"   -> Nạp và validate thành công {len(blocks)} raw blocks")

    # 4. Layout-Aware Semantic Chunking (Gom khối & Enclosing BBox)
    print("\n🧩 [4/5] Thực hiện Layout-Aware Semantic Chunking & Enclosing BBox...")
    chunker = LayoutAwareChunker(min_chars=250, max_chars=800)
    chunks = chunker.chunk_blocks(blocks)
    avg_len = sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
    print(f"   -> Đã tinh chế từ {len(blocks)} raw blocks thành {len(chunks)} Semantic Chunks tối ưu")
    print(f"   -> Độ dài trung bình: {avg_len:.0f} ký tự/chunk (Đã tính Enclosing Bounding Box)")

    # 5. Nạp vào Qdrant Hybrid Collection (Dense + Sparse BM25)
    target_dest = args.qdrant_url if args.qdrant_url else args.qdrant_path
    print(f"\n🧠 [5/5] Khởi tạo & Embed vào Qdrant Hybrid Store ({target_dest})...")
    if args.qdrant_url:
        vec_svc = VectorEngineService(url=args.qdrant_url)
    elif args.qdrant_path == ":memory:":
        vec_svc = VectorEngineService(url=":memory:")
    else:
        vec_svc = VectorEngineService(path=args.qdrant_path)

    ingested_count = vec_svc.ingest_chunks(chunks)
    print(f"   -> Đã index thành công {ingested_count} hybrid points (Dense + Sparse BM25)")

    print("\n" + "=" * 70)
    print("✨ HOÀN TẤT INGESTION GIAI ĐOẠN 1 THÀNH CÔNG RỰC RỠ!")
    print(f"Dữ liệu của {args.company} ({args.year}) đã sẵn sàng cho Dual-Engine Copilot (SQL Facts + Qdrant Hybrid Vector).")


if __name__ == "__main__":
    main()
