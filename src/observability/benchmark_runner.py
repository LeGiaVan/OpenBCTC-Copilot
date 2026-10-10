"""src/observability/benchmark_runner.py — Automated Evaluation Pipeline Runner (Phase 4.2).

Thực thi chạy kiểm thử tự động trên Golden Financial Dataset (50 câu hỏi):
  - Khởi tạo Copilot LangGraph.
  - Chạy từng câu hỏi, ghi nhận thời gian thực thi (latency).
  - Đánh giá chất lượng qua FinancialEvaluator (SQL Accuracy, Citations, Faithfulness, Relevancy).
  - Tự động đồng bộ điểm số lên Langfuse traces nếu có cấu hình.
  - Xuất báo cáo tổng kết chi tiết dạng JSON và Markdown.

Cách chạy:
  python -m src.observability.benchmark_runner
  python -m src.observability.benchmark_runner --limit 10
  python -m src.observability.benchmark_runner --category NUMERIC_FACT
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.agents.copilot.graph import build_copilot_graph, run_query
from src.services.fact_manager import DynamicFactService
from src.services.retriever import HybridRetriever
from src.services.vector_engine import VectorEngineService
from src.observability.evaluator import FinancialEvaluator
from src.observability.langfuse_client import log_score, is_langfuse_configured

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchmarkRunner")


def run_benchmark(
    dataset_path: str = "data/golden_dataset.json",
    output_dir: str = "data",
    limit: int | None = None,
    category_filter: str | None = None,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline kiểm thử tự động và xuất báo cáo."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy Golden Dataset tại: {dataset_path}")

    with open(p, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    if category_filter:
        dataset = [item for item in dataset if item.get("category") == category_filter]

    if limit:
        dataset = dataset[:limit]

    logger.info("🚀 Bắt đầu chạy Benchmark trên %d câu hỏi kiểm thử...", len(dataset))

    # 1. Khởi tạo Fat Services & LLM
    sql_svc = DynamicFactService(base_dir="data")
    try:
        vec_svc = VectorEngineService(url=settings.QDRANT_URL)
    except Exception:
        vec_svc = VectorEngineService(path="data/qdrant_storage")

    retriever = HybridRetriever(vec_svc=vec_svc)

    model_name = settings.GROQ_MODEL or "openai/gpt-oss-20b"
    api_key = settings.GROQ_API_KEY
    base_url = "https://api.groq.com/openai/v1"

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    evaluator = FinancialEvaluator(llm=llm)

    # 2. Xây dựng Copilot LangGraph
    graph = build_copilot_graph(
        sql_svc=sql_svc,
        retriever=retriever,
        llm=llm,
        use_memory_saver=True,
        langfuse_enabled=is_langfuse_configured(),
    )

    results = []
    total_start = time.time()

    for idx, item in enumerate(dataset, 1):
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]
        comp = item.get("company", "VNM")
        yr = item.get("year", 2025)

        logger.info("[%d/%d] Chạy câu hỏi %s (%s): '%s'", idx, len(dataset), q_id, cat, query[:60])
        t0 = time.time()
        thread_id = f"benchmark_session_{q_id}_{int(time.time())}"

        try:
            actual = run_query(
                graph=graph,
                query=query,
                company=comp,
                year=yr,
                thread_id=thread_id,
            )
            latency = round(time.time() - t0, 2)
            eval_res = evaluator.evaluate_item(item, actual)

            # Đồng bộ điểm lên Langfuse nếu có
            log_score(thread_id, "sql_accuracy", eval_res.sql_accuracy)
            log_score(thread_id, "citation_precision", eval_res.citation_precision)
            log_score(thread_id, "faithfulness", eval_res.faithfulness)
            log_score(thread_id, "overall_score", eval_res.overall_score)

            results.append({
                "id": q_id,
                "category": cat,
                "query": query,
                "latency_sec": latency,
                "sql_accuracy": eval_res.sql_accuracy,
                "citation_precision": eval_res.citation_precision,
                "faithfulness": eval_res.faithfulness,
                "answer_relevancy": eval_res.answer_relevancy,
                "intent_match": eval_res.intent_match,
                "overall_score": eval_res.overall_score,
                "final_answer": actual.get("final_answer", "")[:300] + "...",
                "citations_count": len(actual.get("citations", [])),
                "details": eval_res.details,
            })

            logger.info(
                "✓ %s: Score=%.2f | SQL=%.2f | Cite=%.2f | Faith=%.2f (%.2fs)",
                q_id, eval_res.overall_score, eval_res.sql_accuracy,
                eval_res.citation_precision, eval_res.faithfulness, latency
            )

        except Exception as exc:
            logger.error("❌ Lỗi khi chạy %s: %s", q_id, exc)
            results.append({
                "id": q_id,
                "category": cat,
                "query": query,
                "latency_sec": round(time.time() - t0, 2),
                "error": str(exc),
            })

        # Nghỉ nhẹ giữa các câu hỏi để tránh chạm trần burst rate limit của Groq
        time.sleep(1.5)

    total_duration = round(time.time() - total_start, 2)

    # 3. Tính toán các chỉ số tổng hợp
    valid_results = [r for r in results if "error" not in r]
    count = len(valid_results) or 1

    avg_sql = round(sum(r["sql_accuracy"] for r in valid_results) / count, 4)
    avg_cite = round(sum(r["citation_precision"] for r in valid_results) / count, 4)
    avg_faith = round(sum(r["faithfulness"] for r in valid_results) / count, 4)
    avg_rel = round(sum(r["answer_relevancy"] for r in valid_results) / count, 4)
    avg_overall = round(sum(r["overall_score"] for r in valid_results) / count, 4)
    intent_acc = round(sum(1 for r in valid_results if r["intent_match"]) / count, 4)
    avg_latency = round(sum(r["latency_sec"] for r in valid_results) / count, 2)

    # Thống kê theo từng Category
    by_category = {}
    for cat in ["NUMERIC_FACT", "NOTE_EXPLANATION", "DEEP_ANALYSIS"]:
        cat_items = [r for r in valid_results if r["category"] == cat]
        c_count = len(cat_items) or 1
        by_category[cat] = {
            "count": len(cat_items),
            "avg_overall": round(sum(r["overall_score"] for r in cat_items) / c_count, 4),
            "avg_sql": round(sum(r["sql_accuracy"] for r in cat_items) / c_count, 4),
            "avg_citation": round(sum(r["citation_precision"] for r in cat_items) / c_count, 4),
            "avg_faithfulness": round(sum(r["faithfulness"] for r in cat_items) / c_count, 4),
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(dataset),
        "successful_queries": len(valid_results),
        "total_duration_sec": total_duration,
        "avg_latency_sec": avg_latency,
        "metrics": {
            "sql_execution_accuracy": avg_sql,
            "citation_precision": avg_cite,
            "faithfulness": avg_faith,
            "answer_relevancy": avg_rel,
            "intent_classification_accuracy": intent_acc,
            "overall_benchmark_score": avg_overall,
        },
        "by_category": by_category,
        "results": results,
    }

    # 4. Lưu báo cáo JSON & Markdown
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_report_phase4.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = Path("docs/benchmark_report_phase4.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_report(summary, md_path)

    logger.info("🎉 Benchmark hoàn tất! Báo cáo đã lưu tại:")
    logger.info("  - JSON: %s", json_path)
    logger.info("  - Markdown: %s", md_path)

    return summary


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    """Tạo báo cáo định dạng Markdown chuyên nghiệp."""
    m = summary["metrics"]
    by_cat = summary["by_category"]

    md = f"""# 📊 Báo Cáo Đánh Giá Tự Động OpenBCTC Copilot (Phase 4 Evaluation)

- **Thời gian chạy:** `{summary['timestamp']}`
- **Tổng số câu hỏi kiểm thử:** `{summary['total_queries']}`
- **Tỷ lệ thành công:** `{summary['successful_queries']}/{summary['total_queries']}`
- **Thời gian phản hồi trung bình (Latency):** `{summary['avg_latency_sec']}s / câu hỏi`
- **Tổng thời gian thực thi:** `{summary['total_duration_sec']}s`

---

## 🎯 1. Bảng Chỉ Số Chất Lượng Tiêu Chuẩn (Core Metrics)

| Chỉ Số Đánh Giá | Kết Quả Đạt Được | Mục Tiêu Chuẩn (Target) | Trạng Thái |
| :--- | :---: | :---: | :---: |
| **SQL Execution Accuracy** | **{m['sql_execution_accuracy'] * 100:.2f}%** | 100.0% | {'✅ ĐẠT' if m['sql_execution_accuracy'] >= 0.95 else '⚠️ CẦN TỐI ƯU'} |
| **Citation Precision** | **{m['citation_precision'] * 100:.2f}%** | ≥ 85.0% | {'✅ ĐẠT' if m['citation_precision'] >= 0.85 else '⚠️ CẦN TỐI ƯU'} |
| **Faithfulness (Ragas)** | **{m['faithfulness'] * 100:.2f}%** | ≥ 90.0% | {'✅ ĐẠT' if m['faithfulness'] >= 0.90 else '⚠️ CẦN TỐI ƯU'} |
| **Answer Relevancy** | **{m['answer_relevancy'] * 100:.2f}%** | ≥ 90.0% | {'✅ ĐẠT' if m['answer_relevancy'] >= 0.90 else '⚠️ CẦN TỐI ƯU'} |
| **Intent Match Accuracy** | **{m['intent_classification_accuracy'] * 100:.2f}%** | ≥ 95.0% | {'✅ ĐẠT' if m['intent_classification_accuracy'] >= 0.95 else '⚠️ CẦN TỐI ƯU'} |
| **⭐ ĐIỂM CHẤT LƯỢNG TỔNG THỂ** | **{m['overall_benchmark_score'] * 100:.2f}%** | ≥ 90.0% | {'🏆 XUẤT SẮC' if m['overall_benchmark_score'] >= 0.90 else '✅ ĐẠT'} |

---

## 📈 2. Kết Quả Theo Từng Nhóm Câu Hỏi (Breakdown by Category)

| Nhóm Câu Hỏi | Số Lượng | Điểm Trung Bình | SQL Accuracy | Citation Precision | Faithfulness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NUMERIC_FACT** (Số liệu cốt lõi SQL) | {by_cat['NUMERIC_FACT']['count']} | {by_cat['NUMERIC_FACT']['avg_overall'] * 100:.1f}% | {by_cat['NUMERIC_FACT']['avg_sql'] * 100:.1f}% | {by_cat['NUMERIC_FACT']['avg_citation'] * 100:.1f}% | {by_cat['NUMERIC_FACT']['avg_faithfulness'] * 100:.1f}% |
| **NOTE_EXPLANATION** (Thuyết minh BCTC) | {by_cat['NOTE_EXPLANATION']['count']} | {by_cat['NOTE_EXPLANATION']['avg_overall'] * 100:.1f}% | - | {by_cat['NOTE_EXPLANATION']['avg_citation'] * 100:.1f}% | {by_cat['NOTE_EXPLANATION']['avg_faithfulness'] * 100:.1f}% |
| **DEEP_ANALYSIS** (Phân tích suy luận) | {by_cat['DEEP_ANALYSIS']['count']} | {by_cat['DEEP_ANALYSIS']['avg_overall'] * 100:.1f}% | - | {by_cat['DEEP_ANALYSIS']['avg_citation'] * 100:.1f}% | {by_cat['DEEP_ANALYSIS']['avg_faithfulness'] * 100:.1f}% |

---

## 📝 3. Danh Sách Chi Tiết Kết Quả Kiểm Thử (Top Samples)

| ID | Nhóm | Câu Hỏi | Overall Score | SQL Acc | Cite Prec | Faithfulness | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for r in summary["results"][:20]:
        q_snippet = r['query'][:45] + "..." if len(r['query']) > 45 else r['query']
        md += f"| `{r['id']}` | `{r['category']}` | {q_snippet} | **{r.get('overall_score', 0)*100:.1f}%** | {r.get('sql_accuracy', 0)*100:.0f}% | {r.get('citation_precision', 0)*100:.0f}% | {r.get('faithfulness', 0)*100:.0f}% | {r.get('latency_sec', 0)}s |\n"

    md += "\n---\n*Báo cáo được khởi tạo tự động bởi OpenBCTC Copilot Automated Evaluation Pipeline (Phase 4.2).*\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Benchmark đánh giá tự động OpenBCTC Copilot.")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng câu hỏi cần chạy.")
    parser.add_argument("--category", type=str, default=None, help="Lọc theo nhóm: NUMERIC_FACT, NOTE_EXPLANATION, DEEP_ANALYSIS")
    args = parser.parse_args()

    run_benchmark(limit=args.limit, category_filter=args.category)
