import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not any([os.getenv("OPENAI_API_KEY"), os.getenv("GROQ_API_KEY"), os.getenv("OPENROUTER_API_KEY")]):
    print("Vui lòng thiết lập API_KEY (OPENAI / GROQ / OPENROUTER) trong file .env trước khi chạy.")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from src.agents.copilot.graph import build_copilot_graph, run_query
from src.services.sql_engine import SQLiteFactService
from src.services.vector_engine import VectorEngineService
from src.services.retriever import HybridRetriever

def main():
    print("Khởi tạo hệ thống, vui lòng đợi (có thể mất chút thời gian nạp model)...")
    
    # Khởi tạo SQL Engine
    sql_svc = SQLiteFactService(db_path="data/VNM_2025/benchmark_VNM_2025.db")
    
    # Khởi tạo Vector Engine (trỏ tới thư mục dữ liệu đã nạp)
    vec_svc = VectorEngineService(path="data/qdrant_storage")
    
    # Khởi tạo Retriever & LLM
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        if os.getenv("GROQ_API_KEY"):
            base_url = "https://api.groq.com/openai/v1"
        elif os.getenv("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
            
    retriever = HybridRetriever(vec_svc=vec_svc)
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        base_url=base_url,
        temperature=0
    )
    
    # Xây dựng LangGraph
    graph = build_copilot_graph(
        sql_svc=sql_svc,
        retriever=retriever,
        llm=llm,
        use_memory_saver=True,
        langfuse_enabled=False
    )
    
    print("\n" + "="*50)
    print("🤖 OpenBCTC Copilot CLI (VNM 2025)")
    print("Nhập 'quit' hoặc 'exit' để thoát.")
    print("="*50 + "\n")
    
    thread_id = "test-session-1"
    
    while True:
        query = input("\nBạn: ")
        if query.strip().lower() in ['quit', 'exit']:
            break
            
        print("\nCopilot đang suy nghĩ...\n")
        
        try:
            result = run_query(
                graph=graph,
                query=query,
                company="VNM",
                year=2025,
                thread_id=thread_id,
                langfuse_enabled=False
            )
            
            print("🤖 AI:", result["final_answer"])
            print(f"\n[Metadata] Intent: {result['intent'].value if result.get('intent') else 'N/A'}, Fact-Check: {result.get('fact_check_status')}")
            
        except Exception as e:
            print(f"\nLỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
