import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

from src.core.config import settings
from src.agents.copilot.graph import build_copilot_graph
from src.services.sql_engine import SQLiteFactService
from src.services.vector_engine import VectorEngineService
from src.services.retriever import HybridRetriever

from src.api.routes import chat, pdf, ingest, feedback
from src.services.fact_manager import DynamicFactService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo services và graph khi startup
    sql_svc = DynamicFactService(base_dir=settings.FACTS_BASE_DIR)
    
    qdrant_url = settings.QDRANT_URL
    try:
        vec_svc = VectorEngineService(url=qdrant_url)
    except Exception:
        vec_svc = VectorEngineService(path="data/qdrant_storage")
    retriever = HybridRetriever(vec_svc=vec_svc)
    
    # Warm-up pre-load AI models lúc server khởi động để tránh cold start khi user hỏi lần đầu
    logger.info("🔥 Đang khởi động warm-up AI models (Dense, Sparse, Reranker)...")
    try:
        _ = vec_svc.dense_model
        _ = vec_svc.sparse_model
        if retriever.use_reranker:
            _ = retriever.reranker
        logger.info("✅ Warm-up hoàn tất: Tất cả AI models đã sẵn sàng trong RAM.")
    except Exception as e:
        logger.warning("Cảnh báo trong quá trình warm-up models: %s", e)
    
    # Xác định đúng Provider (Groq, OpenAI hoặc OpenRouter)
    groq_key = settings.GROQ_API_KEY
    openai_key = settings.OPENAI_API_KEY
    openrouter_key = settings.OPENROUTER_API_KEY
    
    base_url = settings.LLM_BASE_URL
    model_name = settings.LLM_MODEL
    
    if groq_key and not settings.FORCE_OPENAI:
        api_key = groq_key
        base_url = base_url or "https://api.groq.com/openai/v1"
        model_name = model_name or settings.GROQ_MODEL
    elif openai_key:
        api_key = openai_key
        base_url = base_url or settings.OPENAI_BASE_URL
        model_name = model_name or settings.OPENAI_MODEL
    elif openrouter_key:
        api_key = openrouter_key
        base_url = base_url or "https://openrouter.ai/api/v1"
        model_name = model_name or "openai/gpt-oss-120b"
    else:
        api_key = "dummy-api-key"
        model_name = "openai/gpt-oss-120b"

    logger.info("🤖 Khởi tạo LLM: Provider model=%s, base_url=%s", model_name, base_url)
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0
    )
    
    graph = build_copilot_graph(
        sql_svc=sql_svc,
        retriever=retriever,
        llm=llm,
        use_memory_saver=True,
        langfuse_enabled=False
    )
    
    app.state.graph = graph
    yield
    # Cleanup khi shutdown nếu cần

app = FastAPI(
    title="OpenBCTC Copilot API",
    description="Enterprise Financial Copilot API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.include_router(chat.router, prefix="/api/v1")
app.include_router(pdf.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")

static_dir = Path("src/api/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")

    return HTMLResponse("""
    <html>
        <head><title>OpenBCTC Copilot API</title></head>
        <body>
            <h1>OpenBCTC Copilot API</h1>
            <p>The backend is running successfully.</p>
            <p>Frontend assets are not present yet, so the API is serving a fallback page.</p>
        </body>
    </html>
    """)

@app.get("/healthz")
def health_check():
    return {"status": "ok", "service": "openbctc-copilot-api"}


@app.get("/readyz")
def readiness_check():
    graph_ready = hasattr(app.state, "graph")
    return {
        "status": "ready" if graph_ready else "starting",
        "graph_ready": graph_ready,
    }
