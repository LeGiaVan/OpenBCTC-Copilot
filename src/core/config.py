"""src/core/config.py — Centralized Application Settings (12-Factor App & Externalized Configuration)."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Quản lý toàn bộ cấu hình hệ thống Copilot tập trung qua file .env và biến môi trường."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Ứng dụng & Server Web
    APP_ENV: str = Field(default="dev", description="Môi trường chạy: dev / staging / prod")
    APP_HOST: str = Field(default="0.0.0.0", description="Host bind API")
    APP_PORT: int = Field(default=8000, description="Port chạy API")

    # 2. Database & Data Directory
    FACTS_BASE_DIR: str = Field(default="/app/data", description="Thư mục gốc chứa facts SQLite và data BCTC")
    SQLITE_DB_PATH: str = Field(default="data/financial_reports.db", description="Đường dẫn file SQLite BCTC")
    MONGO_URI: str = Field(default="mongodb://mongodb:27017", description="URI kết nối MongoDB")
    MONGO_DB: str = Field(default="openbctc", description="Tên Database MongoDB")

    # 3. Vector Database (Qdrant)
    QDRANT_URL: str = Field(default="http://qdrant:6333", description="Địa chỉ máy chủ Qdrant")
    QDRANT_API_KEY: str | None = Field(default=None, description="API Key cho Qdrant nếu dùng Cloud")
    QDRANT_COLLECTION_NAME: str = Field(default="financial_blocks", description="Tên collection trong Qdrant")

    # 4. LLM Providers & Models
    GROQ_API_KEY: str | None = Field(default=None, description="API Key cho Groq")
    GROQ_MODEL: str = Field(default="openai/gpt-oss-20b", description="Model mặc định cho Groq")
    OPENAI_API_KEY: str | None = Field(default=None, description="API Key cho OpenAI")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Base URL cho OpenAI API")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="Model mặc định cho OpenAI")
    OPENROUTER_API_KEY: str | None = Field(default=None, description="API Key cho OpenRouter")
    LLM_BASE_URL: str | None = Field(default=None, description="Tuỳ biến Base URL LLM")
    LLM_MODEL: str | None = Field(default=None, description="Tuỳ biến tên Model LLM")
    FORCE_OPENAI: bool = Field(default=False, description="Bắt buộc dùng OpenAI thay vì Groq")

    # 5. Semantic Chunking (Triệt tiêu Magic Numbers)
    CHUNK_MIN_CHARS: int = Field(default=250, description="Độ dài tối thiểu của một semantic chunk (ký tự)")
    CHUNK_MAX_CHARS: int = Field(default=800, description="Độ dài tối đa của một semantic chunk (ký tự)")

    # 6. Observability (Langfuse)
    LANGFUSE_PUBLIC_KEY: str | None = Field(default=None, description="Public Key cho Langfuse")
    LANGFUSE_SECRET_KEY: str | None = Field(default=None, description="Secret Key cho Langfuse")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com", description="Host Langfuse")


# Singleton instance sử dụng xuyên suốt toàn bộ ứng dụng
settings = Settings()
