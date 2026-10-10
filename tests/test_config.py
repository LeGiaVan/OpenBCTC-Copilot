"""tests/test_config.py — Kiểm thử Module Cấu hình Tập trung Settings."""

import os
from src.core.config import Settings, settings


def test_settings_default_values():
    """Kiểm tra các giá trị mặc định của hệ thống."""
    s = Settings(_env_file=None)
    assert s.APP_PORT == 8000
    assert s.APP_HOST == "0.0.0.0"
    assert s.CHUNK_MIN_CHARS == 250
    assert s.CHUNK_MAX_CHARS == 800
    assert s.MONGO_DB == "openbctc"
    assert "financial_blocks" in s.QDRANT_COLLECTION_NAME


def test_settings_environment_override(monkeypatch):
    """Kiểm tra ghi đè cấu hình qua biến môi trường (12-Factor App)."""
    monkeypatch.setenv("APP_PORT", "9090")
    monkeypatch.setenv("CHUNK_MIN_CHARS", "300")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant-prod:6333")

    s = Settings(_env_file=None)
    assert s.APP_PORT == 9090
    assert s.CHUNK_MIN_CHARS == 300
    assert s.QDRANT_URL == "http://qdrant-prod:6333"


def test_singleton_settings_loaded():
    """Kiểm tra instance singleton sẵn sàng sử dụng."""
    assert settings is not None
    assert isinstance(settings.CHUNK_MIN_CHARS, int)
    assert isinstance(settings.CHUNK_MAX_CHARS, int)
