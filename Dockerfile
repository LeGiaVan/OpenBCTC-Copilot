# Builder stage
FROM python:3.11-slim AS builder

# Cài đặt các dependencies hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv
RUN pip install uv

WORKDIR /app

# Copy các file cấu hình project
COPY pyproject.toml Plan.md ./

# Cài đặt dependencies system-wide thông qua uv
RUN uv pip install --system -e .

# Runner stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy packages đã cài từ builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy toàn bộ mã nguồn
COPY . .

# Tạo non-root user để tăng bảo mật
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Khởi chạy server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
