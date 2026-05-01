FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . && pip cache purge

COPY ai_gateway/ /app/ai_gateway/

RUN useradd -r -s /usr/sbin/nologin gateway
USER gateway

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["uvicorn", "ai_gateway.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
