FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY README.md .

# Install the package (with ML support)
RUN pip install --no-cache-dir ".[ml]"

# Create data directory for keys + usage DB
RUN mkdir -p /data/.jpstock

# Default env: SSE transport for Docker, data stored in /data
ENV JPSTOCK_MCP_TRANSPORT=sse
ENV JPSTOCK_MCP_HOST=0.0.0.0
ENV JPSTOCK_MCP_PORT=8000
ENV JPSTOCK_AUTH_KEY_FILE=/data/.jpstock/keys.json
ENV JPSTOCK_RATE_LIMIT_ENABLED=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["jpstock-agent", "serve"]
