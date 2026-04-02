FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir .

# Default to SSE transport for Docker
ENV JPSTOCK_MCP_TRANSPORT=sse
ENV JPSTOCK_MCP_HOST=0.0.0.0
ENV JPSTOCK_MCP_PORT=8000

EXPOSE 8000

CMD ["jpstock-agent", "serve"]
