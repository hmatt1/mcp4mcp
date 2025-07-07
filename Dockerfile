FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY mcp4mcp/ ./mcp4mcp/
COPY server.py .
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 mcp4mcp && chown -R mcp4mcp:mcp4mcp /app
USER mcp4mcp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run server
CMD ["python", "server.py"]
