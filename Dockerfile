# HuggingFace Spaces Dockerfile for CascadeDebug
# Uses Docker SDK — port 7860 is required by HF Spaces

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy all project files
COPY . /app/

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose HF Spaces required port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
