# RagGem - hardened RAG API server, packaged as a single plug-and-play image.
#
#   docker build -t raggem .
#   docker run -p 8100:8100 --env-file .env -v raggem-data:/app/data raggem
#
# The container exposes the HTTP API on port 8100. Knowledge bases ("brains")
# and their vector stores live under /app/data - mount a volume to persist them.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGGEM_PORT=8100

WORKDIR /app

# Dependency manifests first so Docker layer-caches the (slow) pip install
# whenever only source changes.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# All runtime deps ship as manylinux wheels for cp312, so no compiler is
# needed. If a future dependency lacks a wheel, uncomment the build tools:
#   RUN apt-get update && apt-get install -y --no-install-recommends gcc \
#       && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install .

# Persisted knowledge + vector stores live here; declared as a volume so a
# plain `docker run` keeps them across restarts.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8100

# Liveness check against the unauthenticated /health endpoint (no curl in slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health',timeout=4).status==200 else 1)"

# API_ADMIN_KEY must be provided at runtime (the server refuses to start
# without it). GOOGLE_API_KEY is required for embeddings + generation.
CMD ["uvicorn", "raggem.server:app", "--host", "0.0.0.0", "--port", "8100"]
