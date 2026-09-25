# ---------------------------------------------------------------------------
# NEPENTHE | Auto-SOF — container image
# ---------------------------------------------------------------------------
# Build:  docker build -t auto-sof .
# Run:    docker run --rm -p 8501:8501 auto-sof
# Compose: docker compose up
#
# The app serves on http://localhost:8501 (override with --server.port).
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS base

# --- System layer ----------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# --- Dependency layer (cached until requirements.txt changes) ---------------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Application layer ------------------------------------------------------
COPY app.py LICENSE README.md CITATION.cff ./
COPY core ./core
COPY tests ./tests
COPY ui ./ui

# --- Run as an unprivileged user --------------------------------------------
RUN useradd --create-home --uid 1000 sofuser \
    && chown -R sofuser:sofuser /app
USER sofuser

EXPOSE 8501

# Streamlit's built-in health endpoint (plain "ok") for orchestrators.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()==b'ok' else 1)"

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
