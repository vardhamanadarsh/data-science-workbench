FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# libgomp1 is needed by XGBoost and LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Choose extras at build time, e.g.:
#   docker build --build-arg EXTRAS="--extra ml --extra explain --extra reports --extra automl" -t workbench .
ARG EXTRAS="--extra ml --extra explain --extra reports"

# Dependencies first, so this layer is cached when only the code changes
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev ${EXTRAS}

COPY . .

RUN useradd --create-home appuser \
    && mkdir -p /app/experiments \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]