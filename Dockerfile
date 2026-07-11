FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib, scipy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps via pip (poetry export handles this in CI)
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

COPY . .
RUN pip install --no-cache-dir -e .

# Pre-create data directories
RUN mkdir -p data/demo data/charts data/uploads logs

# Generate demo datasets at build time
RUN python scripts/generate_demo_data.py || true

EXPOSE 8501 8000

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
