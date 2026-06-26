FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scientific wheels are prebuilt.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY . .

# Train at build time so the image ships with models, IF the dataset is present.
# (Skipped gracefully if data/raw/online_retail.csv is absent.)
RUN if [ -f data/raw/online_retail.csv ]; then shopper-spectrum train; fi

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
