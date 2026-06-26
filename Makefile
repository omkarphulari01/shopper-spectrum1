.PHONY: help install install-dev train recommend app test lint format clean docker-build docker-run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -e .

install-dev:  ## Install dev dependencies + package in editable mode
	pip install -e ".[dev]"

train:  ## Run the full training pipeline
	shopper-spectrum train

recommend:  ## Get recommendations, e.g. make recommend P="WHITE METAL LANTERN"
	shopper-spectrum recommend "$(P)"

app:  ## Launch the Streamlit app
	streamlit run app/app.py

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Auto-format with ruff
	ruff check --fix src tests

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache __pycache__ */__pycache__ **/__pycache__ .ruff_cache
	rm -rf models/*.pkl reports/figures/*.png data/processed/*.parquet

docker-build:  ## Build the Docker image
	docker build -t shopper-spectrum .

docker-run:  ## Run the app in Docker on port 8501
	docker run -p 8501:8501 shopper-spectrum
