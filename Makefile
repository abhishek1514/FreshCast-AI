.PHONY: setup generate-data train evaluate test dashboard api mlflow-ui dvc-repro docker-build docker-up docker-down lint format clean prepare-real-data train-real

PYTHON := uv run python
STREAMLIT := uv run streamlit
PYTEST := uv run pytest
UVICORN := uv run uvicorn
MLFLOW := uv run mlflow
DVC := uv run dvc
BLACK := uv run black
RUFF := uv run ruff

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

generate-data:
	$(PYTHON) -m freshcast.data.generator

train:
	$(PYTHON) -m freshcast.models.train

prepare-real-data:
	$(PYTHON) scripts/prepare_real_data.py

train-real:
	$(PYTHON) scripts/train_real.py

evaluate:
	$(PYTHON) -m freshcast.metrics.evaluate

test:
	$(PYTEST) tests/ -v

dashboard:
	$(STREAMLIT) run app/streamlit_app.py

api:
	$(UVICORN) freshcast.api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	$(MLFLOW) ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db

dvc-repro:
	$(DVC) repro

dvc-status:
	$(DVC) status

docker-build:
	docker build -t freshcast:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

lint:
	$(BLACK) --check src/ tests/ app/
	$(RUFF) check src/ tests/ app/

format:
	$(BLACK) src/ tests/ app/
	$(RUFF) check --fix src/ tests/ app/

clean:
	rm -rf .pytest_cache
