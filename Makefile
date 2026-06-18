# ==============================================================================
# Breast Cancer Classification - Pipeline MLOps
# ==============================================================================
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL          := /bin/sh
PYTHON         := uv run python
RUN            := uv run
VENV_DIR       := .venv
PYTHONPATH     ?= todo
export PYTHONPATH

API_HOST       ?= 127.0.0.1
API_PORT       ?= 8001
FRONTEND_PORT  ?= 8501
MLFLOW_PORT    ?= 5001
AIRFLOW_PORT   ?= 8080

C              ?= 1.0
MAX_ITER       ?= 1000
CV             ?= 5
SCORING        ?= roc_auc
N_TRIALS       ?= 30

YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
	check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
	data train train-models train-optuna mlflow-server mlflow api predict-client frontend \
	docker-build docker-run docker-up docker-down docker-logs docker-ps \
	airflow-build airflow-up airflow-down airflow-logs workflow-docker \
	lint format type test check

# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ==============================================================================
# Setup
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide
	uv venv $(VENV_DIR)

deps-sync: check-uv ## Installe les dependances projet + dev
	uv sync --extra dev

install: deps-sync ## Installe l'environnement complet

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Actualise uv.lock
	uv lock

reset-env: check-uv ## Reinitialise .venv et uv.lock
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev

doctor: check-uv check-venv ## Diagnostique l'environnement
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"

# ==============================================================================
# Pipeline ML local
# ==============================================================================

data: ## Verifie la presence du dataset
	@test -f data/breast_cancer.csv \
		&& echo "$(GREEN)[OK] data/breast_cancer.csv present$(RESET)" \
		|| { echo "$(RED)[ERREUR] Dataset introuvable$(RESET)"; exit 1; }

train: data ## Entraine la baseline LogisticRegression
	PYTHONPATH=todo $(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train-models: data ## Compare RandomForest / XGBoost / LightGBM
	PYTHONPATH=todo $(PYTHON) -m mlproject.train_models

train-optuna: data ## Optimise les modeles avec Optuna
	PYTHONPATH=todo $(PYTHON) -m mlproject.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

mlflow-server: ## Demarre MLflow localement
	$(RUN) mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

mlflow: mlflow-server ## Alias de mlflow-server

api: ## Lance l'API FastAPI localement
	PYTHONPATH=todo $(RUN) uvicorn mlproject.api:app --reload --host $(API_HOST) --port $(API_PORT)

predict-client: ## Teste l'API locale
	PYTHONPATH=todo $(PYTHON) todo/scripts/predict_client.py --url http://127.0.0.1:$(API_PORT)

frontend: ## Lance Streamlit localement
	API_URL=http://127.0.0.1:$(API_PORT) PYTHONPATH=todo \
	$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)

# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit les images Docker
	docker compose build mlflow api frontend airflow

docker-run: ## Lance l'entrainement Docker one-shot
	docker compose --profile train run --rm train

docker-up: ## Demarre la stack MLOps
	docker compose up -d --build mlflow api frontend airflow

docker-down: ## Arrete et supprime les conteneurs
	docker compose down --remove-orphans

docker-ps: ## Affiche les conteneurs
	docker compose ps

docker-logs: ## Affiche les logs Docker
	docker compose logs --tail=100

# ==============================================================================
# Airflow / Orchestration
# ==============================================================================

airflow-build: ## Construit l'image Airflow
	docker compose build airflow

airflow-up: ## Demarre Airflow avec la stack
	docker compose up -d --build mlflow api frontend airflow

airflow-down: ## Arrete Airflow
	docker compose stop airflow

airflow-logs: ## Affiche les logs Airflow
	docker compose logs airflow --tail=100

workflow-docker: ## Lance le workflow MLOps complet
	docker compose up -d --build mlflow
	docker compose --profile train run --rm train
	docker compose up -d --build api frontend airflow

# ==============================================================================
# Qualite
# ==============================================================================

lint: ## Verifie le style avec Ruff
	PYTHONPATH=todo $(RUN) ruff check todo/mlproject tests todo/scripts frontend

format: ## Formate le code avec Ruff
	PYTHONPATH=todo $(RUN) ruff format todo/mlproject tests todo/scripts frontend

type: ## Verifie les types avec MyPy
	PYTHONPATH=todo $(RUN) mypy todo/mlproject

test: ## Lance les tests Pytest
	PYTHONPATH=todo $(RUN) pytest

check: lint type test ## Lance lint + type + tests