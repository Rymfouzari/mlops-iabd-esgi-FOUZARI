# ==============================================================================
# Projet de classification - Makefile (squelette)
# ==============================================================================
# Seuls les targets d'INSTALLATION sont fournis. Les autres sont a completer
# au fil des TP (un `# TODO (Sx)` indique la commande attendue).
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8001
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5001
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna mlflow api predict-client frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: ## Verifie que le jeu de donnees est present dans data/
	@test -f data/breast_cancer.csv || { \
		echo "$(RED)[ERREUR] Dataset manquant : data/breast_cancer.csv$(RESET)"; \
		exit 1; \
	}
	@echo "$(GREEN)[OK] Dataset trouve : data/breast_cancer.csv$(RESET)"

train: ## Entraine la baseline LogisticRegression
	PYTHONPATH=todo $(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RandomForest / XGBoost / LightGBM avec GridSearchCV
	PYTHONPATH=todo $(PYTHON) -m mlproject.train_models

train-optuna: ## Optimise RandomForest / XGBoost / LightGBM avec Optuna
	PYTHONPATH=todo $(PYTHON) -m mlproject.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

mlflow: ## Demarre MLflow localement
	$(RUN) mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

api: ## Lance l'API FastAPI en local
	PYTHONPATH=todo $(RUN) uvicorn mlproject.api:app --reload --host $(API_HOST) --port $(API_PORT)

predict-client: ## Teste l'API avec quelques exemples du dataset
	PYTHONPATH=todo $(PYTHON) todo/scripts/predict_client.py --url http://127.0.0.1:$(API_PORT)

frontend: ## Lance le frontend Streamlit
	API_URL=http://127.0.0.1:$(API_PORT) PYTHONPATH=todo $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)
# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit les images Docker train et API
	docker compose build train api

docker-run: ## Lance l'entrainement en conteneur Docker
	docker compose --profile train run --rm train

docker-up: ## Demarre la stack Docker MLflow + API
	docker compose up -d mlflow api

docker-down: ## Arrete et supprime les conteneurs Docker
	docker compose down --remove-orphans
# ==============================================================================
# Qualite
# ==============================================================================

lint: ## Verifie le style (ruff)
	PYTHONPATH=todo $(RUN) ruff check todo/mlproject tests todo/scripts

format: ## Formate le code (ruff)
	PYTHONPATH=todo $(RUN) ruff format todo/mlproject tests todo/scripts

type: ## Verifie les types (mypy)
	PYTHONPATH=todo $(RUN) mypy todo/mlproject

test: ## Lance les tests (pytest)
	PYTHONPATH=todo $(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)