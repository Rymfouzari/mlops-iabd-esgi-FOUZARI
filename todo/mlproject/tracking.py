"""Fonctions utilitaires pour le suivi des experiences avec MLflow."""
from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn

from mlproject.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def setup_mlflow() -> None:
    """Configure le serveur MLflow et l'experience courante."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def start_run(run_name: str):
    """Demarre un run MLflow."""
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    """Enregistre les hyperparametres."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    """Enregistre les metriques."""
    mlflow.log_metrics(metrics)


def log_model(model: Any, name: str = "model") -> None:
    """Enregistre un modele scikit-learn."""
    mlflow.sklearn.log_model(model, name=name)


def log_artifact(path: str) -> None:
    """Enregistre un artefact local."""
    mlflow.log_artifact(path)