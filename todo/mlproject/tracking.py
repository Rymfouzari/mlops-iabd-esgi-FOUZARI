"""Fonctions utilitaires pour le suivi des experiences avec MLflow."""
from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from mlproject.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def setup_mlflow() -> None:
    """Configure le serveur MLflow et l'experience courante."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def log_dataset(df: pd.DataFrame, context: str = "training") -> None:
    """Loggue le dataset dans MLflow."""
    mlflow.log_param(f"{context}_rows", df.shape[0])
    mlflow.log_param(f"{context}_columns", df.shape[1])
    mlflow.log_param(f"{context}_missing_values", int(df.isna().sum().sum()))

    dataset = mlflow.data.from_pandas(
        df,
        name="breast_cancer",
    )
    mlflow.log_input(dataset, context=context)


def log_params(params: dict[str, Any]) -> None:
    """Loggue les parametres d'un run MLflow."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    """Loggue les metriques d'un run MLflow."""
    mlflow.log_metrics(metrics)


def log_model(model: Any, name: str = "model") -> None:
    """Loggue un modele scikit-learn dans MLflow."""
    mlflow.sklearn.log_model(model, name=name)


def log_artifact(path: str) -> None:
    """Loggue un artefact local dans MLflow."""
    mlflow.log_artifact(path)