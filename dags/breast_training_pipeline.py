"""Airflow DAG - pipeline d'entrainement Breast Cancer."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ENV = {
    "PYTHONPATH": "/opt/airflow/todo",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}

with DAG(
    dag_id="breast_training_pipeline",
    description="Validation dataset, baseline, comparaison de modeles et optimisation Optuna",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "training", "breast-cancer"],
) as dag:
    validate_dataset = BashOperator(
        task_id="validate_dataset",
        bash_command="test -f /opt/airflow/data/breast_cancer.csv",
        env=DEFAULT_ENV,
    )

    train_baseline = BashOperator(
        task_id="train_baseline",
        bash_command="python -m mlproject.train",
        env=DEFAULT_ENV,
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="python -m mlproject.train_models",
        env=DEFAULT_ENV,
    )

    train_optuna = BashOperator(
        task_id="train_optuna",
        bash_command="python -m mlproject.train_optuna --n-trials ${N_TRIALS:-5} --cv ${CV:-2}",
        env=DEFAULT_ENV,
    )

    validate_dataset >> train_baseline >> train_models >> train_optuna
