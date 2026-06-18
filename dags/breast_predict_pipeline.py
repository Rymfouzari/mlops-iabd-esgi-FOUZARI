"""Airflow DAG - pipeline de prediction Breast Cancer."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ENV = {
    "PYTHONPATH": "/opt/airflow/todo",
    "API_URL": "http://api:8000",
}

with DAG(
    dag_id="breast_predict_pipeline",
    description="Verification API puis envoi de payloads de test vers /predict",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "prediction", "breast-cancer"],
) as dag:
    check_api_health = BashOperator(
        task_id="check_api_health",
        bash_command=(
            "python -c \"import urllib.request; "
            "urllib.request.urlopen('http://api:8000/health')\""
        ),
        env=DEFAULT_ENV,
    )

    run_predict_client = BashOperator(
        task_id="run_predict_client",
        bash_command="python /opt/airflow/todo/scripts/predict_client.py --url http://api:8000",
        env=DEFAULT_ENV,
    )

    check_api_health >> run_predict_client
