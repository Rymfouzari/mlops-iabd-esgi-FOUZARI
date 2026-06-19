"""DAG Airflow - trafic de predictions quotidien Breast Cancer."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "rym-fouzari",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    import httpx

    from mlproject.config import TARGET
    from mlproject.data import load_data

    api_url = os.environ.get("API_URL", "http://api:8000")

    df = load_data()
    features = df.drop(columns=[TARGET])
    sample = features.sample(n=min(N_PREDICTIONS, len(features)), random_state=None)

    sent = 0
    with httpx.Client(base_url=api_url, timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()

        for _, row in sample.iterrows():
            payload = {"features": json.loads(row.to_json())}
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            sent += 1

    logger.info("%d predictions envoyees a %s", sent, api_url)


with DAG(
    dag_id="breast_daily_predictions",
    description="Envoie 20 predictions par jour a l'API pour simuler un trafic de production",
    schedule="0 */5 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["breast-cancer", "predictions", "mlops"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
