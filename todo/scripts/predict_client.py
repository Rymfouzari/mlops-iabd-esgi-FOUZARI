"""Client de test pour l'API FastAPI du modele.

Seance 15 - TP Tests de l'API
"""
from __future__ import annotations

import argparse
import json
import logging

import httpx

from mlproject.config import API_URL, TARGET
from mlproject.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 3


def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construire n payloads de test a partir du jeu de donnees."""
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=n, random_state=42)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=API_URL,
        help="URL de base de l'API (defaut: %(default)s)",
    )
    args = parser.parse_args()

    payloads = build_payloads()

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health")
        logger.info("GET /health -> %s %s", health.status_code, health.json())

        for i, payload in enumerate(payloads):
            response = client.post(
                "/predict",
                json={"features": payload},
            )
            logger.info(
                "POST /predict (#%d) -> %s %s",
                i,
                response.status_code,
                response.json(),
            )

        info = client.get("/model-info")
        logger.info("GET /model-info -> %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()