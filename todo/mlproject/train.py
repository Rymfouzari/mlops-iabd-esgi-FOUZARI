"""Entrainement du modele de classification (baseline).

Seance 5 - TP MLflow Tracking
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
)
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # S5-1 / S5-2
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # S5-3
    with mlflow.start_run(run_name=f"logreg-c{c}"):

        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }

        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        # S5-4
        mlflow.log_params(
            {
                "c": c,
                "max_iter": max_iter,
                "model": "logreg",
            }
        )

        # S5-5
        mlflow.log_metrics(metrics)

        # S5-6
        mlflow.sklearn.log_model(model, name="model")

        # S5-7 bonus
        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.savefig("confusion.png", bbox_inches="tight")
        plt.close()
        mlflow.log_artifact("confusion.png")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)

    args = parser.parse_args()

    train(
        c=args.c,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()