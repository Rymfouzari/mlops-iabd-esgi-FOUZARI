from __future__ import annotations

import warnings

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mlproject.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor

warnings.filterwarnings("ignore", module="mlflow")


MODELS = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 5],
        },
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.05, 0.1],
        },
    ),
}


def train_all() -> None:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for name, (estimator, param_grid) in MODELS.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("clf", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)

        best = search.best_estimator_
        proba = best.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best, name="model")

        print(
            f"{name:20s} -> "
            f"f1={metrics['f1']:.3f}  "
            f"roc_auc={metrics['roc_auc']:.3f}  "
            f"params={search.best_params_}"
        )


if __name__ == "__main__":
    train_all()