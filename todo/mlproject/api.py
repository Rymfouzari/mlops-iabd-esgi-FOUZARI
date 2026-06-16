"""API FastAPI pour servir le modele de classification Breast Cancer."""
from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = MODEL_DIR / "model.joblib"

app = FastAPI(
    title="Breast Cancer Classification API",
    description="API de prediction tumeur maligne / benigne.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Dictionnaire contenant les variables du modele.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "target": "1 = malignant, 0 = benign",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in expected_features if col not in payload.features]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Variables manquantes",
                "missing_features": missing,
            },
        )

    row = {col: payload.features[col] for col in expected_features}
    df = pd.DataFrame([row], columns=expected_features)

    try:
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    label = "malignant" if prediction == 1 else "benign"

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability_malignant=probability,
    )