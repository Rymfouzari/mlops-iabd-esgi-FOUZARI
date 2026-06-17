"""Frontend Streamlit pour l'API Breast Cancer Classification."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001")

DEFAULT_FEATURES: dict[str, float] = {
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189,
}


def call_api(method: str, path: str, api_url: str, **kwargs: Any) -> dict[str, Any]:
    """Appelle l'API FastAPI et retourne la reponse JSON."""
    url = f"{api_url.rstrip('/')}/{path.lstrip('/')}"
    response = httpx.request(method, url, timeout=10.0, **kwargs)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Breast Cancer Classifier")
st.caption("Prédiction bénigne / maligne à partir des 30 variables du dataset Wisconsin.")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("URL de l'API", value=API_URL)

    if st.button("Vérifier l'API", use_container_width=True):
        try:
            health = call_api("GET", "/health", api_url)
        except httpx.HTTPError as exc:
            st.error(f"API indisponible : {exc}")
        else:
            st.success(f"API opérationnelle : {health}")

    try:
        info = call_api("GET", "/model-info", api_url)
    except httpx.HTTPError:
        st.warning("Informations modèle indisponibles.")
    else:
        st.write("Modèle chargé :", info.get("model_exists"))
        st.write("Nombre de variables :", info.get("n_features"))

predict_tab, info_tab, history_tab = st.tabs(
    ["🔍 Prédiction", "ℹ️ Modèle", "🕘 Historique"]
)

with predict_tab:
    st.subheader("Mesures de la tumeur")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        features: dict[str, float] = {}

        with col1:
            st.markdown("**Mesures moyennes**")
            for name in list(DEFAULT_FEATURES.keys())[:10]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        with col2:
            st.markdown("**Erreurs standards**")
            for name in list(DEFAULT_FEATURES.keys())[10:20]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        with col3:
            st.markdown("**Pires valeurs observées**")
            for name in list(DEFAULT_FEATURES.keys())[20:]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

    if submitted:
        payload = {"features": features}

        try:
            result = call_api("POST", "/predict", api_url, json=payload)
        except httpx.HTTPError as exc:
            st.error(f"Erreur lors de l'appel à l'API : {exc}")
        else:
            prediction = int(result["prediction"])
            label = result["label"]
            probability = float(result["probability_malignant"])

            st.divider()
            st.subheader("Résultat de la prédiction")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                if prediction == 1:
                    st.error("⚠️ Tumeur prédite : maligne")
                else:
                    st.success("✅ Tumeur prédite : bénigne")

            with res_col2:
                st.metric(
                    label="Probabilité de malignité",
                    value=f"{probability:.2%}",
                )
                st.progress(min(max(probability, 0.0), 1.0))

            with res_col3:
                st.write("Réponse brute API")
                st.json(result)

            if "predictions" not in st.session_state:
                st.session_state["predictions"] = []

            st.session_state["predictions"].append(
                {
                    "label": label,
                    "prediction": prediction,
                    "probability_malignant": probability,
                }
            )

with info_tab:
    st.subheader("Informations du modèle")

    try:
        info = call_api("GET", "/model-info", api_url)
    except httpx.HTTPError as exc:
        st.error(f"Impossible de récupérer les informations modèle : {exc}")
    else:
        st.json(info)

with history_tab:
    st.subheader("Historique local des prédictions")

    predictions = st.session_state.get("predictions", [])

    if not predictions:
        st.info("Aucune prédiction réalisée dans cette session.")
    else:
        st.dataframe(predictions, use_container_width=True)