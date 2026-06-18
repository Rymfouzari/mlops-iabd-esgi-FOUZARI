"""Frontend Streamlit pour l'API Breast Cancer Classification.

Seance 14 bis - TP Streamlit
    Interface utilisateur pour interroger l'API FastAPI /predict.
    Lancement local :
        API_URL=http://127.0.0.1:8001 PYTHONPATH=todo streamlit run frontend/app.py
    En Docker Compose :
        API_URL=http://api:8000 est injecte automatiquement.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001")
MLFLOW_EXTERNAL_URL = os.environ.get("MLFLOW_EXTERNAL_URL", "http://localhost:5001")

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

MODEL_RESULTS = [
    {"Modele": "Baseline Logistic Regression", "F1": 0.951, "ROC-AUC": 0.996},
    {"Modele": "RandomForest GridSearchCV", "F1": 0.963, "ROC-AUC": 0.993},
    {"Modele": "XGBoost GridSearchCV", "F1": 0.950, "ROC-AUC": 0.995},
    {"Modele": "LightGBM Optuna", "F1": None, "ROC-AUC": 0.995},
]


def call_api(method: str, path: str, api_url: str, **kwargs: Any) -> dict[str, Any]:
    """Appelle l'API FastAPI et retourne la reponse JSON."""
    url = f"{api_url.rstrip('/')}/{path.lstrip('/')}"
    response = httpx.request(method, url, timeout=10.0, **kwargs)
    response.raise_for_status()
    return response.json()


def fetch_health(api_url: str) -> dict[str, Any]:
    """Recupere le statut de l'API."""
    try:
        return call_api("GET", "/health", api_url)
    except httpx.HTTPError:
        return {}


def fetch_model_info(api_url: str) -> dict[str, Any]:
    """Recupere les informations du modele charge par l'API."""
    try:
        return call_api("GET", "/model-info", api_url)
    except httpx.HTTPError:
        return {}


def init_session_state() -> None:
    """Initialise l'historique local des predictions."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🧬",
    layout="wide",
)

init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")
    api_url = st.text_input("URL de l'API", value=API_URL)

    health = fetch_health(api_url)
    model_info = fetch_model_info(api_url)

    if health.get("status") == "ok":
        st.success("API operationnelle")
    else:
        st.error("API indisponible")

    st.divider()

    model_exists = bool(model_info.get("model_exists", False))
    st.metric("Modele charge", "Oui" if model_exists else "Non")
    st.metric("Nombre de variables", model_info.get("n_features", 0))

    st.divider()
    st.link_button("Ouvrir MLflow", MLFLOW_EXTERNAL_URL, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------

tab_home, tab_predict, tab_experiments, tab_architecture, tab_monitoring = st.tabs(
    ["Accueil", "Prediction", "Experiences", "Architecture", "Monitoring"]
)

# ===========================================================================
# ACCUEIL
# ===========================================================================

with tab_home:
    st.title("Breast Cancer Wisconsin - Classification binaire")
    st.markdown(
        """
        Cette application expose un modele de classification binaire capable de predire
        si une tumeur est **benigne** ou **maligne** a partir de mesures numeriques
        extraites d'images medicales.

        Le frontend Streamlit appelle une API FastAPI qui charge le modele entraine
        depuis `models/model.joblib`.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", "569")
    col2.metric("Features", "30")
    col3.metric("Type", "Classification")
    col4.metric("Cible", "0 / 1")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Interpretation metier")
        st.markdown(
            """
            - `0` : tumeur predite comme **benigne**
            - `1` : tumeur predite comme **maligne**
            - la probabilite affichee correspond au risque estime de malignite
            """
        )

    with col_b:
        st.subheader("Stack technique")
        st.markdown(
            """
            - **Tracking** : MLflow
            - **Optimisation** : Optuna
            - **API** : FastAPI
            - **Frontend** : Streamlit
            - **Conteneurisation** : Docker Compose
            - **CI/CD** : GitHub Actions
            """
        )

    st.divider()
    st.subheader("Pipeline MLOps du projet")
    st.markdown(
        """
        `Dataset` → `Baseline` → `MLflow Tracking` → `GridSearchCV` →
        `Optuna` → `Model Registry` → `FastAPI` → `Streamlit` →
        `Docker Compose` → `CI/CD`
        """
    )

    with st.expander("Voir les 30 variables utilisees par le modele"):
        cols = st.columns(3)
        for i, feature in enumerate(DEFAULT_FEATURES):
            cols[i % 3].markdown(f"- `{feature}`")

# ===========================================================================
# PREDICTION
# ===========================================================================

with tab_predict:
    st.title("Tester une prediction")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        features: dict[str, float] = {}

        with col1:
            st.markdown("### Mesures moyennes")
            for name in list(DEFAULT_FEATURES.keys())[:10]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        with col2:
            st.markdown("### Erreurs standards")
            for name in list(DEFAULT_FEATURES.keys())[10:20]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        with col3:
            st.markdown("### Pires valeurs observees")
            for name in list(DEFAULT_FEATURES.keys())[20:]:
                features[name] = st.number_input(
                    name,
                    value=DEFAULT_FEATURES[name],
                    format="%.6f",
                )

        submitted = st.form_submit_button("Predire", use_container_width=True)

    if submitted:
        payload = {"features": features}

        try:
            result = call_api("POST", "/predict", api_url, json=payload)
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
            st.session_state["last_result"] = None
        else:
            prediction = int(result["prediction"])
            label = str(result["label"])
            probability = float(result["probability_malignant"])

            record = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "prediction": prediction,
                "label": label,
                "probability_malignant": probability,
                "feedback": None,
            }

            st.session_state["last_result"] = record
            st.session_state["history"].insert(0, record)

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        prediction = int(result["prediction"])
        probability = float(result["probability_malignant"])

        st.divider()
        st.subheader("Resultat de la prediction")

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if prediction == 1:
                st.error("Tumeur predite : maligne")
            else:
                st.success("Tumeur predite : benigne")

        with col_res2:
            st.metric("Probabilite de malignite", f"{probability:.2%}")
            st.progress(min(max(probability, 0.0), 1.0))

        with col_res3:
            st.metric("Heure", result["timestamp"])
            st.metric("Classe", prediction)

        st.divider()
        st.markdown("**Cette prediction est-elle correcte ?**")

        col_yes, col_no = st.columns(2)
        if col_yes.button("Correcte", use_container_width=True):
            st.session_state["history"][0]["feedback"] = "correct"
            st.success("Feedback enregistre.")
        if col_no.button("Incorrecte", use_container_width=True):
            st.session_state["history"][0]["feedback"] = "incorrect"
            st.warning("Feedback enregistre.")

        st.divider()
        st.subheader("Historique de la session")

        history = st.session_state.get("history", [])
        if history:
            rows = [
                {
                    "Heure": item["timestamp"],
                    "Prediction": "Maligne" if item["prediction"] == 1 else "Benigne",
                    "Probabilite malignite": f"{item['probability_malignant']:.2%}",
                    "Feedback": item.get("feedback") or "-",
                }
                for item in history
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            if st.button("Effacer l'historique"):
                st.session_state["history"] = []
                st.session_state["last_result"] = None
                st.rerun()

# ===========================================================================
# EXPERIENCES
# ===========================================================================

with tab_experiments:
    st.title("Experiences MLflow")

    st.markdown(
        """
        Cette section resume les experimentations realisees pendant le projet.
        Les runs detailles sont consultables directement dans l'interface MLflow.
        """
    )

    st.link_button("Ouvrir MLflow UI", MLFLOW_EXTERNAL_URL, use_container_width=True)

    st.divider()
    st.subheader("Resultats principaux")

    df_results = pd.DataFrame(MODEL_RESULTS)
    st.dataframe(df_results, use_container_width=True)

    chart_df = df_results.set_index("Modele")[["ROC-AUC"]]
    st.bar_chart(chart_df, use_container_width=True)

    st.divider()
    st.subheader("Ce qui est logge dans MLflow")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - Parametres d'entrainement
            - Metriques F1 et ROC-AUC
            - Dataset d'entrainement
            - Modele serialise
            """
        )

    with col2:
        st.markdown(
            """
            - Matrice de confusion
            - Artefacts SHAP
            - Trials Optuna
            - Model Registry
            """
        )

# ===========================================================================
# ARCHITECTURE
# ===========================================================================

with tab_architecture:
    st.title("Architecture MLOps")

    st.subheader("Role des principaux fichiers")
    architecture_rows = [
        {
            "Fichier": "train.py",
            "Role": "entraine la baseline Logistic Regression",
        },
        {
            "Fichier": "train_models.py",
            "Role": "compare plusieurs familles de modeles avec GridSearchCV",
        },
        {
            "Fichier": "train_optuna.py",
            "Role": "optimise RandomForest, XGBoost et LightGBM avec Optuna",
        },
        {
            "Fichier": "tracking.py",
            "Role": "centralise le logging MLflow",
        },
        {
            "Fichier": "api.py",
            "Role": "expose le modele via FastAPI",
        },
        {
            "Fichier": "frontend/app.py",
            "Role": "interface utilisateur Streamlit",
        },
        {
            "Fichier": "docker-compose.yml",
            "Role": "orchestre MLflow, entrainement, API et frontend",
        },
        {
            "Fichier": ".github/workflows/ci.yml",
            "Role": "execute lint, types, tests et entrainement baseline",
        },
        {
            "Fichier": ".github/workflows/cd.yml",
            "Role": "build et push l'image Docker de l'API vers GHCR",
        },
    ]

    st.dataframe(pd.DataFrame(architecture_rows), use_container_width=True)

    st.divider()
    st.subheader("Services Docker")
    st.markdown(
        """
        - `mlflow` : suivi des experiences
        - `train` : entrainement one-shot du modele
        - `api` : service FastAPI d'inference
        - `frontend` : interface Streamlit
        """
    )

# ===========================================================================
# MONITORING
# ===========================================================================

with tab_monitoring:
    st.title("Monitoring local")

    health = fetch_health(api_url)
    info = fetch_model_info(api_url)
    history = st.session_state.get("history", [])

    if health.get("status") == "ok" and info.get("model_exists"):
        st.success("API disponible et modele charge.")
    elif health.get("status") == "ok":
        st.warning("API disponible mais modele non charge.")
    else:
        st.error("API indisponible.")

    st.divider()

    total = len(history)
    malignant = sum(1 for item in history if item["prediction"] == 1)
    benign = total - malignant

    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions session", total)
    col2.metric("Malignes", malignant)
    col3.metric("Benignes", benign)

    if history:
        st.divider()
        st.subheader("Distribution des predictions")
        df_history = pd.DataFrame(history)
        df_distribution = (
            df_history["label"]
            .value_counts()
            .rename_axis("Classe")
            .reset_index(name="Nombre")
        )
        st.bar_chart(df_distribution.set_index("Classe"), use_container_width=True)

    st.divider()
    st.subheader("Liens utiles")
    st.link_button("Ouvrir MLflow UI", MLFLOW_EXTERNAL_URL, use_container_width=True)
