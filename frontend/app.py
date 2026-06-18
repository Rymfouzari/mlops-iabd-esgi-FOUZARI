"""Frontend Streamlit stylé pour l'API Breast Cancer Classification.

Dashboard MLOps complet :
- prédiction via FastAPI
- statut API / modèle
- comparaison d'expériences
- architecture MLOps
- orchestration Airflow
- monitoring de session
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001")
API_DOCS_URL = os.environ.get("API_DOCS_URL", "http://88.96.61.63:8000/docs")
MLFLOW_EXTERNAL_URL = os.environ.get("MLFLOW_EXTERNAL_URL", "http://88.96.61.63:5000")
AIRFLOW_EXTERNAL_URL = os.environ.get("AIRFLOW_EXTERNAL_URL", "http://88.96.61.63:8080")
GITHUB_URL = os.environ.get(
    "GITHUB_URL",
    "https://github.com/Rymfouzari/mlops-iabd-esgi-FOUZARI",
)

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

ALT_EXAMPLE: dict[str, float] = {
    "mean radius": 11.42,
    "mean texture": 20.38,
    "mean perimeter": 77.58,
    "mean area": 386.1,
    "mean smoothness": 0.1425,
    "mean compactness": 0.2839,
    "mean concavity": 0.2414,
    "mean concave points": 0.1052,
    "mean symmetry": 0.2597,
    "mean fractal dimension": 0.09744,
    "radius error": 0.4956,
    "texture error": 1.156,
    "perimeter error": 3.445,
    "area error": 27.23,
    "smoothness error": 0.00911,
    "compactness error": 0.07458,
    "concavity error": 0.05661,
    "concave points error": 0.01867,
    "symmetry error": 0.05963,
    "fractal dimension error": 0.009208,
    "worst radius": 14.91,
    "worst texture": 26.5,
    "worst perimeter": 98.87,
    "worst area": 567.7,
    "worst smoothness": 0.2098,
    "worst compactness": 0.8663,
    "worst concavity": 0.6869,
    "worst concave points": 0.2575,
    "worst symmetry": 0.6638,
    "worst fractal dimension": 0.173,
}

FEATURE_GROUPS = {
    "Mesures moyennes": list(DEFAULT_FEATURES.keys())[:10],
    "Erreurs standards": list(DEFAULT_FEATURES.keys())[10:20],
    "Pires valeurs observées": list(DEFAULT_FEATURES.keys())[20:],
}

MODEL_RESULTS = [
    {"Modèle": "Logistic Regression", "Approche": "Baseline", "F1": 0.951, "ROC-AUC": 0.996},
    {"Modèle": "RandomForest", "Approche": "GridSearchCV", "F1": 0.963, "ROC-AUC": 0.993},
    {"Modèle": "XGBoost", "Approche": "GridSearchCV", "F1": 0.950, "ROC-AUC": 0.995},
    {"Modèle": "LightGBM", "Approche": "Optuna", "F1": None, "ROC-AUC": 0.995},
]

MLOPS_STEPS = [
    ("01", "Data", "Dataset Breast Cancer Wisconsin"),
    ("02", "Baseline", "Pipeline sklearn reproductible"),
    ("03", "Tracking", "MLflow params, métriques, artefacts"),
    ("04", "Search", "GridSearchCV multi-modèles"),
    ("05", "Optuna", "Optimisation automatique"),
    ("06", "Registry", "Sauvegarde du modèle"),
    ("07", "API", "FastAPI pour l'inférence"),
    ("08", "UI", "Dashboard Streamlit"),
    ("09", "Airflow", "Orchestration des DAGs MLOps"),
    ("10", "Docker", "Orchestration compose"),
    ("11", "CI/CD", "GitHub Actions + GHCR"),
    ("12", "Cloud", "Déploiement Oracle"),
]

AIRFLOW_DAGS = [
    {
        "DAG": "breast_training_pipeline",
        "Étape 1": "validate_dataset",
        "Étape 2": "train_baseline",
        "Étape 3": "train_optuna",
        "Rôle": "Orchestration du pipeline d'entraînement",
    },
    {
        "DAG": "breast_predict_pipeline",
        "Étape 1": "check_api_health",
        "Étape 2": "run_predict_client",
        "Étape 3": "-",
        "Rôle": "Validation automatique de l'API de prédiction",
    },
]



def call_api(method: str, path: str, api_url: str, **kwargs: Any) -> dict[str, Any]:
    """Appelle l'API FastAPI et retourne la réponse JSON."""
    url = f"{api_url.rstrip('/')}/{path.lstrip('/')}"
    response = httpx.request(method, url, timeout=10.0, **kwargs)
    response.raise_for_status()
    return response.json()


def fetch_health(api_url: str) -> dict[str, Any]:
    try:
        return call_api("GET", "/health", api_url)
    except httpx.HTTPError:
        return {}


def fetch_model_info(api_url: str) -> dict[str, Any]:
    try:
        return call_api("GET", "/model-info", api_url)
    except httpx.HTTPError:
        return {}


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "active_example" not in st.session_state:
        st.session_state["active_example"] = DEFAULT_FEATURES.copy()


def prediction_label(prediction: int) -> str:
    return "Maligne" if prediction == 1 else "Bénigne"


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "Risque élevé"
    if probability >= 0.40:
        return "Risque intermédiaire"
    return "Risque faible"


def risk_class(probability: float) -> str:
    if probability >= 0.75:
        return "danger"
    if probability >= 0.40:
        return "warning"
    return "success"


def add_history(result: dict[str, Any], features: dict[str, float]) -> None:
    prediction = int(result["prediction"])
    probability = float(result["probability_malignant"])
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "label": prediction_label(prediction),
        "probability_malignant": probability,
        "risk": risk_band(probability),
        "features": features.copy(),
    }
    st.session_state["last_result"] = record
    st.session_state["history"].insert(0, record)


def history_df(history: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": item["timestamp"],
                "Prédiction": item["label"],
                "Probabilité malignité": f"{item['probability_malignant']:.2%}",
                "Niveau": item["risk"],
            }
            for item in history
        ]
    )


st.set_page_config(
    page_title="Rym FOUZARI · MLOps Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 12%, rgba(236, 72, 153, 0.20), transparent 28%),
                radial-gradient(circle at 88% 8%, rgba(56, 189, 248, 0.22), transparent 30%),
                radial-gradient(circle at 50% 95%, rgba(124, 58, 237, 0.16), transparent 33%),
                linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #fdf2f8 100%);
            color: #0f172a;
        }

        @keyframes floatUp {
            0% { transform: translateY(12px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

        @keyframes glowPulse {
            0% { box-shadow: 0 18px 45px rgba(59, 130, 246, 0.22); }
            50% { box-shadow: 0 18px 55px rgba(236, 72, 153, 0.34); }
            100% { box-shadow: 0 18px 45px rgba(59, 130, 246, 0.22); }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.4rem;
            animation: floatUp 0.55s ease-out;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(244, 114, 182, 0.55), transparent 35%),
                radial-gradient(circle at 100% 18%, rgba(34, 211, 238, 0.35), transparent 34%),
                linear-gradient(180deg, #0f172a 0%, #312e81 48%, #0f766e 100%);
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            overflow-y: auto;
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] input {
            color: #0f172a !important;
            background: #ffffff !important;
            border-radius: 0.8rem !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 1rem;
            padding: 0.72rem;
            box-shadow: 0 12px 24px rgba(0,0,0,0.18);
        }

        section[data-testid="stSidebar"] .stLinkButton a {
            background: linear-gradient(135deg, #ffffff, #dbeafe) !important;
            color: #0f172a !important;
            border: 1px solid rgba(255,255,255,0.72) !important;
            border-radius: 0.9rem !important;
            font-weight: 900 !important;
            box-shadow: 0 10px 24px rgba(0,0,0,0.20);
        }

        section[data-testid="stSidebar"] .stLinkButton a p,
        section[data-testid="stSidebar"] .stLinkButton a span {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] .stLinkButton a:hover {
            background: linear-gradient(135deg, #fef3c7, #f9a8d4) !important;
            color: #111827 !important;
        }

        .author-card {
            padding: 1.05rem 0.95rem;
            border-radius: 1.25rem;
            background:
                linear-gradient(135deg, rgba(244, 114, 182, 0.96), rgba(59, 130, 246, 0.92), rgba(45, 212, 191, 0.88));
            box-shadow: 0 18px 45px rgba(0,0,0,0.28);
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255,255,255,0.32);
        }

        .author-name {
            color: white;
            font-size: 1.8rem;
            font-weight: 950;
            line-height: 1;
            letter-spacing: 0.04em;
        }

        .author-role {
            color: rgba(255,255,255,0.90);
            font-size: 0.86rem;
            margin-top: 0.45rem;
            font-weight: 700;
        }

        .hero {
            position: relative;
            padding: 1.7rem 1.85rem;
            border-radius: 1.45rem;
            background:
                linear-gradient(135deg, rgba(15,23,42,0.98), rgba(49,46,129,0.96), rgba(219,39,119,0.92), rgba(6,182,212,0.95));
            background-size: 240% 240%;
            animation: gradientShift 9s ease infinite, glowPulse 3.8s ease-in-out infinite;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 24px 58px rgba(49,46,129,0.28);
            border: 1px solid rgba(255,255,255,0.18);
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            right: -60px;
            top: -70px;
            background: rgba(255,255,255,0.16);
            border-radius: 999px;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.45rem;
            font-size: 2.45rem;
            font-weight: 950;
        }

        .hero p {
            color: rgba(255,255,255,0.91);
            font-size: 1.05rem;
            margin-bottom: 0;
            max-width: 850px;
        }

        .glass-card {
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 1.15rem;
            padding: 1rem;
            box-shadow: 0 16px 36px rgba(15,23,42,0.08);
            backdrop-filter: blur(10px);
        }

        .step-card {
            min-height: 112px;
            border-radius: 1rem;
            padding: 0.9rem;
            margin-bottom: 0.75rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.95), rgba(219,234,254,0.78));
            border: 1px solid rgba(147,197,253,0.45);
            box-shadow: 0 12px 26px rgba(30,41,59,0.08);
        }

        .airflow-card {
            position: relative;
            padding: 1rem;
            border-radius: 1.15rem;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(224,242,254,0.88));
            border: 1px solid rgba(14,165,233,0.35);
            box-shadow: 0 14px 32px rgba(14,165,233,0.12);
            overflow: hidden;
        }

        .airflow-card:before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 90% 10%, rgba(236,72,153,0.18), transparent 32%);
            pointer-events: none;
        }

        .flow-arrow {
            font-size: 1.35rem;
            font-weight: 900;
            color: #2563eb;
            text-align: center;
            margin: 0.15rem 0;
        }

        .step-num {
            display: inline-block;
            padding: 0.18rem 0.45rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #ec4899, #3b82f6);
            color: white;
            font-weight: 900;
            font-size: 0.75rem;
            margin-bottom: 0.3rem;
        }

        .badge {
            display: inline-block;
            padding: 0.34rem 0.68rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 900;
            margin: 0.10rem 0.08rem;
            box-shadow: 0 7px 17px rgba(15,23,42,0.10);
        }

        .success {background: linear-gradient(135deg, #bbf7d0, #34d399); color: #064e3b;}
        .warning {background: linear-gradient(135deg, #fde68a, #f59e0b); color: #78350f;}
        .danger {background: linear-gradient(135deg, #fecdd3, #fb7185); color: #881337;}
        .info {background: linear-gradient(135deg, #bfdbfe, #60a5fa); color: #172554;}

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.74);
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 1rem;
            padding: 0.9rem;
            box-shadow: 0 12px 26px rgba(15,23,42,0.07);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.62rem;
            font-weight: 900;
        }

        .muted {
            color: #64748b;
            font-size: 0.92rem;
        }

        .footer-note {
            color: #64748b;
            font-size: 0.86rem;
            padding-top: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="author-card">
            <div class="author-name">Rym FOUZARI</div>
            <div class="author-role">MLOps · Breast Cancer Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🧭 Pilotage")
    api_url = st.text_input("URL interne de l'API", value=API_URL)

    health = fetch_health(api_url)
    model_info = fetch_model_info(api_url)

    api_ok = health.get("status") == "ok"
    model_ok = bool(model_info.get("model_exists", False))

    if api_ok and model_ok:
        st.success("API disponible · modèle chargé")
    elif api_ok:
        st.warning("API disponible · modèle manquant")
    else:
        st.error("API indisponible")

    st.divider()

    col_api, col_model = st.columns(2)
    col_api.metric("API", "OK" if api_ok else "KO")
    col_model.metric("Modèle", "OK" if model_ok else "KO")
    st.metric("Variables", model_info.get("n_features", 30))

    st.divider()
    st.markdown("## 🔗 Liens utiles")
    st.link_button("MLflow Tracking", MLFLOW_EXTERNAL_URL, use_container_width=True)
    st.link_button("Airflow DAGs", AIRFLOW_EXTERNAL_URL, use_container_width=True)
    st.link_button("FastAPI Docs", API_DOCS_URL, use_container_width=True)
    st.link_button("Dépôt GitHub", GITHUB_URL, use_container_width=True)

    with st.expander("⚙️ Configuration technique"):
        st.code(
            f"""API_URL={api_url}
API_DOCS_URL={API_DOCS_URL}
MLFLOW_EXTERNAL_URL={MLFLOW_EXTERNAL_URL}
AIRFLOW_EXTERNAL_URL={AIRFLOW_EXTERNAL_URL}""",
            language="bash",
        )

st.markdown(
    """
    <div class="hero">
        <h1>🧬 Breast Cancer MLOps Dashboard</h1>
        <p>
            Dashboard complet : prédiction médicale, monitoring, MLflow,
            Airflow, FastAPI, Docker, CI/CD et déploiement Oracle Cloud.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_predict, tab_experiments, tab_model, tab_airflow, tab_architecture, tab_monitoring = st.tabs(
    ["🏠 Accueil", "🔎 Prédiction", "📊 Expériences", "🧠 Modèle", "🌬️ Airflow", "🏗️ Architecture", "📡 Monitoring"]
)

with tab_home:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Observations", "569")
    kpi2.metric("Variables", "30")
    kpi3.metric("ROC-AUC", "0.996")
    kpi4.metric("Déploiement", "Oracle")

    st.divider()
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Objectif")
        st.markdown(
            """
            Ce frontend présente une chaîne MLOps complète autour d'un modèle de
            classification binaire. Il sert de vitrine de projet : entraînement,
            tracking, optimisation, API, conteneurisation, CI/CD et déploiement cloud.
            """
        )
        st.markdown(
            """
            <span class="badge info">MLflow</span>
            <span class="badge info">Optuna</span>
            <span class="badge info">FastAPI</span>
            <span class="badge info">Streamlit</span>
            <span class="badge info">Airflow</span>
            <span class="badge info">Docker</span>
            <span class="badge info">Oracle</span>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Lecture métier")
        st.markdown(
            """
            - `0` : tumeur prédite comme **bénigne**
            - `1` : tumeur prédite comme **maligne**
            - la probabilité affichée correspond au risque estimé de malignité
            """
        )
        st.info("Projet pédagogique MLOps : ne pas utiliser comme outil médical réel.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Pipeline MLOps")

    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(MLOPS_STEPS):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="step-card">
                    <span class="step-num">{num}</span><br>
                    <b>{title}</b><br>
                    <span class="muted">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_predict:
    st.subheader("Prédiction individuelle")

    a, b, c = st.columns([1, 1, 2])
    with a:
        if st.button("Exemple standard", use_container_width=True):
            st.session_state["active_example"] = DEFAULT_FEATURES.copy()
            st.rerun()
    with b:
        if st.button("Exemple alternatif", use_container_width=True):
            st.session_state["active_example"] = ALT_EXAMPLE.copy()
            st.rerun()
    with c:
        st.caption("Les variables sont structurées en trois familles : moyenne, erreur standard et pire valeur.")

    st.divider()

    with st.form("predict_form"):
        values: dict[str, float] = {}
        input_cols = st.columns(3)

        for col, (group_name, features) in zip(input_cols, FEATURE_GROUPS.items(), strict=True):
            with col:
                st.markdown(f"### {group_name}")
                for feature in features:
                    values[feature] = st.number_input(
                        feature,
                        value=float(st.session_state["active_example"][feature]),
                        format="%.6f",
                    )

        submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

    if submitted:
        try:
            api_result = call_api("POST", "/predict", api_url, json={"features": values})
        except httpx.HTTPError as exc:
            st.error(f"Appel API impossible : {exc}")
            st.session_state["last_result"] = None
        else:
            add_history(api_result, values)

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        probability = float(result["probability_malignant"])
        prediction = int(result["prediction"])
        badge = risk_class(probability)

        st.divider()
        st.subheader("Résultat")

        r1, r2, r3, r4 = st.columns(4)
        if prediction == 1:
            r1.error("Prédiction : maligne")
        else:
            r1.success("Prédiction : bénigne")

        r2.metric("Probabilité malignité", f"{probability:.2%}")
        r2.progress(min(max(probability, 0.0), 1.0))

        r3.metric("Niveau", result["risk"])
        r3.markdown(f'<span class="badge {badge}">{result["risk"]}</span>', unsafe_allow_html=True)

        r4.metric("Classe", prediction)
        r4.metric("Heure", result["timestamp"].split(" ")[-1])

        if st.button("Effacer l'historique", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["last_result"] = None
            st.rerun()

    if st.session_state["history"]:
        st.divider()
        st.subheader("Historique")
        st.dataframe(history_df(st.session_state["history"]), use_container_width=True, hide_index=True)

with tab_experiments:
    st.subheader("Expériences MLflow")

    df_results = pd.DataFrame(MODEL_RESULTS)
    best_row = df_results.iloc[df_results["ROC-AUC"].idxmax()]

    e1, e2, e3 = st.columns(3)
    e1.metric("Meilleur modèle", best_row["Modèle"])
    e2.metric("Meilleur ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
    e3.metric("Approches", len(df_results))

    st.divider()
    table_col, chart_col = st.columns([1.1, 1])

    with table_col:
        st.markdown("### Synthèse")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

    with chart_col:
        st.markdown("### ROC-AUC")
        st.bar_chart(df_results.set_index("Modèle")[["ROC-AUC"]], use_container_width=True)

    st.divider()
    x1, x2, x3 = st.columns(3)
    x1.markdown("### Tracking\n- paramètres\n- métriques\n- artefacts")
    x2.markdown("### Optimisation\n- GridSearchCV\n- Optuna\n- validation croisée")
    x3.markdown("### Registry\n- modèle sauvegardé\n- artefact joblib\n- déploiement API")

    st.link_button("Ouvrir MLflow", MLFLOW_EXTERNAL_URL, use_container_width=True)

with tab_model:
    st.subheader("Modèle servi par l'API")

    if not model_info:
        st.error("Impossible de récupérer /model-info.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Modèle chargé", "Oui" if model_info.get("model_exists") else "Non")
        m2.metric("Features", model_info.get("n_features", 0))
        m3.metric("Cible", model_info.get("target", "inconnue"))

        st.divider()
        f1, f2 = st.columns([1, 1])
        with f1:
            st.markdown("### Variables")
            st.dataframe(
                pd.DataFrame({"feature": model_info.get("numeric_features", list(DEFAULT_FEATURES))}),
                use_container_width=True,
                hide_index=True,
            )
        with f2:
            st.markdown("### Réponse API")
            st.json(model_info)

with tab_airflow:
    st.subheader("Orchestration Apache Airflow")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("DAGs", "2")
    a2.metric("Training", "OK")
    a3.metric("Prediction", "OK")
    a4.metric("Port", "8080")

    st.divider()

    flow_left, flow_right = st.columns(2)

    with flow_left:
        st.markdown(
            """
            <div class="airflow-card">
                <h3>🧬 breast_training_pipeline</h3>
                <p class="muted">Pipeline d'entraînement automatisé.</p>
                <div class="badge info">validate_dataset</div>
                <div class="flow-arrow">↓</div>
                <div class="badge info">train_baseline</div>
                <div class="flow-arrow">↓</div>
                <div class="badge info">train_optuna</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with flow_right:
        st.markdown(
            """
            <div class="airflow-card">
                <h3>🚀 breast_predict_pipeline</h3>
                <p class="muted">Pipeline de validation de l'API d'inférence.</p>
                <div class="badge success">check_api_health</div>
                <div class="flow-arrow">↓</div>
                <div class="badge success">run_predict_client</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### DAGs configurés")
    st.dataframe(pd.DataFrame(AIRFLOW_DAGS), use_container_width=True, hide_index=True)

    st.markdown("### Accès Airflow")
    st.info(
        "L'historique détaillé des runs est disponible dans l'interface Airflow. "
        "L'intégration directe dans Streamlit demanderait d'appeler l'API REST Airflow avec authentification."
    )
    st.link_button("Ouvrir Airflow", AIRFLOW_EXTERNAL_URL, use_container_width=True)

    st.code(
        """# Commandes utiles
docker compose up -d airflow
docker compose logs airflow --tail=100
docker compose restart airflow""",
        language="bash",
    )


with tab_architecture:
    st.subheader("Architecture technique")

    arch = [
        {"Bloc": "Entraînement", "Fichier": "todo/mlproject/train.py", "Rôle": "baseline sklearn"},
        {"Bloc": "Comparaison", "Fichier": "todo/mlproject/train_models.py", "Rôle": "GridSearchCV"},
        {"Bloc": "Optimisation", "Fichier": "todo/mlproject/train_optuna.py", "Rôle": "Optuna + MLflow"},
        {"Bloc": "Tracking", "Fichier": "todo/mlproject/tracking.py", "Rôle": "MLflow centralisé"},
        {"Bloc": "API", "Fichier": "todo/mlproject/api.py", "Rôle": "FastAPI inference"},
        {"Bloc": "Frontend", "Fichier": "frontend/app.py", "Rôle": "dashboard Streamlit"},
        {"Bloc": "Airflow", "Fichier": "dags/", "Rôle": "orchestration des workflows"},
        {"Bloc": "Docker", "Fichier": "docker-compose.yml", "Rôle": "orchestration"},
        {"Bloc": "CI", "Fichier": ".github/workflows/ci.yml", "Rôle": "lint, types, tests"},
        {"Bloc": "CD", "Fichier": ".github/workflows/cd.yml", "Rôle": "GHCR"},
        {"Bloc": "Cloud", "Fichier": "Oracle Cloud", "Rôle": "déploiement public"},
    ]
    st.dataframe(pd.DataFrame(arch), use_container_width=True, hide_index=True)

    st.divider()
    d1, d2, d3 = st.columns(3)
    d1.code("docker compose up -d mlflow\ndocker compose --profile train run --rm train\ndocker compose up -d api frontend", language="bash")
    d2.code("GET  /health\nGET  /model-info\nPOST /predict", language="http")
    d3.code("MLflow    : 5000\nFastAPI   : 8000\nStreamlit : 8501", language="text")

with tab_monitoring:
    st.subheader("Monitoring de session")

    hist = st.session_state["history"]
    total = len(hist)
    malignant = sum(1 for item in hist if int(item["prediction"]) == 1)
    benign = total - malignant

    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Prédictions", total)
    z2.metric("Malignes", malignant)
    z3.metric("Bénignes", benign)
    z4.metric("API", "OK" if api_ok else "KO")

    st.divider()

    if hist:
        hdf = pd.DataFrame(hist)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### Distribution des classes")
            distribution = hdf["label"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
            st.bar_chart(distribution.set_index("Classe"), use_container_width=True)

        with g2:
            st.markdown("### Probabilité de malignité")
            probability_df = pd.DataFrame(
                {
                    "prediction": list(range(1, len(hist) + 1)),
                    "probability_malignant": [item["probability_malignant"] for item in reversed(hist)],
                }
            ).set_index("prediction")
            st.line_chart(probability_df, use_container_width=True)

        st.divider()
        st.dataframe(history_df(hist), use_container_width=True, hide_index=True)
    else:
        st.info("Lance une prédiction pour alimenter le monitoring.")

    st.divider()
    st.markdown("### État technique")
    st.json(
        {
            "health": health,
            "model_info": {
                "model_exists": model_info.get("model_exists"),
                "n_features": model_info.get("n_features"),
                "model_path": model_info.get("model_path"),
            },
            "public_links": {
                "api_docs": API_DOCS_URL,
                "mlflow": MLFLOW_EXTERNAL_URL,
                "airflow": AIRFLOW_EXTERNAL_URL,
                "github": GITHUB_URL,
            },
        }
    )

st.markdown(
    '<div class="footer-note">Dashboard Streamlit · MLflow · Airflow · FastAPI · Oracle Cloud · Rym FOUZARI</div>',
    unsafe_allow_html=True,
)
