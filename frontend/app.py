"""Frontend Streamlit pour l'API Breast Cancer Classification.

Dashboard MLOps complet :
- Prediction via API FastAPI
- Statut API / modele
- Resume des experiences MLflow
- Architecture projet
- Monitoring local de session
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
GITHUB_URL = os.environ.get(
    "GITHUB_URL",
    "https://github.com/Rymfouzari/mlops-iabd-esgi-FOUZARI",
)

DEFAULT_FEATURES: dict[str, float] = {
    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8,
    "mean area": 1001.0, "mean smoothness": 0.1184, "mean compactness": 0.2776,
    "mean concavity": 0.3001, "mean concave points": 0.1471,
    "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
    "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589,
    "area error": 153.4, "smoothness error": 0.006399,
    "compactness error": 0.04904, "concavity error": 0.05373,
    "concave points error": 0.01587, "symmetry error": 0.03003,
    "fractal dimension error": 0.006193, "worst radius": 25.38,
    "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
    "worst smoothness": 0.1622, "worst compactness": 0.6656,
    "worst concavity": 0.7119, "worst concave points": 0.2654,
    "worst symmetry": 0.4601, "worst fractal dimension": 0.1189,
}

ALT_EXAMPLE: dict[str, float] = {
    "mean radius": 11.42, "mean texture": 20.38, "mean perimeter": 77.58,
    "mean area": 386.1, "mean smoothness": 0.1425, "mean compactness": 0.2839,
    "mean concavity": 0.2414, "mean concave points": 0.1052,
    "mean symmetry": 0.2597, "mean fractal dimension": 0.09744,
    "radius error": 0.4956, "texture error": 1.156, "perimeter error": 3.445,
    "area error": 27.23, "smoothness error": 0.00911,
    "compactness error": 0.07458, "concavity error": 0.05661,
    "concave points error": 0.01867, "symmetry error": 0.05963,
    "fractal dimension error": 0.009208, "worst radius": 14.91,
    "worst texture": 26.5, "worst perimeter": 98.87, "worst area": 567.7,
    "worst smoothness": 0.2098, "worst compactness": 0.8663,
    "worst concavity": 0.6869, "worst concave points": 0.2575,
    "worst symmetry": 0.6638, "worst fractal dimension": 0.173,
}

MODEL_RESULTS = [
    {"Modele": "Baseline Logistic Regression", "F1": 0.951, "ROC-AUC": 0.996, "Approche": "Baseline"},
    {"Modele": "RandomForest GridSearchCV", "F1": 0.963, "ROC-AUC": 0.993, "Approche": "Grid Search"},
    {"Modele": "XGBoost GridSearchCV", "F1": 0.950, "ROC-AUC": 0.995, "Approche": "Grid Search"},
    {"Modele": "LightGBM Optuna", "F1": None, "ROC-AUC": 0.995, "Approche": "Optuna"},
]

FEATURE_GROUPS = {
    "Mesures moyennes": list(DEFAULT_FEATURES.keys())[:10],
    "Erreurs standards": list(DEFAULT_FEATURES.keys())[10:20],
    "Pires valeurs observées": list(DEFAULT_FEATURES.keys())[20:],
}

MLOPS_STEPS = [
    ("Dataset", "Breast Cancer Wisconsin", "data/breast_cancer.csv"),
    ("Baseline", "LogisticRegression + pipeline sklearn", "train.py"),
    ("Tracking", "Paramètres, métriques, artefacts", "tracking.py"),
    ("Comparaison", "RF / XGBoost / LightGBM", "train_models.py"),
    ("Optimisation", "Recherche d'hyperparamètres", "train_optuna.py"),
    ("Serving", "Inference REST", "api.py"),
    ("Frontend", "Dashboard Streamlit", "frontend/app.py"),
    ("Docker", "MLflow + API + Frontend", "docker-compose.yml"),
    ("CI/CD", "Tests + build + GHCR", ".github/workflows/"),
    ("Cloud", "Déploiement Oracle", "OCI Compute"),
]


def call_api(method: str, path: str, api_url: str, **kwargs: Any) -> dict[str, Any]:
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


def label_from_prediction(prediction: int) -> str:
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
        "display_label": label_from_prediction(prediction),
        "probability_malignant": probability,
        "risk_band": risk_band(probability),
        "feedback": None,
        "features": features.copy(),
    }
    st.session_state["last_result"] = record
    st.session_state["history"].insert(0, record)


def history_table(history: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": item["timestamp"],
                "Prédiction": item["display_label"],
                "Probabilité malignité": f"{item['probability_malignant']:.2%}",
                "Niveau": item["risk_band"],
                "Feedback": item.get("feedback") or "-",
            }
            for item in history
        ]
    )


st.set_page_config(
    page_title="Breast Cancer MLOps Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(236, 72, 153, 0.14), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(14, 165, 233, 0.16), transparent 30%),
                radial-gradient(circle at 45% 95%, rgba(168, 85, 247, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #111827 0%, #312e81 42%, #0f766e 100%);
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            overflow-y: auto;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 1rem;
            padding: 0.65rem;
        }

        section[data-testid="stSidebar"] input {
            color: #0f172a !important;
            background: #ffffff !important;
            border-radius: 0.7rem !important;
        }

        section[data-testid="stSidebar"] .stLinkButton a {
            background: linear-gradient(135deg, #ffffff, #e0f2fe) !important;
            color: #0f172a !important;
            border: 1px solid rgba(255, 255, 255, 0.65) !important;
            border-radius: 0.85rem !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }

        section[data-testid="stSidebar"] .stLinkButton a p,
        section[data-testid="stSidebar"] .stLinkButton a span {
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] .stLinkButton a:hover {
            background: linear-gradient(135deg, #fef3c7, #f9a8d4) !important;
            color: #111827 !important;
            border-color: rgba(255, 255, 255, 0.9) !important;
        }

        .sidebar-small {
            font-size: 0.78rem;
            line-height: 1.25;
            color: rgba(248,250,252,0.82);
        }

        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

        .author-card {
            padding: 0.95rem 0.9rem;
            border-radius: 1.2rem;
            background: linear-gradient(135deg, rgba(244, 114, 182, 0.95), rgba(14, 165, 233, 0.95));
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.28);
        }

        .author-name {
            color: white;
            font-size: 1.65rem;
            font-weight: 900;
            letter-spacing: 0.03em;
            line-height: 1.05;
        }

        .author-role {
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.88rem;
            margin-top: 0.4rem;
            font-weight: 600;
        }

        .hero {
            padding: 1.55rem 1.75rem; border-radius: 1.35rem;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.97) 0%, rgba(49, 46, 129, 0.96) 38%, rgba(219, 39, 119, 0.90) 72%, rgba(6, 182, 212, 0.92) 100%);
            color: white; margin-bottom: 1.2rem;
            box-shadow: 0 20px 50px rgba(49, 46, 129, 0.26);
            border: 1px solid rgba(255,255,255,0.16);
        }
        .hero h1 {color: white; margin-bottom: 0.35rem; font-size: 2.25rem;}
        .hero p {color: rgba(255,255,255,0.90); font-size: 1.04rem; margin-bottom: 0;}

        .step-card {
            border-left: 5px solid #ec4899; border-radius: 0.85rem;
            padding: 0.82rem 0.95rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(239,246,255,0.78));
            margin-bottom: 0.70rem;
            box-shadow: 0 10px 22px rgba(30, 41, 59, 0.07);
        }

        .badge {
            display: inline-block; padding: 0.34rem 0.68rem; border-radius: 999px;
            font-size: 0.82rem; font-weight: 800; margin: 0.12rem 0.08rem;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .success {background: linear-gradient(135deg, #bbf7d0, #34d399); color: #064e3b;}
        .warning {background: linear-gradient(135deg, #fde68a, #f59e0b); color: #78350f;}
        .danger {background: linear-gradient(135deg, #fecdd3, #fb7185); color: #881337;}
        .info {background: linear-gradient(135deg, #bfdbfe, #60a5fa); color: #172554;}

        .muted {color: #64748b; font-size: 0.92rem;}
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 1rem;
            padding: 0.9rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricValue"] {font-size: 1.58rem;}
        .footer-note {color: #64748b; font-size: 0.86rem; padding-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="author-card">
            <div class="author-name">Rym FOUZARI</div>
            <div class="author-role">MLOps Engineer · Breast Cancer Classification</div>
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
    col_a, col_b = st.columns(2)
    col_a.metric("API", "OK" if api_ok else "KO")
    col_b.metric("Modèle", "OK" if model_ok else "KO")
    st.metric("Variables", model_info.get("n_features", 30))

    st.divider()
    st.markdown("## 🔗 Liens utiles")
    st.link_button("MLflow Tracking", MLFLOW_EXTERNAL_URL, use_container_width=True)
    st.link_button("FastAPI Docs", API_DOCS_URL, use_container_width=True)
    st.link_button("Dépôt GitHub", GITHUB_URL, use_container_width=True)

    st.divider()
    with st.expander("⚙️ Configuration technique"):
        st.markdown(
            f"""
            <div class="sidebar-small">
            <b>API interne</b><br>{api_url}<br><br>
            <b>Docs API publiques</b><br>{API_DOCS_URL}<br><br>
            <b>MLflow public</b><br>{MLFLOW_EXTERNAL_URL}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="hero">
        <h1>🧬 Breast Cancer MLOps Dashboard</h1>
        <p>
            Classification binaire, tracking MLflow, API FastAPI, frontend Streamlit,
            Docker Compose, CI/CD GitHub Actions et déploiement Oracle Cloud.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_home, tab_predict, tab_experiments, tab_model, tab_architecture, tab_monitoring = st.tabs(
    ["🏠 Accueil", "🔎 Prédiction", "📊 Expériences", "🧠 Modèle", "🏗️ Architecture", "📡 Monitoring"]
)

with tab_home:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Observations", "569")
    kpi2.metric("Variables", "30")
    kpi3.metric("ROC-AUC baseline", "0.996")
    kpi4.metric("Stack", "MLOps")

    st.divider()
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Objectif du projet")
        st.markdown(
            """
            Ce dashboard sert de vitrine finale du TP : il ne fait pas seulement une
            prédiction, il montre aussi la chaîne MLOps qui permet d'industrialiser
            un modèle de machine learning.
            """
        )
        st.markdown(
            """
            <span class="badge info">Dataset</span>
            <span class="badge info">MLflow</span>
            <span class="badge info">Optuna</span>
            <span class="badge info">FastAPI</span>
            <span class="badge info">Docker</span>
            <span class="badge info">Oracle Cloud</span>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Lecture métier")
        st.markdown(
            """
            - `0` : tumeur prédite comme **bénigne**
            - `1` : tumeur prédite comme **maligne**
            - la probabilité affichée correspond au risque estimé de malignité
            """
        )
        st.info("Cas d'étude MLOps : ne pas utiliser comme outil médical réel.")

    st.divider()
    st.subheader("Pipeline MLOps")

    cols = st.columns(5)
    for index, (step, description, file_name) in enumerate(MLOPS_STEPS):
        with cols[index % 5]:
            st.markdown(
                f"""
                <div class="step-card">
                    <b>{index + 1}. {step}</b><br>
                    <span class="muted">{description}</span><br>
                    <code>{file_name}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_predict:
    st.subheader("Prédiction individuelle")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
    with action_col1:
        if st.button("Charger exemple standard", use_container_width=True):
            st.session_state["active_example"] = DEFAULT_FEATURES.copy()
            st.rerun()
    with action_col2:
        if st.button("Charger exemple alternatif", use_container_width=True):
            st.session_state["active_example"] = ALT_EXAMPLE.copy()
            st.rerun()
    with action_col3:
        st.caption("Les champs sont groupés selon la structure du dataset.")

    st.divider()

    with st.form("predict_form"):
        feature_values: dict[str, float] = {}
        form_cols = st.columns(3)

        for col, (group_name, group_features) in zip(form_cols, FEATURE_GROUPS.items(), strict=True):
            with col:
                st.markdown(f"### {group_name}")
                for feature_name in group_features:
                    feature_values[feature_name] = st.number_input(
                        feature_name,
                        value=float(st.session_state["active_example"][feature_name]),
                        format="%.6f",
                    )

        submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

    if submitted:
        try:
            result = call_api("POST", "/predict", api_url, json={"features": feature_values})
        except httpx.HTTPError as exc:
            st.error(f"Appel API impossible : {exc}")
            st.session_state["last_result"] = None
        else:
            add_history(result, feature_values)

    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        probability = float(result["probability_malignant"])
        prediction = int(result["prediction"])
        badge = risk_class(probability)

        st.divider()
        st.subheader("Résultat")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        with res_col1:
            if prediction == 1:
                st.error("Prédiction : maligne")
            else:
                st.success("Prédiction : bénigne")

        with res_col2:
            st.metric("Probabilité malignité", f"{probability:.2%}")
            st.progress(min(max(probability, 0.0), 1.0))

        with res_col3:
            st.metric("Niveau de risque", result["risk_band"])
            st.markdown(
                f'<span class="badge {badge}">{result["risk_band"]}</span>',
                unsafe_allow_html=True,
            )

        with res_col4:
            st.metric("Heure", result["timestamp"].split(" ")[-1])
            st.metric("Classe", prediction)

        st.markdown("### Feedback utilisateur")
        fb1, fb2, fb3 = st.columns(3)
        if fb1.button("Prédiction correcte", use_container_width=True):
            st.session_state["history"][0]["feedback"] = "correct"
            st.success("Feedback enregistré.")
        if fb2.button("Prédiction incorrecte", use_container_width=True):
            st.session_state["history"][0]["feedback"] = "incorrect"
            st.warning("Feedback enregistré.")
        if fb3.button("Effacer l'historique", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["last_result"] = None
            st.rerun()

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.subheader("Historique de session")
        st.dataframe(history_table(history), use_container_width=True, hide_index=True)

with tab_experiments:
    st.subheader("Comparaison des expérimentations")

    df_results = pd.DataFrame(MODEL_RESULTS)
    best_idx = df_results["ROC-AUC"].idxmax()
    best_model = df_results.loc[best_idx, "Modele"]
    best_auc = df_results.loc[best_idx, "ROC-AUC"]

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    exp_col1.metric("Meilleur modèle", best_model)
    exp_col2.metric("Meilleur ROC-AUC", f"{best_auc:.3f}")
    exp_col3.metric("Nombre d'approches", len(df_results))

    st.divider()
    table_col, chart_col = st.columns([1.1, 1])

    with table_col:
        st.markdown("### Tableau de synthèse")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

    with chart_col:
        st.markdown("### ROC-AUC par modèle")
        st.bar_chart(df_results.set_index("Modele")[["ROC-AUC"]], use_container_width=True)

    st.divider()
    log1, log2, log3 = st.columns(3)
    with log1:
        st.markdown("#### Tracking")
        st.markdown("- paramètres\n- métriques\n- artefacts\n- dataset")
    with log2:
        st.markdown("#### Optimisation")
        st.markdown("- GridSearchCV\n- Optuna\n- validation croisée\n- comparaison")
    with log3:
        st.markdown("#### Registry")
        st.markdown("- modèle sauvegardé\n- artefact joblib\n- versions\n- déploiement API")

    st.link_button("Ouvrir MLflow", MLFLOW_EXTERNAL_URL, use_container_width=True)

with tab_model:
    st.subheader("Informations du modèle servi par l'API")

    if not model_info:
        st.error("Impossible de récupérer /model-info depuis l'API.")
    else:
        mi1, mi2, mi3 = st.columns(3)
        mi1.metric("Modèle disponible", "Oui" if model_info.get("model_exists") else "Non")
        mi2.metric("Nombre de features", model_info.get("n_features", 0))
        mi3.metric("Cible", model_info.get("target", "inconnue"))

        st.divider()
        fcol1, fcol2 = st.columns([1, 1])
        with fcol1:
            st.markdown("### Features numériques")
            st.dataframe(
                pd.DataFrame({"feature": model_info.get("numeric_features", list(DEFAULT_FEATURES))}),
                use_container_width=True,
                hide_index=True,
            )
        with fcol2:
            st.markdown("### Réponse brute /model-info")
            st.json(model_info)

with tab_architecture:
    st.subheader("Architecture technique")

    arch_rows = [
        {"Bloc": "Entraînement", "Fichier": "todo/mlproject/train.py", "Rôle": "baseline Logistic Regression"},
        {"Bloc": "Comparaison", "Fichier": "todo/mlproject/train_models.py", "Rôle": "GridSearchCV multi-modèles"},
        {"Bloc": "Optimisation", "Fichier": "todo/mlproject/train_optuna.py", "Rôle": "Optuna + MLflow"},
        {"Bloc": "Tracking", "Fichier": "todo/mlproject/tracking.py", "Rôle": "centralisation MLflow"},
        {"Bloc": "API", "Fichier": "todo/mlproject/api.py", "Rôle": "inférence FastAPI"},
        {"Bloc": "Frontend", "Fichier": "frontend/app.py", "Rôle": "dashboard Streamlit"},
        {"Bloc": "Docker", "Fichier": "docker-compose.yml", "Rôle": "orchestration services"},
        {"Bloc": "CI", "Fichier": ".github/workflows/ci.yml", "Rôle": "lint, types, tests, train"},
        {"Bloc": "CD", "Fichier": ".github/workflows/cd.yml", "Rôle": "build/push image API GHCR"},
        {"Bloc": "Cloud", "Fichier": "Oracle Cloud Instance", "Rôle": "déploiement public"},
    ]

    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

    st.divider()
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("### Docker")
        st.code(
            "docker compose up -d mlflow\n"
            "docker compose --profile train run --rm train\n"
            "docker compose up -d api frontend",
            language="bash",
        )
    with d2:
        st.markdown("### API")
        st.code("GET  /health\nGET  /model-info\nPOST /predict", language="http")
    with d3:
        st.markdown("### Ports")
        st.code("MLflow    : 5000\nFastAPI   : 8000\nStreamlit : 8501", language="text")

with tab_monitoring:
    st.subheader("Monitoring de session")

    current_history = st.session_state.get("history", [])
    total = len(current_history)
    malignant = sum(1 for item in current_history if int(item["prediction"]) == 1)
    benign = total - malignant

    mon1, mon2, mon3, mon4 = st.columns(4)
    mon1.metric("Prédictions", total)
    mon2.metric("Malignes", malignant)
    mon3.metric("Bénignes", benign)
    mon4.metric("API", "OK" if api_ok else "KO")

    st.divider()

    if current_history:
        df_history = pd.DataFrame(current_history)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Distribution des classes")
            distribution = (
                df_history["display_label"]
                .value_counts()
                .rename_axis("Classe")
                .reset_index(name="Nombre")
            )
            st.bar_chart(distribution.set_index("Classe"), use_container_width=True)

        with chart_col2:
            st.markdown("### Probabilité de malignité")
            probability_df = pd.DataFrame(
                {
                    "prediction": list(range(1, len(current_history) + 1)),
                    "probability_malignant": [
                        item["probability_malignant"] for item in reversed(current_history)
                    ],
                }
            ).set_index("prediction")
            st.line_chart(probability_df, use_container_width=True)

        st.divider()
        st.markdown("### Données de monitoring")
        st.dataframe(history_table(current_history), use_container_width=True, hide_index=True)
    else:
        st.info("Aucune prédiction dans cette session. Lance une prédiction pour alimenter le monitoring.")

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
                "github": GITHUB_URL,
            },
        }
    )

st.markdown(
    '<div class="footer-note">Dashboard Streamlit du projet MLOps Breast Cancer Wisconsin.</div>',
    unsafe_allow_html=True,
)
