from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Classification Breast Cancer Wisconsin")
st.write("Interface de prédiction pour estimer si une tumeur est bénigne ou maligne.")

st.sidebar.header("Configuration")
st.sidebar.write(f"API utilisée : `{API_URL}`")

if st.button("Vérifier l'API"):
    response = httpx.get(f"{API_URL}/health", timeout=10)
    st.success(response.json())

st.header("Données d'entrée")

features = {
    "mean radius": st.number_input("mean radius", value=17.99),
    "mean texture": st.number_input("mean texture", value=10.38),
    "mean perimeter": st.number_input("mean perimeter", value=122.8),
    "mean area": st.number_input("mean area", value=1001.0),
    "mean smoothness": st.number_input("mean smoothness", value=0.1184),
    "mean compactness": st.number_input("mean compactness", value=0.2776),
    "mean concavity": st.number_input("mean concavity", value=0.3001),
    "mean concave points": st.number_input("mean concave points", value=0.1471),
    "mean symmetry": st.number_input("mean symmetry", value=0.2419),
    "mean fractal dimension": st.number_input("mean fractal dimension", value=0.07871),
    "radius error": st.number_input("radius error", value=1.095),
    "texture error": st.number_input("texture error", value=0.9053),
    "perimeter error": st.number_input("perimeter error", value=8.589),
    "area error": st.number_input("area error", value=153.4),
    "smoothness error": st.number_input("smoothness error", value=0.006399),
    "compactness error": st.number_input("compactness error", value=0.04904),
    "concavity error": st.number_input("concavity error", value=0.05373),
    "concave points error": st.number_input("concave points error", value=0.01587),
    "symmetry error": st.number_input("symmetry error", value=0.03003),
    "fractal dimension error": st.number_input("fractal dimension error", value=0.006193),
    "worst radius": st.number_input("worst radius", value=25.38),
    "worst texture": st.number_input("worst texture", value=17.33),
    "worst perimeter": st.number_input("worst perimeter", value=184.6),
    "worst area": st.number_input("worst area", value=2019.0),
    "worst smoothness": st.number_input("worst smoothness", value=0.1622),
    "worst compactness": st.number_input("worst compactness", value=0.6656),
    "worst concavity": st.number_input("worst concavity", value=0.7119),
    "worst concave points": st.number_input("worst concave points", value=0.2654),
    "worst symmetry": st.number_input("worst symmetry", value=0.4601),
    "worst fractal dimension": st.number_input("worst fractal dimension", value=0.1189),
}

if st.button("Prédire"):
    response = httpx.post(
        f"{API_URL}/predict",
        json={"features": features},
        timeout=10,
    )

    if response.status_code == 200:
        result = response.json()
        st.subheader("Résultat")
        st.write(result)

        if result["prediction"] == 1:
            st.error(f"Tumeur prédite : maligne ({result['probability_malignant']:.3f})")
        else:
            st.success(f"Tumeur prédite : bénigne ({result['probability_malignant']:.3f})")
    else:
        st.error(response.json())