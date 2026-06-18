# Projet MLOps – Classification du cancer du sein (Breast Cancer Wisconsin)

## Présentation

Ce projet a pour objectif de mettre en œuvre une chaîne MLOps complète autour d'un problème de classification binaire : prédire si une tumeur est bénigne ou maligne à partir de caractéristiques extraites d'images médicales.

L'application couvre l'ensemble du cycle de vie d'un modèle de Machine Learning :

* préparation et validation des données ;
* entraînement et optimisation des modèles ;
* suivi des expérimentations avec MLflow ;
* interprétabilité avec SHAP ;
* exposition via une API FastAPI ;
* visualisation via une interface Streamlit ;
* orchestration avec Apache Airflow ;
* intégration continue avec GitHub Actions ;
* déploiement conteneurisé avec Docker ;
* hébergement sur Oracle Cloud Infrastructure.

---

# Architecture MLOps

```text
                           ┌──────────────┐
                           │   Airflow    │
                           │ Orchestration│
                           └──────┬───────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼

 ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
 │   Training     │    │    FastAPI     │    │   Streamlit    │
 │  & Optuna      │    │   Inference    │    │   Dashboard    │
 └───────┬────────┘    └───────┬────────┘    └────────────────┘
         │                     │
         ▼                     │
 ┌────────────────┐            │
 │    MLflow      │◄───────────┘
 │ Experiment     │
 │ Tracking       │
 └────────────────┘
```

L'ensemble des services est déployé via Docker Compose sur une machine virtuelle Oracle Cloud.

---

# Choix du dataset

Le dataset Breast Cancer Wisconsin a été retenu pour plusieurs raisons :

* problème de classification binaire clairement défini ;
* taille raisonnable permettant des expérimentations rapides ;
* variables exclusivement numériques ;
* forte utilisation dans la littérature scientifique.

Le dataset contient :

* 569 observations ;
* 30 variables descriptives ;
* une cible binaire (malignant / benign).

---

# Construction de la baseline

Une première baseline a été développée à l'aide d'une régression logistique intégrée dans un pipeline scikit-learn.

Le pipeline comprend :

* prétraitement automatique des variables ;
* standardisation des données ;
* entraînement du modèle ;
* sauvegarde du modèle entraîné.

## Résultats

| Métrique | Valeur |
| -------- | -----: |
| F1-score |  0.951 |
| ROC-AUC  |  0.996 |

La baseline atteint déjà d'excellentes performances, indiquant une bonne séparabilité des classes.

---

# Suivi des expériences avec MLflow

Le suivi des expérimentations a été réalisé avec MLflow.

Un module dédié centralise :

* la configuration du serveur MLflow ;
* le suivi des datasets ;
* l'enregistrement des paramètres ;
* l'enregistrement des métriques ;
* la sauvegarde des modèles ;
* la gestion des artefacts.

Chaque exécution enregistre :

* hyperparamètres ;
* métriques ;
* modèle entraîné ;
* artefacts générés ;
* informations de reproductibilité.

---

# Comparaison de modèles

Plusieurs familles de modèles ont été évaluées :

* Régression Logistique ;
* Random Forest ;
* XGBoost ;
* LightGBM.

Les recherches d'hyperparamètres ont été réalisées avec GridSearchCV.

Les résultats sont automatiquement enregistrés dans MLflow afin de comparer les performances et les paramètres optimaux.

---

# Optimisation automatique avec Optuna

Une étape d'optimisation automatique a été mise en place avec Optuna.

Les familles optimisées sont :

* Random Forest ;
* XGBoost ;
* LightGBM.

Chaque étude Optuna enregistre :

* les essais réalisés ;
* les paramètres testés ;
* les scores ROC-AUC obtenus ;
* le modèle sélectionné.

Cette approche permet une recherche plus efficace que les grilles classiques.

---

# Interprétabilité avec SHAP

Des visualisations SHAP ont été intégrées afin d'améliorer la compréhension des prédictions.

Les graphiques permettent :

* d'identifier les variables les plus importantes ;
* d'expliquer les décisions du modèle ;
* de faciliter l'interprétation métier.

Les figures sont automatiquement sauvegardées dans MLflow.

---

# API FastAPI

Une API REST a été développée avec FastAPI.

Le modèle est chargé automatiquement depuis :

```text
models/model.joblib
```

## Endpoints disponibles

| Endpoint    | Méthode | Description                  |
| ----------- | ------- | ---------------------------- |
| /health     | GET     | Vérification du service      |
| /model-info | GET     | Informations du modèle       |
| /predict    | POST    | Réalisation d'une prédiction |

Exemple de réponse :

```json
{
  "prediction": 1,
  "label": "malignant",
  "probability_malignant": 0.99999
}
```

---

# Dashboard Streamlit

Une interface Streamlit a été développée afin de :

* réaliser des prédictions interactives ;
* consulter les performances du modèle ;
* visualiser l'architecture MLOps ;
* accéder rapidement aux services du projet ;
* suivre l'état du système.

Le dashboard constitue l'interface utilisateur principale du projet.

---

# Orchestration avec Apache Airflow

Apache Airflow a été intégré afin d'automatiser les workflows MLOps.

## Pipeline d'entraînement

DAG :

```text
breast_training_pipeline
```

Workflow :

```text
validate_dataset
        ↓
train_baseline
        ↓
train_optuna
```

Ce pipeline :

* valide les données ;
* entraîne la baseline ;
* lance l'optimisation Optuna ;
* génère le modèle final.

## Pipeline de prédiction

DAG :

```text
breast_predict_pipeline
```

Workflow :

```text
check_api_health
        ↓
run_predict_client
```

Ce pipeline :

* vérifie la disponibilité de l'API ;
* envoie plusieurs requêtes de test ;
* valide le fonctionnement complet du service d'inférence.

---

# Tests automatisés

Le projet inclut :

* tests unitaires ;
* validation des pipelines ;
* vérification des endpoints FastAPI ;
* tests d'intégration.

Pytest est utilisé comme framework principal.

---

# Intégration Continue (CI)

GitHub Actions exécute automatiquement :

* Ruff ;
* MyPy ;
* Pytest ;
* génération du modèle ;
* publication des artefacts.

Chaque modification est ainsi validée automatiquement.

---

# Conteneurisation Docker

Le projet repose sur plusieurs services Docker :

* MLflow ;
* API FastAPI ;
* Dashboard Streamlit ;
* Apache Airflow ;
* conteneur d'entraînement.

Docker Compose orchestre l'ensemble des composants.

---

# Déploiement Oracle Cloud

L'application est déployée sur une machine virtuelle Oracle Cloud.

## Services exposés

| Service           | URL                          |
| ----------------- | ---------------------------- |
| Streamlit         | http://88.96.61.63:8501      |
| FastAPI           | http://88.96.61.63:8000      |
| Documentation API | http://88.96.61.63:8000/docs |
| MLflow            | http://88.96.61.63:5000      |
| Airflow           | http://88.96.61.63:8080      |

---

# Structure du projet

```text
.
├── dags/
├── data/
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.frontend
│   ├── Dockerfile.train
│   └── Dockerfile.airflow
├── frontend/
├── models/
├── tests/
├── todo/
│   ├── mlproject/
│   └── scripts/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

# Conclusion

Le projet implémente une chaîne MLOps complète couvrant :

* la préparation des données ;
* l'entraînement de modèles ;
* l'optimisation d'hyperparamètres ;
* l'interprétabilité ;
* le suivi d'expériences ;
* le déploiement d'une API ;
* la supervision via Streamlit ;
* l'orchestration via Airflow ;
* la conteneurisation Docker ;
* le déploiement Cloud.

Il constitue une plateforme de démonstration complète des bonnes pratiques MLOps appliquées à un cas réel de classification médicale.
