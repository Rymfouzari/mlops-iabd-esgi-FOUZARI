# Projet MLOps – Classification du cancer du sein (Breast Cancer Wisconsin)

## Problématique

L'objectif du projet est de mettre en œuvre une chaîne MLOps complète sur un problème de classification binaire : prédire si une tumeur est maligne ou bénigne à partir de caractéristiques extraites d'images médicales.

Au-delà de la performance prédictive, le projet vise à appliquer les différentes briques étudiées durant le module :

* reproductibilité ;
* suivi des expérimentations ;
* comparaison de modèles ;
* optimisation automatique d'hyperparamètres ;
* exposition du modèle via une API ;
* intégration continue ;
* conteneurisation.

---

## Choix du dataset

Le dataset Breast Cancer Wisconsin a été retenu pour plusieurs raisons :

* problème de classification binaire clairement défini ;
* taille raisonnable permettant des expérimentations rapides ;
* variables exclusivement numériques ;
* forte utilisation dans la littérature, facilitant l'interprétation des résultats.

Le jeu de données contient :

* 569 observations ;
* 30 variables descriptives ;
* une cible binaire (maligne / bénigne).

Ce choix nous a permis de concentrer le travail sur les aspects MLOps plutôt que sur des problématiques complexes de nettoyage de données.

---

## Construction de la baseline

Une première baseline a été développée à l'aide d'une régression logistique intégrée dans un pipeline scikit-learn.

Le pipeline comprend :

* prétraitement automatique des variables ;
* standardisation des données ;
* entraînement du modèle ;
* sauvegarde du modèle entraîné.

### Résultats

| Métrique | Valeur |
| -------- | -----: |
| F1-score |  0.951 |
| ROC-AUC  |  0.996 |

### Analyse

La baseline atteint déjà des performances très élevées. Le ROC-AUC proche de 1 indique que les deux classes sont fortement séparables dans l'espace des caractéristiques.

Cette observation suggère que les gains obtenus par des modèles plus complexes risquent d'être limités.

---

## Mise en place du suivi des expériences

Le suivi des expériences a été réalisé avec MLflow.

Un module dédié (`tracking.py`) a été développé afin de centraliser :

* la configuration du serveur MLflow ;
* l'enregistrement des paramètres ;
* l'enregistrement des métriques ;
* le suivi des datasets ;
* l'enregistrement des modèles ;
* la sauvegarde des artefacts.

Cette centralisation permet de réutiliser les mêmes mécanismes de suivi dans tous les scripts d'entraînement.

### Informations enregistrées

Chaque run MLflow contient :

* les hyperparamètres du modèle ;
* les métriques d'évaluation ;
* les artefacts générés ;
* le modèle entraîné ;
* les informations de reproductibilité.

---

## Comparaison de plusieurs familles de modèles

Nous avons comparé plusieurs approches :

* Régression Logistique ;
* Random Forest ;
* XGBoost ;
* LightGBM.

Chaque modèle a été évalué à l'aide de GridSearchCV afin d'identifier les meilleurs hyperparamètres dans une grille prédéfinie.

Les résultats et paramètres optimaux sont automatiquement enregistrés dans MLflow.

### Observations

Les différentes familles de modèles obtiennent des performances très proches.

Les écarts observés restent faibles, ce qui confirme que le dataset est relativement facile à classifier.

Les méthodes basées sur les arbres obtiennent cependant des ROC-AUC légèrement supérieurs à la régression logistique.

---

## Optimisation automatique avec Optuna

Afin de dépasser l'approche Grid Search, nous avons mis en place une optimisation automatique des hyperparamètres avec Optuna.

Trois familles ont été optimisées :

* Random Forest ;
* XGBoost ;
* LightGBM.

Chaque étude Optuna est suivie dans MLflow :

* suivi des essais (trials) ;
* paramètres testés ;
* score ROC-AUC obtenu ;
* modèle final sélectionné.

L'organisation des runs MLflow permet de visualiser :

* l'étude globale ;
* chaque famille de modèles ;
* chaque essai Optuna individuellement.

### Résultats

Après optimisation, LightGBM a obtenu les meilleures performances globales.

### Conclusion

L'apport principal d'Optuna n'a pas été une amélioration spectaculaire du score, mais l'automatisation de la recherche d'hyperparamètres et la traçabilité complète des expérimentations.

---

## Interprétabilité avec SHAP

Afin d'améliorer la compréhension du comportement des modèles, des graphiques SHAP ont été intégrés au pipeline d'évaluation.

Les valeurs SHAP permettent :

* d'identifier les variables les plus influentes ;
* d'expliquer les prédictions du modèle ;
* de faciliter l'interprétation métier des résultats.

Les graphiques sont automatiquement enregistrés comme artefacts dans MLflow.

Cette étape améliore la transparence du modèle et facilite l'analyse des facteurs contribuant aux prédictions.

---

## Exposition du modèle via une API FastAPI

Une API REST a été développée avec FastAPI afin de rendre le modèle utilisable par des applications externes.

L'API charge automatiquement le modèle entraîné enregistré dans `models/model.joblib`.

### Endpoints disponibles

| Endpoint      | Méthode | Description                                     |
| ------------- | ------- | ----------------------------------------------- |
| `/health`     | GET     | Vérifie que l'API est opérationnelle            |
| `/model-info` | GET     | Affiche les informations du modèle chargé       |
| `/predict`    | POST    | Réalise une prédiction sur de nouvelles données |

### Fonctionnement

Le client envoie les caractéristiques d'une tumeur au format JSON.

L'API :

1. valide les données reçues ;
2. applique le pipeline de prétraitement ;
3. exécute la prédiction ;
4. renvoie la classe prédite ainsi que la probabilité associée.

Exemple de réponse :

```json
{
  "prediction": 1,
  "label": "malignant",
  "probability_malignant": 0.99999
}
```

Cette API constitue l'interface de déploiement du modèle et permet une utilisation en temps réel.

---

## Tests de l'API

Un client Python de test a été développé afin de valider automatiquement le bon fonctionnement de l'API.

Le script :

* vérifie l'état de l'API via `/health` ;
* envoie plusieurs observations du dataset à `/predict` ;
* récupère les informations du modèle via `/model-info`.

Cette étape permet de s'assurer que :

* le modèle est correctement chargé ;
* les prédictions sont produites sans erreur ;
* l'interface REST reste stable après chaque modification du projet.

---

## Intégration Continue (CI)

Un pipeline GitHub Actions a été mis en place afin d'automatiser les contrôles qualité du projet.

À chaque push ou pull request, le workflow exécute automatiquement :

* Ruff pour l'analyse statique du code ;
* MyPy pour la vérification des types ;
* Pytest pour l'exécution des tests.

Le pipeline comporte également une étape d'entraînement automatique qui :

* exécute la baseline ;
* génère le modèle ;
* publie le modèle comme artefact GitHub Actions.

Cette approche garantit que le projet reste fonctionnel après chaque modification du code.

---

## Conteneurisation avec Docker

L'application a été conteneurisée afin de garantir la reproductibilité de l'environnement d'exécution.

### Services Docker

Le projet s'appuie sur plusieurs conteneurs :

* MLflow pour le suivi des expérimentations ;
* un conteneur d'entraînement ;
* une API FastAPI pour l'inférence.

### Docker Compose

Docker Compose permet d'orchestrer l'ensemble des services :

* démarrage de MLflow ;
* entraînement du modèle ;
* partage du modèle entraîné via des volumes Docker ;
* exposition de l'API.

Cette architecture facilite la reproduction complète du projet sur une autre machine.

---

## Architecture du projet

```text
.
├── data/
├── models/
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.api
├── todo/
│   ├── mlproject/
│   │   ├── api.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── evaluation.py
│   │   ├── features.py
│   │   ├── tracking.py
│   │   ├── train.py
│   │   ├── train_models.py
│   │   └── train_optuna.py
│   └── scripts/
│       └── predict_client.py
├── tests/
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Conclusion

Le projet dispose désormais :

* d'une baseline reproductible ;
* d'un suivi MLflow complet ;
* d'une comparaison de modèles ;
* d'une optimisation automatique avec Optuna ;
* d'une interprétabilité via SHAP ;
* d'une API de prédiction ;
* de tests automatisés ;
* d'un pipeline CI GitHub Actions ;
* d'une conteneurisation Docker.

L'ensemble constitue une chaîne MLOps complète couvrant les principales étapes du cycle de vie d'un modèle de machine learning, depuis l'entraînement jusqu'à son exposition sous forme de service utilisable en production.
