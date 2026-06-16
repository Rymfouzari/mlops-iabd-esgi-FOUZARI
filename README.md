# Projet de classification binaire - Breast Cancer Wisconsin

## Objectif

L'objectif de ce projet est de développer un modèle de classification binaire capable de prédire si une tumeur est maligne ou bénigne à partir de mesures numériques extraites d'images médicales.

La classe positive (`1`) correspond à une tumeur maligne, tandis que la classe négative (`0`) correspond à une tumeur bénigne.

---

## Dataset

Le dataset utilisé est le **Breast Cancer Wisconsin Dataset**, disponible dans la bibliothèque scikit-learn.

### Caractéristiques principales

* 569 observations
* 30 variables numériques
* Classification binaire

### Variable cible

* 1 : tumeur maligne
* 0 : tumeur bénigne

---

## Justification du choix

Ce dataset a été choisi car il constitue un problème de classification binaire clair et bien documenté.

Les variables sont exclusivement numériques, ce qui simplifie la mise en place du pipeline de prétraitement et permet de se concentrer sur les différentes étapes du cycle de vie d'un projet MLOps :

* préparation des données ;
* entraînement ;
* suivi des expériences ;
* comparaison de modèles ;
* déploiement futur.

Le contexte médical rend également les résultats faciles à interpréter.

---

## TP0 - Configuration du projet

Les fichiers suivants ont été adaptés pour connecter le projet au dataset choisi :

### config.py

Définition :

* du chemin du dataset ;
* de la variable cible ;
* des variables numériques ;
* de l'expérience MLflow ;
* du nom du modèle.

### data.py

Chargement du dataset et séparation des données en ensembles d'entraînement et de test.

### features.py

Construction du pipeline de prétraitement :

* standardisation des variables numériques avec `StandardScaler` ;
* gestion des variables catégorielles via `OneHotEncoder`.

---

## Baseline - Régression Logistique

Une première baseline a été construite à l'aide d'une régression logistique.

### Résultats

| Métrique | Valeur |
| -------- | -----: |
| F1-score |  0.951 |
| ROC-AUC  |  0.996 |

### Analyse

Le modèle obtient d'excellentes performances sur le jeu de test.

Le ROC-AUC proche de 1 indique une très bonne capacité à distinguer les tumeurs malignes des tumeurs bénignes.

---

## TP5 - Suivi des expériences avec MLflow

Le projet a été instrumenté avec MLflow afin de :

* enregistrer les paramètres d'entraînement ;
* enregistrer les métriques ;
* sauvegarder les modèles ;
* suivre les expériences réalisées ;
* comparer plusieurs exécutions.

Un module dédié `tracking.py` centralise les fonctionnalités de suivi.

Les informations suivantes sont automatiquement enregistrées :

* paramètres ;
* métriques ;
* modèle entraîné ;
* matrice de confusion ;
* informations sur le dataset utilisé.

---

## Comparaison de modèles

Une comparaison de plusieurs modèles a été réalisée à l'aide de `GridSearchCV`.

### Modèles évalués

* Logistic Regression
* Random Forest
* Gradient Boosting

### Résultats

| Modèle              | F1-score | ROC-AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |    0.964 |   0.986 |
| Random Forest       |    0.963 |   0.993 |
| Gradient Boosting   |    0.950 |   0.995 |

### Analyse

Les trois modèles obtiennent des performances très élevées.

* Logistic Regression obtient le meilleur F1-score.
* Gradient Boosting obtient le meilleur ROC-AUC.
* Random Forest offre le meilleur compromis global entre les deux métriques.

Le dataset apparaît donc fortement prédictif et relativement facile à séparer.

---

## Tracking des expériences

Les expériences sont visualisées dans MLflow :

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

