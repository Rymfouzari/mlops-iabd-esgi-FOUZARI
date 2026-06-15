# Projet de classification binaire - Breast Cancer Wisconsin

## Objectif

L'objectif de ce projet est de développer un modèle de classification binaire capable de prédire si une tumeur est maligne ou bénigne à partir de mesures numériques extraites d'images médicales.

## Dataset

Le dataset utilisé est le Breast Cancer Wisconsin Dataset, disponible dans la bibliothèque scikit-learn.

Caractéristiques principales :

* 569 observations
* 30 variables numériques
* Classification binaire

Variable cible :

* 1 : tumeur maligne
* 0 : tumeur bénigne

## Justification du choix

Ce dataset a été choisi car il constitue un problème de classification binaire clair et bien documenté. Les variables sont exclusivement numériques, ce qui facilite la mise en œuvre du pipeline de machine learning tout en permettant d'obtenir des performances élevées.

## Résultats de la baseline

Après configuration du pipeline :

* F1-score : 0.951
* ROC-AUC : 0.996

Ces résultats montrent que le modèle est capable de distinguer efficacement les deux classes.

## Structure du projet

* `data/` : données d'entrée
* `pyproject.toml` : dépendances Python
* `uv.lock` : verrouillage des versions
* `Makefile` : automatisation des commandes du projet
