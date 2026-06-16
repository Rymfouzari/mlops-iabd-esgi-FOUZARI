# Projet MLOps – Classification du cancer du sein (Breast Cancer Wisconsin)

## Problématique

L'objectif du projet est de mettre en œuvre une chaîne MLOps complète sur un problème de classification binaire : prédire si une tumeur est maligne ou bénigne à partir de caractéristiques extraites d'images médicales.

Au-delà de la performance prédictive, le projet vise à appliquer les différentes briques étudiées durant le module :

* reproductibilité ;
* suivi des expérimentations ;
* comparaison de modèles ;
* optimisation automatique d'hyperparamètres ;
* gestion du cycle de vie des modèles.

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

---

## Comparaison de plusieurs familles de modèles

Nous avons comparé trois approches :

* Random Forest ;
* XGBoost ;
* LightGBM.

Chaque modèle a été évalué à l'aide de GridSearchCV afin d'identifier les meilleurs hyperparamètres dans une grille prédéfinie.

Les résultats et paramètres optimaux sont automatiquement enregistrés dans MLflow.

### Observations

Les trois familles de modèles obtiennent des performances très proches.

Les écarts observés sont faibles, ce qui confirme que le dataset est relativement facile à classifier.

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

Le projet dispose désormais :

* d'une baseline reproductible ;
* d'un suivi MLflow complet ;
* d'une comparaison de modèles ;
* d'une optimisation automatique ;
* d'un enregistrement des modèles dans le Model Registry.

Cette architecture constitue une base solide pour les prochaines étapes du projet (API, conteneurisation, déploiement et orchestration).
