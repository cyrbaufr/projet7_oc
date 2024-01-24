# projet7_oc
# Formation Data Scientist Openclassrooms - Projet 7

Copyright Cyril Baudrillart

- Mise au point d'un système de scoring de crédit automatisé grâce au Machine learning
- Création d'un dashboard sous Streamlit
- Mise au point d'une API utilisant FastAPI

# Contenu du repository
Ce répertoire contient l'intégralité du code réalisé dans le cadre du projet 7 de la formation Data Scientist OpenClassrooms.  

Deux répertoires annexes sont rattachés au projet. Ils permettent de gérer le déploiement des applications API & Dashboard sur Heroku.  

Code source API (correspondant à la phase 4 de ce repo):  
https://github.com/cyrbaufr/fastapi  
API déployée disponible ici: 
https://test-cyril-fastapi.herokuapp.com/

Code source Dashboard (correspondant à la phase 5 de ce repo):  
https://github.com/cyrbaufr/credit_heroku  
Application déployée disponible ici: 
https://cyril-credit-scoring.herokuapp.com/  

# Structuration du code source
Le code a été structuré en 5 phases correspondant aux différentes étapes du projet.  
## Phase 1: preprocessing  
Il s'agit de l'ensemble des fichiers utilisés pour nettoyer les données et obtenir les datasets utilisés pour l'entraînement et la validation des modèles.  
## Phase 2: modélisation  
Code source des procédures de test et d'optimisation mises en place afin d'obtenir le modèle de prévision final.  
## Phase 3: interprétabilité
Notebook utilisé pour mieux comprendre le fonctionnement du modèle final retenu et identifier l'impact de chaque feature retenue sur les résultats. Contient à la fois de l'interprétabilité globale et locale.
## Phase 4: API
Ensemble des fichiers utilisés par la construction de l'API avec FastAPI.
## Phase 5: Dashboard
Ensemble des fichiers utilisés pour la construction du dashboard avec Streamlit.

# Données utilisées pour le projet  
Compte tenu de la grande taille des fichiers de données générés dans le cadre du projet, il est impossible de les mettre à disposition des utilisateurs sous github. Les fichiers originaux de données sont disponibles ici:  
https://www.kaggle.com/c/home-credit-default-risk/data
