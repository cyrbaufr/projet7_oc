# credit_heroku
Copyright Cyril Baudrillart

# Accéder à l'application

Cette application a été réalisée dans le cadre du projet 7 de la formation Data Scientist OpenClassrooms.
Elle a été déployée avec Heroku et est accessible à l'adresse suivante:
https://cyril-credit-scoring.herokuapp.com/

# Code source
Le code source de l'application est disponible sur github à l'adresse suivante:
https://github.com/cyrbaufr/credit_heroku

# Utiliser l'application
## Spécifications techniques
L'application a été réalisée avec la librairie Streamlit.
Pour les calculs des scores de crédit,elle l'application utilise une API réalisée avec FastAPI. Cette dernière est accessible à l'adresse suivante: https://test-cyril-fastapi.herokuapp.com/
## Manuel utilisateur
L'application comporte 4 pages différentes afin d'améliorer l'expérience utilisateur.
Son utilisation est destinée aux chargés de clientèle de la société Prêt à dépenser. Elle permet d'obtenir un score en fonction de paramètres saisis par le chargé de compte directement sur la plateforme.
### Page 'Accueil'
Explications concernant le contenu de l'application
### Page 'Comprendre les variables'
Brève description des informations sur le client à saisir par le chargé de compte dans la page 'Score crédit'
### Page Comprendre les scores
Outil d'analyse des variables du modèle. Permet aux chargés de compte de mieux comprendre le fonctionnement du modèle de scoring et surtout l'influence de chaque varible sur les résultats. POur plus de détail, merci de vous référer au manuel remis lors de la formation.
### Page Score crédit'
Page de saisie des informations sur le client
- Renseigner les 11 variables. Pas de saisie manuelle requise. Vous aurez juste à sélectionner la réponse parmi une liste ou bien utiliser le slider.
- Obligation RGPD: ajout d'une option téléchargement des données personnelles du client afin de l'informer des données saisies dans la plateforme en cas de demande.
- Calcul automatique du score de crédit en cliquant sur le bouton "Résultat score"
- Mise à jour de graphiques permettant au chargé de compte d'interpréter les résultats et d'expliquer la décision d'acceptation ou de refus de crédit à son client. Pour plus d'information, merci de vous reporter aux supports de formation remis lors de la séance de prise en main de l'outil.
### Page 'Comparables'
Accès à des statistiques permettant de donner des informations chifrées générales au client afin de l'aider à se situer par rapport aux autres clients de l'échantillon utilisé pour entraîner le modèle de scoring ou bien par rapport à un groupe d'individus comparable.
