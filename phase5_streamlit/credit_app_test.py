"""
Version locale du dashboard pour débogage

Auteur: Cyril Baudrillart

Pour exécuter le code en local:
streamlit run credit_app_test.py

Useful resources:
https://github.com/RihabFekii/streamlit-app/blob/master/frontend/app.py
https://rihab-feki.medium.com/deploying-machine-learning-models-with-streamlit-fastapi-and-docker-bb16bbf8eb91
https://github.com/mullzhang/app-streamlit-fastapi/blob/main/frontend/utils.py
https://towardsdatascience.com/how-you-can-quickly-deploy-your-ml-models-with-fastapi-9428085a87bf
https://www.analyticsvidhya.com/blog/2021/06/deploying-ml-models-as-api-using-fastapi-and-heroku/

"""

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import base64
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'Agg' )
import seaborn as sns
import requests
import json
import shap
from lime import lime_tabular

# use full page length
st.set_page_config(layout="wide")

# ***************** Fonctions used in dashboard ****************

def page_title(my_title="Application de scoring de crédits"):
    """
    Define header of the dashboard pages with logo
    Just enter texte as argument
    """
    # Add logo on the right & title on the left
    image = Image.open('credit_logo.png')
    col1, col2 = st.columns([25,6])
    # Put title in first column
    with col1:
        # center title with logo
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("# ", my_title)
    # Insert logo in second column (on the right)
    with col2:
        st.image(image, use_column_width=False, width=250)

    # Add line below header
    st.write(" *** ")

@st.cache
def get_data(csv_name):
    """ Load data from csv file & put in cache 
    """
    full_url = url + csv_name
    return pd.read_csv(full_url, index_col=0)


@st.cache
def get_pickle(pickle_name):
    """
    Load data from pickle file & put in cache 
    https://stackoverflow.com/questions/48770542/what-is-the-difference-between-save-a-pandas-dataframe-to-pickle-and-to-csv
    """
    full_url = url + pickle_name
    return pd.read_pickle(full_url)


def load_data(df_name):
    """
    Load dataframe
    Remplacer csv par pickle
    """
    my_bar = st.progress(0)
    for pct_complete in range (100):
        time.sleep(0.1)
        my_bar.progress(pct_complete +1)
        if df_name=='X':
            X = get_data('X_train_sample.csv')
        elif df_name=='X_train':
            X = get_data('X_rus_train_sample.csv')
        elif df_name=='X_est':  
            X = get_data('X_rus_test_sample.csv')
    return(X)


def load_pickle_model():
    file = open("classifier_xgb_best.pkl",'rb')
    xgbclassifier = pickle.load(file)
    file.close()
    return xgbclassifier


def load_pickle_shapley():
    """
    Shap Values have been already preprocessed
    """
    file = open("shap_values_test_set.pkl",'rb')
    shap_values = pickle.load(file)
    file.close()
    return shap_values


def load_pickle_obj():
    file = open("shap_obj.pkl",'rb')
    shap_values = pickle.load(file)
    file.close()
    return shap_values


def get_api_response():
    """
    Appelle l'API pour faire une prévision du score de crédit
    retourne un dictionnaire de type
        {"prediction":"Crédit accepté"}
        {"prediction":"Crédit refusé"}
    """
    # fastapi endpoint
    url = 'https://test-cyril-fastapi.herokuapp.com'
    endpoint = '/predict'

    full_url = url + endpoint
    credit = json.dumps({
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "PREV_DAYS_DECISION_MIN": PREV_DAYS_DECISION_MIN,
        "CODE_GENDER": CODE_GENDER,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "PREV_APP_CREDIT_PERC_MIN": PREV_APP_CREDIT_PERC_MIN,
        "INSTAL_DPD_MAX": INSTAL_DPD_MAX,
        "AMT_CREDIT": AMT_CREDIT,
        "DAYS_BIRTH": DAYS_BIRTH,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "NAME_EDUCATION_TYPE_Higher_education": NAME_EDUCATION_TYPE_Higher_education
        })
    headers={'accept': 'application/json',
             'Content-Type': 'application/json'}
    response = requests.request("POST", full_url, headers=headers, data=credit)
    return(response.text)


# ############### LOAD DATA ###############

#https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas
# url = 'https://media.githubusercontent.com/media/cyrbaufr/credit_heroku/master/'
# full_url = url+'X_train_imputed.csv'
url=''
# Voir fichier app_preprocessing.ipynb
# csv_name = 'X_train_raw.csv'

def filedownload(df):
    """ Function to download csv file created a Dataframe df
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="credit.csv">Download CSV File</a>'
    return href

# ***************** MENU **********************

# Create a page dropdown

page = st.sidebar.selectbox("Choisissez votre page",
    ["Accueil", "Comprendre les variables", "Comprendre les scores", "Score crédit", "Comparables"])

# ***************************** Page "Accueil" **********************************

if page == "Accueil":
    page_title("Bienvenue sur l'application de scoring crédit")
    st.header("Comment utiliser ce dashboard")
    st.write("Pour naviguer dans l'application, utilisez le menu de gauche")
    st.write('4 pages sont disponbles:')
    st.subheader("Comprendre les variables")
    st.write("Description des 11 variables utilisées par le modèle de scoring.")
    st.subheader("Comprendre les scores")
    st.write("Cette page permet de mieux comprendre le fonctionnement \
        du modèle de scoring. Nous utilisons une analyse de l'interprétabilité \
        selon les valeurs Shapley, une méthode d'analyse moderne et performante \
        permettant de mieux comprendre l'impact de chaque variable sur la prévision.")
    st.write("")
    st.subheader("Score crédit")
    st.write("Page de saisie des données clients. Une fois les informations renseignées, \
        vous avez juste à cliquer sur un bouton et le résultat du scoring sera\
        retourné automatiquement. Vous saurez en quelques secondes si le crédit est accordé \
        ou refusé en fonction des informations saisies.")
    st.write("")
    st.subheader("Comparables")
    st.write("Cette page vous permet de filtrer le jeu de données selon 3 critères: genre, \
        possession d'une voiture et niveau d'éducation.")
    st.write("Vous pourrez indiquer à votre client le pourcentage de crédits refusés en \
        fonction des critères renseignés (ex: femme n'ayant pas de voiture et un faible \
        niveau d'éducation.")


# ***************************** Page "Comprendre les variables" **********************************

if page == "Comprendre les variables":
    page_title("Mieux comprendre les variables du modèle")
    st.header("Définition des variables")
    st.write("Notre modèle de scoring a été créé à partir de variables séléectionnées \
        avec soin afin d'obtenir des performances optimales.")
    st.write("Nous avons retenu 11 variables parmi des centaines pôur obtenir \
        un scoring pertinent. Voici une brève explication de chacune d'entre elles.")
    st.subheader("EXT_SOURCE_3")
    st.write("Scoring du client réalisé par un organisme externe. Il s'agit d'un \
        chiffre compris entre 0 (mauvais) et 1 (excellent)")
    st.subheader("EXT_SOURCE_2")
    st.write("Autre score client provenant d'une source externe différente. \
        Il s'agit également d'un chiffre compris entre 0 (mauvais) et 1 (excellent)")  
    st.subheader("PREV_DAYS_DECISION_MIN")
    st.write("Nombre de jours écoulés depuis la signature du précédent crédit.")
    st.subheader("CODE_GENDER")
    st.write("Donnée catégorielle binaire décrivant le genre du client: 0 pour les femmes, \
        1 pour les hommes.")
    st.subheader("DAYS_EMPLOYED")
    st.write("Nombre de jours d'ancienneté si le client est employé.")
    st.subheader("PREV_APP_CREDIT_PERC_MIN")
    st.write('Ratio entre le montant initialement demandé et montant accordé lors \
        de la contraction du précédent crédit.')
    st.write("Example: Lors de la négocation du précédent crédit, le client avait demandé \
        1 million d'euros mais il n'a obtenu que 500,000 euros. Le ratio vaut donc 2.")
    st.subheader("INSTAL_DPD_MAX")
    st.write("Plus important retard de paiement en nombre de jours observé sur les crédits précédents.")
    st.subheader("AMT_CREDIT")
    st.write("Montant du précédent crédit contracté par le client.")
    st.subheader("DAYS_BIRTH")
    st.write("Age du client en nombre de jours.")  
    st.subheader("FLAG_OWN_CAR")
    st.write("Donnée catégorielle binaire indiquant si le client possède une voiture:\
         0 si non, 1 si oui.")
    st.subheader("NAME_EDUCATION_TYPE_Higher_education")
    st.write("Donnée catégorielle binaire indiquant si le client a fait des \
        études supérieures: 0 si non, 1 si oui.")


# ************************** Page "Comprendre les scores" ***************************

if page == "Comprendre les scores":
    page_title("Comprendre notre modèle de scoring crédit")
    st.subheader("Influence des variables sur le modèle")
    st.write("En tant que chargé commercial, vous devez pouvoir expliquer \
        à vos clients comment fonctionne notre système de scoring. Pour \
            cela, nous avons mis au point un système visuel permettant de \
                mieux comprendre comment le modèle prend ses décisions.")
    st.write("Ce système utilise les valeurs de Shapley. En quelques mots, ces valeurs \
        sont des coefficients indiquant l'impact de chaque variable sur la prédiction. \
            Plus cette valeur est élevée, plus l'impact de la variable sur la prévision \
                finale sera importante.")

    st.subheader("Les variables importantes")
    st.write("Certaines variables sont plus importantes que d'autres. Rien de \
        mieux qu'un graphique pour mieux comprendre")
    
    st.header("Importance des variables dans le modèle de prévision")
    # X = load_data('X')
    # X_train = load_data('X_train')
    X_train = get_pickle('X_train.pkl')
    # X = X.drop(columns=['TARGET'])
    X_train = X_train.drop(columns=['TARGET'])
    shap_values = load_pickle_shapley()
    st.subheader("Calcul des valeurs de Shapley pour chaque variable")
    fig, ax = plt.subplots(nrows=1, ncols=1)
    class_names=['crédit accepté','crédit refusé']
    shap.summary_plot(shap_values, X_train, plot_type="bar",
                        class_names= class_names,
                        feature_names = X_train.columns)
    # st.pyplot(fig)
    st.write(fig)
    st.write("La variable EXT_SOURCE_3 est la plus importante, suivie de l'ancienneté \
        dans son emploi, etc...")
    st.write("Si vous souhaitez mieux comprendre à quoi correspondent les variables, merci de \
        lire les informations sur la page 'Comprendre les variables'.")

    st.subheader('Importance des variables dans le modèle')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    class_names=['crédit accepté','crédit refusé']
    shap_obj = load_pickle_obj()
    shap.plots.beeswarm(shap_obj)
    st.write(fig)

    # st.pyplot(fig)
    st.write("La couleur rouge indique un niveau élevé de la variable, bleu un niveau faible.")
    st.write('Ce niveau faible ou élevé est à mettre en relation avec le seuil des valeurs de Shapley')
    st.write("En clair, un niveau élevé des scores EXT_SOURCE_2 et 3 (en rouge) va 'pousser' le score\
         vers la classe 0 correspondant à un crédit accepté . A l'inverse, un niveau bas (bleu) de ces \
        variable accroit le risque d'un refus de crédit.")
    st.write(" A l'inverse, plus le montant du dernier contracté crédit est élevé, plus il y a de \
        risque d'avoir un refus de crédit.")



# ***************************** Page "Comparables" **********************************

if page == "Comparables":
    page_title("Analyse des données comparables")
    # X = load_data('X')
    # X = get_pickle('X_train_sample.pkl')
    X = get_pickle('X_train.pkl')  # remplacé par train_sample (à vérifier)
    y = X['TARGET']
    X = X.drop(columns=['TARGET'])
    st.header('Statistiques sur les crédits comparables')
    # st.dataframe(X)
    st.subheader("Tout d'abord quelques chiffres...")
    st.write("Taille de l'échantillon sélectionné:", str(len(X)))
    st.write('Montant moyen des crédits: ',int(X.AMT_CREDIT.mean()), 
        '  ;  Montant maximum: ',int(X.AMT_CREDIT.max()),
        '  ; Montant minimum: ',int(X.AMT_CREDIT.min()))
    st.write("")
    st.write("Ancienneté moyenne pour les employés: ",
        int(X.DAYS_EMPLOYED.mean(skipna=True)/365), "ans",
        '   ; Ancienneté maximum: ',int(X.DAYS_EMPLOYED.max()/365), "ans")

    st.write('Age moyen: ',int(X.DAYS_BIRTH.mean(skipna=True)/365), "ans",
        ' ; Age maximum: ',int(X.DAYS_BIRTH.max()/365), "ans",
        ' ; Age minimum: ',int(X.DAYS_BIRTH.min()/365), "ans")
    st.write("")

    pct_female = len(X.CODE_GENDER[X.CODE_GENDER==0])/len(X)
    st.write("Pourcentage de femmes: ",int(pct_female*100), "%")

    nb_cars = len(X.FLAG_OWN_CAR[X.FLAG_OWN_CAR==1])
    pct_cars = nb_cars/len(X)
    st.write("Pourcentage de clients ayant une voiture: ",int(pct_cars*100), "%")

    nb_studies = len(X.NAME_EDUCATION_TYPE_Higher_education[
        X.NAME_EDUCATION_TYPE_Higher_education==1])
    pct_studies = nb_studies/len(X)
    st.write("Pourcentage de clients ayant fait des études supérieures: ",int(pct_studies*100), "%")

    
    # st.write("Nombre de crédits refusés: ",int(y_train_selected.sum())
    # st.write("Soit {:.0%} des crédits".format(y_train_selected.sum()/len(y_train_selected)))
    # add charts
    # https://docs.streamlit.io/library/api-reference/charts/st.pyplot

    st.subheader("Sélectioner des critères de similarité (plusieurs choix possibles")
    with st.form("my_form"):
        # Barre Gender selection
        sexe_unique = [0, 1]
        selected_sexe = st.multiselect('Sélectionner le genre (0=Femme, 1=Homme)',
                                    sexe_unique, sexe_unique)
        # Barre own car
        car_unique = [0, 1]
        selected_car = st.multiselect('Clients possédant une voiture (0 si non, 1 si oui)',
                                    car_unique, car_unique)
        # Barre études
        studies_unique = [0, 1]
        selected_studies = st.multiselect('Niveau études supérieures (0 si non, 1 si oui)',
                                        studies_unique, studies_unique)


        # Every form must have a submit button
        # Use st.form to avoid reloading all page at each filter change
        # https://docs.streamlit.io/library/api-reference/control-flow/st.form
        # https://stackoverflow.com/questions/69782552/how-to-stop-reloading-the-page-after-selecting-any-widget-with-python-in-streaml
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Filtering data
            df_selected_credits = X[(X.CODE_GENDER.isin(selected_sexe)) & 
                                    (X.FLAG_OWN_CAR.isin(selected_car)) &
                                    (X.NAME_EDUCATION_TYPE_Higher_education.isin(selected_studies))]
            df_selected_idx = df_selected_credits.index
            y_train_selected = y[df_selected_idx]

            st.subheader('Quelques graphiques...')
            
            col1, col2 = st.columns([10,10])
            with col1:
            # center title with logo
                fig = plt.figure(figsize=(5,3))
                ax = plt.axes()
                sns.histplot(data=df_selected_credits, x='EXT_SOURCE_3')
                plt.title('Répartition des ratings source_3')
                # st.write(fig)
                st.pyplot(fig)

            with col2:
                # Add image in 2nd column of the table
                # center title with logo
                fig = plt.figure(figsize=(5,3))
                #ax = plt.axes()
                sns.histplot(data=df_selected_credits, x='EXT_SOURCE_2')
                # ax.hist(X.INSTAL_DPD_MAX, bins=20)
                plt.title('Répartition des ratings source_2')
                # st.write(fig)
                st.pyplot(fig)
            col1, col2 = st.columns([10,10])

            with col1:
                # center title with logo
                fig = plt.figure(figsize=(5,3))
                ax = plt.axes()
                sns.histplot(data=df_selected_credits, x='AMT_CREDIT')
                plt.title('Répartition du montant des crédits ')
                # st.write(fig)
                st.pyplot(fig)

            with col2:
                # Add image in 2nd column of the table
                # center title with logo
                fig = plt.figure(figsize=(5,3))
                #ax = plt.axes()
                sns.histplot(data=df_selected_credits, x='DAYS_EMPLOYED')
                # ax.hist(X.INSTAL_DPD_MAX, bins=20)
                plt.title("Répartition de l'ancienneté des salariés en jours")
                # st.write(fig)
                st.pyplot(fig)
            col1, col2 = st.columns([10,10])
            with col1:
                # Pie chart, where the slices will be ordered and plotted counter-clockwise:
                labels = 'crédits accordés', 'crédits refusés'
                credit_not_ok = (y_train_selected.sum()/len(y_train_selected)) * 100
                sizes = [ 100-credit_not_ok, credit_not_ok]
                explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                # st.write(fig1)
                st.pyplot(fig1)
            with col2:
                # st.dataframe(y_train_selected)
                y_train_selected_df = y_train_selected.to_frame()
                fig = plt.figure()
                sns.countplot(x='TARGET', data=y_train_selected_df)
                plt.title('Nombre de crédits accordés (0) et refusés (1)')
                # st.write(fig)
                st.pyplot(fig)


    # submit_button = st.form_submit_button(label="Submit choice")
    # # select = st.button("Valider la sélection")
    # # if select:
    # if submit_button:
        


# ***************************** Page "score crédit" **********************************
if page == "Score crédit":
    page_title()
    # Select value of features
    st.header('Sélection des paramètres')
    st.write("")
    st.write("Pour mieux comprendre à quoi correspondent les variables ci-dessous \
        merci de vous reporter à la page 'Description des variables' ")

    st.subheader('Veuillez saisir les informations client')

    # EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', float(df_short['EXT_SOURCE_3'].min()),
    #                                  float(df_short['EXT_SOURCE_3'].max()),
    #                                  float(df_short['EXT_SOURCE_3'].mean()))
    GENDER = st.selectbox('Sélectionner le genre', ('Homme', 'Femme'))
    if GENDER=='Homme':
        CODE_GENDER=1
    else:
        CODE_GENDER=0
    # st.write(CODE_GENDER)
    CAR = st.selectbox('Le client possède une voiture', ('Oui', 'Non'))
    if CAR=='Oui':
        FLAG_OWN_CAR=1
    else:
        FLAG_OWN_CAR=0

    STUDIES = st.selectbox("Niveau d'études", ('Etudes supérieures', 'Niveau bac ou inférieur'))
    if STUDIES=='Etudes supérieures':
        NAME_EDUCATION_TYPE_Higher_education=1
    else:
        NAME_EDUCATION_TYPE_Higher_education=0

    # 
    # SEXE = st.sidebar.select_slider('Select sex (O=Female, 1=Male)', options=['Homme', 'Femme'])
    EXT_SOURCE_3 = st.slider("Score 'source 3' (variable EXT_SOURCE_3)",
                            0.0, 1.0, 0.5)
    EXT_SOURCE_2 = st.slider("Score 'source 2' (variable EXT_SOURCE_2)",
                            0.0, 1.0, 0.5)
    PREV_DAYS_DECISION_MIN = st.slider("Nombre de jours écoulés depuis signature du précédent crédit\
        (variable 'PREV_DAYS_DECISION_MIN')", 0, 3000, 1500)
    DAYS_EMPLOYED = st.slider("Nombre de jours d'ancienneté dans son emploi\
        (variable 'DAYS_EMPLOYED')", 0, 20000, 2000)
    PREV_APP_CREDIT_PERC_MIN = st.slider("Ratio montant demandé / accordé lors du précédent crédit. \
        (variable 'PREV_APP_CREDIT_PERC_MIN')", 0.0, 3.0, 0.79)
    INSTAL_DPD_MAX = st.slider("Plus gros retard de paiement en jours observé sur les crédits précédents\
        (variable 'INSTAL_DPD_MAX')", 0, 2800, 20)
    AMT_CREDIT = st.slider("Montant du précédent crédit contracté par le client. \
        (variable 'AMT_CREDIT')", 30000, 3000000, 600000)
    DAYS_BIRTH = st.slider("Age du client en nombre de jours. \
        (variable 'DAYS_BIRTH')", 4000, 30000, 16000)

    # text_test = st.sidebar.text_area("input EXT_source_3", EXT_SOURCE_3_input, height=10 )

    st.subheader("Synthèse des données du client saisies ci-dessus")
    # if CODE_GENDER==0:
    #     st.write('Gender: Female')
    # else:
    #     st.write('Gender: Male')

    # st.write('EXT_SOURCE_3 from slider:', EXT_SOURCE_3)
    # AMT_CREDIT = st.sidebar.slider('AMT_CREDIT', float(df_short['AMT_CREDIT'].min()),
    #                                float(df_short['AMT_CREDIT'].max()),
    #                                float(df_short['AMT_CREDIT'].mean()))
    features_names = ['EXT_SOURCE_3', 'EXT_SOURCE_2',
                    'PREV_DAYS_DECISION_MIN', 'CODE_GENDER',
                    'DAYS_EMPLOYED', 'PREV_APP_CREDIT_PERC_MIN',
                    'INSTAL_DPD_MAX', 'AMT_CREDIT',
                    'DAYS_BIRTH', 'FLAG_OWN_CAR',
                    'NAME_EDUCATION_TYPE_Higher_education'
                    ]
    features_data = [EXT_SOURCE_3, EXT_SOURCE_2,
                    PREV_DAYS_DECISION_MIN, CODE_GENDER,
                    DAYS_EMPLOYED, PREV_APP_CREDIT_PERC_MIN,
                    INSTAL_DPD_MAX, AMT_CREDIT,
                    DAYS_BIRTH, FLAG_OWN_CAR,
                    NAME_EDUCATION_TYPE_Higher_education
                    ]

    list_of_tuples = list(zip(features_names, features_data))
    client_data = pd.DataFrame(list_of_tuples, columns=['Feature_name',
                                                        'Feature_value'])                                            
    client_data = client_data.set_index('Feature_name', drop=True).T
    st.write(client_data)   
    # Add link to download personnal data as csv file (GDPR)
    st.subheader('Télécharger les données personnelles du client')
    st.write('Si votre client le souhaite, vous pouvez lui fournir le fichier contenant les \
        données personnelles utilisées dans le cadre du scoring de crédit.')
    st.markdown(filedownload(client_data), unsafe_allow_html=True)

    st.subheader("Cliquez sur le bouton ci-dessous pour calculer le score")
    st.write("L'affichage du score peut prendre quelques secondes, merci de patienter...")
    prediction = st.button("Résultat score")
    if prediction:
        response = get_api_response()
        st.subheader(response[15:-2])

        st.header('Interprétabilité locale')
        st.write("Chargement des données, merci de patienter...")
        # X = load_data('X')
        X = get_pickle('X_train.pkl')
        # X_train = load_data('X_train')
        X_train = get_pickle('X_rus_train_sample.pkl')
        X = X.drop(columns=['TARGET'])
        X_train = X_train.drop(columns=['TARGET'])
        st.subheader ('Analyse des résultats du scoring avec Lime')
        # Name of labels
        target_names=['Crédit accordé', 'Crédit refusé']
        # Generate LimeTabularExplainer on train set
        
        explainer = lime_tabular.LimeTabularExplainer(X.values,
                                                    mode="classification",
                                                    class_names=target_names,
                                                    feature_names=X.columns,
                                                    discretize_continuous=False  ## add to avoid bug
                                                    )
        classifier = load_pickle_model()
        explanation = explainer.explain_instance(client_data.values.reshape(-1),
                                                 classifier.predict_proba,
                                                 num_features=len(X.columns),
                                                 )
        # https://github.com/streamlit/streamlit/issues/779

        html = explanation.as_html()

        import streamlit.components.v1 as components
        components.html(html, height=350)

        shap_values = load_pickle_shapley()
        st.subheader('Importance des variables locales selon valeurs de Shapley')
        st.text("Classement de l'importance des variables dans la décision d'octroi de crédit de l'individu")


        # Afficher le graphe de force
        def st_shap(plot, height=None):
            """https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/9
            https://towardsdatascience.com/real-time-model-interpretability-api-using-shap-streamlit-and-docker-e664d9797a9a
            """
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        xgboost_explainer = shap.TreeExplainer(classifier, X_train)
        shap_values_client = xgboost_explainer.shap_values(client_data)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        p = shap.force_plot(xgboost_explainer.expected_value, shap_values_client, client_data)
        st_shap(p)
        st.write ('*Analyse locale des résultats du scoring avec les valeurs de Shapley*')
        st.text("Les variables en rouge (partie gauche) contribuent à augmenter la probabilité de refus de crédit.")
        st.text("Les variables en bleu (partie droite) contribuent à l'acceptation du crédit")

        st.subheader('Interprétabilité de chaque variable')
        shap_obj = load_pickle_obj()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.waterfall_plot(shap_obj[1])
        st.write(fig)