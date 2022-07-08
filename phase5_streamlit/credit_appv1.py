"""
Pour exécuter le code en local:
streamlit run credit_app.py
Useful resources:
https://github.com/RihabFekii/streamlit-app/blob/master/frontend/app.py
https://rihab-feki.medium.com/deploying-machine-learning-models-with-streamlit-fastapi-and-docker-bb16bbf8eb91

"""
#from turtle import width
from calendar import c
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import base64
from PIL import Image
import shap
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from lime import lime_tabular
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use( 'Agg' )
import seaborn as sns

# use full page length
st.set_page_config(layout="wide")

# ############### DASHBOARD TITLE ###############

# Add logo on the right & title on the left
image = Image.open('credit_logo.png')
col1, col2 = st.columns([25,6])
with col1:
    # center title with logo
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("# Application de scoring de crédits")

with col2:
    # Add image in 2nd column of the table
    st.image(image, use_column_width=False, width=250)

# Add line below header
st.write(" *** ")

# ############### LOAD DATA ###############

#https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas
# url = 'https://media.githubusercontent.com/media/cyrbaufr/credit_heroku/master/'
# full_url = url+'X_train_imputed.csv'
url=''
# Voir fichier app_preprocessing.ipynb
# csv_name = 'X_train_raw.csv'

@st.cache
def get_data(csv_name):
    """ Load data from csv file & put in cache 
    """
    full_url = url + csv_name
    return pd.read_csv(full_url, index_col=0)

# Add progress bar when loading data
my_bar = st.progress(0)
for pct_complete in range (100):
    time.sleep(0.1)
    my_bar.progress(pct_complete +1)
    X = get_data('X_train_sample.csv')
    X_rus_train = get_data('X_rus_train_sample.csv')
    X_rus_test = get_data('X_rus_test_sample.csv')
# st.write(X)

y = X['TARGET']
X = X.drop(columns=['TARGET'])
# # Standardisation des données
# scaler = StandardScaler()
# X_train_std = scaler.fit_transform(X)
# X_train_std = pd.DataFrame(X_train_std, columns=X.columns)

# ############### CREATE FUNCTION TO DOWNLOAD CSV ###############

def filedownload(df):
    """ Function to download csv file created a Dataframe df
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="credit.csv">Download CSV File</a>'
    return href

# ############### CREATE SIDEBAR ###############

# Select value of features
st.sidebar.header('Sélection des paramètres')

# EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', float(df_short['EXT_SOURCE_3'].min()),
#                                  float(df_short['EXT_SOURCE_3'].max()),
#                                  float(df_short['EXT_SOURCE_3'].mean()))
# CODE_GENDER = st.sidebar.selectbox('Sexe', ('H', 'F'))
# 
# SEXE = st.sidebar.select_slider('Select sex (O=Female, 1=Male)', options=['Homme', 'Femme'])
EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', 0.0, 1.0, 0.5)
EXT_SOURCE_2 = st.sidebar.slider('EXT_SOURCE_2', 0.0, 1.0, 0.5)
PREV_DAYS_DECISION_MIN = st.sidebar.slider('PREV_DAYS_DECISION_MIN', 0, 3000, 1500)

DAYS_EMPLOYED = st.sidebar.slider('DAYS_EMPLOYED', 0, 20000, 2000)
PREV_APP_CREDIT_PERC_MIN = st.sidebar.slider('PREV_APP_CREDIT_PERC_MIN', 0.0, 3.0, 0.79)
INSTAL_DPD_MAX = st.sidebar.slider('INSTAL_DPD_MAX', 0, 2800, 20)
AMT_CREDIT = st.sidebar.slider('AMT_CREDIT', 30000, 3000000, 600000)
DAYS_BIRTH = st.sidebar.slider('DAYS_BIRTH', 4000, 30000, 16000)

FLAG_OWN_CAR = st.sidebar.select_slider('FLAG_OWN_CAR (0=No, 1=Yes)', options=[0, 1])
CODE_GENDER = st.sidebar.select_slider('Select sex (O=Female, 1=Male)', options=[0, 1])
NAME_EDUCATION_TYPE_Higher_education = st.sidebar.select_slider(
    'NAME_EDUCATION_TYPE_Higher_education (0=No, 1=Yes)', options=[0, 1])

# text_test = st.sidebar.text_area("input EXT_source_3", EXT_SOURCE_3_input, height=10 )

st.subheader("Synthèse des données client disponibles")
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
st.sidebar.header('Télécharger les données personnelles du client')
st.sidebar.markdown(filedownload(client_data), unsafe_allow_html=True)

# ############### CALCULATE CREDIT SCORE FROM MODEL ###############

# Read logistic regression model 'classifier_final' from pickle file
# To see how this file was created, look at the following notebook
# /phase2_modelisation/6_modeling_log_reg.ipynb
classifier = pickle.load(open('classifier_xgb_best.pkl', 'rb'))

# Use model to predict 0/1 for a credit with features entered in sidebar
prediction = classifier.predict(client_data)

# st.write('Classe sélectionnée (0= crédit accepté / 1 = crédit refusé')
# st.write (prediction[0])
# # Calculate probability of class 0 & class 1
# prediction_proba = classifier.predict_proba(client_data)
# st.write('Probabilité de crédit remboursé:')
# st.write (prediction_proba[0,0]*100, '%')
# st.write('Probabilité de crédit non remboursé:')
# st.write (prediction_proba[0,1]*100, '%')

st.subheader('Résultat du scoring')
credit_score = np.array(['Crédit approuvé','Crédit refusé'])
if prediction ==0:
    approved = '<p style="font-family:Courier; color:Green; font-size: 20px;">Crédit approuvé</p>'
    st.markdown(approved, unsafe_allow_html=True)
else:
     st.markdown('**Crédit refusé**')
# credit_score[int(prediction)])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)

# ############### LOCAL EXPLAINABILITY WITH LIME ###############

# Name of labels
target_names=['Crédit accordé', 'Crédit refusé']
# Generate LimeTabularExplainer on train set
explainer = lime_tabular.LimeTabularExplainer(X.values,
                                              mode="classification",
                                              class_names=target_names,
                                              feature_names=X.columns,
                                              discretize_continuous=False  ## add to avoid bug
                                              )
# # création d'un pipeline optimisé avec imblearn pipeline
# steps = [('imputer', SimpleImputer(strategy='mean')),
#          ('ros', RandomUnderSampler(random_state=42)),
#          ('scaler', StandardScaler()),
#          ('model', LogisticRegression(penalty='l2',
#                                       C=0.01,
#                                       solver='liblinear'))]
# # imblearn pipeline
# pipeline = Pipeline(steps)
# pipeline.fit(X_train.values, y_train)



st.subheader ('Analyse des résultats du scoring avec Lime')
print("Prediction : ", target_names[int(classifier.predict(client_data.values.reshape(1,-1))[0])])
explanation = explainer.explain_instance(client_data.values.reshape(-1),
                                         classifier.predict_proba,
                                         num_features=len(X.columns),
                                        )
# https://github.com/streamlit/streamlit/issues/779

html = explanation.as_html()

import streamlit.components.v1 as components
components.html(html, height=350)

# ############### LOCAL EXPLAINABILITY WITH SHAP ###############

st.subheader ('Analyse locale des résultats du scoring avec les valeurs de Shapley')
st.text("Les variables en rouge à gauche augmentent la probabilité de refus de crédit.")
st.text("Les variables Celles en bleu à droite contribuent à l'acceptation du crédit")

# Echantillonnage des données avec RandomUnderSampler
# Add progress bar when loading data

X_rus_train = X_rus_train.drop(columns=['TARGET'])
X_rus_test = X_rus_test.drop(columns=['TARGET'])
# Calcul des valeurs de Shap
# https://github.com/slundberg/shap/issues/1373

def subsample_data(X, y, n_sample=100, seed_temp=1234):
    """Subsample data, stratified by target variable y
    https://github.com/Chancylin/shap_loss/blob/master/helper_functions/shap_help.py
    https://towardsdatascience.com/use-shap-loss-values-to-debug-monitor-your-model-83f7808af40f
    """
    frac = n_sample / X.shape[0]

    data = X.copy(deep=True)
    data["label"] = y
    strata = ["label"]

    data_subsample = data.groupby(strata, group_keys=False)\
        .apply(lambda x: x.sample(frac=frac, replace=False, random_state=seed_temp))

    # assert
    #print("imbalance ratio()")

    return data_subsample[X.columns]

xgboost_explainer = shap.TreeExplainer(classifier.named_steps['model'], X_rus_train)
shap_values = xgboost_explainer.shap_values(X_rus_test)

prediction = classifier.predict(client_data.values.reshape(1, -1))[0]
st.write("Classe prédite: ", prediction)

# ############### GLOBAL EXPLAINABILITY WITH SHAP ###############

# Afficher le graphe de force
def st_shap(plot, height=None):
    """https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/9
    https://towardsdatascience.com/real-time-model-interpretability-api-using-shap-streamlit-and-docker-e664d9797a9a
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


shap_values_client = xgboost_explainer.shap_values(client_data)
fig, ax = plt.subplots(nrows=1, ncols=1)
p = shap.force_plot(xgboost_explainer.expected_value, shap_values_client, client_data)
st_shap(p)

st.subheader('Importance des variables locales selon valeurs de Shapley')
st.text("Classement de l'importance des variables dans la décision d'octroi de crédit de l'individu")
fig, ax = plt.subplots(nrows=1, ncols=1)
class_names=['crédit accepté','crédit refusé']
shap.summary_plot(shap_values, X_rus_train, plot_type="bar",
                  class_names= class_names,
                  feature_names = X.columns)
# st.pyplot(fig)
st.write(fig)

st.subheader('Importance des variables dans le modèle')
fig, ax = plt.subplots(nrows=1, ncols=1)
class_names=['crédit accepté','crédit refusé']
shap_obj = xgboost_explainer(X_rus_train)
shap.plots.beeswarm(shap_obj)
st.write(fig)
# st.pyplot(fig)
st.write("Analyse: Un chiffre élevé des scores EXT_SOURCE_2 et 3 permet d'améliorer le score.")
st.write(" Plus le montant du bien financé est élevé, plus il y a de chance d'avoir le crédit")


st.subheader ('Sélection des critères pour analyse des comparables')

st.write ("3 crières disponibles: genre, niveau d'éducation et possession d'une voiture")

# explainer = shap.LinearExplainer(lr_classifier, X_train)
# shap_values = explainer.shap_values(client_data)
# shap.plots.waterfall(shap_values[0])

# Barre Gender selection
gender_unique = [0, 1]
selected_gender = st.multiselect('Sélectionner le genre (0=Femme, 1=Homme)',
                               gender_unique, gender_unique)
# Barre education
education_unique = [0, 1]
selected_education = st.multiselect("Sélectionner le niveau d'éducation (1 si supérieur, 0 sinon)",
                                      education_unique, education_unique)
# Barre FLAG_OWN_CAR
car_unique = [0, 1]
selected_car = st.multiselect('Le client possède une voiture (0=non, 1=oui)',
                                      car_unique, car_unique)
# Filtering data
df_selected_credits = X[(X.CODE_GENDER.isin(selected_gender)) & 
                        (X.NAME_EDUCATION_TYPE_Higher_education.isin(selected_education)) &
                        (X.FLAG_OWN_CAR.isin(selected_car))]

st.header('Statistiques sur les crédits comparables')
# st.dataframe(df_selected_credits)
st.subheader('Quelques chiffres...')
st.write('Montant moyen des crédits: ',int(df_selected_credits.AMT_CREDIT.mean(skipna=True)), 
    '  ;  Montant maximum: ',int(df_selected_credits.AMT_CREDIT.max()),
    '  ; Montant minimum: ',int(df_selected_credits.AMT_CREDIT.min()))
st.write("")
st.write("Ancienneté moyenne pour les employés: ",
    int(df_selected_credits.DAYS_EMPLOYED.mean(skipna=True)/365), "ans",
    '   ; Ancienneté maximum: ',int(df_selected_credits.DAYS_EMPLOYED.max()/365), "ans")

st.write('Age moyen: ',int(df_selected_credits.DAYS_BIRTH.mean(skipna=True)/365), "ans",
    ' ; Age maximum: ',int(df_selected_credits.DAYS_BIRTH.max()/365), "ans",
    ' ; Age minimum: ',int(df_selected_credits.DAYS_BIRTH.min()/365), "ans")
st.write("")
nb_female = len(df_selected_credits.CODE_GENDER[df_selected_credits.CODE_GENDER==0])
pct_f = nb_female/len(df_selected_credits)
st.write("Pourcentage de femmes dans l'échantillon: ",int(pct_f*100), "%")

nb_cars = len(df_selected_credits.FLAG_OWN_CAR[df_selected_credits.FLAG_OWN_CAR==1])
pct_cars = nb_cars/len(df_selected_credits)
st.write("Pourcentage de clients de l'échantillon ayant une voiture: ",int(pct_cars*100), "%")

nb_studies = len(df_selected_credits.NAME_EDUCATION_TYPE_Higher_education[
    df_selected_credits.NAME_EDUCATION_TYPE_Higher_education==1])
pct_studies = nb_studies/len(df_selected_credits)
st.write("Pourcentage de clients ayant fait des études supérieures: ",int(pct_studies*100), "%")

st.write("Nombre de crédits de l'échantillon: ",len(df_selected_credits))

df_selected_idx = df_selected_credits.index
y_train_selected = y[df_selected_idx]
# st.write("Nombre de crédits refusés: ",int(y_train_selected.sum())
# st.write("Soit {:.0%} des crédits".format(y_train_selected.sum()/len(y_train_selected)))
# add charts
# https://docs.streamlit.io/library/api-reference/charts/st.pyplot

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
    # ax.hist(df_selected_credits.INSTAL_DPD_MAX, bins=20)
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
    # ax.hist(df_selected_credits.INSTAL_DPD_MAX, bins=20)
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
    

# UserWarning: X has feature names, but LogisticRegression was fitted without feature names