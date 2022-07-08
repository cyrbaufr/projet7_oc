"""
Pour exécuter le code en local:
streamlit run credit_app.py

"""


#from turtle import width
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

# use full page length
st.set_page_config(layout="wide")
# Add logo on the right & title on the right
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

# # #### Add table with test set ####
# st.subheader("Affichage du tableau de données avec tous les crédits du test set")
# # load test set
# df_test = pd.read_csv('../data_models/X_test_main_features_XGBoost_weighted_class_top50.csv',
#                       index_col=0)
# df_train = pd.read_csv('../data_models/X_train_main_features_XGBoost_weighted_class_top50.csv',
#                       index_col=0)
# df_train = df_train.drop(columns='TARGET')
# # Delete lines with missing values
# df_test = df_test.dropna()
# # Show first 20 lines of the table
# df_short = df_test.iloc[:20,:]
# st.write(df_short)
# # store indexes in a list
# credit_idx = df_short.index.to_list()

# #### create a sidebar on the left ####

# # Add box to select a credit according to its index
# st.sidebar.write("")
# a = st.sidebar.selectbox('Sélectionner un numéro de crédit', credit_idx)
# st.write("### Crédit sélectioné: ",a)
# # Add a button to download data of a client (GDRP)

def filedownload(df):
    """ Function to download csv file created from a Dataframe df
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="credit.csv">Download CSV File</a>'
    return href

# Select value of features
st.sidebar.header('Sélection des paramètres')

# EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', float(df_short['EXT_SOURCE_3'].min()),
#                                  float(df_short['EXT_SOURCE_3'].max()),
#                                  float(df_short['EXT_SOURCE_3'].mean()))
# CODE_GENDER = st.sidebar.selectbox('Sexe', ('H', 'F'))
# 
# SEXE = st.sidebar.select_slider('Select sex (O=Female, 1=Male)', options=['Homme', 'Femme'])
CODE_GENDER = st.sidebar.select_slider('Select sex (O=Female, 1=Male)', options=[0, 1])
FLAG_DOCUMENT_3 = st.sidebar.select_slider('FLAG_DOCUMENT_3 (0=No, 1=Yes)', options=[0, 1]) 
EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', 0.0, 1.0, 0.5)
EXT_SOURCE_2 = st.sidebar.slider('EXT_SOURCE_2', 0.0, 1.0, 0.5)
AMT_CREDIT = st.sidebar.slider('AMT_CREDIT', 0.0, 100000.0, 50000.0)
AMT_GOODS_PRICE = st.sidebar.slider('AMT_GOODS_PRICE', 0.0, 100000.0, 50000.0)
NAME_INCOME_TYPE_Working = st.sidebar.select_slider('INCOME from Working (0=No, 1=Yes)', options=[0, 1])
DAYS_EMPLOYED = st.sidebar.slider('DAYS_EMPLOYED', 0, 1000, 0)
INSTAL_DAYS_ENTRY_PAYMENT_MEAN = st.sidebar.slider('INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 0, 100, 0)
INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.slider('INSTAL_DAYS_ENTRY_PAYMENT_MAX', 0, 100, 0)
# EXT_SOURCE_3_input = 0
# text_test = st.sidebar.text_area("input EXT_source_3", EXT_SOURCE_3_input, height=10 )

st.subheader("Valeur des variables retenues pour le scoring")
if CODE_GENDER==0:
    st.write('Gender: Female')
else:
    st.write('Gender: Male')

st.write('EXT_SOURCE_3 from slider:', EXT_SOURCE_3)
# AMT_CREDIT = st.sidebar.slider('AMT_CREDIT', float(df_short['AMT_CREDIT'].min()),
#                                float(df_short['AMT_CREDIT'].max()),
#                                float(df_short['AMT_CREDIT'].mean()))
# AMT_GOODS_PRICE = st.sidebar.slider('AMT_GOODS_PRICE', float(df_short['AMT_GOODS_PRICE'].min()),
#                                     float(df_short['AMT_GOODS_PRICE'].max()),
#                                     float(df_short['AMT_GOODS_PRICE'].mean()))
# FLAG_DOCUMENT_3 = st.sidebar.slider('FLAG_DOCUMENT_3', float(df_short['FLAG_DOCUMENT_3'].min()),
#                                     float(df_short['FLAG_DOCUMENT_3'].max()),
#                                     float(df_short['FLAG_DOCUMENT_3'].mean()))
# CODE_GENDER = st.sidebar.slider('CODE_GENDER', df_short['CODE_GENDER'].min(),
#                                 df_short['CODE_GENDER'].max(),0)
# NAME_INCOME_TYPE_Working = st.sidebar.slider('NAME_INCOME_TYPE_Working',
#                                              int(df_short['NAME_INCOME_TYPE_Working'].min()),
#                                              int(df_short['NAME_INCOME_TYPE_Working'].max()),
#                                              0)
# INSTAL_DAYS_ENTRY_PAYMENT_MEAN = st.sidebar.slider('INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'].min()),
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'].max()),
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MEAN'].mean()))
# INSTAL_DAYS_ENTRY_PAYMENT_MAX = st.sidebar.slider('INSTAL_DAYS_ENTRY_PAYMENT_MAX',
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MAX'].min()),
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MAX'].max()),
#     float(df_short['INSTAL_DAYS_ENTRY_PAYMENT_MAX'].mean()))
# DAYS_EMPLOYED = st.sidebar.slider('DAYS_EMPLOYED', float(df_short['DAYS_EMPLOYED'].min()),
#                                   float(df_short['DAYS_EMPLOYED'].max()),
#                                   float(df_short['DAYS_EMPLOYED'].mean()))

# Create dataframe from input data
features_names = ['EXT_SOURCE_3', 'EXT_SOURCE_2',
                  'AMT_CREDIT', 'FLAG_DOCUMENT_3',
                  'AMT_GOODS_PRICE', 'CODE_GENDER',
                  'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
                  'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
                  'DAYS_EMPLOYED',
                  'NAME_INCOME_TYPE_Working']
features_data = [EXT_SOURCE_3, EXT_SOURCE_2,
                 AMT_CREDIT, FLAG_DOCUMENT_3,
                 AMT_GOODS_PRICE, CODE_GENDER,
                 INSTAL_DAYS_ENTRY_PAYMENT_MAX,
                 INSTAL_DAYS_ENTRY_PAYMENT_MEAN,
                 DAYS_EMPLOYED,
                 NAME_INCOME_TYPE_Working]
list_of_tuples = list(zip(features_names, features_data))
client_data = pd.DataFrame(list_of_tuples, columns=['Feature_name',
                                                    'Feature_value'])                                            
client_data = client_data.set_index('Feature_name', drop=True).T
st.write(client_data)   
# Add link to download personnal data as csv file (GDPR)
st.sidebar.header('Télécharger les données personnelles du client')
st.sidebar.markdown(filedownload(client_data), unsafe_allow_html=True)

# Read logistic regression model from pickle file
lr_classifier = pickle.load(open('classifier_final.pkl', 'rb'))
# Use model to predict 0/1 for a credit selected in test set
prediction = lr_classifier.predict(client_data)
st.write('Classe sélectionnée (0= crédit accepté / 1 = crédit refusé')
st.write (prediction[0])

prediction_proba = lr_classifier.predict_proba(client_data)
st.write('Probabilité de crédit accepté:')
st.write (prediction_proba[0,0]*100, '%')
st.write('Probabilité de crédit refusé:')
st.write (prediction_proba[0,1]*100, '%')

st.subheader('Résultat du scoring')
credit_score = np.array(['Crédit approuvé','Crédit refusé'])
st.write(credit_score[int(prediction)])

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Explainability
#https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas
url = 'https://media.githubusercontent.com/media/cyrbaufr/credit_heroku/master/'
full_url = url+'X_train_main_features_XGBoost_weighted_class_top50.csv'
X = pd.read_csv(full_url, index_col=0)
y = X['TARGET']
X = X[features_names]
# explainer = shap.LinX = X[features_names]
# création des train et test sets avec stratification (classes déséquilibrées)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2, stratify=y)
# Nom des labels
target_names=['Crédit accordé', 'Crédit refusé']
# Génération du LimeTabularExplainer sur le train set
explainer = lime_tabular.LimeTabularExplainer(X_train.values,
                                              mode="classification",
                                              class_names=target_names,
                                              feature_names=X.columns,
                                              discretize_continuous=False  ## ajouté pour éviter bug
                                              )
# création d'un pipeline optimisé avec imblearn pipeline
steps = [('imputer', SimpleImputer(strategy='mean')),
         ('ros', RandomUnderSampler(random_state=42)),
         ('scaler', StandardScaler()),
         ('model', LogisticRegression(penalty='l2',
                                      C=0.01,
                                      solver='liblinear'))]
# imblearn pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train.values, y_train)

st.subheader ('Analyse des résultats du scoring avec Lime')
print("Prediction : ", target_names[int(pipeline.predict(client_data.values.reshape(1,-1))[0])])
explanation = explainer.explain_instance(client_data.values.reshape(-1), pipeline.predict_proba,
                                         num_features=len(X.columns),
                                        )
# https://github.com/streamlit/streamlit/issues/779

html = explanation.as_html()

import streamlit.components.v1 as components
components.html(html, height=350)

st.subheader ('Analyse locale des résultats du scoring avec les valeurs de Shapley')
st.text("Les variables en rouge à gauche augmentent la probabilité de refus de crédit. "
    "Celles en bleu à droite contribuent positivement à l'acceptation du crédit")
# Echantillonnage des données avec RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=True)
x_rus, y_rus = rus.fit_resample(X.values, y)
# Création d'un train & test set à partir des données sous-échantillonnées
X_rus_train, X_rus_test, y_rus_train, y_rus_test = train_test_split(x_rus, y_rus,
                                                                    test_size=0.3,
                                                                    random_state=2)
# Imputation des valeurs manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train_imputed = imputer.fit_transform(X_rus_train)
X_test_imputed = imputer.transform(X_rus_test)
# Standardisation des données
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_imputed)
X_test_std = scaler.transform(client_data.values)
# Transformation en dataframe
X_train_std = pd.DataFrame(X_train_std, columns=X.columns)
X_test_std = pd.DataFrame(X_test_std, columns=client_data.columns)
# Régression logistique
lr = LogisticRegression(penalty='l2',
                        C=0.01,
                        solver='liblinear')
# Entrainement du modèle
lr.fit(X_train_std.values, y_rus_train)
# Calcul des valeurs de Shap
log_reg_explainer = shap.LinearExplainer(lr, X_train_std)
shap_values = log_reg_explainer.shap_values(X_test_std)

prediction = lr.predict(X_test_std.values.reshape(1, -1))[0]
st.write("Classe prédite: ", prediction)
# Afficher le graphe de force
def st_shap(plot, height=None):
    """https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/9
    https://towardsdatascience.com/real-time-model-interpretability-api-using-shap-streamlit-and-docker-e664d9797a9a
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

fig, ax = plt.subplots(nrows=1, ncols=1)
p = shap.force_plot(log_reg_explainer.expected_value, 
                shap_values, X_test_std)
st_shap(p)

st.subheader('Importance des variables locles selon valeurs de Shapley')
st.text("Classement de l'importance des variables dans la décision d'octroi de crédit de l'individu")
fig, ax = plt.subplots(nrows=1, ncols=1)
class_names=['crédit accepté','crédit refusé']
shap.summary_plot(shap_values, X_train_std.values, plot_type="bar",
                  class_names= class_names,
                  feature_names = X.columns)
st.pyplot(fig)

st.subheader('Importance des variables dasn le modèle')
fig, ax = plt.subplots(nrows=1, ncols=1)
class_names=['crédit accepté','crédit refusé']
shap_obj = log_reg_explainer(X_train_std)
shap.plots.beeswarm(shap_obj)
st.pyplot(fig)
st.write("Analyse: Un chiffre élevé des scores EXT_SOURCE_2 et 3 permet d'améliorer le score.")
st.write(" Plus le montant du bien financé est élevé, plus il y a de chance d'avoir le crédit")


st.subheader ('Analyse des résultats du scoring avec Lime')

# explainer = shap.LinearExplainer(lr_classifier, X_train)
# shap_values = explainer.shap_values(client_data)
# shap.plots.waterfall(shap_values[0])

# Barre Gender selection
sexe_unique = [0, 1]
selected_sexe = st.multiselect('Sélectionner le sexe (0=Femme, 1=Homme)',
                               sexe_unique, sexe_unique)
# Barre INCOME from Working
worker_unique = [0, 1]
selected_income_work = st.multiselect('Sélectionner le type de revenus (1 si salarié, 0 sinon)',
                                      worker_unique, worker_unique)
# Barre FLAG_DOCUMENT_3
doc3_unique = [0, 1]
selected_doc3 = st.multiselect('Document 3 rempli (0=non, 1=oui)',
                                      doc3_unique, doc3_unique)
# Filtering data
df_selected_credits = X_train[(X_train.CODE_GENDER.isin(selected_sexe)) & 
                              (X_train.NAME_INCOME_TYPE_Working.isin(selected_income_work)) &
                              (X_train.FLAG_DOCUMENT_3.isin(selected_doc3))]

st.header('Statistiques sur les crédits')
st.dataframe(df_selected_credits)
st.write('Montant moyen du crédit: ',int(df_selected_credits.AMT_CREDIT.mean(skipna=True)))
st.write('Montant maximum du crédit: ',int(df_selected_credits.AMT_CREDIT.max()))
st.write('Montant minimum du crédit: ',int(df_selected_credits.AMT_CREDIT.min()))
st.write("")
st.write('Montant moyen du bien acheté: ',int(df_selected_credits.AMT_GOODS_PRICE.mean(skipna=True)))
st.write('Montant maximum du crédit: ',int(df_selected_credits.AMT_GOODS_PRICE.max()))
st.write('Montant minimum du crédit: ',int(df_selected_credits.AMT_GOODS_PRICE.min()))
st.write("")
st.write('Montant moyen INSTAL_DAYS_ENTRY_PAYMENT_MEAN: ',int(df_selected_credits.INSTAL_DAYS_ENTRY_PAYMENT_MEAN.mean(skipna=True)))
st.write('Montant maximum INSTAL_DAYS_ENTRY_PAYMENT_MEAN: ',int(df_selected_credits.INSTAL_DAYS_ENTRY_PAYMENT_MEAN.max()))
st.write('Montant minimum INSTAL_DAYS_ENTRY_PAYMENT_MEAN: ',int(df_selected_credits.INSTAL_DAYS_ENTRY_PAYMENT_MEAN.min()))
st.write("")
st.write("Nombre de crédits sélectionnés: ",len(df_selected_credits))
df_selected_idx = df_selected_credits.index
y_train_selected = y_train[df_selected_idx]
st.write("Nombre de crédits refusés: ",y_train_selected.sum())
st.write("Soit {:.0%} des crédits".format(y_train_selected.sum()/len(y_train_selected)))
# add charts
# https://docs.streamlit.io/library/api-reference/charts/st.pyplot
col1, col2 = st.columns([10,10])
with col1:
    # center title with logo
    fig = plt.figure(figsize=(5,3))
    ax = plt.axes()
    ax.hist(df_selected_credits.AMT_CREDIT, bins=20)
    plt.title('Répartition du montant des crédits ')
    st.pyplot(fig)

with col2:
    # Add image in 2nd column of the table
    # center title with logo
    fig = plt.figure(figsize=(5,3))
    ax = plt.axes()
    ax.hist(df_selected_credits.AMT_GOODS_PRICE, bins=20)
    plt.title('Répartition du prix des biens ')
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
    st.write(fig1)
with col2:
    # st.dataframe(y_train_selected)
    y_train_selected_df = y_train_selected.to_frame()
    fig = plt.figure()
    sns.countplot(x='TARGET', data=y_train_selected_df)
    plt.title('Nombre de crédits accordés (0) et refusés (1)')
    st.write(fig)
    

# UserWarning: X has feature names, but LogisticRegression was fitted without feature names