# Importation des modules
import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image

# local
API_URL = "http://51.21.116.35:8000/"
# deployment cloud


# Chargement des dataset
data_train = pd.read_csv('train_df_sample.csv')
data_test = pd.read_csv('test_df_sample.csv')


# Fonctions

##Pretraitement des données
#***************************************************#
def prepocessing_var(df, scaler):
    cols = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    if scaler == 'minmax':
        scal = MinMaxScaler()
    else:
        scal = StandardScaler()

    df_scaled[cols] = scal.fit_transform(df[cols])
    return df_scaled

data_train_mm = prepocessing_var(data_train, 'minmax')
data_test_mm = prepocessing_var(data_test, 'minmax')


##Récuperation de la probabilité de defaut de paiement
#***************************************************#
def prediction_client(client_id):
    url_get_pred = API_URL + "prediction/" + str(client_id)
    response = requests.get(url_get_pred)
    proba_default = round(float(response.content), 3)
    best_threshold = 0.54
    if proba_default >= best_threshold:
        decision = "Refusé"
    else:
        decision = "Accordé"

    return proba_default, decision


##Construction graphique jauge qui va retourner le score du client
#***************************************************#
def graph_jauge(proba):
    """Affiche une jauge indiquant le score du client.
    :param: proba (float).
    """
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number",
        title={'text': "Jauge de score"},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "grey"},
                   {'range': [20, 45], 'color': "grey"},
                   {'range': [45, 54], 'color': "grey"},
                   {'range': [54, 100], 'color': "grey"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 1, 'value': 54}}))

    st.plotly_chart(fig)



##Interpretation locale
#***************************************************#
def valeur_shape(client_id):
    url_get_shap_local = API_URL + "shaplocal/" + str(client_id)

    response = requests.get(url_get_shap_local)
    res = json.loads(response.content)
    shap_val_local = res['shap_values']
    base_value = res['base_value']
    feat_values = res['data']
    feat_names = res['feature_names']

    explanation = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), (1, -1)),
                                   base_value,
                                   data=np.reshape(np.array(feat_values, dtype='float'), (1, -1)),
                                   feature_names=feat_names)

    return explanation[0]


##Interpretation globale
#***************************************************#
def get_shap_val():
    url_get_shap = API_URL + "shap/"
    response = requests.get(url_get_shap)
    content = json.loads(response.content)
    shap_val_glob_0 = content['shap_values_0']
    shap_val_glob_1 = content['shap_values_1']
    shap_globales = np.array([shap_val_glob_0, shap_val_glob_1])

    return shap_globales

##Clients similaires
#***************************************************#
def df_voisins(id_client):
    url_get_df_voisins = API_URL + "clients_similaires/" + str(id_client)
    response = requests.get(url_get_df_voisins)
    try:
        # Utilisez directement response.json() pour obtenir le JSON parsé
        data_voisins = pd.read_json(response.json())
    except Exception as e:
        print(f"Erreur lors de la lecture JSON : {e}")
        data_voisins = pd.DataFrame()  # Ou une autre valeur par défaut

    #data_voisins = pd.read_json(eval(response.content))

    return data_voisins


##Distribution du feature et affichage de la position du client
#***************************************************#
def distribution(feature, id_client, df):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='accordé')
    ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='refusé')

    observation_value = data_test.loc[data_test['SK_ID_CURR'] == id_client][feature].values
    ax.axvline(observation_value, color='green', linestyle='dashed', linewidth=2, label='Client')

    ax.set_xlabel('Valeur de la feature', fontsize=20)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=20)
    ax.set_title(f'Histogramme de la feature "{feature}" pour les cibles accordé et refusé', fontsize=22)
    ax.legend(fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    st.pyplot(fig)


##Nuage de point de la variable X en fonction de Y et affichage de la position du client
#***************************************************#
def scatter(id_client, feature_x, feature_y, df):
    fig, ax = plt.subplots(figsize=(10, 6))

    data_accord = df[df['TARGET'] == 0]
    data_refus = df[df['TARGET'] == 1]
    ax.scatter(data_accord[feature_x], data_accord[feature_y], color='blue',
               alpha=0.5, label='accordé')
    ax.scatter(data_refus[feature_x], data_refus[feature_y], color='red',
               alpha=0.5, label='refusé')

    data_client = data_test.loc[data_test['SK_ID_CURR'] == id_client]
    observation_x = data_client[feature_x]
    observation_y = data_client[feature_y]
    ax.scatter(observation_x, observation_y, marker='*', s=200, color='black', label='Client')

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Analyse bivariée des caractéristiques sélectionnées')
    ax.legend()

    st.pyplot(fig)


##Representation de la boite à moustache et affichage de la position du client et les 10 plus proches voisins
#***************************************************#
def boxplot_graph(id_client, feat, df_vois): 

    df_box = data_train_mm.melt(id_vars=['TARGET'], value_vars=feat,
                                var_name="variables", value_name="values")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_box, x='variables', y='values', hue='TARGET', ax=ax)


    df_voisins_scaled = prepocessing_var(df_vois, 'minmax')
    df_voisins_box = df_voisins_scaled.melt(id_vars=['TARGET'], value_vars=feat,
                                            var_name="var", value_name="val")
    sns.swarmplot(data=df_voisins_box, x='var', y='val', hue='TARGET', size=8,
                  palette=['green', 'red'], ax=ax)

    data_client = data_test_mm.loc[data_test['SK_ID_CURR'] == id_client][feat]
    categories = ax.get_xticks()
    for cat in categories:
        plt.scatter(cat, data_client.iloc[:, cat], marker='*', s=250, color='blueviolet', label='Client')

    ax.set_title(f'Boxplot des caractéristiques sélectionnées')
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) < 8:
        ax.legend(handles[:4], ['Accordé', 'Refusé', 'Voisins', 'Client'])
    else:
        ax.legend(handles[:5], ['Accordé', 'Refusé', 'Voisins (accordés)', 'Voisins (refusés)', 'Client'])

    st.pyplot(fig)


#Titre de la page
st.set_page_config(page_title="Interface de prédiction Prêt à dépenser", layout="wide")

# Sidebar
with st.sidebar:
    logo = Image.open('img/pret_depenser.png')
    st.image(logo, width=200)
    # Page selection
    page = st.selectbox('Navigation', ["Acceuil", "Information du client", "Interprétation locale",
                                               "Interprétation globale"])

    # ID Selection
    st.markdown("""---""")

    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    id_client_dash = st.selectbox("ID Client", list_id_client)
    st.write('Vous avez choisi le client ID : '+str(id_client_dash))

    st.markdown("""---""")
    st.write("Crée par Sayoba GANSANE, Data Sientist||Market analyst")


if page == "Acceuil":
    st.title("Interface de prédiction - Page d'acceuil")



    st.markdown("Bienvenue dans notre interface dédiée à l'explication des décisions\n"
                "de crédit. Cette plateforme a été conçue pour offrir à nos clients\n"
                "une compréhension claire des facteurs qui influent sur\n"
                "l'approbation ou le refus de leur demande de crédit.\n"
                
                "\nLes prévisions générées reposent sur un modèle Light GBM (Light Gradient Boosting Machine). "
                " Les résultats sont calculés à partir d'un ensemble de données disponible [ici](https://www.kaggle.com/c/home-credit-default-risk/data). "
                "Lors du déploiement, un échantillon de ces données a été utilisé.\n"
                
                "\nDécouvrez les différentes fonctionnalités de notre tableau de bord :\n"
                "- **Information du client**: Cette page regroupe toutes les informations pertinentes"
                "concernant le client sélectionné, ainsi que le résultat de sa demande de crédit."
                "N'hésitez pas à explorer cette section pour démarrer.\n"
                "- **Interprétation locale**: Identifiez les caractéristiques spécifiques du client qui ont eu"
                "le plus d'influence sur la décision d'approbation ou de refus de la demande de crédit.\n"
                "- **Intérprétation globale**:  Comparez le profil du client avec d'autres clients"
                "de notre base de données et avec des clients similaires.\n"
                
                "\nNous vous invitons à explorer les différentes pages pour une expérience complète et transparente. En cas de questions, n'hésitez pas à nous contacter.")





if page == "Information du client":
    st.title("Interface de prédiction - Informations du client")

    st.write("Cliquez sur le bouton ci-dessous pour debuter l'analyse de la demande :")
    button_start = st.button("Statut de la demande")
    if button_start:
        if id_client_dash != '<Select>':
            # Calcul des prédictions et affichage des résultats
            st.markdown("RÉSULTAT DE LA DEMANDE")
            probability, decision = prediction_client(id_client_dash)

            if decision == 'Accordé':
                st.success("PRÊT ACCORDÉ")
            else:
                st.error("Crédit refusé")

            # Affichage de la jauge
            graph_jauge(probability)

    # Affichage des informations client
    with st.expander("Afficher les informations du client", expanded=False):
        st.info("Voici les informations du client:")
        st.write(pd.DataFrame(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash]))


if page == "Interprétation locale":
    st.title("Interface de prédiction - Page Interprétation locale")

    locale = st.checkbox("Interprétation locale")
    if locale:
        st.info("Interprétation locale de la prédiction")
        shap_val = valeur_shape(id_client_dash)
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
        # Affichage du waterfall plot : shap local
        fig = shap.waterfall_plot(shap_val, max_display=nb_features, show=False)
        st.pyplot(fig)

        with st.expander("Explication du graphique", expanded=False):
            st.caption("Cette section met en lumière les particularités locales qui ont impacté la prise de "
                       "décision. Autrement dit, elle expose les caractéristiques qui ont joué un rôle "
                       "déterminant dans la décision spécifique à ce client..")


if page == "Interprétation globale":
    st.title("Interface de prédiction - Page Interprétation globale")
    # Création du dataframe de voisins similaires
    data_voisins = df_voisins(id_client_dash)

    globale = st.checkbox("Importance globale")
    if globale:
        st.info("Importance globale")
        shap_values = get_shap_val()
        data_test_std = prepocessing_var(data_test.drop('SK_ID_CURR', axis=1), 'std')
        nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)
        fig, ax = plt.subplots()
        # Affichage du summary plot : shap global
        ax = shap.summary_plot(shap_values[1], data_test_std, plot_type='bar', max_display=nb_features)
        st.pyplot(fig)

        with st.expander("Explication du graphique", expanded=False):
            st.caption("Ce graphique represente les caractéristiques qui influent de manière globale la décision du client.")

    distrib = st.checkbox("Comparaison des distributions")
    if distrib:
        st.info("Comparaison des distributions de plusieurs features")
        distrib_compa = st.radio("Choisir type de comparaison :", ('Tous', 'Clients similaires'), key='distrib')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        with st.spinner(text="Chargement des graphiques..."):
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('AMT_CREDIT'))
                if distrib_compa == 'Tous':
                    distribution(feature1, id_client_dash, data_train)
                else:
                    distribution(feature1, id_client_dash, data_voisins)
            with col2:
                feature2 = st.selectbox("Choisissez une caractéristique", list_features,
                                        index=list_features.index('EXT_SOURCE_2'))
                if distrib_compa == 'Tous':
                    distribution(feature2, id_client_dash, data_train)
                else:
                    distribution(feature2, id_client_dash, data_voisins)

            with st.expander("Explication des distributions", expanded=False):
                st.caption("Choisissez le feature dont vous voulez analyser la distribution. En bleu, on a la distribution "
                           "des clients solvable dont le prêt a été accordé et en orange, ceux dont le prêt "
                           "a été réfusé. Les pointillées en verts indiquent la position du client par rapport "
                           "aux autres clients.")

    bivar = st.checkbox("Analyse bivariée")
    if bivar:
        st.info("Analyse bivariée")
        bivar_compa = st.radio("Choisir type de comparaison :", ('Tous', 'Clients similaires'), key='bivar')

        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        list_features.insert(0, '<Select>')

        # Selection des features à afficher
        c1, c2 = st.columns(2)
        with c1:
            feat1 = st.selectbox("Sélectionner une variable X ", list_features)
        with c2:
            feat2 = st.selectbox("Sélectionner une variable Y", list_features)
        if (feat1 != '<Select>') & (feat2 != '<Select>'):
            if bivar_compa == 'Tous':
                scatter(id_client_dash, feat1, feat2, data_train)
            else:
                scatter(id_client_dash, feat1, feat2, data_voisins)
            with st.expander("Explication des scatter plot", expanded=False):
                st.caption("Ici, on peut afficher une variable en fonction d'une autre. "
                           "En bleu sont indiqués les clients solvables et en rouge les clients non solvables."
                           "L'étoile noire correspond au client en cours d'analyse et permet donc de le situer par rapport "
                           "à la base de données clients.")

    boxplot = st.checkbox("Analyse des boxplot")
    if boxplot:
        st.info("Comparaison des distributions de plusieurs feautures en utilisant les boxplot.")

        feat_quanti = data_train.select_dtypes(['float64']).columns
        # Selection des features à afficher
        features = st.multiselect("Choisir les features à visualiser: ",
                                  sorted(feat_quanti),
                                  default=['AMT_CREDIT', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

        # Affichage des boxplot
        boxplot_graph(id_client_dash, features, data_voisins)
        with st.expander("Explication des boxplot", expanded=False):
            st.caption("Les boxplot permettent d'analyser les distributions des variables renseignées. "
                       "Une étoile violette représente le client. Ses plus proches voisins sont également "
                       "renseignés sous forme de points de couleurs (rouge pour ceux étant qualifiés comme "
                       "étant en défaut et vert pour les autres).")
