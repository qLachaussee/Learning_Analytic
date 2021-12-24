import streamlit as st

import pandas as pd
import numpy as np

import random
import os
import glob
import argparse
from tqdm import tqdm
from datetime import datetime

from google.oauth2.service_account import Credentials
from google.cloud import storage
import google.auth
from google.cloud import bigquery

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from streamlit.script_runner import StopException, RerunException

st.set_option('deprecation.showPyplotGlobalUse', False)
pd.options.mode.chained_assignment = None  

map_months = dict(zip(range(1,13), ["janvier","février","mars","avril","mai","juin","juillet","aout","septembre","octobre","novembre","décembre"]))
day = None


users = {"denis":"surv"}


def get_dataframe_from_gbq_api(adv_name:str , sensor_type:str, year:str, month=None, day=None, only_month=False, position = None):
    if type(year) != str:
        year = str(year)
    if month and type(month) != str:
        month = str(month)
    if day and type(day) != str:
        day = str(day)
    table_name = adv_name+"_"+sensor_type
    if month and type(month) is str and len(month) == 2 and month[0] == '0':
        month = month[1]

    if day and type(day) is str and len(day) == 2 and day[0] == '0':
        day = day[1]

    if only_month:
        QUERY = f"SELECT DISTINCT Month \
                    FROM `lisea-mesea-sandbox-272216.survaig_aiguilles_safe.{table_name}` \
                    WHERE Year={year}"
    else:
        QUERY = f"SELECT * \
                    FROM `lisea-mesea-sandbox-272216.survaig_aiguilles_safe.{table_name}` \
                    WHERE Year={year}"

    if month:
        QUERY = QUERY + f" AND Month={month}"
    if day:
        QUERY = QUERY + f" AND Day={day}"
    if position:
        QUERY = QUERY + f" AND Position={position}"

    # instanciation du client bigquery
    bqclient = bigquery.Client(project = "lisea-mesea-sandbox-272216")
    # Requete API
    query_job = bqclient.query(QUERY)
    # tranformation requete en dataframe
    return query_job.to_dataframe(create_bqstorage_client=False)


def generate_name(adv_name, sensor, year, month, feature, day=None):

    if day and day !="all":
        n_day = f"{day} "
    else:
        n_day = ""
    n_feature = feature.replace('_',' ')
    if month != "all":
        return f"{n_day}{map_months[month]} {year} , {adv_name} : {sensor} , {n_feature}"
    return f"{year} , {adv_name} : {sensor} , {n_feature}"


def extract_adv_sensor_names_available(all_tables):
    dict_output = dict()
    for elem in all_tables:
        adv_name, sensor_type = elem.split('_')[0], elem.split('_')[1]
        if adv_name not in dict_output:
            dict_output[adv_name] = [sensor_type]
        else:
            dict_output[adv_name].append(sensor_type)
    return dict_output


def generate_color_discrete_map(list_elem):
    viridis = cm.get_cmap('viridis', len(list_elem))

    dic_output = dict()
    for i, elem in enumerate(list_elem):
        r,g,b,a = viridis(i/len(list_elem))
        a = 0.5
        dic_output[elem] = f"rgba({r},{g},{b},{a})"
    return dic_output

def generate_legend_adding_position(df):
    poz_dict = {0:' (D)', 1:' (G)' }
    releve_dict = {0: " (Circulation)", 1:" (Manoeuvre)"}
    n_df = df.copy()
    n_df.drop_duplicates('Date du relevé', inplace = True)

    n_df.reset_index(inplace=True)
    all_poz = list(n_df["Position"].apply(lambda x:poz_dict[x]))
    all_releve = list(n_df["Type_releve"].apply(lambda x:releve_dict[x]))

    releve = [str(elem)+all_poz[i]+all_releve[i] for i, elem in enumerate(n_df['Date du relevé'])]
    return dict(zip(list(n_df['Date du relevé']),  releve  ))


##############################
### Découper le signal PEL ###
##############################

def show_split_pel_9_v2(df_total):
    def check_decontrole(liste):
        start = np.mean(liste[:10])
        start_std = np.std(liste[:100])
        start_max = np.max(liste[:10])
        end = np.mean(liste[-10:])
        end_std = np.std(liste[-100:])
        max = np.max(liste[100:200])
        # si ça ne monte pas
        if start < 1 or start_std < 100:
            return "Debut"
        # si ça ne remonte pas ou descend pas 
        elif end > 1 or end_std < 100:
            return "Fin"
        # si ça ne descend pas 
        elif max == start_max:
            return "Milieu"
        # si tout est ok
        else:
            return "Ok"
    
    df_total["Phase"] = 4
    liste_phase = []

    for date in df_total.Date.unique():
        df = df_total.copy()

        df = df[df['Date'] == date].sort_values(by="Id").reset_index(drop=True)
        df["Diff"] = df.Puissance_valeur.diff()

        df_reverse = df.sort_values(by="Id", ascending=False).reset_index(drop=True)
        df_reverse["Diff"] = df_reverse.Puissance_valeur.diff()

        number_roll_plat = 5
        number_roll_monte = 6

        # print(date)
        # print(check_decontrole(df.Puissance_valeur))

        # quand le problème c'est la fin
        if check_decontrole(df.Puissance_valeur) == "Fin":

            # trouver le point 3 -> la remonter de la manoeuvre
            try:
                index_3 = df[(df.Diff.rolling(number_roll_monte).sum() > 0) & (df.index>10)].index[0] - (number_roll_monte/2)
            except :
                index_3 = df[(df.Diff.rolling(number_roll_plat).std() < 0.5)].index[0] - (number_roll_monte/2)
            
            # identifier la phase de déverouillage
            df_start = df[df.index <= index_3]
            
            # trouver le point 1 -> le début de la descente du U
            index_1 = df_start[df_start.Diff == 0].index[0] - 1

            index_2 = df_start[(df_start.index < 10) &
                               (round(df_start.Diff) == 0)].index[-1]

            df["Phase"][df.index <= index_1] = 1
            df["Phase"][(df.index > index_1) & (df.index <= index_2)] = 2
            df["Phase"][(df.index > index_2) & (df.index <= index_3)] = 3

            # on affiche les 4 premieres phases
            df.Puissance_valeur[df.Phase == 1].plot(color="grey")
            df.Puissance_valeur[df.Phase == 2].plot(color="black")
            df.Puissance_valeur[df.Phase == 3].plot(color="blue")
            df.Puissance_valeur[df.Phase == 4].plot(color="orange")
        
        # quand le problème c'est le début
        elif check_decontrole(df.Puissance_valeur) == "Debut":
            print("Début")
        
        # quand le problème c'est le milieu
        elif check_decontrole(df.Puissance_valeur) == "Milieu":
            
            print("Anomalie")
            df.Courant_Controle_valeur.plot(color="orange")
        
        # sinon tout vas bien
        else:
            ######################
            # 3 premieres phases #
            ######################

            # trouver le point 3 -> la remonter de la manoeuvre
            try:
                index_3 = df[(df.Diff.rolling(number_roll_monte).sum() > 0) & (df.index>10)].index[0] - (number_roll_monte/2)
            except :
                index_3 = df[(df.Diff.rolling(number_roll_plat).std() < 0.5)].index[0] - (number_roll_monte/2)
            
            # identifier la phase de déverouillage
            df_start = df[df.index <= index_3]
            
            # trouver le point 1 -> le début de la descente du U
            index_1 = df_start[df_start.Diff == 0].index[0] - 1

            index_2 = df_start[(df_start.index < 10) &
                               (round(df_start.Diff) == 0)].index[-1]

            df["Phase"][df.index <= index_1] = 1
            df["Phase"][(df.index > index_1) & (df.index <= index_2)] = 2
            df["Phase"][(df.index > index_2) & (df.index <= index_3)] = 3

            ######################
            # 5 dernieres phases #
            ######################

            # on identifi le nombre d'index pour faire la conversion
            total = len(df.Puissance_valeur)
            
            # trouver le point 8 -> la fin de la chute du deverrouillage
            index_8 = df_reverse[(df_reverse.Diff > 1)].index[0]
            
            # trouver le point 7 -> la fin du plateau du deverrouillage
            index_7 = df_reverse[(df_reverse.index > index_8) & 
                                 (df_reverse.Diff < 0.5)].index[0]

            # trouver le point 4 -> le debut de la phase de deverrouillage
            index_4 = df_reverse[(df_reverse.index > index_7 + 50) & 
                                 (df_reverse.Diff.rolling(number_roll_monte).sum() > 10)].index[0] - (number_roll_monte)

            # trouver le point 5 -> le debut de la remonté du deverrouillage
            index_5 = df_reverse[(df_reverse.index > index_7) & 
                                 (df_reverse.index < index_4) &
                                 (df_reverse.Diff.rolling(number_roll_monte).sum() < -100)].index[-1] - (number_roll_monte/2)

            # trouver le point 6 -> la fin de la remonté du deverrouillage
            index_6 = df_reverse[(df_reverse.index > index_7) & 
                                 (df_reverse.index < index_5) &
                                 (df_reverse.Diff.rolling(number_roll_plat).mean() > 0)].index[-1]

            df["Phase"][(df.index > total - index_4) & (df.index <= total - index_5)] = 5
            df["Phase"][(df.index >= total - index_5) & (df.index <= total - index_6)] = 6
            df["Phase"][(df.index > total - index_6) & (df.index <= total - index_7)] = 7
            df["Phase"][(df.index >= total - index_7) & (df.index <= total - index_8)] = 8
            df["Phase"][(df.index > total - index_8)] = 9

            # on affiche les 9 phases
            df.Puissance_valeur[df.Phase == 1].plot(color="grey")
            df.Puissance_valeur[df.Phase == 2].plot(color="black")
            df.Puissance_valeur[df.Phase == 3].plot(color="blue")
            df.Puissance_valeur[df.Phase == 4].plot(color="orange")
            df.Puissance_valeur[df.Phase == 5].plot(color="red")
            df.Puissance_valeur[df.Phase == 6].plot(color="green")
            df.Puissance_valeur[df.Phase == 7].plot(color="brown")
            df.Puissance_valeur[df.Phase == 8].plot(color="purple")
            df.Puissance_valeur[df.Phase == 9].plot(color="grey")
            
        liste_phase.extend(df.Phase)
        
    st.pyplot()

    df_total = df_total.assign(Phase=liste_phase)
    df_total["Date"] = df_total["Date"].apply(lambda x: datetime.strftime(x, "%Y%m%d %H:%M:%S"))
    
    return df_total
            
##############################
### Découper le signal RKA ###
##############################

def show_split_rka_5_v2(df_total):
    def check_decontrole(liste):
        start = np.mean(liste[:10])
        start_std = np.std(liste[:10])
        end = np.mean(liste[-10:])
        end_std = np.std(liste[-10:])
        max = np.max(liste)
        # 0 - 30
        if end - start > 10 or start_std > 3:
            return "Debut"
        # 30 - 0
        elif end - start < -10 or end_std > 3:
            return "Fin"
        # si ça ne descend pas
        elif max > start + 5 or max > end + 5 or max < 1 or (start < 10 and end < 10):
            return "Milieu"
        # si tout est ok
        else:
            return "Ok"
    
    df_total["Phase"] = 3
    liste_phase = []

    for date in df_total.Date.unique():
        df = df_total.copy()

        df = df[df['Date'] == date].sort_values(by="Id").reset_index(drop=True)
        df["Diff"] = df.Courant_Controle_valeur.diff()

        df_reverse = df.sort_values(by="Id", ascending=False).reset_index(drop=True)
        df_reverse["Diff"] = df_reverse.Courant_Controle_valeur.diff()

        number_roll_plat = 6
        number_roll_descend = 6
        number_roll_monte = 4

        # print(date)
        # print(check_decontrole(df.Courant_Controle_valeur))

        # quand le problème c'est la fin
        if check_decontrole(df.Courant_Controle_valeur) == "Fin":
            
            # trouver le point 2 -> le début du bas du U
            try:
                index_2 = df[(df.Diff.rolling(number_roll_descend).sum() < -1) & (df.index <= df[df.Diff.rolling(number_roll_monte).sum() > 1].index[0])].index[-1]
            except:
                index_2 = df[(df.Diff.rolling(number_roll_descend).sum() < -1)].index[-1]

            # identifier le début du U
            df_start = df[df.index <= index_2]
            
            # trouver le point 1 -> le début de la descente du U
            index_1 = df_start[round(df_start.Diff.rolling(number_roll_plat).sum()) == 0].index[-1]
            
            df["Phase"][df.index <= index_1] = 1
            df["Phase"][(df.index >= index_1) & (df.index <= index_2)] = 2
            df["Phase"][df.index >= index_2] = 3

            # on affiche les 3 premières phases
            df.Courant_Controle_valeur[df.Phase == 1].plot(color="black")
            df.Courant_Controle_valeur[df.Phase == 2].plot(color="blue")
            df.Courant_Controle_valeur[df.Phase == 3].plot(color="orange")

        # quand le problème c'est le début
        elif check_decontrole(df.Courant_Controle_valeur) == "Debut":
            # on identifie le nombre d'index pour faire la conversion
            total = len(df.Courant_Controle_valeur)

            # trouver le point 3 -> la fin du bas du U
            try:
                index_3 = df_reverse[(df_reverse.Diff.rolling(number_roll_monte).sum() < -1) & (df_reverse.index <= df_reverse[df_reverse.Diff.rolling(number_roll_monte).sum() > 1].index[0])].index[-1]
            except:
                index_3 = df_reverse[(df_reverse.Diff.rolling(number_roll_monte).sum() < -1)].index[-1]

            # identifier la fin du U
            df_end = df_reverse[df_reverse.index <= index_3]
            
            # trouver le point 4 -> la fin de la remonté du U
            index_4 = df_end[round(df_end.Diff.rolling(number_roll_plat).sum()) == 0].index[-1]
            
            df["Phase"][df.index <= total - index_3] = 3
            df["Phase"][(df.index >= total - index_3) & (df.index <= total - index_4)] = 4
            df["Phase"][df.index >= total - index_4] = 5

            # on affiche les 3 dernières phases
            df.Courant_Controle_valeur[df.Phase == 3].plot(color="orange")
            df.Courant_Controle_valeur[df.Phase == 4].plot(color="green")
            df.Courant_Controle_valeur[df.Phase == 5].plot(color="brown")

        # quand le problème c'est le milieu
        elif check_decontrole(df.Courant_Controle_valeur) == "Milieu":
            
            print("Anomalie")
            df.Courant_Controle_valeur.plot(color="orange")
        
        # sinon tout vas bien
        else:
            ######################
            # 2 premieres phases #
            ######################

            # trouver le point 2 -> le début du bas du U
            try:
                index_2 = df[(df.Diff.rolling(number_roll_descend).sum() < -1) & (df.index <= df[df.Diff.rolling(number_roll_monte).sum() > 1].index[0])].index[-1]
            except:
                index_2 = df[(df.Diff.rolling(number_roll_descend).sum() < -1)].index[-1]
            
            # identifier le début du U
            df_start = df[df.index <= index_2]
            
            # trouver le point 1 -> le début de la descente du U
            index_1 = df_start[round(df_start.Diff.rolling(number_roll_plat).sum()) == 0].index[-1]
            
            df["Phase"][df.index <= index_1] = 1
            df["Phase"][(df.index >= index_1) & (df.index <= index_2)] = 2
            
            ######################
            # 2 dernieres phases #
            ######################

            # on identifi le nombre d'index pour faire la conversion
            total = len(df.Courant_Controle_valeur)
            
            # trouver le point 3 -> la fin du bas du U
            try:
                index_3 = df_reverse[(df_reverse.Diff.rolling(number_roll_monte).sum() < -1) & (df_reverse.index <= df_reverse[df_reverse.Diff.rolling(number_roll_monte).sum() > 1].index[0])].index[-1]
            except:
                index_3 = df_reverse[(df_reverse.Diff.rolling(number_roll_monte).sum() < -1)].index[-1]
            
            # identifier la fin du U
            df_end = df_reverse[df_reverse.index <= index_3]
            
            # trouver le point 4 -> la fin de la remonté du U
            index_4 = df_end[round(df_end.Diff.rolling(number_roll_plat).sum()) == 0].index[-1]
            
            df["Phase"][(df.index >= total - index_3) & (df.index <= total - index_4)] = 4
            df["Phase"][df.index >= total - index_4] = 5

            ###############
            # 3ème phases #
            ###############

            df["Phase"][(df.index >= index_2) & (df.index <= index_3)] = 3

            # on affiche les 5 phases
            df.Courant_Controle_valeur[df.Phase == 1].plot(color="black")
            df.Courant_Controle_valeur[df.Phase == 2].plot(color="blue")
            df.Courant_Controle_valeur[df.Phase == 3].plot(color="orange")
            df.Courant_Controle_valeur[df.Phase == 4].plot(color="green")
            df.Courant_Controle_valeur[df.Phase == 5].plot(color="brown")

        liste_phase.extend(df.Phase)
        
    st.pyplot()

    df_total = df_total.assign(Phase=liste_phase)
    df_total["Date"] = df_total["Date"].apply(lambda x: datetime.strftime(x, "%Y%m%d %H:%M:%S"))
    
    return df_total

#############################
### Afficher les Features ###
#############################

def show_features(df_total, feature):
    if feature == "Puissance_valeur":
        size_phase = 9
    elif feature == "Courant_Controle_valeur":
        size_phase = 5
    with st.expander(f'Moyenne :'):
        for i in range(1, size_phase + 1):
            features = df_total[df_total.Phase == i].groupby(by='Date')[feature].mean()
            fig_scatter = px.line(x=features.index, y=features.values,
                                    title=f"Phase {i}", width=650, height=400)
            fig_scatter.update_layout(yaxis_range=[-1, max(features.values)*1.3])
            st.plotly_chart(fig_scatter)

    with st.expander(f'Variance :'):
        for i in range(1, size_phase + 1):
            features = df_total[df_total.Phase == i].groupby(by='Date')[feature].std()
            fig_scatter = px.line(x=features.index, y=features.values,
                                    title=f"Phase {i}", width=650, height=400)
            fig_scatter.update_layout(yaxis_range=[-1, max(features.values)*1.3])
            st.plotly_chart(fig_scatter)
    
    with st.expander(f'Somme :'):
        for i in range(1, size_phase + 1):
            features = df_total[df_total.Phase == i].groupby(by='Date')[feature].sum()
            fig_scatter = px.line(x=features.index, y=features.values,
                                    title=f"Phase {i}", width=650, height=400)
            fig_scatter.update_layout(yaxis_range=[-1, max(features.values)*1.3])
            st.plotly_chart(fig_scatter)
    
    with st.expander(f'Longueur :'):
        for i in range(1, size_phase + 1):
            features = df_total[df_total.Phase == i].groupby(by='Date')[feature].count()
            fig_scatter = px.line(x=features.index, y=features.values,
                                    title=f"Phase {i}", width=650, height=400)
            fig_scatter.update_layout(yaxis_range=[-1, max(features.values)*1.3])
            st.plotly_chart(fig_scatter)

def main():

    credentials = Credentials.from_service_account_file("C:/Users/qlachaussee/Downloads/lisea-mesea-sandbox-272216-127d5e43c8e5.json")
    storage_client = storage.Client(project="lisea-mesea-sandbox-272216", credentials=credentials)
    bucket = storage_client.bucket("lisea-mesea-sea-cloud-data-collection-safe")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/qlachaussee/Downloads/lisea-mesea-sandbox-272216-127d5e43c8e5.json"
    bqclient = bigquery.Client(project = "lisea-mesea-sandbox-272216")



    tables = bqclient.list_tables("survaig_aiguilles_safe")
    all_tables = [tbl.table_id for tbl in tables]


    dic = extract_adv_sensor_names_available(all_tables)

    st.header('Survaig dashboard')

    adv_name = st.sidebar.selectbox('Choisir Aiguille à explorer', list(dic.keys()))


    sensor = st.sidebar.selectbox('Choisir Capteur à explorer', dic[adv_name])

    list_years = [2017, 2018, 2019, 2020, 2021]

    year = st.sidebar.selectbox('Choisir Année à explorer', list_years)


    df = get_dataframe_from_gbq_api(adv_name, sensor, year, only_month=True)  ### add month option from available months

    unik_months = list(df['Month'].unique())
    unik_months.sort()
    unik_months.append("all")
    month = st.sidebar.selectbox('Choisir Mois à explorer', unik_months)
    my_position = st.radio("Type de position à visualiser",('Toutes', 'Aiguille à Droite', 'Aiguille à Gauche'))


    if my_position and my_position != "Toutes":
        map_position = {"Aiguille à Droite":0, 'Aiguille à Gauche':1}
        my_position = map_position[my_position]
    else:
        my_position = None

    if month != "all":
        df = get_dataframe_from_gbq_api(adv_name, sensor, year, month, position=my_position)  ### add month option from available months
        unik_day = list(df['Day'].unique())
        unik_day.sort()
        unik_day.append("all")
        day = st.sidebar.selectbox('Choisir Jour à explorer', unik_day, index=len(unik_day)-1)

        if day != "all":
            df = get_dataframe_from_gbq_api(adv_name, sensor, year, month, day, position=my_position)  ### add month option from available months

    else:
        df = get_dataframe_from_gbq_api(adv_name, sensor, year, position=my_position)
        day = None



    if "pel" in sensor.lower():
        features = ["Puissance_valeur", "Tension_valeur", "Courant_valeur"]
    if "rka" in sensor.lower():
        features = ["Courant_Controle_valeur"]



    df["Date du relevé"] = df["Date"].apply(lambda x:datetime.strftime(x, "%d %B %Y %H:%M:%S"))
    df = df.sort_values(by=["Date", "Id"], ascending=[True, True])



    my_color_map = generate_color_discrete_map(list(df['Date du relevé'].unique()))


    for feature in features:
        n_feature = feature.replace('_',' ')
        with st.expander(f'Visualiser {n_feature} :'):
            fig = px.line(df, x="Id", y=feature, color="Date du relevé", line_group="Date du relevé", hover_name="Date du relevé",
                    line_shape="spline", render_mode="svg",color_discrete_map=my_color_map, title=generate_name(adv_name, sensor, year, month, feature, day))

            newnames = generate_legend_adding_position(df)
            fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                                  legendgroup = newnames[t.name],
                                                  hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                                 )
                              )
            st.plotly_chart(fig)

        if n_feature == "Puissance valeur":
            with st.expander(f'Découper {n_feature} :'):
                # try:
                df_total = show_split_pel_9_v2(df)
                # except:
                #     print("pel_9_v2_fail")  
            show_features(df_total, "Puissance_valeur")
            # except:
            #     print("Pas de Features")

        if n_feature == "Courant Controle valeur":
            with st.expander(f'Découper {n_feature} :'):
                try:
                    df_total = show_split_rka_5_v2(df)
                except:
                    print("rka_5_v2_fail")
            try:
                show_features(df_total, "Courant_Controle_valeur")
            except:
                print("Pas de Features")
                   

if __name__ == '__main__':
    # Initialization
    if 'logged_in' not in st.session_state:
        print("noo")
        with st.form(key='login_form'):
            if "username" not in st.session_state:
                my_user = st.text_input("Username or e-mail")
                password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                if my_user.lower() in users and users[my_user.lower()] == password:
                    st.success("Succesfully logged in! :tada:")
                    st.session_state.logged_in = True
                    st.session_state.key = 'OK'
                    with st.spinner("Redirecting to application..."):
                        time.sleep(1)
                        print("okkkkkk")
                        st.experimental_rerun()
                        # st.experimental_rerun()
                else:
                    st.error("Invalid User/Password. Try again. :no_entry:")

    else:
        main()
