#%%
import streamlit as st
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import pickle

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt
import time

st.set_option('deprecation.showPyplotGlobalUse', False)

id_student = pickle.load(open( "id_student.p", "rb" ) )

users = {id:"20/20" for id in id_student}

@st.experimental_memo
def load_data():
    return pickle.load(open( "dataset_dict.p", "rb" ) )

def print_load_data():
        
    status_text = st.sidebar.markdown('<p style="font-size:15px;color:blue">Downloading the dataset...</p>', unsafe_allow_html=True)
    progress_bar = st.sidebar.progress(0)
    
    # Chargement des datasets
    progress_bar.progress(50)
    dataset_dict = load_data()
    
    status_text.text("Dataset downloaded !")
    progress_bar.progress(100)
    
    return dataset_dict


def getOneCourse(dataset_dict, code_module, code_presentation):
    '''
    Fusionne toutes les tables par leur clés primaires pour un seul cours,
    remplace NaNs valeurs des dates Exam/unregistration manquantes avec module_presentation_length
    '''
    
    def filterByCode(table):
        '''retourne un booléen'''
        if type(code_module) == str:
            if type(code_presentation) == str:
                return  (table['code_module'] == code_module) & \
                        (table['code_presentation'] == code_presentation)
            else:
                return  (table['code_module'] == code_module) & \
                        (table['code_presentation'].isin(list(code_presentation)))
        else:
            if type(code_presentation) == str:
                return  (table['code_module'].isin(list(code_module))) & \
                        (table['code_presentation'] == code_presentation)
            else:
                return  (table['code_module'].isin(list(code_module))) & \
                        (table['code_presentation'].isin(list(code_presentation)))
            
    course = dataset_dict['courses']
    course = course[filterByCode(course)]
    module_presentation_length = course['module_presentation_length'].values[0]
    
    # studentAssessment -> complete studentAssessments avec les infos assessments
    assessments = dataset_dict['assessments']
    assessments.loc[ (assessments['assessment_type'] == 'Exam') & \
        filterByCode(assessments), \
        'date'] = module_presentation_length
    assessments = assessments[filterByCode(assessments)]
    studentAssessment = pd.merge(dataset_dict['studentAssessment'], assessments, \
                                 how='inner', on='id_assessment')

    # studentInfo -> complete studentInfo avec studentRegistration
    studentRegistration = dataset_dict['studentRegistration']
    studentRegistration.loc[studentRegistration['date_unregistration'].isna() & \
        filterByCode(studentRegistration), \
        'date_unregistration'] = module_presentation_length
    studentRegistration = studentRegistration[ \
                           filterByCode(studentRegistration)]
    studentInfo = pd.merge(dataset_dict['studentInfo'], studentRegistration, \
                           how='inner', on=['id_student', 'code_module', 'code_presentation'])

    # studentVle = complete vle avec studentVle
    vle = dataset_dict['vle']
    vle = vle[filterByCode(vle)]
    studentVle = pd.merge(dataset_dict['studentVle'], vle, \
                          how='inner', on=['id_site', 'code_module', 'code_presentation'])
    del studentVle['id_site']
    
    # complete avec studentVle avec studentInfo, complete studentAssessment avec studentInfo
    # puis ajoute studentVle avec l'étudiantAssessment, trié par date
    studentVleInfo = pd.merge(studentVle, studentInfo, \
                              how='inner', on=['id_student', 'code_module', 'code_presentation'])
    studentAssessmentInfo = pd.merge(studentAssessment, studentInfo, \
                              how='inner', on=['id_student', 'code_presentation', 'code_module'])
    
    combined_df = studentAssessmentInfo.append(studentVleInfo)
    del combined_df['code_module']
    del combined_df['code_presentation']
    del combined_df['id_assessment']
    combined_df = combined_df.sort_values(by=['date'])
    return combined_df

def restructure(oneCourse, days):
    '''
    agréger la séquence de chaque élève pour que chaque élève ne contienne qu'une seule ligne,
    en ne conservant que les données des deux premières semaines
    
    '''
    # La prédiction n'est intéressante que lorsqu'il s'agit du début du cours, et non de la fin !
    first14Days_oneCourse = oneCourse[oneCourse['date'] <= days]
    # supprimer ceux qui se sont désinscrits avant le début car nous ne pouvons rien faire pour eux.
    first14Days_oneCourse = first14Days_oneCourse[first14Days_oneCourse['date_unregistration'] \
                                                  > 0]
    # Liste de types d'activités uniques pour créer des caractéristiques.
    activity_types_df = first14Days_oneCourse['activity_type'].unique()
    # supprimer le type d'activité NaN
    activity_types_df = [x for x in activity_types_df if type(x) == str]
    #  nous voulons un étudiant par ligne (la manière facile)
    final_df = first14Days_oneCourse.groupby('id_student').agg({
        'score': [np.mean, np.sum],
        'date_submitted': [np.mean],
        'is_banked': [np.sum],
        'assessment_type': [('CMA_count', lambda x: x.values[x.values == 'CMA'].size), \
                            ('TMA_count', lambda x: x.values[x.values == 'TMA'].size)],
        'date':[np.mean],
        'weight': [np.mean, np.sum],
        'gender': [('first', lambda x: x.values[0])],
        'region': [('first', lambda x: x.values[0])],
        'highest_education': [('first', lambda x: x.values[0])],
        'imd_band': [('first', lambda x: x.values[0])],
        'age_band': [('first', lambda x: x.values[0])],
        'num_of_prev_attempts': [('first', lambda x: x.values[0])],
        'studied_credits': [('first', lambda x: x.values[0])],
        'disability': [('first', lambda x: x.values[0])],
        'final_result': [('first', lambda x: x.values[0])],
        'date_registration': [('first', lambda x: x.values[0])],
        'date_unregistration': [('first', lambda x: x.values[0])],
        'sum_click': [np.mean, np.sum],
        'activity_type': [('list', lambda x: [x.values[x.values == activity].size \
                                              for activity in activity_types_df])]
    })
    # ne conserver qu'un seul niveau de noms de colonnes
    final_df.columns = ["_".join(x) for x in final_df.columns]
    custom_columns = ['activity_type_' + str(x) for x in activity_types_df]
    # diviser et concaténer les caractéristiques créées (il devrait y avoir une meilleure façon de le faire...)
    final_df = final_df.join(pd.DataFrame(final_df.activity_type_list.values.tolist(), \
                                          columns=custom_columns, index=final_df.index))
    del final_df['activity_type_list']
    return final_df

def cleanAndMap(final_df, encode=True):
    '''
    remplace les NaNs introduits par les nouvelles caractéristiques et si encode=True -> map 
    les colonnes catogoriques en nombres
    retourner l'objet encodeur pour décoder les étiquettes plus tard
    '''
    # remplacer les NaNs valeurs
    final_df.loc[final_df['weight_mean'].isna(), 'weight_mean'] = 0
    final_df.loc[final_df['score_mean'].isna(), 'score_mean'] = 0
    final_df.loc[final_df['sum_click_mean'].isna(), 'sum_click_mean'] = 0
    final_df.loc[final_df['date_submitted_mean'].isna(), 'date_submitted_mean'] = -1

    activities = [x for x in final_df.columns if re.search("^activity_type", x)]
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    final_df.loc[:,activities] = imp.fit_transform(final_df.loc[:,activities])

    if not encode:
        return
        
    # remplacement des variables catégoriques par des nombres
    # ordre de manipulation manuelle pour les caractéristiques ordinales
    categorical_columns = ['gender_first', 'disability_first', 'region_first', 'age_band_first']
    ordinal_columns = {
        'highest_education_first': {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        },
        'imd_band_first': {
            '0-100%': 0, #mettre 50 ?
            '0-10%': 5,
            '10-20': 15,
            '20-30%': 25,
            '30-40%': 35,
            '40-50%': 45,
            '50-60%': 55,
            '60-70%': 65,
            '70-80%': 75,
            '80-90%': 85,
            '90-100%': 95
        },
        'final_result_first': {
            'Withdrawn': 0,
            'Fail': 1,
            'Pass': 2,
            'Distinction': 3,
        }
    }
    categorical_encoders = {col: LabelEncoder() for col in categorical_columns }
    for col in categorical_columns:
        final_df.loc[:, col] = categorical_encoders[col].fit_transform(final_df[col])
    for col in ordinal_columns:
        final_df.loc[:, col] = final_df.loc[:, col].map(ordinal_columns[col])
        
    return categorical_encoders

def filtre_par_3(table, id_student, code_module, code_presentation):
    '''retourne un booléen'''
    if type(code_module) == str:
        if type(code_presentation) == str:
            filtre = (table['code_module'] == code_module) & \
                    (table['code_presentation'] == code_presentation)
        else:
            filtre = (table['code_module'] == code_module) & \
                    (table['code_presentation'].isin(list(code_presentation)))
    else:
        if type(code_presentation) == str:
            filtre = (table['code_module'].isin(list(code_module))) & \
                    (table['code_presentation'] == code_presentation)
        else:
            filtre = (table['code_module'].isin(list(code_module))) & \
                    (table['code_presentation'].isin(list(code_presentation)))
    return table[(table["id_student"] == int(id_student)) & (filtre)]
    

def one(final_df):
    plt.figure(figsize=(10,8))
    final_df['final_result_first'].value_counts(dropna=False).plot(kind='bar')
    plt.ylabel('Number of data points')
    plt.xlabel('final result')
    plt.show()
    st.pyplot()
 
def two(final_df):
    counts = final_df['imd_band_first'].value_counts(dropna=False)
    plt.figure(figsize=(10,10))
    plt.pie(counts, labels=counts.index, colors=['green', 'blue', 'red'])
    plt.title('Pie chart showing counts for\nstudentInfo imd_band categories')
    plt.show()
    st.pyplot()
 
def three(final_df):
    sns.set()
    final_df.groupby(['imd_band_first','final_result_first']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,8))
    plt.show()
    st.pyplot()
 
def main():
    
    st.set_page_config(page_title="Meilleur site", page_icon=":mortar_board:")
    st.header("Projet Learning Analityc")    
    
    dataset_dict = print_load_data()
    st.sidebar.write('<p style="color:red; font-size: 12px;">* : facultatif</p>', unsafe_allow_html=True)

    
    id_student = st.sidebar.text_input("Votre identifiant d'étudiant ?", "")
    
    if not id_student:
        st.warning("Veuillez rentre un identifiant d'étudiant")
        st.stop()
    
    st.balloons()
    
    student_registration = dataset_dict["studentRegistration"][dataset_dict["studentRegistration"]["id_student"] == int(id_student)]
    
    list_module = student_registration["code_module"].unique()
    code_module = st.sidebar.multiselect("Quels modules à analyser ? *", list_module, default=list_module)
    
    if not code_module:
        st.warning("Veuillez sélectionner un ou plusieurs module(s)")
        st.stop()
    
    list_presentation = student_registration[student_registration["code_module"].isin(code_module)]["code_presentation"].unique()
    code_presentation = st.sidebar.multiselect("Quelle présentation à analyser ? *", list_presentation, default=list_presentation)

    if not code_presentation:
        st.warning("Veuillez sélectionner une ou plusieurs présentation(s)")
        st.stop()
        
    oneCourse = getOneCourse(dataset_dict, code_module, code_presentation)
    final_df = restructure(oneCourse, 14)
    encoders = cleanAndMap(final_df, encode=False)
    
    if id_student != "":
        final_df = final_df[final_df.index == int(id_student)]
    
    graph = ("Description de l'étudiant", "Description des modules", "Prédiction")
    st.subheader("Explorer")    
    graph_to_show = st.selectbox("", graph)

    if graph_to_show == "Description de l'étudiant":
        
        final_df_2 = final_df[["gender_first", "region_first", "highest_education_first", "imd_band_first", "age_band_first"]]
        final_df_2.rename(columns = {'gender_first':'Genre', 'region_first':'Région', 'Plus haut diplôme':'Genre', 'imd_band_first':'Niveau de pauvreté', 'age_band_first':"Tranche d'âge"}, inplace = True)
        st.dataframe(final_df_2.T)
                
        student_vle = filtre_par_3(dataset_dict["studentVle"], id_student, code_module, code_presentation)
        student_vle.groupby(['code_module']).sum()["sum_click"].plot(kind='bar', stacked=True, figsize=(12,8), title="Nombre de clique total par module")
        plt.ylabel("Nombre de clique total")
        plt.show()
        st.pyplot()
        
    elif graph_to_show == "Description des modules":
        
        dataset_dict["studentVle"].groupby(['code_module']).mean()["sum_click"].plot(kind='bar', stacked=True, figsize=(12,8), title="Nombre de clique moyen par module")
        plt.ylabel("Nombre de clique moyen")
        plt.show()
        st.pyplot()

    elif graph_to_show == "Prédiction":
        
        graph = ('Table des données', '1er Graph', '2eme Graph', '3eme Graph')
        graph_to_show = st.selectbox("Quel graphique à afficher ?", graph)
    
        if graph_to_show == "Table des données":
            st.write(final_df)
        elif graph_to_show == "1er Graph":
            one(final_df)
        elif graph_to_show == "2eme Graph":
            two(final_df)
        elif graph_to_show == "3eme Graph":
            three(final_df)
    
    st.sidebar.caption("Quentin LACHAUSSEE")
    st.sidebar.caption("Adrien GOLEBIEWSKI")
    st.sidebar.caption("Vladimir GUIGNARD")

    st.write('<p style="color:grey;line-height:14px;"><br><br><br><br><br>&nbsp &nbsp &nbsp &nbsp.__(.)< (COIN COIN) <br>&nbsp &nbsp &nbsp &nbsp \___)<br>~~~~~~~~~~~~~~~~~~', unsafe_allow_html=True)



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

# %%
