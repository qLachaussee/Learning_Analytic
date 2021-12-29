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
from itertools import cycle, islice
import time
from zipfile import ZipFile

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

id_student = pickle.load(open( "id_student_petit.p", "rb" ) )
users = {str(id):"mdp" for id in id_student.unique()}

@st.experimental_memo(suppress_st_warning=True, show_spinner=False)
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
            
    status_markdown = st.sidebar.markdown('<p style="font-size:16px;color:blue">Downloading the dataset...</p>', unsafe_allow_html=True)
    status_text = st.sidebar.text("courses.csv")
    progress_bar = st.sidebar.progress(0)
    course = dataset_dict['courses']
    course = course[filterByCode(course)]
    module_presentation_length = course['module_presentation_length'].values[0]
    
    # studentAssessment -> complete studentAssessments avec les infos assessments
    status_text.text("studentAssessments.csv")
    progress_bar.progress(20)
    assessments = dataset_dict['assessments']
    assessments.loc[ (assessments['assessment_type'] == 'Exam') & \
        filterByCode(assessments), \
        'date'] = module_presentation_length
    assessments = assessments[filterByCode(assessments)]
    studentAssessment = pd.merge(dataset_dict['studentAssessment'], assessments, \
                                 how='inner', on='id_assessment')

    # studentInfo -> complete studentInfo avec studentRegistration
    status_text.text("studentInfo.csv")
    progress_bar.progress(40)
    studentRegistration = dataset_dict['studentRegistration']
    studentRegistration.loc[studentRegistration['date_unregistration'].isna() & \
        filterByCode(studentRegistration), \
        'date_unregistration'] = module_presentation_length
    studentRegistration = studentRegistration[ \
                           filterByCode(studentRegistration)]
    studentInfo = pd.merge(dataset_dict['studentInfo'], studentRegistration, \
                           how='inner', on=['id_student', 'code_module', 'code_presentation'])

    # studentVle = complete vle avec studentVle
    status_text.text("studentVle.csv")
    progress_bar.progress(60)
    vle = dataset_dict['vle']
    vle = vle[filterByCode(vle)]
    studentVle = pd.merge(dataset_dict['studentVle'], vle, \
                          how='inner', on=['id_site', 'code_module', 'code_presentation'])
    del studentVle['id_site']
    
    # complete avec studentVle avec studentInfo, complete studentAssessment avec studentInfo
    # puis ajoute studentVle avec l'étudiantAssessment, trié par date
    status_text.text("studentAssessment.csv")
    progress_bar.progress(80)
    studentVleInfo = pd.merge(studentVle, studentInfo, \
                              how='inner', on=['id_student', 'code_module', 'code_presentation'])
    studentAssessmentInfo = pd.merge(studentAssessment, studentInfo, \
                              how='inner', on=['id_student', 'code_presentation', 'code_module'])
    
    combined_df = studentAssessmentInfo.append(studentVleInfo)
    del combined_df['code_module']
    del combined_df['code_presentation']
    del combined_df['id_assessment']
    combined_df = combined_df.sort_values(by=['date'])

    status_markdown.text("")
    status_text.text("")
    progress_bar.progress(100)
    status_markdown = st.sidebar.markdown('<p style="font-size:16px;color:red">Dataset downloaded !</p>', unsafe_allow_html=True)

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
            '10-20%': 15,
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

def filtre_par_3(table, code_module, code_presentation, id_student=None):
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
    if id_student == None:
        return table[filtre]
    else:
        return table[(table["id_student"] == int(id_student)) & (filtre)]

def one(df):
    plt.figure(figsize=(10,8))
    df['final_result_first'].value_counts(dropna=False).plot(kind='bar')
    plt.ylabel("Nombre d'étudiant")
    plt.xlabel('Résultat')
    st.pyplot()
 
def two(df):
    counts = df['imd_band_first'].value_counts(dropna=False)
    plt.figure(figsize=(10,10))
    plt.pie(counts, labels=counts.index, colors=['green', 'blue', 'red'])
    plt.title('Pie chart showing counts for\nstudentInfo imd_band categories')
    st.pyplot()
 
def three(df):
    sns.set()
    df.groupby(['imd_band_first','final_result_first']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,8))
    st.pyplot()
 
def main():
    
    st.set_page_config(page_title="Meilleur site", page_icon=":mortar_board:")
    st.header("Projet Learning Analityc")    
    
    dataset_dict = st.session_state.data

    id_student = int(st.session_state.id_student)
    st.sidebar.markdown(f"<p style='text-align:center; font-size:1.8rem;'><b>Bonjour n°{id_student}</b></p>", unsafe_allow_html=True)
    
    st.balloons()

    student_registration = dataset_dict["studentRegistration"][dataset_dict["studentRegistration"]["id_student"] == id_student]
    
    all_module = student_registration["code_module"].unique()
    code_module = st.sidebar.multiselect("Quels modules à analyser ?", all_module, default=all_module)
    
    if not code_module:
        st.warning("Veuillez sélectionner un ou plusieurs module(s)")
        st.stop()
    
    all_presentation = code_presentation = student_registration[student_registration["code_module"].isin(code_module)]["code_presentation"].unique()
    code_presentation = st.sidebar.multiselect("Quelle présentation à analyser ?", all_presentation, default=all_presentation)

    if not code_presentation:
        st.warning("Veuillez sélectionner une ou plusieurs présentation(s)")
        st.stop()

    # df_all = getOneCourse(dataset_dict, all_module, all_presentation)
    # df_all = restructure(df_all, 14)

    df_filtered_MP = getOneCourse(dataset_dict, code_module, code_presentation)
    df_filtered_MP = restructure(df_filtered_MP, 14)
    df_filtered_MPS = df_filtered_MP[df_filtered_MP.index == id_student]

    # encoders = cleanAndMap(df_filtered_MPS, encode=False)

    graph = ("Description de l'étudiant", "Description des modules", "Prédiction")
    st.subheader("Explorer")    
    graph_to_show = st.selectbox("", graph)

    if graph_to_show == "Description de l'étudiant":
        
        df_filtered_MPS_desc = df_filtered_MPS[["gender_first", "region_first", "highest_education_first", "imd_band_first", "age_band_first", "final_result_first"]].copy()
        df_filtered_MPS_desc.rename(columns = {'gender_first':'Genre', 'region_first':'Région', 'highest_education_first':'Plus haut diplôme', 'imd_band_first':'Niveau de pauvreté', 'age_band_first':"Tranche d'âge", "final_result_first": "Résultat final"}, inplace = True)
        st.dataframe(df_filtered_MPS_desc.T)
        
        with st.expander("Nombre de clique moyen :"):    

            plt.subplot(1,2,1)
            student_vle_filtered_MP = filtre_par_3(dataset_dict["studentVle"], code_module, code_presentation)
            df = student_vle_filtered_MP.groupby(['code_module']).mean()["sum_click"]
            df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
            plt.ylabel("Nombre de clique moyen")
            plt.xlabel("")
            plt.title("De la promotion")
            plt.ylim([0,10])

            plt.subplot(1,2,2)
            student_vle_filtered_MPS = filtre_par_3(dataset_dict["studentVle"], code_module, code_presentation, id_student=id_student)
            df = student_vle_filtered_MPS.groupby(['code_module']).mean()["sum_click"]
            df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
            plt.xlabel("")
            plt.title("De l'étudiant")
            plt.ylim([0,10])

            plt.tight_layout()
            st.pyplot()

        with st.expander("Note moyenne par module :"):    

            student_assessment = dataset_dict["assessments"].merge(dataset_dict["studentAssessment"])
            student_assessment_filtered_MP = filtre_par_3(student_assessment, code_module, code_presentation)
            student_assessment_filtered_MPS = filtre_par_3(student_assessment, code_module, code_presentation, id_student=id_student)
            
            plt.subplot(1,2,1)
            df = student_assessment_filtered_MP.groupby(['code_module']).mean()["score"]
            df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
            plt.ylabel("Note moyenne")
            plt.xlabel("")
            plt.title("De la promotion")
            plt.ylim([0,100])

            plt.subplot(1,2,2)
            df = student_assessment_filtered_MPS.groupby(['code_module']).mean()["score"]
            df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
            plt.xlabel("")
            plt.title("De l'étudiant")
            plt.ylim([0,100])

            st.pyplot()

        with st.expander("Note moyenne par module selon la date :"):    
            
            list_presentation = student_assessment_filtered_MP.code_presentation.sort_values().unique()
            i = 1
            for prez in list_presentation:
                plt.subplot(len(list_presentation), 2, i)
                df = student_assessment_filtered_MP[student_assessment_filtered_MP.code_presentation == prez].groupby(['code_module']).mean()["score"]
                df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
                plt.ylabel(prez.replace("B", " Février").replace("J", " Octobre"))
                plt.xlabel("")
                plt.title("De la promotion")
                plt.ylim([0,100])

                plt.subplot(len(list_presentation), 2, i+1)
                df = student_assessment_filtered_MPS[student_assessment_filtered_MPS.code_presentation == prez].groupby(['code_module']).mean()["score"]
                df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
                plt.title("De l'étudiant")
                plt.xlabel("")
                plt.ylim([0,100])

                i+=2

            plt.tight_layout()
            st.pyplot()
        
        with st.expander("Note moyenne par module selon le type d'évaluation :"):    
            
            list_assessment = student_assessment_filtered_MP.assessment_type.unique()
            i = 1
            for asses in list_assessment:
                plt.subplot(len(list_assessment), 2, i)
                df = student_assessment_filtered_MP[student_assessment_filtered_MP.assessment_type == asses].groupby(['code_module']).mean()["score"]
                df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
                plt.ylabel(asses.replace("B", " Février").replace("J", " Octobre"))
                plt.xlabel("")
                plt.title("De la promotion")
                plt.ylim([0,100])

                plt.subplot(len(list_assessment), 2, i+1)
                df = student_assessment_filtered_MPS[student_assessment_filtered_MPS.assessment_type == asses].groupby(['code_module']).mean()["score"]
                df.plot(kind='bar', stacked=True, figsize=(12,8), color=list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df))))
                plt.title("De l'étudiant")
                plt.xlabel("")
                plt.ylim([0,100])

                i+=2

            plt.tight_layout()
            st.pyplot()

        with st.expander("Résultat final par niveau de pauvreté :"):    
        #     plt.subplot(1,2,1)
        #     counts = df_filtered_MP['imd_band_first'].value_counts(dropna=False)
        #     plt.pie(counts, labels=counts.index, colors=['green', 'blue', 'red'])
        #     plt.title('Proportion du niveau de pauvreté')

        #     plt.subplot(1,2,2)
        #     counts = df_filtered_MPS['imd_band_first'].value_counts(dropna=False)
        #     plt.pie(counts, labels=counts.index, colors=['green', 'blue', 'red'])
        #     plt.title("Proportion du niveau de pauvreté de l'étudiant")
            
        #     plt.tight_layout()
        #     st.pyplot()

            df_filtered_MP.groupby(['imd_band_first','final_result_first']).size().unstack().plot(kind='bar', stacked=True, figsize=(12,8))
            plt.xlabel("Niveau de pauvreté")
            plt.ylabel("Nombre d'étudiant")
            plt.title("Résultat final de la promotion")
            st.pyplot()
            st.write("Résultat final de l'étudiant")
            st.write(df_filtered_MPS_desc[["Niveau de pauvreté", "Résultat final"]].reset_index(drop=True))
    
    if graph_to_show == "Description des modules":
        
        student_vle_filtered_MP = filtre_par_3(dataset_dict["studentVle"], code_module, code_presentation)
        student_vle_filtered_MP.groupby(['code_module']).mean()["sum_click"].plot(kind='bar', stacked=True, figsize=(12,8))
        plt.title("Nb de clique moyen par module")
        plt.ylabel("Nb de clique moyen")
        st.pyplot()

        with st.expander("Note moyenne par module selon la date :"):    
            
            assessment_filtered_MP = filtre_par_3(dataset_dict["assessments"], code_module, code_presentation)
            student_assessment_filtered_MP = assessment_filtered_MP.merge(dataset_dict["studentAssessment"])
            for i, prez in enumerate(np.sort(student_assessment_filtered_MP.code_presentation.unique())):
                plt.subplot(2, 2, i+1)
                student_assessment_filtered_MP[student_assessment_filtered_MP.code_presentation == prez].groupby(['code_module']).mean()["score"].plot(kind='bar', stacked=True, figsize=(12,8))
                
                plt.title(prez.replace("B", " Février").replace("J", " Octobre"))
                plt.xlabel("")
                plt.ylim([0,100])

            plt.tight_layout()
            st.pyplot()
        
        with st.expander("Note moyenne par module selon le type d'évaluation :"):    
            
            assessment_filtered_MP = filtre_par_3(dataset_dict["assessments"], code_module, code_presentation)
            student_assessment_filtered_MP = assessment_filtered_MP.merge(dataset_dict["studentAssessment"])
            for i, prez in enumerate(student_assessment_filtered_MP.assessment_type.unique()):
                plt.subplot(2, 2, i+1)
                student_assessment_filtered_MP[student_assessment_filtered_MP.assessment_type == prez].groupby(['code_module']).mean()["score"].plot(kind='bar', stacked=True, figsize=(12,8))
                
                plt.title(prez.replace("B", " Février").replace("J", " Octobre"))
                plt.xlabel("")
                plt.ylim([0,100])

            plt.tight_layout()
            st.pyplot()

    if graph_to_show == "Prédiction":

        with st.expander("Table des données :"):    
            # st.dataframe(final_df)
            st.write('<p style="color:grey;line-height:14px;"><br><br><br><br><br>&nbsp &nbsp &nbsp &nbsp.__(.)< (COIN COIN) <br>&nbsp &nbsp &nbsp &nbsp \___)<br>~~~~~~~~~~~~~~~~~~', unsafe_allow_html=True)
    
    st.sidebar.caption("Quentin LACHAUSSEE")
    st.sidebar.caption("Adrien GOLEBIEWSKI")
    st.sidebar.caption("Vladimir GUIGNARD")




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
                    st.session_state.id_student = my_user
                    with st.spinner("Redirecting to application..."):
                        st.session_state.data = pickle.load(open("dataset_dict_petit.p", "rb" ))
                        time.sleep(1)
                        print("okkkkkk")
                        st.experimental_rerun()
                        # st.experimental_rerun()
                else:
                    st.error("Invalid User/Password. Try again. :no_entry:")

    else:
        main()

