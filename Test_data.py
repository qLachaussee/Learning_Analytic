#%%

import pandas as pd
import pickle

#%%
student = pd.read_csv("./../Input/studentRegistration.csv")
id_student = student.id_student
id_student.to_pickle("id_student.p")

#%%

path = "./../Input/"
assessments_df         = pd.read_csv(path+'assessments.csv')
courses_df             = pd.read_csv(path+'courses.csv')
studentAssessment_df   = pd.read_csv(path+'studentAssessment.csv')
studentInfo_df         = pd.read_csv(path+'studentInfo.csv')
studentRegistration_df = pd.read_csv(path+'studentRegistration.csv')
studentVle_df          = pd.read_csv(path+'studentVle.csv')
vle_df                 = pd.read_csv(path+'vle.csv')

list_etu = studentVle_df[["code_module", "code_presentation", "id_student"]].drop_duplicates().groupby(["id_student"]).count().sort_values(by="code_module")
etu = list_etu[list_etu.code_module >2].index
etu.to_pickle("id_student_petit.p")

# on stocke les références
dataset_dict = {
    'assessments': assessments_df,
    'courses': courses_df,
    'studentAssessment': studentAssessment_df,
    'studentInfo': studentInfo_df,
    'studentRegistration': studentRegistration_df,
    'studentVle': studentVle_df,
    'vle': vle_df
}
pickle.dump(dataset_dict, open("dataset_dict.p", "wb"))

# on stocke les références
dataset_dict_petit = {
    'assessments': assessments_df,
    'courses': courses_df,
    'studentAssessment': studentAssessment_df,
    'studentInfo': studentInfo_df,
    'studentRegistration': studentRegistration_df,
    'studentVle': studentVle_df[studentVle_df.id_student.isin(etu)],
    'vle': vle_df
}
pickle.dump(dataset_dict_petit, open("dataset_dict_petit.p", "wb"))

#%%
student = pickle.load(open("id_student_petit.p", "rb"))
master = pickle.load(open("master.p", "rb"))
master[master.id_student.isin(student)].id_student.to_csv("list_student.csv", index=False)

#%%
from zipfile import ZipFile
with ZipFile("dataset_dict.zip", 'w') as zip:
    zip.write('dataset_dict.p')
with ZipFile("dataset_dict.zip", 'r') as zip:
    zip.extract('dataset_dict.p')

#%%
# %%
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    
    combined_df = combined_df.sort_values(by=['date'])

    return combined_df

def restructure(oneCourse, days):
    '''
    agréger la séquence de chaque élève pour que chaque élève ne contienne qu'une seule ligne,
    en ne conservant que les données des deux premières semaines
    
    '''
    # del oneCourse['code_module']
    # del oneCourse['code_presentation']
    # del oneCourse['id_assessment']
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
    final_df = first14Days_oneCourse.groupby(['id_student', 'code_module']).agg({
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

dataset_dict = pickle.load(open("dataset_dict.p", "rb" ))
id_student = 80329

student_registration = dataset_dict["studentRegistration"][dataset_dict["studentRegistration"]["id_student"] == id_student]
code_module = student_registration["code_module"].unique()
code_presentation = student_registration[student_registration["code_module"].isin(code_module)]["code_presentation"].unique()

all_module = student_registration["code_module"].unique()
all_presentation = student_registration[student_registration["code_module"].isin(code_module)]["code_presentation"].unique()

df_all = getOneCourse(dataset_dict, all_module, all_presentation)
df_filtered_S = df_all[df_all.index == id_student]

df_filtered_MP = getOneCourse(dataset_dict, code_module, code_presentation)
df_filtered_MP = restructure(df_filtered_MP, 14)
df_filtered_MPS = df_filtered_MP.reset_index()[df_filtered_MP.reset_index().id_student==id_student]

#%%

import pandas as pd
import numpy as np
import pickle 

dataset_dict = pickle.load(open("dataset_dict.p", "rb" ))

# Student Info
student_info = dataset_dict["studentInfo"]
student_info = student_info.query('final_result != "Withdrawn"')

# Données d'évaluation
# Lire uniquement les colonnes utiles à fusionner

assesments = dataset_dict["assessments"][['code_module', 'code_presentation',
                                          'id_assessment', 'assessment_type']]

student_assesments = dataset_dict["studentAssessment"][['id_assessment', 'id_student', 'score']]

merged_assessments = student_assesments.merge(assesments, on='id_assessment')

# Grouper les évaluations par id_étudiant, code_module et code_présentation de façon à ce qu'elles
# aient une corrélation bi-univoque avec la table student_info.

merged_assessments = merged_assessments.groupby(
    ['id_student', 'code_module', 'code_presentation',
     'assessment_type'])['score'].mean().reset_index()

# Créez un tableau croisé dynamique pour fusionner les colonnes de type d'évaluation et de score en deux colonnes (une pour CMA et TMA) sur une ligne unique d'étudiant/cours.
# (une pour les deux CMA et TMA) sur une ligne unique étudiant/cours.

merged_assessments = (merged_assessments.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="assessment_type")['score'].reset_index().rename_axis(None,
                                                                       axis=1))

# Lire uniquement les colonnes pertinentes de la table vle générique.
vle = dataset_dict["vle"][['id_site', 'activity_type']]

# Student data
student_vle = dataset_dict["studentVle"]

# Fusionner les types d'activités sur la VLE des élèves.
merged_vle = student_vle.merge(vle, on='id_site')
# Encore une fois, le groupe d'étudiants/module/présentation sera un à un avec des informations sur les étudiants.
merged_vle = merged_vle.groupby(['id_student', 'code_module', 'code_presentation', 'activity_type'])

# Visites uniques pour chaque type d'activité en comptant les jours uniques.
vle_uniq_visits = merged_vle['date'].count().reset_index()
# Tableau croisé dynamique ici pour transformer la colonne des visites en colonnes individuelles pour
# chaque activité avec des valeurs comme le nombre de visites
vle_uniq_visits = (vle_uniq_visits.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['date'].reset_index().rename_axis(None,
                                                                    axis=1))

#Nombre total d'interactions avec chaque type d'activité
vle_interactions = merged_vle['sum_click'].sum().reset_index()
# Encore un tableau croisé dynamique pour transformer les colonnes d'interactions en colonnes individuelles pour
# chaque activité
vle_interactions = (vle_interactions.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['sum_click'].reset_index().rename_axis(
    None, axis=1))

# Fusionner d'abord les informations sur les étudiants et les évaluations préparées.
master = pd.merge(student_info,
                  merged_assessments,
                  on=['id_student', 'code_module', 'code_presentation'])

# Fusionner sur la table vle_visit en ajoutant un suffixe pour la différencier de la table interactiosns.
master = master.merge(
    vle_uniq_visits.add_suffix('_uniq_visits'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_uniq_visits', 'code_module_uniq_visits',
        'code_presentation_uniq_visits'
    ])

# Fusion sur vle_interactions
master = master.merge(
    vle_interactions.add_suffix('_interactions'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_interactions', 'code_module_interactions',
        'code_presentation_interactions'
    ])

# Suppression des lignes redondantes des fusions suffixées de la table vle
master.drop([
    'id_student_uniq_visits', 'code_module_uniq_visits',
    'code_presentation_uniq_visits', 'id_student_interactions',
    'code_module_interactions', 'code_presentation_interactions'
],
            axis=1,
            inplace=True)

imd_dict = {
    '0-10%': 5,
    '10-20': 15,
    '10-20%': 15,
    '20-30%': 25,
    '30-40%': 35,
    '40-50%': 45,
    '50-60%': 55,
    '60-70%': 65,
    '70-80%': 75,
    '80-90%': 90,
    '90-100%': 95
}
age_dict = {'0-35': 17.5, '35-55': 45, '55<=': 82.5}

master.replace({"age_band": age_dict, "imd_band": imd_dict}, inplace=True)
master.query('final_result != "Withdrawn"', inplace=True)

# master.to_pickle("master.p")

#%%
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Pipeline pour la standardisation du tableau pour les modèles
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_labels(df):
    # Retirer la colonne de résultat final d'une table et retourner les deux colonnes.
    values = df.drop('final_result', axis=1)
    labels = df['final_result'].copy()
    return values, labels

def pipeline(df):
    # Les données VLE sont sélectionnées séparément des autres données numériques car la stratégie de remplissage
    # doit être différente
    vle_types = df.filter(
        regex='_uniq_visits$', axis=1).columns.values.tolist() + df.filter(
            regex='_interactions$', axis=1).columns.values.tolist()
    #toutes les autres colomnes numériques
    other_numeric = [
        'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits',
        'CMA', 'TMA'
    ]

    # Imputers
    # Les données VLE sont remplies de 0 car c'est ce que les données représentent réellement.
    # Si une valeur est NA, cela signifie que l'utilisateur n'a pas interagi avec/visité cette activité.
    df[vle_types] = df[vle_types].fillna(0)
    # Les autres données numériques sont remplies avec la moyenne
    df[other_numeric] = df[other_numeric].fillna(
        df[other_numeric].mean(axis=0)+1)

    # Transformateur de colonnes
    # Oneencoder encode les données catégorielles restantes.
    # StandardEncode met à l'échelle toutes les données numériques
    ct = ColumnTransformer([('cat', OneHotEncoder(), [
        'code_module', 'code_presentation', 'gender', 'region',
        'highest_education'
    ]), ('std_scaler', StandardScaler(), vle_types + other_numeric)],
                           remainder='drop')

    return ct.fit_transform(df)

def prepare_labels(labels):
    # Comme nous ne prévoyons que des succès et des échecs, nous réétiquetons la disctinction en tant que
    # réussite et retrait comme échec
    # Nous utilisons 1 pour représenter la réussite et 0 pour l'échec pour les fonctions de la métrique de notation.
    lab_dict = {'Pass': 1, 'Fail': 0, 'Withdrawn': 0, 'Distinction': 1}
    return labels.replace(lab_dict)

def load_model(file):
    return joblib.load(file)

def load_file(file):
    return pickle.load(open(file, "rb"))

def save_model(file, model):
    joblib.dump(model, file)


def train_k_neighbors_classifier(values, labels):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(values, labels)
    return model

def train_svm_classifier(values, labels):
    from sklearn.svm import SVC
    model = SVC()
    model.fit(values, labels)
    return model

def train_tree_classifier(values, labels):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(values, labels)
    return model

def train_forest_classifier(values, labels):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(values, labels)
    return model

def train_MLP_classifier(values, labels):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier()
    model.fit(values, labels)
    return model

def train_ada_boost_classifier(values, labels):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier()
    model.fit(values, labels)
    return model

def train_bayes_classifier(values, labels):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(values, labels)
    return model


def load_data():
    data = pickle.load(open("master.p", "rb" )) #à ajouter
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


master = pickle.load(open("master.p", "rb" ))

# Crer un  train test split
train_set, test_set = train_test_split(master, test_size=0.2, random_state=20)

# Séparer les étiquettes
train_values, train_labels = split_labels(train_set)
test_values, test_labels = split_labels(test_set)

# Pipeline les données pour préparer la formation et les tests
train_values = pipeline(train_values)
train_labels = prepare_labels(train_labels)
test_values = pipeline(test_values)
test_labels = prepare_labels(test_labels)

#%%

# create contrôle l'exécution de la recherche de grille sur les hyperparamètres.
create = True

# Entrainement classifier
k_neighbors_classifier = train_k_neighbors_classifier(train_values, train_labels)
svm_classifier = train_svm_classifier(train_values, train_labels)
forest_classifier = train_forest_classifier(train_values, train_labels)
tree_classifier = train_tree_classifier(train_values, train_labels)
MLP_classifier = train_MLP_classifier(train_values, train_labels)
ada_boost_classifier = train_ada_boost_classifier(train_values, train_labels)
bayes_classifier = train_bayes_classifier(train_values, train_labels)

#%%
classifiers = [k_neighbors_classifier, svm_classifier, forest_classifier,
               tree_classifier, MLP_classifier, ada_boost_classifier, bayes_classifier]
classifiers = [ada_boost_classifier, bayes_classifier]
# classifiers = [k_neighbors_classifier]
#%%
from time import time

for classifier in classifiers:
    timet = time()
    print(f'\n{str(classifier.__class__).split(".")[-1][:-2]} Cross Validation Scores')
    for test in ['accuracy', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(classifier,
                                train_values,
                                train_labels,
                                scoring=test,
                                cv=5)
        print(test, np.mean(scores))
    print(time() - timet)


#%%
if create:
    # Recherche d'une grille aléatoire pour le réglage des hyperparamètres.
    for classifier in classifiers:
        name = str(classifier.__class__).split(".")[-1][:-2]
        if name == "DecisionTreeClassifier":
            random_grid = {
                'max_features': ['sqrt', 'log2'],
                'max_depth': [None, 20, 40, 60, 80, 100, 120],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4],
            }
        elif name == "RandomForestClassifier":
            random_grid = {
                'n_estimators': [100, 200, 400, 600, 800, 1000, 1200],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [None, 20, 40, 60, 80, 100, 120],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif name == "KNeighborsClassifier":
            random_grid = {
                'n_neighbors': list(range(1, 31)),
                'weights': ['uniform', 'distance']
            }
        elif name == "SVC":
            random_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                'class_weight':['balanced', None]
            }
        elif name == "MLPClassifier":
            random_grid = {
                'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                'activation': ['tanh', 'relu', 'logistic'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0.0001, 0.01, 0.1, 1],
                'learning_rate': ['constant','adaptive']
            }
        elif name == "AdaBoostClassifier":
            random_grid = {
                'n_estimators':[500,1000,2000],
                'learning_rate':[.001,0.01,.1]
            }
        elif name == "GaussianNB":
            random_grid = {
                'var_smoothing': np.logspace(0,-9, num=100)
            }
        else:
            print(name)
            
        print(f'\n{name} Model Creation')
        # Nous réalisons d'abord l'optimisation de l'arbre de décision
        search = RandomizedSearchCV(classifier,
                                    param_distributions=random_grid,
                                    n_iter=75,
                                    cv=5,
                                    n_jobs=-1,
                                    scoring='roc_auc',
                                    random_state=20)
        search.fit(train_values, train_labels)
        # Sauvegarde du meilleur modèle
        classifier = search.best_estimator_
        save_model(f'Best {name}', classifier)

else:
    k_neighbors_classifier = load_model('Best KNeighborsClassifier')
    svm_classifier = load_model('Best SVC')
    forest_classifier = load_model('Best RandomForestClassifier')
    tree_classifier = load_model('Best DecisionTreeClassifier')
    MLP_classifier = load_model('Best MLPClassifier')
    ada_boost_classifier = load_model('Best AdaBoostClassifier')
    bayes_classifier = load_model('Best GaussianNB')

#%%

for classifier in classifiers:
    print(f'\n{str(classifier.__class__).split(".")[-1][:-2]} Test Scores')

    predictions = classifier.predict(test_values)
    predictions_proba = classifier.predict_proba(test_values)[:, 1]

    print('Confusion Matrix', confusion_matrix(test_labels, predictions))
    print('Accuracy', accuracy_score(test_labels, predictions))
    print('Recall', recall_score(test_labels, predictions))
    print('F1', f1_score(test_labels, predictions))
    print('ROC AUC', roc_auc_score(test_labels, predictions_proba))
    plot_confusion_matrix(classifier, test_values, test_labels, display_labels=["Fail", "Pass"], cmap="Blues")
    plt.show()
    plot_roc_curve(classifier, test_values, test_labels)
    plt.show()
    plot_precision_recall_curve(classifier, test_values, test_labels)
    plt.show()

#%%
