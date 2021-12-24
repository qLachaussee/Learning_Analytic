#%%

import pandas as pd
import pickle

#%%
student = pd.read_csv("./../../Input/studentRegistration.csv")
id_student = student.id_student
id_student.to_pickle("id_student.p")
# %%
id_student = pickle.load(open( "id_student.p", "rb" ) )
print(id_student)

#%%
path = "./../../Input/"
assessments_df         = pd.read_csv(path+'assessments.csv')
courses_df             = pd.read_csv(path+'courses.csv')
studentAssessment_df   = pd.read_csv(path+'studentAssessment.csv')
studentInfo_df         = pd.read_csv(path+'studentInfo.csv')
studentRegistration_df = pd.read_csv(path+'studentRegistration.csv')
studentVle_df          = pd.read_csv(path+'studentVle.csv')
vle_df                 = pd.read_csv(path+'vle.csv')

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
# %%
from zipfile import ZipFile
with ZipFile("dataset_dict.zip", 'r') as zip:
    zip.extract('dataset_dict.p')
# %%
