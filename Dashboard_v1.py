from Dashboard_v2 import *

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import re
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle, islice
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, roc_auc_score, accuracy_score, recall_score, f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

id_student = pickle.load(open( "id_student_petit.p", "rb" ) )
users = {str(id):"mdp" for id in id_student.unique()}

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
