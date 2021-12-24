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

import matplotlib.pyplot as plt
import time
from zipfile import ZipFile

# st.set_option('deprecation.showPyplotGlobalUse', False)

id_student = pickle.load(open( "id_student.p", "rb" ) )
users = {str(id):"20/20" for id in id_student.unique()[:100]}


def main():
    

    st.set_page_config(page_title="Meilleur site", page_icon=":mortar_board:")
    st.header("Projet Learning Analityc")    
    
    dataset_dict = print_load_data()

    id_student = st.session_state.id_student
    st.sidebar.write(f"Bonjour n°{id_student}")
    
    st.balloons()



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
                        time.sleep(1)
                        print("okkkkkk")
                        st.experimental_rerun()
                        # st.experimental_rerun()
                else:
                    st.error("Invalid User/Password. Try again. :no_entry:")

    else:
        main()

# %%
