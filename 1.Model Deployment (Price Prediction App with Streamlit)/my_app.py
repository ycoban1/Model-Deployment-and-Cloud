import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("AutoScout ML Deployment Project")
st.text('Check value of your car!')

#image
img = Image.open("car_image.jpg")
st.image(img, width=500)

#sidebar hearder
st.sidebar.header('Car Price Calculator')

# make model
make_model=st.sidebar.selectbox("Make & Model", ["Audi A1", "Audi A3", "Opel Astra", "Opel Corsa", "Opel Insignia", "Renault Clio", "Renault Duster", "Renault Espace"])

# Gearing Type
Gearing_Type=st.sidebar.selectbox("Gearing Type", ["Manual", "Atomatic", "Semi-automatic"])

#Fuel
Fuel=st.sidebar.selectbox("Fuel", ["Benzine", "Diesel","LPG/CNG", "Electric"])

#Age
age = st.sidebar.number_input("Age:",min_value=0)

#Km
km=st.sidebar.number_input("Km:",min_value=0, step=1000)

#hp_kW
hp_kW=st.sidebar.number_input("HP:",min_value=0, step=10)

import pickle
filename = 'autoscout_lasso_model'
model = pickle.load(open(filename, 'rb'))

my_dict = {
    "hp_kW": hp_kW,
    "km": km,
    "age":age,
    "make_model": make_model,
    "Gearing_Type": Gearing_Type,
    "Fuel": Fuel,
}

my_dict=pd.DataFrame.from_dict([my_dict])
my_dict=pd.get_dummies(my_dict)
columns_name=['hp_kW', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
       'make_model_Opel Astra', 'make_model_Opel Corsa',
       'make_model_Opel Insignia', 'make_model_Renault Clio',
       'make_model_Renault Duster', 'make_model_Renault Espace',
       'Gearing_Type_Automatic', 'Gearing_Type_Manual',
       'Gearing_Type_Semi-automatic', 'Fuel_Benzine', 'Fuel_Diesel',
       'Fuel_Electric', 'Fuel_LPG/CNG']
my_dict = my_dict.reindex(columns=columns_name, fill_value=0)


if st.sidebar.button("Check"):
    pred = model.predict(my_dict)
    st.success("The estimated value of sales is {}. ".format(pred))
st.sidebar.info("Please fill all required fields..")