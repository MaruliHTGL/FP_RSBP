import streamlit as st
import numpy as np

import joblib
import os   

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def load_normalizer(normalizer_file):
    loaded_normalizer = joblib.load(open(os.path.join(normalizer_file), 'rb'))
    return loaded_normalizer

def run_ml_app():
    st.subheader("Enter Your Penguin's Measurements")
    bill_length_mm = st.number_input("Bill length in milimeters", 1, 100)
    bill_depth_mm = st.number_input("Bill depth in milimeters", 1, 50)
    flipper_length_mm = st.number_input("Flipper in milimeters", 1, 250)
    body_mass_g = st.number_input("Body mass in grams", 1, 6500)

    st.subheader("Your Penguin's Measurement Input")
    with st.expander("Your Selected Options"):
        result = {
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm':bill_depth_mm,
            'flipper_length_mm':flipper_length_mm,
            'body_mass_g':body_mass_g,
        }
    st.write(result)

    single_array = np.array(list(result.values())).reshape(1, -1)
    st.write("Raw Input Array:")
    st.write(single_array)

    normalizer = load_normalizer("normalize (1).pkl")
    normalized_array = normalizer.transform(single_array)
    st.write("Normalized Input Array:")
    st.write(normalized_array)

    st.subheader('Your Penguin Analysis Result')
    model = load_model("model.pkl")
    prediction = model.predict(normalized_array)

    if prediction == 'Adelie':
        st.success("Your Penguin is Adelie")
    else:
        st.success("Your Penguin is Gentoo")