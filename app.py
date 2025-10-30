# app.py (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and preprocessors
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature mappings for selectboxes
chestpain_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

restingrelectro_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}

noofmajorvessels_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}

st.title("Heart Disease Prediction")


# Input widgets for numeric features
age = st.number_input("Age (In Years)", value=40, min_value=1, max_value=120)
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=50, max_value=250)
serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=100, max_value=600)
maxheartrate = st.number_input("Max Heart Rate Achieved", value=150, min_value=60, max_value=220)
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, min_value=0.0, max_value=10.0)

# Input widgets for categorical features
chestpain = st.selectbox("Chest Pain Type", list(chestpain_map.keys()))
restingrelectro = st.selectbox("Resting Electrocardiogram", list(restingrelectro_map.keys()))
slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_map.keys()))
noofmajorvessels = st.selectbox("Number of Major Vessels Colored", list(noofmajorvessels_map.keys()))

# Input widgets for binary features
gender = st.radio("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
exerciseangia = st.radio("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

if st.button("Predict"):
    # Map categorical input labels to codes
    chestpain_code = chestpain_map[chestpain]
    restingrelectro_code = restingrelectro_map[restingrelectro]
    slope_code = slope_map[slope]
    noofmajorvessels_code = noofmajorvessels_map[noofmajorvessels]

    # Create DataFrame for preprocessing
    input_df = pd.DataFrame({
        'age': [age],
        'restingBP': [restingBP],
        'serumcholestrol': [serumcholestrol],
        'maxheartrate': [maxheartrate],
        'oldpeak': [oldpeak],
        'chestpain': [chestpain_code],
        'restingrelectro': [restingrelectro_code],
        'slope': [slope_code],
        'noofmajorvessels': [noofmajorvessels_code],
        'gender': [gender],
        'fastingbloodsugar': [fastingbloodsugar],
        'exerciseangia': [exerciseangia]
    })

    # Preprocess input
    numeric_cols = ['age', 'restingBP', 'maxheartrate', 'oldpeak', 'serumcholestrol']
    cat_cols = ['chestpain', 'restingrelectro', 'slope', 'noofmajorvessels']
    binary_cols = ['gender', 'fastingbloodsugar', 'exerciseangia']

    scaled_numeric = scaler.transform(input_df[numeric_cols])
    encoded_cat = encoder.transform(input_df[cat_cols])
    binary_features = input_df[binary_cols].to_numpy()

    # Combine all
    features = np.hstack([scaled_numeric, encoded_cat, binary_features])

    # Predict
    prediction = model.predict(features)[0]

    result = "Presence of Heart Disease" if prediction == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
