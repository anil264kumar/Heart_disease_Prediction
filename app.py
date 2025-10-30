import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing

from models import scaler, encoder, svm_model, rf_model, xgb_model
from preprocessing import preprocess_input

st.title("Heart Disease Prediction")

tabs = st.tabs([
   
    "Single Prediction",
    "Results Visualization",
    "Model Comparison",    
    "Bulk Predict (CSV)",
])


# Single Prediction Tab
with tabs[0]:
    st.header("Single Patient Prediction")
    age = st.number_input("Age (In Years)", value=40, min_value=1, max_value=120)
    restingBP = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=50, max_value=250)
    serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=100, max_value=600)
    maxheartrate = st.number_input("Max Heart Rate Achieved", value=150, min_value=60, max_value=220)
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, min_value=0.0, max_value=10.0)
    chestpain = st.selectbox("Chest Pain Type", list(preprocessing.chestpain_map.keys()))
    restingrelectro = st.selectbox("Resting Electrocardiogram", list(preprocessing.restingrelectro_map.keys()))
    slope = st.selectbox("Slope of Peak Exercise ST Segment", list(preprocessing.slope_map.keys()))
    noofmajorvessels = st.selectbox("Number of Major Vessels Colored", list(preprocessing.noofmajorvessels_map.keys()))
    gender = st.radio("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
    exerciseangia = st.radio("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    if st.button("Predict"):
        input_df = pd.DataFrame({
            'age': [age], 'restingBP': [restingBP], 'serumcholestrol': [serumcholestrol],
            'maxheartrate': [maxheartrate], 'oldpeak': [oldpeak], 'chestpain': [chestpain],
            'restingrelectro': [restingrelectro], 'slope': [slope], 'noofmajorvessels': [noofmajorvessels],
            'gender': [gender], 'fastingbloodsugar': [fastingbloodsugar], 'exerciseangia': [exerciseangia]
        })
        features = preprocess_input(input_df, scaler, encoder)
        prediction = xgb_model.predict(features)[0]
        proba = xgb_model.predict_proba(features)[0][prediction]
        result = "Presence of Heart Disease" if prediction == 1 else "No Heart Disease"
        st.success(f"Prediction: {result}")
        st.session_state['pred_result'] = result
        st.session_state['pred_proba'] = proba



# Results Visualization Tab
with tabs[1]:
    st.header("Prediction Results Visualization")
    if 'pred_result' in st.session_state:
        fig, ax = plt.subplots()
        labels = ['Predicted probability', 'Model accuracy']
        values = [
            st.session_state['pred_proba'],
            0.985  # Replace with actual accuracy if available
        ]
        ax.bar(labels, values, color=['#FF6347', '#4682B4'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f"{st.session_state['pred_result']}")
        st.pyplot(fig)
    else:
        st.info("Make a prediction first in the Single Prediction tab.")



# Model Comparison Tab
with tabs[2]:
    st.header("Model Comparison")
    # Same inputs as Single Prediction
    age = st.number_input("Age (In Years)", value=40, min_value=1, max_value=120, key="comp_age")
    restingBP = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=50, max_value=250, key="comp_restingbp")
    serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=100, max_value=600, key="comp_chol")
    maxheartrate = st.number_input("Max Heart Rate Achieved", value=150, min_value=60, max_value=220, key="comp_hr")
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, min_value=0.0, max_value=10.0, key="comp_oldpeak")
    chestpain = st.selectbox("Chest Pain Type", list(preprocessing.chestpain_map.keys()), key="comp_cp")
    restingrelectro = st.selectbox("Resting Electrocardiogram", list(preprocessing.restingrelectro_map.keys()), key="comp_re")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", list(preprocessing.slope_map.keys()), key="comp_slope")
    noofmajorvessels = st.selectbox("Number of Major Vessels Colored", list(preprocessing.noofmajorvessels_map.keys()), key="comp_vessels")
    gender = st.radio("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], key="comp_gender")[1]
    fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0], key="comp_fbs")[1]
    exerciseangia = st.radio("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="comp_ea")[1]

    if st.button("Compare Models"):
        input_df = pd.DataFrame({
            'age': [age], 'restingBP': [restingBP], 'serumcholestrol': [serumcholestrol],
            'maxheartrate': [maxheartrate], 'oldpeak': [oldpeak], 'chestpain': [chestpain],
            'restingrelectro': [restingrelectro], 'slope': [slope], 'noofmajorvessels': [noofmajorvessels],
            'gender': [gender], 'fastingbloodsugar': [fastingbloodsugar], 'exerciseangia': [exerciseangia]
        })

        features = preprocess_input(input_df, scaler, encoder)

        preds = {
            'SVM': svm_model.predict(features)[0],
            'Random Forest': rf_model.predict(features)[0],
            'XGBoost': xgb_model.predict(features)[0],
        }

        def label(p): return "Presence of Heart Disease" if p == 1 else "No Heart Disease"

        results = pd.DataFrame({
            'Model': list(preds.keys()),
            'Prediction': [label(pred) for pred in preds.values()]
        })

        st.table(results)


# Bulk Predict Tab
with tabs[3]:
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        features = preprocess_input(df, scaler, encoder)

        if st.button("Bulk Predict"):
            predictions = xgb_model.predict(features)
            predicted_labels = ["Presence of Heart Disease" if p == 1 else "No Heart Disease" for p in predictions]
            df['Prediction'] = predicted_labels
            probas = xgb_model.predict_proba(features)
            df['Prob_HeartDisease'] = probas[:, 1]

            st.success(f"Predicted {len(df)} records.")
            st.dataframe(df[['Prediction', 'Prob_HeartDisease']])

            output_csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=output_csv,
                file_name="heart_disease_predictions.csv",
                mime="text/csv"
            )
