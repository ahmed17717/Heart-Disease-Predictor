import numpy as np
import pickle
import streamlit as st
import os

# Load the saved model
model_path = os.path.join('models', 'heart_disease_model.sav')
model = pickle.load(open(model_path, 'rb'))

# Prediction function
def heart_disease_prediction(input_data):
    input_np = np.asarray(input_data, dtype=float)
    input_reshaped = input_np.reshape(1, -1)
    prediction = model.predict(input_reshaped)
    
    return prediction[0]

# App layout
def main():
    st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")
    st.title("🫀 Heart Disease Risk Predictor")
    st.markdown("---")
    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)
        sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
        trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=250)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])
        thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250)
        exang = st.selectbox('Exercise-induced Angina?', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input('ST Depression (Oldpeak)', step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2])
        ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3, 4])
        thal = st.number_input('Thalassemia', min_value=0, step=1)


    if st.button("💡 Predict Heart Disease Risk"):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]

        result = heart_disease_prediction(input_data)

        if result == 0:
            st.success("✅ The person doesn't have a heart disease")
        else:
            st.error("⚠️ The person has a heart disease")

    st.markdown("---")
    st.caption("Built by Ahmed Ragab")

if __name__ == '__main__':
    main()
