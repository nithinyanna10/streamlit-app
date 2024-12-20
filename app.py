import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load("best_xgb_model_with_scaler.pkl")

# Streamlit app UI
st.title("Loan Default Prediction")

# Define input fields with Streamlit widgets
st.header("Enter the following details:")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length (in years)", min_value=0, max_value=50, value=5)
loan_grade = st.number_input("Loan Grade", min_value=1, max_value=10, value=5)
loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=5.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=10.0)
cb_person_default_on_file = st.selectbox("Default on File (0=No, 1=Yes)", options=[0, 1], index=0)
cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, value=5)

# Create a DataFrame with the input data
input_data = pd.DataFrame([[person_age, person_income, person_emp_length, loan_grade,
                            loan_amnt, loan_int_rate, loan_percent_income,
                            cb_person_default_on_file, cb_person_cred_hist_length]],
                          columns=['person_age', 'person_income', 'person_emp_length', 'loan_grade',
                                   'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                                   'cb_person_default_on_file', 'cb_person_cred_hist_length'])

# Display the input data (optional, for debugging purposes)
st.write("Input Data:")
st.write(input_data)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: **no loan**")
    else:
        st.write("Prediction: **loan grants**")
