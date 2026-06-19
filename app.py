import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter Applicant Details")

# Numeric Inputs
age = st.number_input("Age", min_value=21, max_value=75, value=30)

gender = st.number_input("Gender (Female=0, Male=1, Other=2)",
                         min_value=0, max_value=2, value=1)

marital_status = st.number_input(
    "Marital Status (Divorced=0, Married=1, Single=2)",
    min_value=0, max_value=2, value=1
)

education_level = st.number_input(
    "Education Level (Bachelor=0, High School=1, Master's=2, Other=3, PhD=4)",
    min_value=0, max_value=4, value=0
)

annual_income = st.number_input("Annual Income", value=50000.0)

employment_status = st.number_input(
    "Employment Status (Employed=0, Retired=1, Self-Employed=2, Unemployed=3)",
    min_value=0, max_value=3, value=0
)

debt_to_income_ratio = st.number_input(
    "Debt to Income Ratio", value=0.20, format="%.3f"
)

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=900, value=700
)

loan_amount = st.number_input("Loan Amount", value=10000.0)

loan_purpose = st.number_input(
    "Loan Purpose (0-6)", min_value=0, max_value=6, value=0
)

interest_rate = st.number_input("Interest Rate", value=10.0)

loan_term = st.number_input("Loan Term", value=36)

installment = st.number_input("Installment", value=300.0)

num_of_open_accounts = st.number_input(
    "Number of Open Accounts", value=5
)

total_credit_limit = st.number_input(
    "Total Credit Limit", value=40000.0
)

current_balance = st.number_input(
    "Current Balance", value=15000.0
)

delinquency_history = st.number_input(
    "Delinquency History", value=1
)

public_records = st.number_input(
    "Public Records", value=0
)

num_of_delinquencies = st.number_input(
    "Number of Delinquencies", value=1
)

credit_utilization = st.number_input(
    "Credit Utilization", value=0.30, format="%.3f"
)

loan_income_ratio = st.number_input(
    "Loan Income Ratio", value=0.20, format="%.3f"
)

# Predict Button
if st.button("Predict"):

    input_data = np.array([[
        age,
        gender,
        marital_status,
        education_level,
        annual_income,
        employment_status,
        debt_to_income_ratio,
        credit_score,
        loan_amount,
        loan_purpose,
        interest_rate,
        loan_term,
        installment,
        num_of_open_accounts,
        total_credit_limit,
        current_balance,
        delinquency_history,
        public_records,
        num_of_delinquencies,
        credit_utilization,
        loan_income_ratio
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Likely To Be Repaid")
    else:
        st.error("❌ High Risk of Default")