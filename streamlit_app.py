import streamlit as st
import requests

st.title("🏦 Loan Credo")

st.write("Enter loan details to predict the default risk:")

property_value = st.number_input("Property Value", min_value=0.0)
income = st.number_input("Applicant Income", min_value=0.0)
rate_of_interest = st.number_input("Rate of Interest", min_value=0.0)
LTV = st.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0)
dtir1 = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0)

if st.button("Predict"):
    data = {
        "property_value": property_value,
        "income": income,
        "rate_of_interest": rate_of_interest,
        "LTV": LTV,
        "dtir1": dtir1
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        if prediction == 1:
            st.error("⚠️ High risk of loan default")
        else:
            st.success("✅ Low risk of loan default")

    else:
        st.warning("Error connecting to prediction API.")
