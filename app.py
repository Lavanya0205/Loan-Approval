import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(r"C:\Users\chopr\OneDrive\Desktop\Loan Approval\loan_approval_model_v2.pkl")

st.title("Loan Approval Prediction")

# Collect all required inputs
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)

no_of_dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Create dataframe with same columns as training
input_data = pd.DataFrame({
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'no_of_dependents': [no_of_dependents],
    'education': [education],
    'self_employed': [self_employed],
    'residential_assets_value': [residential_assets_value],
    'commercial_assets_value': [commercial_assets_value],
    'luxury_assets_value': [luxury_assets_value],
    'bank_asset_value': [bank_asset_value]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

