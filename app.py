import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn probability.")

# User inputs
tenure = st.number_input("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)
senior_citizen = st.selectbox("Senior Citizen", [0, 1])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])

# Build input dataframe
input_data = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'PaperlessBilling_Yes': [1 if paperless == "Yes" else 0],
    'OnlineSecurity_Yes': [1 if online_security == "Yes" else 0],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
})

# Load feature columns
feature_columns = joblib.load("feature_columns.pkl")

# Add missing columns
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure correct column order
input_data = input_data[feature_columns]

st.markdown("---")

if st.button("🔍 Predict Churn"):

    # Scale input INSIDE button
    input_scaled = scaler.transform(input_data)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader(f"Churn Probability: {probability:.2%}")

    if probability >= 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")
