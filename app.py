"""

Telco Customer Churn Predictor - Streamlit App

Run: streamlit run app.py

"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

model = joblib.load("models/churn_model.pkl")

st.title("📉 Telco Customer Churn Predictor")
st.write("Enter a customer's details to estimate their churn probability.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges

if st.button("Predict Churn"):
    service_cols_values = [
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies
    ]
    num_services = sum(v == "Yes" for v in service_cols_values)

    if tenure <= 12:
        tenure_group = "0-1yr"
    elif tenure <= 24:
        tenure_group = "1-2yr"
    elif tenure <= 48:
        tenure_group = "2-4yr"
    elif tenure <= 60:
        tenure_group = "4-5yr"
    else:
        tenure_group = "5-6yr"

    input_df = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges, "tenure_group": tenure_group,
        "avg_charge_per_tenure": total_charges / max(tenure, 1),
        "num_services": num_services,
    }])

    proba = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{proba:.1%}")

    if proba > 0.5:
        st.error("⚠️ High risk of churn — consider a retention offer.")
    else:
        st.success("✅ Low risk of churn.")