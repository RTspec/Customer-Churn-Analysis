"""
Customer Churn Prediction — Streamlit App

The user only fills in customer details. Behind the scenes, all three
trained models (Logistic Regression, Decision Tree, Random Forest) each
make a prediction, and the final result shown is a MAJORITY VOTE across
the three — the user never sees or picks individual models.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================
# Load saved artifacts (cached so they only load once per session)
# ======================================

@st.cache_resource
def load_artifacts():
    encoders = joblib.load("model/label_encoders.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
    }
    return encoders, scaler, feature_columns, models


encoders, scaler, feature_columns, models = load_artifacts()

st.title("Customer Churn Prediction")
st.write("Enter customer details below to predict whether they are likely to churn.")

# ======================================
# Input form
# ======================================

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=0.5)

    submitted = st.form_submit_button("Predict")

# ======================================
# Prediction
# ======================================

if submitted:
    # Build a single-row dataframe matching the raw column names used during training
    raw_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    # Apply the same label encoders used during training
    for col, encoder in encoders.items():
        if col == "Churn":
            continue
        raw_input[col] = encoder.transform(raw_input[col])

    # Reorder columns to match training order, then scale
    raw_input = raw_input[feature_columns]
    scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=feature_columns)

    # Get each model's prediction (0 = No churn, 1 = Churn)
    predictions = {}
    for name, model in models.items():
        pred = model.predict(scaled_input)[0]
        predictions[name] = pred

    # Majority vote across the three models
    votes = list(predictions.values())
    final_prediction = 1 if sum(votes) >= 2 else 0

    st.subheader("Result")
    if final_prediction == 1:
        st.error("This customer is likely to CHURN.")
    else:
        st.success("This customer is likely to STAY.")

    with st.expander("See individual model predictions (for reference)"):
        for name, pred in predictions.items():
            label = "Churn" if pred == 1 else "Stay"
            st.write(f"{name}: {label}")
