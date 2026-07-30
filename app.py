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
# Page config (must be the first Streamlit command)
# ======================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================
# Custom styling
# ======================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #f59e0b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .result-card-churn {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
    }
    .result-card-stay {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-title-churn { color: #b91c1c; }
    .result-title-stay { color: #15803d; }
    .result-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
    }
    .vote-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .vote-churn {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .vote-stay {
        background-color: #dcfce7;
        color: #15803d;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ======================================
# Load saved artifacts (cached so they only load once per session)
# ======================================

@st.cache_resource
def load_artifacts():
    encoders = joblib.load("model/label_encoders.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    
    log_reg = joblib.load("model/logistic_regression.pkl")
    
    # Compatibility fix for newer scikit-learn versions
    if not hasattr(log_reg, 'multi_class'):
        log_reg.multi_class = 'auto'
        
    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
    }
    return encoders, scaler, feature_columns, models


encoders, scaler, feature_columns, models = load_artifacts()

# ======================================
# Header
# ======================================

st.markdown('<div class="main-header">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enter customer details in the sidebar, then click Predict to see the churn risk.</div>',
    unsafe_allow_html=True,
)

# ======================================
# Sidebar — input form, grouped into logical sections
# ======================================

with st.sidebar:
    st.header("Customer Details")

    with st.form("churn_form"):
        st.subheader("👤 Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

        st.subheader("📄 Account Info")
        tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=0.5)

        st.subheader("📡 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# ======================================
# Prediction
# ======================================

if submitted:
    raw_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": "No",              # <-- Hidden from UI, hardcoded default
        "Dependents": "No",           # <-- Hidden from UI, hardcoded default
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
        "PaperlessBilling": "Yes",    # <-- Hidden from UI, hardcoded default
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    # Apply the same label encoders used during training
    for col, encoder in encoders.items():
        if col == "Churn":
            continue
        # Only encode columns that actually exist in the current raw_input
        if col in raw_input.columns:
            raw_input[col] = encoder.transform(raw_input[col])

    # Reorder columns to match training order, then scale
    raw_input = raw_input[feature_columns]
    scaled_input = pd.DataFrame(scaler.transform(raw_input), columns=feature_columns)

    # Get each model's prediction and probability (0 = No churn, 1 = Churn)
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        pred = model.predict(scaled_input)[0]
        predictions[name] = pred
        if hasattr(model, "predict_proba"):
            probabilities[name] = model.predict_proba(scaled_input)[0][1]

    # Majority vote across the three models
    votes = list(predictions.values())
    final_prediction = 1 if sum(votes) >= 2 else 0

    # Average predicted churn probability across models, for the risk gauge
    avg_probability = np.mean(list(probabilities.values())) if probabilities else None

    # ======================================
    # Result card
    # ======================================

    left, right = st.columns([1.3, 1])

    with left:
        if final_prediction == 1:
            st.markdown("""
            <div class="result-card result-card-churn">
                <div class="result-title result-title-churn">⚠️ Likely to CHURN</div>
                <div class="result-subtitle">This customer shows a high risk of leaving. Consider a retention offer.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-card-stay">
                <div class="result-title result-title-stay">✅ Likely to STAY</div>
                <div class="result-subtitle">This customer shows a low risk of churning.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Individual model votes:**")
        badges_html = ""
        for name, pred in predictions.items():
            css_class = "vote-churn" if pred == 1 else "vote-stay"
            label = "Churn" if pred == 1 else "Stay"
            badges_html += f'<span class="vote-badge {css_class}">{name}: {label}</span>'
        st.markdown(badges_html, unsafe_allow_html=True)

    with right:
        if avg_probability is not None:
            st.metric("Average Churn Probability", f"{avg_probability * 100:.1f}%")
            st.progress(float(avg_probability))
        st.metric("Tenure", f"{tenure} months")
        st.metric("Monthly Charges", f"${monthly_charges:,.2f}")

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict** to see the result.")