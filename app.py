import streamlit as st
import joblib
import pickle
import joblib

encoder = joblib.load("encoder.pkl")
model = joblib.load("churn_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Churn Prediction System")

st.markdown("""
Predict whether a customer is likely to churn based on their information.
""")

# Sidebar
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    max_value=100,
    value=1
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

st.write("### Customer Details")

col1, col2 = st.columns(2)

with col1:
    st.write("**Gender:**", gender)
    st.write("**Senior Citizen:**", senior)
    st.write("**Partner:**", partner)
    st.write("**Dependents:**", dependents)

with col2:
    st.write("**Tenure:**", tenure)
    st.write("**Monthly Charges:**", monthly)
    st.write("**Total Charges:**", total)
    st.write("**Contract:**", contract)

if st.button("Predict Churn"):
    st.success("Prediction feature will be added after training the model.")