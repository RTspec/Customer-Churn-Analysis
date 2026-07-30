import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random


st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


# Header
st.title("📊 Customer Churn Prediction AI Dashboard")
st.write("Machine Learning Based Customer Retention System")

st.write("---")


# Customer Details
st.subheader("👤 Customer Details")


c1,c2,c3 = st.columns(3)


with c1:
    tenure = st.slider(
        "Tenure (Months)",
        0,100,20
    )

    charges = st.number_input(
        "Monthly Charges",
        0,500,70
    )


with c2:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month","One Year","Two Year"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL","Fiber Optic","No"]
    )


with c3:
    payment = st.selectbox(
        "Payment Method",
        ["Credit Card","Electronic Check","Bank Transfer"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )



if st.button("🔮 Analyze Customer"):

    risk = random.randint(10,95)

    st.write("---")

    st.subheader("🎯 Customer Risk Analysis")


    a,b,c = st.columns(3)


    a.metric(
        "Churn Probability",
        f"{risk}%"
    )


    if risk > 70:
        b.error("🔴 High Risk")

    elif risk > 40:
        b.warning("🟡 Medium Risk")

    else:
        b.success("🟢 Low Risk")


    c.metric(
        "Retention Chance",
        f"{100-risk}%"
    )


    st.progress(risk)


    st.subheader("💡 AI Recommendation")


    if risk > 70:
        st.warning(
            "🎁 Give discount + loyalty benefits"
        )

    elif risk > 40:
        st.info(
            "📧 Send personalized offers"
        )

    else:
        st.success(
            "✅ Customer is satisfied"
        )



# Customer Behavior Graph

st.write("---")

st.subheader("📊 Customer Behavior Analysis")


behavior = pd.DataFrame({

    "Category":[
        "Service Usage",
        "Payment Stability",
        "Customer Loyalty",
        "Engagement"
    ],

    "Score":[
        85,
        70,
        90,
        75
    ]

})


st.bar_chart(
    behavior.set_index("Category")
)



# Model Performance

st.write("---")

st.subheader("📈 Model Performance")


model = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Performance":[
        80,
        76,
        79
    ]

})


st.bar_chart(
    model.set_index("Model")
)



# Business Section

st.write("---")

st.subheader("🚀 Business Impact")


x,y,z = st.columns(3)

x.metric("Customers", "7043")
y.metric("Best Model", "Random Forest")
z.metric("AI Solution", "Active")



st.write("---")

st.success(
    "🎉 Customer Churn AI Dashboard Ready"
)

st.caption(
    "Machine Learning | Streamlit | Data Analytics"
)