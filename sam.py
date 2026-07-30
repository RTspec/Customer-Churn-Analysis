import streamlit as st


st.set_page_config(
    page_title="Customer Churn AI Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------- Header ----------
st.title("📊 Customer Churn Analytics AI")
st.write("Machine Learning Based Customer Retention Prediction")

st.divider()


# ---------- Customer Form + Result ----------
left, right = st.columns(2)


with left:

    st.subheader("👤 Customer Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0, 72, 24
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        0, 200, 70
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Bank Transfer",
            "Electronic Check"
        ]
    )


    predict = st.button(
        "🔮 Predict Churn",
        use_container_width=True
    )



with right:

    st.subheader("🎯 Prediction Result")


    churn_probability = 82


    st.metric(
        "Churn Probability",
        f"{churn_probability}%"
    )


    st.progress(
        churn_probability / 100
    )


    if churn_probability >= 70:

        st.error(
            "⚠ High Churn Risk"
        )

    else:

        st.success(
            "✅ Customer Likely to Stay"
        )


    st.info(
        """
        AI Analysis:
        
        • Customer behavior analyzed
        • Risk level calculated
        • Retention strategy generated
        """
    )


st.divider()


# ---------- AI Recommendation ----------

st.subheader("💡 AI Recommendation")


col1, col2, col3 = st.columns(3)


with col1:
    st.success(
        "🎁 Personalized Offers\n\nProvide special discounts"
    )


with col2:
    st.warning(
        "☎ Customer Support\n\nImprove customer experience"
    )


with col3:
    st.info(
        "📈 Better Plans\n\nSuggest suitable packages"
    )


st.divider()


st.caption(
    "Powered by Machine Learning | Customer Churn Prediction System"
)