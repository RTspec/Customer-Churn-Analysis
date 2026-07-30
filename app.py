import streamlit as st


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ================= CSS =================

st.markdown("""
<style>

body {
    background-color:#f5f7fb;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:800;
    color:#1f4e79;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#666;
}


.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.12);
}


.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)



# ================= HEADER =================


st.markdown(
    "<div class='title'>📊 Customer Churn Prediction System</div>",
    unsafe_allow_html=True
)


st.markdown(
    "<div class='subtitle'>AI Powered Customer Retention Dashboard</div>",
    unsafe_allow_html=True
)


st.write("")



# ================= IMAGE =================

st.image(
    "https://cdn-icons-png.flaticon.com/512/4149/4149677.png",
    width=180
)



# ================= CUSTOMER FORM =================


st.markdown(
"<div class='card'>",
unsafe_allow_html=True
)


st.header("👤 Customer Information")


col1,col2,col3 = st.columns(3)


with col1:

    st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    st.number_input(
        "Tenure (Months)",
        0,
        100
    )


    st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )



with col2:

    st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


    st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Electronic Check",
            "Bank Transfer"
        ]
    )


    st.number_input(
        "Monthly Charges"
    )



with col3:

    st.number_input(
        "Total Charges"
    )


    st.selectbox(
        "Senior Citizen",
        [0,1]
    )


    st.selectbox(
        "Partner",
        ["Yes","No"]
    )



st.markdown(
"</div>",
unsafe_allow_html=True
)



st.write("")



# ================= BUTTON =================


if st.button(
    "🚀 Predict Customer Status",
    use_container_width=True
):

    st.markdown(
    """
    <div class='result-box'>
    ✅ Prediction Result Will Appear Here
    </div>
    """,
    unsafe_allow_html=True
    )



# ================= FOOTER =================


st.write("")

st.markdown(
"""
<center>
<b>Customer Churn Analysis | Machine Learning Project</b>
</center>
""",
unsafe_allow_html=True
)