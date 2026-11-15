import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# CUSTOM DESIGN
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #F4F8FC;
}

h1 {
    color: #123B5D;
    text-align: center;
}

h2, h3 {
    color: #164E70;
}

.stButton button {
    background-color: #087E8B;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    background-color: #05616B;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL AND SCALER
# =========================

@st.cache_resource
def load_model():

    with open("diabetes_classifier (5).pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler (5).pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


try:
    model, scaler = load_model()

except Exception as e:
    st.error("Unable to load model or scaler.")
    st.exception(e)
    st.stop()


# =========================
# HEADER
# =========================

st.title("🩺 Diabetes Prediction System")

st.markdown(
    "<center>Machine Learning Based Diabetes Prediction</center>",
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter patient information below "
    "to get a prediction from our Machine Learning Model."
)


# =========================
# INPUT FORM
# =========================

st.header("Patient Information")

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=80.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# =========================
# PREDICTION
# =========================

if predict_button:

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    try:

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)

        st.divider()

        st.header("Prediction Result")

        if prediction[0] == 1:

            st.error(
                "⚠️ The Model Predicts: Diabetes"
            )

        else:

            st.success(
                "✅ The Model Predicts: No Diabetes"
            )

        st.caption(
            "This is an educational machine learning prediction, "
            "not a medical diagnosis."
        )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)


# =========================
# FOOTER
# =========================

st.divider()

st.markdown("""
<center>
<p style="color:gray;">
Diabetes Prediction System
<br>
Powered by SVM - Linear Kernel
</p>
</center>
""", unsafe_allow_html=True)
   
