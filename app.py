import streamlit as st
import pandas as pd
import pickle

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.main {
    background-color: #F5F8FC;
}

.stApp {
    background-color: #F5F8FC;
}

.title {
    text-align: center;
    color: #123B5D;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #607D8B;
    font-size: 17px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 23px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# LOAD MODEL AND SCALER
# ==============================

@st.cache_resource
def load_files():

    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


try:
    model, scaler = load_files()

except Exception as e:
    st.error("Unable to load model or scaler.")
    st.exception(e)
    st.stop()


# ==============================
# HEADER
# ==============================

st.markdown(
    '<p class="title">🩺 Diabetes Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Diabetes Prediction</p>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter the patient's medical information "
    "to predict the likelihood of diabetes."
)


# ==============================
# PATIENT INPUT FORM
# ==============================

st.subheader("Patient Information")

with st.form("diabetes_form"):

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0
        )

        glucose = st.number_input(
            "Glucose Level",
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
            value=25.0,
            step=0.1
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    submit = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# ==============================
# PREDICTION
# ==============================

if submit:

    input_data = pd.DataFrame([{

        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age

    }])

    try:

        # Apply the saved scaler
        scaled_data = scaler.transform(input_data)

        # Model prediction
        prediction = model.predict(scaled_data)

        st.divider()

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error(
                "⚠️ The Model Predicts: Diabetes"
            )

            st.write(
                "The model classified this input "
                "as belonging to the diabetes class."
            )

        else:

            st.success(
                "✅ The Model Predicts: No Diabetes"
            )

            st.write(
                "The model classified this input "
                "as belonging to the non-diabetes class."
            )

        st.caption(
            "This result is a machine learning prediction "
            "for educational purposes, not a medical diagnosis."
        )

    except Exception as e:

        st.error("Prediction Error")
        st.exception(e)


# ==============================
# FOOTER
# ==============================

st.divider()

st.markdown(
    """
    <center>
    <p style='color:gray;'>
    Diabetes Prediction System | Powered by Machine Learning
    <br>
    SVM Linear Kernel
    </p>
    </center>
    """,
    unsafe_allow_html=True
)
