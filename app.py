```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_classifier.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: bold;
    color: #173b67;
    text-align: center;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    text-align: center;
}

div.stButton > button {
    background-color: #1677c8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #105b9b;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL AND SCALER
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "diabetes_classifier.pkl was not found."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "scaler.pkl was not found."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("🩺 Diabetes System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Diabetes Prediction", "About"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Based Diabetes Prediction System"
)


# ==========================================
# 5. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<p class="title">Diabetes Prediction System</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Machine Learning Based Healthcare Application</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("Welcome!")

        st.write("""
        This application uses Machine Learning to estimate
        the likelihood of diabetes based on patient health
        measurements.

        The system uses a trained machine learning model
        and a scaler to process patient information.
        """)

        st.success(
            "Navigate to Diabetes Prediction to start."
        )

    with col2:

        st.subheader("Application Features")

        st.write("✅ Patient data input")
        st.write("✅ Machine Learning prediction")
        st.write("✅ Data scaling")
        st.write("✅ Prediction results")
        st.write("✅ Simple interface")


# ==========================================
# 6. DIABETES PREDICTION PAGE
# ==========================================

elif page == "Diabetes Prediction":

    st.title("🩺 Diabetes Prediction")

    st.write(
        "Enter the patient's information below."
    )

    st.markdown("---")

    try:

        model, scaler = load_model()

        st.success(
            "Model and Scaler loaded successfully!"
        )

    except Exception as e:

        st.error("Unable to load model or scaler.")

        st.exception(e)

        st.stop()


    st.subheader("Patient Information")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=30,
                value=0
            )

            glucose = st.number_input(
                "Glucose Level",
                min_value=0.0,
                max_value=500.0,
                value=120.0
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=300.0,
                value=70.0
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=150.0,
                value=20.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=3000.0,
                value=80.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=150.0,
                value=25.0
            )

            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=5.0,
                value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 Predict Diabetes"
        )


    # ==========================================
    # 7. PREDICTION PROCESS
    # ==========================================

    if submitted:

        try:

            feature_names = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]

            input_data = pd.DataFrame(
                [[
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    diabetes_pedigree,
                    age
                ]],
                columns=feature_names
            )

            if hasattr(scaler, "n_features_in_"):

                if scaler.n_features_in_ != 8:

                    raise ValueError(
                        "Scaler must have exactly 8 input features."
                    )

            scaled_data = scaler.transform(input_data)

            prediction = model.predict(scaled_data)[0]

            st.markdown("---")

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ Positive Diabetes Prediction"
                )

                st.warning(
                    "The model predicts an increased likelihood "
                    "of diabetes."
                )

            else:

                st.success(
                    "✅ Negative Diabetes Prediction"
                )

                st.info(
                    "The model does not predict diabetes "
                    "based on the provided measurements."
                )


            # ==================================
            # 8. PREDICTION PROBABILITY
            # ==================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    scaled_data
                )[0]

                classes = list(model.classes_)

                col1, col2 = st.columns(2)

                with col1:

                    if 0 in classes:

                        no_diabetes_probability = (
                            probabilities[classes.index(0)] * 100
                        )

                        st.metric(
                            "No Diabetes",
                            f"{no_diabetes_probability:.2f}%"
                        )

                with col2:

                    if 1 in classes:

                        diabetes_probability = (
                            probabilities[classes.index(1)] * 100
                        )

                        st.metric(
                            "Diabetes",
                            f"{diabetes_probability:.2f}%"
                        )

                        st.progress(
                            float(probabilities[classes.index(1)])
                        )

            st.markdown("---")

            st.caption(
                "Disclaimer: This application is for educational "
                "and research purposes only. It does not provide "
                "a medical diagnosis. Please consult a healthcare "
                "professional for medical advice."
            )

        except Exception as e:

            st.error(
                "An error occurred during prediction."
            )

            st.exception(e)


# ==========================================
# 9. ABOUT PAGE
# ==========================================

elif page == "About":

    st.title("About The Project")

    st.markdown("""
    ### Diabetes Prediction Using Machine Learning

    This project uses Machine Learning to estimate
    diabetes likelihood based on patient health measurements.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Pickle

    ### Project Components

    1. Trained Machine Learning Model
    2. Data Scaler
    3. Streamlit User Interface
    4. Prediction System

    ### Objective

    To provide a simple interface for exploring
    machine learning-based diabetes predictions.
    """)


# ==========================================
# 10. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>
    Diabetes Prediction System | Machine Learning Project
    </center>
    """,
    unsafe_allow_html=True
)
```
             
