import streamlit as st
import pandas as pd
import joblib
import sklearn

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Heart Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS (Dark + Glass + Animation)
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

.heartbeat {
    animation: heartbeat 1.5s infinite;
    font-size: 50px;
    text-align: center;
}

@keyframes heartbeat {
    0% { transform: scale(1); }
    25% { transform: scale(1.1); }
    40% { transform: scale(1); }
    60% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff1e1e;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# ---------------------------------------------------
# TITLE SECTION
# ---------------------------------------------------
st.markdown('<div class="heartbeat">❤️</div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>AI Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built by Faez | Machine Learning Powered</p>", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
with st.sidebar:
    st.header("Patient Details")

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------------------------------------------
# PREDICTION AREA
# ---------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

if st.button("🔍 Predict Risk"):

    with st.spinner("Analyzing patient data..."):

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.markdown(
            '<div class="result-box" style="background-color:#ff4b4b;">⚠️ HIGH RISK DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#00c853;">✅ LOW RISK</div>',
            unsafe_allow_html=True
        )

    st.subheader("Risk Probability")
    st.progress(float(probability))
    st.write(f"Risk Score: {probability*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)