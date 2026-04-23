import streamlit as st
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
data = joblib.load("disease_model.pkl")
model = data["model"]
mlb = data["mlb"]
le = data["le"]
scaler = data["scaler"]

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Disease Predictor AI",
    page_icon="💀",
    layout="centered"
)

# -----------------------------
# 🔥 DARK "HELL MODE" UI
# -----------------------------
st.markdown("""
<style>

/* FULL BLACK BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #0a0000, #000000);
    color: white;
}

/* glowing title */
h1 {
    text-align: center;
    color: #ff1a1a;
    text-shadow: 0 0 10px #ff0000, 0 0 25px #990000;
    animation: flicker 1.5s infinite;
}

/* flicker animation (hell vibe) */
@keyframes flicker {
    0% {opacity: 1;}
    50% {opacity: 0.6;}
    100% {opacity: 1;}
}

/* container glow */
.main {
    background: rgba(20, 0, 0, 0.7);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 0 25px #ff0000;
}

/* buttons */
.stButton button {
    background: linear-gradient(90deg, #ff0000, #660000);
    color: white;
    border-radius: 12px;
    padding: 10px;
    font-size: 18px;
    transition: 0.3s;
    box-shadow: 0 0 10px red;
}

.stButton button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 25px red;
}

/* floating red particles */
@keyframes floatUp {
    0% {transform: translateY(100vh); opacity: 0;}
    50% {opacity: 1;}
    100% {transform: translateY(-10vh); opacity: 0;}
}

.particle {
    position: fixed;
    width: 6px;
    height: 6px;
    background: red;
    border-radius: 50%;
    animation: floatUp 6s infinite;
}

</style>

<div class="particle" style="left:10%"></div>
<div class="particle" style="left:30%"></div>
<div class="particle" style="left:50%"></div>
<div class="particle" style="left:70%"></div>
<div class="particle" style="left:90%"></div>

""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1>💀 AI DOOM DISEASE PREDICTOR</h1>", unsafe_allow_html=True)

st.write("Enter symptoms, age, gender... let fate decide.")

# -----------------------------
# INPUTS
# -----------------------------
symptoms = st.text_input("Enter symptoms (comma separated)")
age = st.number_input("Age", 1, 120, 25)
gender = st.selectbox("Gender", ["male", "female"])

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(symptoms, age, gender):
    sym_list = [s.strip().lower() for s in symptoms.split(",")]
    sym_vec = mlb.transform([sym_list])
    age_scaled = scaler.transform([[age]])
    gender_val = 0 if gender == "male" else 1
    return np.hstack([sym_vec, age_scaled, [[gender_val]]])

# -----------------------------
# PREDICT
# -----------------------------
if st.button("UNLEASH PREDICTION 💀"):

    if symptoms == "":
        st.warning("Enter symptoms first...")
    else:
        X = preprocess(symptoms, age, gender)
        pred = model.predict(X)
        disease = le.inverse_transform(pred)[0]

        st.success(f"☠️ Predicted Disease: {disease}")

        # dramatic effect
        st.markdown("""
        <h2 style="color:red; text-align:center;
        text-shadow:0 0 10px red;">
        ☠️ YOU ARE ENTERING THE GRIM REALM...
        </h2>
        """, unsafe_allow_html=True)

        st.balloons()
