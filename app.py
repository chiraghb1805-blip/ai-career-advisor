import streamlit as st
import joblib
import numpy as np

model = joblib.load("career_model.pkl")

st.title("AI Career Advisor")

st.write("Enter your skills level (1-10)")

python = st.slider("Python Skill", 1, 10)
maths = st.slider("Maths Skill", 1, 10)
design = st.slider("Design Skill", 1, 10)
communication = st.slider("Communication Skill", 1, 10)

interest_ai = st.checkbox("Interested in AI")
interest_web = st.checkbox("Interested in Web Development")
interest_govt = st.checkbox("Interested in Government Jobs")

if st.button("Predict Career"):

    input_data = np.array([[python, maths, design, communication,
                            int(interest_ai),
                            int(interest_web),
                            int(interest_govt)]])

    prediction = model.predict(input_data)

    st.success(f"Recommended Career: {prediction[0]}")
