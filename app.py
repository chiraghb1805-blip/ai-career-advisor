import streamlit as st
import joblib
import numpy as np

model = joblib.load("career_model.pkl")

st.title("AI Career Advisor")

st.write("Enter your skills level (1-10)")

python_skill = st.slider("Python Skill", 1, 10, key="python")
maths = st.slider("Maths Skill", 1, 10, key="maths")
design = st.slider("Design Skill", 1, 10, key="design")
communication = st.slider("Communication Skill", 1, 10, key="communication")

interest_ai = st.checkbox("Interested in AI", key="ai")
interest_web = st.checkbox("Interested in Web Development", key="web")
interest_govt = st.checkbox("Interested in Government Jobs", key="govt")

if st.button("Predict Career"):

    input_data = np.array([[python_skill, maths, design, communication,
                            int(interest_ai),
                            int(interest_web),
                            int(interest_govt)]])

    prediction = model.predict(input_data)

    st.success(f"Recommended Career: {prediction[0]}")
