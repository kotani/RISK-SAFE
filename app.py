import streamlit as st
import pickle
import numpy as np

# model
with open("SVC_revise.pkl", "rb") as f:
    model = pickle.load(f) 

scaler = model.named_steps["scaler"]
svm = model.named_steps["clf"]

# title
st.title("Brugada Syndrome Risk Prediction\n RISK-SAFE score")

# user input
st.sidebar.header("Input Features")

# input filed
r_J_interval = st.sidebar.number_input("r-J interval in V1 (ms)", min_value=0.0, max_value=400.0, step=0.1)
QRS_V6 = st.sidebar.number_input("QRS duration in V6 (ms)", min_value=0.0, max_value=1000.0, step=0.1)
T_peak_to_T_end = st.sidebar.number_input("T-peak-to-T-end interval (ms)", min_value=0.0, max_value=400.0, step=0.1)
age = st.sidebar.number_input("Age", min_value=0, max_value=150, step=1)

syncope = st.sidebar.radio("Syncope", [0, 1])
frag_QRS = st.sidebar.radio("Fragmented QRS", [0, 1])
ER_presence = st.sidebar.radio("Presence of ER", [0, 1])


if st.sidebar.button("Predict"):
    # data to numpy
    input_data = np.array([[r_J_interval, syncope, frag_QRS, ER_presence, T_peak_to_T_end, QRS_V6, age]])
    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Predicted probability of major arrhythmic events in Brugada syndrome: {probability * 100:.2f}%")

 
