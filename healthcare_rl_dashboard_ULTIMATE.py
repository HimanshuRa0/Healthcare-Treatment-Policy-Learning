# =========================================================
# HEALTHCARE RL ULTIMATE DASHBOARD
# =========================================================
# Features Included:
# ✔ Simple Login System
# ✔ Q-Learning RL Model
# ✔ Doctor Chatbot
# ✔ Disease Prediction Model
# ✔ PDF Report Generator
# ✔ Alerts System
# ✔ Live Simulation
# ✔ Patient Lookup
# ✔ Download Buttons
# ✔ Simple Visualizations
# ✔ Everything in ONE FILE
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from sklearn.linear_model import LogisticRegression
from reportlab.pdfgen import canvas
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Healthcare RL Dashboard",
                   layout="wide",
                   page_icon="🧠")

# ---------------------------------------------------------
# SIMPLE LOGIN SYSTEM
# ---------------------------------------------------------
def login_page():
    st.title("🔐 Login to Healthcare RL Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    btn = st.button("Login")

    if btn:
        if username == "admin" and password == "1592005":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()


# ---------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------
st.title("🧠 Healthcare Reinforcement Learning Dashboard – Ultimate Edition")
st.caption("Advanced + RL + Predictions + Chatbot + PDF + Alerts + Simulation – All in one web app")


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Dashboard Settings")

num_patients = st.sidebar.slider("Synthetic Patients", 10, 200, 50)
lr = st.sidebar.slider("RL Learning Rate", 0.01, 1.0, 0.3)
gamma = st.sidebar.slider("Discount Factor", 0.5, 0.99, 0.9)
episodes = st.sidebar.slider("Training Episodes", 10, 200, 50)

train_rl = st.sidebar.button("Train RL Model")
simulate = st.sidebar.button("Run Simulation")


# =========================================================
# SYNTHETIC DATA GENERATION
# =========================================================
def generate_data(n):
    return pd.DataFrame({
        "Patient_ID": range(1, n + 1),
        "HR": np.random.randint(60, 150, n),
        "SBP": np.random.randint(90, 180, n),
        "RR": np.random.randint(10, 30, n),
        "Temp": np.round(np.random.normal(37, 0.6, n), 1),
        "Risk": np.random.choice(["Low", "Medium", "High"], size=n, p=[0.5, 0.3, 0.2]),
    })

df = generate_data(num_patients)


# =========================================================
# RL MODEL – Q LEARNING
# =========================================================
states = ["Low", "Medium", "High"]
actions = ["No Action", "IV Fluids", "Vasopressor", "Call Doctor"]

Q = pd.DataFrame(0.0, index=states, columns=actions)


def reward(state, action):
    if state == "High":
        return 12 if action in ["Vasopressor", "Call Doctor"] else -6
    if state == "Medium":
        return 8 if action == "IV Fluids" else -4
    if state == "Low":
        return 5 if action == "No Action" else -2
    return 0


def train_q_learning():
    for ep in range(episodes):
        s = random.choice(states)
        a = random.choice(actions)
        r = reward(s, a)
        Q.loc[s, a] += lr * (r + gamma * Q.loc[s].max() - Q.loc[s, a])


if train_rl:
    train_q_learning()
    st.sidebar.success("🎉 Q-Learning Model Trained!")


# =========================================================
# ALERT SYSTEM
# =========================================================
def show_alerts(hr, sbp, temp, rr):
    if hr > 120:
        st.warning("⚠️ High Heart Rate Detected (HR > 120)")
    if sbp > 160:
        st.error("🚨 High Blood Pressure (SBP > 160)")
    if temp > 38.5:
        st.warning("🌡️ Fever Detected (Temp > 38.5°C)")
    if rr > 24:
        st.warning("😮 High Respiratory Rate (RR > 24)")

# =========================================================
# MACHINE LEARNING – DISEASE PREDICTION MODEL
# =========================================================
# We create a simple model predicting "High Risk" from vitals
df_ml = df.copy()
df_ml["Label"] = df_ml["Risk"].apply(lambda x: 1 if x == "High" else 0)

model = LogisticRegression()
model.fit(df_ml[["HR", "SBP", "Temp", "RR"]], df_ml["Label"])


def predict_disease(hr, sbp, temp, rr):
    pred = model.predict([[hr, sbp, temp, rr]])[0]
    prob = model.predict_proba([[hr, sbp, temp, rr]])[0][1]
    return pred, prob


# =========================================================
# DOCTOR CHATBOT (RULE-BASED)
# =========================================================
def doctor_chatbot(user_input):
    user_input = user_input.lower()

    if "fever" in user_input or "temperature" in user_input:
        return "High temperature detected. Stay hydrated and monitor fever. Consider paracetamol if above 38.5°C."

    if "bp" in user_input or "blood pressure" in user_input or "sbp" in user_input:
        return "For high BP, reduce salt intake, rest, and seek medical advice if SBP > 160."

    if "heart" in user_input or "hr" in user_input:
        return "High heart rate can be due to stress, dehydration, or infection. Monitor closely."

    if "recommend" in user_input or "treatment" in user_input:
        return "Treatments depend on risk level. Use RL engine to see AI recommendations."

    return "I am a medical assistant bot. Please describe symptoms or ask health questions."


# =========================================================
# DOWNLOAD PDF REPORT
# =========================================================
def generate_pdf(patient):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 18)

    c.drawString(50, 800, "Healthcare Patient Report")

    c.setFont("Helvetica", 12)
    y = 760
    for key, value in patient.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Vitals Explorer",
    "RL Engine",
    "Patient Lookup",
    "Prediction Model",
    "Doctor Chatbot"
])


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.subheader("📊 Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    avg_hr = df.HR.mean()
    avg_sbp = df.SBP.mean()
    avg_temp = df.Temp.mean()
    avg_rr = df.RR.mean()

    col1.metric("Average HR", f"{avg_hr:.1f}")
    col2.metric("Average SBP", f"{avg_sbp:.1f}")
    col3.metric("Average Temp", f"{avg_temp:.1f}")
    col4.metric("Average RR", f"{avg_rr:.1f}")

    show_alerts(avg_hr, avg_sbp, avg_temp, avg_rr)

    st.write("### Risk Level Distribution")
    st.plotly_chart(px.bar(df["Risk"].value_counts(), title="Risk Distribution"), use_container_width=True)


# =========================================================
# TAB 2 — VITALS EXPLORER
# =========================================================
with tab2:
    st.subheader("📉 Vitals Explorer")

    st.plotly_chart(px.histogram(df, x="HR", title="Heart Rate Distribution"), use_container_width=True)
    st.plotly_chart(px.line(df.sort_values("Patient_ID"), x="Patient_ID", y="SBP",
                            title="SBP Line Trend"), use_container_width=True)
    st.plotly_chart(px.box(df, y="Temp", title="Temperature Box Plot"), use_container_width=True)
    st.plotly_chart(px.scatter(df, x="HR", y="SBP", color="Risk",
                               title="HR vs SBP Scatter"), use_container_width=True)


# =========================================================
# TAB 3 — RL ENGINE
# =========================================================
with tab3:
    st.subheader("🤖 RL Treatment Recommendation Engine")

    if train_rl:
        st.write("### Q-Table")
        st.dataframe(Q.style.highlight_max(axis=1))

        st.plotly_chart(px.imshow(Q, text_auto=True, title="Q-Table Heatmap"), use_container_width=True)

        best_actions = Q.idxmax(axis=1)
        st.write("### Best Actions")
        st.write(best_actions)

        st.plotly_chart(px.bar(best_actions, title="Best Action by Risk Level"), use_container_width=True)

    else:
        st.info("Train the RL model from the sidebar to enable this section.")


# =========================================================
# TAB 4 — PATIENT LOOKUP
# =========================================================
with tab4:
    st.subheader("🔍 Patient Lookup")

    pid = st.number_input("Enter Patient ID", min_value=1, max_value=num_patients)

    patient = df[df["Patient_ID"] == pid]

    if not patient.empty:
        p = patient.iloc[0]

        st.success(f"Patient {pid} Found")

        st.metric("Heart Rate", p.HR)
        st.metric("SBP", p.SBP)
        st.metric("Temperature", p.Temp)
        st.metric("Resp Rate", p.RR)
        st.metric("Risk Level", p.Risk)

        show_alerts(p.HR, p.SBP, p.Temp, p.RR)

        if train_rl:
            st.info(f"💡 RL Recommendation: **{Q.idxmax(axis=1)[p.Risk]}**")

        # Export PDF
        pdf_buffer = generate_pdf({
            "Patient ID": p.Patient_ID,
            "HR": p.HR,
            "SBP": p.SBP,
            "Temp": p.Temp,
            "RR": p.RR,
            "Risk": p.Risk
        })

        st.download_button("📄 Download Patient PDF Report", data=pdf_buffer,
                           file_name=f"patient_{pid}_report.pdf")


# =========================================================
# TAB 5 — DISEASE PREDICTION
# =========================================================
with tab5:
    st.subheader("🧬 Disease Prediction Model")

    hr_i = st.slider("HR", 40, 170, 90)
    sbp_i = st.slider("SBP", 80, 200, 120)
    temp_i = st.slider("Temperature", 35.0, 41.0, 37.0)
    rr_i = st.slider("RR", 8, 35, 16)

    pred, prob = predict_disease(hr_i, sbp_i, temp_i, rr_i)

    if pred == 1:
        st.error(f"🔴 High Disease Risk Detected — Probability: {prob:.2f}")
    else:
        st.success(f"🟢 Low Disease Risk — Probability: {prob:.2f}")


# =========================================================
# TAB 6 — DOCTOR CHATBOT
# =========================================================
with tab6:
    st.subheader("🩺 Doctor Chatbot")

    user_msg = st.text_input("Ask your health question:")

    if user_msg:
        response = doctor_chatbot(user_msg)
        st.info(response)
# =========================================================
# RAW DATA + DOWNLOAD
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Data")

csv_data = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download Patient Dataset (CSV)",
                           data=csv_data,
                           file_name="patient_dataset.csv",
                           mime="text/csv")

if train_rl:
    q_csv = Q.to_csv().encode("utf-8")
    st.sidebar.download_button("Download Q-Table (CSV)",
                               data=q_csv,
                               file_name="q_table.csv",
                               mime="text/csv")


# =========================================================
# LIVE SIMULATION (ICU FEED)
# =========================================================
st.markdown("---")
st.subheader("🚨 Live Vital Simulation (ICU Mode)")

if simulate:
    placeholder = st.empty()

    for i in range(20):
        hr = random.randint(60, 160)
        sbp = random.randint(90, 180)
        temp = round(np.random.normal(37, 0.7), 1)
        rr = random.randint(10, 35)

        with placeholder.container():
            st.markdown("### 🔴 Live Patient Vitals")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("HR", hr)
            col2.metric("SBP", sbp)
            col3.metric("Temp", temp)
            col4.metric("RR", rr)

            # ⚠️ realtime alerts
            show_alerts(hr, sbp, temp, rr)

        time.sleep(1)


# =========================================================
# RAW DATA TAB (INSIDE MAIN UI)
# =========================================================
st.markdown("---")
st.subheader("📄 Full Patient Dataset")
st.dataframe(df, use_container_width=True)


# =========================================================
# LOGOUT BUTTON
# =========================================================
st.sidebar.markdown("---")
logout = st.sidebar.button("Logout")

if logout:
    st.session_state["logged_in"] = False
    st.rerun()


# =========================================================
# END OF FILE
# =========================================================
st.success("Loaded Healthcare RL Dashboard Successfully!")

 # streamlit run healthcare_rl_dashboard_ULTIMATE.py
# admin 1234
#Why is my heart rate high?
#How to control blood pressure?
#I have a fever, what should I take?