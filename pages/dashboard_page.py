import os
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor

st.set_page_config(page_title="AltScore", page_icon="⚡", layout="wide")

DATA_FILE = "data/dataset.csv"

# ----------------------------------------------------
# 🎨 CASCADE STYLE THEME
# ----------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(145deg, #111111, #1c1c1c);
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Reduce spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #222;
}

/* Vertical Tabs Style */
div[role="radiogroup"] > label {
    display: block;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-radius: 10px;
    background: #1a1a1a;
    cursor: pointer;
    transition: 0.3s ease;
    font-weight: 500;
}

div[role="radiogroup"] > label:hover {
    background: #262626;
    transform: translateX(5px);
}

/* Section Titles */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: #1a1a1a;
    padding: 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
}

/* Metric */
.metric {
    font-size: 28px;
    font-weight: 700;
}

/* Risk Colors */
.low { color: #00ffae; font-weight: 600; }
.medium { color: #ffd166; font-weight: 600; }
.high { color: #ff4d6d; font-weight: 600; }

/* Table */
[data-testid="stDataFrame"] {
    background: #111;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.warning("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

if "alt_credit_score" in df.columns and "credit_score" not in df.columns:
    df.rename(columns={"alt_credit_score": "credit_score"}, inplace=True)

df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")

def risk_level(score):
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"

df["risk"] = df["credit_score"].apply(risk_level)

# ----------------------------------------------------
# Layout (Vertical Tabs)
# ----------------------------------------------------
left, right = st.columns([1,4])

with left:
    st.markdown("## ⚡ AltScore")
    tab = st.radio(
        "Navigation",
        ["Dashboard Overview", "Registered Users", "Model Predictions"],
        label_visibility="collapsed"
    )

with right:

    # -----------------------------------------
    # Dashboard Overview
    # -----------------------------------------
    if tab == "Dashboard Overview":

        st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

        total = len(df)
        low = (df["risk"] == "Low").sum()
        medium = (df["risk"] == "Medium").sum()
        high = (df["risk"] == "High").sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="card">Total Users<div class="metric">{total}</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="card">Low Risk<div class="metric low">{low}</div></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="card">Medium Risk<div class="metric medium">{medium}</div></div>', unsafe_allow_html=True)

        with col4:
            st.markdown(f'<div class="card">High Risk<div class="metric high">{high}</div></div>', unsafe_allow_html=True)

    # -----------------------------------------
    # Registered Users
    # -----------------------------------------
    elif tab == "Registered Users":

        st.markdown('<div class="section-title">Registered Users</div>', unsafe_allow_html=True)

        df_sorted = df.sort_values("credit_score", ascending=False)

        def highlight_risk(val):
            if val == "Low":
                return "color:#00ffae; font-weight:bold;"
            if val == "Medium":
                return "color:#ffd166; font-weight:bold;"
            if val == "High":
                return "color:#ff4d6d; font-weight:bold;"
            return ""

        styled = df_sorted.style.map(highlight_risk, subset=["risk"])

        st.dataframe(styled, use_container_width=True, height=500)

    # -----------------------------------------
    # Model Predictions
    # -----------------------------------------
    elif tab == "Model Predictions":

        st.markdown('<div class="section-title">Latest Model Predictions</div>', unsafe_allow_html=True)

        @st.cache_resource
        def load_models():
            return load_onnx_sessions()

        try:
            _, xgb_sess, _ = load_models()
        except:
            st.warning("Model not loaded.")
            st.stop()

        latest = df.tail(5).iloc[::-1]
        pred_rows = []

        for _, row in latest.iterrows():
            try:
                input_df = pd.DataFrame([row])
                score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
            except:
                score = "Error"

            pred_rows.append({
                "User ID": row["user_id"],
                "Actual Score": row["credit_score"],
                "Predicted Score": score,
                "Risk": row["risk"]
            })

        pred_df = pd.DataFrame(pred_rows)

        st.dataframe(pred_df, use_container_width=True)
