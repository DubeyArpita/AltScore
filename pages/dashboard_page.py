import os
import numpy as np
import pandas as pd
import streamlit as st

from onnx_utils import load_onnx_sessions, onnx_predict_regressor

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Analytics",
    page_icon="💳",
    layout="wide"
)

DATA_FILE = "data/dataset.csv"

# --------------------------------------------------
# CLEAN PROFESSIONAL UI CSS
# --------------------------------------------------
st.markdown("""
<style>

/* Global Background */
.stApp {
    background-color: #F8FAFC;
}

/* Main Container Width */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title */
.page-title {
    font-size: 34px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #64748B;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section Header */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #0F172A;
    margin-top: 40px;
    margin-bottom: 15px;
}

/* Cards */
.card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
}

/* Metric Numbers */
.metric-number {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

/* Risk Colors */
.low { color: #16A34A; }
.medium { color: #F59E0B; }
.high { color: #DC2626; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def compute_risk_level(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"

def risk_class(val):
    if val == "Low":
        return "low"
    if val == "Medium":
        return "medium"
    if val == "High":
        return "high"
    return ""

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## ALTSCORE")
    st.write("Credit Intelligence Platform")
    st.divider()

    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/Add_user_page.py", label="➕ Add User")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="page-title">Credit Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Monitor alternative credit risk portfolio performance</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.warning("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

if df.empty:
    st.warning("No users registered yet.")
    st.stop()

if "alt_credit_score" in df.columns and "credit_score" not in df.columns:
    df = df.rename(columns={"alt_credit_score": "credit_score"})

df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
df["risk_level"] = df["credit_score"].apply(compute_risk_level)

# --------------------------------------------------
# Metrics Section
# --------------------------------------------------
st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

total = len(df)
low = int((df["credit_score"] >= 70).sum())
medium = int(df["credit_score"].between(40, 69).sum())
high = int((df["credit_score"] < 40).sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
        Total Users
        <div class="metric-number">{total}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        Low Risk
        <div class="metric-number low">{low}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        Medium Risk
        <div class="metric-number medium">{medium}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
        High Risk
        <div class="metric-number high">{high}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# Users Table
# --------------------------------------------------
st.markdown('<div class="section-title">Registered Users</div>', unsafe_allow_html=True)

df_sorted = df.sort_values("credit_score", ascending=False).reset_index(drop=True)
df_sorted.index += 1

st.dataframe(
    df_sorted[[
        "user_id",
        "employment_type",
        "monthly_income",
        "credit_score",
        "risk_level"
    ]],
    use_container_width=True,
    height=500
)

# --------------------------------------------------
# AI Predictions (Last 5)
# --------------------------------------------------
st.markdown('<div class="section-title">Latest 5 Model Predictions</div>', unsafe_allow_html=True)

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
        "Risk": row["risk_level"]
    })

pred_df = pd.DataFrame(pred_rows)

st.dataframe(pred_df, use_container_width=True)
