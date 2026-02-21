import os
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor

st.set_page_config(page_title="AltScore", page_icon="⚡", layout="wide")

DATA_FILE = "data/dataset.csv"

# -----------------------------------------------------
# 🌌 FUTURISTIC GLASSMORPHISM THEME
# -----------------------------------------------------
st.markdown("""
<style>

/* Background Glow */
.stApp {
    background: radial-gradient(circle at 20% 20%, #3b1c71 0%, transparent 40%),
                radial-gradient(circle at 80% 80%, #0f3460 0%, transparent 40%),
                linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Main container tighter spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    background: linear-gradient(90deg,#8E2DE2,#4A00E0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section headers */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 25px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: all 0.3s ease-in-out;
}

.glass:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

/* Metric number */
.metric-number {
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
}

/* Risk colors */
.low { color: #00ffae; }
.medium { color: #ffd166; }
.high { color: #ff4d6d; }

/* Table styling */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Helper
# -----------------------------------------------------
def compute_risk(score):
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ AltScore")
    st.caption("AI Credit Intelligence")
    st.divider()
    st.page_link("app.py", label="🏠 Home")
    st.page_link("pages/Add_user_page.py", label="➕ Register User")

# -----------------------------------------------------
# Header
# -----------------------------------------------------
st.markdown('<div class="main-title">Credit Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Gen Alternative Credit Risk Intelligence</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
if not os.path.exists(DATA_FILE):
    st.warning("No dataset found.")
    st.stop()

df = pd.read_csv(DATA_FILE)

if "alt_credit_score" in df.columns and "credit_score" not in df.columns:
    df.rename(columns={"alt_credit_score": "credit_score"}, inplace=True)

df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
df["risk"] = df["credit_score"].apply(compute_risk)

# -----------------------------------------------------
# Metrics (Glass Cards)
# -----------------------------------------------------
st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

total = len(df)
low = (df["risk"] == "Low").sum()
medium = (df["risk"] == "Medium").sum()
high = (df["risk"] == "High").sum()

cols = st.columns(4)

metrics = [
    ("Total Users", total, ""),
    ("Low Risk", low, "low"),
    ("Medium Risk", medium, "medium"),
    ("High Risk", high, "high"),
]

for col, (title, value, color) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="glass">
            {title}
            <div class="metric-number {color}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------
# Users Table
# -----------------------------------------------------
st.markdown('<div class="section-title">Registered Users</div>', unsafe_allow_html=True)

df_sorted = df.sort_values("credit_score", ascending=False)
st.dataframe(
    df_sorted[["user_id","employment_type","monthly_income","credit_score","risk"]],
    use_container_width=True,
    height=420
)

# -----------------------------------------------------
# AI Prediction Section
# -----------------------------------------------------
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

st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)
