import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Credit Analytics Dashboard",
    page_icon="💳",
    layout="wide"
)

DATA_FILE = "data/dataset.csv"

# -------------------------------------------------------
# Modern UI Styling
# -------------------------------------------------------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Main Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
    color: #00D1FF;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    color: #cbd5e1;
}

/* Section Header */
.section-header {
    font-size: 24px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 15px;
    color: #00D1FF;
}

/* Glass Metric Card */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Cache Models
# -------------------------------------------------------
@st.cache_resource
def load_models():
    return load_onnx_sessions()

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def compute_risk_level(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"

def color_risk(val):
    if val == "Low":
        return "background-color: #6bcf7f; color: white; font-weight: bold;"
    elif val == "Medium":
        return "background-color: #ffd93d; color: black; font-weight: bold;"
    elif val == "High":
        return "background-color: #ff6b6b; color: white; font-weight: bold;"
    return ""

def build_input_df_from_row(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "employment_type": str(row.get("employment_type", "salaried")).strip().lower(),
        "income_range": str(row.get("income_range", "10000-30000")).strip().lower(),
        "city_tier": int(pd.to_numeric(row.get("city_tier", 2), errors="coerce") or 2),
        "bank_account_age_months": int(pd.to_numeric(row.get("bank_account_age_months", 24), errors="coerce") or 24),
        "num_bank_accounts": int(pd.to_numeric(row.get("num_bank_accounts", 1), errors="coerce") or 1),
        "monthly_income": float(pd.to_numeric(row.get("monthly_income", 30000), errors="coerce") or 30000),
        "rent_paid_on_time": float(pd.to_numeric(row.get("rent_paid_on_time", 1.0), errors="coerce") or 1.0),
        "utility_delay_days": float(pd.to_numeric(row.get("utility_delay_days", 0.0), errors="coerce") or 0.0),
        "upi_txn_count": float(pd.to_numeric(row.get("upi_txn_count", 20.0), errors="coerce") or 20.0),
        "avg_month_end_balance": float(pd.to_numeric(row.get("avg_month_end_balance", 5000.0), errors="coerce") or 5000.0),
        "overdraft_event": int(pd.to_numeric(row.get("overdraft_event", 0), errors="coerce") or 0),
    }])

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('<div class="main-title">💳 Credit Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Alternative Credit Risk Insights</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_FILE):
    st.warning("Dataset not found. Please register users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)

if df_raw.empty:
    st.warning("Dataset is empty.")
    st.stop()

if "alt_credit_score" in df_raw.columns and "credit_score" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"alt_credit_score": "credit_score"})

df_raw["credit_score"] = pd.to_numeric(df_raw["credit_score"], errors="coerce")
df_raw["risk_level"] = df_raw["credit_score"].apply(compute_risk_level)

# -------------------------------------------------------
# Metrics Section
# -------------------------------------------------------
st.markdown('<div class="section-header">📈 Portfolio Overview</div>', unsafe_allow_html=True)

total_users = len(df_raw)
low_users = int((df_raw["credit_score"] >= 70).sum())
medium_users = int(df_raw["credit_score"].between(40, 69).sum())
high_users = int((df_raw["credit_score"] < 40).sum())

col1, col2, col3, col4 = st.columns(4)

def metric_card(title, value, color):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color:{color};">{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

with col1:
    metric_card("Total Users", total_users, "#00D1FF")

with col2:
    metric_card("Low Risk", low_users, "#6bcf7f")

with col3:
    metric_card("Medium Risk", medium_users, "#ffd93d")

with col4:
    metric_card("High Risk", high_users, "#ff6b6b")

# -------------------------------------------------------
# Risk Distribution Chart
# -------------------------------------------------------
st.markdown('<div class="section-header">📊 Risk Distribution</div>', unsafe_allow_html=True)

risk_counts = df_raw["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

fig = px.pie(
    risk_counts,
    names="Risk Level",
    values="Count",
    hole=0.5,
    color="Risk Level",
    color_discrete_map={
        "Low": "#6bcf7f",
        "Medium": "#ffd93d",
        "High": "#ff6b6b"
    }
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# Users Table
# -------------------------------------------------------
st.markdown('<div class="section-header">👥 Registered Users</div>', unsafe_allow_html=True)

display_df = df_raw.sort_values("credit_score", ascending=False).copy()
display_df.index = range(1, len(display_df) + 1)

st.dataframe(
    display_df.style.map(color_risk, subset=["risk_level"]),
    use_container_width=True,
    height=500
)

# -------------------------------------------------------
# AI Predictions (Last 5 Users)
# -------------------------------------------------------
st.markdown('<div class="section-header">🤖 AI Model Insights (Latest 5 Users)</div>', unsafe_allow_html=True)

try:
    lr_sess, xgb_sess, rf_sess = load_models()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

df_predict = df_raw.tail(5).iloc[::-1].reset_index(drop=True)

pred_rows = []

for idx, row in df_predict.iterrows():
    try:
        input_df = build_input_df_from_row(row)
        xgb_score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
    except:
        xgb_score = "Error"

    pred_rows.append({
        "User ID": row.get("user_id", f"User_{idx+1}"),
        "Credit Score": row.get("credit_score", np.nan),
        "Risk Level": row.get("risk_level", "Unknown"),
        "XGB Predicted Score": xgb_score
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

st.dataframe(
    pred_df.style.map(color_risk, subset=["Risk Level"]),
    use_container_width=True,
    height=250
)
