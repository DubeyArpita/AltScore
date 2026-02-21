import os
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(page_title="AltScore | Dashboard", layout="wide", page_icon="📊")
DATA_FILE = "data/dataset.csv"

# -------------------------------------
# Custom CSS — Blue theme
# -------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:ital,wght@0,700;0,900;1,700&display=swap');

    /* GLOBAL */
    * { font-family: 'Manrope', sans-serif !important; box-sizing: border-box; }

    /* APP BG */
    .stApp {
        background-color: #e3f2fd;
        background-image: linear-gradient(150deg, #e3f2fd 0%, #eff7ff 40%, #e3f2fd 70%, #eaf4fd 100%);
        color: #071a2e;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: #040d1a !important;
        border-right: 1px solid rgba(21,101,192,0.22) !important;
    }

    .sidebar-logo {
        font-family: 'Fraunces', serif !important;
        font-size: 1.9rem; font-weight: 900; text-align: center; color: #90caf9;
    }

    /* MAIN CONTAINER */
    section.main > div.block-container {
        background: rgba(255, 255, 255, 0.93) !important;
        border-radius: 22px !important;
        padding: 44px 52px !important;
        backdrop-filter: blur(8px) !important;
    }

    .page-title {
        font-family: 'Fraunces', serif !important;
        font-size: 3rem; font-weight: 900; color: #071a2e;
    }
    .page-title .grad { color: #1565c0; }

    /* KPI METRICS */
    div.stMetric {
        border-radius: 16px !important;
        padding: 22px 20px !important;
        color: white !important;
    }
    div[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; color: white !important;}
    div[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.8) !important; font-weight: 800 !important;}

    div.stMetric:nth-of-type(1) { background: linear-gradient(135deg, #0d47a1, #1976d2) !important; }
    div.stMetric:nth-of-type(2) { background: linear-gradient(135deg, #1565c0, #42a5f5) !important; }
    div.stMetric:nth-of-type(3) { background: linear-gradient(135deg, #1976d2, #00acc1) !important; }
    div.stMetric:nth-of-type(4) { background: linear-gradient(135deg, #283593, #1565c0) !important; }

    /* ══════════════════════════════════════════
       HTML TABLE CSS (The "Third Way")
    ══════════════════════════════════════════ */
    .custom-table-container {
        overflow-x: auto;
        border-radius: 14px;
        border: 1.5px solid #90caf9;
        box-shadow: 0 4px 20px rgba(13,71,161,0.10);
        margin-bottom: 20px;
    }
    
    table.altscore-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Manrope', sans-serif;
        background-color: white;
    }

    table.altscore-table thead th {
        background: #0d47a1 !important;
        color: #ffffff !important;
        padding: 14px;
        text-align: left;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 3px solid #1565c0;
    }

    table.altscore-table tbody tr:nth-child(odd) { background: #ffffff; }
    table.altscore-table tbody tr:nth-child(even) { background: #e8f4fd; }
    table.altscore-table tbody tr:hover { background: #bbdefb; }

    table.altscore-table td {
        padding: 12px 14px;
        font-size: 0.88rem;
        color: #071a2e;
        border-bottom: 1px solid #c5dff8;
    }

    /* Risk Badge Styling inside HTML Table */
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .badge-low { background: #6bcf7f; color: white; }
    .badge-medium { background: #FFD93D; color: #374151; }
    .badge-high { background: #FF6B6B; color: white; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# Helpers
# -------------------------------------
@st.cache_resource
def load_models():
    return load_onnx_sessions()

def compute_risk_level(score):
    if pd.isna(score): return "Unknown"
    if score >= 70:    return "Low"
    elif score >= 40:  return "Medium"
    else:              return "High"

def get_risk_badge(level):
    """Returns HTML for a risk badge based on level."""
    if level == "Low": return f'<span class="badge badge-low">{level}</span>'
    if level == "Medium": return f'<span class="badge badge-medium">{level}</span>'
    if level == "High": return f'<span class="badge badge-high">{level}</span>'
    return level

def render_html_table(df):
    """Converts a dataframe to the custom HTML structure."""
    # Apply badge logic to the risk level column if it exists
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].apply(get_risk_badge)
    if "Risk Level" in df.columns:
        df["Risk Level"] = df["Risk Level"].apply(get_risk_badge)
        
    html = df.to_html(classes='altscore-table', escape=False, index=False)
    st.markdown(f'<div class="custom-table-container">{html}</div>', unsafe_allow_html=True)

def build_input_df_from_row(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "employment_type":        str(row.get("employment_type", "salaried")).strip().lower(),
        "income_range":           str(row.get("income_range", "10000-30000")).strip().lower(),
        "city_tier":              int(pd.to_numeric(row.get("city_tier", 2), errors="coerce") or 2),
        "bank_account_age_months":int(pd.to_numeric(row.get("bank_account_age_months", 24), errors="coerce") or 24),
        "num_bank_accounts":      int(pd.to_numeric(row.get("num_bank_accounts", 1), errors="coerce") or 1),
        "monthly_income":         float(pd.to_numeric(row.get("monthly_income", 30000), errors="coerce") or 30000),
        "rent_paid_on_time":      float(pd.to_numeric(row.get("rent_paid_on_time", 1.0), errors="coerce") or 1.0),
        "utility_delay_days":     float(pd.to_numeric(row.get("utility_delay_days", 0.0), errors="coerce") or 0.0),
        "upi_txn_count":          float(pd.to_numeric(row.get("upi_txn_count", 20.0), errors="coerce") or 20.0),
        "avg_month_end_balance":  float(pd.to_numeric(row.get("avg_month_end_balance", 5000.0), errors="coerce") or 5000.0),
        "overdraft_event":        int(pd.to_numeric(row.get("overdraft_event", 0), errors="coerce") or 0),
    }])

# -------------------------------------
# Sidebar
# -------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#90caf9; font-size:0.7rem;'>AI CREDIT INTEL</p>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠  Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊  Dashboard", use_container_width=True): st.rerun()

# -------------------------------------
# Main Data Logic
# -------------------------------------
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_FILE) or pd.read_csv(DATA_FILE).empty:
    st.warning("📭 Dataset is empty. Please register some users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)
df_added_order = df_raw.copy()

if "alt_credit_score" in df_raw.columns:
    df_raw = df_raw.rename(columns={"alt_credit_score": "credit_score"})

df_raw["credit_score"] = pd.to_numeric(df_raw["credit_score"], errors="coerce")
df_raw = df_raw.sort_values(by="credit_score", ascending=False).reset_index(drop=True)
df_raw["risk_level"] = df_raw["credit_score"].apply(compute_risk_level)

lr_sess, xgb_sess, rf_sess = load_models()

# -------------------------------------
# UI: Header & Metrics
# -------------------------------------
st.markdown("<h1 class='page-title'>Credit <span class='grad'>Dashboard</span></h1>", unsafe_allow_html=True)

total_users = len(df_raw)
low_users = (df_raw["risk_level"] == "Low").sum()
medium_users = (df_raw["risk_level"] == "Medium").sum()
high_users = (df_raw["risk_level"] == "High").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Users", f"{total_users}")
col2.metric("✅ Low Risk", f"{low_users}")
col3.metric("⚠️ Medium Risk", f"{medium_users}")
col4.metric("❌ High Risk", f"{high_users}")

# -------------------------------------
# UI: Main Table (The HTML Way)
# -------------------------------------
st.subheader("📊 Portfolio Overview")
display_cols = ["user_id", "employment_type", "monthly_income", "upi_txn_count", "credit_score", "risk_level"]
render_html_table(df_raw[display_cols].head(15))

# -------------------------------------
# UI: AI Predictions (Last 5)
# -------------------------------------
st.subheader("🤖 Recent AI Insights")
df_predict = df_added_order.tail(5).copy().iloc[::-1]
pred_rows = []
for _, row in df_predict.iterrows():
    input_df = build_input_df_from_row(row)
    lr_risk, _ = onnx_predict_classifier_label_and_proba(lr_sess, input_df)
    xgb_score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
    rf_score = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))
    
    pred_rows.append({
        "User ID": row.get("user_id"),
        "Score": int(row.get("credit_score", 0)),
        "XGB": f"{xgb_score:.1f}",
        "RF": f"{rf_score:.1f}",
        "Risk Level": compute_risk_level(row.get("credit_score"))
    })

render_html_table(pd.DataFrame(pred_rows))
