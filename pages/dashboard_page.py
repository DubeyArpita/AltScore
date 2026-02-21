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

    
     /* ══════════════════════════════════════════
       Hides the default Streamlit Page List
    ══════════════════════════════════════════ */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* ══════════════════════════════════════════
       GLOBAL
    ══════════════════════════════════════════ */
    * { font-family: 'Manrope', sans-serif !important; box-sizing: border-box; }

    /* ══════════════════════════════════════════
       APP BG — blue gradient + money SVG watermark
    ══════════════════════════════════════════ */
    .stApp {
        background-color: #e3f2fd;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' viewBox='0 0 1000 700'%3E%3C!-- Large banknote top-right --%3E%3Crect x='680' y='30' width='280' height='140' rx='18' fill='none' stroke='%231565c0' stroke-width='2' opacity='0.18'/%3E%3Crect x='700' y='50' width='240' height='100' rx='10' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ccircle cx='820' cy='100' r='30' fill='none' stroke='%231565c0' stroke-width='1.5' opacity='0.18'/%3E%3Ccircle cx='820' cy='100' r='20' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ctext x='808' y='107' font-family='monospace' font-size='18' fill='%231565c0' opacity='0.28' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='695' y='80' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3EALT SCORE CREDIT%3C/text%3E%3Ctext x='695' y='148' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3E100 00000 0001%3C/text%3E%3C!-- Coin stack bottom-left --%3E%3Cellipse cx='90' cy='590' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='572' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='554' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='536' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='35' y1='536' x2='35' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='145' y1='536' x2='145' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Ctext x='73' y='567' font-family='monospace' font-size='14' fill='%230d47a1' opacity='0.25' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Large rupee symbols --%3E%3Ctext x='50' y='80' font-family='monospace' font-size='42' fill='%231565c0' opacity='0.07' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='880' y='620' font-family='monospace' font-size='56' fill='%231565c0' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='430' y='660' font-family='monospace' font-size='36' fill='%230d47a1' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Percent symbol --%3E%3Ctext x='910' y='200' font-family='monospace' font-size='80' fill='%231565c0' opacity='0.05' font-weight='bold'%3E%25%3C/text%3E%3C!-- Mini banknote bottom-right --%3E%3Crect x='750' y='560' width='210' height='110' rx='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.15'/%3E%3Crect x='766' y='576' width='178' height='78' rx='8' fill='none' stroke='%230d47a1' stroke-width='0.8' opacity='0.1'/%3E%3Ccircle cx='855' cy='615' r='22' fill='none' stroke='%230d47a1' stroke-width='1.2' opacity='0.15'/%3E%3Ctext x='845' y='621' font-family='monospace' font-size='13' fill='%230d47a1' opacity='0.22' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Stock trend line --%3E%3Cpolyline points='200,680 270,640 340,655 420,590 500,600 580,520 660,490 740,440 820,420' fill='none' stroke='%231565c0' stroke-width='1.8' opacity='0.1'/%3E%3C!-- Credit score arc --%3E%3Cpath d='M 30 220 A 120 120 0 0 1 230 180' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.12' stroke-dasharray='6 6'/%3E%3Ctext x='30' y='270' font-family='monospace' font-size='9' fill='%230d47a1' opacity='0.2'%3ECREDIT SCORE%3C/text%3E%3C!-- Bar chart --%3E%3Crect x='30' y='370' width='14' height='50' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='52' y='350' width='14' height='70' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='74' y='360' width='14' height='60' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='96' y='335' width='14' height='85' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='118' y='345' width='14' height='75' rx='3' fill='%231565c0' opacity='0.1'/%3E%3C/svg%3E"),
            radial-gradient(ellipse 55vw 45vh at 5%  5%,  rgba(21,101,192,0.10) 0%, transparent 65%),
            radial-gradient(ellipse 45vw 50vh at 95% 8%,  rgba(13,71,161,0.08)  0%, transparent 60%),
            radial-gradient(ellipse 50vw 40vh at 80% 92%, rgba(21,101,192,0.09) 0%, transparent 60%),
            radial-gradient(ellipse 40vw 45vh at 10% 90%, rgba(13,71,161,0.07)  0%, transparent 55%),
            linear-gradient(150deg, #e3f2fd 0%, #eff7ff 40%, #e3f2fd 70%, #eaf4fd 100%);
        background-size: 1000px 700px, cover, cover, cover, cover, cover;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat;
        color: #071a2e;
    }

    /* ══════════════════════════════════════════
       SIDEBAR — deep navy
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #040d1a !important;
        border-right: 1px solid rgba(21,101,192,0.22) !important;
        box-shadow: 4px 0 28px rgba(0,0,0,0.45) !important;
    }

    .sidebar-logo {
        font-family: 'Fraunces', serif !important;
        font-size: 1.9rem;
        font-weight: 900;
        text-align: center;
        color: #90caf9;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        text-align: center;
        color: #ffffff;
        font-size: 0.63rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(21,101,192,0.09) !important;
        border: 1px solid rgba(21,101,192,0.2) !important;
        color: #3a6ea8 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        transition: background 0.2s, border-color 0.2s, color 0.2s !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(21,101,192,0.2) !important;
        border-color: rgba(21,101,192,0.45) !important;
        color: #90caf9 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══════════════════════════════════════════
       MAIN CONTENT — white frosted card
    ══════════════════════════════════════════ */
    section.main > div.block-container {
        background: rgba(255, 255, 255, 0.93) !important;
        border-radius: 22px !important;
        border: 1px solid rgba(21,101,192,0.13) !important;
        box-shadow:
            0 0 0 1px rgba(21,101,192,0.05),
            0 8px 48px rgba(13,71,161,0.10),
            0 2px 10px rgba(0,0,0,0.05) !important;
        margin: 20px 20px 20px 8px !important;
        padding: 44px 52px 52px 52px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    .page-eyebrow {
        font-size: 0.9rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #1565c0;
        font-weight: 700;
        margin-bottom: 10px;
        display: block;
    }

    .page-title {
        font-family: 'Fraunces', serif !important;
        font-size: clamp(2.6rem, 4.5vw, 3.8rem);
        font-weight: 900;
        color: #071a2e;
        line-height: 1.08;
        margin: 0 0 16px 0;
        letter-spacing: -0.02em;
    }

    .page-title .grad { color: #1565c0; }

    .page-desc {
        color: #0d2a4a;
        font-size: 1.15rem;
        font-weight: 500;
        max-width: 680px;
        line-height: 1.85;
    }

    .header-rule {
        height: 2px;
        background: linear-gradient(90deg, #1565c0, #64b5f6, transparent);
        border-radius: 2px;
        margin: 26px 0 34px 0;
        opacity: 0.45;
    }

    /* ══════════════════════════════════════════
       SECTION CARDS
    ══════════════════════════════════════════ */
    .sec-card {
        background: #eff7ff;
        border: 1px solid rgba(21,101,192,0.13);
        border-radius: 16px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 18px;
        box-shadow: 0 2px 12px rgba(13,71,161,0.06);
        position: relative;
        overflow: hidden;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0d47a1, #1565c0, #64b5f6);
        border-radius: 16px 16px 0 0;
    }

    .sec-title {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #1565c0;
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(21,101,192,0.11);
    }

    .sec-icon {
        width: 27px; height: 27px;
        background: linear-gradient(135deg, #bbdefb, #90caf9);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════════
       KPI METRIC CARDS — blue gradient family
    ══════════════════════════════════════════ */
    div.stMetric {
        border-radius: 16px !important;
        padding: 22px 20px !important;
        position: relative !important;
        overflow: hidden !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif !important;
        font-size: 2.2rem !important;
        font-weight: 900 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }

    /* Card 1 — Total Users: deep navy → royal blue */
    div.stMetric:nth-of-type(1) {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%) !important;
        border: 1px solid rgba(144,202,249,0.3) !important;
        box-shadow: 0 4px 20px rgba(13,71,161,0.3) !important;
    }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricLabel"] { color: #bbdefb !important; }

    /* Card 2 — Low Risk: royal blue → sky blue */
    div.stMetric:nth-of-type(2) {
        background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%) !important;
        border: 1px solid rgba(144,202,249,0.3) !important;
        box-shadow: 0 4px 20px rgba(21,101,192,0.25) !important;
    }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricLabel"] { color: #e3f2fd !important; }

    /* Card 3 — Medium Risk: steel blue → cyan */
    div.stMetric:nth-of-type(3) {
        background: linear-gradient(135deg, #1976d2 0%, #00acc1 100%) !important;
        border: 1px solid rgba(178,235,242,0.3) !important;
        box-shadow: 0 4px 20px rgba(25,118,210,0.22) !important;
    }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricLabel"] { color: #e0f7fa !important; }

    /* Card 4 — High Risk: indigo → blue */
    div.stMetric:nth-of-type(4) {
        background: linear-gradient(135deg, #283593 0%, #1565c0 100%) !important;
        border: 1px solid rgba(144,202,249,0.25) !important;
        box-shadow: 0 4px 20px rgba(40,53,147,0.25) !important;
    }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricLabel"] { color: #c5cae9 !important; }

    /* ══════════════════════════════════════════
       DATAFRAME — blue/white theme, keep risk colours
    ══════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1.5px solid #90caf9 !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(13,71,161,0.10) !important;
    }

    /* Header row — deep navy with white text */
    [data-testid="stDataFrame"] thead tr th {
        background: #0d47a1 !important;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid #1565c0 !important;
    }

    /* Odd rows — pure white */
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
        background: #ffffff !important;
        color: #071a2e !important;
    }

    /* Even rows — very light blue */
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background: #e8f4fd !important;
        color: #0d2a4a !important;
    }

    /* Hover — soft blue highlight */
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: #bbdefb !important;
        color: #071a2e !important;
    }

    /* Cell base */
    [data-testid="stDataFrame"] td {
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border-color: #c5dff8 !important;
    }

    /* Index column — steel blue tint */
    [data-testid="stDataFrame"] th:first-child,
    [data-testid="stDataFrame"] td:first-child {
        background: #bbdefb !important;
        color: #0d47a1 !important;
        font-weight: 700 !important;
        border-right: 1px solid #90caf9 !important;
    }

    /* ══════════════════════════════════════════
       INFO BOX
    ══════════════════════════════════════════ */
    [data-testid="stInfo"] {
        background: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        border-radius: 10px !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(21,101,192,0.12) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(21,101,192,0.22); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(21,101,192,0.4); }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# Cached model sessions
# -------------------------------------
@st.cache_resource
def load_models():
    return load_onnx_sessions()

# -------------------------------------
# Helpers
# -------------------------------------
def compute_risk_level(score):
    if pd.isna(score): return "Unknown"
    if score >= 70:    return "Low"
    elif score >= 40:  return "Medium"
    else:              return "High"

def color_risk(val):
    if val == "Low":    return "background-color: #6bcf7f; color: #ffffff; font-weight: bold;"
    elif val == "Medium": return "background-color: #FFD93D; color: #374151; font-weight: bold;"
    elif val == "High":   return "background-color: #FF6B6B; color: #ffffff; font-weight: bold;"
    return ""

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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.jpeg", width=110)

    # st.image("logo.jpeg", width=100) 
    st.write("---")
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Credit Beyond Cards</div>", unsafe_allow_html=True)

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True):
        st.rerun()
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

    st.markdown("<p style='text-align:center;color:#0d2a4a;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# -------------------------------------
# Page Header
# -------------------------------------
st.markdown("""
    <div style='margin-bottom: 0;'>
        <h1 class='page-title'>Credit <span class='grad'>Analytics Dashboard</span></h1>
        <p class='page-desc'>Monitor and assess user credit health with AI-powered insights across all registered profiles.</p>
    </div>
    <div class='header-rule'></div>
""", unsafe_allow_html=True)

# -------------------------------------
# Data Load
# -------------------------------------
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_FILE):
    st.warning("📭 Dataset file not found. Please add users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)
df_added_order = df_raw.copy()

if df_raw.empty:
    st.warning("📭 Dataset is empty. Please register some users first.")
    st.stop()

if "alt_credit_score" in df_raw.columns and "credit_score" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"alt_credit_score": "credit_score"})

if "credit_score" not in df_raw.columns:
    st.error("❌ Column 'credit_score' not found.")
    st.stop()

df_raw["credit_score"] = pd.to_numeric(df_raw["credit_score"], errors="coerce")
df_raw = df_raw.sort_values(by="credit_score", ascending=False, na_position="last").reset_index(drop=True)
df_raw["risk_level"] = df_raw["credit_score"].apply(compute_risk_level)

# -------------------------------------
# Load ONNX models
# -------------------------------------
try:
    lr_sess, xgb_sess, rf_sess = load_models()
except Exception as e:
    st.error(f"ONNX model loading error: {e}")
    st.stop()

# -------------------------------------
# Section: KPI Summary
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>📈</div>Portfolio Summary</div>
</div>""", unsafe_allow_html=True)

total_users   = len(df_raw)
low_users     = (df_raw["credit_score"] >= 70).sum()
high_users    = (df_raw["credit_score"] < 40).sum()
medium_users  = df_raw["credit_score"].between(40, 70).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥  Total Users",          f"{total_users}")
col2.metric("✅  Low Risk  ≥ 70",      f"{low_users}")
col3.metric("⚠️  Medium Risk  40–69", f"{medium_users}")
col4.metric("❌  High Risk  < 40",     f"{high_users}")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------------------------
# Section: All Users Overview Table
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>👥</div>User Financial Profiles & AI Predictions</div>
</div>""", unsafe_allow_html=True)

cols = [
    "user_id", "employment_type", "income_range", "city_tier", "monthly_income",
    "bank_account_age_months", "num_bank_accounts", "rent_paid_on_time",
    "utility_delay_days", "upi_txn_count", "avg_month_end_balance",
    "overdraft_event", "credit_score", "risk_level"
]
display_df = df_raw.head(2000)[cols].copy()
display_df.index = range(1, len(display_df) + 1)

st.dataframe(
    display_df.style.map(color_risk, subset=["risk_level"]),
    use_container_width=True,
    height=500
)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------------------------
# Section: AI Predictions (Last 5 Users)
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>🤖</div>AI Predictions — Last 5 Registered Users</div>
</div>""", unsafe_allow_html=True)

df_predict = df_added_order.tail(5).copy().iloc[::-1].reset_index(drop=True)
pred_rows = []
for idx, row in df_predict.iterrows():
    try:
        input_df = build_input_df_from_row(row)
        lr_risk, _ = onnx_predict_classifier_label_and_proba(lr_sess, input_df)
        xgb_score  = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
        rf_score   = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))
    except Exception:
        lr_risk, xgb_score, rf_score = "Error", "Error", "Error"

    pred_rows.append({
        "User ID":           row.get("user_id", f"User_{idx+1}"),
        "Credit Score":      row.get("credit_score", np.nan),
        "Predicted LR Risk": lr_risk,
        "XGB Score":         xgb_score,
        "RF Score":          rf_score,
        "Risk Level":        compute_risk_level(row.get("credit_score", np.nan)),
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

st.dataframe(
    pred_df.style.map(color_risk, subset=["Risk Level"]),
    use_container_width=True,
    height=300
)
