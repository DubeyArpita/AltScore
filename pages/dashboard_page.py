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
# Custom CSS — matches Register page theme
# -------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,700;0,900;1,700&family=Manrope:wght@400;500;600;700;800&display=swap');

    /* ══════════════════════════════════════════
       ROOT — purple/white theme
    ══════════════════════════════════════════ */
    .stApp {
        background: #f3f0ff;
        font-family: 'Manrope', sans-serif;
        color: #1e0a3c;
    }

    /* Subtle static purple radial tint in corners */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 60vw 50vh at 0%   0%,   rgba(139,92,246,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 50vw 45vh at 100% 100%, rgba(109,40,217,0.10) 0%, transparent 65%),
            radial-gradient(ellipse 40vw 40vh at 100% 0%,   rgba(196,181,253,0.15) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* Finance SVG watermark — static, very subtle */
    .stApp > div:first-child::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='500' viewBox='0 0 800 500'%3E%3Cg opacity='0.045' stroke='%236d28d9' fill='none'%3E%3Cpolyline stroke-width='2' points='0,400 80,340 160,360 240,280 320,300 400,200 480,170 560,120 640,140 720,80 800,50'/%3E%3Cpolyline stroke-width='1.5' stroke='%238b5cf6' points='0,450 100,420 200,430 300,390 400,360 500,310 600,280 700,240 800,200'/%3E%3Cline x1='0' y1='166' x2='800' y2='166' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='0' y1='333' x2='800' y2='333' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='200' y1='0' x2='200' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='400' y1='0' x2='400' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='600' y1='0' x2='600' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Ccircle cx='680' cy='400' r='48' stroke-width='1.5'/%3E%3Ccircle cx='680' cy='400' r='34' stroke-width='1'/%3E%3Ctext x='666' y='406' font-family='monospace' font-size='16' fill='%236d28d9' opacity='1'%3E%E2%82%B9%3C/text%3E%3Crect x='40' y='60' width='12' height='60' fill='%236d28d9'/%3E%3Crect x='65' y='40' width='12' height='80' fill='%238b5cf6'/%3E%3Crect x='90' y='70' width='12' height='50' fill='%236d28d9'/%3E%3Crect x='115' y='30' width='12' height='90' fill='%238b5cf6'/%3E%3Crect x='140' y='55' width='12' height='65' fill='%236d28d9'/%3E%3C/g%3E%3C/svg%3E");
        background-size: 800px 500px;
        background-repeat: repeat;
        background-position: center;
        pointer-events: none;
        z-index: 0;
    }

    /* ══════════════════════════════════════════
       SIDEBAR — deep dark purple
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #12002e  !important;
        border-right: 1px solid rgba(139,92,246,0.2) !important;
        box-shadow: 4px 0 28px rgba(0,0,0,0.5) !important;
    }

    .sidebar-logo {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        color: #c4b5fd;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        text-align: center;
        color: #3b1f6a;
        font-size: 0.63rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(167,139,250,0.05) !important;
        border: 1px solid rgba(167,139,250,0.12) !important;
        color: #4a2e7a !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.92rem !important;
        font-family: 'Manrope', sans-serif !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        transition: background 0.2s, border-color 0.2s, color 0.2s !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139,92,246,0.14) !important;
        border-color: rgba(139,92,246,0.35) !important;
        color: #c4b5fd !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══════════════════════════════════════════
       MAIN CONTENT — white card
    ══════════════════════════════════════════ */
    section.main > div.block-container {
        background: #ffffff !important;
        border-radius: 24px !important;
        border: 1px solid #ede9fe !important;
        box-shadow:
            0 0 0 1px rgba(139,92,246,0.06),
            0 8px 48px rgba(109,40,217,0.09),
            0 2px 8px rgba(0,0,0,0.04) !important;
        margin: 20px 20px 20px 8px !important;
        padding: 44px 52px 52px 52px !important;
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    .page-eyebrow {
        font-family: 'Manrope', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #7c3aed;
        font-weight: 700;
        margin-bottom: 10px;
        display: block;
    }

    .page-title {
        font-family: 'Fraunces', serif;
        font-size: clamp(2.8rem, 4.5vw, 4rem);
        font-weight: 900;
        color: #1e0a3c;
        line-height: 1.08;
        margin: 0 0 18px 0;
        letter-spacing: -0.02em;
    }

    .page-title .grad {
        color: #7c3aed;
    }

    .page-desc {
        color: #3b1f5e;
        font-size: 1.18rem;
        font-weight: 500;
        max-width: 600px;
        line-height: 1.85;
    }

    .header-rule {
        height: 2px;
        background: linear-gradient(90deg, #7c3aed, #c4b5fd, transparent);
        border-radius: 2px;
        margin: 28px 0 36px 0;
        opacity: 0.5;
    }

    /* ══════════════════════════════════════════
       SECTION CARDS
    ══════════════════════════════════════════ */
    .sec-card {
        background: #faf8ff;
        border: 1px solid #ede9fe;
        border-radius: 16px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 18px;
        box-shadow: 0 2px 12px rgba(109,40,217,0.05);
        position: relative;
        overflow: hidden;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #7c3aed, #a78bfa, #c4b5fd);
        border-radius: 16px 16px 0 0;
    }

    .sec-title {
        font-family: 'Manrope', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #6d28d9;
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid #ede9fe;
    }

    .sec-icon {
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════════
       KPI METRIC CARDS — per-card colour theming
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

    /* Card 1 — Total Users: deep indigo */
    div.stMetric:nth-of-type(1) {
        background: linear-gradient(135deg, #1e0a3c 0%, #3b1a7a 100%) !important;
        border: 1px solid rgba(167,139,250,0.25) !important;
        box-shadow: 0 4px 20px rgba(30,10,60,0.3) !important;
    }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricValue"] { color: #e9d5ff !important; }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricLabel"] { color: #a78bfa !important; }

    /* Card 2 — Low Risk: emerald green */
    div.stMetric:nth-of-type(2) {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%) !important;
        border: 1px solid rgba(52,211,153,0.25) !important;
        box-shadow: 0 4px 20px rgba(6,78,59,0.3) !important;
    }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricValue"] { color: #a7f3d0 !important; }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricLabel"] { color: #6ee7b7 !important; }

    /* Card 3 — Medium Risk: amber/gold */
    div.stMetric:nth-of-type(3) {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%) !important;
        border: 1px solid rgba(251,191,36,0.25) !important;
        box-shadow: 0 4px 20px rgba(120,53,15,0.3) !important;
    }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricValue"] { color: #fde68a !important; }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricLabel"] { color: #fcd34d !important; }

    /* Card 4 — High Risk: crimson red */
    div.stMetric:nth-of-type(4) {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important;
        border: 1px solid rgba(252,165,165,0.25) !important;
        box-shadow: 0 4px 20px rgba(127,29,29,0.3) !important;
    }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricValue"] { color: #fecaca !important; }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricLabel"] { color: #fca5a5 !important; }

    /* ══════════════════════════════════════════
       DATAFRAMES — styled to match theme
    ══════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid #ddd6fe !important;
        overflow: hidden !important;
        box-shadow: 0 2px 16px rgba(109,40,217,0.08) !important;
    }

    /* Header row */
    [data-testid="stDataFrame"] thead tr th {
        background: #1e0a3c !important;
        color: #a78bfa !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid #3b1a7a !important;
    }

    /* Alternating rows */
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
        background: #f8f5ff !important;
        color: #2d0f6b !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background: #ede9fe !important;
        color: #1e0a3c !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: #ddd6fe !important;
        color: #1e0a3c !important;
    }

    /* Cell text */
    [data-testid="stDataFrame"] td {
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border-color: #ddd6fe !important;
    }

    /* Index column */
    [data-testid="stDataFrame"] th:first-child,
    [data-testid="stDataFrame"] td:first-child {
        background: #2d0f6b !important;
        color: #c4b5fd !important;
        font-weight: 700 !important;
        border-right: 1px solid #4c1d95 !important;
    }

    /* ══════════════════════════════════════════
       INFO / WARNING / ERROR
    ══════════════════════════════════════════ */
    [data-testid="stInfo"] {
        background: #f5f3ff !important;
        border: 1px solid #ddd6fe !important;
        border-radius: 10px !important;
        color: #5b21b6 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ══════════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════════ */
    hr {
        border: none !important;
        border-top: 1px solid rgba(139,92,246,0.1) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.36); }
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
    if score >= 70: return "Low"
    elif score >= 40: return "Medium"
    else: return "High"

def color_risk(val):
    if val == "Low": return "background-color: #6bcf7f; color: white; font-weight: bold;"
    elif val == "Medium": return "background-color: #FFD93D; color: black; font-weight: bold;"
    elif val == "High": return "background-color: #FF6B6B; color: white; font-weight: bold;"
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

# -------------------------------------
# Sidebar
# -------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>AI Credit Intelligence</div>", unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠  Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📊  Dashboard", use_container_width=True):
        st.rerun()

    if st.button("➕  New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")

    st.write("---")

    if st.button("🗑️  Delete Last Entry", use_container_width=True):
        if os.path.exists(DATA_FILE):
            try:
                df_del = pd.read_csv(DATA_FILE)
                if not df_del.empty:
                    deleted_user = df_del.iloc[-1].get("user_id", "Unknown")
                    df_del = df_del.iloc[:-1]
                    df_del.to_csv(DATA_FILE, index=False)
                    st.success(f"✅ Deleted: {deleted_user}")
                    st.rerun()
                else:
                    st.warning("No entries to delete.")
            except Exception as e:
                st.error(f"Error deleting entry: {e}")
        else:
            st.warning("No dataset found.")

    st.write("---")
    st.markdown("<p style='text-align:center;color:#2a1550;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# -------------------------------------
# Page Header
# -------------------------------------
st.markdown("""
    <div style='margin-bottom: 0;'>
        <span class='page-eyebrow'>// Analytics Overview</span>
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

total_users = len(df_raw)
low_users   = (df_raw["credit_score"] >= 70).sum()
high_users  = (df_raw["credit_score"] < 40).sum()
medium_users = df_raw["credit_score"].between(40, 70).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥  Total Users",           f"{total_users}")
col2.metric("✅  Low Risk  ≥ 70",       f"{low_users}")
col3.metric("⚠️  Medium Risk  40–69",  f"{medium_users}")
col4.metric("❌  High Risk  < 40",      f"{high_users}")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------------------------
# Section: All Users Overview Table
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>👥</div>All Users Overview</div>
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
