import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# --- Page Config ---
st.set_page_config(page_title="AltScore | Register", layout="wide", page_icon="📝")

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
        background: #12002e !important;
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
       INPUTS — white, dark text
    ══════════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: #ffffff !important;
        color: #1e0a3c !important;
        border: 1.5px solid #ddd6fe !important;
        border-radius: 10px !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(109,40,217,0.06) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    }

    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #ffffff !important;
        border: 1.5px solid #ddd6fe !important;
        border-radius: 10px !important;
        color: #1e0a3c !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 3px rgba(109,40,217,0.06) !important;
    }

    /* Labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #3b1f5e !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        font-family: 'Manrope', sans-serif !important;
    }

    /* Radio options */
    .stRadio [role="radiogroup"] label div p {
        color: #2d1450 !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
    }

    /* Slider thumb */
    [data-baseweb="slider"] [role="slider"] {
        background: #7c3aed !important;
        border: 2px solid #fff !important;
        box-shadow: 0 0 8px rgba(124,58,237,0.4) !important;
    }

    /* Slider track fill */
    [data-baseweb="slider"] [data-testid="stSlider"] div[role="progressbar"] {
        background: #7c3aed !important;
    }

    /* Info */
    [data-testid="stInfo"] {
        background: #f5f3ff !important;
        border: 1px solid #ddd6fe !important;
        border-radius: 10px !important;
        color: #5b21b6 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Toggle */
    div[data-testid="stToggle"] p {
        color: #3b1f5e !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }

    /* ══════════════════════════════════════════
       CENTERED SUBMIT BUTTON
    ══════════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 320px !important;
        padding: 17px 60px !important;
        border-radius: 50px !important;
        border: none !important;
        background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 50%, #a78bfa 100%) !important;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        box-shadow: 0 4px 24px rgba(109,40,217,0.3), 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 36px rgba(109,40,217,0.45), 0 4px 12px rgba(0,0,0,0.12) !important;
    }

    /* ══════════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: #faf8ff !important;
        border: 1px solid #ddd6fe !important;
        border-radius: 12px !important;
        color: #5b21b6 !important;
        font-family: 'Manrope', sans-serif !important;
    }

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

# --- Logic Functions ---
DATA_FILE = "data/dataset.csv"
REQUIRED_COLUMNS = [
    "user_id", "employment_type", "income_range", "city_tier",
    "bank_account_age_months", "num_bank_accounts", "monthly_income",
    "rent_paid_on_time", "utility_delay_days", "upi_txn_count",
    "avg_month_end_balance", "overdraft_event", "alt_credit_score"
]

@st.cache_resource
def load_models():
    return load_onnx_sessions()

def ensure_dataset_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(DATA_FILE, index=False)

def generate_user_id():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty and "user_id" in df.columns:
                last_id = str(df.iloc[-1]["user_id"])
                m = re.search(r"(\d+)$", last_id)
                if m:
                    return f"USER_{int(m.group(1)) + 1:04d}"
        except: pass
    return "USER_0001"

def get_dropdown_options_from_dataset():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                return (
                    sorted(df["employment_type"].dropna().unique().tolist()),
                    sorted(df["income_range"].dropna().unique().tolist()),
                    sorted(list(set([int(x) for x in df["city_tier"].dropna().tolist()])))
                )
        except: pass
    return (["gig", "salaried", "self_employed"], ["0-15000", "10000-30000", "30000-50000", "50000-100000"], [1, 2, 3])

def predict_all(input_df, lr_sess, xgb_sess, rf_sess):
    lr_risk, lr_probs = onnx_predict_classifier_label_and_proba(lr_sess, input_df)
    risk_to_score = {"Low Risk": 85, "Medium Risk": 55, "High Risk": 25}
    lr_score = int(risk_to_score.get(lr_risk, 50))
    xgb_score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
    rf_score  = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))

    if lr_risk == "High Risk": final_score = min(xgb_score, rf_score)
    elif lr_risk == "Low Risk": final_score = max(xgb_score, rf_score)
    else: final_score = round((xgb_score + rf_score) / 2)

    if final_score >= 70:  eligibility, risk_level = "✅ ELIGIBLE",    "Low Risk"
    elif final_score >= 40: eligibility, risk_level = "⚠️ CONDITIONAL", "Medium Risk"
    else:                   eligibility, risk_level = "❌ RISKY",        "High Risk"

    return {
        "lr_risk": lr_risk, "lr_probs": lr_probs, "lr_score": lr_score,
        "xgb_score": xgb_score, "rf_score": rf_score, "final_score": final_score,
        "eligibility": eligibility, "risk_level": risk_level,
    }

# --- App Init ---
ensure_dataset_file()
lr_sess, xgb_sess, rf_sess = load_models()
employment_options, income_options, city_tier_options = get_dropdown_options_from_dataset()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>AI Credit Intelligence</div>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠  Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊  Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.write("---")
    st.markdown("<p style='text-align:center;color:#2a1550;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div style='margin-bottom: 0;'>
        <span class='page-eyebrow'>✦ New Application</span>
        <h1 class='page-title'>Register <span class='grad'>User Profile</span></h1>
        <p class='page-desc'>Complete the financial profile below to generate an instant AI-powered alternative credit score tailored to your unique financial footprint.</p>
    </div>
    <div class='header-rule'></div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("user_registration_form", border=False):

    # Section 1
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📍</div>Basic Profile</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: employment_type = st.selectbox("Employment Type", employment_options)
    with c2: income_range    = st.selectbox("Income Range (Monthly)", income_options)
    with c3: city_tier       = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Section 2
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>💰</div>Financial Capacity</div>
    </div>""", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        monthly_income          = st.number_input("Exact Monthly Income (₹)", min_value=0, value=30000, step=1000)
        bank_account_age_months = st.number_input("Account Age (Months)", min_value=0, max_value=240, value=24)
    with c5:
        num_bank_accounts       = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance   = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Section 3
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>⚡</div>Digital Footprint & Reliability</div>
    </div>""", unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        upi_txn_count      = st.number_input("Monthly UPI Transactions", min_value=0.0, value=20.0)
        utility_delay_days = st.number_input("Utility Delay Days", min_value=0.0, value=0.0)
    with c7:
        overdraft_event  = st.radio("Overdraft Facility Used?", ["No", "Yes"], horizontal=True)
        pays_rent_toggle = st.toggle("Monthly Rent Payer", value=True)
        if pays_rent_toggle:
            rent_paid_on_time = st.slider("Rent Timeliness Score", 0.0, 1.0, 1.0, 0.05)
        else:
            rent_paid_on_time = 1.0
            st.info("Non-renter: Neutral behaviour applied.")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✦  Analyze & Generate Score")

# --- Submission ---
if submitted:
    user_id  = generate_user_id()
    input_df = pd.DataFrame([{
        "employment_type":       str(employment_type).strip().lower(),
        "income_range":          str(income_range).strip().lower(),
        "city_tier":             int(city_tier),
        "bank_account_age_months": int(bank_account_age_months),
        "num_bank_accounts":     int(num_bank_accounts),
        "monthly_income":        float(monthly_income),
        "rent_paid_on_time":     float(rent_paid_on_time),
        "utility_delay_days":    float(utility_delay_days),
        "upi_txn_count":         float(upi_txn_count),
        "avg_month_end_balance": float(avg_month_end_balance),
        "overdraft_event":       1 if overdraft_event == "Yes" else 0,
    }])

    with st.status("🧠  Running AI analysis...", expanded=True) as status:
        st.write("Initialising ONNX inference sessions...")
        import time; time.sleep(1)
        out = predict_all(input_df, lr_sess, xgb_sess, rf_sess)
        st.write("Aggregating multi-model insights...")
        status.update(label="Analysis complete!", state="complete", expanded=False)

        st.session_state["report_data"] = {
            "user_id": user_id, "lr": out["lr_score"], "xgb": out["xgb_score"],
            "rf": out["rf_score"], "final": out["final_score"],
            "lr_risk": out["lr_risk"], "lr_probs": out["lr_probs"],
            "eligibility": out["eligibility"], "risk_level": out["risk_level"],
        }

        new_entry = input_df.iloc[0].to_dict()
        new_entry.update({"user_id": user_id, "alt_credit_score": out["final_score"]})
        df_csv = pd.read_csv(DATA_FILE)
        df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
        df_csv.to_csv(DATA_FILE, index=False)

        st.toast(f"User {user_id} saved successfully!", icon='✅')
        st.switch_page("pages/user_report_page.py")
