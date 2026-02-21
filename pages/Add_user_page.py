import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# --- Page Config ---
st.set_page_config(page_title="AltScore | Register", layout="wide", page_icon="📝")

# --- UI Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ========== ANIMATED BACKGROUND ========== */
    .stApp {
        background: #060a14;
        font-family: 'DM Sans', sans-serif;
        color: #e2e8f0;
        overflow-x: hidden;
    }

    /* Aurora orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -20%;
        left: -10%;
        width: 70vw;
        height: 70vw;
        border-radius: 50%;
        background: radial-gradient(ellipse at center,
            rgba(0, 180, 255, 0.12) 0%,
            rgba(0, 100, 255, 0.07) 40%,
            transparent 70%
        );
        animation: orbDrift1 18s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: '';
        position: fixed;
        bottom: -20%;
        right: -10%;
        width: 60vw;
        height: 60vw;
        border-radius: 50%;
        background: radial-gradient(ellipse at center,
            rgba(120, 0, 255, 0.10) 0%,
            rgba(60, 0, 180, 0.06) 40%,
            transparent 70%
        );
        animation: orbDrift2 22s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes orbDrift1 {
        0%   { transform: translate(0, 0) scale(1); }
        50%  { transform: translate(5vw, 8vh) scale(1.1); }
        100% { transform: translate(-3vw, 4vh) scale(0.95); }
    }

    @keyframes orbDrift2 {
        0%   { transform: translate(0, 0) scale(1); }
        50%  { transform: translate(-6vw, -5vh) scale(1.15); }
        100% { transform: translate(4vw, 3vh) scale(0.9); }
    }

    /* Grid overlay texture */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0, 180, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 180, 255, 0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
    }

    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: rgba(6, 10, 20, 0.95) !important;
        border-right: 1px solid rgba(0, 180, 255, 0.12) !important;
        backdrop-filter: blur(20px);
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: 0.15em;
        background: linear-gradient(135deg, #00b4ff 0%, #7b2fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
    }

    .sidebar-sub {
        text-align: center;
        color: #4a6080;
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 24px;
    }

    /* ========== PAGE HEADER ========== */
    .page-header {
        margin-bottom: 40px;
    }

    .page-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #00b4ff;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin: 0 0 12px 0;
    }

    .page-title span {
        background: linear-gradient(90deg, #00b4ff, #7b2fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .page-desc {
        color: #64748b;
        font-size: 1rem;
        font-weight: 300;
        max-width: 520px;
        line-height: 1.7;
    }

    /* ========== SECTION CARDS ========== */
    .section-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease;
    }

    .section-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 255, 0.4), transparent);
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #00b4ff;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .section-icon {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, rgba(0,180,255,0.2), rgba(123,47,255,0.2));
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }

    /* ========== INPUTS ========== */
    .stNumberInput input,
    .stTextInput input {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.2s ease !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(0, 180, 255, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(0, 180, 255, 0.08) !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* Labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Toggle */
    .stToggle label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #00b4ff, #7b2fff) !important;
        border: none !important;
    }

    /* Radio */
    .stRadio div[role="radiogroup"] label {
        color: #94a3b8 !important;
    }

    /* Info box */
    .stInfo {
        background: rgba(0, 180, 255, 0.06) !important;
        border: 1px solid rgba(0, 180, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #7ecfff !important;
    }

    /* ========== SUBMIT BUTTON ========== */
    .stForm [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        padding: 18px 40px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #00b4ff 0%, #7b2fff 100%);
        color: #ffffff;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 24px rgba(0, 180, 255, 0.2);
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0, 180, 255, 0.35);
    }

    /* Sidebar nav buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #94a3b8 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.88rem !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0, 180, 255, 0.08) !important;
        border-color: rgba(0, 180, 255, 0.25) !important;
        color: #e2e8f0 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
        margin: 16px 0 !important;
    }

    /* Status widget */
    [data-testid="stStatus"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(0, 180, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #7ecfff !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 180, 255, 0.2); border-radius: 3px; }
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

    if final_score >= 70: eligibility, risk_level = "✅ ELIGIBLE", "Low Risk"
    elif final_score >= 40: eligibility, risk_level = "⚠️ CONDITIONAL", "Medium Risk"
    else: eligibility, risk_level = "❌ RISKY", "High Risk"

    return {
        "lr_risk": lr_risk, "lr_probs": lr_probs, "lr_score": lr_score,
        "xgb_score": xgb_score, "rf_score": rf_score, "final_score": final_score,
        "eligibility": eligibility, "risk_level": risk_level,
    }

# --- App Initialization ---
ensure_dataset_file()
lr_sess, xgb_sess, rf_sess = load_models()
employment_options, income_options, city_tier_options = get_dropdown_options_from_dataset()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>ALTSCORE</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>AI Credit Intelligence</div>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠  Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊  Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.write("---")
    st.markdown("<p style='text-align:center;color:#2d3f55;font-size:0.7rem;letter-spacing:0.08em;'>v2.1 · SECURE · ENCRYPTED</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div class='page-header'>
        <div class='page-eyebrow'>New Application</div>
        <h1 class='page-title'>Register <span>User Profile</span></h1>
        <p class='page-desc'>Complete the financial profile below to generate an instant AI-powered alternative credit score.</p>
    </div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("user_registration_form", border=False):

    # Section 1: Basic Profile
    st.markdown("""
        <div class='section-card'>
            <div class='section-title'>
                <div class='section-icon'>📍</div>
                Basic Profile
            </div>
        </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        employment_type = st.selectbox("Employment Type", employment_options)
    with c2:
        income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3:
        city_tier = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Section 2: Financial Capacity
    st.markdown("""
        <div class='section-card'>
            <div class='section-title'>
                <div class='section-icon'>💰</div>
                Financial Capacity
            </div>
        </div>
    """, unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        monthly_income = st.number_input("Exact Monthly Income (₹)", min_value=0, value=30000, step=1000)
        bank_account_age_months = st.number_input("Account Age (Months)", min_value=0, max_value=240, value=24)
    with c5:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Section 3: Digital Footprint
    st.markdown("""
        <div class='section-card'>
            <div class='section-title'>
                <div class='section-icon'>⚡</div>
                Digital Footprint & Reliability
            </div>
        </div>
    """, unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        upi_txn_count = st.number_input("Monthly UPI Transactions", min_value=0.0, value=20.0)
        utility_delay_days = st.number_input("Utility Delay Days", min_value=0.0, value=0.0)
    with c7:
        overdraft_event = st.radio("Overdraft Facility Used?", ["No", "Yes"], horizontal=True)
        pays_rent_toggle = st.toggle("Monthly Rent Payer", value=True)
        if pays_rent_toggle:
            rent_paid_on_time = st.slider("Rent Timeliness Score", 0.0, 1.0, 1.0, 0.05)
        else:
            rent_paid_on_time = 1.0
            st.info("Non-renter: Neutral behaviour applied.")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✦  Analyze & Generate Score")

# --- Form Submission ---
if submitted:
    user_id = generate_user_id()
    input_df = pd.DataFrame([{
        "employment_type": str(employment_type).strip().lower(),
        "income_range": str(income_range).strip().lower(),
        "city_tier": int(city_tier),
        "bank_account_age_months": int(bank_account_age_months),
        "num_bank_accounts": int(num_bank_accounts),
        "monthly_income": float(monthly_income),
        "rent_paid_on_time": float(rent_paid_on_time),
        "utility_delay_days": float(utility_delay_days),
        "upi_txn_count": float(upi_txn_count),
        "avg_month_end_balance": float(avg_month_end_balance),
        "overdraft_event": 1 if overdraft_event == "Yes" else 0,
    }])

    with st.status("🧠  Running AI analysis...", expanded=True) as status:
        st.write("Initialising ONNX inference sessions...")
        import time
        time.sleep(1)
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
