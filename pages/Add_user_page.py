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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ══════════════════════════════════════
       ANIMATED LIGHT BACKGROUND
    ══════════════════════════════════════ */
    .stApp {
        background: #f0f4ff;
        font-family: 'DM Sans', sans-serif;
        color: #1a2340;
        overflow-x: hidden;
    }

    /* Animated gradient mesh background */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 80vw 60vh at 10% 10%, rgba(99, 179, 255, 0.22) 0%, transparent 60%),
            radial-gradient(ellipse 60vw 70vh at 90% 20%, rgba(167, 139, 250, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse 70vw 50vh at 50% 90%, rgba(52, 211, 153, 0.15) 0%, transparent 60%),
            radial-gradient(ellipse 50vw 60vh at 80% 70%, rgba(251, 191, 36, 0.10) 0%, transparent 55%),
            linear-gradient(135deg, #eef2ff 0%, #f5f0ff 50%, #ecfdf5 100%);
        animation: meshShift 12s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes meshShift {
        0%   { filter: hue-rotate(0deg) brightness(1); }
        50%  { filter: hue-rotate(15deg) brightness(1.03); }
        100% { filter: hue-rotate(-10deg) brightness(0.98); }
    }

    /* Floating decorative circle */
    .stApp > div:first-child::after {
        content: '';
        position: fixed;
        top: 5%;
        right: 8%;
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(129, 140, 248, 0.12), rgba(52, 211, 153, 0.08));
        animation: float1 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
        border: 1px solid rgba(129, 140, 248, 0.15);
    }

    @keyframes float1 {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50%       { transform: translateY(-20px) rotate(5deg); }
    }

    @keyframes cardRise {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ══════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.80) !important;
        border-right: 1px solid rgba(129, 140, 248, 0.18) !important;
        backdrop-filter: blur(20px) !important;
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.9rem;
        letter-spacing: 0.12em;
        text-align: center;
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .sidebar-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 0.68rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(99, 102, 241, 0.06) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        color: #4b5563 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.88rem !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.12) !important;
        border-color: rgba(99, 102, 241, 0.35) !important;
        color: #4f46e5 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════ */
    .page-eyebrow {
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 3.5vw, 2.8rem);
        font-weight: 800;
        color: #1e1b4b;
        line-height: 1.15;
        margin: 0 0 10px 0;
    }

    .page-title .grad {
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .page-desc {
        color: #64748b;
        font-size: 0.98rem;
        font-weight: 400;
        max-width: 500px;
        line-height: 1.75;
    }

    /* ══════════════════════════════════════
       SECTION CARDS
    ══════════════════════════════════════ */
    .sec-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 20px;
        padding: 22px 26px 8px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(16px);
        box-shadow: 0 2px 20px rgba(99, 102, 241, 0.07), 0 1px 4px rgba(0,0,0,0.03);
        animation: cardRise 0.5s ease both;
        position: relative;
        overflow: hidden;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #06b6d4, #34d399);
        opacity: 0.7;
    }

    .sec-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6366f1;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
    }

    .sec-icon {
        width: 26px;
        height: 26px;
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(6,182,212,0.12));
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
    }

    /* ══════════════════════════════════════
       INPUTS — WHITE & READABLE
    ══════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: #ffffff !important;
        color: #1e1b4b !important;
        border: 1.5px solid #c7d2fe !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #ffffff !important;
        border: 1.5px solid #c7d2fe !important;
        border-radius: 10px !important;
        color: #1e1b4b !important;
    }

    /* All labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    /* Radio option text */
    .stRadio [role="radiogroup"] label div p {
        color: #374151 !important;
        font-size: 0.9rem !important;
    }

    /* Slider thumb */
    [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
    }

    /* Info box */
    [data-testid="stInfo"] {
        background: rgba(99, 102, 241, 0.05) !important;
        border: 1px solid rgba(99, 102, 241, 0.18) !important;
        border-radius: 10px !important;
        color: #4338ca !important;
    }

    /* ══════════════════════════════════════
       SUBMIT BUTTON
    ══════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        padding: 17px 40px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        color: #ffffff !important;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.28);
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.42);
    }

    /* Status */
    [data-testid="stStatus"] {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #4338ca !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(99, 102, 241, 0.1) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.2); border-radius: 3px; }
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

# --- App Init ---
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
    st.markdown("<p style='text-align:center;color:#cbd5e1;font-size:0.68rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div style='margin-bottom: 36px; animation: cardRise 0.4s ease both;'>
        <div class='page-eyebrow'>New Application</div>
        <h1 class='page-title'>Register <span class='grad'>User Profile</span></h1>
        <p class='page-desc'>Complete the financial profile below to generate an instant AI-powered alternative credit score.</p>
    </div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("user_registration_form", border=False):

    # Section 1
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📍</div> Basic Profile</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: employment_type = st.selectbox("Employment Type", employment_options)
    with c2: income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3: city_tier = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Section 2
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>💰</div> Financial Capacity</div>
    </div>""", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        monthly_income = st.number_input("Exact Monthly Income (₹)", min_value=0, value=30000, step=1000)
        bank_account_age_months = st.number_input("Account Age (Months)", min_value=0, max_value=240, value=24)
    with c5:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Section 3
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>⚡</div> Digital Footprint & Reliability</div>
    </div>""", unsafe_allow_html=True)
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

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("✦  Analyze & Generate Score")

# --- Submission ---
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
