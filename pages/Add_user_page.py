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

    /* ══════════════════════════════════════════
       GLOBAL BASE — Soft dark-neutral canvas
    ══════════════════════════════════════════ */
    .stApp {
        background: #0e1117;
        font-family: 'DM Sans', sans-serif;
        color: #cbd5e1;
        overflow-x: hidden;
        min-height: 100vh;
    }

    /* Ambient glow orbs — subtle, not blinding */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 55vw 45vh at 15% 15%, rgba(99, 102, 241, 0.09) 0%, transparent 65%),
            radial-gradient(ellipse 40vw 55vh at 85% 25%, rgba(6, 182, 212, 0.07) 0%, transparent 60%),
            radial-gradient(ellipse 50vw 40vh at 60% 85%, rgba(168, 85, 247, 0.07) 0%, transparent 60%);
        animation: ambientShift 16s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes ambientShift {
        0%   { opacity: 1; transform: scale(1); }
        50%  { opacity: 0.85; transform: scale(1.04); }
        100% { opacity: 1; transform: scale(0.97); }
    }

    /* Subtle noise texture overlay */
    .stApp > div:first-child::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
        opacity: 0.4;
        pointer-events: none;
        z-index: 1;
    }

    @keyframes cardRise {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ══════════════════════════════════════════
       SIDEBAR — DARK PANEL
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #080b11 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
        backdrop-filter: none !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }

    [data-testid="stSidebarNav"] {
        background: transparent !important;
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.8rem;
        letter-spacing: 0.14em;
        text-align: center;
        background: linear-gradient(135deg, #818cf8 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        text-align: center;
        color: #334155;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        margin-bottom: 22px;
    }

    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 14px 0;
    }

    .sidebar-version {
        text-align: center;
        color: #1e293b;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: #64748b !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.87rem !important;
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        color: #a5b4fc !important;
        transform: none !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.12) !important;
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    .page-eyebrow {
        font-size: 0.68rem;
        letter-spacing: 0.32em;
        text-transform: uppercase;
        color: #818cf8;
        font-weight: 600;
        margin-bottom: 8px;
        opacity: 0.9;
    }

    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(1.9rem, 3vw, 2.6rem);
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.15;
        margin: 0 0 10px 0;
    }

    .page-title .grad {
        background: linear-gradient(90deg, #818cf8, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .page-desc {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 400;
        max-width: 480px;
        line-height: 1.8;
    }

    /* ══════════════════════════════════════════
       GLASSMORPHIC SECTION CARDS
    ══════════════════════════════════════════ */
    .sec-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 16px;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow:
            0 0 0 1px rgba(99, 102, 241, 0.05),
            0 4px 32px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: cardRise 0.45s ease both;
        position: relative;
        overflow: hidden;
    }

    /* Glowing top border line */
    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 16px; right: 16px;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.5), rgba(34, 211, 238, 0.4), transparent);
    }

    /* Inner glow on hover */
    .sec-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        box-shadow:
            0 0 0 1px rgba(99, 102, 241, 0.1),
            0 8px 40px rgba(0, 0, 0, 0.3),
            0 0 30px rgba(99, 102, 241, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.07);
        transition: all 0.3s ease;
    }

    .sec-card:nth-child(2) { animation-delay: 0.08s; }
    .sec-card:nth-child(3) { animation-delay: 0.16s; }

    .sec-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #818cf8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .sec-icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(6,182,212,0.12));
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        box-shadow: 0 0 8px rgba(99, 102, 241, 0.15);
    }

    /* ══════════════════════════════════════════
       INPUTS — LIGHT FILL, DARK READABLE TEXT
    ══════════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: rgba(241, 245, 249, 0.92) !important;
        color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(129, 140, 248, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15), 0 0 12px rgba(99, 102, 241, 0.1) !important;
        background: #f8faff !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: rgba(241, 245, 249, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] [role="listbox"] {
        background: #1e293b !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
    }

    [data-baseweb="popover"] [role="option"] {
        color: #cbd5e1 !important;
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.12) !important;
        color: #a5b4fc !important;
    }

    /* All form labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: 0.03em !important;
    }

    /* Radio option text */
    .stRadio [role="radiogroup"] label div p {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
    }

    /* Slider */
    [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #818cf8, #22d3ee) !important;
        border: 2px solid rgba(255,255,255,0.15) !important;
        box-shadow: 0 0 10px rgba(129, 140, 248, 0.5) !important;
    }

    /* Toggle */
    [data-testid="stToggle"] [role="checkbox"] {
        background-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* Info box */
    [data-testid="stInfo"] {
        background: rgba(99, 102, 241, 0.07) !important;
        border: 1px solid rgba(99, 102, 241, 0.18) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
    }

    /* Number input arrows */
    .stNumberInput [data-testid="stNumberInputContainer"] button {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.1) !important;
        color: #94a3b8 !important;
    }

    /* ══════════════════════════════════════════
       SUBMIT BUTTON — GLOWING
    ══════════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] > button {
        width: 100%;
        padding: 17px 40px;
        border-radius: 14px;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        background: linear-gradient(135deg, rgba(99,102,241,0.85) 0%, rgba(6,182,212,0.75) 100%) !important;
        color: #ffffff !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.25), 0 4px 16px rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(8px) !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 32px rgba(99, 102, 241, 0.45), 0 8px 24px rgba(0,0,0,0.4) !important;
    }

    /* ══════════════════════════════════════════
       STATUS & MISC
    ══════════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #a5b4fc !important;
        backdrop-filter: blur(12px) !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 14px 0 !important;
    }

    /* Main content scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.35); }
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
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    if st.button("🏠  Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊  Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-version'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div style='margin-bottom: 32px; animation: cardRise 0.4s ease both;'>
        <div class='page-eyebrow'>New Application</div>
        <h1 class='page-title'>Register <span class='grad'>User Profile</span></h1>
        <p class='page-desc'>Complete the financial profile below to generate an instant AI-powered alternative credit score.</p>
    </div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("user_registration_form", border=False):

    # Section 1 — Basic Profile
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📍</div>Basic Profile</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: employment_type = st.selectbox("Employment Type", employment_options)
    with c2: income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3: city_tier = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Section 2 — Financial Capacity
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>💰</div>Financial Capacity</div>
    </div>""", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        monthly_income = st.number_input("Exact Monthly Income (₹)", min_value=0, value=30000, step=1000)
        bank_account_age_months = st.number_input("Account Age (Months)", min_value=0, max_value=240, value=24)
    with c5:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Section 3 — Digital Footprint
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>⚡</div>Digital Footprint & Reliability</div>
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

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
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
