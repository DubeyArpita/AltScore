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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* ══════════════════════════════════════
       GLOBAL — light grey background
    ══════════════════════════════════════ */
    * { font-family: 'Poppins', sans-serif !important; box-sizing: border-box; }

    .stApp {
        background: #f0f2f8 !important;
        color: #1a1f3c;
    }

    /* ══════════════════════════════════════
       SIDEBAR — dark navy, exact match
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #1a1f3c !important;
        border-right: none !important;
        box-shadow: 3px 0 18px rgba(0,0,0,0.35) !important;
        min-width: 200px !important;
    }

    [data-testid="stSidebarNav"] { background: transparent !important; }

    .sidebar-logo {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        line-height: 1.3;
        padding: 8px 4px 4px 4px;
        letter-spacing: 0.01em;
    }

    .sidebar-logo span {
        display: block;
        font-size: 0.75rem;
        font-weight: 400;
        color: #8892b0;
        margin-top: 2px;
        letter-spacing: 0.05em;
    }

    /* Sidebar nav buttons — orange gradient like the screenshot */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #f7971e, #f4455a) !important;
        border: none !important;
        color: #ffffff !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        border-radius: 8px !important;
        padding: 9px 14px !important;
        text-transform: none !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 3px 10px rgba(244,69,90,0.3) !important;
        transition: opacity 0.2s ease, transform 0.15s ease !important;
        width: 100% !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        opacity: 0.88 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 14px rgba(244,69,90,0.4) !important;
    }

    /* ══════════════════════════════════════
       MAIN CONTENT WRAPPER
    ══════════════════════════════════════ */
    section.main > div.block-container {
        background: transparent !important;
        border-radius: 0 !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 28px 40px 28px !important;
        margin: 0 !important;
        max-width: 100% !important;
    }

    /* ══════════════════════════════════════
       HERO BANNER — coral-to-pink gradient
    ══════════════════════════════════════ */
    .hero-banner {
        background: linear-gradient(135deg, #f7971e 0%, #f4455a 55%, #c850c0 100%);
        border-radius: 18px;
        padding: 38px 48px 36px 48px;
        margin-bottom: 28px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(244,69,90,0.22);
        position: relative;
        overflow: hidden;
    }

    /* subtle white shimmer overlay */
    .hero-banner::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(255,255,255,0.07) 0%, transparent 60%);
        border-radius: 18px;
    }

    .hero-title {
        font-size: clamp(1.8rem, 3.5vw, 2.8rem);
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.01em;
        line-height: 1.15;
        text-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }

    .hero-sub {
        font-size: 1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.88);
        margin: 0;
        line-height: 1.6;
        max-width: 520px;
        margin: 0 auto;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #fff;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* ══════════════════════════════════════
       SECTION CARDS — white with purple accent
    ══════════════════════════════════════ */
    .sec-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px 26px 14px 26px;
        margin-bottom: 18px;
        box-shadow: 0 2px 14px rgba(26,31,60,0.07);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(108,92,231,0.08);
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        border-radius: 14px 14px 0 0;
    }

    .sec-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #6c5ce7;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f0f0f8;
    }

    .sec-icon {
        width: 26px; height: 26px;
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════
       INPUTS — clean white
    ══════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: #f8f8fc !important;
        color: #1a1f3c !important;
        border: 1.5px solid #e0ddf7 !important;
        border-radius: 9px !important;
        font-size: 0.96rem !important;
        font-weight: 500 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #6c5ce7 !important;
        box-shadow: 0 0 0 3px rgba(108,92,231,0.1) !important;
        background: #fff !important;
    }

    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #f8f8fc !important;
        border: 1.5px solid #e0ddf7 !important;
        border-radius: 9px !important;
        color: #1a1f3c !important;
        font-weight: 500 !important;
        font-size: 0.96rem !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] [role="listbox"] {
        background: #fff !important;
        border: 1px solid #e0ddf7 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(26,31,60,0.12) !important;
    }

    [data-baseweb="popover"] [role="option"] { color: #1a1f3c !important; }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: #ede9fe !important;
        color: #6c5ce7 !important;
    }

    /* Labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #4a4a6a !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    .stRadio [role="radiogroup"] label div p {
        color: #2d2d50 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }

    /* Slider */
    [data-baseweb="slider"] [role="slider"] {
        background: #6c5ce7 !important;
        border: 2px solid #fff !important;
        box-shadow: 0 0 8px rgba(108,92,231,0.4) !important;
    }

    /* Info */
    [data-testid="stInfo"] {
        background: #f5f3ff !important;
        border: 1px solid #ddd6fe !important;
        border-radius: 9px !important;
        color: #5b21b6 !important;
        font-weight: 500 !important;
    }

    /* Toggle */
    div[data-testid="stToggle"] p {
        color: #4a4a6a !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Number input step buttons */
    .stNumberInput [data-testid="stNumberInputContainer"] button {
        background: #ede9fe !important;
        border-color: #e0ddf7 !important;
        color: #6c5ce7 !important;
    }

    /* ══════════════════════════════════════
       CENTERED SUBMIT BUTTON — purple gradient pill
    ══════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 300px !important;
        padding: 15px 60px !important;
        border-radius: 50px !important;
        border: none !important;
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        box-shadow: 0 6px 24px rgba(108,92,231,0.35) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(108,92,231,0.5) !important;
    }

    /* ══════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: #ffffff !important;
        border: 1px solid #e0ddf7 !important;
        border-radius: 12px !important;
        color: #6c5ce7 !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(108,92,231,0.1) !important;
        margin: 12px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(108,92,231,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(108,92,231,0.38); }
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

    if final_score >= 70:   eligibility, risk_level = "✅ ELIGIBLE",    "Low Risk"
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
    st.markdown("""
        <div class='sidebar-logo'>
            Credit Analysis<br>
            <span>Register User</span>
        </div>
    """, unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠  Back to Home",      use_container_width=True): st.switch_page("app.py")
    if st.button("📝  New User Registration", use_container_width=True): st.switch_page("pages/register_page.py")
    if st.button("📊  Dashboard",         use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.write("---")
    st.markdown("<p style='text-align:center;color:#3d4270;font-size:0.62rem;letter-spacing:0.08em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Hero Banner ---
st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>✦ New Application</div>
        <h1 class='hero-title'>Register New User Profile</h1>
        <p class='hero-sub'>Complete the financial profile below to generate an instant AI-powered alternative credit score tailored to your unique financial footprint.</p>
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
    with c2: income_range    = st.selectbox("Income Range (Monthly)", income_options)
    with c3: city_tier       = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Section 2 — Financial Capacity
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

    # Section 3 — Digital Footprint
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
        "employment_type":         str(employment_type).strip().lower(),
        "income_range":            str(income_range).strip().lower(),
        "city_tier":               int(city_tier),
        "bank_account_age_months": int(bank_account_age_months),
        "num_bank_accounts":       int(num_bank_accounts),
        "monthly_income":          float(monthly_income),
        "rent_paid_on_time":       float(rent_paid_on_time),
        "utility_delay_days":      float(utility_delay_days),
        "upi_txn_count":           float(upi_txn_count),
        "avg_month_end_balance":   float(avg_month_end_balance),
        "overdraft_event":         1 if overdraft_event == "Yes" else 0,
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
