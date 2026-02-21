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
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:ital,wght@0,700;0,900;1,700&display=swap');

    /* ══════════════════════════════════════════
       Hides the default Streamlit Page List
    ══════════════════════════════════════════ */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* GLOBAL */
    * { font-family: 'Manrope', sans-serif !important; box-sizing: border-box; }

    /* APP BG — blue gradient + money SVG watermark */
    .stApp {
        background-color: #e3f2fd;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' viewBox='0 0 1000 700'%3E%3Crect x='680' y='30' width='280' height='140' rx='18' fill='none' stroke='%231565c0' stroke-width='2' opacity='0.18'/%3E%3Ccircle cx='820' cy='100' r='30' fill='none' stroke='%231565c0' stroke-width='1.5' opacity='0.18'/%3E%3Ctext x='808' y='107' font-family='monospace' font-size='18' fill='%231565c0' opacity='0.28' font-weight='bold'%3E₹%3C/text%3E%3C/svg%3E"),
            radial-gradient(ellipse 55vw 45vh at 5%  5%,  rgba(21,101,192,0.10) 0%, transparent 65%),
            linear-gradient(150deg, #e3f2fd 0%, #eff7ff 40%, #e3f2fd 70%, #eaf4fd 100%);
        background-size: 1000px 700px, cover, cover;
        background-repeat: repeat, no-repeat, no-repeat;
        color: #071a2e;
    }

    /* SIDEBAR — deep navy dark */
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

    /* Styled Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(21,101,192,0.09) !important;
        border: 1px solid rgba(21,101,192,0.2) !important;
        color: #90caf9 !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        border-radius: 10px !important;
        transition: 0.2s all ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #00D1FF !important;
        color: #000 !important;
        border-color: #00D1FF !important;
    }

    /* MAIN CONTENT CARD */
    section.main > div.block-container {
        background: rgba(255, 255, 255, 0.93) !important;
        border-radius: 22px !important;
        border: 1px solid rgba(21,101,192,0.13) !important;
        padding: 44px 52px !important;
        backdrop-filter: blur(8px) !important;
    }

    .page-title {
        font-family: 'Fraunces', serif !important;
        font-size: clamp(2.6rem, 4.5vw, 3.8rem);
        font-weight: 900;
        color: #071a2e;
        margin: 0 0 16px 0;
    }
    .page-title .grad { color: #1565c0; }

    /* SECTION CARDS */
    .sec-card {
        background: #eff7ff;
        border: 1px solid rgba(21,101,192,0.13);
        border-radius: 16px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 18px;
    }
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
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>AI Credit Intelligence</div>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠  Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊  Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.write("---")
    st.markdown("<p style='text-align:center;color:#0d2a4a;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

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
