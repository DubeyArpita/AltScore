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
    @import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Cabinet+Grotesk:wght@400;500;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ══════════════════════════════════════════
       BASE
    ══════════════════════════════════════════ */
    .stApp {
        background: #07050f;
        font-family: 'Outfit', sans-serif;
        color: #e2e0f0;
        overflow-x: hidden;
    }

    /* ══════════════════════════════════════════
       ANIMATED ORB BACKGROUND — red, purple, blue, yellow, pink
    ══════════════════════════════════════════ */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 50vw 45vh at 8%  12%,  rgba(239, 68,  68,  0.18) 0%, transparent 60%),
            radial-gradient(ellipse 45vw 40vh at 88% 18%,  rgba(168, 85,  247, 0.16) 0%, transparent 60%),
            radial-gradient(ellipse 55vw 40vh at 75% 80%,  rgba(59,  130, 246, 0.15) 0%, transparent 60%),
            radial-gradient(ellipse 40vw 35vh at 20% 78%,  rgba(234, 179, 8,   0.12) 0%, transparent 55%),
            radial-gradient(ellipse 35vw 50vh at 50% 45%,  rgba(236, 72,  153, 0.10) 0%, transparent 55%);
        animation: orbPulse 14s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes orbPulse {
        0%   { opacity: 1;    transform: scale(1)    rotate(0deg); }
        33%  { opacity: 0.88; transform: scale(1.05) rotate(1deg); }
        66%  { opacity: 0.95; transform: scale(0.97) rotate(-1deg); }
        100% { opacity: 1;    transform: scale(1.03) rotate(0.5deg); }
    }

    /* Floating glowing blobs that drift independently */
    .blob {
        position: fixed;
        border-radius: 50%;
        filter: blur(72px);
        opacity: 0.22;
        pointer-events: none;
        z-index: 0;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-direction: alternate;
    }

    /* Inline blobs injected via HTML below */

    @keyframes blobDrift1 {
        0%   { transform: translate(0,   0)   scale(1); }
        100% { transform: translate(40px, 55px) scale(1.12); }
    }
    @keyframes blobDrift2 {
        0%   { transform: translate(0,   0)   scale(1); }
        100% { transform: translate(-50px, 30px) scale(0.92); }
    }
    @keyframes blobDrift3 {
        0%   { transform: translate(0,   0)   scale(1); }
        100% { transform: translate(30px, -40px) scale(1.08); }
    }
    @keyframes blobDrift4 {
        0%   { transform: translate(0,   0)   scale(1); }
        100% { transform: translate(-30px, -35px) scale(1.05); }
    }
    @keyframes blobDrift5 {
        0%   { transform: translate(0,   0)   scale(1); }
        100% { transform: translate(20px, 45px) scale(0.95); }
    }

    /* Page enter animation */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes cardRise {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ══════════════════════════════════════════
       SIDEBAR — SOLID DARK
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #060409 !important;
        border-right: 1px solid rgba(168, 85, 247, 0.12) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.6) !important;
    }

    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.9rem;
        letter-spacing: 0.14em;
        text-align: center;
        background: linear-gradient(135deg, #f472b6 0%, #a855f7 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        text-align: center;
        color: #2d2440;
        font-size: 0.63rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-bottom: 20px;
        font-family: 'Outfit', sans-serif;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #4a3f60 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        font-family: 'Outfit', sans-serif !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(168, 85, 247, 0.08) !important;
        border-color: rgba(168, 85, 247, 0.25) !important;
        color: #c084fc !important;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.1) !important;
        transform: none !important;
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    .page-header-wrap {
        margin-bottom: 36px;
        animation: fadeUp 0.5s ease both;
    }

    .page-eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #f472b6, #a855f7, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        margin-bottom: 10px;
        display: block;
    }

    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.4rem, 4vw, 3.4rem);
        font-weight: 700;
        color: #f1eeff;
        line-height: 1.1;
        margin: 0 0 14px 0;
        letter-spacing: -0.01em;
    }

    .page-title .grad {
        background: linear-gradient(100deg, #f472b6 0%, #a855f7 45%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .page-desc {
        color: #8b7fa8;
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 560px;
        line-height: 1.85;
        letter-spacing: 0.01em;
    }

    /* ══════════════════════════════════════════
       SECTION CARDS — GLASSMORPHIC
    ══════════════════════════════════════════ */
    .sec-card {
        background: rgba(255, 255, 255, 0.028);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 24px 28px 12px 28px;
        margin-bottom: 18px;
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        box-shadow:
            0 0 0 1px rgba(168, 85, 247, 0.04),
            0 6px 40px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        animation: cardRise 0.5s ease both;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 20px; right: 20px;
        height: 1px;
        background: linear-gradient(90deg,
            transparent,
            rgba(244, 114, 182, 0.5),
            rgba(168, 85, 247, 0.6),
            rgba(96, 165, 250, 0.5),
            transparent
        );
    }

    .sec-card:hover {
        border-color: rgba(168, 85, 247, 0.14);
        box-shadow:
            0 0 0 1px rgba(168, 85, 247, 0.08),
            0 10px 50px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(168, 85, 247, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.07);
    }

    .sec-card:nth-child(2) { animation-delay: 0.1s; }
    .sec-card:nth-child(3) { animation-delay: 0.2s; }

    .sec-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #f472b6, #a855f7, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .sec-icon {
        width: 26px;
        height: 26px;
        background: linear-gradient(135deg, rgba(244,114,182,0.15), rgba(168,85,247,0.18), rgba(96,165,250,0.12));
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        flex-shrink: 0;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.12);
    }

    /* ══════════════════════════════════════════
       INPUTS
    ══════════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: rgba(240, 235, 255, 0.9) !important;
        color: #1a1030 !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.97rem !important;
        font-weight: 500 !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(168, 85, 247, 0.55) !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.12), 0 0 14px rgba(168, 85, 247, 0.1) !important;
        background: rgba(248, 244, 255, 0.97) !important;
    }

    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: rgba(240, 235, 255, 0.9) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 10px !important;
        color: #1a1030 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
    }

    /* All labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #9d8fbe !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    .stRadio [role="radiogroup"] label div p {
        color: #9d8fbe !important;
        font-size: 0.92rem !important;
    }

    [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #f472b6, #a855f7, #60a5fa) !important;
        border: 2px solid rgba(255,255,255,0.18) !important;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.5) !important;
    }

    [data-testid="stInfo"] {
        background: rgba(168, 85, 247, 0.07) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 10px !important;
        color: #c084fc !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ══════════════════════════════════════════
       CENTERED SUBMIT BUTTON
    ══════════════════════════════════════════ */
    .btn-center-wrap {
        display: flex;
        justify-content: center;
        margin-top: 8px;
    }

    /* Target the submit button specifically and center it */
    .stForm [data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 320px !important;
        padding: 17px 52px !important;
        border-radius: 50px !important;
        border: 1px solid rgba(244, 114, 182, 0.3) !important;
        background: linear-gradient(135deg,
            rgba(239,68,68,0.7) 0%,
            rgba(168,85,247,0.85) 35%,
            rgba(59,130,246,0.75) 70%,
            rgba(236,72,153,0.7) 100%
        ) !important;
        background-size: 200% 200% !important;
        animation: btnGradientShift 4s ease infinite !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow:
            0 0 28px rgba(168, 85, 247, 0.3),
            0 0 60px rgba(236, 72, 153, 0.12),
            0 6px 20px rgba(0,0,0,0.4) !important;
    }

    @keyframes btnGradientShift {
        0%   { background-position: 0%   50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0%   50%; }
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow:
            0 0 40px rgba(168, 85, 247, 0.5),
            0 0 80px rgba(236, 72, 153, 0.2),
            0 10px 30px rgba(0,0,0,0.5) !important;
    }

    /* ══════════════════════════════════════════
       STATUS & MISC
    ══════════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: rgba(10, 6, 20, 0.85) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 12px !important;
        color: #c084fc !important;
        backdrop-filter: blur(12px) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(168, 85, 247, 0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.38); }
    </style>

    <!-- Floating color blobs -->
    <div class="blob" style="
        width:420px; height:420px;
        background: radial-gradient(circle, rgba(239,68,68,0.55), transparent 70%);
        top: 5%; left: 3%;
        animation: blobDrift1 11s ease-in-out infinite alternate;
    "></div>
    <div class="blob" style="
        width:380px; height:380px;
        background: radial-gradient(circle, rgba(168,85,247,0.5), transparent 70%);
        top: 10%; right: 4%;
        animation: blobDrift2 13s ease-in-out infinite alternate;
    "></div>
    <div class="blob" style="
        width:360px; height:360px;
        background: radial-gradient(circle, rgba(59,130,246,0.45), transparent 70%);
        bottom: 12%; right: 8%;
        animation: blobDrift3 10s ease-in-out infinite alternate;
    "></div>
    <div class="blob" style="
        width:300px; height:300px;
        background: radial-gradient(circle, rgba(234,179,8,0.38), transparent 70%);
        bottom: 8%; left: 6%;
        animation: blobDrift4 15s ease-in-out infinite alternate;
    "></div>
    <div class="blob" style="
        width:340px; height:340px;
        background: radial-gradient(circle, rgba(236,72,153,0.42), transparent 70%);
        top: 45%; left: 42%;
        animation: blobDrift5 12s ease-in-out infinite alternate;
    "></div>
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
    st.markdown("<p style='text-align:center;color:#1a1030;font-size:0.65rem;letter-spacing:0.1em;font-family:Outfit,sans-serif;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div class='page-header-wrap'>
        <span class='page-eyebrow'>✦ New Application</span>
        <h1 class='page-title'>Register <span class='grad'>User Profile</span></h1>
        <p class='page-desc'>Complete the financial profile below to generate an instant AI-powered alternative credit score tailored to your unique financial footprint.</p>
    </div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("user_registration_form", border=False):

    # Section 1
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📍</div>Basic Profile</div>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: employment_type = st.selectbox("Employment Type", employment_options)
    with c2: income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3: city_tier = st.selectbox("City Tier", city_tier_options)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Section 2
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

    # Section 3
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

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
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
