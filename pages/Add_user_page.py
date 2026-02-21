import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# --- Page Config ---
st.set_page_config(page_title="AltScore | Register", layout="wide", page_icon="📝")

# --- UI Styling: Deep Theme & Glassmorphism ---
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
    }

    /* Sidebar Panel Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    [data-testid="stSidebarNav"] {
        background-color: transparent !important;
    }

    /* Glassmorphic Card Container */
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Typography */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    .header-subtext {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Input Field Customization */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    /* Modernized Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #00d1ff 0%, #0076ff 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-weight: bold;
        font-size: 1rem;
        transition: 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0px 0px 20px rgba(0, 209, 255, 0.5);
        transform: translateY(-2px);
    }

    /* Sidebar buttons styling */
    section[data-testid="stSidebar"] .stButton>button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-transform: none;
        letter-spacing: 0;
        padding: 8px 15px;
    }

    /* Horizontal Rules */
    hr {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
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
    st.markdown("<h1 style='text-align: center; color: #00D1FF; font-size: 28px;'>ALTSCORE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>AI-Powered Credit Insights</p>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")
    st.write("---")
    st.caption("v2.1 | Secure & Encrypted")

# --- Main Layout ---
st.markdown("<h1>📝 Register New User</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtext'>Complete the financial profile to generate an instant alternative credit score.</p>", unsafe_allow_html=True)

with st.form("user_registration_form", border=False):
    
    # --- Section 1: Identity & Location ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Basic Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        employment_type = st.selectbox("Employment Type", employment_options)
    with c2:
        income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3:
        city_tier = st.selectbox("City Tier", city_tier_options)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Section 2: Financial Health ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Financial Capacity")
    c4, c5 = st.columns(2)
    with c4:
        monthly_income = st.number_input("Exact Monthly Income (₹)", min_value=0, value=30000, step=1000)
        bank_account_age_months = st.number_input("Account Age (Months)", min_value=0, max_value=240, value=24)
    with c5:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Section 3: Digital Behavior ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Digital Footprint & Reliability")
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
            st.info("Non-renter: Neutral behavior applied.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    submitted = st.form_submit_button("💾 Analyze & Generate Score")

# --- Form Submission Processing ---
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

    with st.status("🧠 Processing AI Models...", expanded=True) as status:
        st.write("Fetching ONNX sessions...")
        # out = predict_all(input_df, lr_sess, xgb_sess, rf_sess) # In reality use this
        # Simulate slight delay for "intelligence" feel
        import time
        time.sleep(1)
        out = predict_all(input_df, lr_sess, xgb_sess, rf_sess)
        st.write("Aggregating model insights...")
        status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Store in session state for report page
        st.session_state["report_data"] = {
            "user_id": user_id, "lr": out["lr_score"], "xgb": out["xgb_score"],
            "rf": out["rf_score"], "final": out["final_score"],
            "lr_risk": out["lr_risk"], "lr_probs": out["lr_probs"],
            "eligibility": out["eligibility"], "risk_level": out["risk_level"],
        }

        # Update CSV Database
        new_entry = input_df.iloc[0].to_dict()
        new_entry.update({"user_id": user_id, "alt_credit_score": out["final_score"]})
        df_csv = pd.read_csv(DATA_FILE)
        df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
        df_csv.to_csv(DATA_FILE, index=False)

        st.toast(f"User {user_id} saved successfully!", icon='✅')
        st.switch_page("pages/user_report_page.py")
