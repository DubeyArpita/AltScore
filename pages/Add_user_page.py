import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# --- Page Config ---
st.set_page_config(page_title="AltScore | Register", layout="wide", page_icon="📝")

# --- Custom CSS for Beauty ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        background-color: #00D1FF;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00B8E6;
        box-shadow: 0px 4px 15px rgba(0, 209, 255, 0.3);
    }
    .header-text {
        color: #1E1E1E;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- Initialization ---
ensure_dataset_file()
lr_sess, xgb_sess, rf_sess = load_models()
employment_options, income_options, city_tier_options = get_dropdown_options_from_dataset()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>ALTSCORE AI</h2>", unsafe_allow_html=True)
    st.info("Alternative Credit Scoring System using ONNX Runtime.")
    st.divider()
    if st.button("🏠 Home", use_container_width=True): st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True): st.switch_page("pages/dashboard_page.py")

# --- Main UI ---
st.markdown("<h1 class='header-text'>📝 User Registration</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: gray; margin-bottom: 2rem;'>Fill in the details below to compute the alternative credit score.</p>", unsafe_allow_html=True)

with st.form("user_registration_form", border=False):
    # Card 1: Basic Information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Basic Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        employment_type = st.selectbox("Employment Type", employment_options)
    with c2:
        income_range = st.selectbox("Income Range (Monthly)", income_options)
    with c3:
        city_tier = st.selectbox("City Tier", city_tier_options)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 2: Financial Profile
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Financial Profile")
    c1, c2 = st.columns(2)
    with c1:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=30000)
        bank_account_age_months = st.number_input("Bank Account Age (Months)", min_value=0, max_value=240, value=24)
    with c2:
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=1, max_value=15, value=1)
        avg_month_end_balance = st.number_input("Avg Month-End Balance (₹)", min_value=0.0, value=5000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 3: Behavioral Data
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Behavioral Data")
    c1, c2 = st.columns(2)
    with c1:
        upi_txn_count = st.number_input("Monthly UPI Transactions", min_value=0.0, value=20.0)
        utility_delay_days = st.number_input("Utility Delay Days", min_value=0.0, value=0.0)
    with c2:
        overdraft_event = st.radio("Overdraft Availed?", ["No", "Yes"], horizontal=True)
        pays_rent_toggle = st.toggle("I pay monthly rent", value=True)
        
        if pays_rent_toggle:
            rent_paid_on_time = st.slider("Rent Payment Timeliness (1.0 = Always On Time)", 0.0, 1.0, 1.0)
        else:
            rent_paid_on_time = 1.0
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    submitted = st.form_submit_button("🚀 GENERATE CREDIT SCORE", use_container_width=True)

# --- Logic on Submission ---
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

    with st.spinner("🧠 AI Models Analyzing Risk Profile..."):
        out = predict_all(input_df, lr_sess, xgb_sess, rf_sess)

        # Store in session
        st.session_state["report_data"] = {
            "user_id": user_id, "lr": out["lr_score"], "xgb": out["xgb_score"],
            "rf": out["rf_score"], "final": out["final_score"],
            "lr_risk": out["lr_risk"], "lr_probs": out["lr_probs"],
            "eligibility": out["eligibility"], "risk_level": out["risk_level"],
        }

        # Update CSV
        new_entry = input_df.iloc[0].to_dict()
        new_entry.update({"user_id": user_id, "alt_credit_score": out["final_score"]})
        df_csv = pd.read_csv(DATA_FILE)
        df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
        df_csv.to_csv(DATA_FILE, index=False)

        st.toast(f"User {user_id} saved!", icon='✅')
        st.switch_page("pages/user_report_page.py")
