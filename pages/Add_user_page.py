import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

st.set_page_config(page_title="Register User", layout="centered")

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
                    n = int(m.group(1))
                    return f"USER_{n+1:04d}"
        except Exception:
            pass
    return "USER_0001"

def get_dropdown_options_from_dataset():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if not df.empty:
                emp = sorted(df["employment_type"].dropna().unique().tolist())
                inc = sorted(df["income_range"].dropna().unique().tolist())
                tiers = sorted(list(set([int(x) for x in df["city_tier"].dropna().tolist()])))
                if emp and inc and tiers:
                    return emp, inc, tiers
        except Exception:
            pass
    return (["gig", "salaried", "self_employed"], ["0-15000", "10000-30000", "30000-50000", "50000-100000"], [1,2,3])

def predict_all(input_df, lr_sess, xgb_sess, rf_sess):
    lr_risk, lr_probs = onnx_predict_classifier_label_and_proba(lr_sess, input_df)

    # Convert risk -> score (your rule)
    risk_to_score = {"Low Risk": 85, "Medium Risk": 55, "High Risk": 25}
    lr_score = int(risk_to_score.get(lr_risk, 50))

    xgb_score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
    rf_score  = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))

    final_score = int(round((lr_score + xgb_score + rf_score) / 3))

    if final_score >= 70:
        eligibility = "✅ ELIGIBLE"
        risk_level = "Low Risk"
    elif final_score >= 40:
        eligibility = "⚠️ CONDITIONAL"
        risk_level = "Medium Risk"
    else:
        eligibility = "❌ RISKY"
        risk_level = "High Risk"

    return {
        "lr_risk": lr_risk,
        "lr_probs": lr_probs,
        "lr_score": lr_score,
        "xgb_score": xgb_score,
        "rf_score": rf_score,
        "final_score": final_score,
        "eligibility": eligibility,
        "risk_level": risk_level,
    }

# -------- startup
ensure_dataset_file()
lr_sess, xgb_sess, rf_sess = load_models()
employment_options, income_options, city_tier_options = get_dropdown_options_from_dataset()

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    st.write("---")

st.markdown("<h1>📝 User Registration</h1>", unsafe_allow_html=True)

pays_rent = st.selectbox("Do you pay rent?", ["No", "Yes"], index=0)

with st.form("user_registration_form"):

    # =========================
    # BASIC DETAILS
    # =========================
    st.subheader("Basic Details")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        employment_type = st.selectbox("Employment Type *", employment_options)
    with col2:
        income_range = st.selectbox("Income Range (Monthly) *", income_options)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        city_tier = st.selectbox("City Tier *", city_tier_options)
    with col4:
        bank_account_age_months = st.number_input(
            "Bank Account Age (Months) *",
            min_value=0, max_value=240, value=24, step=1
        )

    st.divider()

    # =========================
    # INCOME & TRANSACTIONS
    # =========================
    st.subheader("Income & Transactions")

    col5, col6 = st.columns(2, gap="large")
    with col5:
        monthly_income = st.number_input(
            "Monthly Income (₹) *",
            min_value=0, value=30000, step=1000
        )
    with col6:
        num_bank_accounts = st.number_input(
            "Number of Bank Accounts *",
            min_value=1, max_value=15, value=1, step=1
        )

    col7, col8 = st.columns(2, gap="large")
    with col7:
        upi_txn_count = st.number_input(
            "Monthly UPI Transaction Count *",
            min_value=0.0, value=20.0, step=1.0
        )
    with col8:
        avg_month_end_balance = st.number_input(
            "Average Month-End Balance (₹) *",
            min_value=0.0, value=5000.0, step=100.0
        )

    st.divider()

    # =========================
    # PAYMENT BEHAVIOR
    # =========================

    
    st.subheader("Payment Behavior")

    col9, col10 = st.columns(2, gap="large")
    with col9:
        
        if pays_rent == "Yes":
            rent_paid_on_time = st.slider("Rent Paid On Time (0 to 1)", 0.0, 1.0, 1.0, 0.1)

        if pays_rent == "No":
            st.info("Rent not applicable. We will treat rent behavior as neutral during scoring.")

    with col10:
        utility_delay_days = st.number_input(
            "Utility Delay Days *",
            min_value=0.0, value=0.0, step=1.0
        )

    col11, col12 = st.columns(2, gap="large")
    with col11:
        overdraft_event = st.selectbox(
            "Overdraft Availed? *",
            ["No", "Yes"]
        )
    with col12:
        st.markdown(" ")  # keeps grid symmetric
        st.caption("Overdraft usage may increase credit risk.")

    st.write("")
    submitted = st.form_submit_button(
        "💾 Save User & Generate Score 🚀",
        use_container_width=True
    )


if submitted:
    user_id = generate_user_id()
    if pays_rent == "No":
        rent_paid_on_time = 1.0

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

    with st.spinner("Generating score..."):
        out = predict_all(input_df, lr_sess, xgb_sess, rf_sess)

        st.session_state["report_data"] = {
            "user_id": user_id,
            "lr": out["lr_score"],
            "xgb": out["xgb_score"],
            "rf": out["rf_score"],
            "final": out["final_score"],
            "lr_risk": out["lr_risk"],
            "lr_probs": out["lr_probs"],
            "eligibility": out["eligibility"],
            "risk_level": out["risk_level"],
        }

        new_entry = input_df.iloc[0].to_dict()
        new_entry["user_id"] = user_id
        new_entry["alt_credit_score"] = out["final_score"]

        df_csv = pd.read_csv(DATA_FILE)
        df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
        df_csv.to_csv(DATA_FILE, index=False)

        st.success(f"✅ User {user_id} registered successfully!")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Logistic Regression", f"{out['lr_score']}", out["lr_risk"])
        with c2: st.metric("XGBoost", f"{out['xgb_score']:.1f}")
        with c3: st.metric("Random Forest", f"{out['rf_score']:.1f}")
        with c4: st.metric("Final Score", f"{out['final_score']}", out["risk_level"])

        st.switch_page("pages/user_report_page.py")
