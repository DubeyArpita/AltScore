import os
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(page_title="Credit Analytics Dashboard", layout="wide", page_icon="📊")

DATA_FILE = "data/dataset.csv"

# -------------------------------------
# Custom CSS Styling (Modern UI)
# -------------------------------------
st.markdown("""
<style>

/* Global font & text tweaks */
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f6f8fa;
}

h1, h2, h3, h4 {
    color: #222831;
    font-weight: 600;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1f2937;
    color: white;
}
[data-testid="stSidebar"] h2 {
    text-align: center;
    color: #00D1FF;
}
[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Metric cards */
div[data-testid="stMetricValue"] {
    color: #111827;
}
div.stMetric {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    text-align: center;
}

/* Dataframes */
table {
    border-radius: 8px!important;
}
thead tr th {
    background-color: #00D1FF;
    color: white;
    font-weight: 600;
}

/* General spacing */
.block-container {
    padding: 1rem 3rem;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------
# Cached model sessions
# -------------------------------------
@st.cache_resource
def load_models():
    return load_onnx_sessions()

# -------------------------------------
# Helpers
# -------------------------------------
def compute_risk_level(score):
    if pd.isna(score): return "Unknown"
    if score >= 70: return "Low"
    elif score >= 40: return "Medium"
    else: return "High"

def color_risk(val):
    if val == "Low": return "background-color: #6bcf7f; color: white; font-weight: bold;"
    elif val == "Medium": return "background-color: #FFD93D; color: black; font-weight: bold;"
    elif val == "High": return "background-color: #FF6B6B; color: white; font-weight: bold;"
    return ""

def build_input_df_from_row(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "employment_type": str(row.get("employment_type", "salaried")).strip().lower(),
        "income_range": str(row.get("income_range", "10000-30000")).strip().lower(),
        "city_tier": int(pd.to_numeric(row.get("city_tier", 2), errors="coerce") or 2),
        "bank_account_age_months": int(pd.to_numeric(row.get("bank_account_age_months", 24), errors="coerce") or 24),
        "num_bank_accounts": int(pd.to_numeric(row.get("num_bank_accounts", 1), errors="coerce") or 1),
        "monthly_income": float(pd.to_numeric(row.get("monthly_income", 30000), errors="coerce") or 30000),
        "rent_paid_on_time": float(pd.to_numeric(row.get("rent_paid_on_time", 1.0), errors="coerce") or 1.0),
        "utility_delay_days": float(pd.to_numeric(row.get("utility_delay_days", 0.0), errors="coerce") or 0.0),
        "upi_txn_count": float(pd.to_numeric(row.get("upi_txn_count", 20.0), errors="coerce") or 20.0),
        "avg_month_end_balance": float(pd.to_numeric(row.get("avg_month_end_balance", 5000.0), errors="coerce") or 5000.0),
        "overdraft_event": int(pd.to_numeric(row.get("overdraft_event", 0), errors="coerce") or 0),
    }])

# -------------------------------------
# Sidebar
# -------------------------------------
with st.sidebar:
    st.markdown("<h2>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📊 Dashboard", use_container_width=True):
        st.rerun()

    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")

    st.write("---")
    if st.button("🗑️ Delete Last Entry", use_container_width=True):
        if os.path.exists(DATA_FILE):
            try:
                df_del = pd.read_csv(DATA_FILE)
                if not df_del.empty:
                    deleted_user = df_del.iloc[-1].get("user_id", "Unknown")
                    df_del = df_del.iloc[:-1]
                    df_del.to_csv(DATA_FILE, index=False)
                    st.success(f"✅ Deleted: {deleted_user}")
                    st.rerun()
                else:
                    st.warning("No entries to delete.")
            except Exception as e:
                st.error(f"Error deleting entry: {e}")
        else:
            st.warning("No dataset found.")


# -------------------------------------
# Header
# -------------------------------------
st.markdown("<h1 style='color:#00B4D8'>📊 Credit Analytics Dashboard</h1>", unsafe_allow_html=True)
st.caption("Monitor and assess user credit health with AI-powered insights.")

# -------------------------------------
# Data Load
# -------------------------------------
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_FILE):
    st.warning("📭 Dataset file not found. Please add users first.")
    st.stop()

df_raw = pd.read_csv(DATA_FILE)
df_added_order = df_raw.copy()

if df_raw.empty:
    st.warning("📭 Dataset is empty. Please register some users first.")
    st.stop()

if "alt_credit_score" in df_raw.columns and "credit_score" not in df_raw.columns:
    df_raw = df_raw.rename(columns={"alt_credit_score": "credit_score"})

if "credit_score" not in df_raw.columns:
    st.error("❌ Column 'credit_score' not found.")
    st.stop()

df_raw["credit_score"] = pd.to_numeric(df_raw["credit_score"], errors="coerce")
df_raw = df_raw.sort_values(by="credit_score", ascending=False, na_position="last").reset_index(drop=True)
df_raw["risk_level"] = df_raw["credit_score"].apply(compute_risk_level)

# -------------------------------------
# Load ONNX models
# -------------------------------------
try:
    lr_sess, xgb_sess, rf_sess = load_models()
except Exception as e:
    st.error(f"ONNX model loading error: {e}")
    st.stop()

# -------------------------------------
# KPIs
# -------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_users = len(df_raw)
low_users = (df_raw["credit_score"] >= 70).sum()
high_users = (df_raw["credit_score"] < 40).sum()
medium_users = df_raw["credit_score"].between(40, 70).sum()

col1.metric("👥 Total Users", f"{total_users}")
col2.metric("🟢 Low Risk (≥70)", f"{low_users}")
col3.metric("🟡 Medium Risk (40–69)", f"{medium_users}")
col4.metric("🔴 High Risk (<40)", f"{high_users}")

st.write("")

# -------------------------------------
# Main Data Table
# -------------------------------------
st.markdown("### ✅ All Users Overview")

cols = ["user_id","employment_type","income_range","city_tier","monthly_income",
        "bank_account_age_months","num_bank_accounts","rent_paid_on_time",
        "utility_delay_days","upi_txn_count","avg_month_end_balance",
        "overdraft_event","credit_score","risk_level"]

display_df = df_raw.head(2000)[cols].copy()
display_df.index = range(1, len(display_df)+1)

st.dataframe(
    display_df.style.map(color_risk, subset=["risk_level"]),
    use_container_width=True, height=500
)

# -------------------------------------
# Predictions (Last 5 Added)
# -------------------------------------
st.markdown("### 🤖 AI Predictions (Last 5 Registered Users)")

df_predict = df_added_order.tail(5).copy().iloc[::-1].reset_index(drop=True)

pred_rows = []
for idx, row in df_predict.iterrows():
    try:
        input_df = build_input_df_from_row(row)
        lr_risk, _ = onnx_predict_classifier_label_and_proba(lr_sess, input_df)
        xgb_score = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
        rf_score  = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))
    except Exception:
        lr_risk, xgb_score, rf_score = "Error", "Error", "Error"

    pred_rows.append({
        "User ID": row.get("user_id", f"User_{idx+1}"),
        "Credit Score": row.get("credit_score", np.nan),
        "Predicted LR Risk": lr_risk,
        "XGB Score": xgb_score,
        "RF Score": rf_score,
        "Risk Level": compute_risk_level(row.get("credit_score", np.nan)),
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

st.dataframe(
    pred_df.style.map(color_risk, subset=["Risk Level"]),
    use_container_width=True, height=300
)
