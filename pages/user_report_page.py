import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Analysis Report", layout="centered")

# --------------------------------------------------
# CSS (keep / adjust if you want)
# --------------------------------------------------
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }

.stApp {
    background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)),
                url("https://images.unsplash.com/photo-1451187580459-43490279c0fa");
    background-size: cover;
}


.good { color: #28a745; font-weight: bold; }
.medium { color: #ffc107; font-weight: bold; }
.bad { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Validate session state
# --------------------------------------------------
if "report_data" not in st.session_state:
    st.error("❌ No report data found. Please register a user first.")
    st.stop()

data = st.session_state["report_data"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#00D1FF;'>📄 Personalized Credit Report</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<h3 style='text-align:center;color:white;'>User ID: {data['user_id']}</h3>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Prepare display values
# --------------------------------------------------
final_score = int(data["final"])
risk_level = data["risk_level"]

if risk_level == "Low Risk":
    risk_class = "good"
elif risk_level == "Medium Risk":
    risk_class = "medium"
else:
    risk_class = "bad"

# --------------------------------------------------
# Results table
# --------------------------------------------------
report_df = pd.DataFrame({
    "Model": [
        "Logistic Regression (Risk Classifier)",
        "Random Forest (Score)",
        "XGBoost (Score)",
        "FINAL CREDIT SCORE"
    ],
    "Result": [
        data["lr_risk"],
        f"{data['rf']}/100",
        f"{data['xgb']}/100",
        f"{final_score}/100"
    ],
    "Remarks": [
        "Probability-based classification",
        "Regression estimate",
        "Regression estimate",
        data["eligibility"]
    ]
})

st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.table(report_df)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Score Progress
# --------------------------------------------------
st.markdown("<h3 style='color:white;text-align:center;'>Overall Credit Health</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.progress(final_score / 100)
    st.markdown(
        f"<p class='{risk_class}' style='text-align:center;font-size:24px;'>"
        f"{final_score}/100 — {risk_level}</p>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Final Verdict Message
# --------------------------------------------------
if final_score >= 70:
    st.success("🎉 **Excellent profile!** This user is eligible for credit facilities.")
elif final_score >= 40:
    st.warning("⚠️ **Moderate risk.** Conditional approval is recommended.")
else:
    st.error("❌ **High risk profile.** Credit extension is not recommended.")

# --------------------------------------------------
# Optional: Probability breakdown (Logistic Regression)
# --------------------------------------------------
if data.get("lr_probs"):
    st.markdown("<h3 style='color:white;'>📊 Risk Probability Breakdown</h3>", unsafe_allow_html=True)
    st.json(data["lr_probs"])

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")

    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
