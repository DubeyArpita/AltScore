import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AltScore Dashboard",
    layout="wide"
)

# -------------------------
# CUSTOM STYLING
# -------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #111827;
    color: #F9FAFB;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 1rem;
    color: #F9FAFB;
}

.metric-card {
    background-color: #1F2937;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.metric-title {
    font-size: 14px;
    color: #9CA3AF;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    margin-top: 8px;
}

.low-risk {
    background-color: rgba(34,197,94,0.15);
    color: #22C55E;
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: 600;
}

.medium-risk {
    background-color: rgba(245,158,11,0.15);
    color: #F59E0B;
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: 600;
}

.high-risk {
    background-color: rgba(239,68,68,0.15);
    color: #EF4444;
    padding: 6px 12px;
    border-radius: 12px;
    font-weight: 600;
}

.dataframe {
    background-color: #1F2937 !important;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.markdown("## 📊 Alternative Credit Risk Dashboard")
st.markdown("---")

# -------------------------
# SAMPLE DATA
# -------------------------
df = pd.DataFrame({
    "Name": ["Rahul", "Anita", "John", "Meena", "David"],
    "Score": [750, 620, 580, 710, 500],
    "Risk Level": ["Low", "Medium", "High", "Low", "High"]
})

# -------------------------
# METRICS
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Total Applicants</div>
        <div class="metric-value">5</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Average Score</div>
        <div class="metric-value">632</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">High Risk Cases</div>
        <div class="metric-value" style="color:#EF4444;">2</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# CHART
# -------------------------
st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)

fig = px.pie(
    df,
    names="Risk Level",
    hole=0.6,
    color="Risk Level",
    color_discrete_map={
        "Low": "#22C55E",
        "Medium": "#F59E0B",
        "High": "#EF4444"
    }
)

fig.update_layout(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font_color="white",
    margin=dict(t=10, b=10, l=10, r=10)
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# TABLE WITH RISK HIGHLIGHT
# -------------------------
st.markdown('<div class="section-header">Applicant Details</div>', unsafe_allow_html=True)

def highlight_risk(val):
    if val == "Low":
        return 'background-color: rgba(34,197,94,0.2); color: #22C55E; font-weight: bold;'
    elif val == "Medium":
        return 'background-color: rgba(245,158,11,0.2); color: #F59E0B; font-weight: bold;'
    elif val == "High":
        return 'background-color: rgba(239,68,68,0.2); color: #EF4444; font-weight: bold;'
    return ''

styled_df = df.style.applymap(highlight_risk, subset=["Risk Level"])

st.dataframe(styled_df, use_container_width=True)
