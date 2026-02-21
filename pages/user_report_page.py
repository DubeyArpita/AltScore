import streamlit as st
import pandas as pd

st.set_page_config(page_title="AltScore | Credit Report", layout="wide", page_icon="📄")

# --------------------------------------------------
# CSS — blue theme from Register page
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:ital,wght@0,700;0,900;1,700&display=swap');

[data-testid="stSidebarNav"] { display: none !important; }

/* ══════════════════════════════════════════
   GLOBAL
══════════════════════════════════════════ */
* { font-family: 'Manrope', sans-serif !important; box-sizing: border-box; }

/* ══════════════════════════════════════════
   APP BG — blue gradient + money SVG watermark
══════════════════════════════════════════ */
.stApp {
    background-color: #e3f2fd;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' viewBox='0 0 1000 700'%3E%3Crect x='680' y='30' width='280' height='140' rx='18' fill='none' stroke='%231565c0' stroke-width='2' opacity='0.18'/%3E%3Crect x='700' y='50' width='240' height='100' rx='10' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ccircle cx='820' cy='100' r='30' fill='none' stroke='%231565c0' stroke-width='1.5' opacity='0.18'/%3E%3Ccircle cx='820' cy='100' r='20' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ctext x='808' y='107' font-family='monospace' font-size='18' fill='%231565c0' opacity='0.28' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='695' y='80' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3EALT SCORE CREDIT%3C/text%3E%3Ctext x='695' y='148' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3E100 00000 0001%3C/text%3E%3Cellipse cx='90' cy='590' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='572' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='554' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='536' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='35' y1='536' x2='35' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='145' y1='536' x2='145' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Ctext x='73' y='567' font-family='monospace' font-size='14' fill='%230d47a1' opacity='0.25' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='50' y='80' font-family='monospace' font-size='42' fill='%231565c0' opacity='0.07' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='880' y='620' font-family='monospace' font-size='56' fill='%231565c0' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='430' y='660' font-family='monospace' font-size='36' fill='%230d47a1' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='910' y='200' font-family='monospace' font-size='80' fill='%231565c0' opacity='0.05' font-weight='bold'%3E%25%3C/text%3E%3Crect x='750' y='560' width='210' height='110' rx='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.15'/%3E%3Crect x='766' y='576' width='178' height='78' rx='8' fill='none' stroke='%230d47a1' stroke-width='0.8' opacity='0.1'/%3E%3Ccircle cx='855' cy='615' r='22' fill='none' stroke='%230d47a1' stroke-width='1.2' opacity='0.15'/%3E%3Ctext x='845' y='621' font-family='monospace' font-size='13' fill='%230d47a1' opacity='0.22' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Cpolyline points='200,680 270,640 340,655 420,590 500,600 580,520 660,490 740,440 820,420' fill='none' stroke='%231565c0' stroke-width='1.8' opacity='0.1'/%3E%3Cpath d='M 30 220 A 120 120 0 0 1 230 180' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.12' stroke-dasharray='6 6'/%3E%3Cpath d='M 30 240 A 140 140 0 0 1 250 180' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.08' stroke-dasharray='3 9'/%3E%3Ctext x='30' y='270' font-family='monospace' font-size='9' fill='%230d47a1' opacity='0.2'%3ECREDIT SCORE%3C/text%3E%3Crect x='30' y='370' width='14' height='50' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='52' y='350' width='14' height='70' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='74' y='360' width='14' height='60' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='96' y='335' width='14' height='85' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='118' y='345' width='14' height='75' rx='3' fill='%231565c0' opacity='0.1'/%3E%3C/svg%3E"),
        radial-gradient(ellipse 55vw 45vh at 5%  5%,  rgba(21,101,192,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 45vw 50vh at 95% 8%,  rgba(13,71,161,0.08)  0%, transparent 60%),
        radial-gradient(ellipse 50vw 40vh at 80% 92%, rgba(21,101,192,0.09) 0%, transparent 60%),
        radial-gradient(ellipse 40vw 45vh at 10% 90%, rgba(13,71,161,0.07)  0%, transparent 55%),
        linear-gradient(150deg, #e3f2fd 0%, #eff7ff 40%, #e3f2fd 70%, #eaf4fd 100%);
    background-size: 1000px 700px, cover, cover, cover, cover, cover;
    background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat;
    color: #071a2e;
}

/* ══════════════════════════════════════════
   SIDEBAR — deep navy dark
══════════════════════════════════════════ */
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
    color: #0d2a4a;
    font-size: 0.63rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

section[data-testid="stSidebar"] .stButton > button {
    background: rgba(21,101,192,0.09) !important;
    border: 1px solid rgba(21,101,192,0.2) !important;
    color: #3a6ea8 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 10px 16px !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    transition: background 0.2s, border-color 0.2s, color 0.2s !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(21,101,192,0.2) !important;
    border-color: rgba(21,101,192,0.45) !important;
    color: #90caf9 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ══════════════════════════════════════════
   MAIN CONTENT — white frosted card
══════════════════════════════════════════ */
section.main > div.block-container {
    background: rgba(255,255,255,0.93) !important;
    border-radius: 22px !important;
    border: 1px solid rgba(21,101,192,0.13) !important;
    box-shadow:
        0 0 0 1px rgba(21,101,192,0.05),
        0 8px 48px rgba(13,71,161,0.10),
        0 2px 10px rgba(0,0,0,0.05) !important;
    margin: 20px 20px 20px 8px !important;
    padding: 44px 52px 52px 52px !important;
    backdrop-filter: blur(8px) !important;
}

/* ══════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════ */
.page-eyebrow {
    font-size: 0.9rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #1565c0;
    font-weight: 700;
    margin-bottom: 10px;
    display: block;
}

.page-title {
    font-family: 'Fraunces', serif !important;
    font-size: clamp(2.6rem, 4.5vw, 3.8rem);
    font-weight: 900;
    color: #071a2e;
    line-height: 1.08;
    margin: 0 0 16px 0;
    letter-spacing: -0.02em;
}

.page-title .grad { color: #1565c0; }

.page-desc {
    color: #0d2a4a;
    font-size: 1.15rem;
    font-weight: 500;
    max-width: 680px;
    line-height: 1.85;
}

.header-rule {
    height: 2px;
    background: linear-gradient(90deg, #1565c0, #64b5f6, transparent);
    border-radius: 2px;
    margin: 26px 0 34px 0;
    opacity: 0.45;
}

/* ══════════════════════════════════════════
   SECTION CARDS
══════════════════════════════════════════ */
.sec-card {
    background: #eff7ff;
    border: 1px solid rgba(21,101,192,0.13);
    border-radius: 16px;
    padding: 22px 26px 10px 26px;
    margin-bottom: 18px;
    box-shadow: 0 2px 12px rgba(13,71,161,0.06);
    position: relative;
    overflow: hidden;
}

.sec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0d47a1, #1565c0, #64b5f6);
    border-radius: 16px 16px 0 0;
}

.sec-title {
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #1565c0;
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(21,101,192,0.11);
}

.sec-icon {
    width: 27px; height: 27px;
    background: linear-gradient(135deg, #bbdefb, #90caf9);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    flex-shrink: 0;
}

/* ══════════════════════════════════════════
   SCORE DISPLAY
══════════════════════════════════════════ */
.score-box {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
    border-radius: 20px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(13,71,161,0.28), 0 2px 8px rgba(0,0,0,0.1);
    margin: 16px 0;
}

.score-number {
    font-family: 'Fraunces', serif !important;
    font-size: 5rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 6px;
}

.score-label {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #90caf9;
    margin-bottom: 14px;
}

.score-risk-badge {
    display: inline-block;
    padding: 7px 22px;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}

 .badge-low    { background: rgba(40,167,69,0.22);  color: #6bcf7f; border: 1.5px solid rgba(40,167,69,0.4); }
 .badge-medium { background: rgba(255,193,7,0.22);  color: #ffd93d; border: 1.5px solid rgba(255,193,7,0.4); }
 .badge-high   { background: rgba(220,53,69,0.22);  color: #ff6b6b; border: 1.5px solid rgba(220,53,69,0.4); }


/* ══════════════════════════════════════════
   RESULTS TABLE
══════════════════════════════════════════ */
# .stTable table {
#     border-radius: 12px !important;
#     overflow: hidden !important;
#     border: 1px solid rgba(21,101,192,0.15) !important;
#     width: 100% !important;
# }

# .stTable thead tr th {
#     background: #0d47a1 !important;
#     color: #e3f2fd !important;
#     font-family: 'Manrope', sans-serif !important;
#     font-weight: 800 !important;
#     font-size: 0.75rem !important;
#     letter-spacing: 0.14em !important;
#     text-transform: uppercase !important;
#     padding: 14px 16px !important;
#     border: none !important;
# }

# .stTable tbody tr:nth-child(odd) td {
#     background: #ffffff !important;
#     color: #374151 !important;
#     font-family: 'Manrope', sans-serif !important;
#     font-weight: 500 !important;
#     padding: 14px 16px !important;
#     border-color: #e3f2fd !important;
# }

# .stTable tbody tr:nth-child(even) td {
#     background: #f0f7ff !important;
#     color: #4b5563 !important;
#     font-family: 'Manrope', sans-serif !important;
#     font-weight: 500 !important;
#     padding: 14px 16px !important;
#     border-color: #e3f2fd !important;
# }

# .stTable tbody tr:last-child td {
#     background: #e3f2fd !important;
#     color: #0d47a1 !important;
#     font-weight: 800 !important;
#     font-size: 1rem !important;
# }

/* Force consistent layout */
.stTable table {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(21,101,192,0.15) !important;
    width: 100% !important;
    table-layout: fixed !important;   /* Important */
    border-collapse: collapse !important;
}

/* Make header + body identical in height */
.stTable th,
.stTable td {
    padding: 14px 16px !important;
    line-height: 1.4 !important;     /* Force uniform text height */
    height: 56px !important;         /* Fixed row height */
    vertical-align: middle !important;
}

/* Header Styling */
.stTable thead tr th {
    background: #0d47a1 !important;
    color: #e3f2fd !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;     /* Reduced slightly from 800 */
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; /* Reduced from 0.14em */
    text-transform: uppercase !important;
    border: none !important;
}

/* Odd Rows */
.stTable tbody tr:nth-child(odd) td {
    background: #ffffff !important;
    color: #374151 !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 500 !important;
    border-color: #e3f2fd !important;
}

/* Even Rows */
.stTable tbody tr:nth-child(even) td {
    background: #f0f7ff !important;
    color: #4b5563 !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 500 !important;
    border-color: #e3f2fd !important;
}

/* Final Score Row */
.stTable tbody tr:last-child td {
    background: #e3f2fd !important;
    color: #0d47a1 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* ══════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0d47a1, #1976d2, #64b5f6) !important;
    border-radius: 8px !important;
}

[data-testid="stProgress"] > div {
    background: #e3f2fd !important;
    border-radius: 8px !important;
    height: 12px !important;
}

/* ══════════════════════════════════════════
   STATUS / ALERTS
══════════════════════════════════════════ */
[data-testid="stInfo"] {
    background: #e3f2fd !important;
    border: 1px solid #90caf9 !important;
    border-radius: 10px !important;
    color: #0d47a1 !important;
    font-weight: 600 !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(21,101,192,0.12) !important;
    margin: 14px 0 !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(21,101,192,0.22); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(21,101,192,0.4); }
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
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>AI Credit Intelligence</div>", unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠  Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📊  Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")

    if st.button("➕  New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")

    st.write("---")
    st.markdown("<p style='text-align:center;color:#0d2a4a;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --------------------------------------------------
# Prepare values
# --------------------------------------------------
final_score = int(data["final"])
risk_level  = data["risk_level"]

if risk_level == "Low Risk":
    risk_class   = "badge-low"
    risk_emoji   = "🟢"
elif risk_level == "Medium Risk":
    risk_class   = "badge-medium"
    risk_emoji   = "🟡"
else:
    risk_class   = "badge-high"
    risk_emoji   = "🔴"

# --------------------------------------------------
# Page Header
# --------------------------------------------------
st.markdown(f"""
    <div style='margin-bottom: 0;'>
        <span class='page-eyebrow'>✦ Analysis Complete</span>
        <h1 class='page-title'>Credit <span class='grad'>Intelligence Report</span></h1>
        <p class='page-desc'>AI-powered credit assessment for <strong>User ID: {data['user_id']}</strong> — generated instantly from your financial footprint.</p>
    </div>
    <div class='header-rule'></div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Score + Table layout
# --------------------------------------------------
col_score, col_table = st.columns([1, 2], gap="large")

with col_score:
    st.markdown(f"""
        <div class='score-box'>
            <div class='score-label'>Overall Credit Score</div>
            <div class='score-number'>{final_score}</div>
            <div style='color:#bbdefb;font-size:0.9rem;margin-bottom:14px;font-weight:600;'>out of 100</div>
            <span class='score-risk-badge {risk_class}'>{risk_emoji} {risk_level}</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.progress(final_score / 100)

    if final_score >= 70:
        st.success("🎉 **Excellent profile!** Eligible for credit facilities.")
    elif final_score >= 40:
        st.warning("⚠️ **Moderate risk.** Conditional approval recommended.")
    else:
        st.error("❌ **High risk profile.** Credit extension not recommended.")

with col_table:
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📊</div>Model-by-Model Breakdown</div>
    </div>""", unsafe_allow_html=True)

    report_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "FINAL SCORE"
        ],
        "Result": [
            data["lr_risk"],
            f"{data['rf']}/100",
            f"{data['xgb']}/100",
            f"{final_score}/100"
        ],
        "Remarks": [
            "Probability-based risk classification",
            "Ensemble regression estimate",
            "Gradient boosting estimate",
            data["eligibility"]
        ]
    })

    st.table(report_df)

# --------------------------------------------------
# Risk Probability Breakdown (optional)
# --------------------------------------------------
if data.get("lr_probs"):
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("""<div class='sec-card'>
        <div class='sec-title'><div class='sec-icon'>📈</div>Risk Probability Breakdown</div>
    </div>""", unsafe_allow_html=True)

    probs = data["lr_probs"]
    cols = st.columns(len(probs))
    for i, (label, prob) in enumerate(probs.items()):
        with cols[i]:
            pct = round(float(prob) * 100, 1)
            st.metric(label=label, value=f"{pct}%")
