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
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Manrope:wght@300;400;500;600;700;800&display=swap');

    /* ══════════════════════════════════════════
       ROOT CANVAS — dark bg with color orbs + finance SVG pattern
    ══════════════════════════════════════════ */
    .stApp {
        font-family: 'Manrope', sans-serif;
        color: #1a1030;
        overflow-x: hidden;
        /* Finance-themed SVG pattern + gradient */
        background-color: #07050f;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='600' viewBox='0 0 900 600'%3E%3Cdefs%3E%3Cstyle%3E.a%7Bfill:none;stroke-width:1.2;stroke-linecap:round;stroke-linejoin:round;opacity:0.55%7D.b%7Bfill:none;stroke-width:0.8;opacity:0.3%7D.c%7Bfont-family:monospace;font-size:10px;opacity:0.25%7D%3C/style%3E%3C/defs%3E%3C!-- Graph trend lines --%3E%3Cpolyline class='a' stroke='%23f472b6' points='40,480 90,440 140,460 200,380 260,400 320,320 380,290 440,250 500,230 560,180 620,200 680,150 740,120 800,90 860,70'/%3E%3Cpolyline class='a' stroke='%23818cf8' points='40,520 100,500 160,510 220,470 280,480 340,430 400,410 460,370 520,350 580,300 640,320 700,270 760,240 820,210 870,190'/%3E%3Cpolyline class='a' stroke='%2334d399' points='40,560 110,540 180,545 250,510 310,520 370,480 430,460 490,420 550,400 610,360 670,375 730,340 790,310 850,290'/%3E%3C!-- Candlestick bars --%3E%3Cg stroke='%23f59e0b' stroke-width='1' opacity='0.35'%3E%3Crect x='55' y='200' width='8' height='40' fill='%23f59e0b' rx='1'/%3E%3Cline x1='59' y1='195' x2='59' y2='245'/%3E%3Crect x='80' y='180' width='8' height='50' fill='%23f59e0b' rx='1'/%3E%3Cline x1='84' y1='175' x2='84' y2='235'/%3E%3Crect x='105' y='210' width='8' height='30' fill='%23ef4444' rx='1' stroke='%23ef4444'/%3E%3Cline x1='109' y1='205' x2='109' y2='245'/%3E%3Crect x='130' y='190' width='8' height='45' fill='%23f59e0b' rx='1'/%3E%3Cline x1='134' y1='185' x2='134' y2='240'/%3E%3Crect x='155' y='170' width='8' height='55' fill='%23f59e0b' rx='1'/%3E%3Cline x1='159' y1='165' x2='159' y2='230'/%3E%3Crect x='780' y='80' width='8' height='35' fill='%23f59e0b' rx='1'/%3E%3Cline x1='784' y1='75' x2='784' y2='120'/%3E%3Crect x='805' y='60' width='8' height='45' fill='%2334d399' rx='1' stroke='%2334d399'/%3E%3Cline x1='809' y1='55' x2='809' y2='110'/%3E%3Crect x='830' y='50' width='8' height='40' fill='%23f59e0b' rx='1'/%3E%3Cline x1='834' y1='45' x2='834' y2='95'/%3E%3C/g%3E%3C!-- Coin circles --%3E%3Ccircle cx='760' cy='480' r='32' fill='none' stroke='%23f59e0b' stroke-width='1.5' opacity='0.2'/%3E%3Ccircle cx='760' cy='480' r='24' fill='none' stroke='%23f59e0b' stroke-width='0.8' opacity='0.15'/%3E%3Ctext x='754' y='484' class='c' fill='%23f59e0b' font-size='14px' opacity='0.3' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ccircle cx='120' cy='100' r='28' fill='none' stroke='%23818cf8' stroke-width='1.5' opacity='0.2'/%3E%3Ccircle cx='120' cy='100' r='20' fill='none' stroke='%23818cf8' stroke-width='0.8' opacity='0.15'/%3E%3Ctext x='114' y='104' class='c' fill='%23818cf8' font-size='14px' opacity='0.3' font-weight='bold'%3E%24%3C/text%3E%3C!-- Grid lines --%3E%3Cg class='b' stroke='%23ffffff'%3E%3Cline x1='0' y1='150' x2='900' y2='150'/%3E%3Cline x1='0' y1='300' x2='900' y2='300'/%3E%3Cline x1='0' y1='450' x2='900' y2='450'/%3E%3Cline x1='150' y1='0' x2='150' y2='600'/%3E%3Cline x1='300' y1='0' x2='300' y2='600'/%3E%3Cline x1='450' y1='0' x2='450' y2='600'/%3E%3Cline x1='600' y1='0' x2='600' y2='600'/%3E%3Cline x1='750' y1='0' x2='750' y2='600'/%3E%3C/g%3E%3C!-- Credit score arc --%3E%3Cpath d='M 820 530 A 60 60 0 0 1 880 480' fill='none' stroke='%23a855f7' stroke-width='2' opacity='0.25' stroke-dasharray='4 4'/%3E%3Cpath d='M 820 530 A 80 80 0 0 1 900 480' fill='none' stroke='%23f472b6' stroke-width='1' opacity='0.18' stroke-dasharray='2 6'/%3E%3C!-- Small data labels --%3E%3Ctext x='500' y='220' class='c' fill='%2322d3ee'%3ESCORE: 847%3C/text%3E%3Ctext x='300' y='310' class='c' fill='%23818cf8'%3ERISK: LOW%3C/text%3E%3Ctext x='650' y='420' class='c' fill='%23f472b6'%3EELIGIBLE%3C/text%3E%3C/svg%3E"),
            radial-gradient(ellipse 55vw 50vh at 8%  12%,  rgba(239,68,68,0.14)  0%, transparent 60%),
            radial-gradient(ellipse 45vw 45vh at 90% 15%,  rgba(168,85,247,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 50vw 40vh at 70% 82%,  rgba(59,130,246,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 38vw 38vh at 18% 80%,  rgba(234,179,8,0.09)  0%, transparent 55%),
            radial-gradient(ellipse 30vw 50vh at 50% 48%,  rgba(236,72,153,0.08) 0%, transparent 55%);
        background-size: 900px 600px, cover, cover, cover, cover, cover;
        background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat;
        background-attachment: fixed;
        animation: bgPan 40s linear infinite;
    }

    @keyframes bgPan {
        0%   { background-position: 0px 0px,    center, center, center, center, center; }
        100% { background-position: 900px 600px, center, center, center, center, center; }
    }

    /* ══════════════════════════════════════════
       SIDEBAR — SOLID DEEP DARK
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #060409 !important;
        border-right: 1px solid rgba(168,85,247,0.12) !important;
        box-shadow: 4px 0 32px rgba(0,0,0,0.7) !important;
    }

    .sidebar-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f472b6 0%, #a855f7 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: 0.06em;
    }

    .sidebar-sub {
        text-align: center;
        color: #2a1f3d;
        font-size: 0.63rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        color: #42325a !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.9rem !important;
        font-family: 'Manrope', sans-serif !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        transition: all 0.25s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(168,85,247,0.09) !important;
        border-color: rgba(168,85,247,0.28) !important;
        color: #c084fc !important;
        box-shadow: 0 0 14px rgba(168,85,247,0.1) !important;
        transform: none !important;
    }

    /* ══════════════════════════════════════════
       MAIN CONTENT AREA — light frosted glass panel
    ══════════════════════════════════════════ */
    /* Wrap the entire main content in a light glass panel */
    section.main > div.block-container {
        background: rgba(255, 255, 255, 0.82) !important;
        backdrop-filter: blur(32px) !important;
        -webkit-backdrop-filter: blur(32px) !important;
        border-radius: 28px !important;
        border: 1px solid rgba(255,255,255,0.9) !important;
        box-shadow:
            0 0 0 1px rgba(168,85,247,0.06),
            0 20px 80px rgba(0,0,0,0.25),
            inset 0 1px 0 rgba(255,255,255,1) !important;
        margin: 24px 24px 24px 8px !important;
        padding: 40px 48px 48px 48px !important;
        animation: panelRise 0.6s cubic-bezier(0.16,1,0.3,1) both;
    }

    @keyframes panelRise {
        from { opacity: 0; transform: translateY(24px) scale(0.99); }
        to   { opacity: 1; transform: translateY(0)   scale(1); }
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes cardRise {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .page-eyebrow {
        font-family: 'Manrope', sans-serif;
        font-size: 1.05rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #e11d48, #9333ea, #2563eb, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 10px;
        display: block;
        animation: fadeUp 0.5s ease both;
    }

    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: clamp(2.5rem, 4vw, 3.6rem);
        font-weight: 400;
        color: #120a2e;
        line-height: 1.1;
        margin: 0 0 16px 0;
        animation: fadeUp 0.55s ease 0.05s both;
    }

    .page-title .grad {
        background: linear-gradient(100deg, #e11d48 0%, #9333ea 45%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-style: italic;
    }

    .page-desc {
        color: #64748b;
        font-size: 1.15rem;
        font-weight: 400;
        max-width: 580px;
        line-height: 1.9;
        letter-spacing: 0.01em;
        animation: fadeUp 0.55s ease 0.1s both;
        margin-bottom: 8px;
    }

    /* Decorative divider under header */
    .header-rule {
        height: 2px;
        background: linear-gradient(90deg, #e11d48, #9333ea, #2563eb, #ec4899, transparent);
        border-radius: 2px;
        margin: 24px 0 32px 0;
        opacity: 0.35;
        animation: fadeUp 0.6s ease 0.15s both;
    }

    /* ══════════════════════════════════════════
       SECTION CARDS — light glass on light bg
    ══════════════════════════════════════════ */
    .sec-card {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 18px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
        box-shadow:
            0 2px 16px rgba(0,0,0,0.06),
            0 1px 4px rgba(0,0,0,0.04),
            inset 0 1px 0 rgba(255,255,255,0.9);
        animation: cardRise 0.5s ease both;
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 16px; right: 16px;
        height: 2px;
        background: linear-gradient(90deg,
            transparent,
            rgba(225,29,72,0.4),
            rgba(147,51,234,0.5),
            rgba(37,99,235,0.4),
            rgba(236,72,153,0.4),
            transparent
        );
        border-radius: 2px;
    }

    .sec-card:hover {
        border-color: rgba(147,51,234,0.18);
        box-shadow:
            0 6px 32px rgba(147,51,234,0.08),
            0 2px 8px rgba(0,0,0,0.06),
            inset 0 1px 0 rgba(255,255,255,0.95);
    }

    .sec-card:nth-child(2) { animation-delay: 0.08s; }
    .sec-card:nth-child(3) { animation-delay: 0.16s; }

    .sec-title {
        font-family: 'Manrope', sans-serif;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #7c3aed;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }

    .sec-icon {
        width: 26px; height: 26px;
        background: linear-gradient(135deg, rgba(225,29,72,0.12), rgba(147,51,234,0.14), rgba(37,99,235,0.1));
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════════
       INPUTS — crisp white on light panel
    ══════════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: #ffffff !important;
        color: #0f0a20 !important;
        border: 1.5px solid #e2d9f3 !important;
        border-radius: 10px !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #9333ea !important;
        box-shadow: 0 0 0 3px rgba(147,51,234,0.1), 0 2px 8px rgba(147,51,234,0.08) !important;
        background: #fdfbff !important;
    }

    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #ffffff !important;
        border: 1.5px solid #e2d9f3 !important;
        border-radius: 10px !important;
        color: #0f0a20 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.96rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    /* All widget labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: 0.025em !important;
    }

    /* Radio text */
    .stRadio [role="radiogroup"] label div p {
        color: #475569 !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }

    /* Slider */
    [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #e11d48, #9333ea) !important;
        border: 2px solid #fff !important;
        box-shadow: 0 0 10px rgba(147,51,234,0.35) !important;
    }

    /* Info */
    [data-testid="stInfo"] {
        background: rgba(147,51,234,0.05) !important;
        border: 1px solid rgba(147,51,234,0.2) !important;
        border-radius: 10px !important;
        color: #6d28d9 !important;
    }

    /* ══════════════════════════════════════════
       CENTERED PILL BUTTON
    ══════════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 300px !important;
        padding: 16px 56px !important;
        border-radius: 50px !important;
        border: none !important;
        background: linear-gradient(135deg,
            #e11d48 0%,
            #9333ea 35%,
            #2563eb 70%,
            #ec4899 100%
        ) !important;
        background-size: 200% 200% !important;
        animation: btnGlow 4s ease infinite !important;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow:
            0 0 24px rgba(147,51,234,0.28),
            0 4px 20px rgba(0,0,0,0.2) !important;
    }

    @keyframes btnGlow {
        0%   { background-position: 0%   50%; box-shadow: 0 0 24px rgba(225,29,72,0.28),  0 4px 20px rgba(0,0,0,0.2); }
        33%  { background-position: 60%  50%; box-shadow: 0 0 28px rgba(147,51,234,0.35), 0 4px 20px rgba(0,0,0,0.2); }
        66%  { background-position: 100% 50%; box-shadow: 0 0 24px rgba(37,99,235,0.28),  0 4px 20px rgba(0,0,0,0.2); }
        100% { background-position: 0%   50%; box-shadow: 0 0 24px rgba(225,29,72,0.28),  0 4px 20px rgba(0,0,0,0.2); }
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow:
            0 0 40px rgba(147,51,234,0.5),
            0 8px 32px rgba(0,0,0,0.25) !important;
    }

    /* ══════════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(147,51,234,0.2) !important;
        border-radius: 12px !important;
        color: #6d28d9 !important;
        font-family: 'Manrope', sans-serif !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(147,51,234,0.18); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(147,51,234,0.32); }
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
    st.markdown("<p style='text-align:center;color:#1a1030;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# --- Page Header ---
st.markdown("""
    <div style='margin-bottom: 0; animation: fadeUp 0.5s ease both;'>
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
