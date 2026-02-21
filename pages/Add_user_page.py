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
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' viewBox='0 0 1000 700'%3E%3C!-- Large banknote top-right --%3E%3Crect x='680' y='30' width='280' height='140' rx='18' fill='none' stroke='%231565c0' stroke-width='2' opacity='0.18'/%3E%3Crect x='700' y='50' width='240' height='100' rx='10' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ccircle cx='820' cy='100' r='30' fill='none' stroke='%231565c0' stroke-width='1.5' opacity='0.18'/%3E%3Ccircle cx='820' cy='100' r='20' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ctext x='808' y='107' font-family='monospace' font-size='18' fill='%231565c0' opacity='0.28' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='695' y='80' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3EALT SCORE CREDIT%3C/text%3E%3Ctext x='695' y='148' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3E100 00000 0001%3C/text%3E%3C!-- Coin stack bottom-left --%3E%3Cellipse cx='90' cy='590' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='572' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='554' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='536' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='35' y1='536' x2='35' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='145' y1='536' x2='145' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Ctext x='73' y='567' font-family='monospace' font-size='14' fill='%230d47a1' opacity='0.25' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Large rupee symbols --%3E%3Ctext x='50' y='80' font-family='monospace' font-size='42' fill='%231565c0' opacity='0.07' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='880' y='620' font-family='monospace' font-size='56' fill='%231565c0' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='430' y='660' font-family='monospace' font-size='36' fill='%230d47a1' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Percent symbol --%3E%3Ctext x='910' y='200' font-family='monospace' font-size='80' fill='%231565c0' opacity='0.05' font-weight='bold'%3E%25%3C/text%3E%3C!-- Mini banknote bottom-right --%3E%3Crect x='750' y='560' width='210' height='110' rx='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.15'/%3E%3Crect x='766' y='576' width='178' height='78' rx='8' fill='none' stroke='%230d47a1' stroke-width='0.8' opacity='0.1'/%3E%3Ccircle cx='855' cy='615' r='22' fill='none' stroke='%230d47a1' stroke-width='1.2' opacity='0.15'/%3E%3Ctext x='845' y='621' font-family='monospace' font-size='13' fill='%230d47a1' opacity='0.22' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Stock trend line --%3E%3Cpolyline points='200,680 270,640 340,655 420,590 500,600 580,520 660,490 740,440 820,420' fill='none' stroke='%231565c0' stroke-width='1.8' opacity='0.1'/%3E%3C!-- Credit score arc --%3E%3Cpath d='M 30 220 A 120 120 0 0 1 230 180' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.12' stroke-dasharray='6 6'/%3E%3Cpath d='M 30 240 A 140 140 0 0 1 250 180' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.08' stroke-dasharray='3 9'/%3E%3Ctext x='30' y='270' font-family='monospace' font-size='9' fill='%230d47a1' opacity='0.2'%3ECREDIT SCORE%3C/text%3E%3C!-- Bar chart --%3E%3Crect x='30' y='370' width='14' height='50' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='52' y='350' width='14' height='70' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='74' y='360' width='14' height='60' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='96' y='335' width='14' height='85' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='118' y='345' width='14' height='75' rx='3' fill='%231565c0' opacity='0.1'/%3E%3C/svg%3E"),
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
        background: rgba(255, 255, 255, 0.93) !important;
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
        max-width: 600px;
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
       INPUTS
    ══════════════════════════════════════════ */
    .stNumberInput input,
    .stTextInput input {
        background: #ffffff !important;
        color: #071a2e !important;
        border: 1.5px solid #90caf9 !important;
        border-radius: 10px !important;
        font-size: 0.97rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(13,71,161,0.07) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #1565c0 !important;
        box-shadow: 0 0 0 3px rgba(21,101,192,0.13) !important;
    }

    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: #ffffff !important;
        border: 1.5px solid #90caf9 !important;
        border-radius: 10px !important;
        color: #071a2e !important;
        font-weight: 600 !important;
        font-size: 0.97rem !important;
        box-shadow: 0 1px 3px rgba(13,71,161,0.07) !important;
    }

    /* Dropdown */
    [data-baseweb="popover"] [role="listbox"] {
        background: #fff !important;
        border: 1px solid #90caf9 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(13,71,161,0.12) !important;
    }
    [data-baseweb="popover"] [role="option"] { color: #071a2e !important; }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [aria-selected="true"] {
        background: #e3f2fd !important;
        color: #1565c0 !important;
    }

    /* All labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label,
    .stRadio > label,
    .stToggle label,
    div[data-testid="stWidgetLabel"] p {
        color: #0d2a4a !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
    }

    .stRadio [role="radiogroup"] label div p {
        color: #071a2e !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
    }

    /* ══════════════════════════════════════════
       RADIO — purple
    ══════════════════════════════════════════ */
    .stRadio [role="radiogroup"] label span:first-child {
        border-color: #7c3aed !important;
    }
    .stRadio [role="radiogroup"] [aria-checked="true"] span:first-child {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
    }
    .stRadio [role="radiogroup"] [aria-checked="true"] span:first-child::after {
        background: #ffffff !important;
    }
    input[type="radio"] { accent-color: #7c3aed !important; }

    /* ══════════════════════════════════════════
       SLIDER — purple
    ══════════════════════════════════════════ */
    [data-baseweb="slider"] [role="progressbar"],
    [data-baseweb="slider"] div[style*="background"] {
        background: #7c3aed !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        background: #7c3aed !important;
        border: 2.5px solid #ffffff !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.22), 0 2px 8px rgba(124,58,237,0.3) !important;
    }
    input[type="range"] { accent-color: #7c3aed !important; }

    /* ══════════════════════════════════════════
       TOGGLE — purple
    ══════════════════════════════════════════ */
    div[data-testid="stToggle"] > label > div:first-child {
        background-color: #e9d5ff !important;
        border-color: #c4b5fd !important;
    }
    div[data-testid="stToggle"] > label > div[data-checked="true"],
    div[data-testid="stToggle"] input:checked + div {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
    }
    div[data-testid="stToggle"] > label > div > div {
        background-color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(124,58,237,0.25) !important;
    }
    input[type="checkbox"] { accent-color: #7c3aed !important; }

    div[data-testid="stToggle"] p {
        color: #0d2a4a !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }

    /* Info box */
    [data-testid="stInfo"] {
        background: #e3f2fd !important;
        border: 1px solid #90caf9 !important;
        border-radius: 10px !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
    }

    /* Number input step buttons */
    .stNumberInput [data-testid="stNumberInputContainer"] button {
        background: #e3f2fd !important;
        border-color: #90caf9 !important;
        color: #1565c0 !important;
    }

    /* ══════════════════════════════════════════
       CENTERED SUBMIT BUTTON — deep blue gradient
    ══════════════════════════════════════════ */
    .stForm [data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 320px !important;
        padding: 17px 62px !important;
        border-radius: 50px !important;
        border: none !important;
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 45%, #1976d2 100%) !important;
        color: #ffffff !important;
        font-size: 1.02rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        box-shadow: 0 5px 24px rgba(21,101,192,0.38), 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }

    .stForm [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 36px rgba(21,101,192,0.52), 0 4px 12px rgba(0,0,0,0.12) !important;
    }

    /* ══════════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════════ */
    [data-testid="stStatus"] {
        background: #eff7ff !important;
        border: 1px solid #90caf9 !important;
        border-radius: 12px !important;
        color: #0d47a1 !important;
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
        color: #00000 !important;
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
