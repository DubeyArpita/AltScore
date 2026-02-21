import os
import numpy as np
import pandas as pd
import streamlit as st
from onnx_utils import load_onnx_sessions, onnx_predict_regressor, onnx_predict_classifier_label_and_proba

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(page_title="AltScore | Dashboard", layout="wide", page_icon="📊")

DATA_FILE = "data/dataset.csv"

# -------------------------------------
# Custom CSS — matches Register page theme
# -------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,700;0,900;1,700&family=Manrope:wght@400;500;600;700;800&display=swap');

    /* ══════════════════════════════════════════
       ROOT — purple/white theme
    ══════════════════════════════════════════ */
    .stApp {
        background: #f3f0ff;
        font-family: 'Manrope', sans-serif;
        color: #1e0a3c;
    }

    /* Subtle static purple radial tint in corners */
    .stApp > div:first-child::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 60vw 50vh at 0%   0%,   rgba(139,92,246,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 50vw 45vh at 100% 100%, rgba(109,40,217,0.10) 0%, transparent 65%),
            radial-gradient(ellipse 40vw 40vh at 100% 0%,   rgba(196,181,253,0.15) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* Finance SVG watermark — static, very subtle */
    .stApp > div:first-child::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='500' viewBox='0 0 800 500'%3E%3Cg opacity='0.045' stroke='%236d28d9' fill='none'%3E%3Cpolyline stroke-width='2' points='0,400 80,340 160,360 240,280 320,300 400,200 480,170 560,120 640,140 720,80 800,50'/%3E%3Cpolyline stroke-width='1.5' stroke='%238b5cf6' points='0,450 100,420 200,430 300,390 400,360 500,310 600,280 700,240 800,200'/%3E%3Cline x1='0' y1='166' x2='800' y2='166' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='0' y1='333' x2='800' y2='333' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='200' y1='0' x2='200' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='400' y1='0' x2='400' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Cline x1='600' y1='0' x2='600' y2='500' stroke-width='0.8' stroke-dasharray='4 8'/%3E%3Ccircle cx='680' cy='400' r='48' stroke-width='1.5'/%3E%3Ccircle cx='680' cy='400' r='34' stroke-width='1'/%3E%3Ctext x='666' y='406' font-family='monospace' font-size='16' fill='%236d28d9' opacity='1'%3E%E2%82%B9%3C/text%3E%3Crect x='40' y='60' width='12' height='60' fill='%236d28d9'/%3E%3Crect x='65' y='40' width='12' height='80' fill='%238b5cf6'/%3E%3Crect x='90' y='70' width='12' height='50' fill='%236d28d9'/%3E%3Crect x='115' y='30' width='12' height='90' fill='%238b5cf6'/%3E%3Crect x='140' y='55' width='12' height='65' fill='%236d28d9'/%3E%3C/g%3E%3C/svg%3E");
        background-size: 800px 500px;
        background-repeat: repeat;
        background-position: center;
        pointer-events: none;
        z-index: 0;
    }

    /* ══════════════════════════════════════════
       SIDEBAR — deep dark purple
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: #12002e !important;
        border-right: 1px solid rgba(139,92,246,0.2) !important;
        box-shadow: 4px 0 28px rgba(0,0,0,0.5) !important;
    }

    .sidebar-logo {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        color: #c4b5fd;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        text-align: center;
        color: #7c3aed;
        font-size: 0.63rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(124,58,237,0.08) !important;
        border: 1px solid rgba(124,58,237,0.2) !important;
        color: #c4b5fd !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        padding: 10px 16px !important;
        font-size: 0.92rem !important;
        font-family: 'Manrope', sans-serif !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        transition: background 0.2s, border-color 0.2s, color 0.2s !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(124,58,237,0.22) !important;
        border-color: #7c3aed !important;
        color: #ffffff !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ══════════════════════════════════════════
       MAIN CONTENT — white card
    ══════════════════════════════════════════ */
    section.main > div.block-container {
        background: #ffffff !important;
        border-radius: 24px !important;
        border: 1px solid #ede9fe !important;
        box-shadow:
            0 0 0 1px rgba(139,92,246,0.06),
            0 8px 48px rgba(109,40,217,0.09),
            0 2px 8px rgba(0,0,0,0.04) !important;
        margin: 20px 20px 20px 8px !important;
        padding: 44px 52px 52px 52px !important;
    }

    /* ══════════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════════ */
    .page-eyebrow {
        font-family: 'Manrope', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #7c3aed;
        font-weight: 700;
        margin-bottom: 10px;
        display: block;
    }

    .page-title {
        font-family: 'Fraunces', serif;
        font-size: clamp(2.8rem, 4.5vw, 4rem);
        font-weight: 900;
        color: #1e0a3c;
        line-height: 1.08;
        margin: 0 0 18px 0;
        letter-spacing: -0.02em;
    }

    .page-title .grad {
        color: #7c3aed;
    }

    .page-desc {
        color: #3b1f5e;
        font-size: 1.18rem;
        font-weight: 500;
        max-width: 600px;
        line-height: 1.85;
    }

    .header-rule {
        height: 2px;
        background: linear-gradient(90deg, #7c3aed, #c4b5fd, transparent);
        border-radius: 2px;
        margin: 28px 0 36px 0;
        opacity: 0.5;
    }

    /* ══════════════════════════════════════════
       SECTION CARDS
    ══════════════════════════════════════════ */
    .sec-card {
        background: #faf8ff;
        border: 1px solid #ede9fe;
        border-radius: 16px;
        padding: 22px 26px 10px 26px;
        margin-bottom: 18px;
        box-shadow: 0 2px 12px rgba(109,40,217,0.05);
        position: relative;
        overflow: hidden;
    }

    .sec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #7c3aed, #a78bfa, #c4b5fd);
        border-radius: 16px 16px 0 0;
    }

    .sec-title {
        font-family: 'Manrope', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #6d28d9;
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid #ede9fe;
    }

    .sec-icon {
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    /* ══════════════════════════════════════════
       KPI METRIC CARDS — per-card colour theming
    ══════════════════════════════════════════ */
    div.stMetric {
        border-radius: 16px !important;
        padding: 22px 20px !important;
        position: relative !important;
        overflow: hidden !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif !important;
        font-size: 2.2rem !important;
        font-weight: 900 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }

    /* Card 1 — Total Users: lighter brand purple */
    div.stMetric:nth-of-type(1) {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%) !important;
        border: 1px solid rgba(196,181,253,0.4) !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
    }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(1) div[data-testid="stMetricLabel"] { color: #ede9fe !important; }

    /* Card 2 — Low Risk: lighter purple-green */
    div.stMetric:nth-of-type(2) {
        background: linear-gradient(135deg, #a78bfa 0%, #34d399 100%) !important;
        border: 1px solid rgba(110,231,183,0.35) !important;
        box-shadow: 0 4px 20px rgba(167,139,250,0.2) !important;
    }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(2) div[data-testid="stMetricLabel"] { color: #d1fae5 !important; }

    /* Card 3 — Medium Risk: lighter purple-amber */
    div.stMetric:nth-of-type(3) {
        background: linear-gradient(135deg, #a78bfa 0%, #fbbf24 100%) !important;
        border: 1px solid rgba(253,230,138,0.35) !important;
        box-shadow: 0 4px 20px rgba(167,139,250,0.2) !important;
    }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(3) div[data-testid="stMetricLabel"] { color: #fef3c7 !important; }

    /* Card 4 — High Risk: lighter purple-red */
    div.stMetric:nth-of-type(4) {
        background: linear-gradient(135deg, #a78bfa 0%, #f87171 100%) !important;
        border: 1px solid rgba(252,165,165,0.35) !important;
        box-shadow: 0 4px 20px rgba(167,139,250,0.2) !important;
    }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stMetric:nth-of-type(4) div[data-testid="stMetricLabel"] { color: #fee2e2 !important; }

    /* ══════════════════════════════════════════
       DATAFRAMES — styled to match theme
    ══════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid #ddd6fe !important;
        overflow: hidden !important;
        box-shadow: 0 2px 16px rgba(109,40,217,0.08) !important;
    }

    /* Header row */
    [data-testid="stDataFrame"] thead tr th {
        background: #1e0a3c !important;
        color: #a78bfa !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid #3b1a7a !important;
    }

    /* Alternating rows */
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) td {
        background: #f8f5ff !important;
        color: #2d0f6b !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background: #ede9fe !important;
        color: #1e0a3c !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: #ddd6fe !important;
        color: #1e0a3c !important;
    }

    /* Cell text */
    [data-testid="stDataFrame"] td {
        font-family: 'Manrope', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border-color: #ddd6fe !important;
    }

    /* Index column */
    [data-testid="stDataFrame"] th:first-child,
    [data-testid="stDataFrame"] td:first-child {
        background: #2d0f6b !important;
        color: #c4b5fd !important;
        font-weight: 700 !important;
        border-right: 1px solid #4c1d95 !important;
    }

    /* ══════════════════════════════════════════
       INFO / WARNING / ERROR
    ══════════════════════════════════════════ */
    [data-testid="stInfo"] {
        background: #f5f3ff !important;
        border: 1px solid #ddd6fe !important;
        border-radius: 10px !important;
        color: #5b21b6 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ══════════════════════════════════════════
       STATUS + MISC
    ══════════════════════════════════════════ */
    hr {
        border: none !important;
        border-top: 1px solid rgba(139,92,246,0.1) !important;
        margin: 14px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.2); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.36); }
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
    st.markdown("""
        <div style='text-align:center; padding: 18px 12px 6px 12px;'>
            <img src='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAH0AfQDASIAAhEBAxEB/8QAHgABAAIDAAMBAQAAAAAAAAAAAAgJBQYHAgMEAQr/xABUEAABAwMDAgMFBAUFDAgEBwABAAIDBAUGBwgREiEJEzEUIkFRYRUycYEjQlKRoRZicnOCFyQzNENTY5Kio7HBGCV0k5SywsMmg6SzOER1hJWl0f/EABkBAQADAQEAAAAAAAAAAAAAAAABAgQDBf/EAC8RAQACAgICAAUDAwMFAAAAAAABAgMRITEEEhMiMkFRI2GBFDNxQqGxJDRi0fH/2gAMAwEAAhEDEQA/ALUERFnXEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBEWnaiax6V6S0YrtSdQLHjzHN6o2V1Yxk0o/wBHFz1yfg1pREzEcy3FFDXPPFW20Yx5kGKRZJl87eQx1FQezU5P1fUFjwPqGFR+y7xhNQatz24Ho9j9rbzwx92r5q8kfMiIQcH6cnj5lWisy428jHX7rS0VNN38U3ddcur2O44vaufT2SzNdx+HnOkWsv8AEg3mveXN1hawE9mtx61cD99MSp9Jc58zH+67xFSNTeJJvKglEkurUVQ0escuPWwNP+rTg/xW22jxWd0tt6fbY8NuvHr7ZaHt5/HyZY09JI8zH+64xFWThvjD3iN8cOoWi1HUNPHmVNmuj4S36iGVj+fw8wfipDYH4nO1PMjHDdsjvGJVMnAEd6tj+nq+XmU5lYB9XFo/D0UTWYda+Rjt1KWCLC4lm2HZ7amX3CMqtN/tz+OKq21k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EREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREH//2Q=='
                 style='width:170px; border-radius:12px; filter: brightness(0) invert(1);
                        opacity:0.92; margin-bottom:10px;' />
        </div>
        <div class='sidebar-sub' style='color:#a78bfa;'>Credit Beyond Cards</div>
    """, unsafe_allow_html=True)
    st.write("---")

    if st.button("🏠  Home", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📊  Dashboard", use_container_width=True):
        st.rerun()

    st.write("---")
    st.markdown("<p style='text-align:center;color:#7c3aed;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# -------------------------------------
# Page Header
# -------------------------------------
st.markdown("""
    <div style='margin-bottom: 0;'>
        <span class='page-eyebrow'>// Analytics Overview</span>
        <h1 class='page-title'>Credit <span class='grad'>Analytics Dashboard</span></h1>
        <p class='page-desc'>Monitor and assess user credit health with AI-powered insights across all registered profiles.</p>
    </div>
    <div class='header-rule'></div>
""", unsafe_allow_html=True)

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
# Section: KPI Summary
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>📈</div>Portfolio Summary</div>
</div>""", unsafe_allow_html=True)

total_users = len(df_raw)
low_users   = (df_raw["credit_score"] >= 70).sum()
high_users  = (df_raw["credit_score"] < 40).sum()
medium_users = df_raw["credit_score"].between(40, 70).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥  Total Users",           f"{total_users}")
col2.metric("✅  Low Risk  ≥ 70",       f"{low_users}")
col3.metric("⚠️  Medium Risk  40–69",  f"{medium_users}")
col4.metric("❌  High Risk  < 40",      f"{high_users}")

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------------------------
# Section: All Users Overview Table
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>👥</div>All Users Overview</div>
</div>""", unsafe_allow_html=True)

cols = [
    "user_id", "employment_type", "income_range", "city_tier", "monthly_income",
    "bank_account_age_months", "num_bank_accounts", "rent_paid_on_time",
    "utility_delay_days", "upi_txn_count", "avg_month_end_balance",
    "overdraft_event", "credit_score", "risk_level"
]

display_df = df_raw.head(2000)[cols].copy()
display_df.index = range(1, len(display_df) + 1)

st.dataframe(
    display_df.style.map(color_risk, subset=["risk_level"]),
    use_container_width=True,
    height=500
)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# -------------------------------------
# Section: AI Predictions (Last 5 Users)
# -------------------------------------
st.markdown("""<div class='sec-card'>
    <div class='sec-title'><div class='sec-icon'>🤖</div>AI Predictions — Last 5 Registered Users</div>
</div>""", unsafe_allow_html=True)

df_predict = df_added_order.tail(5).copy().iloc[::-1].reset_index(drop=True)

pred_rows = []
for idx, row in df_predict.iterrows():
    try:
        input_df = build_input_df_from_row(row)
        lr_risk, _ = onnx_predict_classifier_label_and_proba(lr_sess, input_df)
        xgb_score  = float(np.clip(onnx_predict_regressor(xgb_sess, input_df), 0, 100))
        rf_score   = float(np.clip(onnx_predict_regressor(rf_sess, input_df), 0, 100))
    except Exception:
        lr_risk, xgb_score, rf_score = "Error", "Error", "Error"

    pred_rows.append({
        "User ID":           row.get("user_id", f"User_{idx+1}"),
        "Credit Score":      row.get("credit_score", np.nan),
        "Predicted LR Risk": lr_risk,
        "XGB Score":         xgb_score,
        "RF Score":          rf_score,
        "Risk Level":        compute_risk_level(row.get("credit_score", np.nan)),
    })

pred_df = pd.DataFrame(pred_rows)
pred_df.index = range(1, len(pred_df) + 1)

st.dataframe(
    pred_df.style.map(color_risk, subset=["Risk Level"]),
    use_container_width=True,
    height=300
)
