import streamlit as st

st.set_page_config(page_title="AltScore India | Home", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.jpeg", width=110)

    # st.image("logo.jpeg", width=100) 
    st.write("---")
    if st.button("🏠 Home", use_container_width=True):
        st.rerun()
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

# ---------- CSS ----------
st.markdown(
    """
    <style>

    /* ══════════════════════════════════════════
       Hides the default Streamlit Page List
    ══════════════════════════════════════════ */
    [data-testid="stSidebarNav"] {
        display: none !important;
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
    color: #ffffff;
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

    # [data-testid="stSidebarNav"] { display: none; }
    # [data-testid="stSidebar"] { background-color: #0b0b0b !important; border-right: 1px solid #333; }

    # [data-testid="stSidebar"] .stButton button {
    #     background-color: #00D1FF !important;
    #     color: #000 !important;
    #     border-radius: 10px !important;
    #     font-weight: 800 !important;
    #     border: none !important;
    #     transition: 0.2s all ease;
    #     margin-bottom: 8px !important;
    # }
    # [data-testid="stSidebar"] .stButton button:hover {
    #     background-color: #fff !important;
    #     transform: scale(1.02);
    # }

    .stApp {
        background: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                    url("https://images.unsplash.com/photo-1705313382153-ed294af4b0a7?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
    }

    .hero-container { text-align: center; padding-top: 60px; }
    .app-name { font-size: 50px; font-weight: 900; letter-spacing: 6px; color: #24bcdd; margin-bottom: 16px; }
    .hero-headline { font-size: 56px; font-weight: 800; color: #fff; margin-bottom: 10px; }
    .hero-subtext { font-size: 22px; color: #e0e0e0; max-width: 900px; margin: 0 auto; line-height: 1.6; }

    .feature-card {
        background-color: #fff;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.32);
        min-height: 240px;
        width: 90%;
        margin-top: 110px;
        word-wrap: break-word;
    }
    .feature-card h4 { color: #1E1E1E; font-size: 24px; font-weight: 800; margin-bottom: 14px; }
    .feature-card p { color: #444; font-size: 16px; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero-container">
        <div class="app-name">ALTSCORE INDIA</div>
        <div class="hero-headline">Credit Identity for the Next Billion.</div>
        <div class="hero-subtext">Redefining creditworthiness by unlocking the power of alternative data.</div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.markdown(
        '<div class="feature-card"><h4>Alternative Data</h4>'
        '<p>Analyze rent, utilities, and UPI patterns to build a robust financial profile without needing traditional credit history.</p></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        '<div class="feature-card"><h4>AI-Driven Insights</h4>'
        '<p>Uses Logistic Regression (risk class) + XGBoost & Random Forest (score) to generate a fair and explainable score.</p></div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        '<div class="feature-card"><h4>Financial Inclusion</h4>'
        '<p>Empowering students, gig workers, and underserved segments with responsible access to credit.</p></div>',
        unsafe_allow_html=True
    )

st.write("")
c1, c2, c3 = st.columns([1, 1.2, 1])
with c2:
    if st.button("🚀 Get Started (Register User)", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
