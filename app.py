import streamlit as st

st.set_page_config(page_title="AltScore India | Home", layout="wide", page_icon="🏠")

# ---------- CSS Styling ----------
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
    * { 
        font-family: 'Manrope', sans-serif; 
        box-sizing: border-box; 
    }

    /* ══════════════════════════════════════════
       APP BG — blue gradient + money SVG watermark
    ══════════════════════════════════════════ */
    .stApp {
        background-color: #e3f2fd;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1000' height='700' viewBox='0 0 1000 700'%3E%3C!-- Large banknote top-right --%3E%3Crect x='680' y='30' width='280' height='140' rx='18' fill='none' stroke='%231565c0' stroke-width='2' opacity='0.18'/%3E%3Crect x='700' y='50' width='240' height='100' rx='10' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ccircle cx='820' cy='100' r='30' fill='none' stroke='%231565c0' stroke-width='1.5' opacity='0.18'/%3E%3Ccircle cx='820' cy='100' r='20' fill='none' stroke='%231565c0' stroke-width='1' opacity='0.12'/%3E%3Ctext x='808' y='107' font-family='monospace' font-size='18' fill='%231565c0' opacity='0.28' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='695' y='80' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3EALT SCORE CREDIT%3C/text%3E%3Ctext x='695' y='148' font-family='monospace' font-size='8' fill='%231565c0' opacity='0.2'%3E100 00000 0001%3C/text%3E%3C!-- Coin stack bottom-left --%3E%3Cellipse cx='90' cy='590' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='572' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='554' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cellipse cx='90' cy='536' rx='55' ry='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='35' y1='536' x2='35' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Cline x1='145' y1='536' x2='145' y2='590' stroke='%230d47a1' stroke-width='1.5' opacity='0.2'/%3E%3Ctext x='73' y='567' font-family='monospace' font-size='14' fill='%230d47a1' opacity='0.25' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Large rupee symbols --%3E%3Ctext x='50' y='80' font-family='monospace' font-size='42' fill='%231565c0' opacity='0.07' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='880' y='620' font-family='monospace' font-size='56' fill='%231565c0' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3Ctext x='430' y='660' font-family='monospace' font-size='36' fill='%230d47a1' opacity='0.06' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Percent symbol --%3E%3Ctext x='910' y='200' font-family='monospace' font-size='80' fill='%231565c0' opacity='0.05' font-weight='bold'%3E%25%3C/text%3E%3C!-- Mini banknote bottom-right --%3E%3Crect x='750' y='560' width='210' height='110' rx='14' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.15'/%3E%3Crect x='766' y='576' width='178' height='78' rx='8' fill='none' stroke='%230d47a1' stroke-width='0.8' opacity='0.1'/%3E%3Ccircle cx='855' cy='615' r='22' fill='none' stroke='%230d47a1' stroke-width='1.2' opacity='0.15'/%3E%3Ctext x='845' y='621' font-family='monospace' font-size='13' fill='%230d47a1' opacity='0.22' font-weight='bold'%3E%E2%82%B9%3C/text%3E%3C!-- Stock trend line --%3E%3Cpolyline points='200,680 270,640 340,655 420,590 500,600 580,520 660,490 740,440 820,420' fill='none' stroke='%231565c0' stroke-width='1.8' opacity='0.1'/%3E%3C!-- Credit score arc --%3E%3Cpath d='M 30 220 A 120 120 0 0 1 230 180' fill='none' stroke='%230d47a1' stroke-width='1.5' opacity='0.12' stroke-dasharray='6 6'/%3E%3Ctext x='30' y='270' font-family='monospace' font-size='9' fill='%230d47a1' opacity='0.2'%3ECREDIT SCORE%3C/text%3E%3C!-- Bar chart --%3E%3Crect x='30' y='370' width='14' height='50' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='52' y='350' width='14' height='70' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='74' y='360' width='14' height='60' rx='3' fill='%231565c0' opacity='0.1'/%3E%3Crect x='96' y='335' width='14' height='85' rx='3' fill='%230d47a1' opacity='0.1'/%3E%3Crect x='118' y='345' width='14' height='75' rx='3' fill='%231565c0' opacity='0.1'/%3E%3C/svg%3E"),
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
       HERO SECTION
    ══════════════════════════════════════════ */
    .hero-container {
        text-align: center;
        padding: 40px 20px 60px 20px;
        margin-bottom: 40px;
    }

    .hero-eyebrow {
        font-size: 0.85rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #1565c0;
        font-weight: 700;
        margin-bottom: 16px;
        display: block;
    }

    .hero-title {
        font-family: 'Fraunces', serif !important;
        font-size: clamp(2.8rem, 5vw, 4.2rem);
        font-weight: 900;
        color: #071a2e;
        line-height: 1.1;
        margin: 0 0 20px 0;
        letter-spacing: -0.02em;
    }

    .hero-title .grad {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #42a5f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        color: #0d2a4a;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 0 auto 10px auto;
        line-height: 1.75;
        text-align: center;
        padding: 0 20px;
    }

    .hero-rule {
        height: 2px;
        width: 120px;
        background: linear-gradient(90deg, transparent, #1565c0, transparent);
        border-radius: 2px;
        margin: 32px auto 0 auto;
        opacity: 0.5;
    }

    /* ══════════════════════════════════════════
       FEATURE CARDS
    ══════════════════════════════════════════ */
    .feature-card {
        background: linear-gradient(135deg, #eff7ff 0%, #ffffff 100%);
        border: 1.5px solid rgba(21,101,192,0.15);
        border-radius: 18px;
        padding: 32px;
        min-height: 250px;
        box-shadow: 
            0 4px 16px rgba(13,71,161,0.08),
            0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0d47a1, #1565c0, #42a5f5);
        border-radius: 18px 18px 0 0;
    }

    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 
            0 12px 32px rgba(13,71,161,0.15),
            0 4px 12px rgba(0,0,0,0.08);
        border-color: rgba(21,101,192,0.3);
    }

    .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 18px;
        display: block;
    }

    .feature-card h4 {
        font-family: 'Fraunces', serif !important;
        color: #071a2e;
        font-size: 1.45rem;
        font-weight: 800;
        # margin-bottom: 10px;
        letter-spacing: -0.01em;
    }

    .feature-card p {
        color: #0d2a4a;
        font-size: 1.02rem;
        line-height: 1.5;
        font-weight: 500;
    }

    /* ══════════════════════════════════════════
       CTA BUTTON
    ══════════════════════════════════════════ */
    .cta-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px;
        margin-bottom: 20px;
        width: 100%;
    }

    .cta-container > div {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .cta-container .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    .cta-container .stButton > button {
        width: auto !important;
        min-width: 340px !important;
        padding: 18px 64px !important;
        border-radius: 50px !important;
        border: none !important;
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 45%, #1976d2 100%) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        box-shadow: 0 6px 28px rgba(21,101,192,0.40), 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        margin: 0 auto !important;
    }

    .cta-container .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 40px rgba(21,101,192,0.55), 0 4px 14px rgba(0,0,0,0.12) !important;
    }

    /* ══════════════════════════════════════════
       MISC
    ══════════════════════════════════════════ */
    hr {
        border: none !important;
        border-top: 1px solid rgba(21,101,192,0.12) !important;
        margin: 18px 0 !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(21,101,192,0.22); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(21,101,192,0.4); }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.jpeg", width=110)

    st.write("---")
    st.markdown("<div class='sidebar-logo'>AltScore</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Credit Beyond Cards</div>", unsafe_allow_html=True)

    if st.button("🏠 Home", use_container_width=True):
        st.rerun()
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    if st.button("➕ New Registration", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
    st.write("---")

    st.markdown("<p style='text-align:center;color:#90caf9;font-size:0.65rem;letter-spacing:0.1em;'>v2.1 · Secure & Encrypted</p>", unsafe_allow_html=True)

# ---------- Hero Section ----------
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title" style="font-size: 3rem;"><span class="grad">AltScore</span></h1>
        <span class="hero-eyebrow">Empowering Financial Inclusion</span>
        <h1 class="hero-title">Credit Identity for the <span class="grad">Next Billion</span></h1>
        <p class="hero-subtitle">Redefining credit-worthiness by unlocking the power of alternative data : helping millions access credit without traditional barriers.</p>
        <div class="hero-rule"></div>
    </div>
""", unsafe_allow_html=True)

# ---------- Feature Cards ----------
col1, col2, col3 = st.columns(3, gap="small")

with col1:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h4>Alternative Data</h4>
            <p>Analyze rent, utilities, and UPI patterns to build a robust financial profile without needing credit history.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <h4>AI-Driven Insights</h4>
            <p>Powered by Logistic Regression, XGBoost & Random Forest to generate fair and accurate credit scores.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🌍</span>
            <h4>Financial Inclusion</h4>
            <p>Empowering students, gig workers, and underserved segments with responsible access to credit opportunities.</p>
        </div>
    """, unsafe_allow_html=True)

# ---------- CTA Button ----------
st.markdown('<div class="cta-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀  Get Started — Register User", key="cta_button", use_container_width=True):
        st.switch_page("pages/Add_user_page.py")
st.markdown('</div>', unsafe_allow_html=True)