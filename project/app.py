import streamlit as st

st.set_page_config(page_title="AltScore India | Home", layout="wide")

# Custom Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    if st.button(" Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button(" Dashboard", use_container_width=True):
        st.switch_page("pages/dashboard_page.py")
    st.write("---")

# Updated CSS - Focus on Spacing and Box Width
st.markdown("""
    <style>
    /* Hide default sidebar nav */
    [data-testid="stSidebarNav"] { display: none; }

    /* 1. Sidebar Background - Dark/Solid Black for Neon Contrast */
    [data-testid="stSidebar"] {
        background-color: #0b0b0b !important;
        border-right: 1px solid #333; /* Halka sa border separate karne ke liye */
    }

    /* 2. Sidebar Buttons - Neon Blue Look */
    [data-testid="stSidebar"] .stButton button {
        background-color: #00D1FF !important;
        color: #000000 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        border: none !important;
        transition: 0.3s all ease;
    }

    /* 3. Button Hover Effect */
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #ffffff !important; /* Hover pe white glow */
        transform: scale(1.02);
    }

    /* 4. Sidebar Text/Header Color */
    [data-testid="stSidebar"] h2 {
        color: #00D1FF !important;
        text-shadow: 0 0 10px rgba(0, 209, 255, 0.4);
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #00D1FF !important;
        opacity: 0.3;
    }

    /* Sidebar ke buttons ke beech ka gap kam karne ke liye */
    [data-testid="stSidebar"] div.stButton {
        margin-bottom: -20px !important; /* Negative margin se gap kam hoga */
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }

    /* Agar divider (---) ke beech ka gap bhi kam karna hai */
    [data-testid="stSidebar"] hr {
        margin-top: 50px !important;
        margin-bottom: 10px !important;
    }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1772&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Hero Container Spacing */
    .hero-container {
        text-align: center;
        padding-top: 60px;
    }
    .app-name { font-size: 50px; font-weight: 900; letter-spacing: 6px; color: #00D1FF; margin-bottom: 20px; }
    .hero-headline { font-size: 60px; font-weight: 800; color: #ffffff; margin-bottom: 10px; }
    .hero-subtext { font-size: 24px; color: #e0e0e0; max-width: 850px; margin: 0 auto; text-align: center; line-height: 1.6; }

    /* Feature Cards - Fixed Width and Spacing */
    .feature-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        
        /* Box ko wide aur text ke acc expand hone ke liye */
        min-height: 250px;
        height: auto;
        width: 85%;
        
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        
        /* GAP FROM REDEFINING LINE: Boxes ko neeche shift karne ke liye */
        margin-top: 120px; 
        word-wrap: break-word;
    }

    .feature-card h4 { color: #1E1E1E; font-size: 26px; font-weight: bold; margin-bottom: 20px; }
    .feature-card p { color: #444444; font-size: 17px; line-height: 1.6; }
    
    /* Button Styling */
    .stButton button {
        background-color: #00D1FF !important;
        color: black !important;
        border-radius: 10px !important;
        font-size: 22px !important;
        font-weight: bold !important;
        padding: 15px !important;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown(f"""
    <div class="hero-container">
        <div class="app-name">ALTSCORE INDIA</div>
        <h1 class="hero-headline" style="color: white;">Credit Identity for the Next Billion.</h1>
        <p class="hero-subtext" style="text-align: center; width: 100%; max-width: 100%;">Redefining creditworthiness by unlocking the power of alternative data.</p>
    </div>
    """, unsafe_allow_html=True)

# Feature section - 'gap="large"' adds horizontal space between boxes
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="feature-card"><h4>Alternative Data</h4><p>Analyze rent, utilities, and UPI patterns to build a robust financial profile without needing traditional credit cards.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card"><h4>AI-Driven Insights</h4><p>Leveraging advanced Machine Learning models like XGBoost and Random Forest to provide precise and fair credit scoring.</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card"><h4>Financial Inclusion</h4><p>Empowering students, gig workers, and small business owners to access formal credit and grow their financial future.</p></div>', unsafe_allow_html=True)

# Bottom Redirect Button
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1.5, 1])
