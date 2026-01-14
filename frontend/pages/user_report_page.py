# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Credit Analysis Report", layout="centered")

# # --- CSS Section: Pure Styling ---
# st.markdown("""
#     <style>
#     /* Default Sidebar Hide */
#     [data-testid="stSidebarNav"] { display: none !important; }
#     /* Sidebar ke buttons ka naya Neon Blue look */
#     [data-testid="stSidebar"] .stButton button {
#         background: linear-gradient(135deg, #00D1FF 0%, #0072FF 100%) !important;
#         color: white !important;
#         border: none !important;
#         border-radius: 12px !important; /* Thode square but rounded corners professional lagte hain */
#         padding: 10px 20px !important;
#         font-weight: 700 !important;
#         font-size: 16px !important;
#         transition: all 0.3s ease !important;
#         box-shadow: 0 4px 15px rgba(0, 209, 255, 0.3) !important;
#         text-transform: uppercase; /* Buttons ko bold look dene ke liye */
#         letter-spacing: 1px;
#     }

#     /* Hover effect: Mouse le jaane par glow badhega */
#     [data-testid="stSidebar"] .stButton button:hover {
#         transform: translateY(-2px) !important;
#         box-shadow: 0 8px 25px rgba(0, 209, 255, 0.5) !important;
#         background: #00D1FF !important; /* Hover pe solid neon blue */
#         color: black !important; /* Contrast ke liye text black ho jayega */
#     }

#     /* Sidebar ke andar ke dividers (hr lines) ko visible banane ke liye */
#     [data-testid="stSidebar"] hr {
#         border-top: 2px solid #00D1FF !important; /* Neon Blue color */
#         opacity: 0.5 !important;
#         margin-top: 10px !important;
#         margin-bottom: 10px !important;
#     }
#     /* Background Setup */
#     .stApp {
#         background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
#                     url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1744&q=80");
#         background-size: cover;
#     }

#     /* Sidebar Styling */
#     [data-testid="stSidebar"] { background-color: #001529 !important; }
#     [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { background-color: #001529 !important; }

#     /* 1. SINGLE WHITE BOX: Table ke liye main container */

#     /* 2. TABLE SIZE & SPACING: Table ko bada aur saaf banane ke liye */
#     [data-testid="stTable"] {
#         width: 100% !important;
#         background-color: white !important;
#     }

#     [data-testid="stTable"] td {
#         padding: 25px !important; /* Table rows ki height badhane ke liye */
#         font-size: 18px !important;
#         color: #1E1E1E !important;
#         text-align: left !important;
#         border-bottom: 1px solid #f0f0f0 !important;
#     }

#     [data-testid="stTable"] th {
#         padding: 20px !important;
#         font-size: 20px !important;
#         background-color: #f8f9fa !important;
#         color: #333 !important;
#         text-align: left !important;
#         font-weight: bold !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# if 'report_data' in st.session_state:
#     # --- Balloons Logic ---
#     if 'show_balloons' not in st.session_state:
#         st.balloons()
#         st.session_state.show_balloons = False

#     data = st.session_state['report_data']
    
#     # Page Titles
#     st.markdown(f"<h1 style='text-align: center; color: #00D1FF;'> Personalized Credit Report</h1>", unsafe_allow_html=True)
#     st.markdown(f"<h3 style='text-align: center; color: white;'>User ID: {data['user_id']}</h3>", unsafe_allow_html=True)

#     # --- Logistic Regression Range Logic ---
#     lr_val = data['lr']
#     if lr_val <= 40:
#     lr_display = "🔴 High Risk"
#     elif 40 < lr_val <= 70:
#         lr_display = "🟡 Medium Risk"
#     else:
#         lr_display = "🟢 Low Risk"

#     # --- Final Table Data ---
#     report_df = pd.DataFrame({
#         "Analysis Model": ["Logistic Regression", "Random Forest Score", "XGBoost Score", "FINAL SCORE"],
#         "Result": [lr_display, f"{data['rf']}/100", f"{data['xgb']}/100", f"{data['final']}/100"],
#         "Verdict": [f"Score: {lr_val}", "RF Prediction", "XGB Prediction", "✅ ELIGIBLE" if data['final'] > 60 else "❌ RISKY"]
#     })
#     report_df.index = report_df.index + 1

#     # --- Table in a Single White Card ---
#     st.table(report_df)
#     st.markdown('</div>', unsafe_allow_html=True)
#     st.write("---")
#     st.markdown("<h4 style='color: white;'>Overall Financial Health</h4>", unsafe_allow_html=True)
#     st.progress(data['final'] / 100)

#     # Sidebar Navigation
#    # Sidebar Navigation - Corrected Indentation
#    # Sidebar Navigation - With Visible Lines
#     with st.sidebar:
#         st.markdown("<h2 style='text-align: center; color: #00D1FF;'>User Credit Analysis</h2>", unsafe_allow_html=True)
        
#         # Line 1
#         st.markdown("<hr style='border: 1px solid #00D1FF;'>", unsafe_allow_html=True)
        
#         if st.button(" Home", use_container_width=True): 
#             st.switch_page("main_app.py")
            
#         if st.button(" Dashboard", use_container_width=True): 
#             st.switch_page("Pages/2_Dashboard.py")
            
#         if st.button(" New Registration", use_container_width=True): 
#             st.switch_page("Pages/3_Add_User.py")
            
#         # Line 2
#         st.markdown("<hr style='border: 1px solid #00D1FF;'>", unsafe_allow_html=True)

# else:
#     st.error("No data found! Please register first.")*

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Analysis Report", layout="centered")

# --- CSS Section: Pure Styling ---
st.markdown("""
    <style>
    /* Default Sidebar Hide */
    [data-testid="stSidebarNav"] { display: none !important; }
    /* Sidebar ke buttons ka naya Neon Blue look */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #00D1FF 0%, #0072FF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 209, 255, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hover effect */
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 209, 255, 0.5) !important;
        background: #00D1FF !important;
        color: black !important;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-top: 2px solid #00D1FF !important;
        opacity: 0.5 !important;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    
    /* Background Setup */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1744&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #001529 !important; 
    }

    /* Main container for white box */
    .main-white-box {
        background-color: white !important;
        padding: 40px !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
        margin-top: 20px !important;
        margin-bottom: 40px !important;
    }

    /* Table styling */
    table {
        width: 100% !important;
        border-collapse: collapse !important;
        background-color: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    th {
        background-color: #f0f8ff !important;
        padding: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        color: #333 !important;
        text-align: left !important;
        border-bottom: 2px solid #00D1FF !important;
    }

    td {
        padding: 20px !important;
        font-size: 16px !important;
        color: #333 !important;
        text-align: left !important;
        border-bottom: 1px solid #eee !important;
    }

    tr:last-child td {
        border-bottom: none !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #00D1FF !important;
    }

    /* Center content */
    .stApp > div > div {
        padding: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

if 'report_data' in st.session_state:
    # --- Balloons Logic ---
    if 'show_balloons' not in st.session_state:
        st.balloons()
        st.session_state.show_balloons = True

    data = st.session_state['report_data']
    
    # Page Titles
    st.markdown(f"<h1 style='text-align: center; color: #00D1FF;'> Personalized Credit Report</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: white;'>User ID: {data['user_id']}</h3>", unsafe_allow_html=True)

    # --- Logistic Regression Range Logic (FIXED INDENTATION) ---
    lr_val = data['lr']
    if lr_val <= 40:
        lr_display = "🔴 High Risk"
    elif 40 < lr_val <= 70:
        lr_display = "🟡 Medium Risk"
    else:
        lr_display = "🟢 Low Risk"

    # --- Final Table Data ---
    report_df = pd.DataFrame({
        "Analysis Model": ["Logistic Regression", "Random Forest Score", "XGBoost Score", "FINAL SCORE"],
        "Result": [lr_display, f"{data['rf']}/100", f"{data['xgb']}/100", f"{data['final']}/100"],
        "Verdict": [f"Score: {lr_val}", "RF Prediction", "XGB Prediction", "✅ ELIGIBLE" if data['final'] > 60 else "❌ RISKY"]
    })
    report_df.index = report_df.index + 1

    # --- Table in a White Box ---
    st.markdown('<div class="main-white-box">', unsafe_allow_html=True)
    st.table(report_df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Progress bar section
    st.write("---")
    st.markdown("<h4 style='color: white; text-align: center;'>Overall Financial Health</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.progress(data['final'] / 100)
        st.markdown(f"<p style='color: white; text-align: center; font-size: 24px; font-weight: bold;'>{data['final']}/100</p>", unsafe_allow_html=True)

    # Final verdict message
    if data['final'] > 60:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-top: 30px;'>
            <h4 style='color: #155724; margin: 0;'>🎉 Congratulations!</h4>
            <p style='color: #155724; margin: 10px 0 0 0;'>Based on our analysis, this user has a good credit profile and is eligible for credit facilities.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; margin-top: 30px;'>
            <h4 style='color: #721c24; margin: 0;'>⚠️ Caution Advised</h4>
            <p style='color: #721c24; margin: 10px 0 0 0;'>This user's credit profile shows higher risk. Consider additional verification before extending credit.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("No data found! Please register first.")

# Sidebar Navigation (fixed indentation)
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>User Credit Analysis</h2>", unsafe_allow_html=True)
    
    # Line 1
    st.markdown("<hr style='border: 1px solid #00D1FF;'>", unsafe_allow_html=True)
    
    if st.button("🏠 Home", use_container_width=True): 
        st.switch_page("main_app.py")
        
    if st.button("📊 Dashboard", use_container_width=True): 
        st.switch_page("Pages/2_Dashboard.py")
        
    if st.button("➕ New Registration", use_container_width=True): 
        st.switch_page("Pages/3_Add_User.py")
        
    # Line 2
    st.markdown("<hr style='border: 1px solid #00D1FF;'>", unsafe_allow_html=True)
    
    # Additional information
    st.markdown("""
    <div style='padding: 10px; color: #00D1FF; font-size: 14px;'>
    <p><strong>Report Generated:</strong><br>Real-time AI Analysis</p>
    <p><strong>Confidential:</strong><br>For internal use only</p>
    </div>
    """, unsafe_allow_html=True)
