import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load models
try:
    lr_model = joblib.load('models/logistic_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    rf_columns = joblib.load('models/model_columns.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

st.set_page_config(page_title="Credit Analytics Dashboard", layout="wide")

def apply_status_color(val):
    if val == "High Risk" or val == "High": 
        return "background-color: #ff6b6b; color: white; font-weight: bold;"
    elif val == "Medium Risk" or val == "Medium": 
        return "background-color: #ffd93d; color: black; font-weight: bold;"
    elif val == "Low Risk" or val == "Low": 
        return "background-color: #6bcf7f; color: white; font-weight: bold;"
    return ""

# COMPLETE CSS STYLING FROM YOUR ORIGINAL DASHBOARD
st.markdown("""
    <style>
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%) !important; 
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #ffc75f !important;
    }
    
    [data-testid="stSidebar"] hr { 
        border-color: #ff7e8b !important; 
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #ff7e8b 0%, #ffc75f 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        margin: 5px 0 !important;
        transition: transform 0.3s ease !important;
        width: 100% !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(255, 126, 139, 0.4) !important;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(rgba(245, 247, 250, 0.95), rgba(245, 247, 250, 0.95)), 
                    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1740&q=80") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    
    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #ff7e8b 0%, #ffc75f 100%);
        color: white; 
        padding: 30px; 
        border-radius: 15px; 
        text-align: center; 
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid transparent;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        color: white;
        margin: 10px 0;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Cards */
    .section-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #ff7e8b;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Hide Streamlit Navigation */
    [data-testid="stSidebarNav"] { 
        display: none !important; 
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
        border: none !important;
    }
    
    .dataframe th {
        background-color: #1e3a8a !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 15px !important;
        border: none !important;
    }
    
    .dataframe td {
        padding: 10px 15px !important;
        border-bottom: 1px solid #e9ecef !important;
    }
    
    .dataframe tr:hover {
        background-color: #f8f9fa !important;
    }
    
    /* Status Badges for Tables */
    [data-testid="stDataFrame"] td[data-color="high"] {
        background-color: #ff6b6b !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    [data-testid="stDataFrame"] td[data-color="medium"] {
        background-color: #ffd93d !important;
        color: black !important;
        font-weight: bold !important;
    }
    
    [data-testid="stDataFrame"] td[data-color="low"] {
        background-color: #6bcf7f !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Custom Sidebar Logo */
    .sidebar-logo {
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-logo h2 {
        color: #00D1FF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* Button Icons */
    [data-testid="stSidebar"] .stButton button::before {
        margin-right: 8px;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff7e8b 0%, #ffc75f 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff6b7a 0%, #ffc04d 100%);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 32px;
        }
        
        .dashboard-header {
            padding: 20px;
        }
        
        .section-card {
            padding: 15px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="dashboard-header"><h1>📊 Credit Analytics Dashboard</h1><p>Advanced AI-powered credit scoring insights</p></div>', unsafe_allow_html=True)

file_name = "data/dataset.csv"
if os.path.exists(file_name):
    try:
        df_raw = pd.read_csv(file_name)
        
        if not df_raw.empty:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                overall_avg = df_raw['alt_credit_score'].mean() if 'alt_credit_score' in df_raw.columns else 0
                st.markdown(f'<div class="metric-card"><div class="metric-value">{overall_avg:.1f}</div><div class="metric-label">Avg Score / 100</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_raw)}</div><div class="metric-label">Total Users</div></div>', unsafe_allow_html=True)
            
            with col3:
                if 'alt_credit_score' in df_raw.columns:
                    good_users = (df_raw['alt_credit_score'] >= 70).sum()
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{good_users}</div><div class="metric-label">Good (≥70)</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">--</div><div class="metric-label">Good Users</div></div>', unsafe_allow_html=True)
            
            with col4:
                if 'alt_credit_score' in df_raw.columns:
                    risky_users = (df_raw['alt_credit_score'] < 50).sum()
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{risky_users}</div><div class="metric-label">Risky (<50)</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">--</div><div class="metric-label">Risky Users</div></div>', unsafe_allow_html=True)
            
            # Display registered users
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📋 Registered Users Data")
            
            display_df = df_raw.tail(10).copy()  # Show last 10 users
            display_df.index = range(1, len(display_df) + 1)
            
            # Select columns to display
            available_cols = ['user_id', 'employment_type', 'income_range', 
                             'monthly_income', 'upi_txn_count', 'alt_credit_score']
            available_cols = [col for col in available_cols if col in display_df.columns]
            
            st.dataframe(display_df[available_cols])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Try to generate predictions for the dashboard
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("🤖 AI Model Predictions (Latest 5 Users)")
            
            # Take last 5 users
            df_predict = df_raw.tail(5).copy()
            
            predictions_data = []
            
            for idx, row in df_predict.iterrows():
                user_pred = {
                    'User ID': row.get('user_id', f'User_{idx}'),
                    'Actual Score': row.get('alt_credit_score', 'N/A')
                }
                
                # Prepare input for this user (same as in registration)
                input_data = {
                    "employment_type": row.get('employment_type', 'Salaried'),
                    "income_range": row.get('income_range', '20000-40000'),
                    "city_tier": row.get('city_tier', 2),
                    "bank_account_age_months": float(row.get('bank_account_age_months', 24)), 
                    "num_bank_accounts": row.get('num_bank_accounts', 2),
                    "monthly_income": row.get('monthly_income', 30000),
                    "rent_paid_on_time": row.get('rent_paid_on_time', 1),
                    "utility_delay_days": row.get('utility_delay_days', 0),
                    "upi_txn_count": row.get('upi_txn_count', 20),
                    "avg_month_end_balance": row.get('avg_month_end_balance', 5000),
                    "overdraft_event": row.get('overdraft_event', 0)
                }
                
                input_df = pd.DataFrame([input_data])
                
                try:
                    # Logistic Regression
                    lr_pred = lr_model.predict(input_df)[0]
                    user_pred['LR Risk'] = lr_pred
                except:
                    user_pred['LR Risk'] = "Error"
                
                try:
                    # XGBoost
                    xgb_pred = xgb_model.predict(input_df)[0]
                    user_pred['XGB Score'] = f"{xgb_pred:.1f}"
                except:
                    user_pred['XGB Score'] = "Error"
                
                try:
                    # Random Forest
                    # Preprocess for RF
                    rf_input = input_df.copy()
                    
                    # Convert categorical
                    employment_map = {"Salaried": 0, "Self-Employed": 1, "Freelancer": 2, "Unemployed": 3}
                    income_map = {"0-20000": 0, "20000-40000": 1, "40000-60000": 2, "60000+": 3}
                    account_map = {"Savings": 0, "Current": 1, "Salary": 2}
                    
                    rf_input['employment_type'] = rf_input['employment_type'].map(employment_map).fillna(0)
                    rf_input['income_range'] = rf_input['income_range'].map(income_map).fillna(0)
                    rf_input['bank_account_age_months'] = rf_input['bank_account_age_months'].map(account_map).fillna(0)
                    
                    # Create dummies
                    rf_encoded = pd.get_dummies(rf_input)
                    
                    # Align columns
                    missing_cols = set(rf_columns) - set(rf_encoded.columns)
                    for col in missing_cols:
                        rf_encoded[col] = 0
                    
                    rf_encoded = rf_encoded[rf_columns]
                    
                    # Ensure numeric
                    for col in rf_encoded.columns:
                        rf_encoded[col] = pd.to_numeric(rf_encoded[col], errors='coerce')
                    
                    rf_encoded = rf_encoded.fillna(0)
                    
                    # Predict
                    rf_array = rf_encoded.values.astype(np.float32)
                    rf_pred = rf_model.predict(rf_array)[0]
                    user_pred['RF Score'] = f"{rf_pred:.1f}"
                except:
                    user_pred['RF Score'] = "Error"
                
                predictions_data.append(user_pred)
            
            # Display predictions table
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df.index = range(1, len(predictions_df) + 1)
                
                # Apply color coding to LR Risk column
                if 'LR Risk' in predictions_df.columns:
                    styled_df = predictions_df.style.map(apply_status_color, subset=['LR Risk'])
                    st.dataframe(styled_df)
                else:
                    st.dataframe(predictions_df)
            else:
                st.info("No predictions available")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.warning("📭 Dataset is empty. Please register some users first.")
            
    except Exception as e:
        st.error(f"❌ Error processing dashboard: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.warning("📭 Dataset file not found. Please add users first.")

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <h2>ALTSCORE</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Navigation Buttons with Icons
    if st.button("🏠 Home", use_container_width=True): 
        st.switch_page("app.py")
    
    if st.button("📊 Dashboard", use_container_width=True): 
        st.rerun()
    
    if st.button("➕ New Registration", use_container_width=True): 
        st.switch_page("pages/Add_user_page.py")
    
    st.write("---")
    
    # Delete last entry button
    if st.button("🗑️ Delete Last Entry", use_container_width=True):
        if os.path.exists(file_name):
            try:
                df_edit = pd.read_csv(file_name)
                if not df_edit.empty:
                    deleted_user = df_edit.iloc[-1]['user_id'] if 'user_id' in df_edit.columns else "Unknown"
                    df_edit = df_edit.iloc[:-1]
                    df_edit.to_csv(file_name, index=False)
                    st.success(f"✅ Deleted: {deleted_user}")
                    st.rerun()
                else:
                    st.warning("No entries to delete.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("No dataset found.")
    
    st.write("---")
    
    # Footer
    st.markdown("""
        <div style="text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 12px; margin-top: 20px;">
            <p>© AI Model Predictions</p>
            <p>Advanced Credit Analytics</p>
        </div>
    """, unsafe_allow_html=True)
