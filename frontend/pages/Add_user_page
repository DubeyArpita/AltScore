import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load models
try:
    lr_model = joblib.load('logistic_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    rf_columns = joblib.load('model_columns.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

st.set_page_config(page_title="Register User", layout="centered")

# CSS Styling (keep your existing CSS here)

# ... [Your existing CSS code] ...

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D1FF;'>ALTSCORE</h2>", unsafe_allow_html=True)
    st.write("---")
    
    if st.button("🏠 Home", use_container_width=True): 
        st.switch_page("main_app.py")
    
    if st.button("📊 Dashboard", use_container_width=True): 
        st.switch_page("Pages/2_Dashboard.py")
    
    st.write("---")

st.markdown("<h1>📝 User Registration</h1>", unsafe_allow_html=True)

# Check if dataset exists
file_name = "final_dataset_v3.csv"
if not os.path.exists(file_name):
    df_template = pd.DataFrame(columns=[
        "user_id", "employment_type", "income_range", "city_tier", 
        "bank_account_age_months", "num_bank_accounts", "monthly_income", 
        "rent_paid_on_time", "utility_delay_days", "upi_txn_count", 
        "avg_month_end_balance", "overdraft_event", "alt_credit_score"
    ])
    df_template.to_csv(file_name, index=False)

# Form for user registration
with st.form("user_registration_form"):
    st.markdown('<div class="registration-form">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.text_input("User ID *", placeholder="Enter unique User ID")
        employment_type = st.selectbox("Employment Type *", ["Salaried", "Self-Employed", "Freelancer", "Unemployed"])
        income_range = st.selectbox("Income Range (Monthly) *", ["0-20000", "20000-40000", "40000-60000", "60000+"])
        city_tier = st.selectbox("City Tier *", [1, 2, 3])
        bank_account_age_months = st.number_input("Bank Account Age (Months) *", min_value=0, max_value=240,value=24,step=1,
        help="Enter number of months (e.g., 24 for 2 years old account)"
)
    
    with col2:
        num_bank_accounts = st.number_input("Number of Bank Accounts *", min_value=1, max_value=10, value=1, step=1)
        monthly_income = st.number_input("Monthly Income (₹) *", min_value=0, value=30000, step=1000)
        rent_paid_on_time = st.selectbox("Rent Paid on Time? *", ["Yes", "No"])
        utility_delay_days = st.number_input("Utility Bill Defaults (Count) *", min_value=0, value=0, step=1)
        upi_txn_count = st.number_input("Monthly UPI Transaction Count *", min_value=0, value=20, step=1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        avg_month_end_balance = st.number_input("Average Monthly Balance (₹) *", min_value=0, value=5000, step=500)
    
    with col4:
        overdraft_event = st.selectbox("Overdraft Availed? *", ["No", "Yes"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    submit_button = st.form_submit_button("💾 Save User & Generate Score 🚀", use_container_width=True)

if submit_button:
    if not user_id:
        st.error("❌ Please enter a User ID!")
    else:
        with st.spinner("Processing user data and generating score..."):
            # Prepare input data EXACTLY as the models expect
            input_data = {
                "employment_type": employment_type,  # String
                "income_range": income_range,        # String
                "city_tier": city_tier,              # Integer
                "bank_account_age_months": bank_account_age_months,  # String
                "num_bank_accounts": num_bank_accounts,              # Integer
                "monthly_income": monthly_income,                    # Float
                "rent_paid_on_time": 1 if rent_paid_on_time == "Yes" else 0,  # 0/1
                "utility_delay_days": utility_delay_days,            # Integer
                "upi_txn_count": upi_txn_count,                      # Integer
                "avg_month_end_balance": avg_month_end_balance,      # Float
                "overdraft_event": 1 if overdraft_event == "Yes" else 0  # 0/1
            }
            
            # Create DataFrame with EXACT column names and order
            input_df = pd.DataFrame([input_data])
            
            try:
                # ============================================
                # 1. LOGISTIC REGRESSION (PIPELINE)
                # ============================================
                # Pipeline will handle preprocessing internally
                lr_prediction = lr_model.predict(input_df)[0]  # Returns: 'Low Risk', 'Medium Risk', or 'High Risk'
                
                # Convert risk level to score
                risk_to_score = {"Low Risk": 85, "Medium Risk": 55, "High Risk": 25}
                lr_score = risk_to_score.get(lr_prediction, 50)
                
                # ============================================
                # 2. XGBOOST (PIPELINE)
                # ============================================
                # Pipeline will handle preprocessing internally
                xgb_pred_raw = xgb_model.predict(input_df)[0]  # Returns numeric score
                xgb_score = float(np.clip(xgb_pred_raw, 0, 100))
                
                # ============================================
                # 3. RANDOM FOREST (PLAIN MODEL)
                # ============================================
                # Need to preprocess data to match what RF was trained on
                
                # Step 1: Convert categorical strings to numbers
                # (This must match how you preprocessed during training)
                rf_input = input_df.copy()
                
                # Convert employment_type
                employment_map = {"Salaried": 0, "Self-Employed": 1, "Freelancer": 2, "Unemployed": 3}
                rf_input['employment_type'] = rf_input['employment_type'].map(employment_map).fillna(0)
                
                # Convert income_range
                income_map = {"0-20000": 0, "20000-40000": 1, "40000-60000": 2, "60000+": 3}
                rf_input['income_range'] = rf_input['income_range'].map(income_map).fillna(0)
                
                # Convert bank_account_age_months (treating as account type)
                account_map = {"Savings": 0, "Current": 1, "Salary": 2}
                rf_input['bank_account_age_months'] = rf_input['bank_account_age_months'].map(account_map).fillna(0)
                
                # Step 2: Create one-hot encoding if needed
                # Based on rf_columns, it seems RF was trained on 21 features
                # This likely includes one-hot encoded versions
                
                # Create dummy variables
                rf_encoded = pd.get_dummies(rf_input)
                
                # Step 3: Align with the columns RF was trained on
                # Add missing columns with 0 values
                missing_cols = set(rf_columns) - set(rf_encoded.columns)
                for col in missing_cols:
                    rf_encoded[col] = 0
                
                # Reorder columns to match training
                rf_encoded = rf_encoded[rf_columns]
                
                # Step 4: Ensure all values are numeric
                for col in rf_encoded.columns:
                    rf_encoded[col] = pd.to_numeric(rf_encoded[col], errors='coerce')
                
                rf_encoded = rf_encoded.fillna(0)
                
                # Step 5: Convert to numpy array
                rf_array = rf_encoded.values.astype(np.float32)
                
                # Step 6: Make prediction
                rf_pred_raw = rf_model.predict(rf_array)[0]
                rf_score = float(np.clip(rf_pred_raw, 0, 100))
                
                # ============================================
                # 4. CALCULATE FINAL SCORE
                # ============================================
                final_score = int((lr_score + xgb_score + rf_score) / 3)
                
                # Determine eligibility
                if final_score >= 70:
                    eligibility = "✅ ELIGIBLE"
                    risk_level = "Low Risk"
                elif final_score >= 50:
                    eligibility = "⚠️ CONDITIONAL"
                    risk_level = "Medium Risk"
                else:
                    eligibility = "❌ RISKY"
                    risk_level = "High Risk"
                
                # ============================================
                # 5. STORE RESULTS
                # ============================================
                st.session_state['report_data'] = {
                    'user_id': user_id,
                    'lr': lr_score,
                    'xgb': xgb_score,
                    'rf': rf_score,
                    'final': final_score,
                    'lr_risk': lr_prediction,
                    'eligibility': eligibility,
                    'risk_level': risk_level
                }
                
                # ============================================
                # 6. SAVE TO DATASET
                # ============================================
                new_entry = {
                    "user_id": user_id,
                    "employment_type": employment_type,
                    "income_range": income_range,
                    "city_tier": city_tier,
                    "bank_account_age_months": bank_account_age_months,
                    "num_bank_accounts": num_bank_accounts,
                    "monthly_income": monthly_income,
                    "rent_paid_on_time": 1 if rent_paid_on_time == "Yes" else 0,
                    "utility_delay_days": utility_delay_days,
                    "upi_txn_count": upi_txn_count,
                    "avg_month_end_balance": avg_month_end_balance,
                    "overdraft_event": 1 if overdraft_event == "Yes" else 0,
                    "alt_credit_score": final_score
                }
                
                # Append to CSV
                if os.path.exists(file_name):
                    df_csv = pd.read_csv(file_name)
                    df_csv = pd.concat([df_csv, pd.DataFrame([new_entry])], ignore_index=True)
                else:
                    df_csv = pd.DataFrame([new_entry])
                
                df_csv.to_csv(file_name, index=False)
                
                # ============================================
                # 7. SHOW RESULTS
                # ============================================
                st.success(f"✅ User {user_id} registered successfully!")
                
                # Show scores in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Logistic Regression", f"{lr_score:.0f}", lr_prediction)
                
                with col2:
                    st.metric("XGBoost", f"{xgb_score:.1f}")
                
                with col3:
                    st.metric("Random Forest", f"{rf_score:.1f}")
                
                with col4:
                    st.metric("Final Score", f"{final_score}", risk_level)
                
                # Navigate to report page
                import time
                time.sleep(3)
                st.switch_page("Pages/4_User_Report.py")
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                import traceback
                st.code(traceback.format_exc())
