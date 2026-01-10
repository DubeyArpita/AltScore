import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------- STEP 1: Load Dataset --------
df = pd.read_csv("final_dataset_v3.csv")

# -------- STEP 2: Feature & Target Split --------
X = df.drop(columns=["user_id", "alt_credit_score"])
y = df["alt_credit_score"]

# -------- STEP 3: Encode Categorical Variables --------
X = pd.get_dummies(X, drop_first=True)

# -------- STEP 4: Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- STEP 5: Train Random Forest Regressor --------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=8,
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------- STEP 6: Evaluate Model --------
y_pred_test = rf_model.predict(X_test)

print("\nMODEL EVALUATION")
print("MAE:", round(mean_absolute_error(y_test, y_pred_test), 2))
print("R2 Score:", round(r2_score(y_test, y_pred_test), 2))

# -------- STEP 7: Verdict Logic --------
def credit_verdict(score):
    if score <= 39:
        return "Low"
    elif score <= 69:
        return "Fair"
    else:
        return "Good"

# -------- STEP 8: Test on Dataset (Sample Output) --------
df_test_output = X_test.copy()
df_test_output["Actual_Score"] = y_test.values
df_test_output["Predicted_Score"] = np.clip(y_pred_test, 0, 100).astype(int)
df_test_output["Verdict"] = df_test_output["Predicted_Score"].apply(credit_verdict)

print("\nSAMPLE TEST PREDICTIONS")
print(df_test_output[["Actual_Score", "Predicted_Score", "Verdict"]].head())

# -------- STEP 9: MANUAL USER INPUT --------
new_user = {
   "employment_type": "salaried",
    "income_range": "25000-30000",
    "city_tier": 3,
    "bank_account_age_months": 112,
    "num_bank_accounts": 1,
    "monthly_income": 52000,
    "rent_paid_on_time": 0.9,
    "utility_delay_days": 2.5,
    "upi_txn_count": 87.9,
    "avg_month_end_balance": 12000,
    "overdraft_event": 0
}

# -------- STEP 10: Predict for New User --------
new_user_df = pd.DataFrame([new_user])
new_user_encoded = pd.get_dummies(new_user_df)

new_user_encoded = new_user_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_score = int(np.clip(rf_model.predict(new_user_encoded)[0], 0, 100))
verdict = credit_verdict(predicted_score)

print("\nNEW USER PREDICTION")
print("Predicted Credit Score (0–100):", predicted_score)
print("Credit Verdict:", verdict)
