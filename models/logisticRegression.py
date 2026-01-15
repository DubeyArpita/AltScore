#STEP 1: Upload CSV
from google.colab import files
uploaded = files.upload()

#STEP 2: Load dataset
import pandas as pd
df = pd.read_csv(list(uploaded.keys())[0])
df.head()

#STEP 3: Create risk category
def risk_label(score):
    if score < 40:
        return "High Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "Low Risk"
df['risk_category'] = df['alt_credit_score'].apply(risk_label)
df[['alt_credit_score', 'risk_category']].head()

#STEP 4: Feature/Target separation
X = df.drop(columns=['user_id', 'alt_credit_score', 'risk_category'])
y = df['risk_category']

#STEP 5: Encode categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_cols = ['employment_type', 'income_range', 'city_tier']
numeric_cols = [col for col in X.columns if col not in categorical_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

#STEP 6: Multiclass logistic regression model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'
    ))
])

#STEP 7: Train-test split and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
model.fit(X_train, y_train)

#STEP 8: Model evaluation
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#STEP 9: Live prediction function
def predict_credit_risk(input_data):
    input_df = pd.DataFrame([input_data])
    predicted_risk = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))
    return {
        "Predicted Risk Category": predicted_risk,
        "Class Probabilities": class_probabilities
    }

#STEP 10: Example manual input 
sample_user = {
    "employment_type": "Gig",
    "income_range": "20000-40000",
    "city_tier": "2",
    "bank_account_age_months": 20,
    "num_bank_accounts": 1,
    "monthly_income": 32000,
    "rent_paid_on_time": 1,
    "utility_delay_days": 5,
    "upi_txn_count": 70,
    "avg_month_end_balance": 6000,
    "overdraft_event": 0
}
predict_credit_risk(sample_user)

