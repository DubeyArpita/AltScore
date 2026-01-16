import os
import numpy as np
import pandas as pd
import onnxruntime as ort

MODELS_ONNX_DIR = "models_onnx"

def load_onnx_sessions():
    lr_sess  = ort.InferenceSession(os.path.join(MODELS_ONNX_DIR, "logistic_model.onnx"), providers=["CPUExecutionProvider"])
    xgb_sess = ort.InferenceSession(os.path.join(MODELS_ONNX_DIR, "xgb_model.onnx"), providers=["CPUExecutionProvider"])
    rf_sess  = ort.InferenceSession(os.path.join(MODELS_ONNX_DIR, "rf_model.onnx"), providers=["CPUExecutionProvider"])
    return lr_sess, xgb_sess, rf_sess

def _as_col(x, dtype):
    return np.array([[x]], dtype=dtype)

def build_onnx_inputs(input_df: pd.DataFrame) -> dict:
    row = input_df.iloc[0].to_dict()

    emp = str(row["employment_type"]).strip().lower()
    inc = str(row["income_range"]).strip().lower()

    return {
        "employment_type": _as_col(emp, object),
        "income_range":    _as_col(inc, object),

        "city_tier":               _as_col(int(row["city_tier"]), np.int64),
        "bank_account_age_months": _as_col(int(row["bank_account_age_months"]), np.int64),
        "num_bank_accounts":       _as_col(int(row["num_bank_accounts"]), np.int64),
        "overdraft_event":         _as_col(int(row["overdraft_event"]), np.int64),

        "monthly_income":         _as_col(float(row["monthly_income"]), np.float32),
        "rent_paid_on_time":      _as_col(float(row["rent_paid_on_time"]), np.float32),
        "utility_delay_days":     _as_col(float(row["utility_delay_days"]), np.float32),
        "upi_txn_count":          _as_col(float(row["upi_txn_count"]), np.float32),
        "avg_month_end_balance":  _as_col(float(row["avg_month_end_balance"]), np.float32),
    }

def onnx_predict_regressor(sess: ort.InferenceSession, input_df: pd.DataFrame) -> float:
    inputs = build_onnx_inputs(input_df)
    out = sess.run(None, inputs)[0]
    return float(out.ravel()[0])

def onnx_predict_classifier_label_and_proba(sess, input_df: pd.DataFrame):
    """
    Logistic Regression ONNX output handler
    Class order (confirmed):
    ['High Risk', 'Low Risk', 'Medium Risk']
    """
    inputs = build_onnx_inputs(input_df)
    outputs = sess.run(None, inputs)

    # ----- LABEL -----
    label = outputs[0]
    label = label.ravel()[0]
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8")

    # ----- PROBABILITIES -----
    probs = {}
    if len(outputs) > 1:
        proba = np.array(outputs[1])

        # Flatten safely (works for (1,3), (1,1,3), etc.)
        proba = proba.reshape(-1)

        classes = ["High Risk", "Low Risk", "Medium Risk"]

        if len(proba) == len(classes):
            probs = {
                classes[i]: float(proba[i])
                for i in range(len(classes))
            }

    return str(label), probs

def onnx_predict_regressor(sess, input_df):
    """
    Works for regressors even if output comes as:
    - [array([score], dtype=float32)]
    - [array([[score]], dtype=float32)]
    - multiple outputs (we pick first numeric)
    """
    inputs = build_onnx_inputs(input_df)
    outputs = sess.run(None, inputs)

    # Find first numeric output and return scalar float
    for out in outputs:
        arr = np.array(out)
        if arr.dtype.kind in ("f", "i"):  # float or int
            return float(arr.reshape(-1)[0])

    # If we reached here, no numeric output found -> debug info
    debug = [(np.array(o).dtype, np.array(o).shape, np.array(o)[:5]) for o in outputs]
    raise TypeError(f"Regressor ONNX produced no numeric output. Outputs summary: {debug}")

