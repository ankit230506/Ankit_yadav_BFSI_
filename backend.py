

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
import numpy as np
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fraud Prediction API")

# Allow CORS from file:// or local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
ART_PATH = "artifacts"
model = joblib.load(f"{ART_PATH}/model.pkl")
encoders = joblib.load(f"{ART_PATH}/encoders.pkl")
scaler = joblib.load(f"{ART_PATH}/scaler.pkl")
with open(f"{ART_PATH}/features.json","r") as f:
    FEATURES = json.load(f)

# Pydantic model for incoming JSON (all fields optional because frontend may send strings)
class Transaction(BaseModel):
    Customer_ID: Optional[str] = ""
    Customer_Name: Optional[str] = ""
    Gender: Optional[str] = ""
    Age: Optional[float] = 0
    State: Optional[str] = ""
    City: Optional[str] = ""
    Bank_Branch: Optional[str] = ""
    Account_Type: Optional[str] = ""
    Transaction_ID: Optional[str] = ""
    Transaction_Date: Optional[str] = ""
    Transaction_Time: Optional[str] = ""
    Transaction_Amount: Optional[float] = 0.0
    Merchant_ID: Optional[str] = ""
    Transaction_Type: Optional[str] = ""
    Merchant_Category: Optional[str] = ""
    Account_Balance: Optional[float] = 0.0
    Transaction_Device: Optional[str] = ""
    Transaction_Location: Optional[str] = ""
    Device_Type: Optional[str] = ""
    Transaction_Currency: Optional[str] = ""
    Customer_Contact: Optional[str] = ""
    Transaction_Description: Optional[str] = ""

@app.get("/")
def root():
    return {"status": "ok", "features_expected": FEATURES}

@app.post("/predict")
def predict(txn: Transaction):
    # Build dataframe with features in right order
    data = {f: getattr(txn, f, "") for f in FEATURES}
    df = pd.DataFrame([data], columns=FEATURES)

    # Handle missing types & cast numeric fields if possible
    # Numeric fields we assume: Age, Transaction_Amount, Account_Balance
    for col in ["Age","Transaction_Amount","Account_Balance"]:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            except:
                df[col] = 0.0

    # Apply encoders for categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            # unseen labels handling: map to -1 then transform if possible
            vals = df[col].astype(str).tolist()
            encoded = []
            for v in vals:
                if v in le.classes_:
                    encoded.append(int(le.transform([v])[0]))
                else:
                    # add fallback: if unseen label, map to most frequent (0) or new index
                    encoded.append(-1)
            df[col] = encoded

    
    try:
        # If scaler is StandardScaler wrapped in ColumnTransformer (we didn't use that), this code still works
        numeric_cols = ["Age","Transaction_Amount","Account_Balance"]
        X_to_scale = df[numeric_cols].astype(float).values
        X_scaled = scaler.transform(X_to_scale)
        # replace scaled numeric values
        for i, col in enumerate(numeric_cols):
            if col in df.columns:
                df[col] = X_scaled[:, i]
    except Exception:
        # If scaler can't be applied directly, skip scaling
        pass

    # Replace any remaining non-numeric with 0
    X = df.fillna(0)

    # Ensure order
    X = X[FEATURES]

    # If model expects numeric-only array, convert to numeric
    X_matrix = X.values.astype(float)

    prob = float(model.predict_proba(X_matrix)[0][1])
    pred = int(model.predict(X_matrix)[0])

    label = "Fraud" if pred == 1 else "Not Fraud"
    return {
        "prediction": label,
        "probability": round(prob, 5),
        "probability_pct": f"{prob*100:.2f}%",
        "raw": {"pred": int(pred), "prob": prob}
    }
