# model_train.py
# Train XGBoost fraud model and save model + preprocess artifacts

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import os
import json

CSV_PATH = r"C:\Users\Ankit Yadav\Downloads\Certificates\Bank_Transaction_Fraud_Detection.csv"  # put your CSV here
TARGET = "Is_Fraud"

# FEATURES in exact order required by UI/back-end
FEATURES = [
    "Customer_ID","Customer_Name","Gender","Age","State","City","Bank_Branch",
    "Account_Type","Transaction_ID","Transaction_Date","Transaction_Time",
    "Transaction_Amount","Merchant_ID","Transaction_Type","Merchant_Category",
    "Account_Balance","Transaction_Device","Transaction_Location","Device_Type",
    "Transaction_Currency","Customer_Contact","Transaction_Description"
]

assert os.path.exists(CSV_PATH), f"{CSV_PATH} not found. Place the CSV in the same folder."

# Load
df = pd.read_csv(CSV_PATH, low_memory=False)
if TARGET not in df.columns:
    raise ValueError(f"{TARGET} column not found in dataset.")

# Keep only features that exist in CSV
present = [c for c in FEATURES if c in df.columns]
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    print("Warning: these fields are missing from CSV and will be created blank:", missing)
    for c in missing:
        df[c] = ""  # create empty column so feature order is consistent

# Drop rows with missing target
df = df.dropna(subset=[TARGET])

# Basic cleanup: fillna
for col in present:
    if df[col].dtype == object:
        df[col] = df[col].fillna("").astype(str)
    else:
        df[col] = df[col].fillna(0)

# Separate X,y and enforce column order
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

# Determine categorical vs numeric automatically (treat object as categorical)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [c for c in FEATURES if c not in categorical_cols]

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# Label encode categoricals
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale numeric columns
scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
print("Training XGBoost...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
joblib.dump(encoders, "artifacts/encoders.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
with open("artifacts/features.json", "w") as f:
    json.dump(FEATURES, f)

print("Saved: artifacts/model.pkl, encoders.pkl, scaler.pkl, features.json")
