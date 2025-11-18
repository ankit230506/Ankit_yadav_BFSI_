# dashboard.py
import streamlit as st
import pandas as pd
import joblib, json
import numpy as np

st.set_page_config(page_title="Fraud Dashboard", layout="wide")
st.title("Fraud Detection — Streamlit Demo")

import os
ART = os.path.dirname(os.path.abspath(__file__))   # Auto detect current folder

model = joblib.load(f"{ART}/model.pkl")
encoders = joblib.load(f"{ART}/encoders.pkl")
scaler = joblib.load(f"{ART}/scaler.pkl")
with open(f"{ART}/features.json") as f:
    FEATURES = json.load(f)

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, low_memory=False)
else:
    st.sidebar.write("No file uploaded. Will use sample if available.")
    df = None

if df is not None:
    st.write("Preview:", df.head())
    if "Is_Fraud" in df.columns:
        st.bar_chart(df["Is_Fraud"].value_counts())

st.sidebar.header("Predict single transaction")
inputs = {}
for fld in FEATURES:
    if fld in ["Age"]:
        inputs[fld] = st.sidebar.number_input(fld, value=30)
    elif fld in ["Transaction_Amount","Account_Balance"]:
        inputs[fld] = st.sidebar.number_input(fld, value=1000.0, format="%.2f")
    else:
        inputs[fld] = st.sidebar.text_input(fld, "")

if st.sidebar.button("Predict locally"):
    # build df
    X = pd.DataFrame([inputs], columns=FEATURES)
    # numeric conversion
    for c in ["Age","Transaction_Amount","Account_Balance"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    # encode categorical
    for col, le in encoders.items():
        if col in X.columns:
            vals = []
            for v in X[col].astype(str):
                if v in le.classes_:
                    vals.append(int(le.transform([v])[0]))
                else:
                    vals.append(-1)
            X[col] = vals
    # scale numeric
    try:
        num_cols = ["Age","Transaction_Amount","Account_Balance"]
        X[num_cols] = scaler.transform(X[num_cols])
    except Exception:
        pass
    X_matrix = X.fillna(0).values.astype(float)
    pred = int(model.predict(X_matrix)[0])
    prob = float(model.predict_proba(X_matrix)[0][1])
    st.metric("Prediction", "Fraud" if pred==1 else "Not Fraud", delta=f"{prob*100:.2f}%")
