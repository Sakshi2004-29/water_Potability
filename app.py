# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Water Potability - Demo", layout="wide", page_icon="💧")

# -----------------------
# Utility functions
# -----------------------
MODEL_FILES = {
    "CatBoost": "catboost_model.pkl",
    "RandomForest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "LGBM": "lgb_model.pkl",
    "AdaBoost": "ada_model.pkl"
}

FEATURES = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def preprocess_Xy(df):
    df = df.copy()
    # rename columns if needed: expects the features as listed above or similar
    # Ensure columns exist
    df = df.rename(columns=lambda x: x.strip())
    X = df[FEATURES]
    y = df["Potability"] if "Potability" in df.columns else None
    # simple fillna
    X = X.fillna(X.median())
    return X, y

def train_and_save_models(X_train, X_test, y_train, y_test):
    results = []
    # scale numeric features for models that benefit (RF/XGB)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # 1) Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, MODEL_FILES["RandomForest"])
    res = evaluate_metrics("RandomForest", rf, X_test_scaled, y_test)
    results.append(res)

    # 2) AdaBoost
    base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada = AdaBoostClassifier(estimator=base_tree, n_estimators=150, learning_rate=0.1, random_state=42)
    ada.fit(X_train.fillna(X_train.median()), y_train)
    joblib.dump(ada, MODEL_FILES["AdaBoost"])
    res = evaluate_metrics("AdaBoost", ada, X_test, y_test)
    results.append(res)

    # 3) CatBoost
    cat = CatBoostClassifier(iterations=400, depth=8, learning_rate=0.05, verbose=0, random_seed=42)
    cat.fit(X_train.fillna(X_train.median()), y_train)
    joblib.dump(cat, MODEL_FILES["CatBoost"])
    res = evaluate_metrics("CatBoost", cat, X_test, y_test)
    results.append(res)

    # 4) LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
    lgbm.fit(X_train.fillna(X_train.median()), y_train)
    joblib.dump(lgbm, MODEL_FILES["LGBM"])
    res = evaluate_metrics("LGBM", lgbm, X_test, y_test)
    results.append(res)

    # 5) XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train.fillna(X_train.median()), y_train)
    joblib.dump(xgb_model, MODEL_FILES["XGBoost"])
    res = evaluate_metrics("XGBoost", xgb_model, X_test, y_test)
    results.append(res)

    return pd.DataFrame(results)

def evaluate_metrics(name, model, X_test, y_test):
    # if scaler exists, caller should pass scaled X if needed
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    r2 = r2_score(y_test, y_pred)
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "R2 Score": r2}

def load_models_if_exist():
    loaded = {}
    for k, fname in MODEL_FILES.items():
        if os.path.exists(fname):
            loaded[k] = joblib.load(fname)
    if os.path.exists("scaler.pkl"):
        loaded["scaler"] = joblib.load("scaler.pkl")
    return loaded

# -----------------------
# UI
# -----------------------
st.title("💧 Water Potability — Deploy & Test")
st.write("Deploy your trained models with a simple and friendly interface. You can train models, compare them, test samples, and download predictions.")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Upload & EDA", "Train & Compare", "Predict"])

# Home
if page == "Home":
    st.header("Welcome")
    st.markdown("""
    **What you can do here:**  
    - Upload your `water_potability_final.csv` or use sample data.  
    - Train and compare 5 ensemble models (CatBoost, RandomForest, XGBoost, LGBM, AdaBoost).  
    - Predict single sample or batch CSV and download predictions.  
    """)
    st.info("If trained model files are present in the app folder, the app will load them to skip retraining.")

    # show sample data button
    if st.button("Show sample rows"):
        sample = pd.DataFrame([{
            "ph":7.0,"Hardness": 150,"Solids":20000,"Chloramines": 7.0,"Sulfate":330,"Conductivity": 450,"Organic_carbon": 10,"Trihalomethanes": 70,"Turbidity": 3,"Potability":1
        }])
        st.dataframe(sample)

# Upload & EDA
elif page == "Upload & EDA":
    st.header("Upload Dataset & Quick EDA")
    uploaded = st.file_uploader("Upload your CSV (water_potability_final.csv)", type=["csv"])
    if uploaded:
        df = load_data(uploaded)
        st.subheader("First 5 rows")
        st.dataframe(df.head())

        st.subheader("Dataset Info")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("Missing Values (count)")
        mv = df.isnull().sum()
        st.bar_chart(mv)

        st.subheader("Potability Distribution")
        if "Potability" in df.columns:
            fig = px.pie(df, names='Potability', title="Potable (1) vs Non-potable (0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Potability' column found for distribution.")

        st.subheader("Correlation Heatmap (top)")
        corr = df[FEATURES + (["Potability"] if "Potability" in df.columns else [])].corr()
        fig2 = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Histogram — pH distribution")
        fig3 = px.histogram(df, x='ph', nbins=30, title="pH distribution (example)")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("Upload the dataset to see EDA visuals. Use the sample data if you don't have the CSV.")

# Train & Compare
elif page == "Train & Compare":
    st.header("Train Models & Compare Performance")

    uploaded = st.file_uploader("Upload CSV for training (with Potability column)", type=["csv"], key="train_csv")
    if uploaded is None and st.button("Use demo (sample) dataset"):
        # create a tiny synthetic dataset (only for demo - better to upload real CSV)
        st.warning("Demo dataset is very small; real results require your full dataset.")
        demo = pd.DataFrame({
            "ph":[7.0,6.5,8.2,7.1,6.8,5.5,7.9,6.7,7.2,8.0],
            "Hardness":[150,180,120,160,140,200,130,170,155,145],
            "Solids":[21000,25000,18000,23000,20000,30000,17000,22000,20500,19500],
            "Chloramines":[7.1,8.0,6.5,7.5,6.9,9.0,6.0,7.8,7.0,6.6],
            "Sulfate":[330,360,300,345,320,400,290,355,335,315],
            "Conductivity":[450,480,420,460,445,510,410,470,455,440],
            "Organic_carbon":[10,12,9,11,10,14,8,13,10,9],
            "Trihalomethanes":[70,75,65,72,69,80,60,74,71,68],
            "Turbidity":[3,4,2,3,3,5,2,4,3,2],
            "Potability":[1,0,1,1,1,0,1,0,1,1]
        })
        df = demo
    elif uploaded:
        df = load_data(uploaded)
    else:
        st.info("Please upload your dataset to train models (CSV must contain 'Potability' column).")
        df = None

    if df is not None:
        if "Potability" not in df.columns:
            st.error("Uploaded CSV must have a 'Potability' column (1 = Safe, 0 = Unsafe).")
        else:
            st.subheader("Data preview")
            st.dataframe(df.head())

            X, y = preprocess_Xy(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # check for existing models
            existing = load_models_if_exist()
            if len(existing) == len(MODEL_FILES):
                st.success("Found saved models on disk. Loaded them for evaluation.")
                # evaluate each loaded model
                scaler = existing.get("scaler", None)
                results = []
                for name in MODEL_FILES.keys():
                    model = existing[name]
                    if name == "RandomForest":
                        X_test_for = scaler.transform(X_test)
                    else:
                        X_test_for = X_test.fillna(X_test.median())
                    results.append(evaluate_metrics(name, model, X_test_for, y_test))
                df_results = pd.DataFrame(results)
            else:
                st.info("Training models now. This may take a few minutes depending on your machine.")
                df_results = train_and_save_models(X_train, X_test, y_train, y_test)
                st.success("Training complete. Models saved to disk.")

            st.subheader("Model Comparison Table")
            st.table(df_results.set_index("Model").round(3))

            st.subheader("Accuracy Comparison")
            fig = px.bar(df_results, x="Model", y="Accuracy", text=df_results["Accuracy"].round(3),
                         title="Accuracy Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # Offer CSV download
            csv = df_results.to_csv(index=False).encode()
            st.download_button("Download metrics CSV", data=csv, file_name="model_metrics.csv", mime="text/csv")

# Predict
elif page == "Predict":
    st.header("Predict Potability — Single or Batch")

    # load models
    loaded = load_models_if_exist()
    if len(loaded) == 0:
        st.warning("No trained models found. Please go to Train & Compare to train models first (or put trained model .pkl files in the folder).")
    else:
        st.write("Loaded models: " + ", ".join([k for k in loaded.keys() if k != "scaler"]))

    # sidebar choose model
    model_choice = st.selectbox("Choose model for prediction", ["CatBoost","RandomForest","XGBoost","LGBM","AdaBoost"])
    model = loaded.get(model_choice, None)
    scaler = loaded.get("scaler", None)

    st.subheader("Single sample input")
    cols = st.columns(3)
    sample = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            val = st.number_input(feat, value=0.0, format="%.3f")
            sample[feat] = val
    if st.button("Predict single sample"):
        if model is None:
            st.error("Selected model not loaded. Train models first.")
        else:
            Xs = pd.DataFrame([sample])
            Xs = Xs[FEATURES].astype(float).fillna(Xs.median())
            Xs_for = scaler.transform(Xs) if (scaler is not None and model_choice == "RandomForest") else Xs
            pred = model.predict(Xs_for)[0]
            proba = model.predict_proba(Xs_for)[0] if hasattr(model, "predict_proba") else None
            st.write("**Prediction:**", "🟢 Safe (1)" if pred==1 else "🔴 Unsafe (0)")
            if proba is not None:
                st.write("Probability:", { "Unsafe": float(proba[0]), "Safe": float(proba[1]) })

    st.markdown("---")
    st.subheader("Batch prediction (CSV)")
    up = st.file_uploader("Upload CSV with features (no Potability column needed) for batch prediction", type=["csv"])
    if up is not None:
        df_batch = load_data(up)
        st.write("First 5 rows:")
        st.dataframe(df_batch.head())

        # prepare
        Xbatch = df_batch.copy()
        Xbatch = Xbatch[FEATURES].fillna(Xbatch.median())
        Xbatch_for = scaler.transform(Xbatch) if (scaler is not None and model_choice == "RandomForest") else Xbatch
        if st.button("Run batch prediction"):
            preds = model.predict(Xbatch_for)
            df_out = df_batch.copy()
            df_out["Predicted_Potability"] = preds
            st.dataframe(df_out.head())
            csv2 = df_out.to_csv(index=False).encode()
            st.download_button("Download predictions CSV", data=csv2, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Built for demo: train on your full dataset for real results. Models saved to disk for faster subsequent runs.")
