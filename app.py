import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from imblearn.over_sampling import SMOTE
import io
import joblib
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Bias Auditor", layout="wide")
st.title("🧠 AI Bias Auditor — Make Your Dataset Fairer")

st.write("""
Upload your dataset, select your target column and sensitive feature (like gender or race),  
and this app will train a model, detect bias, debias the data, and let you download an unbiased version.
""")

# --- SIDEBAR OPTION ---
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose Mode", ["Upload Dataset", "Load Notebook Results"])

# --- OPTION 1: UPLOAD MODE ---
if mode == "Upload Dataset":
    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### 📊 Dataset Preview")
        st.dataframe(data.head())

        target_col = st.selectbox("🎯 Select Target Column", data.columns)
        sensitive_col = st.selectbox("⚖️ Select Sensitive Column (e.g. gender, race)", data.columns)

        if st.button("🚀 Run Bias Detection & Debiasing"):
            st.info("Processing... please wait.")

            df = data.dropna().copy()
            le = LabelEncoder()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = le.fit_transform(df[col])

            X = df.drop(target_col, axis=1)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.metric("✅ Model Accuracy (Before Debiasing)", f"{acc:.3f}")

            sensitive = X_test[sensitive_col]
            metric_before = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive)
            dpd_before = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)

            st.write("### ⚖️ Bias Before Debiasing")
            fig1, ax1 = plt.subplots()
            metric_before.by_group.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
            ax1.set_title(f"Selection Rate by {sensitive_col} (Before)")
            st.pyplot(fig1)
            st.metric("Demographic Parity Difference", f"{dpd_before:.3f}")

            sm = SMOTE()
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            model.fit(X_train_res, y_train_res)
            y_pred_res = model.predict(X_test)

            acc_res = accuracy_score(y_test, y_pred_res)
            dpd_after = demographic_parity_difference(y_test, y_pred_res, sensitive_features=sensitive)

            st.metric("✅ Accuracy (After Debiasing)", f"{acc_res:.3f}")
            st.metric("⚖️ Demographic Parity Difference (After)", f"{dpd_after:.3f}")

            metric_after = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred_res, sensitive_features=sensitive)
            st.write("### 📊 Bias After Debiasing")
            fig2, ax2 = plt.subplots()
            metric_after.by_group.plot(kind='bar', ax=ax2, color=['lightgreen', 'orange'])
            ax2.set_title(f"Selection Rate by {sensitive_col} (After)")
            st.pyplot(fig2)

            unbiased_df = X_test.copy()
            unbiased_df[target_col] = y_pred_res
            buffer = io.BytesIO()
            unbiased_df.to_csv(buffer, index=False)
            st.download_button(
                label="⬇️ Download Unbiased Dataset (CSV)",
                data=buffer.getvalue(),
                file_name="unbiased_dataset.csv",
                mime="text/csv"
            )

            st.success("✅ Bias detection and correction complete!")

    else:
        st.info("👆 Upload a dataset to start the bias analysis.")

# --- OPTION 2: LOAD NOTEBOOK RESULTS ---
else:
    st.subheader("📈 Load Results from bias_auditor.ipynb")
    if os.path.exists("data/results.pkl"):
        X_test, y_test, y_pred, metric = joblib.load("data/results.pkl")
        st.success("Loaded saved bias analysis results.")

        fig, ax = plt.subplots()
        metric.by_group.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
        ax.set_title("Selection Rate by Sensitive Feature (Notebook)")
        st.pyplot(fig)

        unbiased_path = "data/unbiased_dataset.csv"
        if os.path.exists(unbiased_path):
            with open(unbiased_path, "rb") as f:
                st.download_button("⬇️ Download Unbiased Dataset", f, file_name="unbiased_dataset.csv")
    else:
        st.warning("⚠️ Run bias_auditor.ipynb first to generate results.pkl")
