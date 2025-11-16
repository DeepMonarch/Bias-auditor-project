import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import io

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="AI Bias Auditor", layout="wide")
st.title("🧠 AI Bias Auditor — Detect & Reduce Dataset Bias (User Input Only)")

st.write("""
Upload your dataset, select a **target column** and a **sensitive feature**, and the tool will:

- Train a model  
- Measure fairness  
- Attempt debiasing  
- Produce an unbiased dataset  
""")

# ----------------------------
# HELPER
# ----------------------------
def safe_read_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

# ----------------------------
# UPLOAD DATASET
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])

if uploaded_file:
    try:
        df = safe_read_csv(uploaded_file)

        df = df.replace(["∞", "inf", "-inf"], np.nan)
        df = df.replace([np.inf, -np.inf], np.nan)

        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    columns = list(df.columns)

    # ----------------------------
    # SELECT COLUMNS
    # ----------------------------
    target_col = st.selectbox("🎯 Target Column (label)", columns)
    sensitive_col = st.selectbox("⚖ Sensitive Column (e.g., gender, age, region)", columns)

    if sensitive_col == target_col:
        st.error("❌ Sensitive column cannot be the same as the target column.")
        st.stop()

    # ----------------------------
    # RUN BIAS AUDITOR
    # ----------------------------
    if st.button("🚀 Run Bias Auditor"):
        try:
            st.info("Processing... Please wait...")

            working = df.copy()

            # Normalize text
            for col in working.columns:
                if working[col].dtype == object:
                    working[col] = working[col].astype(str).str.strip()

            working = working.replace({"": "missing", "nan": "missing", None: "missing"})

            # Save sensitive original
            working[sensitive_col] = working[sensitive_col].astype(str).fillna("missing")
            sensitive_original = working[sensitive_col].copy()

            # Target encoding
            working[target_col] = working[target_col].astype(str).fillna("missing")
            target_cat = pd.Categorical(working[target_col])
            y_all = pd.Series(target_cat.codes, index=working.index)
            target_mapping = dict(enumerate(target_cat.categories))

            # ----------------------------
            # ENCODE ALL CATEGORICAL FEATURES (EXCEPT TARGET)
            # ----------------------------
            cat_cols = [c for c in working.columns if working[c].dtype == object and c != target_col]

            if len(cat_cols) > 0:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                working[cat_cols] = encoder.fit_transform(working[cat_cols])
                working[cat_cols] = working[cat_cols].astype(float)

            # ----------------------------
            # IMPUTE MISSING VALUES
            # ----------------------------
            imputer = SimpleImputer(strategy="median")
            working = pd.DataFrame(imputer.fit_transform(working), columns=working.columns)

            # ----------------------------
            # SPLIT
            # ----------------------------
            X = working.drop(columns=[target_col])
            y = y_all

            if len(np.unique(y)) < 2:
                st.error("❌ Target column must have at least 2 classes.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            sensitive_test = sensitive_original.loc[X_test.index]

            # ----------------------------
            # MODEL BEFORE DEBIASING
            # ----------------------------
            model = LogisticRegression(max_iter=1000)

            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc_before = accuracy_score(y_test, y_pred)
            st.metric("Accuracy (Before Debiasing)", f"{acc_before:.3f}")

            metric_before = MetricFrame(
                metrics=selection_rate,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sensitive_test
            )

            dpd_before = demographic_parity_difference(
                y_test, y_pred, sensitive_features=sensitive_test
            )

            st.write("### ⚖ Bias Before Debiasing")
            fig1, ax1 = plt.subplots()
            metric_before.by_group.plot(kind="bar", ax=ax1)
            ax1.set_title(f"Selection Rate by {sensitive_col} (Before)")
            st.pyplot(fig1)
            st.metric("Demographic Parity Difference (Before)", f"{dpd_before:.3f}")

            # ----------------------------
            # APPLY SMOTE
            # ----------------------------
            if y_train.value_counts().min() >= 2:
                sm = SMOTE(random_state=42)
                X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            else:
                st.warning("⚠ SMOTE skipped (class too small).")
                X_train_res, y_train_res = X_train, y_train

            model.fit(X_train_res, y_train_res)
            y_pred_after = model.predict(X_test)

            dpd_after = demographic_parity_difference(
                y_test, y_pred_after, sensitive_features=sensitive_test
            )

            st.metric("Accuracy (After Debiasing)", f"{accuracy_score(y_test, y_pred_after):.3f}")
            st.metric("Demographic Parity Difference (After)", f"{dpd_after:.3f}")

            # ----------------------------
            # EXPORT
            # ----------------------------
            unbiased_df = X_test.copy()
            unbiased_df[target_col + "_predicted"] = [
                target_mapping.get(int(v), str(v)) for v in y_pred_after
            ]
            unbiased_df[sensitive_col + "_original"] = sensitive_original.loc[unbiased_df.index]

            buffer = io.BytesIO()
            unbiased_df.to_csv(buffer, index=False)

            st.download_button(
                "⬇ Download Unbiased Dataset (CSV)",
                data=buffer.getvalue(),
                file_name="unbiased_dataset.csv",
                mime="text/csv"
            )

            st.success("✔ Bias detection completed!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload a CSV file to start.")
