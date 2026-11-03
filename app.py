import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Risk Alert Classifier", page_icon="🏦", layout="wide")
st.title("🏦 Risk Alert Classifier")
st.write("An educational early-warning dashboard for identifying customers who may need a closer risk review.")
st.warning("For learning and demonstration only. Do not use these predictions as the sole basis for real banking decisions.")

@st.cache_resource
def load_artifact():
    return joblib.load("risk_alert_model.joblib")

try:
    artifact = load_artifact()
except Exception:
    st.error("Model file not found. Run all cells in Risk_Alert_Classifier_Project.ipynb first and keep risk_alert_model.joblib beside app.py.")
    st.stop()

st.caption(f"Model loaded: {artifact.get('model_name', 'Risk classifier')}")
st.markdown("### 📂 Upload customer data")
st.write("Upload a CSV with the same customer feature columns as the project dataset. The `risk_status` target column is optional and is not used as an input feature.")
upload = st.file_uploader("Choose customer CSV", type=["csv"])

if upload is not None:
    try:
        raw = pd.read_csv(upload)
        st.subheader("Preview")
        st.dataframe(raw.head(10), use_container_width=True)
        work = raw.copy()
        work = work.drop(columns=[c for c in ["risk_status", "customer_id"] if c in work.columns], errors="ignore")
        if "last_transaction_date" in work.columns:
            dates = pd.to_datetime(work["last_transaction_date"], errors="coerce")
            work["last_transaction_month"] = dates.dt.month
            work["last_transaction_day"] = dates.dt.day
            work = work.drop(columns=["last_transaction_date"])
        required = artifact["numeric_cols"] + artifact["categorical_cols"]
        missing_columns = [c for c in required if c not in work.columns]
        if missing_columns:
            st.error("These required feature columns are missing: " + ", ".join(missing_columns))
            st.stop()

        num = work[artifact["numeric_cols"]].apply(pd.to_numeric, errors="coerce")
        num_scaled = artifact["numeric_scaler"].transform(num)
        num_imputed = artifact["knn_imputer"].transform(num_scaled)
        num_df = pd.DataFrame(num_imputed, columns=artifact["numeric_cols"], index=work.index)

        cat = work[artifact["categorical_cols"]].copy()
        for col in artifact["categorical_cols"]:
            cat[col] = cat[col].fillna(artifact["category_fill_values"].get(col, "Unknown")).astype(str)
        prepared = pd.concat([num_df, cat], axis=1)
        prepared = pd.get_dummies(prepared, columns=artifact["categorical_cols"], dtype=int)
        prepared = prepared.reindex(columns=artifact["encoded_columns"], fill_value=0)

        model = artifact["model"]
        prediction = model.predict(prepared)
        probability = model.predict_proba(prepared)[:, 1]
        results = raw.copy()
        results["predicted_risk_status"] = prediction.astype(int)
        results["predicted_risk_label"] = np.where(prediction == 1, "High Risk", "Low Risk")
        results["high_risk_probability"] = probability.round(4)

        high_count = int((prediction == 1).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers scored", f"{len(results):,}")
        c2.metric("Flagged high risk", f"{high_count:,}")
        c3.metric("High-risk share", f"{(high_count / len(results) * 100 if len(results) else 0):.1f}%")

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.subheader("Predicted risk distribution")
        counts = results["predicted_risk_label"].value_counts().reindex(["Low Risk", "High Risk"], fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Predicted Customer Risk Categories")
        ax.set_xlabel("Predicted category")
        ax.set_ylabel("Number of customers")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        st.info("Interpretation: high-risk predictions are alerts for further review, not proof of fraud or default.")

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions as CSV", data=csv, file_name="risk_alert_predictions.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"The file could not be processed. Check that it matches the project dataset columns. Details: {exc}")
else:
    st.info("Upload the provided Risk_Alert_Classifier_Dataset_4600.csv to test the dashboard.")
