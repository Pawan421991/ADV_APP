
import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Advertising Sales Prediction", page_icon="📈", layout="centered")
st.title("📈 Advertising Sales Prediction App")
st.write("Upload a CSV with features (TV, Radio, Newspaper) to predict **Sales**.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.joblib")
COLS_PATH  = os.path.join(BASE_DIR, "feature_columns.json")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, "r") as f:
        cols = json.load(f)
    return model, cols

model, feature_columns = load_artifacts()

st.subheader("1) Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Drop common index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X = df[feature_columns]
        preds = model.predict(X)

        out = df.copy()
        out["Predicted_Sales"] = preds

        st.success("✅ Prediction completed!")
        st.dataframe(out)

        st.download_button(
            "📥 Download predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

st.subheader("2) Single prediction (manual input)")
cols = st.columns(len(feature_columns))
manual = {}
for i, c in enumerate(feature_columns):
    manual[c] = cols[i].number_input(c, value=0.0)

if st.button("Predict Sales"):
    X_one = pd.DataFrame([manual])[feature_columns]
    pred_one = float(model.predict(X_one)[0])
    st.info(f"✅ Predicted Sales: **{pred_one:.2f}**")
