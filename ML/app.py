import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/ml_pipe.joblib"
DATA_PATH = "data/full_data.csv"

st.set_page_config(page_title="Stroke Prediction (Test Only)", layout="wide")
st.title("🧠 Stroke Prediction (Test Only)")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data_preview():
    return pd.read_csv(DATA_PATH)

# โหลดเพื่อโชว์ค่าใน dropdown + median
df = load_data_preview()

target_col = "stroke"
X_cols = [c for c in df.columns if c != target_col]
cat_cols = df[X_cols].select_dtypes(include=["object"]).columns.tolist()

# โหลดโมเดลที่เทรนไว้แล้ว
try:
    model = load_model()
except Exception as e:
    st.error("โหลดโมเดลไม่สำเร็จ (ต้องรัน train_ml.py ก่อน)")
    st.code(str(e))
    st.stop()

st.subheader("กรอกข้อมูลเพื่อทำนาย")

input_data = {}
cols = st.columns(3)

for i, col in enumerate(X_cols):
    with cols[i % 3]:
        if col in cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(col, options)
        else:
            default = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
            input_data[col] = st.number_input(col, value=default)

input_df = pd.DataFrame([input_data])

if st.button("Predict (ML Ensemble)"):
    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    st.write({"stroke_pred": pred, "probability": float(prob)})