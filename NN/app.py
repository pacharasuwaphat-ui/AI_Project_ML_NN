import json
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

MODEL_PATH = "models/nn_fraud.keras"
SCALER_PATH = "models/scaler.joblib"
META_PATH = "models/meta.json"

st.set_page_config(page_title="Fraud Detection (NN Test)", layout="wide")
st.title("💳 Credit Card Fraud Detection (Neural Network)")
st.caption("เว็บนี้ใช้โมเดลที่เทรนไว้แล้ว (Test Only)")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta

try:
    model, scaler, meta = load_artifacts()
    feature_cols = meta["feature_cols"]
except Exception as e:
    st.error("โหลดโมเดลไม่สำเร็จ (ต้องรัน train_nn.py ให้สร้างไฟล์ในโฟลเดอร์ models ก่อน)")
    st.code(str(e))
    st.stop()

threshold = st.slider("Threshold (ค่าตัดสินว่าเป็น Fraud)", 0.01, 0.99, 0.50, 0.01)

tab1, tab2 = st.tabs(["ทดสอบทีละรายการ", "อัปโหลด CSV ทำนายหลายแถว"])

with tab1:
    st.subheader("ทดสอบทีละรายการ (กรอกค่าเอง)")
    st.info("Dataset นี้มีคอลัมน์เยอะ (Time, V1..V28, Amount) — ถ้าไม่อยากกรอกทีละช่อง แนะนำแท็บอัปโหลด CSV")

    cols = st.columns(3)
    input_data = {}
    for i, c in enumerate(feature_cols):
        with cols[i % 3]:
            input_data[c] = st.number_input(c, value=0.0)

    x = pd.DataFrame([input_data], columns=feature_cols)
    x_s = scaler.transform(x)

    if st.button("Predict (Single)"):
        prob = float(model.predict(x_s, verbose=0)[0][0])
        pred = int(prob >= threshold)
        st.write({"fraud_pred": pred, "probability": prob, "threshold": float(threshold)})

with tab2:
    st.subheader("อัปโหลดไฟล์ CSV แล้วทำนายหลายแถว")
    st.write("ไฟล์ต้องมีคอลัมน์ feature เหมือนตอนเทรน (อย่างน้อยต้องมีทุกคอลัมน์ใน meta.json)")
    upload = st.file_uploader("Upload CSV", type=["csv"])

    if upload is not None:
        df_up = pd.read_csv(upload)

        missing = [c for c in feature_cols if c not in df_up.columns]
        if missing:
            st.error("ไฟล์ของคุณขาดคอลัมน์เหล่านี้:")
            st.write(missing)
            st.stop()

        x = df_up[feature_cols].copy()
        x = x.replace([float("inf"), float("-inf")], pd.NA)
        x = x.fillna(x.median(numeric_only=True))

        x_s = scaler.transform(x)
        prob = model.predict(x_s, verbose=0).ravel()
        pred = (prob >= threshold).astype(int)

        out = df_up.copy()
        out["fraud_probability"] = prob
        out["fraud_pred"] = pred

        st.success("ทำนายเสร็จแล้ว ✅")
        st.dataframe(out.head(20), use_container_width=True)

        # download result
        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download ผลลัพธ์เป็น CSV",
            data=csv_bytes,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )