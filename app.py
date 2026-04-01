import streamlit as st
import pandas as pd
import joblib
import json
import tensorflow as tf

st.set_page_config(page_title="AI Project ML & NN", layout="wide")
st.title("AI Project: Machine Learning & Neural Network")

# ================= PATHS (อ้างจาก root) =================
ML_MODEL_PATH = "ML/models/ml_pipe.joblib"
ML_DATA_PATH  = "ML/data/full_data.csv"

NN_MODEL_PATH  = "NN/models/nn_fraud.keras"
NN_SCALER_PATH = "NN/models/scaler.joblib"
NN_META_PATH   = "NN/models/meta.json"

# ================= TOP TABS =================
tab_ml_exp, tab_ml_test, tab_nn_exp, tab_nn_test = st.tabs([
    "ML Explanation",
    "ML Test",
    "NN Explanation",
    "NN Test"
])

# ================= ML EXPLANATION =================
with tab_ml_exp:
    st.header("Machine Learning Model Explanation")

    st.subheader("1) Dataset Description")
    st.write("""
Dataset ที่ใช้คือ Stroke Prediction Dataset จาก Kaggle:
https://www.kaggle.com/code/mahendrasinghrajpoot/brain-stroke-dataset/input

เป้าหมายคือทำนายว่า บุคคลมีโอกาสเป็นโรคหลอดเลือดสมอง (stroke) หรือไม่

Target:
- stroke (0 = ไม่เป็น, 1 = เป็น)

ข้อมูลประกอบด้วย:
- เพศ, อายุ, โรคประจำตัว, ระดับน้ำตาล, BMI ฯลฯ
""")

    st.subheader("2) Data Preparation")
    st.write("""
ขั้นตอนการเตรียมข้อมูล:
1. ตรวจสอบ Missing Value
2. เติมค่า Missing:
   - ตัวเลข → median
   - categorical → mode
3. แปลง categorical → One-Hot Encoding
4. ปรับ scale ข้อมูลด้วย StandardScaler
""")

    st.subheader("3) Model & Theory")
    st.write("""
ใช้ Ensemble Learning แบบ Soft Voting ซึ่งรวมหลายโมเดลเพื่อลด bias และ variance

โมเดลที่ใช้:

- Random Forest
  - เป็น ensemble ของ decision tree
  - ลด overfitting ด้วยการสุ่ม feature และ data
  - เหมาะกับข้อมูลที่มีความสัมพันธ์ไม่เชิงเส้น

- Gradient Boosting
  - เรียนรู้จาก error ของ model ก่อนหน้า
  - เหมาะกับการจับ pattern ที่ซับซ้อน

- Logistic Regression
  - เป็น linear model ที่ตีความง่าย
  - ใช้เป็น baseline ที่ดี

เหตุผลที่ใช้ Ensemble:
- เพิ่มความแม่นยำ
- ลด overfitting
- รวมข้อดีของหลายโมเดล
""")

    st.subheader("4) Model Development Pipeline")
    st.write("""
ขั้นตอนการพัฒนาโมเดล:

1. Load dataset
2. Data cleaning & preprocessing
3. แบ่ง train / test
4. Train model (RF, GB, LR)
5. รวมโมเดลด้วย Soft Voting
6. Evaluate model
7. Save model เป็นไฟล์ (.joblib)
8. Deploy ผ่าน Streamlit
""")

    st.subheader("5) Evaluation")
    st.write("""
ใช้ metrics:
- Accuracy
- F1-score
- Confusion Matrix

เนื่องจากข้อมูลมี class imbalance จึงให้ความสำคัญกับ F1-score
""")

    st.subheader("6) System Integration")
    st.write("""
โมเดลถูกนำมาใช้ในระบบดังนี้:
- โหลดโมเดลด้วย joblib
- ผู้ใช้กรอกข้อมูลผ่านหน้าเว็บ
- ระบบ preprocess ข้อมูล
- ส่งเข้าโมเดลเพื่อ predict
- แสดงผลลัพธ์ probability และ class
""")

# ================= ML TEST =================
with tab_ml_test:
    st.header("🧠 Stroke Prediction (ML Test Only)")

    @st.cache_resource
    def load_ml_model():
        return joblib.load(ML_MODEL_PATH)

    @st.cache_data
    def load_ml_data():
        return pd.read_csv(ML_DATA_PATH)

    try:
        df = load_ml_data()
        model = load_ml_model()
    except Exception as e:
        st.error("โหลด ML ไม่สำเร็จ (เช็ค path ไฟล์ data/models หรือยังไม่ได้ train)")
        st.code(str(e))
        st.stop()

    target_col = "stroke"
    if target_col not in df.columns:
        st.error("ไม่พบคอลัมน์ 'stroke' ใน ML/data/full_data.csv")
        st.stop()

    X_cols = [c for c in df.columns if c != target_col]
    cat_cols = df[X_cols].select_dtypes(include=["object"]).columns.tolist()

    st.subheader("กรอกข้อมูลเพื่อทำนาย")
    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(X_cols):
        with cols[i % 3]:
            if col in cat_cols:
                options = sorted(df[col].dropna().unique().tolist())
                input_data[col] = st.selectbox(col, options, key=f"ml_{col}")
            else:
                default = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                input_data[col] = st.number_input(col, value=default, key=f"ml_{col}")

    input_df = pd.DataFrame([input_data])

    if st.button("Predict (ML Ensemble)"):
        prob = float(model.predict_proba(input_df)[0][1])
        pred = int(prob >= 0.5)
        st.write({"stroke_pred": pred, "probability": prob})

# ================= NN EXPLANATION =================
with tab_nn_exp:
    st.header("Neural Network Model Explanation")

    st.subheader("1) Dataset Description")
    st.write("""
Dataset: Credit Card Fraud Detection จาก Kaggle:
https://www.kaggle.com/code/basmalaawad/credit-card-fraud-dataset/input

เป้าหมาย:
- ตรวจจับธุรกรรมที่เป็น fraud

Target:
- Class (0 = ปกติ, 1 = Fraud)

Features:
- Time, Amount
- V1 - V28 (ผ่าน PCA เพื่อปกปิดข้อมูลจริง)
""")

    st.subheader("2) Data Preparation")
    st.write("""
ขั้นตอน:
1. แบ่งข้อมูลแบบ Stratified (รักษาสัดส่วน fraud)
2. ใช้ StandardScaler ปรับ scale
3. จัดการ missing/inf values
4. ใช้ class_weight แก้ imbalance
""")

    st.subheader("3) Model & Theory")
    st.write("""
ใช้ Neural Network (Deep Learning)

โครงสร้าง:
- Dense 64 (ReLU)
- Dropout
- Dense 32 (ReLU)
- Dropout
- Output (Sigmoid)

ทฤษฎีที่ใช้:
- ReLU: ช่วยให้เรียนรู้ nonlinear pattern
- Dropout: ลด overfitting
- Sigmoid: แปลงค่าเป็น probability (0-1)

เหตุผลที่เลือก Neural Network:
- สามารถจับ pattern ซับซ้อนของ fraud ได้ดี
- เหมาะกับข้อมูลที่มี feature จำนวนมาก
""")

    st.subheader("4) Model Development Pipeline")
    st.write("""
1. Load dataset
2. Data preprocessing
3. Train/Test split (Stratified)
4. Scale data
5. Train Neural Network
6. Evaluate model
7. Save model (.keras)
8. Deploy ผ่าน Streamlit
""")

    st.subheader("5) Evaluation")
    st.write("""
ใช้ metrics ที่เหมาะกับ imbalance:
- AUC
- Precision
- Recall

เน้น Recall สูงเพื่อไม่พลาด fraud
""")

    st.subheader("6) System Integration")
    st.write("""
- โหลดโมเดลด้วย TensorFlow
- โหลด scaler และ meta data
- รับ input จาก user หรือ CSV
- preprocess → scale
- predict probability
- เปรียบเทียบ threshold
- แสดงผลลัพธ์
""")

# ================= NN TEST =================
with tab_nn_test:
    st.header("💳 Fraud Detection (NN Test Only)")
    st.caption("เว็บนี้ใช้โมเดลที่เทรนไว้แล้ว")

    @st.cache_resource
    def load_nn_artifacts():
        model = tf.keras.models.load_model(NN_MODEL_PATH)
        scaler = joblib.load(NN_SCALER_PATH)
        with open(NN_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, scaler, meta

    try:
        nn_model, scaler, meta = load_nn_artifacts()
        feature_cols = meta["feature_cols"]
    except Exception as e:
        st.error("โหลด NN ไม่สำเร็จ (เช็ค path ไฟล์ models หรือยังไม่ได้ train)")
        st.code(str(e))
        st.stop()

    threshold = st.slider("Threshold (ค่าตัดสินว่าเป็น Fraud)", 0.01, 0.99, 0.50, 0.01)

    subtab_single, subtab_upload = st.tabs(["ทดสอบทีละรายการ", "อัปโหลด CSV ทำนายหลายแถว"])

    with subtab_single:
        st.subheader("ทดสอบทีละรายการ")
        cols = st.columns(3)
        input_data = {}
        for i, c in enumerate(feature_cols):
            with cols[i % 3]:
                input_data[c] = st.number_input(c, value=0.0, key=f"nn_{c}")

        x = pd.DataFrame([input_data], columns=feature_cols)
        x_s = scaler.transform(x)

        if st.button("Predict (Single)"):
            prob = float(nn_model.predict(x_s, verbose=0)[0][0])
            pred = int(prob >= threshold)
            st.write({"fraud_pred": pred, "probability": prob, "threshold": float(threshold)})

    with subtab_upload:
        st.subheader("อัปโหลด CSV แล้วทำนายหลายแถว")
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
            prob = nn_model.predict(x_s, verbose=0).ravel()
            pred = (prob >= threshold).astype(int)

            out = df_up.copy()
            out["fraud_probability"] = prob
            out["fraud_pred"] = pred

            st.success("ทำนายเสร็จแล้ว ✅")
            st.dataframe(out.head(20), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download ผลลัพธ์เป็น CSV",
                data=csv_bytes,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )