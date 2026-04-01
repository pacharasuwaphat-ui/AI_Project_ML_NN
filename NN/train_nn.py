import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "nn_fraud.keras")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์ {DATA_PATH} (กรุณาวาง creditcard.csv ไว้ใน NN/data/)")

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("ไฟล์นี้ต้องมีคอลัมน์เป้าหมายชื่อ 'Class' (0=normal, 1=fraud)")

    # เลือก feature ทั้งหมด ยกเว้น Class
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"].astype(int)

    # จัดการ missing ถ้ามี (โดยปกติ dataset นี้ไม่มี แต่กันไว้)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Split แบบ stratify (สำคัญมากสำหรับ fraud ที่ไม่สมดุล)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # class_weight ช่วยแก้ class imbalance
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    if n1 == 0:
        raise ValueError("Train set ไม่มีตัวอย่าง fraud (Class=1) เลย — แปลกมาก ตรวจสอบไฟล์อีกครั้ง")

    class_weight = {0: 1.0, 1: float(n0 / n1)}

    model = build_model(X_train_s.shape[1])

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=3, restore_best_weights=True
    )

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=2048,
        class_weight=class_weight,
        callbacks=[early],
        verbose=1
    )

    # Evaluate (prob + auc)
    prob = model.predict(X_test_s, verbose=0).ravel()
    auc = float(roc_auc_score(y_test, prob))

    # ใช้ threshold 0.5 เป็นค่าเริ่มต้น (เว็บจะให้ปรับได้)
    pred = (prob >= 0.5).astype(int)

    print("\n=== Evaluation (threshold=0.5) ===")
    print("AUC:", auc)
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, digits=4))

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    model.save(MODEL_PATH)

    meta = {
        "data_path": DATA_PATH,
        "feature_cols": feature_cols,
        "class_weight": class_weight,
        "auc": auc,
        "epochs_trained": len(history.history.get("loss", [])),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved scaler:", SCALER_PATH)
    print("Saved model :", MODEL_PATH)
    print("Saved meta  :", META_PATH)


if __name__ == "__main__":
    main()