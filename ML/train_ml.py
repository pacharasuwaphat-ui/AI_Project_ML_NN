import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/full_data.csv"
MODEL_PATH = "models/ml_pipe.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "stroke"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    rf = RandomForestClassifier(n_estimators=250, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=3000)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft"
    )

    ml_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", ensemble),
    ])

    # split (ไม่จำเป็นต่อการเซฟ แต่ทำไว้เป็นมาตรฐาน)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ml_pipe.fit(X_train, y_train)

    # เซฟ pipeline ทั้งก้อน (รวม preprocess + model)
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(ml_pipe, MODEL_PATH)

    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()