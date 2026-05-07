from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import xgboost as xgb
import pandas as pd
import os

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
S3_MODEL_JSON_KEY = "models/latest/model.json"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")
MODEL_JSON_PATH = os.path.expanduser("~/models/model.json")


def download_model():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    s3.download_file(S3_BUCKET, S3_MODEL_JSON_KEY, MODEL_JSON_PATH)
    print("Model downloaded from S3.")


def load_model():
    # Prefer native JSON (cross-version compatible), fallback to pickle
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_JSON_PATH)
        print("Loaded model from native JSON.")
        return model
    except Exception:
        print("JSON load failed, trying pickle...")
        return joblib.load(MODEL_PATH)


download_model()
model = load_model()


FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def add_features(df):
    df = df.copy()
    df['alcohol_sulphates']     = df['alcohol'] * df['sulphates']
    df['volatile_pH']           = df['volatile acidity'] * df['pH']
    df['free_total_so2_ratio']  = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1)
    df['citric_pH']             = df['citric acid'] * df['pH']
    df['alcohol_density']       = df['alcohol'] * df['density']
    df['sugar_ratio']           = df['residual sugar'] / (df['density'] * 100)
    df['acid_sum']              = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['sulphate_alcohol']      = df['sulphates'] / (df['alcohol'] + 0.1)
    return df


class PredictRequest(BaseModel):
    features: List[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    X = pd.DataFrame([req.features], columns=FEATURE_NAMES)
    X = add_features(X)
    pred = model.predict(X)[0]

    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
