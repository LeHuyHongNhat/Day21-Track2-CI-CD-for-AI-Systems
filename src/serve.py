from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    # TODO 1-4: Tai file model.pkl tu S3 ve may khi server khoi dong
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print("Model downloaded from S3.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # TODO 6: Kiem tra so luong dac trung
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # TODO 7: Goi model.predict
    pred = model.predict([req.features])[0]

    # TODO 8: Tra ve dict chua "prediction" va "label"
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
