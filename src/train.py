import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.69


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


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # Doc du lieu
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Feature engineering
    X_train = add_features(X_train)
    X_eval  = add_features(X_eval)

    with mlflow.start_run():

        mlflow.log_params(params)

        # XGBoost classifier
        model = xgb.XGBClassifier(**params, objective='multi:softmax', num_class=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:  # load best params
        params = yaml.safe_load(f)
    train(params)

