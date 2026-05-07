import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

EVAL_THRESHOLD = 0.65
CLASS_LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


def setup_mlflow():
    """Configure MLflow tracking. Use DagsHub if creds available, else local."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    user = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    if uri and "dagshub" in uri and user and pwd:
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
        mlflow.set_tracking_uri(uri)
        try:
            # Quick smoke test — try listing experiments
            mlflow.search_experiments()
            print(f"MLflow tracking: {uri}")
            return
        except Exception as e:
            print(f"DagsHub connection failed ({e}), falling back to local MLflow.")

    # Fallback to local
    mlflow.set_tracking_uri("")
    print("MLflow tracking: local (sqlite or file store)")


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


def check_label_distribution(y, dataset_name="train"):
    """Bonus 5: Canh bao lech lac du lieu neu lop nao < 10%."""
    total = len(y)
    dist = {}
    for label, count in y.value_counts().items():
        pct = count / total * 100
        dist[int(label)] = {"count": int(count), "percent": round(pct, 2)}
        if pct < 10:
            print(f"WARNING [{dataset_name}]: Class {label} ({CLASS_LABELS.get(label, label)}) "
                  f"chiem {pct:.1f}% < 10%. Co nguy co mat can bang du lieu.")
    print(f"Label distribution [{dataset_name}]: "
          + ", ".join(f"{CLASS_LABELS.get(k, k)}={v['percent']:.1f}%" for k, v in sorted(dist.items())))
    return dist


def create_model(params):
    """Bonus 2: Ho tro nhieu thuat toan qua model_type trong params."""
    model_type = params.pop("model_type", "xgboost")
    if model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=params.get("random_state", 42))
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params, random_state=params.get("random_state", 42))
    else:  # xgboost
        return xgb.XGBClassifier(**params, objective='multi:softmax', num_class=3)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi nhan
    train_dist = check_label_distribution(y_train, "train")
    eval_dist  = check_label_distribution(y_eval, "eval")

    X_train = add_features(X_train)
    X_eval  = add_features(X_eval)

    model_type = params.get("model_type", "xgboost")

    setup_mlflow()

    with mlflow.start_run():

        mlflow.log_params(params)

        model = create_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: Confusion matrix + per-class precision/recall
        cm = confusion_matrix(y_eval, preds)
        precision_per_class = precision_score(y_eval, preds, average=None, labels=[0, 1, 2])
        recall_per_class    = recall_score(y_eval, preds, average=None, labels=[0, 1, 2])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for i in range(3):
            mlflow.log_metric(f"precision_class_{i}", precision_per_class[i])
            mlflow.log_metric(f"recall_class_{i}", recall_per_class[i])
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
            "confusion_matrix": cm.tolist(),
            "precision_per_class": {str(i): round(p, 4) for i, p in enumerate(precision_per_class)},
            "recall_per_class":    {str(i): round(r, 4) for i, r in enumerate(recall_per_class)},
            "train_distribution": train_dist,
            "eval_distribution": eval_dist,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f)

        # Bonus 3: Generate report.txt
        with open("outputs/report.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("BAO CAO HIỆU SUAT MO HINH\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model type: {model_type}\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n\n")
            f.write("Per-class metrics:\n")
            f.write(f"{'Class':<10} {'Label':<15} {'Precision':<12} {'Recall':<12}\n")
            f.write("-" * 49 + "\n")
            for i in range(3):
                f.write(f"{i:<10} {CLASS_LABELS[i]:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"{'':<10} {'Pred0':<8} {'Pred1':<8} {'Pred2':<8}\n")
            for i in range(3):
                f.write(f"{'Actual'+str(i):<10} {cm[i][0]:<8} {cm[i][1]:<8} {cm[i][2]:<8}\n")
            f.write("\nLabel Distribution (train):\n")
            for k, v in sorted(train_dist.items()):
                f.write(f"  Class {k} ({CLASS_LABELS.get(k, '?')}): {v['percent']:.1f}%\n")

        print("Report saved to outputs/report.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        model.save_model("models/model.json")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
