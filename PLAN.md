# PLAN.md — MLOps Lab: CI/CD for AI Systems

## Mục tiêu

Xây dựng pipeline CI/CD hoàn chỉnh cho bài toán phân loại chất lượng rượu vang (Wine Quality) với 3 bước:

1. **Bước 1** — Thực nghiệm cục bộ, theo dõi thí nghiệm bằng MLflow, chọn siêu tham số tốt nhất.
2. **Bước 2** — Pipeline CI/CD trên GitHub Actions: test → train → eval gate → deploy lên Cloud VM.
3. **Bước 3** — Huấn luyện liên tục: thêm dữ liệu mới, pipeline tự động chạy lại.

Cloud provider: **AWS**.

---

## Bước 0: Chuẩn bị môi trường

### 0.1 Yêu cầu phần mềm
- Python 3.10+
- Git + tài khoản GitHub (tạo repo public rỗng)
- Tài khoản AWS (gói free tier đủ dùng)
- AWS CLI (`aws --version`)
- Conda (môi trường `aithucchien`)

### 0.2 Cấu hình AWS CLI
```bash
aws configure
# Nhập AWS Access Key ID, AWS Secret Access Key, region (vd: us-east-1)
```

### 0.3 Clone và cài đặt
```bash
git clone <repo-url> && cd <repo>
conda activate aithucchien
pip install -r requirements.txt
```

### 0.4 Tạo dữ liệu
```bash
python generate_data.py
# Kết quả: data/train_phase1.csv (2998), data/eval.csv (500), data/train_phase2.csv (2998)
ls data/
```

---

## Bước 1: Thực nghiệm cục bộ và MLflow tracking

### 1.1 Cấu hình MLflow
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

### 1.2 Hoàn thiện `src/train.py`

Hoàn thành 10 TODO trong file. Dưới đây là code cho từng TODO:

**Đọc dữ liệu (TODO 1):**
```python
df_train = pd.read_csv(data_path)
df_eval  = pd.read_csv(eval_path)
```

**Tách đặc trưng & nhãn (TODO 2):**
```python
X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]
X_eval  = df_eval.drop(columns=["target"])
y_eval  = df_eval["target"]
```

**Bắt đầu MLflow run (TODO 3):**
```python
with mlflow.start_run():
```

**Ghi nhận siêu tham số (TODO 3 - trong with block):**
```python
    mlflow.log_params(params)
```

**Huấn luyện mô hình (TODO 4):**
```python
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
```

**Dự đoán & tính chỉ số (TODO 5):**
```python
    preds = model.predict(X_eval)
    acc   = accuracy_score(y_eval, preds)
    f1    = f1_score(y_eval, preds, average="weighted")
```

**Ghi nhận chỉ số vào MLflow (TODO 6):**
```python
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
```

**In kết quả (TODO 7):**
```python
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
```

**Lưu metrics.json (TODO 8):**
```python
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)
```

**Lưu model.pkl (TODO 9):**
```python
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
```

**Trả về accuracy (TODO 10):**
```python
    return acc
```

### 1.3 Chạy ít nhất 3 thí nghiệm

Thay đổi `params.yaml` giữa các lần chạy. Gợi ý:

```bash
# Lần 1: n_estimators=100, max_depth=5, min_samples_split=2
python src/train.py

# Lần 2: n_estimators=50, max_depth=3, min_samples_split=2
# (sửa params.yaml rồi chạy lại)
python src/train.py

# Lần 3: n_estimators=200, max_depth=10, min_samples_split=5
python src/train.py
```

### 1.4 Phân tích kết quả trong MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Mở http://localhost:5000
```

So sánh các lần chạy, chọn bộ siêu tham số có accuracy cao nhất, cập nhật vào `params.yaml`.

### 1.5 Đầu ra Bước 1
- [ ] `src/train.py` hoàn thiện, chạy không lỗi
- [ ] `outputs/metrics.json` tồn tại (có `accuracy` và `f1_score`)
- [ ] `models/model.pkl` tồn tại
- [ ] MLflow UI hiển thị ≥ 3 lần chạy với siêu tham số khác nhau
- [ ] `params.yaml` chứa bộ siêu tham số tốt nhất
- [ ] Chụp màn hình MLflow UI

---

## Bước 2: Pipeline CI/CD tự động

### 2.1 Tạo S3 Bucket

#### Kiểm tra AWS CLI

Trước khi tạo bucket, xác nhận AWS CLI đã được cài đặt và cấu hình:

```bash
# Kiểm tra AWS CLI đã cài đặt
aws --version
# → aws-cli/2.x.x ...

# Kiểm tra credentials đã cấu hình
aws sts get-caller-identity
# → Trả về UserId, Account, Arn của IAM user hiện tại

# Nếu chưa cấu hình, chạy lệnh sau và nhập Access Key + Secret Key + region:
aws configure
# AWS Access Key ID: <your-access-key>
# AWS Secret Access Key: <your-secret-key>
# Default region name: us-east-1
# Default output format: json (hoặc để trống)
```

Nếu bạn chưa có Access Key, vào AWS Console → IAM → Users → chọn user → Security credentials → Create access key. Lưu ngay Access Key ID và Secret Access Key (chỉ hiện một lần).

#### Chọn tên bucket

Tên S3 bucket phải **duy nhất trên toàn cầu** (không trùng với bất kỳ bucket nào của bất kỳ tài khoản AWS nào). Quy tắc đặt tên:

- Độ dài: 3-63 ký tự
- Chỉ gồm chữ thường, số, dấu gạch ngang (`-`) và dấu chấm (`.`)
- Phải bắt đầu và kết thúc bằng chữ thường hoặc số
- Không được dùng IP address (vd: 192.168.1.1)

Gợi ý đặt tên: `mlops-lab-<your-name>-<random-suffix>` để tránh trùng.

```bash
# Ví dụ
BUCKET="mlops-lab-nhat-2026"
```

#### Chọn region

AWS region quyết định vị trí vật lý của dữ liệu. Chọn region gần bạn để giảm độ trễ:

| Region Code | Tên | Vị trí |
|---|---|---|
| `us-east-1` | US East (N. Virginia) | Mỹ - phổ biến nhất, free tier |
| `us-west-2` | US West (Oregon) | Mỹ - bờ Tây |
| `ap-southeast-1` | Asia Pacific (Singapore) | Gần Việt Nam |
| `ap-northeast-1` | Asia Pacific (Tokyo) | Nhật Bản |

```bash
AWS_REGION="us-east-1"
```

#### Tạo bucket

```bash
aws s3 mb s3://$BUCKET --region $AWS_REGION
# → make_bucket: <bucket-name>
```

Nếu bucket chạy EC2 cùng region, dữ liệu truyền giữa EC2 và S3 trong cùng region là miễn phí.

#### Xác nhận bucket đã tạo

```bash
# Liệt kê tất cả bucket
aws s3 ls

# Kiểm tra bucket cụ thể
aws s3 ls s3://$BUCKET/

# Xem chi tiết bucket (region, creation date)
aws s3api get-bucket-location --bucket $BUCKET
```

#### Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `BucketAlreadyExists` | Tên bucket đã bị người khác dùng | Đổi tên khác (thêm số/hậu tố ngẫu nhiên) |
| `InvalidBucketName` | Tên không đúng quy tắc (chữ hoa, ký tự đặc biệt...) | Kiểm tra lại quy tắc đặt tên ở trên |
| `Access Denied` | IAM user không có quyền `s3:CreateBucket` | Gắn policy `AmazonS3FullAccess` cho user |
| `Could not connect to endpoint URL` | Region sai hoặc mạng không kết nối được | Kiểm tra region code, kiểm tra internet |

#### Ghi nhớ

```bash
# Lưu tên bucket và region để dùng ở các bước sau
echo "BUCKET=$BUCKET"
echo "AWS_REGION=$AWS_REGION"
```

### 2.2 Tạo IAM User và Access Key

Tạo IAM User với quyền S3 (chỉ trên bucket của bạn):

```bash
# Tạo IAM User
aws iam create-user --user-name mlops-lab-user

# Gắn policy AmazonS3FullAccess (hoặc tự tạo policy giới hạn 1 bucket)
aws iam attach-user-policy \
  --user-name mlops-lab-user \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Tạo Access Key
aws iam create-access-key --user-name mlops-lab-user
# Lưu lại: AccessKeyId và SecretAccessKey (chỉ hiện MỘT LẦN)
```

Lưu credentials vào file JSON để DVC dùng cục bộ (file này không commit):
```bash
cat > sa-key.json <<'EOF'
{
  "aws_access_key_id": "<YOUR_ACCESS_KEY_ID>",
  "aws_secret_access_key": "<YOUR_SECRET_ACCESS_KEY>"
}
EOF
```

`.gitignore` đã có `sa-key.json` — không cần thêm.

### 2.3 Cài đặt DVC với S3 remote

**Lưu ý:** `requirements.txt` cần có `dvc[s3]` thay vì `dvc[gs]`. Cập nhật `requirements.txt`:
```
dvc[s3]==3.50.1
```
Sau đó cài lại:
```bash
pip install -r requirements.txt
```

```bash
dvc init
dvc remote add -d myremote s3://$BUCKET/dvc

# AWS DVC dùng AWS CLI profile hoặc biến môi trường, không cần credentialpath
# Nhưng nếu dùng file JSON riêng, set biến môi trường trước khi chạy dvc:
export AWS_ACCESS_KEY_ID=$(python -c "import json; print(json.load(open('sa-key.json'))['aws_access_key_id'])")
export AWS_SECRET_ACCESS_KEY=$(python -c "import json; print(json.load(open('sa-key.json'))['aws_secret_access_key'])")

dvc add data/train_phase1.csv
dvc add data/eval.csv
dvc add data/train_phase2.csv

git add data/*.csv.dvc .gitignore .dvc/config requirements.txt
git commit -m "feat: track datasets with DVC on S3"

dvc push   # Đẩy dữ liệu lên S3
```

Kiểm tra trên AWS S3 Console: các file CSV xuất hiện trong bucket dưới prefix `dvc/`.

### 2.4 Tạo EC2 Instance

```bash
# Tạo security group cho phép SSH (22) và API (8000)
aws ec2 create-security-group \
  --group-name mlops-serve-sg \
  --description "MLOps serve: SSH + API"

aws ec2 authorize-security-group-ingress \
  --group-name mlops-serve-sg \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-name mlops-serve-sg \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Tạo key pair cho SSH
aws ec2 create-key-pair \
  --key-name mlops-serve-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/mlops-serve-key.pem
chmod 400 ~/.ssh/mlops-serve-key.pem

# Lấy ID của Amazon Linux 2 AMI (free tier eligible)
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

# Tạo EC2 instance (t2.micro = free tier)
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t2.micro \
  --key-name mlops-serve-key \
  --security-groups mlops-serve-sg \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=mlops-serve}]'

# Lấy IP công khai
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=mlops-serve" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
# Lưu IP này lại
```

### 2.5 Cấu hình EC2 (thủ công, một lần)

```bash
# Amazon Linux 2 dùng ec2-user
ssh -i ~/.ssh/mlops-serve-key.pem ec2-user@<EC2_PUBLIC_IP>
```

Trong EC2:
```bash
# Amazon Linux 2 dùng yum
sudo yum update -y && sudo yum install -y python3-pip
pip3 install fastapi uvicorn scikit-learn joblib boto3
mkdir -p ~/models ~/src
exit
```

Copy credentials và file lên EC2:
```bash
scp -i ~/.ssh/mlops-serve-key.pem sa-key.json ec2-user@<EC2_PUBLIC_IP>:~/
scp -i ~/.ssh/mlops-serve-key.pem src/serve.py ec2-user@<EC2_PUBLIC_IP>:~/src/
```

### 2.6 Hoàn thiện `src/serve.py`

Cập nhật `src/serve.py` dùng `boto3` thay vì `google.cloud.storage`:

**Import thay đổi:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os
```

**Biến môi trường:**
```python
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")
```

**TODO 1-4: Hàm download_model()**
```python
def download_model():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print("Model downloaded from S3.")
```

**TODO 5: Health endpoint**
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

**TODO 6-8: Predict endpoint**
```python
@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = model.predict([req.features])[0]
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}
```

### 2.7 Cấu hình systemd service trên EC2

```bash
ssh -i ~/.ssh/mlops-serve-key.pem ec2-user@<EC2_PUBLIC_IP>
```

```bash
# Lấy credentials từ sa-key.json
ACCESS_KEY=$(python3 -c "import json; print(json.load(open('/home/ec2-user/sa-key.json'))['aws_access_key_id'])")
SECRET_KEY=$(python3 -c "import json; print(json.load(open('/home/ec2-user/sa-key.json'))['aws_secret_access_key'])")

sudo tee /etc/systemd/system/mlops-serve.service > /dev/null <<EOF
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment="S3_BUCKET=<YOUR_BUCKET_NAME>"
Environment="AWS_ACCESS_KEY_ID=$ACCESS_KEY"
Environment="AWS_SECRET_ACCESS_KEY=$SECRET_KEY"
Environment="AWS_DEFAULT_REGION=us-east-1"
ExecStart=/usr/bin/python3 /home/ec2-user/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
```

Thay `<YOUR_BUCKET_NAME>` bằng tên bucket thật. Chưa start service — model chưa có trên S3.

### 2.8 Tạo SSH key cho GitHub Actions deploy

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"

# Thêm public key vào EC2
ssh -i ~/.ssh/mlops-serve-key.pem ec2-user@<EC2_PUBLIC_IP> \
  "echo '$(cat ~/.ssh/mlops_deploy.pub)' >> ~/.ssh/authorized_keys"
```

### 2.9 Thêm GitHub Secrets

Vào repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Giá trị |
|---|---|
| `CLOUD_CREDENTIALS` | Nội dung file `sa-key.json` (JSON chứa `aws_access_key_id` và `aws_secret_access_key`) |
| `CLOUD_BUCKET` | Tên S3 bucket |
| `VM_HOST` | IP công khai của EC2 instance |
| `VM_USER` | `ec2-user` |
| `VM_SSH_KEY` | Toàn bộ nội dung `~/.ssh/mlops_deploy` (private key) |

### 2.10 Hoàn thiện `tests/test_train.py`

**TODO 1-5: Hàm _make_temp_data()**
```python
def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    train_path = str(tmp_path / "train.csv")
    eval_path  = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path,  index=False)
    return train_path, eval_path
```

**TODO 6-7: test_train_returns_float**
```python
def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
```

**TODO 8: test_metrics_file_created**
```python
def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
```

**TODO 9: test_model_file_created**
```python
def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)
    assert os.path.exists("models/model.pkl")
```

Chạy thử:
```bash
pytest tests/ -v
```

### 2.11 Hoàn thiện `.github/workflows/mlops.yml`

**TODO 1 — Job test, Run tests:**
```yaml
        run: pytest tests/ -v
```

**TODO 2 — Job train, Authenticate to Cloud Storage (AWS):**
```yaml
        run: |
          echo '${{ secrets.CLOUD_CREDENTIALS }}' > /tmp/creds.json
          ACCESS_KEY=$(python -c "import json; print(json.load(open('/tmp/creds.json'))['aws_access_key_id'])")
          SECRET_KEY=$(python -c "import json; print(json.load(open('/tmp/creds.json'))['aws_secret_access_key'])")
          echo "AWS_ACCESS_KEY_ID=$ACCESS_KEY" >> $GITHUB_ENV
          echo "AWS_SECRET_ACCESS_KEY=$SECRET_KEY" >> $GITHUB_ENV
          echo "AWS_DEFAULT_REGION=us-east-1" >> $GITHUB_ENV
```

**TODO 3 — Job train, Pull data with DVC:**
```yaml
        run: dvc pull data/train_phase1.csv.dvc data/eval.csv.dvc
```

**TODO 4 — Job train, Read metrics:**
```yaml
        run: |
          ACC=$(python -c "import json; d=json.load(open('outputs/metrics.json')); print(d['accuracy'])")
          echo "accuracy=$ACC" >> $GITHUB_OUTPUT
```

**TODO 5 — Job train, Upload model to S3:**
```yaml
        run: |
          python - <<'EOF'
          import boto3
          import os
          s3 = boto3.client("s3")
          s3.upload_file("models/model.pkl", os.environ["CLOUD_BUCKET"], "models/latest/model.pkl")
          print("Model uploaded to S3.")
          EOF
        env:
          CLOUD_BUCKET: ${{ secrets.CLOUD_BUCKET }}
```

**TODO 6 — Job eval, Check eval gate:**
```yaml
        run: |
          python - <<'EOF'
          acc = float("${{ needs.train.outputs.accuracy }}")
          if acc < 0.70:
              raise SystemExit(f"FAILED: accuracy {acc:.4f} < 0.70. Huy deploy.")
          print(f"PASSED: accuracy {acc:.4f} >= 0.70. Dang trien khai model.")
          EOF
```

**TODO 7-8 — Job deploy, SSH script:**
```yaml
          script: |
            sudo systemctl restart mlops-serve
            sleep 5
            curl -sf http://localhost:8000/health && echo "Health check passed." || exit 1
```

### 2.12 Lần chạy pipeline đầu tiên

```bash
touch src/__init__.py tests/__init__.py   # nếu chưa có
git add .
git commit -m "feat: add CI/CD pipeline, tests, and serving API"
git push origin main
```

Theo dõi tab **Actions** trên GitHub.

### 2.13 Khởi động service và kiểm tra

Sau khi pipeline xanh:
```bash
ssh -i ~/.ssh/mlops-serve-key.pem ec2-user@<EC2_PUBLIC_IP> \
  "sudo systemctl start mlops-serve"

VM_IP=<EC2_PUBLIC_IP>

curl http://$VM_IP:8000/health
# → {"status": "ok"}

curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,0]}'
# → {"prediction":1,"label":"trung_binh"}
```

### 2.14 Đầu ra Bước 2
- [ ] Cả 4 GitHub Actions jobs (Test, Train, Eval, Deploy) màu xanh
- [ ] `curl /health` → `{"status": "ok"}`
- [ ] `curl /predict` → kết quả dự đoán hợp lệ
- [ ] AWS S3 Console hiển thị file dữ liệu (`dvc/`) và model (`models/latest/model.pkl`)
- [ ] Chụp màn hình tab Actions + kết quả curl

---

## Bước 3: Huấn luyện liên tục khi có dữ liệu mới

### 3.1 Thêm dữ liệu mới
```bash
python add_new_data.py
# → Cập nhật dữ liệu: 2998 -> 5996 mẫu
```

### 3.2 Phiên bản hóa và kích hoạt pipeline

Trước khi chạy `dvc push`, đảm bảo AWS credentials đã được export:
```bash
export AWS_ACCESS_KEY_ID=$(python -c "import json; print(json.load(open('sa-key.json'))['aws_access_key_id'])")
export AWS_SECRET_ACCESS_KEY=$(python -c "import json; print(json.load(open('sa-key.json'))['aws_secret_access_key'])")
```

```bash
dvc add data/train_phase1.csv          # DVC ghi nhận file đã thay đổi
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"
dvc push                                # QUAN TRỌNG: đẩy dữ liệu lên S3 TRƯỚC
git push origin main                    # Sau đó push code để kích hoạt CI/CD
```

### 3.3 Theo dõi pipeline
Vào tab **Actions** — pipeline tự động chạy khi `.dvc` file thay đổi.
Xác nhận commit message hiển thị trong tên lần chạy là `data: bổ sung...`.

### 3.4 Kiểm tra mô hình mới
```bash
curl http://$VM_IP:8000/health
curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,0]}'
```

### 3.5 So sánh kết quả hai lần chạy
Tải `metrics.json` từ Artifacts của cả 2 lần chạy Bước 2 và Bước 3 để so sánh accuracy, f1_score.

### 3.6 Đầu ra Bước 3
- [ ] Pipeline được kích hoạt bởi commit dữ liệu (không cần thao tác thủ công)
- [ ] Cả 4 jobs màu xanh
- [ ] Chụp màn hình Actions run được kích hoạt bởi commit dữ liệu
- [ ] Bảng so sánh accuracy Bước 2 vs Bước 3

---

## Bonus (không bắt buộc, tối đa +20 điểm)

### Bonus 1: Tracking MLflow từ xa với DagsHub (+4đ)
- Tạo tài khoản tại https://dagshub.com và kết nối repo GitHub
- Thêm `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` vào GitHub Secrets
- Sửa `mlops.yml`: thêm `env` block cho MLflow variables, bỏ sqlite local

### Bonus 2: Nhiều thuật toán (+4đ)
- Thêm `model_type` vào `params.yaml` (vd: `random_forest`, `gradient_boosting`, `logistic_regression`)
- Trong `src/train.py`: if/elif chọn model theo `model_type`, log `model_type` vào MLflow
- Chạy thí nghiệm với ≥ 2 thuật toán, so sánh trên MLflow UI

### Bonus 3: Báo cáo hiệu suất tự động (+4đ)
- Thêm bước trong `mlops.yml`:
  - Tính confusion matrix ở dạng text
  - Tính precision/recall cho từng lớp (0, 1, 2)
  - Ghi vào `outputs/report.txt`
  - Upload làm artifact cùng `metrics.json`

### Bonus 4: Hoàn trả về phiên bản trước (+4đ)
- Trước deploy: tải `outputs/metrics.json` của lần chạy trước từ S3
- So sánh accuracy mới vs cũ
- Chỉ deploy nếu accuracy mới ≥ accuracy cũ
- Ghi kết quả so sánh vào log

### Bonus 5: Cảnh báo lệch lạc dữ liệu (+4đ)
- Thêm bước kiểm tra phân phối nhãn trước khi train
- Tính tỷ lệ mẫu của từng lớp (0, 1, 2)
- Nếu lớp nào < 10% → in cảnh báo
- Ghi phân phối nhãn vào `outputs/metrics.json`

---

## Tổng kết thứ tự thực hiện

```
Bước 0: Setup môi trường, generate data
  ↓
Bước 1: Viết train.py → Chạy 3+ experiments → Phân tích MLflow → Chọn best params
  ↓
Bước 2: Tạo S3 bucket → IAM User → DVC init → Tạo EC2 → Cấu hình EC2
  ↓       Viết serve.py (boto3) → systemd service → SSH key → GitHub Secrets
  ↓       Viết test_train.py → Viết mlops.yml (AWS) → Push → Pipeline chạy
  ↓       Start service → Test curl /health & /predict
  ↓
Bước 3: Chạy add_new_data.py → dvc add → commit .dvc → dvc push → git push
  ↓       Theo dõi pipeline → Xác nhận model mới → So sánh metrics
  ↓
Bonus (optional): Hoàn thành 1-5 thử thách để đạt tối đa 100 điểm
```

## Nộp bài

1. URL repo GitHub public
2. Ảnh chụp màn hình:
   - MLflow UI (≥ 3 experiments)
   - GitHub Actions tab (4 jobs xanh cho Bước 2 và Bước 3)
   - `curl /health` và `curl /predict`
   - AWS S3 Console (dữ liệu + model)
3. Báo cáo ngắn (≤ 1 trang A4): siêu tham số đã chọn + lý do, khó khăn & cách giải quyết
