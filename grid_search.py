import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from itertools import product
import random
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

# ============================================================
# Data + feature engineering (same as src/train.py)
# ============================================================
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

t = pd.read_csv('data/train_phase1.csv')
e = pd.read_csv('data/eval.csv')

X_train = add_features(t.drop(columns=['target']))
y_train = t['target']
X_eval  = add_features(e.drop(columns=['target']))
y_eval  = e['target']

print(f'Train: {len(X_train)} ({X_train.shape[1]} feats) | Eval: {len(X_eval)}')

# ============================================================
# Search space — conservative, anti-overfit ranges
# ============================================================
PARAM_SPACE = {
    'n_estimators':     [80, 100, 120, 150, 180, 200],
    'max_depth':        [4, 5, 6, 7],
    'learning_rate':    [0.03, 0.05, 0.07, 0.1],
    'subsample':        [0.7, 0.75, 0.8, 0.85, 0.9],
    'colsample_bytree': [0.6, 0.65, 0.7, 0.75, 0.8],
    'min_child_weight': [3, 5, 7, 10],
    'reg_alpha':        [0.3, 0.5, 0.7, 1.0],
    'reg_lambda':       [1.0, 1.5, 2.0, 3.0],
    'gamma':            [0.0, 0.1, 0.2, 0.3, 0.5],
}

FIXED = dict(objective='multi:softmax', num_class=3, n_jobs=1, verbosity=0, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate all combos then sample
all_combos = list(product(*PARAM_SPACE.values()))
print(f'Full grid: {len(all_combos)} combos — sampling 500')

sampled = random.sample(all_combos, min(500, len(all_combos)))

results = []

print(f'Running {len(sampled)} combos...\n')

# ============================================================
# Run search
# ============================================================
for i, combo in enumerate(sampled):
    params = dict(zip(PARAM_SPACE.keys(), combo))
    model = xgb.XGBClassifier(**params, **FIXED)

    # 5-fold CV
    cv_scores = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = model.predict(X_train.iloc[val_idx])
        cv_scores.append(accuracy_score(y_train.iloc[val_idx], preds))

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)

    # Holdout eval
    model.fit(X_train, y_train)
    eval_acc = accuracy_score(y_eval, model.predict(X_eval))
    eval_f1  = f1_score(y_eval, model.predict(X_eval), average='weighted')
    gap = cv_mean - eval_acc

    results.append((cv_mean, eval_acc, eval_f1, gap, cv_std, params))

    if (i + 1) % 50 == 0:
        print(f'  [{i+1:4d}/500] done')

# ============================================================
# Sort & report top 20
# ============================================================
results.sort(key=lambda r: (r[1], -r[3]), reverse=True)

print(f'\n========== TOP 20 (sorted by eval acc, then larger eval margin) ==========')
for rank, (cv, ea, ef1, gap, cvstd, p) in enumerate(results[:20]):
    flag = '⚠️ OVERFIT' if gap > 0.03 else ('✓ tight' if gap < 0.02 else '~ ok')
    print(f'{rank+1:2d}. CV={cv:.4f}±{cvstd:.3f}  Eval={ea:.4f}  F1={ef1:.4f}  Gap={gap:+.4f}  {flag}')
    print(f'    n={p["n_estimators"]:3d}  d={p["max_depth"]}  lr={p["learning_rate"]:.3f}  '
          f'ss={p["subsample"]:.2f}  cb={p["colsample_bytree"]:.2f}  mcw={p["min_child_weight"]}  '
          f'a={p["reg_alpha"]:.1f}  l={p["reg_lambda"]:.1f}  g={p["gamma"]:.2f}')

# ============================================================
# Recommended: tight gap + good eval
# ============================================================
valid = [(cv, ea, ef1, gap, cvstd, p) for cv, ea, ef1, gap, cvstd, p in results if gap < 0.025 and ea >= 0.70]
if not valid:
    valid = results[:5]
valid.sort(key=lambda r: (r[1], abs(r[3])))

print(f'\n========== RECOMMENDED (gap < 0.025, eval >= 0.70) ==========')
for rank, (cv, ea, ef1, gap, cvstd, p) in enumerate(valid[:5]):
    print(f'{rank+1}. CV={cv:.4f}±{cvstd:.3f}  Eval={ea:.4f}  F1={ef1:.4f}  Gap={gap:+.4f}')
    print(f'   {p}')

if valid:
    best = valid[0]
    print(f'\n>>> COPY THIS TO params.yaml:')
    print(f'n_estimators: {best[5]["n_estimators"]}')
    print(f'max_depth: {best[5]["max_depth"]}')
    print(f'learning_rate: {best[5]["learning_rate"]}')
    print(f'subsample: {best[5]["subsample"]}')
    print(f'colsample_bytree: {best[5]["colsample_bytree"]}')
    print(f'min_child_weight: {best[5]["min_child_weight"]}')
    print(f'gamma: {best[5]["gamma"]}')
    print(f'reg_alpha: {best[5]["reg_alpha"]}')
    print(f'reg_lambda: {best[5]["reg_lambda"]}')
    print(f'random_state: 42')
    print(f'n_jobs: 1')
    print(f'verbosity: 0')
