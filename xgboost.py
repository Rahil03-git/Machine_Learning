import json
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# =========================
# LOAD JSONL FUNCTION
# =========================
def load_jsonl(file_path):
    """
    Load EMBER numeric features
    histogram (256) + byteentropy (256) = 512 features
    """
    X, y = [], []

    with open(file_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                features = obj["histogram"] + obj["byteentropy"]
                label = obj["label"]
                X.append(features)
                y.append(label)
            except Exception:
                continue

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

# =========================
# LOAD TRAINING DATA
# =========================
print("Loading EMBER training data...")

X_all, y_all = [], []

for i in range(6):
    X, y = load_jsonl(DATA_DIR / f"train_{i}.jsonl")
    X_all.append(X)
    y_all.append(y)

X_train = np.vstack(X_all)
y_train = np.hstack(y_all)

# Remove unlabeled samples (-1)
mask = y_train != -1
X_train = X_train[mask]
y_train = y_train[mask]

X_train = np.nan_to_num(X_train)

print("Training samples:", X_train.shape[0])

# =========================
# OOF STORAGE (CRITICAL)
# =========================
oof_probs = np.zeros(len(y_train), dtype=np.float32)
oof_labels = y_train.copy()

# =========================
# STRATIFIED K-FOLD CV
# =========================
print("\nRunning Stratified 5-Fold Cross Validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc, cv_prec, cv_rec, cv_f1, cv_auc = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nFold {fold}")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    scaler_cv = StandardScaler()
    X_tr = scaler_cv.fit_transform(X_tr)
    X_val = scaler_cv.transform(X_val)

    model_cv = XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model_cv.fit(X_tr, y_tr)

    y_val_pred = model_cv.predict(X_val)
    y_val_prob = model_cv.predict_proba(X_val)[:, 1]

    # STORE OOF PROBABILITIES
    oof_probs[val_idx] = y_val_prob

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_prob)

    cv_acc.append(acc)
    cv_prec.append(prec)
    cv_rec.append(rec)
    cv_f1.append(f1)
    cv_auc.append(auc)

    print(
        f"Accuracy: {acc:.4f} | "
        f"Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | "
        f"F1-score: {f1:.4f} | "
        f"ROC-AUC: {auc:.4f}"
    )

print("\n===== CV MEAN PERFORMANCE =====")
print(f"Accuracy : {np.mean(cv_acc)*100:.2f}%")
print(f"Precision: {np.mean(cv_prec):.4f}")
print(f"Recall   : {np.mean(cv_rec):.4f}")
print(f"F1-score : {np.mean(cv_f1):.4f}")
print(f"ROC-AUC  : {np.mean(cv_auc):.4f}")

# =========================
# SAVE OOF FILES (STACKING)
# =========================
np.save("xgb_train_oof_probs.npy", oof_probs)
np.save("xgb_train_labels.npy", oof_labels)

print("\nSaved:")
print("xgb_train_oof_probs.npy")
print("xgb_train_labels.npy")

# =========================
# FINAL TRAINING (FULL DATA)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

final_model = XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_scaled, y_train)

# =========================
# LOAD TEST DATA
# =========================
X_test, y_test = load_jsonl(DATA_DIR / "test.jsonl")

mask = y_test != -1
X_test = X_test[mask]
y_test = y_test[mask]

X_test = np.nan_to_num(X_test)
X_test = scaler.transform(X_test)

# =========================
# TEST PREDICTIONS
# =========================
y_test_prob = final_model.predict_proba(X_test)[:, 1]

# =========================
# SAVE TEST FILES (STACKING)
# =========================
np.save("xgb_test_probs.npy", y_test_prob)
np.save("xgb_test_labels.npy", y_test)

print("\nSaved:")
print("xgb_test_probs.npy")
print("xgb_test_labels.npy")

print("\nAll stacking files generated successfully.")
