# ========================================================= 
# Logistic Regression (SGD) – EMBER 
# 5-Fold Stratified CV + OOF + Test 
# ========================================================= 
import json 
import numpy as np 
import pandas as pd 
import glob 
import os 
from tqdm import tqdm 
from sklearn.linear_model import SGDClassifier 
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
# ========================= 
# PATHS 
# ========================= 
BASE_DIR = r"C:\Users\al3x_X_y\Downloads\Logistic_EMBER" 
DATA_DIR = os.path.join(BASE_DIR, "Data") 
TRAIN_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "train_features_*.jsonl"))) 
TEST_FILE = os.path.join(DATA_DIR, "test_features.jsonl") 
CHUNK_SIZE = 5000 
N_SPLITS = 5 
# ========================= 
# MODEL CONFIG 
# ========================= 
MODEL_PARAMS = { 
    "loss": "log_loss", 
    "penalty": "l2", 
    "alpha": 1e-3, 
    "learning_rate": "optimal", 
    # "early_stopping": True, 
    "validation_fraction": 0.1, 
    "n_iter_no_change": 5, 
    "random_state": 42 
} 
# ========================= 
# FEATURE FLATTENING 
# ========================= 
def flatten_features(obj): 
    flat = {} 
    for k, v in obj.items(): 
        if k == "label": 
            flat["label"] = v 
        elif isinstance(v, (int, float)): 
            flat[k] = v 
        elif isinstance(v, list): 
            for i, val in enumerate(v): 
                if isinstance(val, (int, float)): 
                    flat[f"{k}_{i}"] = val 
    return flat 
# ========================= 
# CHUNK LOADER 
# ========================= 
def load_jsonl_chunks(path, chunk_size): 
    buf = [] 
    with open(path, "r") as f: 
        for i, line in enumerate(f): 
            buf.append(flatten_features(json.loads(line))) 
            if (i + 1) % chunk_size == 0: 
                yield pd.DataFrame(buf) 
                buf = [] 
    if buf: 
        yield pd.DataFrame(buf) 
# ========================= 
# LOAD LABELS ONLY (FOR CV) 
# ========================= 
print("Loading labels for CV...") 
labels = [] 
for file in TRAIN_FILES: 
    for chunk in load_jsonl_chunks(file, CHUNK_SIZE): 
        y = chunk.get("label") 
        if y is not None: 
            mask = y != -1 
            labels.append(y[mask]) 
y_all = pd.concat(labels, ignore_index=True) 
print("Total training samples:", len(y_all)) 
# ========================= 
# OOF STORAGE (CRITICAL) 
# ========================= 
oof_probs = np.zeros(len(y_all), dtype=np.float32) 
oof_labels = y_all.values.copy() 
# ========================= 
# STRATIFIED CV 
# ========================= 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42) 
cv_acc, cv_prec, cv_rec, cv_f1, cv_auc = [], [], [], [], [] 
print("\nRunning 5-Fold Stratified CV...") 
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), 1): 
    print(f"\n===== Fold {fold} =====") 
    model = SGDClassifier(**MODEL_PARAMS) 
    scaler = StandardScaler(with_mean=False) 
    first_fit = True 
    y_val_true, y_val_pred, y_val_prob = [], [], [] 

    global_ptr = 0 

    for file in TRAIN_FILES: 
        for chunk in tqdm(load_jsonl_chunks(file, CHUNK_SIZE), desc=f"Fold {fold}"): 
            if "label" not in chunk: 
                continue 

            y = chunk["label"] 
            X = chunk.drop(columns=["label"]).fillna(0) 

            mask = y != -1 
            y = y[mask] 
            X = X.loc[y.index] 

            idx_range = np.arange(global_ptr, global_ptr + len(y)) 
            global_ptr += len(y) 

            is_train = np.isin(idx_range, train_idx) 
            is_val = np.isin(idx_range, val_idx) 

            # TRAIN 
            if is_train.any(): 
                X_tr = X.iloc[is_train] 
                y_tr = y.iloc[is_train] 

                if first_fit: 
                    scaler.fit(X_tr) 

                X_tr = scaler.transform(X_tr) 

                if first_fit: 
                    model.partial_fit(X_tr, y_tr, classes=[0, 1]) 
                    first_fit = False 
                else: 
                    model.partial_fit(X_tr, y_tr) 

            # VALIDATION 
            if is_val.any(): 
                X_v = scaler.transform(X.iloc[is_val]) 
                y_v = y.iloc[is_val] 
                prob = model.predict_proba(X_v)[:, 1] 
                pred = (prob >= 0.5).astype(int) 
                oof_probs[val_idx[np.isin(val_idx, idx_range)]] = prob 
                y_val_true.extend(y_v) 
                y_val_pred.extend(pred) 
                y_val_prob.extend(prob) 
    
    # METRICS 
    acc = accuracy_score(y_val_true, y_val_pred) 
    prec = precision_score(y_val_true, y_val_pred) 
    rec = recall_score(y_val_true, y_val_pred) 
    f1 = f1_score(y_val_true, y_val_pred) 
    auc = roc_auc_score(y_val_true, y_val_prob) 
    cm = confusion_matrix(y_val_true, y_val_pred) 
    cv_acc.append(acc) 
    cv_prec.append(prec) 
    cv_rec.append(rec) 
    cv_f1.append(f1) 
    cv_auc.append(auc) 
    print( 
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | " 
        f"Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}" 
    ) 
    print("Confusion Matrix:\n", cm) 
# ========================= 
# CV SUMMARY 
# ========================= 
print("\n===== CV MEAN PERFORMANCE =====") 
print(f"Accuracy : {np.mean(cv_acc):.4f}") 
print(f"Precision: {np.mean(cv_prec):.4f}") 
print(f"Recall   : {np.mean(cv_rec):.4f}") 
print(f"F1-score : {np.mean(cv_f1):.4f}") 
print(f"ROC-AUC  : {np.mean(cv_auc):.4f}") 
# ========================= 
# SAVE OOF FILES 
# ========================= 
np.save("lr_train_oof_probs.npy", oof_probs) 
np.save("lr_train_labels.npy", oof_labels) 
print("\nSaved:") 
print("lr_train_oof_probs.npy") 
print("lr_train_labels.npy") 
print("Missing OOF values:", np.isnan(oof_probs).sum()) 
# ========================= 
# FINAL TRAINING (FULL DATA) 
# ========================= 
print("\nTraining final LR model on full data...") 
scaler = StandardScaler(with_mean=False) 
model = SGDClassifier(**MODEL_PARAMS) 
first_fit = True 
for file in TRAIN_FILES: 
    for chunk in load_jsonl_chunks(file, CHUNK_SIZE): 
        if "label" not in chunk: 
            continue 
        y = chunk["label"] 
        X = chunk.drop(columns=["label"]).fillna(0) 
        mask = y != -1 
        y = y[mask] 
        X = X.loc[y.index] 
        if first_fit: 
            scaler.fit(X) 
        X = scaler.transform(X) 
        if first_fit: 
            model.partial_fit(X, y, classes=[0, 1]) 
            first_fit = False 
        else: 
            model.partial_fit(X, y) 
# ========================= 
# TEST PREDICTION 
# ========================= 
print("\nPredicting on test set...") 
y_test_all, y_test_prob = [], [] 
for chunk in load_jsonl_chunks(TEST_FILE, CHUNK_SIZE): 
    if "label" not in chunk: 
        continue 
    y = chunk["label"] 
    X = chunk.drop(columns=["label"]).fillna(0) 
    mask = y != -1 
    y = y[mask] 
    X = X.loc[y.index] 
    X = scaler.transform(X) 
    prob = model.predict_proba(X)[:, 1] 
    y_test_all.extend(y) 
    y_test_prob.extend(prob) 
y_test_all = np.array(y_test_all) 
y_test_prob = np.array(y_test_prob) 
y_test_pred = (y_test_prob >= 0.5).astype(int) 
print("\n===== TEST PERFORMANCE (Logistic Regression) =====") 
print("Accuracy :", accuracy_score(y_test_all, y_test_pred)) 
print("Precision:", precision_score(y_test_all, y_test_pred)) 
print("Recall   :", recall_score(y_test_all, y_test_pred)) 
print("F1-score :", f1_score(y_test_all, y_test_pred)) 
print("ROC-AUC  :", roc_auc_score(y_test_all, y_test_prob)) 
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_test_pred)) 
# ========================= 
# SAVE TEST FILES 
# ========================= 
np.save("lr_test_probs.npy", y_test_prob) 
np.save("lr_test_labels.npy", y_test_all) 
print("\nSaved:") 
print("lr_test_probs.npy") 
print("lr_test_labels.npy") 
print("\nAll Logistic Regression stacking files generated successfully.")