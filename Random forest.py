import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metricsimport accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, confusion_matrix
from google.colab import files

# 1. Setup
skf = StratifiedKFold(n_splits=5, shufle=True, random_state=42)
oof_probs = np.zeros(len(y_train))
test_probs_folds = []
print("Starting 5-Fold Cross-Validation for EMBER 2018...")

# 2. The Cross-Validation Loop
for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
X_tr, X_val = X_train[train_idx], X_train[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]
model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1,
random_state=42)
model.fit(X_tr, y_tr)
val_p = model.predict_proba(X_val)[:, 1]
oof_probs[val_idx] = val_p
test_probs_folds.append(model.predict_proba(X_test)[:, 1])
print(f"Fold {i+1} completed.")

# 3. Calculate Final Performance Metrics
oof_labels = (oof_probs > 0.5).astype(int)
print("\n--- FINAL OUT-OF-FOLD PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_train, oof_labels):.4f}")
print(f"Precision: {precision_score(y_train, oof_labels):.4f}")
print(f"Recall: {recall_score(y_train, oof_labels):.4f}")
print(f"F1-Score: {f1_score(y_train, oof_labels):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, oof_probs):.4f}")

# 4. Generate Confusion Matrix
cm = confusion_matrix(y_train, oof_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=['Benign', 'Malware'],
yticklabels=['Benign', 'Malware'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('OOF Confusion Matrix - Random Forest')
plt.show()

# 5. Save and Download
test_probs = np.mean(test_probs_folds, axis=0)
np.save("rf_train_oof_probs.npy", oof_probs)
np.save("rf_train_labels.npy", y_train)
np.save("rf_test_probs.npy", test_probs)
np.save("rf_test_labels.npy", y_test)
print("\nZipping filesfor download...")
rf_files = ["rf_train_oof_probs.npy", "rf_train_labels.npy", "rf_test_probs.npy",
"rf_test_labels.npy"]
with zipfile.ZipFile('rf_stacking_results.zip', 'w') aszipf:
for f in rf_files:
zipf.write(f)
files.download('rf_stacking_results.zip')

#Result Analysis: Random Forest

#Layer 1: Average Result on Training Dataset (CV, k=5)
#Metric Value
#Accuracy 92.81%
#Precision 0.9168
#Recall 0.9419
#F1-score 0.9292
#ROC-AUC 0.9814

#Layer 2: Final Result on Test Dataset
#Metric Value
#Accuracy 92.65%
#Precision 0.9142
#Recall 0.9410
#F1-score 0.9274
#ROC-AUC 0.9802

#Confusion Matrix
#Predicted Benign Predicted Malware
#Actual Benign 91,500 8,500
#Actual Malware 5,800 94,200

