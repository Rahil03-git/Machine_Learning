# ===============================
# STEP 0: IMPORT LIBRARIES
# ===============================

import os
import json
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ===============================
# STEP 1: DATASET PATHS
# ===============================

DATA_PATH = r"D:\10th Trimester\CSE 4889 (Machine Learning)\Research_Paper\Ember-Malware-Detection\Data"

TRAIN_FILES = [
    "train_features_0.jsonl",
    "train_features_1.jsonl",
    "train_features_2.jsonl",
    "train_features_3.jsonl",
    "train_features_4.jsonl",
    "train_features_5.jsonl"
]

TEST_FILE = "test_features.jsonl"


# ===============================
# STEP 2: HELPER FUNCTION
# ===============================

def extract_features(sample):
    """
    Converts EMBER JSON object to numeric feature vector.
    We use EMBER static features only.
    """

    feature_vector = []

    # Byte histogram (256)
    feature_vector.extend(sample["histogram"])

    # Byte entropy (256)
    feature_vector.extend(sample["byteentropy"])

    # String features
    strings = sample["strings"]
    feature_vector.append(strings["numstrings"])
    feature_vector.append(strings["avlength"])
    feature_vector.append(strings["entropy"])
    feature_vector.append(strings["paths"])
    feature_vector.append(strings["urls"])
    feature_vector.append(strings["registry"])
    feature_vector.append(strings["MZ"])

    # General PE features
    general = sample["general"]
    feature_vector.extend([
        general["size"],
        general["vsize"],
        general["imports"],
        general["exports"],
        general["has_debug"],
        general["has_resources"],
        general["has_signature"],
        general["has_tls"]
    ])

    return np.array(feature_vector, dtype=np.float32)


# ===============================
# STEP 3: INITIALIZE MODEL
# ===============================

scaler = StandardScaler()

svm = SGDClassifier(
    loss="hinge",        # Linear SVM
    max_iter=1,          # Needed for partial_fit
    learning_rate="optimal",
    tol=None,
    random_state=42
)

classes = np.array([0, 1])  # 0 = Benign, 1 = Malware


# ===============================
# STEP 4: INCREMENTAL TRAINING
# ===============================

print("\nStarting training on ALL EMBER train files...\n")

first_batch = True

for file_name in TRAIN_FILES:
    file_path = os.path.join(DATA_PATH, file_name)
    print(f"Processing {file_path}")

    X_batch = []
    y_batch = []

    with open(file_path, "r") as f:
        for line in tqdm(f, desc=file_name):
            sample = json.loads(line)

            # Ignore unlabeled samples
            if sample["label"] == -1:
                continue

            X_batch.append(extract_features(sample))
            y_batch.append(sample["label"])

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)

    # Scale features incrementally
    if first_batch:
        scaler.fit(X_batch)
        X_scaled = scaler.transform(X_batch)
        svm.partial_fit(X_scaled, y_batch, classes=classes)
        first_batch = False
    else:
        X_scaled = scaler.transform(X_batch)
        svm.partial_fit(X_scaled, y_batch)

print("\nTraining completed successfully.\n")


# ===============================
# STEP 5: LOAD TEST DATA
# ===============================

print("Loading test data...\n")

X_test = []
y_test = []

with open(os.path.join(DATA_PATH, TEST_FILE), "r") as f:
    for line in tqdm(f, desc="Testing"):
        sample = json.loads(line)

        if sample["label"] == -1:
            continue

        X_test.append(extract_features(sample))
        y_test.append(sample["label"])

X_test = scaler.transform(np.array(X_test))
y_test = np.array(y_test)


# ===============================
# STEP 6: EVALUATION
# ===============================

y_pred = svm.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malware"]))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))







# =====================================================
# STEP 8: RESULT ANALYSIS
# =====================================================

# Classification Report:

#               precision    recall  f1-score   support

#       Benign       0.76      0.58      0.66    100000
#      Malware       0.66      0.82      0.73    100000

#     accuracy                           0.70    200000
#    macro avg       0.71      0.70      0.69    200000
# weighted avg       0.71      0.70      0.69    200000

# Confusion Matrix:

# [[58227 41773]
#  [18424 81576]]

# [Done] exited with code=0 in 204.487 seconds