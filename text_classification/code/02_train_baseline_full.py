import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 1. Fast load data
print(" Loading 300k cleaned data...")
start_time = time.time()
df = pd.read_pickle("reuters_clean_all.pkl")
print(f" Load done! Time: {time.time() - start_time:.2f} s")

# Extract X and Y
X = df["text"]
Y = np.array(df["labels_vec"].tolist())

# Split train and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f" Train size: {len(X_train)} | Val size: {len(X_val)}")


# 2. Feature extraction (memory control)
print("\n Building TF-IDF features (top 20000 words)...")
vectorizer = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2)
)  # Day 2 improvement: add bi-gram
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)


# 3. Train baseline model
print("\n Training Logistic Regression (OneVsRest multi-label)...")
print("  (This may take 5-15 minutes, please wait...)")
train_start = time.time()

# Use liblinear solver for high-dim sparse matrix
base_lr = LogisticRegression(solver="liblinear", class_weight="balanced")
model = OneVsRestClassifier(base_lr)
model.fit(X_train_vec, Y_train)

print(f" Training done! Time: {time.time() - train_start:.2f} s")


# 4. Prediction and threshold tuning
print("\n Predicting and finding best threshold...")
# Key: use predict_proba instead of predict
Y_val_prob = model.predict_proba(X_val_vec)

# Test different thresholds to find best F1
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

for t in thresholds:
    # If prob >= t, assign 1
    Y_pred_t = (Y_val_prob >= t).astype(int)

    micro = f1_score(Y_val, Y_pred_t, average="micro", zero_division=0)
    macro = f1_score(Y_val, Y_pred_t, average="macro", zero_division=0)

    print("-" * 30)
    print(f" Threshold = {t}")
    print(f"  Micro-F1: {micro:.4f}")
    print(f"  Macro-F1: {macro:.4f}")

print("\n Baseline test done!")
