import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# === Load datasets ===
fake_df = pd.read_csv("data/fusers.csv").fillna(0)
real_df = pd.read_csv("data/users.csv").fillna(0)

fake_df["label"] = 1
real_df["label"] = 0

# === Balance dataset ===
min_size = min(len(fake_df), len(real_df))
fake_df = resample(fake_df, n_samples=min_size, random_state=42)
real_df = resample(real_df, n_samples=min_size, random_state=42)

df = pd.concat([real_df, fake_df], ignore_index=True)

# === Drop non-numeric and text columns ===
drop_cols = ['username', 'full_name', 'biography', 'name', 'screen_name', 'description', 'created_at', 'updated', 'url', 'dataset']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

# === Derived features ===
df["follower_following_ratio"] = df["edge_followed_by"] / (df["edge_follow"] + 1)
df["following_to_follower_ratio"] = df["edge_follow"] / (df["edge_followed_by"] + 1)

# === Clean up ===
df.replace([np.inf, -np.inf], 0, inplace=True)
df = df.select_dtypes(include=[np.number])

# === Features and target ===
X = df.drop(columns=["label"])
y = df["label"]
feature_columns = X.columns.tolist()

# === Scale ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === XGBoost model ===
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# === Random Forest with calibration ===
rf_base = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_base.fit(X_train, y_train)

rf_model = CalibratedClassifierCV(base_estimator=rf_base, method='sigmoid', cv=3)
rf_model.fit(X_train, y_train)

# === Evaluation ===
print("\n✅ XGBoost Evaluation:")
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, xgb_preds))
print("AUC:", roc_auc_score(y_test, xgb_probs))

print("\n✅ Calibrated Random Forest Evaluation:")
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, rf_preds))
print("AUC:", roc_auc_score(y_test, rf_probs))

# === Save models and artifacts ===
os.makedirs("model", exist_ok=True)
pickle.dump(xgb_model, open("model/xgb_model.pkl", "wb"))
pickle.dump(rf_model, open("model/rf_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(feature_columns, open("model/feature_order.pkl", "wb"))

print("\n✅ Models, scaler, and feature list saved in /model/")

# === Confusion Matrix for XGBoost ===
xgb_cm = confusion_matrix(y_test, xgb_preds)
fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))  # Smaller figure
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'], ax=ax1)
ax1.set_title("XGBoost Confusion Matrix", fontsize=10)
ax1.set_xlabel("Predicted", fontsize=8)
ax1.set_ylabel("Actual", fontsize=8)
plt.tight_layout(pad=1)
plt.savefig("model/xgb_confusion_matrix.png", dpi=150)
plt.close(fig1)

# === Confusion Matrix for Random Forest ===
rf_cm = confusion_matrix(y_test, rf_preds)
fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))  # Smaller figure
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'], ax=ax2)
ax2.set_title("Random Forest Confusion Matrix", fontsize=10)
ax2.set_xlabel("Predicted", fontsize=8)
ax2.set_ylabel("Actual", fontsize=8)
plt.tight_layout(pad=1)
plt.savefig("model/rf_confusion_matrix.png", dpi=150)
plt.close(fig2)