"""
================================================
PHASE 4: CLASSIFICATION — PREDICT VISIT MODE
Tourism Experience Analytics Project
================================================
Goal: Predict the Visit Mode a user will use
      (Business / Couples / Family / Friends / Solo)

Models trained:
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - Gradient Boosting Classifier  ← best saved

Evaluation: Accuracy, Precision, Recall, F1
================================================
Run AFTER phase1_data_preparation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  PHASE 4: CLASSIFICATION MODEL")
print("=" * 55)

df = pd.read_csv('data/master_cleaned.csv')
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# ENCODE TARGET & FEATURES
# ─────────────────────────────────────────────
print("\n[1/5] Preparing features and target...")

le_target = LabelEncoder()
df['VisitMode_Label'] = le_target.fit_transform(df['VisitModeName'].astype(str))

# Save the target encoder for Streamlit
os.makedirs('models', exist_ok=True)
with open('models/visit_mode_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)

# Encode categorical features
le = LabelEncoder()
df['AttractionType_enc'] = le.fit_transform(df['AttractionType'].astype(str))
df['UserContinent_enc']  = le.fit_transform(df['UserContinent'].astype(str))
df['UserRegion_enc']     = le.fit_transform(df['UserRegion'].astype(str))
df['Season_enc']         = le.fit_transform(df['Season'].astype(str))

FEATURES = [
    'Rating',
    'VisitYear', 'VisitMonth', 'IsPeakSeason', 'Season_enc',
    'UserContinent_enc', 'UserRegion_enc',
    'AttractionType_enc',
    'AttractionAvgRating', 'AttractionVisitCount',
    'UserAvgRating', 'UserVisitCount',
]
TARGET = 'VisitMode_Label'
CLASS_NAMES = list(le_target.classes_)

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"  Features: {len(FEATURES)}")
print(f"  Classes : {CLASS_NAMES}")
print(f"  Class distribution:\n{df['VisitModeName'].value_counts()}")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[2/5] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# HELPER: EVALUATE
# ─────────────────────────────────────────────
def evaluate_clf(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average='weighted')
    pre = precision_score(y_te, preds, average='weighted', zero_division=0)
    rec = recall_score(y_te, preds, average='weighted', zero_division=0)

    print(f"\n  {name}:")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    F1 Score  = {f1:.4f}  (weighted)")
    print(f"    Precision = {pre:.4f}")
    print(f"    Recall    = {rec:.4f}")

    return model, preds, {'Accuracy': acc, 'F1': f1, 'Precision': pre, 'Recall': rec}

# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[3/5] Training classification models...")
results = {}

# Model 1: Logistic Regression (baseline)
lr_model, lr_preds, lr_metrics = evaluate_clf(
    "Logistic Regression",
    LogisticRegression(max_iter=500, random_state=42),
    X_train, y_train, X_test, y_test
)
results['Logistic Regression'] = lr_metrics

# Model 2: Random Forest
rf_model, rf_preds, rf_metrics = evaluate_clf(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test
)
results['Random Forest'] = rf_metrics

# Model 3: Gradient Boosting
gb_model, gb_preds, gb_metrics = evaluate_clf(
    "Gradient Boosting",
    GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    X_train, y_train, X_test, y_test
)
results['Gradient Boosting'] = gb_metrics

# ─────────────────────────────────────────────
# STEP 4: COMPARE & SAVE BEST MODEL
# ─────────────────────────────────────────────
print("\n[4/5] Comparing models and saving best...")

results_df = pd.DataFrame(results).T
print("\n  Model Comparison Table:")
print(results_df.to_string())

best_name = results_df['F1'].idxmax()
best_model = {'Logistic Regression': lr_model, 'Random Forest': rf_model, 'Gradient Boosting': gb_model}[best_name]
best_preds = {'Logistic Regression': lr_preds, 'Random Forest': rf_preds, 'Gradient Boosting': gb_preds}[best_name]

print(f"\n  Best model: {best_name}  (F1 = {results_df.loc[best_name, 'F1']:.4f})")

with open('models/classification_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'features': FEATURES,
        'model_name': best_name,
        'class_names': CLASS_NAMES
    }, f)
print("  Saved: models/classification_model.pkl")

print("\n  Full Classification Report (best model):")
print(classification_report(y_test, best_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────
print("[5/5] Generating evaluation plots...")

os.makedirs('outputs/classification', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Classification Evaluation — {best_name}', fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')
axes[0].tick_params(axis='x', rotation=25)

# Plot 2: Per-class F1
from sklearn.metrics import f1_score as f1_per
f1_per_class = f1_score(y_test, best_preds, average=None)
axes[1].bar(CLASS_NAMES, f1_per_class, color='#4C72B0', edgecolor='white')
axes[1].set_title('F1 Score per Class')
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1.1)
for i, v in enumerate(f1_per_class):
    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
axes[1].tick_params(axis='x', rotation=20)

# Plot 3: Feature importance
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fi.plot(kind='barh', ax=axes[2], color='#55A868')
    axes[2].set_title('Feature Importance')
    axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('outputs/classification/classification_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/classification/classification_evaluation.png")

# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ['Accuracy', 'F1', 'Precision', 'Recall']
x = np.arange(len(metrics_to_plot))
width = 0.25
colors = ['#4C72B0', '#55A868', '#DD8452']
for i, (model_name, row) in enumerate(results_df.iterrows()):
    ax.bar(x + i * width, [row[m] for m in metrics_to_plot],
           width, label=model_name, color=colors[i], edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.1)
ax.set_title('Model Comparison — All Metrics', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/classification/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/classification/model_comparison.png")

print(f"\n{'='*55}")
print("  PHASE 4 COMPLETE ✓")
print(f"{'='*55}")
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {results_df.loc[best_name, 'Accuracy']:.4f}")
print(f"  F1 Score   : {results_df.loc[best_name, 'F1']:.4f}")
