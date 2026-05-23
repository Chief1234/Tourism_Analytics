"""
================================================
PHASE 3: REGRESSION — PREDICT ATTRACTION RATING
Tourism Experience Analytics Project
================================================
Goal: Predict the rating (1–5) a user will give
      to a tourist attraction.

Models trained:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor  ← best model saved

Evaluation: R², MAE, RMSE
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  PHASE 3: REGRESSION MODEL")
print("=" * 55)

df = pd.read_csv('data/master_cleaned.csv')
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n[1/5] Selecting features...")

# Encode any text columns needed
le = LabelEncoder()
df['VisitModeName_enc']  = le.fit_transform(df['VisitModeName'].astype(str))
df['AttractionType_enc'] = le.fit_transform(df['AttractionType'].astype(str))
df['UserContinent_enc']  = le.fit_transform(df['UserContinent'].astype(str))
df['UserRegion_enc']     = le.fit_transform(df['UserRegion'].astype(str))
df['Season_enc']         = le.fit_transform(df['Season'].astype(str))

FEATURES = [
    'VisitYear', 'VisitMonth', 'IsPeakSeason',
    'VisitModeName_enc',
    'UserContinent_enc', 'UserRegion_enc',
    'AttractionType_enc',
    'AttractionAvgRating', 'AttractionVisitCount',
    'UserAvgRating', 'UserVisitCount',
    'Season_enc'
]
TARGET = 'Rating'

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f"  Features used: {len(FEATURES)}")
print(f"  Feature list: {FEATURES}")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\n[2/5] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# HELPER: EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    # Clip predictions to valid rating range 1–5
    preds = np.clip(preds, 1, 5)

    r2   = r2_score(y_te, preds)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))

    print(f"\n  {name}:")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

    return model, preds, {'R2': r2, 'MAE': mae, 'RMSE': rmse}

# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[3/5] Training regression models...")
results = {}

# Model 1: Linear Regression (baseline)
lr_model, lr_preds, lr_metrics = evaluate_model(
    "Linear Regression", LinearRegression(),
    X_train, y_train, X_test, y_test
)
results['Linear Regression'] = lr_metrics

# Model 2: Random Forest
rf_model, rf_preds, rf_metrics = evaluate_model(
    "Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test
)
results['Random Forest'] = rf_metrics

# Model 3: Gradient Boosting (XGBoost alternative, no extra install needed)
gb_model, gb_preds, gb_metrics = evaluate_model(
    "Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
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

# Best model = highest R²
best_name = results_df['R2'].idxmax()
best_model = {'Linear Regression': lr_model, 'Random Forest': rf_model, 'Gradient Boosting': gb_model}[best_name]
best_preds = {'Linear Regression': lr_preds, 'Random Forest': rf_preds, 'Gradient Boosting': gb_preds}[best_name]

print(f"\n  Best model: {best_name}  (R² = {results_df.loc[best_name, 'R2']:.4f})")

os.makedirs('models', exist_ok=True)
with open('models/regression_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'features': FEATURES, 'model_name': best_name}, f)
print("  Saved: models/regression_model.pkl")

# Also save feature list for Streamlit
with open('models/regression_features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

# ─────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[5/5] Generating evaluation plots...")

os.makedirs('outputs/regression', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Regression Evaluation — {best_name}', fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, best_preds, alpha=0.3, color='#4C72B0', s=10)
axes[0, 0].plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect fit')
axes[0, 0].set_xlabel('Actual Rating')
axes[0, 0].set_ylabel('Predicted Rating')
axes[0, 0].set_title('Actual vs Predicted Rating')
axes[0, 0].legend()

# Plot 2: Residuals
residuals = y_test - best_preds
axes[0, 1].hist(residuals, bins=40, color='#DD8452', edgecolor='white')
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Residual (Actual − Predicted)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')

# Plot 3: Model comparison bar chart
metric_names = ['R2', 'MAE', 'RMSE']
x = np.arange(len(metric_names))
width = 0.25
colors = ['#4C72B0', '#55A868', '#DD8452']
for i, (model_name, row) in enumerate(results_df.iterrows()):
    axes[1, 0].bar(x + i * width, [row['R2'], row['MAE'], row['RMSE']],
                   width, label=model_name, color=colors[i], edgecolor='white')
axes[1, 0].set_xticks(x + width)
axes[1, 0].set_xticklabels(metric_names)
axes[1, 0].set_title('Model Comparison')
axes[1, 0].legend(fontsize=8)

# Plot 4: Feature importance (Random Forest)
if hasattr(rf_model, 'feature_importances_'):
    fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
    fi.plot(kind='barh', ax=axes[1, 1], color='#C44E52')
    axes[1, 1].set_title('Feature Importance (Random Forest)')
    axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('outputs/regression/regression_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/regression/regression_evaluation.png")

print(f"\n{'='*55}")
print("  PHASE 3 COMPLETE ✓")
print(f"{'='*55}")
print(f"\n  Best Model : {best_name}")
print(f"  R²         : {results_df.loc[best_name, 'R2']:.4f}  (1.0 = perfect)")
print(f"  MAE        : {results_df.loc[best_name, 'MAE']:.4f}  (avg error in rating points)")
print(f"  RMSE       : {results_df.loc[best_name, 'RMSE']:.4f}")
