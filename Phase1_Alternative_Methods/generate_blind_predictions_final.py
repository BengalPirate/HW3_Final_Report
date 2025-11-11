"""
Generate Blind Predictions using the Ultimate Pipeline Strategy
Replicates the exact preprocessing and ensemble strategy from ultimate_pipeline.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("GENERATING BLIND PREDICTIONS - ULTIMATE PIPELINE")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
blind_data = pd.read_csv('blindData.txt', header=None)

print(f"Training: {train_data.shape}, Blind: {blind_data.shape}")

# Preprocess
print("\n[2/7] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
blind_data = blind_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_blind_base = imputer.transform(blind_data.values)

# Feature Selection using Mutual Information
print("\n[3/7] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)  # Keep top 90%
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")

X_base_selected = X_base[:, selected_features]
X_blind_base_selected = X_blind_base[:, selected_features]

# Feature Engineering
print("\n[4/7] Advanced feature engineering...")
top_mi_features = np.argsort(mi_scores)[::-1][:15]

def create_advanced_features(X_data, top_feat_idx):
    features = [X_data]

    # Polynomial features for top 15
    for feat in top_feat_idx[:15]:
        if feat < X_data.shape[1]:
            features.append((X_data[:, feat] ** 2).reshape(-1, 1))
            features.append((X_data[:, feat] ** 3).reshape(-1, 1))

    # Interactions for top 8
    for i in range(min(8, len(top_feat_idx))):
        for j in range(i+1, min(8, len(top_feat_idx))):
            if top_feat_idx[i] < X_data.shape[1] and top_feat_idx[j] < X_data.shape[1]:
                interaction = (X_data[:, top_feat_idx[i]] * X_data[:, top_feat_idx[j]]).reshape(-1, 1)
                features.append(interaction)
                # Ratio
                ratio = np.divide(X_data[:, top_feat_idx[i]],
                                 X_data[:, top_feat_idx[j]] + 1e-8).reshape(-1, 1)
                features.append(ratio)

    # Statistical features
    top_data = X_data[:, [f for f in top_feat_idx[:15] if f < X_data.shape[1]]]
    features.append(np.mean(top_data, axis=1).reshape(-1, 1))
    features.append(np.std(top_data, axis=1).reshape(-1, 1))
    features.append(np.max(top_data, axis=1).reshape(-1, 1))
    features.append(np.min(top_data, axis=1).reshape(-1, 1))
    features.append(np.median(top_data, axis=1).reshape(-1, 1))
    features.append(np.percentile(top_data, 25, axis=1).reshape(-1, 1))
    features.append(np.percentile(top_data, 75, axis=1).reshape(-1, 1))

    return np.hstack(features)

X_full = create_advanced_features(X_base_selected, top_mi_features)
X_blind = create_advanced_features(X_blind_base_selected, top_mi_features)

print(f"After feature engineering: {X_full.shape}")

# Scaling
scaler = RobustScaler()
X_full = scaler.fit_transform(X_full)
X_blind = scaler.transform(X_blind)

y_zero_based = y - 1

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5

# Best hyperparameters from ultimate_pipeline.py (Config 3 - Aggressive)
best_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.025,
    'num_leaves': 70,
    'max_depth': 14,
    'min_data_in_leaf': 12,
    'feature_fraction': 0.88,
    'bagging_fraction': 0.88,
    'bagging_freq': 2,
    'lambda_l1': 0.1,
    'lambda_l2': 0.4,
    'min_gain_to_split': 0.003,
    'verbose': -1,
    'is_unbalance': is_imbalanced,
}

# Second-best config for diversity
second_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.02,
    'num_leaves': 45,
    'max_depth': 10,
    'min_data_in_leaf': 18,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 2,
    'lambda_l1': 0.15,
    'lambda_l2': 0.6,
    'min_gain_to_split': 0.008,
    'verbose': -1,
    'is_unbalance': is_imbalanced,
}

# Train large ensemble (18 models matching ultimate_pipeline)
print("\n[5/7] Training 18-model ensemble on full training data...")

ensemble_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777,
                  1111, 2222, 5555, 8888, 9999]
ensemble_preds = []

# Use average best iteration from ultimate_pipeline (around 600-800)
avg_best_iter = 700  # Conservative estimate

# Train 15 models with best config
print(f"Training 15 models with best config...")
for seed_idx, seed in enumerate(ensemble_seeds):
    print(f"  Model {seed_idx+1}/15 (seed={seed})...", end='')

    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed

    full_train = lgb.Dataset(X_full, label=y_zero_based)
    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    blind_pred = model.predict(X_blind)
    ensemble_preds.append(blind_pred)
    print(" Done")

# Train 3 models with second-best config for diversity
print(f"\nTraining 3 models with 2nd-best config for diversity...")
for seed_idx, seed in enumerate([42, 123, 456]):
    print(f"  Model {seed_idx+1}/3 (seed={seed})...", end='')

    params_with_seed = second_params.copy()
    params_with_seed['seed'] = seed

    full_train = lgb.Dataset(X_full, label=y_zero_based)
    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    blind_pred = model.predict(X_blind)
    ensemble_preds.append(blind_pred)
    print(" Done")

# Weighted average (same as ultimate_pipeline)
print("\n[6/7] Computing weighted ensemble predictions...")
n_best = 15
n_second = 3

weights = [1.0] * n_best + [0.7] * n_second
weights = np.array(weights) / np.sum(weights)

blind_pred_final = np.average(ensemble_preds, axis=0, weights=weights)
blind_labels = np.argmax(blind_pred_final, axis=1) + 1

# Save in required format
print("\n[7/7] Saving predictions...")
output = np.column_stack([blind_pred_final, blind_labels])
np.savetxt('blindLabel.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\n" + "="*80)
print("✓ BLIND PREDICTIONS COMPLETE!")
print("="*80)
print(f"\nBlind dataset: {blind_data.shape[0]} samples")
print(f"Total ensemble models: {len(ensemble_preds)}")
print(f"Features used: {X_full.shape[1]} (from {X_base.shape[1]} original)")
print("\nPredicted class distribution:")
for i in range(1, 5):
    count = np.sum(blind_labels == i)
    pct = count / len(blind_labels) * 100
    print(f"  Class {i}: {count:5d} samples ({pct:5.2f}%)")

print("\nOutput file:")
print("  - blindLabel.txt")
print("\nFormat: 5 columns (4 class probabilities + final predicted class)")
print("="*80)
