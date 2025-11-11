"""
Optuna Hyperparameter Tuning Pipeline - Bayesian Optimization
Systematically searches for optimal hyperparameters using Optuna.
Expected gain: 0.3-0.6% (or improve on 94.40%)
Time: ~1-2 hours (50 trials)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
import optuna
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("OPTUNA HYPERPARAMETER TUNING - BAYESIAN OPTIMIZATION")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"Training: {train_data.shape}, Test: {test_data.shape}")

# Preprocess
print("\n[2/7] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Selection
print("\n[3/7] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")

X_base_selected = X_base[:, selected_features]
X_test_base_selected = X_test_base[:, selected_features]

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
X_test = create_advanced_features(X_test_base_selected, top_mi_features)

print(f"After feature engineering: {X_full.shape}")

# Scaling
scaler = RobustScaler()
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

y_zero_based = y - 1

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5

# Optuna objective function
print("\n[5/7] Setting up Optuna optimization...")

def objective(trial):
    # Suggest hyperparameters
    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 40, 100),
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 8, 25),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.5),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.01, log=True),
        'verbose': -1,
        'is_unbalance': is_imbalanced,
        'seed': 42
    }

    # 3-fold CV for faster evaluation
    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        # Train model without pruning callback (simpler approach)
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1500,
            valid_sets=[valid_dataset],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80),
                lgb.log_evaluation(period=0)
            ]
        )

        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_labels = np.argmax(y_val_pred, axis=1)
        accuracy = accuracy_score(y_val, y_val_labels)

        cv_scores.append(accuracy)

    return np.mean(cv_scores)

# Create study
print("\n[6/7] Running Optuna optimization (50 trials)...")
print("This will take approximately 1-2 hours...")
print("")

study = optuna.create_study(
    direction='maximize',
    study_name='lightgbm_optimization'
)

# Run optimization
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Print results
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

print(f"\nBest trial: {study.best_trial.number}")
print(f"Best CV accuracy: {study.best_trial.value:.4f} ({study.best_trial.value*100:.2f}%)")

print("\nBest hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

# Compare with baseline
baseline_acc = 0.9158
pseudo_acc = 0.9440
print(f"\nBaseline (no tuning): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Pseudo-labeling: {pseudo_acc:.4f} ({pseudo_acc*100:.2f}%)")
print(f"Optuna optimized: {study.best_trial.value:.4f} ({study.best_trial.value*100:.2f}%)")

if study.best_trial.value > pseudo_acc:
    improvement = (study.best_trial.value - pseudo_acc) * 100
    print(f"✓ Improvement over pseudo-labeling: +{improvement:.2f}%")
elif study.best_trial.value > baseline_acc:
    improvement = (study.best_trial.value - baseline_acc) * 100
    print(f"✓ Improvement over baseline: +{improvement:.2f}%")

# Train final model with best parameters
print("\n[7/7] Training final model with optimized parameters...")

best_params = study.best_trial.params
best_params.update({
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'verbose': -1,
    'is_unbalance': is_imbalanced
})

# 5-fold CV for final evaluation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_scores = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]

    train_dataset = lgb.Dataset(X_train, label=y_train)
    valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    model = lgb.train(
        best_params,
        train_dataset,
        num_boost_round=2000,
        valid_sets=[valid_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=120),
            lgb.log_evaluation(period=0)
        ]
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_val_labels = np.argmax(y_val_pred, axis=1)
    accuracy = accuracy_score(y_val, y_val_labels)

    cv_scores.append(accuracy)
    fold_models.append(model)

    print(f"  Fold {fold+1}: Acc={accuracy:.4f}, Iter={model.best_iteration}")

final_cv_score = np.mean(cv_scores)
print(f"\nFinal 5-fold CV Score: {final_cv_score:.4f} ± {np.std(cv_scores):.4f}")

# Train ensemble on full data
print("\nTraining final ensemble (15 models)...")

ensemble_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777,
                  1111, 2222, 5555, 8888, 9999]

avg_best_iter = int(np.mean([m.best_iteration for m in fold_models]))
print(f"Using average iteration count: {avg_best_iter}")

full_train = lgb.Dataset(X_full, label=y_zero_based)
ensemble_preds = []

for seed_idx, seed in enumerate(ensemble_seeds):
    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed

    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    test_pred = model.predict(X_test)
    ensemble_preds.append(test_pred)
    print(f"  Model {seed_idx+1}/{len(ensemble_seeds)} complete")

# Average ensemble predictions
test_pred_final = np.mean(ensemble_preds, axis=0)
test_labels = np.argmax(test_pred_final, axis=1) + 1

# Save
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel_optuna_tuned.txt', output, delimiter='\t',
           fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\n" + "="*80)
print("✓ OPTUNA HYPERPARAMETER TUNING COMPLETE!")
print("="*80)

print(f"\nFinal Results:")
print(f"  5-fold CV Accuracy: {final_cv_score:.4f} ({final_cv_score*100:.2f}%)")
print(f"  Baseline: 91.58%")
print(f"  Pseudo-labeling: 94.40%")
print(f"  Target: 92.82%")
print(f"  Partner: 92.58%")

if final_cv_score >= 0.9440:
    print(f"✓✓ IMPROVED ON PSEUDO-LABELING!")
elif final_cv_score >= 0.9282:
    print(f"✓ EXCEEDED TARGET!")
else:
    gap = (0.9440 - final_cv_score) * 100
    print(f"Gap to pseudo-labeling: {gap:.2f}%")

print(f"\nFile saved: testLabel_optuna_tuned.txt")
print(f"\nBest hyperparameters saved in study object")
