"""
Stacking Ensemble Pipeline - Meta-Learner Approach
Trains diverse base models and combines with meta-learner.
Expected gain: 0.4-0.7% (or maintain/improve on 94.40%)
Time: ~45 minutes
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("STACKING ENSEMBLE PIPELINE - META-LEARNER APPROACH")
print("="*80)

# Load data
print("\n[1/8] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"Training: {train_data.shape}, Test: {test_data.shape}")

# Preprocess
print("\n[2/8] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Selection
print("\n[3/8] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")

X_base_selected = X_base[:, selected_features]
X_test_base_selected = X_test_base[:, selected_features]

# Feature Engineering
print("\n[4/8] Advanced feature engineering...")
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

print("\n[5/8] Setting up diverse base models...")

# Base Model 1: LightGBM (best from previous)
base_model_1_params = {
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

# Base Model 2: LightGBM (conservative)
base_model_2_params = {
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

# Base Model 3: XGBoost
base_model_3 = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100,
    eval_metric='mlogloss'
)

# Base Model 4: Neural Network
base_model_4 = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

print("Base models configured:")
print("  1. LightGBM (Aggressive)")
print("  2. LightGBM (Conservative)")
print("  3. XGBoost")
print("  4. Neural Network (MLP)")

# Stacking with 5-fold CV
print("\n[6/8] Training base models with 5-fold CV stacking...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store meta-features for training meta-learner
meta_features_train = np.zeros((X_full.shape[0], 4 * 4))  # 4 models × 4 classes
meta_features_test = np.zeros((X_test.shape[0], 4 * 4))

# Store base model predictions for averaging
test_preds_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
    print(f"\nFold {fold+1}/{n_folds}")
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]

    fold_test_preds = []

    # Model 1: LightGBM Aggressive
    print("  Training LightGBM (Aggressive)...")
    train_data_lgb1 = lgb.Dataset(X_train, label=y_train)
    valid_data_lgb1 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb1)

    model_1 = lgb.train(
        base_model_1_params,
        train_data_lgb1,
        num_boost_round=2000,
        valid_sets=[valid_data_lgb1],
        callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)]
    )

    val_pred_1 = model_1.predict(X_val, num_iteration=model_1.best_iteration)
    test_pred_1 = model_1.predict(X_test, num_iteration=model_1.best_iteration)
    meta_features_train[val_idx, 0:4] = val_pred_1
    fold_test_preds.append(test_pred_1)

    # Model 2: LightGBM Conservative
    print("  Training LightGBM (Conservative)...")
    train_data_lgb2 = lgb.Dataset(X_train, label=y_train)
    valid_data_lgb2 = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb2)

    model_2 = lgb.train(
        base_model_2_params,
        train_data_lgb2,
        num_boost_round=2000,
        valid_sets=[valid_data_lgb2],
        callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)]
    )

    val_pred_2 = model_2.predict(X_val, num_iteration=model_2.best_iteration)
    test_pred_2 = model_2.predict(X_test, num_iteration=model_2.best_iteration)
    meta_features_train[val_idx, 4:8] = val_pred_2
    fold_test_preds.append(test_pred_2)

    # Model 3: XGBoost
    print("  Training XGBoost...")
    model_3 = xgb.XGBClassifier(**base_model_3.get_params())
    model_3.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_3 = model_3.predict_proba(X_val)
    test_pred_3 = model_3.predict_proba(X_test)
    meta_features_train[val_idx, 8:12] = val_pred_3
    fold_test_preds.append(test_pred_3)

    # Model 4: Neural Network
    print("  Training Neural Network...")
    model_4 = MLPClassifier(**base_model_4.get_params())
    model_4.fit(X_train, y_train)

    val_pred_4 = model_4.predict_proba(X_val)
    test_pred_4 = model_4.predict_proba(X_test)
    meta_features_train[val_idx, 12:16] = val_pred_4
    fold_test_preds.append(test_pred_4)

    # Fold validation accuracy
    fold_avg_pred = np.mean([val_pred_1, val_pred_2, val_pred_3, val_pred_4], axis=0)
    fold_pred_labels = np.argmax(fold_avg_pred, axis=1)
    fold_acc = accuracy_score(y_val, fold_pred_labels)
    print(f"  Fold {fold+1} base ensemble accuracy: {fold_acc:.4f}")

    # Average test predictions for this fold
    test_preds_per_fold.append(np.mean(fold_test_preds, axis=0))

# Average test predictions across folds for meta-features
meta_features_test = np.mean([
    np.hstack([test_preds_per_fold[fold][i] for i in range(4)])
    for fold in range(n_folds)
], axis=0).reshape(X_test.shape[0], -1)

print(f"\nMeta-features shape - Train: {meta_features_train.shape}, Test: {meta_features_test.shape}")

# Train meta-learner
print("\n[7/8] Training meta-learner (Logistic Regression)...")

meta_learner = LogisticRegression(
    C=0.1,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

meta_learner.fit(meta_features_train, y_zero_based)

# Evaluate meta-learner on training set (should be high)
meta_train_pred = meta_learner.predict(meta_features_train)
meta_train_acc = accuracy_score(y_zero_based, meta_train_pred)
print(f"Meta-learner training accuracy: {meta_train_acc:.4f}")

# Final test predictions
print("\n[8/8] Generating final stacked predictions...")
test_pred_proba = meta_learner.predict_proba(meta_features_test)
test_labels = np.argmax(test_pred_proba, axis=1) + 1

# Save
output = np.column_stack([test_pred_proba, test_labels])
np.savetxt('testLabel_stacked.txt', output, delimiter='\t',
           fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\n" + "="*80)
print("✓ STACKING ENSEMBLE PIPELINE COMPLETE!")
print("="*80)

print(f"\\nMeta-learner training accuracy: {meta_train_acc:.4f} ({meta_train_acc*100:.2f}%)")
print("\\nBase Models:")
print("  - LightGBM (Aggressive)")
print("  - LightGBM (Conservative)")
print("  - XGBoost")
print("  - Neural Network (MLP)")
print("\\nMeta-Learner: Logistic Regression")

print(f"\\nFile saved: testLabel_stacked.txt")

print(f"\\nTarget: 92.82%")
print(f"Partner: 92.58%")
print(f"Pseudo-labeling: 94.40%")

print("\\nNote: Final test accuracy unknown until submission.")
print("Stacking should capture complementary strengths of diverse models!")
