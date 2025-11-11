# Phase 1: Alternative Methods and Exploratory Work
**Author:** Brandon Newton
**Test Accuracy Achieved:** 92.00%
**Cross-Validation:** 91.58% ± 0.09%

This folder contains the exploratory work and alternative methods developed during Phase 1 of the project. While these approaches achieved solid 92% test accuracy, they were ultimately not selected for final submission in favor of the Phase 2 noise-robust approach (92.82%).

## Overview

Phase 1 focused on sophisticated feature engineering and large ensemble strategies to maximize performance through complexity. Key contributions include:

1. **Advanced Feature Engineering**
   - Mutual Information feature selection (411 → 314 features)
   - Polynomial features (x², x³) for top 15 features
   - Interaction features (products and ratios)
   - Statistical aggregations (mean, std, min, max, etc.)

2. **Large Ensemble Strategy**
   - 18-model weighted ensemble
   - Multiple hyperparameter configurations
   - Seed-based diversity

3. **Experimental Approaches**
   - SMOTE for class imbalance (failed - 87.36%)
   - Pseudo-labeling semi-supervised learning
   - Stacking ensemble with diverse models
   - Optuna Bayesian hyperparameter optimization

## Files in This Directory

### Core Implementation Files
- `ultimate_pipeline.py` - Main Phase 1 pipeline with feature engineering
- `generate_blind_predictions_final.py` - Blind data prediction script
- `fast_xgboost.py` - XGBoost baseline implementation
- `hypertuned_pipeline.py` - LightGBM hyperparameter tuning

### Experimental Files (Failed Approaches)
- `optimized_pipeline.py` - SMOTE attempt (87.36% accuracy)
- `pseudo_labeling_pipeline.py` - Semi-supervised learning
- `stacking_ensemble_pipeline.py` - Stacking meta-learner
- `optuna_tuning_pipeline.py` - Bayesian optimization

### Analysis and Visualization
- `explore_data.py` - Exploratory data analysis
- `partner_inspired_pipeline.py` - Noise-robust exploration

### Documentation
- `FINAL_REPORT.md` - Original Phase 1 report
- `OPTIMIZATION_README.md` - Optimization journey
- `WRITEUP.md` - Detailed write-up

## Key Results

### Performance Comparison

| Approach | CV Accuracy | Test Accuracy | Key Strategy |
|----------|-------------|---------------|--------------|
| Random Forest Baseline | ~85% | ~85% | Initial exploration |
| XGBoost Baseline | 90.01% CV | 90% | Fast gradient boosting |
| LightGBM + MI Selection | 91.58% CV | 92% | Feature engineering |
| 18-Model Ensemble | 91.58% CV | 92% | Large weighted ensemble |

### Why Phase 1 Didn't Win

Despite achieving 92% accuracy, Phase 1 was not selected for final submission because:

1. **Complexity vs. Performance Trade-off**
   - 407 engineered features vs. 411 original features (Phase 2)
   - 4 preprocessing steps vs. 1 step (Phase 2)
   - 18 models vs. 8 models (Phase 2)
   - 30-minute training vs. 15 minutes (Phase 2)

2. **Generalization Gap**
   - Phase 1: 91.58% CV → 92.00% test (+0.42% gain)
   - Phase 2: 91.07% CV → 92.82% test (+1.75% gain)
   - Phase 2 showed better real-world generalization

3. **Noise Amplification Risk**
   - Feature engineering on noisy data may amplify noise
   - Polynomial and interaction terms create noise-noise interactions
   - Phase 2's approach of keeping original features proved more robust

4. **Assignment Context**
   - Assignment stated: "All three datasets cannot get away of embedded noise"
   - Phase 1 treated noise as signal to extract
   - Phase 2 treated noise as noise to resist

## Failed Experiments - Valuable Lessons

### 1. SMOTE (Synthetic Minority Over-sampling)
**Result:** 87.36% accuracy (dropped from 90%)

**Why it failed:**
- High-dimensional space (411 features) makes interpolation unreliable
- Synthetic samples may not represent true distribution with embedded noise
- Class weighting (`is_unbalance=True`) sufficient for this problem

### 2. Pseudo-Labeling
**Result:** No improvement (~90-91%)

**Why it failed:**
- Risk of error propagation with noisy data
- With 27K training samples, adding 2K pseudo-labels insufficient
- Cannot validate properly when using test set for training

### 3. Stacking Ensemble
**Result:** Similar to best single model (~91%)

**Why it failed:**
- All tree models highly correlated (made similar predictions)
- LightGBM already near-optimal
- Little room for meta-learner improvement
- Computational cost not justified

### 4. Optuna Hyperparameter Tuning
**Result:** Confirmed manual tuning was good (~91%)

**Finding:**
- After 50-100 trials, converged to similar configurations
- Manual tuning based on experience was sufficient
- Diminishing returns for this problem

## How to Use This Code

### 1. Ultimate Pipeline (Best Phase 1 Result)

```bash
cd Phase1_Alternative_Methods
python ultimate_pipeline.py
```

**Outputs:**
- `testLabel_ultimate.txt` (5-column format)
- Console output with CV results and metrics
- Training time: ~30 minutes

### 2. XGBoost Baseline

```bash
python fast_xgboost.py
```

**Outputs:**
- `testLabel_final.txt`
- Training time: ~5 minutes
- Expected accuracy: ~90%

### 3. Experimental Pipelines

```bash
# SMOTE attempt (will show 87.36% - demonstrates failure)
python optimized_pipeline.py

# Pseudo-labeling
python pseudo_labeling_pipeline.py

# Stacking ensemble
python stacking_ensemble_pipeline.py
```

## Key Insights from Phase 1

### Technical Learnings

1. **Feature Engineering on Noisy Data**
   - Creating polynomial/interaction features can amplify noise
   - Regularization alone may be better than feature engineering
   - Let the model's built-in mechanisms handle feature selection

2. **Ensemble Size Trade-offs**
   - Large ensembles (18 models) needed to average out overfitting
   - Better to fix base model robustness than scale ensemble
   - Diminishing returns beyond 8-10 diverse models

3. **Cross-Validation Metrics**
   - Low CV variance (±0.09%) doesn't guarantee best test performance
   - Generalization gap (CV→test) more informative than CV variance
   - Phase 1 had stable CV but lower test generalization (+0.42% vs +1.75%)

4. **Class Imbalance Handling**
   - Simple class weighting (`is_unbalance=True`) often sufficient
   - Complex resampling (SMOTE) can hurt in high dimensions
   - Don't over-engineer solutions to standard problems

### Strategic Learnings

1. **Read Problem Statement Carefully**
   - "Embedded noise" hint was crucial but initially overlooked
   - Understanding data characteristics more important than technique sophistication

2. **Simpler Can Be Better**
   - 92.82% (simple, 411 features, 1 step) beat 92.00% (complex, 407 features, 4 steps)
   - Occam's Razor applies in machine learning

3. **Failed Experiments Have Value**
   - Each failure taught us what doesn't work and why
   - SMOTE failure led to noise-robust thinking
   - Documented failures help future work

4. **Iterate and Learn**
   - 10+ different approaches tested over 3 weeks
   - Each iteration built on previous insights
   - Systematic experimentation led to understanding

## Dependencies

```bash
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn imbalanced-learn optuna
```

**Versions:**
- Python: 3.8+
- numpy: 1.21+
- pandas: 1.3+
- scikit-learn: 1.0+
- lightgbm: 3.3+
- xgboost: 1.5+
- imbalanced-learn: 0.8+ (for SMOTE experiments)
- optuna: 2.10+ (for Bayesian optimization)

## References

All Phase 1 work used standard libraries and documented APIs. No external code snippets were used. Key references:

1. **LightGBM Documentation:** https://lightgbm.readthedocs.io/
2. **Scikit-learn User Guide:** https://scikit-learn.org/stable/user_guide.html
3. **XGBoost Documentation:** https://xgboost.readthedocs.io/

## Comparison with Phase 2 (Final Submission)

| Aspect | Phase 1 (This Folder) | Phase 2 (Final) |
|--------|----------------------|-----------------|
| Test Accuracy | 92.00% | **92.82%** ✓ |
| CV Accuracy | 91.58% ± 0.09% | 91.07% ± 0.64% |
| Generalization | +0.42% (CV→test) | **+1.75%** ✓ |
| Feature Count | 407 (engineered) | 411 (original) |
| Preprocessing | 4 steps | 1 step |
| Ensemble Size | 18 models | 8 models |
| Training Time | ~30 min | ~15 min |
| Philosophy | Extract signal from noise | Resist noise through regularization |

## Conclusion

Phase 1 demonstrated that complex feature engineering and large ensembles can achieve competitive performance (92%). However, the simpler Phase 2 noise-robust approach achieved superior test accuracy (92.82%) with half the complexity.

**Key Takeaway:** When data has embedded noise, it's better to resist the noise through robust hyperparameters and regularization rather than trying to extract more signal through feature engineering.

This exploratory work was essential for understanding the problem and ultimately led to the insights that guided Phase 2's successful approach.

---

**Author:** Brandon Newton
**Date:** November 11, 2025
**Part of:** HW3 Supervised Learning Project - CSC 621
