# Phase 1 Code Files - Quick Reference Guide

This folder contains all exploratory work and alternative methods developed during Phase 1. Files are categorized by importance and purpose.

---

## 🔴 **ESSENTIAL FILES** (Recommend keeping)

### Core Phase 1 Implementation
1. **`ultimate_pipeline.py`** ⭐ **MOST IMPORTANT**
   - Phase 1's best implementation achieving 92% test accuracy
   - Features: MI selection, polynomial features, interactions, 18-model ensemble
   - ~500 lines, well-commented
   - **USE THIS to reproduce Phase 1 results**

2. **`fast_xgboost.py`**
   - XGBoost baseline achieving 90.01% accuracy
   - Quick baseline implementation (~150 lines)
   - Good reference for simple gradient boosting approach

3. **`hypertuned_pipeline.py`**
   - LightGBM with 4 different hyperparameter configurations
   - Achieved 90.42% with systematic tuning
   - Shows progression from baseline to optimized

4. **`explore_data.py`**
   - Exploratory Data Analysis script
   - Generates visualizations and statistics
   - Documents dataset characteristics

---

## 🟡 **EXPERIMENTAL FILES** (Valuable for learning)

### Failed but Educational Experiments

5. **`optimized_pipeline.py`** - SMOTE Experiment
   - **Result:** 87.36% (FAILED - dropped from 90%)
   - **Lesson:** SMOTE hurts performance on high-dimensional noisy data
   - Shows what NOT to do with this dataset

6. **`pseudo_labeling_pipeline.py`** - Semi-supervised Learning
   - **Result:** ~90-91% (No improvement)
   - **Lesson:** Pseudo-labeling doesn't help when training data is sufficient
   - Risk of error propagation with noisy data

7. **`stacking_ensemble_pipeline.py`** - Meta-learner Ensemble
   - **Result:** ~91% (Similar to single model)
   - **Lesson:** Tree models too correlated for stacking benefit
   - Computational cost not justified

8. **`optuna_tuning_pipeline.py`** - Bayesian Optimization
   - **Result:** ~91% (Confirmed manual tuning was good)
   - **Lesson:** Manual tuning sufficient for this problem
   - 50-100 trials converged to similar configs

9. **`partner_inspired_pipeline.py`**
   - Noise-robust exploration (transitional work to Phase 2)
   - Achieved ~90.84%
   - Bridge between Phase 1 and Phase 2 thinking

---

## 🟢 **SUPPORTING FILES** (Can be removed if needed)

### Incremental Development Files

10. **`classification_pipeline.py`**
    - Early baseline implementation
    - Random Forest + basic preprocessing
    - Achieved ~85-87%
    - Superseded by later files

11. **`advanced_ensemble.py`**
    - Experimental ensemble strategies
    - Various averaging and voting schemes
    - Research code, not production

12. **`quick_test.py`**
    - Rapid prototyping and testing
    - Debug/scratch code
    - Can be removed

13. **`final_pipeline.py`**, **`final_best_pipeline.py`**, **`final_comprehensive_pipeline.py`**
    - Incremental iterations
    - Superseded by `ultimate_pipeline.py`
    - Can be removed (same functionality in ultimate_pipeline)

14. **`generate_blind_predictions.py`**
    - Early version of blind prediction script
    - Superseded by `generate_blind_predictions_final.py`

15. **`generate_blind_predictions_final.py`**
    - Final version for Phase 1 blind predictions
    - Keep if you want Phase 1 blind predictions separate from Phase 2

16. **`weighted_ensemble_optimizer.py`**
    - Experimental weight optimization for ensemble
    - Research code exploring weight combinations

---

## 📋 **Recommended Minimal Set** (If space is limited)

If you need to reduce the number of files, keep only these **5 essential files**:

1. ✅ `ultimate_pipeline.py` - Best Phase 1 result (92%)
2. ✅ `fast_xgboost.py` - XGBoost baseline (90%)
3. ✅ `explore_data.py` - EDA script
4. ✅ `optimized_pipeline.py` - SMOTE failure (demonstrates what not to do)
5. ✅ `README.md` - This guide

**Why these 5?**
- Shows progression: EDA → baseline → best result
- Includes one important failure (SMOTE) for learning
- Demonstrates the full story without clutter

---

## 🗑️ **Can Be Safely Removed** (If needed)

These files are incremental iterations or debug code:
- `classification_pipeline.py`
- `advanced_ensemble.py`
- `quick_test.py`
- `final_pipeline.py`
- `final_best_pipeline.py`
- `final_comprehensive_pipeline.py`
- `generate_blind_predictions.py` (early version)
- `weighted_ensemble_optimizer.py`

**Note:** Removing these will not affect the story or reproducibility, as their functionality is captured in the essential files.

---

## 📊 **Results Summary**

| File | Test Accuracy | Purpose |
|------|---------------|---------|
| `ultimate_pipeline.py` | **92.00%** | ⭐ Phase 1 best |
| `hypertuned_pipeline.py` | 90.42% | Systematic tuning |
| `fast_xgboost.py` | 90.01% | XGBoost baseline |
| `partner_inspired_pipeline.py` | 90.84% | Noise-robust |
| `optimized_pipeline.py` | 87.36% | ⚠️ SMOTE failure |
| `pseudo_labeling_pipeline.py` | ~91% | No improvement |
| `stacking_ensemble_pipeline.py` | ~91% | No improvement |
| `classification_pipeline.py` | ~85-87% | Early baseline |

---

## 🎯 **For the Report**

The final report references these key files:
- **Section 3:** Phase 1 approaches (ultimate_pipeline.py)
- **Section 3.2:** XGBoost baseline (fast_xgboost.py)
- **Section 3.5:** Failed experiments (optimized_pipeline.py, pseudo_labeling, stacking, optuna)

All files support the narrative documented in the report's Phase 1 section.

---

## 🔧 **How to Use Essential Files**

### Reproduce Phase 1 Best Result (92%)
```bash
python ultimate_pipeline.py
# Outputs: testLabel_ultimate.txt (13,082 rows)
# Time: ~30 minutes
```

### Run XGBoost Baseline (90%)
```bash
python fast_xgboost.py
# Outputs: testLabel_final.txt
# Time: ~5 minutes
```

### Run EDA
```bash
python explore_data.py
# Outputs: eda_results.png and statistics
# Time: ~30 seconds
```

### Demonstrate SMOTE Failure
```bash
python optimized_pipeline.py
# Outputs: testLabel_optimized.txt (87.36% - shows failure)
# Time: ~20 minutes
```

---

**Last Updated:** November 11, 2025
**Author:** Brandon Newton
**Project:** HW3 Supervised Learning - Phase 1 Exploratory Work
