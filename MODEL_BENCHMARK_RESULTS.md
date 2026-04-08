# ESG Greenwashing Detection -- Model Benchmark Results

## Experimental Setup

- **Dataset:** ESG Feature Matrix (480 companies x 205 engineered features)
- **Task:** Binary classification (greenwashing vs genuine)
- **Class balance:** 335 negative / 145 positive (30.2% positive)
- **Split:** 60/20/20 train/val/test, stratified, random_state=42
- **Train size:** 288 | **Validation size:** 96 | **Test size:** 96
- **Preprocessing:** StandardScaler fit on training set only
- **Decision threshold:** 0.5

## Models Evaluated

| # | Model | Type | Training time (s) |
|---|-------|------|---|
| 1 | Logistic Regression | Linear | 0.01 |
| 2 | SVM (RBF) | Kernel SVM | 0.02 |
| 3 | KNN (k=5) | Instance-based | 0.00 |
| 4 | Decision Tree | Tree | 0.01 |
| 5 | Random Forest | Bagging ensemble | 0.18 |
| 6 | Gradient Boosting | Boosting ensemble | 1.37 |
| 7 | XGBoost | Boosting ensemble | 2.02 |
| 8 | CatBoost | Boosting ensemble | 2.56 |
| 9 | Hybrid XGB-CatBoost (opt) | Optimized blend (alpha=0.20) | - |
| 10 | Hybrid XGB-CatBoost (50/50) | Equal blend | - |
| 11 | Voting (RF+XGB+Cat) | Soft voting ensemble | - |

## Training Set: 5-Fold Cross-Validation Scores

Plain training-set metrics are saturated (~1.0000) for tree models because they memorize the 288 samples. The 5-fold CV scores below are computed by training on 4/5 of the training data and validating on the held-out fold, repeated 5 times. This is the **true generalization estimate during training**, computed entirely within the training set (no leakage from val or test).

| Rank | Model | CV Accuracy (mean ± std) | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) |
|------|-------|--------------------------|--------------------|-------------------------|
| 1 | CatBoost | 0.9619 ± 0.0275 | 0.9305 ± 0.0522 | 0.9944 ± 0.0098 |
| 2 | XGBoost | 0.9619 ± 0.0275 | 0.9305 ± 0.0522 | 0.9941 ± 0.0071 |
| 3 | Random Forest | 0.9411 ± 0.0319 | 0.8936 ± 0.0655 | 0.9792 ± 0.0215 |
| 4 | SVM (RBF) | 0.8857 ± 0.0412 | 0.7877 ± 0.0923 | 0.9543 ± 0.0248 |
| 5 | Logistic Regression | 0.8786 ± 0.0303 | 0.8065 ± 0.0468 | 0.9483 ± 0.0252 |
| 6 | Gradient Boosting | 0.9446 ± 0.0480 | 0.9001 ± 0.0879 | 0.9420 ± 0.0658 |
| 7 | Decision Tree | 0.9135 ± 0.0447 | 0.8557 ± 0.0749 | 0.9013 ± 0.0641 |
| 8 | KNN (k=5) | 0.8541 ± 0.0359 | 0.7177 ± 0.0918 | 0.9005 ± 0.0290 |

## Training Set Results -- Full Fit (memorization) (n = 288)

*Note: tree-based ensembles reach 1.0000 because they perfectly memorize the 288 training samples. The 5-fold CV scores above are the meaningful training metric for these models.*

| Rank | Model | Accuracy | Bal Acc | Precision | Recall | F1 | ROC-AUC | Log Loss | MCC |
|------|-------|---------|---------|-----------|--------|-----|---------|---------|-----|
| 1 | Logistic Regression | 0.9965 | 0.9943 | 1.0000 | 0.9885 | 0.9942 | 1.0000 | 0.0450 | 0.9918 |
| 2 | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0411 | 1.0000 |
| 3 | Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | CatBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0023 | 1.0000 |
| 5 | Hybrid XGB-CatBoost (opt) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0042 | 1.0000 |
| 6 | Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 7 | XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0119 | 1.0000 |
| 8 | Hybrid XGB-CatBoost (50/50) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0070 | 1.0000 |
| 9 | Voting (RF+XGB+Cat) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0178 | 1.0000 |
| 10 | SVM (RBF) | 0.9826 | 0.9745 | 0.9881 | 0.9540 | 0.9708 | 0.9967 | 0.0809 | 0.9587 |
| 11 | KNN (k=5) | 0.9028 | 0.8586 | 0.9155 | 0.7471 | 0.8228 | 0.9635 | 0.2259 | 0.7642 |

## Validation Set Results (n = 96)

| Rank | Model | Accuracy | Bal Acc | Precision | Recall | F1 | ROC-AUC | Log Loss | MCC |
|------|-------|---------|---------|-----------|--------|-----|---------|---------|-----|
| 1 | Hybrid XGB-CatBoost (opt) | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9943 | 0.1088 | 0.9012 |
| 2 | CatBoost | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9943 | 0.1098 | 0.9012 |
| 3 | Hybrid XGB-CatBoost (50/50) | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9923 | 0.1105 | 0.9012 |
| 4 | XGBoost | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9913 | 0.1205 | 0.9012 |
| 5 | Voting (RF+XGB+Cat) | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9882 | 0.1204 | 0.9012 |
| 6 | Gradient Boosting | 0.9583 | 0.9604 | 0.9032 | 0.9655 | 0.9333 | 0.9861 | 0.3055 | 0.9041 |
| 7 | Random Forest | 0.9583 | 0.9506 | 0.9310 | 0.9310 | 0.9310 | 0.9779 | 0.1574 | 0.9012 |
| 8 | SVM (RBF) | 0.8646 | 0.8541 | 0.7500 | 0.8276 | 0.7869 | 0.9578 | 0.2515 | 0.6898 |
| 9 | Logistic Regression | 0.8854 | 0.8592 | 0.8214 | 0.7931 | 0.8070 | 0.9187 | 0.5285 | 0.7258 |
| 10 | Decision Tree | 0.9062 | 0.9035 | 0.8125 | 0.8966 | 0.8525 | 0.9035 | 1.5111 | 0.7860 |
| 11 | KNN (k=5) | 0.8750 | 0.8420 | 0.8148 | 0.7586 | 0.7857 | 0.8870 | 0.9209 | 0.6985 |

## Test Set Results (n = 96)

| Rank | Model | Accuracy | Bal Acc | Precision | Recall | F1 | ROC-AUC | Log Loss | MCC |
|------|-------|---------|---------|-----------|--------|-----|---------|---------|-----|
| 1 | CatBoost | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9871 | 0.2545 | 0.7994 |
| 2 | Hybrid XGB-CatBoost (opt) | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9840 | 0.2445 | 0.7994 |
| 3 | Hybrid XGB-CatBoost (50/50) | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9815 | 0.2404 | 0.7994 |
| 4 | XGBoost | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9738 | 0.2494 | 0.7994 |
| 5 | Voting (RF+XGB+Cat) | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9691 | 0.2385 | 0.7994 |
| 6 | Gradient Boosting | 0.9062 | 0.8839 | 0.8571 | 0.8276 | 0.8421 | 0.9655 | 0.6687 | 0.7757 |
| 7 | Random Forest | 0.9167 | 0.8914 | 0.8889 | 0.8276 | 0.8571 | 0.9377 | 0.2711 | 0.7994 |
| 8 | Logistic Regression | 0.8542 | 0.8173 | 0.7778 | 0.7241 | 0.7500 | 0.9063 | 0.5380 | 0.6481 |
| 9 | SVM (RBF) | 0.8646 | 0.8248 | 0.8077 | 0.7241 | 0.7636 | 0.8811 | 0.4083 | 0.6711 |
| 10 | Decision Tree | 0.8958 | 0.8765 | 0.8276 | 0.8276 | 0.8276 | 0.8765 | 1.6790 | 0.7530 |
| 11 | KNN (k=5) | 0.7917 | 0.7138 | 0.7143 | 0.5172 | 0.6000 | 0.8384 | 0.9462 | 0.4750 |

## Test Set Confusion Matrices

| Model | TN | FP | FN | TP |
|-------|-----|-----|-----|-----|
| CatBoost | 64 | 3 | 5 | 24 |
| Hybrid XGB-CatBoost (opt) | 64 | 3 | 5 | 24 |
| Hybrid XGB-CatBoost (50/50) | 64 | 3 | 5 | 24 |
| XGBoost | 64 | 3 | 5 | 24 |
| Voting (RF+XGB+Cat) | 64 | 3 | 5 | 24 |
| Gradient Boosting | 63 | 4 | 5 | 24 |
| Random Forest | 64 | 3 | 5 | 24 |
| Logistic Regression | 61 | 6 | 8 | 21 |
| SVM (RBF) | 62 | 5 | 8 | 21 |
| Decision Tree | 62 | 5 | 5 | 24 |
| KNN (k=5) | 61 | 6 | 14 | 15 |

## Best Models on Test Set

| Metric | Best Model | Value |
|--------|-----------|-------|
| ROC-AUC | CatBoost | 0.9871 |
| F1 Score | Random Forest | 0.8571 |
| Accuracy | Random Forest | 0.9167 |
| Log Loss (lower=better) | Voting (RF+XGB+Cat) | 0.2385 |

## Overfitting Analysis (Train F1 - Test F1)

| Model | Train F1 | Test F1 | Gap | Assessment |
|-------|---------|---------|-----|------------|
| CatBoost | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| Hybrid XGB-CatBoost (50/50) | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| Hybrid XGB-CatBoost (opt) | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| Random Forest | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| Voting (RF+XGB+Cat) | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| XGBoost | 1.0000 | 0.8571 | +0.1429 | Mild overfit |
| Gradient Boosting | 1.0000 | 0.8421 | +0.1579 | Overfitting |
| Decision Tree | 1.0000 | 0.8276 | +0.1724 | Overfitting |
| SVM (RBF) | 0.9708 | 0.7636 | +0.2071 | Overfitting |
| KNN (k=5) | 0.8228 | 0.6000 | +0.2228 | Overfitting |
| Logistic Regression | 0.9942 | 0.7500 | +0.2442 | Overfitting |

## Final Recommendation

**Best model for production: `CatBoost`**

- Highest test ROC-AUC: **0.9871**
- Test set: Accuracy = 0.9167, F1 = 0.8571, MCC = 0.7994
- Confusion matrix: TN=64, FP=3, FN=5, TP=24
- Generalization gap (Train F1 - Test F1): +0.1429 (mild overfit, acceptable)

Tree-based ensemble methods (CatBoost, XGBoost, Random Forest, hybrid blends) all cluster at the top with similar test performance. The **Hybrid XGBoost-CatBoost** adds the benefit of ensemble diversity and **Monte Carlo uncertainty quantification** (95% confidence intervals per company) for production risk decisions.

## Files Generated

| File | Description |
|------|-------------|
| `data/processed/all_models_train_test_results.csv` | Full results table (long format) |
| `data/processed/all_models_summary.csv` | Test-set summary ranked by ROC-AUC |
| `data/processed/all_models_results.json` | Complete metadata with all metrics |
| `data/processed/hybrid_predictions.csv` | Per-company hybrid predictions + 95% CI |
| `data/processed/hybrid_model_metrics.json` | Hybrid model configuration |
| `MODEL_BENCHMARK_RESULTS.md` | This report |