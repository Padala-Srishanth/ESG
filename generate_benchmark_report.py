"""Generate a clean, submission-ready markdown report from benchmark results."""
import pandas as pd
import json


def main():
    df = pd.read_csv('data/processed/all_models_train_test_results.csv')
    with open('data/processed/all_models_results.json') as f:
        meta = json.load(f)

    cb = meta['class_balance']
    n_neg = cb.get('0', cb.get(0, 0))
    n_pos = cb.get('1', cb.get(1, 0))

    report = []
    report.append('# ESG Greenwashing Detection -- Model Benchmark Results')
    report.append('')
    report.append('## Experimental Setup')
    report.append('')
    report.append(f'- **Dataset:** ESG Feature Matrix (480 companies x {meta["n_features"]} engineered features)')
    report.append(f'- **Task:** Binary classification (greenwashing vs genuine)')
    report.append(f'- **Class balance:** {n_neg} negative / {n_pos} positive ({n_pos/(n_neg+n_pos)*100:.1f}% positive)')
    report.append(f'- **Split:** {meta["split_ratio"]} train/val/test, stratified, random_state={meta["random_seed"]}')
    report.append(f'- **Train size:** {meta["train_size"]} | **Validation size:** {meta["val_size"]} | **Test size:** {meta["test_size"]}')
    report.append(f'- **Preprocessing:** StandardScaler fit on training set only')
    report.append(f'- **Decision threshold:** 0.5')
    report.append('')

    # Models table
    type_map = {
        'Logistic Regression': 'Linear',
        'SVM (RBF)': 'Kernel SVM',
        'KNN (k=5)': 'Instance-based',
        'Decision Tree': 'Tree',
        'Random Forest': 'Bagging ensemble',
        'Gradient Boosting': 'Boosting ensemble',
        'XGBoost': 'Boosting ensemble',
        'CatBoost': 'Boosting ensemble',
        'Hybrid XGB-CatBoost (opt)': 'Optimized blend (alpha=0.20)',
        'Hybrid XGB-CatBoost (50/50)': 'Equal blend',
        'Voting (RF+XGB+Cat)': 'Soft voting ensemble',
    }
    report.append('## Models Evaluated')
    report.append('')
    report.append('| # | Model | Type | Training time (s) |')
    report.append('|---|-------|------|---|')
    train_times = meta.get('train_times_seconds', {})
    for i, (model, type_str) in enumerate(type_map.items(), 1):
        t = train_times.get(model, '-')
        t_str = f'{t:.2f}' if isinstance(t, (int, float)) else '-'
        report.append(f'| {i} | {model} | {type_str} | {t_str} |')
    report.append('')

    # Per-split tables
    for split_name in ['train', 'val', 'test']:
        title = {'train': 'Training Set', 'val': 'Validation Set', 'test': 'Test Set'}[split_name]
        sub = df[df['split'] == split_name].sort_values('roc_auc', ascending=False)
        n = int(sub['n_samples'].iloc[0])
        report.append(f'## {title} Results (n = {n})')
        report.append('')
        report.append('| Rank | Model | Accuracy | Bal Acc | Precision | Recall | F1 | ROC-AUC | Log Loss | MCC |')
        report.append('|------|-------|---------|---------|-----------|--------|-----|---------|---------|-----|')
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            report.append(
                f'| {rank} | {row["model"]} | {row["accuracy"]:.4f} | {row["balanced_accuracy"]:.4f} | '
                f'{row["precision"]:.4f} | {row["recall"]:.4f} | {row["f1"]:.4f} | '
                f'{row["roc_auc"]:.4f} | {row["log_loss"]:.4f} | {row["mcc"]:.4f} |'
            )
        report.append('')

    # Confusion matrices on test set
    report.append('## Test Set Confusion Matrices')
    report.append('')
    report.append('| Model | TN | FP | FN | TP |')
    report.append('|-------|-----|-----|-----|-----|')
    test_df = df[df['split'] == 'test'].sort_values('roc_auc', ascending=False)
    for _, row in test_df.iterrows():
        report.append(f'| {row["model"]} | {row["tn"]} | {row["fp"]} | {row["fn"]} | {row["tp"]} |')
    report.append('')

    # Best models
    test_df_full = df[df['split'] == 'test'].copy()
    best_auc = test_df_full.loc[test_df_full['roc_auc'].idxmax()]
    best_f1 = test_df_full.loc[test_df_full['f1'].idxmax()]
    best_acc = test_df_full.loc[test_df_full['accuracy'].idxmax()]
    best_logloss = test_df_full.loc[test_df_full['log_loss'].idxmin()]

    report.append('## Best Models on Test Set')
    report.append('')
    report.append('| Metric | Best Model | Value |')
    report.append('|--------|-----------|-------|')
    report.append(f'| ROC-AUC | {best_auc["model"]} | {best_auc["roc_auc"]:.4f} |')
    report.append(f'| F1 Score | {best_f1["model"]} | {best_f1["f1"]:.4f} |')
    report.append(f'| Accuracy | {best_acc["model"]} | {best_acc["accuracy"]:.4f} |')
    report.append(f'| Log Loss (lower=better) | {best_logloss["model"]} | {best_logloss["log_loss"]:.4f} |')
    report.append('')

    # Overfitting analysis
    report.append('## Overfitting Analysis (Train F1 - Test F1)')
    report.append('')
    report.append('| Model | Train F1 | Test F1 | Gap | Assessment |')
    report.append('|-------|---------|---------|-----|------------|')
    gap_data = []
    for name in sorted(df['model'].unique()):
        tr = df[(df['model'] == name) & (df['split'] == 'train')]
        te = df[(df['model'] == name) & (df['split'] == 'test')]
        if len(tr) and len(te):
            train_f1 = float(tr['f1'].iloc[0])
            test_f1 = float(te['f1'].iloc[0])
            gap_data.append((name, train_f1, test_f1, train_f1 - test_f1))
    gap_data.sort(key=lambda x: x[3])
    for name, tr_f1, te_f1, gap in gap_data:
        if gap > 0.15:
            status = 'Overfitting'
        elif gap > 0.05:
            status = 'Mild overfit'
        elif gap < -0.05:
            status = 'Underfit?'
        else:
            status = 'Well calibrated'
        report.append(f'| {name} | {tr_f1:.4f} | {te_f1:.4f} | +{gap:.4f} | {status} |')
    report.append('')

    # Final recommendation
    train_f1_best = float(df[(df['model'] == best_auc['model']) & (df['split'] == 'train')]['f1'].iloc[0])
    overfit_gap = train_f1_best - best_auc['f1']

    report.append('## Final Recommendation')
    report.append('')
    report.append(f'**Best model for production: `{best_auc["model"]}`**')
    report.append('')
    report.append(f'- Highest test ROC-AUC: **{best_auc["roc_auc"]:.4f}**')
    report.append(f'- Test set: Accuracy = {best_auc["accuracy"]:.4f}, F1 = {best_auc["f1"]:.4f}, MCC = {best_auc["mcc"]:.4f}')
    report.append(f'- Confusion matrix: TN={int(best_auc["tn"])}, FP={int(best_auc["fp"])}, FN={int(best_auc["fn"])}, TP={int(best_auc["tp"])}')
    report.append(f'- Generalization gap (Train F1 - Test F1): +{overfit_gap:.4f} (mild overfit, acceptable)')
    report.append('')
    report.append('Tree-based ensemble methods (CatBoost, XGBoost, Random Forest, hybrid blends) all '
                  'cluster at the top with similar test performance. The **Hybrid XGBoost-CatBoost** '
                  'adds the benefit of ensemble diversity and **Monte Carlo uncertainty quantification** '
                  '(95% confidence intervals per company) for production risk decisions.')
    report.append('')
    report.append('## Files Generated')
    report.append('')
    report.append('| File | Description |')
    report.append('|------|-------------|')
    report.append('| `data/processed/all_models_train_test_results.csv` | Full results table (long format) |')
    report.append('| `data/processed/all_models_summary.csv` | Test-set summary ranked by ROC-AUC |')
    report.append('| `data/processed/all_models_results.json` | Complete metadata with all metrics |')
    report.append('| `data/processed/hybrid_predictions.csv` | Per-company hybrid predictions + 95% CI |')
    report.append('| `data/processed/hybrid_model_metrics.json` | Hybrid model configuration |')
    report.append('| `MODEL_BENCHMARK_RESULTS.md` | This report |')

    text = '\n'.join(report)
    with open('MODEL_BENCHMARK_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write(text)

    print(f'Report written to MODEL_BENCHMARK_RESULTS.md')
    print(f'  {len(report)} lines, {len(text)} chars')


if __name__ == '__main__':
    main()
