"""
================================================================================
COMPREHENSIVE MODEL BENCHMARK -- ALL MODELS
================================================================================
Trains and evaluates ALL classifiers on the same train/val/test split for the
ESG greenwashing detection task. Reports train + validation + test metrics
for direct comparison and submission.

Models benchmarked:
    1. Logistic Regression
    2. Support Vector Machine (RBF)
    3. K-Nearest Neighbors
    4. Decision Tree
    5. Random Forest
    6. Gradient Boosting (sklearn)
    7. XGBoost
    8. CatBoost
    9. LightGBM (if installed)
    10. Hybrid XGBoost-CatBoost (optimal blend)
    11. Hybrid Ensemble (Voting)

Metrics reported per (model, split):
    - Accuracy
    - Balanced Accuracy
    - Precision
    - Recall
    - F1 score
    - ROC-AUC
    - Log Loss
    - Matthews Correlation Coefficient
    - Confusion matrix counts (TN, FP, FN, TP)

Outputs:
    - data/processed/all_models_train_test_results.csv  (long format)
    - data/processed/all_models_summary.csv               (wide test-set summary)
    - data/processed/all_models_results.json              (full metadata)
================================================================================
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')


def build_proxy_labels(fm):
    """Build the same 5-indicator proxy labels as model_training.py."""
    ind1 = (fm['esg_controversy_divergence'] > fm['esg_controversy_divergence'].quantile(0.75)).astype(int)
    ind2 = (fm['greenwashing_signal_score'] > fm['greenwashing_signal_score'].quantile(0.75)).astype(int)
    ind3 = fm.get('risk_controversy_mismatch', pd.Series([0] * len(fm))).fillna(0).astype(int)
    ind4 = (fm['controversy_risk_ratio'] > fm['controversy_risk_ratio'].quantile(0.75)).astype(int)
    ind5 = (fm['combined_anomaly_score'] > fm['combined_anomaly_score'].quantile(0.75)).astype(int)
    return ((ind1 + ind2 + ind3 + ind4 + ind5) >= 2).astype(int).values


def prepare_features(fm):
    """Extract numeric features, excluding labels and IDs."""
    exclude = {
        'gw_indicator_1', 'gw_indicator_2', 'gw_indicator_3', 'gw_indicator_4', 'gw_indicator_5',
        'gw_proxy_score', 'gw_label_binary', 'gw_label_multiclass',
        'symbol', 'company_name', 'sector', 'industry', 'description', 'source',
        'esg_controversy_segment', 'sector_risk_segment',
        'ESG_Risk_Level', 'Controversy_Level',
    }
    feature_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = fm[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return X, feature_cols


def evaluate(name, split, y_true, y_pred, y_prob):
    """Compute all metrics for one (model, split) combination."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        log_loss, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'model': name,
        'split': split,
        'n_samples': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def find_optimal_blend_weight(p_xgb, p_cat, y_true):
    """Grid search optimal alpha for hybrid XGBoost-CatBoost blend."""
    from sklearn.metrics import log_loss
    best_alpha, best_loss = 0.5, float('inf')
    for alpha in np.arange(0.0, 1.01, 0.01):
        p = np.clip(alpha * p_xgb + (1 - alpha) * p_cat, 1e-7, 1 - 1e-7)
        loss = log_loss(y_true, p)
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    return best_alpha, float(best_loss)


def main():
    print('=' * 80)
    print('  COMPREHENSIVE MODEL BENCHMARK -- ALL MODELS')
    print('=' * 80)

    start = time.time()

    # === Load data ===
    print('\n[1] Loading feature matrix...')
    fm = pd.read_csv('data/processed/feature_matrix.csv')
    print(f'  Shape: {fm.shape}')

    y = build_proxy_labels(fm)
    X, feature_cols = prepare_features(fm)
    print(f'  Features: {len(feature_cols)}')
    print(f'  Class balance: {dict(pd.Series(y).value_counts())}')

    # === Split data 60/20/20 ===
    print('\n[2] Splitting train/val/test = 60/20/20...')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # === Define all models ===
    print('\n[3] Initializing models...')
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb
    from catboost import CatBoostClassifier

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        'SVM (RBF)': SVC(probability=True, random_state=42, kernel='rbf', C=1.0),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1,
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            random_seed=42, loss_function='Logloss', verbose=False,
        ),
    }

    # Try LightGBM if available
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
    except ImportError:
        print('  (LightGBM not installed -- skipping)')

    print(f'  {len(models)} base models ready')

    # === Train all models ===
    print('\n[4] Training all models...')
    trained_models = {}
    train_times = {}
    for name, model in models.items():
        t0 = time.time()
        try:
            if name == 'CatBoost':
                model.fit(X_train_s, y_train, verbose=False)
            else:
                model.fit(X_train_s, y_train)
            trained_models[name] = model
            train_times[name] = time.time() - t0
            print(f'  [+] {name:<22s}  trained in {train_times[name]:.2f}s')
        except Exception as e:
            print(f'  [!] {name:<22s}  FAILED: {e}')

    # === Evaluate on all 3 splits ===
    print('\n[5] Evaluating on Train / Validation / Test...')
    all_results = []

    splits = {
        'train': (X_train_s, y_train),
        'val':   (X_val_s,   y_val),
        'test':  (X_test_s,  y_test),
    }

    for name, model in trained_models.items():
        for split_name, (X_split, y_split) in splits.items():
            try:
                p = model.predict_proba(X_split)[:, 1]
                pred = (p >= 0.5).astype(int)
                all_results.append(evaluate(name, split_name, y_split, pred, p))
            except Exception as e:
                print(f'  [!] {name} on {split_name}: {e}')

    # === Hybrid XGBoost-CatBoost (optimized blend) ===
    print('\n[6] Building Hybrid XGBoost-CatBoost (optimized blend)...')
    if 'XGBoost' in trained_models and 'CatBoost' in trained_models:
        xgb_model = trained_models['XGBoost']
        cat_model = trained_models['CatBoost']

        # Find optimal alpha on validation set
        p_xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
        p_cat_val = cat_model.predict_proba(X_val_s)[:, 1]
        alpha, val_loss = find_optimal_blend_weight(p_xgb_val, p_cat_val, y_val)
        print(f'  Optimal alpha: {alpha:.3f} (XGBoost weight)')
        print(f'  Validation log-loss: {val_loss:.4f}')

        for split_name, (X_split, y_split) in splits.items():
            p_x = xgb_model.predict_proba(X_split)[:, 1]
            p_c = cat_model.predict_proba(X_split)[:, 1]
            p_h = alpha * p_x + (1 - alpha) * p_c
            pred = (p_h >= 0.5).astype(int)
            all_results.append(evaluate('Hybrid XGB-CatBoost (opt)', split_name, y_split, pred, p_h))

        # Also include a fixed 50/50 hybrid for comparison
        for split_name, (X_split, y_split) in splits.items():
            p_x = xgb_model.predict_proba(X_split)[:, 1]
            p_c = cat_model.predict_proba(X_split)[:, 1]
            p_h = 0.5 * p_x + 0.5 * p_c
            pred = (p_h >= 0.5).astype(int)
            all_results.append(evaluate('Hybrid XGB-CatBoost (50/50)', split_name, y_split, pred, p_h))

    # === Cross-validation training scores (5-fold) ===
    # Plain train metrics are saturated (all ~1.0 because of memorization).
    # 5-fold CV on the training set gives meaningful generalization estimates
    # WITHOUT touching the held-out validation or test sets.
    print('\n[6.5] Computing 5-fold cross-validation scores on training set...')
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in trained_models.items():
        try:
            cv_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_f1 = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
            cv_auc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_results[name] = {
                'cv_accuracy_mean': float(cv_acc.mean()),
                'cv_accuracy_std': float(cv_acc.std()),
                'cv_f1_mean': float(cv_f1.mean()),
                'cv_f1_std': float(cv_f1.std()),
                'cv_roc_auc_mean': float(cv_auc.mean()),
                'cv_roc_auc_std': float(cv_auc.std()),
                'cv_folds': 5,
            }
            print(f'  [+] {name:<22s}  CV ROC-AUC = {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}')
        except Exception as e:
            print(f'  [!] {name}: CV failed -- {e}')

    # === Voting Ensemble (soft voting of top 3) ===
    print('\n[7] Building Soft Voting Ensemble (RF + XGB + CatBoost)...')
    if all(m in trained_models for m in ['Random Forest', 'XGBoost', 'CatBoost']):
        for split_name, (X_split, y_split) in splits.items():
            p_rf = trained_models['Random Forest'].predict_proba(X_split)[:, 1]
            p_xg = trained_models['XGBoost'].predict_proba(X_split)[:, 1]
            p_cb = trained_models['CatBoost'].predict_proba(X_split)[:, 1]
            p_vote = (p_rf + p_xg + p_cb) / 3
            pred = (p_vote >= 0.5).astype(int)
            all_results.append(evaluate('Voting (RF+XGB+Cat)', split_name, y_split, pred, p_vote))

    # === Save outputs ===
    print('\n[8] Saving results...')
    results_df = pd.DataFrame(all_results)
    col_order = ['model', 'split', 'n_samples', 'accuracy', 'balanced_accuracy',
                 'precision', 'recall', 'f1', 'roc_auc', 'log_loss', 'mcc',
                 'tn', 'fp', 'fn', 'tp']
    results_df = results_df[col_order]
    results_df.to_csv('data/processed/all_models_train_test_results.csv', index=False)
    print('  Saved data/processed/all_models_train_test_results.csv  '
          f'({len(results_df)} rows)')

    # === Wide format: test-set summary ===
    test_summary = results_df[results_df['split'] == 'test'].copy()
    test_summary = test_summary.sort_values('roc_auc', ascending=False).reset_index(drop=True)
    test_summary.insert(0, 'rank', range(1, len(test_summary) + 1))
    test_summary[['rank', 'model', 'accuracy', 'balanced_accuracy', 'precision',
                  'recall', 'f1', 'roc_auc', 'log_loss', 'mcc']].to_csv(
        'data/processed/all_models_summary.csv', index=False
    )
    print('  Saved data/processed/all_models_summary.csv')

    # === JSON metadata ===
    metadata = {
        'n_features': len(feature_cols),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'class_balance': {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        'random_seed': 42,
        'split_ratio': '60/20/20',
        'train_times_seconds': train_times,
        'cv_results': cv_results,
        'results': all_results,
    }
    with open('data/processed/all_models_results.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print('  Saved data/processed/all_models_results.json')

    # === CV results CSV ===
    if cv_results:
        cv_rows = []
        for name, res in cv_results.items():
            cv_rows.append({
                'model': name,
                'cv_accuracy_mean': res['cv_accuracy_mean'],
                'cv_accuracy_std': res['cv_accuracy_std'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std'],
                'cv_roc_auc_mean': res['cv_roc_auc_mean'],
                'cv_roc_auc_std': res['cv_roc_auc_std'],
            })
        cv_df = pd.DataFrame(cv_rows).sort_values('cv_roc_auc_mean', ascending=False)
        cv_df.to_csv('data/processed/all_models_cv_results.csv', index=False)
        print('  Saved data/processed/all_models_cv_results.csv')

    # === Print final tables ===
    print('\n' + '=' * 100)
    print('  FINAL RESULTS BY SPLIT')
    print('=' * 100)

    for split_name in ['train', 'val', 'test']:
        title = {'train': 'TRAIN SET', 'val': 'VALIDATION SET', 'test': 'TEST SET'}[split_name]
        rows = [r for r in all_results if r['split'] == split_name]
        n = rows[0]['n_samples'] if rows else 0
        print(f'\n  ===== {title} (n = {n}) =====')
        print(f'  {"Model":<28s}  {"Acc":>7s}  {"BalAcc":>7s}  {"Prec":>7s}  '
              f'{"Recall":>7s}  {"F1":>7s}  {"AUC":>7s}  {"LogLoss":>8s}  {"MCC":>7s}')
        print('  ' + '-' * 110)
        # Sort by ROC-AUC descending
        rows_sorted = sorted(rows, key=lambda r: -r['roc_auc'])
        for r in rows_sorted:
            print(f'  {r["model"]:<28s}  {r["accuracy"]:>7.4f}  {r["balanced_accuracy"]:>7.4f}  '
                  f'{r["precision"]:>7.4f}  {r["recall"]:>7.4f}  {r["f1"]:>7.4f}  '
                  f'{r["roc_auc"]:>7.4f}  {r["log_loss"]:>8.4f}  {r["mcc"]:>7.4f}')

    # === 5-fold CV training scores table ===
    if cv_results:
        print('\n  ===== 5-FOLD CV TRAINING SCORES (true generalization) =====')
        print(f'  {"Model":<28s}  {"CV Acc (mean+/-std)":>22s}  {"CV F1 (mean+/-std)":>22s}  {"CV AUC (mean+/-std)":>22s}')
        print('  ' + '-' * 100)
        cv_sorted = sorted(cv_results.items(), key=lambda kv: -kv[1]['cv_roc_auc_mean'])
        for name, r in cv_sorted:
            acc_str = f'{r["cv_accuracy_mean"]:.4f} +/- {r["cv_accuracy_std"]:.4f}'
            f1_str = f'{r["cv_f1_mean"]:.4f} +/- {r["cv_f1_std"]:.4f}'
            auc_str = f'{r["cv_roc_auc_mean"]:.4f} +/- {r["cv_roc_auc_std"]:.4f}'
            print(f'  {name:<28s}  {acc_str:>22s}  {f1_str:>22s}  {auc_str:>22s}')

    # === Identify best model on test set ===
    test_only = [r for r in all_results if r['split'] == 'test']
    best_by_auc = max(test_only, key=lambda r: r['roc_auc'])
    best_by_f1 = max(test_only, key=lambda r: r['f1'])
    best_by_acc = max(test_only, key=lambda r: r['accuracy'])

    print('\n  ===== BEST MODELS ON TEST SET =====')
    print(f'  Best ROC-AUC : {best_by_auc["model"]:<28s}  AUC = {best_by_auc["roc_auc"]:.4f}')
    print(f'  Best F1      : {best_by_f1["model"]:<28s}  F1  = {best_by_f1["f1"]:.4f}')
    print(f'  Best Accuracy: {best_by_acc["model"]:<28s}  Acc = {best_by_acc["accuracy"]:.4f}')

    # === Train-Test gap analysis (overfitting check) ===
    print('\n  ===== OVERFITTING CHECK (Train F1 - Test F1) =====')
    print(f'  {"Model":<28s}  {"Train F1":>10s}  {"Test F1":>10s}  {"Gap":>10s}  {"Status":<15s}')
    print('  ' + '-' * 80)
    model_names = sorted(set(r['model'] for r in all_results))
    for name in model_names:
        train_r = next((r for r in all_results if r['model'] == name and r['split'] == 'train'), None)
        test_r = next((r for r in all_results if r['model'] == name and r['split'] == 'test'), None)
        if train_r and test_r:
            gap = train_r['f1'] - test_r['f1']
            if gap > 0.15:
                status = 'OVERFITTING'
            elif gap > 0.05:
                status = 'mild overfit'
            elif gap < -0.05:
                status = 'underfit?'
            else:
                status = 'OK'
            print(f'  {name:<28s}  {train_r["f1"]:>10.4f}  {test_r["f1"]:>10.4f}  '
                  f'{gap:>+10.4f}  {status:<15s}')

    elapsed = time.time() - start
    print(f'\n  Total benchmark time: {elapsed:.1f}s')
    print('=' * 100)


if __name__ == '__main__':
    main()
