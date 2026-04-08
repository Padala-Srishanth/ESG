"""
================================================================================
HYBRID XGBOOST-CATBOOST MODEL WITH MONTE CARLO UNCERTAINTY
================================================================================
Inspired by:
    "Optimized hybrid XGBoost-CatBoost model for enhanced prediction... using
     Monte Carlo simulations" -- ScienceDirect

Adapted for:
    ESG Greenwashing Detection (binary classification)

Approach:
    1. Train an XGBoost classifier on the engineered feature matrix
    2. Train a CatBoost classifier on the same features
    3. Combine predictions via OPTIMIZED WEIGHTED BLENDING
       - Weights are tuned by minimizing log-loss on validation split
    4. Run Monte Carlo simulation: bootstrap-resample training data N times,
       retrain hybrid, capture prediction variance per company
    5. Output: hybrid probability + 95% confidence interval per company

Why a hybrid:
    - XGBoost excels at numerical features and complex interactions
    - CatBoost excels at categorical features and is robust to overfitting
    - Their decision boundaries are different, so blending them captures
      complementary signals (similar to model stacking)
    - Monte Carlo gives uncertainty estimates -- critical for risk decisions

Outputs:
    - data/processed/hybrid_predictions.csv      (per-company hybrid prediction + CI)
    - data/processed/hybrid_model_metrics.json   (accuracy, blend weights, MC config)
    - data/processed/hybrid_feature_importance.csv (averaged from both models)
================================================================================
"""

import pandas as pd
import numpy as np
import json
import warnings
import time
warnings.filterwarnings('ignore')


def build_proxy_labels(fm):
    """Build the same 5-indicator proxy labels as model_training.py."""
    ind1 = (fm['esg_controversy_divergence'] > fm['esg_controversy_divergence'].quantile(0.75)).astype(int)
    ind2 = (fm['greenwashing_signal_score'] > fm['greenwashing_signal_score'].quantile(0.75)).astype(int)
    ind3 = fm.get('risk_controversy_mismatch', pd.Series([0] * len(fm))).fillna(0).astype(int)
    ind4 = (fm['controversy_risk_ratio'] > fm['controversy_risk_ratio'].quantile(0.75)).astype(int)
    ind5 = (fm['combined_anomaly_score'] > fm['combined_anomaly_score'].quantile(0.75)).astype(int)
    proxy_score = ind1 + ind2 + ind3 + ind4 + ind5
    return (proxy_score >= 2).astype(int).values, proxy_score.values


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


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier with early stopping."""
    import xgboost as xgb
    print('  [XGB] Training XGBoost...')
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost classifier with early stopping."""
    from catboost import CatBoostClassifier
    print('  [CAT] Training CatBoost...')
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=30,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def find_optimal_blend_weight(p_xgb, p_cat, y_true):
    """
    Find the optimal alpha in [0, 1] that minimizes log-loss for the blend:
        p_hybrid = alpha * p_xgb + (1 - alpha) * p_cat

    Uses fine grid search over alpha in steps of 0.01.
    """
    from sklearn.metrics import log_loss
    best_alpha = 0.5
    best_loss = float('inf')
    for alpha in np.arange(0.0, 1.01, 0.01):
        p_blend = alpha * p_xgb + (1 - alpha) * p_cat
        # Clip to valid probability range to avoid log(0)
        p_blend = np.clip(p_blend, 1e-7, 1 - 1e-7)
        loss = log_loss(y_true, p_blend)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha, best_loss


def monte_carlo_uncertainty(X_train, y_train, X_all, alpha, n_runs=20, sample_frac=0.8):
    """
    Run Monte Carlo simulation by bootstrap-resampling training data.

    For each of n_runs iterations:
        1. Sample sample_frac of training data with replacement
        2. Retrain XGBoost and CatBoost on the bootstrap sample
        3. Predict on the FULL dataset
        4. Blend with the optimal alpha
        5. Store predictions

    Returns:
        mean_pred  : average prediction per company across MC runs
        std_pred   : standard deviation per company (uncertainty)
        ci_low     : 2.5th percentile (95% CI lower bound)
        ci_high    : 97.5th percentile (95% CI upper bound)
    """
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.utils import resample

    print(f'\n  Running Monte Carlo simulation ({n_runs} runs)...')
    n_samples = int(len(X_train) * sample_frac)
    mc_predictions = np.zeros((n_runs, len(X_all)))

    for run in range(n_runs):
        # Bootstrap resample training set
        X_boot, y_boot = resample(X_train, y_train, n_samples=n_samples, random_state=run)

        # Train both models on bootstrap sample
        xgb_mc = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=run,
            eval_metric='logloss', verbosity=0, n_jobs=-1,
        )
        xgb_mc.fit(X_boot, y_boot)

        cat_mc = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_seed=run, loss_function='Logloss', verbose=False,
        )
        cat_mc.fit(X_boot, y_boot, verbose=False)

        # Predict on FULL dataset
        p_xgb_mc = xgb_mc.predict_proba(X_all)[:, 1]
        p_cat_mc = cat_mc.predict_proba(X_all)[:, 1]
        p_hybrid_mc = alpha * p_xgb_mc + (1 - alpha) * p_cat_mc

        mc_predictions[run] = p_hybrid_mc
        if (run + 1) % 5 == 0:
            print(f'    MC run {run + 1}/{n_runs} complete')

    mean_pred = mc_predictions.mean(axis=0)
    std_pred = mc_predictions.std(axis=0)
    ci_low = np.percentile(mc_predictions, 2.5, axis=0)
    ci_high = np.percentile(mc_predictions, 97.5, axis=0)

    return mean_pred, std_pred, ci_low, ci_high


def main():
    print('=' * 70)
    print('  HYBRID XGBOOST-CATBOOST MODEL WITH MONTE CARLO UNCERTAINTY')
    print('=' * 70)

    start_time = time.time()

    # === STEP 1: Load data ===
    print('\n[1/7] Loading feature matrix...')
    fm = pd.read_csv('data/processed/feature_matrix.csv')
    print(f'  Shape: {fm.shape}')

    # === STEP 2: Build labels and prepare features ===
    print('\n[2/7] Building labels and preparing features...')
    y, proxy_score = build_proxy_labels(fm)
    X, feature_cols = prepare_features(fm)
    print(f'  Features: {len(feature_cols)}')
    print(f'  Class balance: {dict(pd.Series(y).value_counts())}')

    # === STEP 3: Train/val/test split ===
    print('\n[3/7] Splitting data (train/val/test = 60/20/20)...')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    X_all_s = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # === STEP 4: Train both models ===
    print('\n[4/7] Training base models...')
    xgb_model = train_xgboost(X_train_s, y_train, X_val_s, y_val)
    cat_model = train_catboost(X_train_s, y_train, X_val_s, y_val)

    # === STEP 5: Optimal blend weight ===
    print('\n[5/7] Finding optimal blend weight on validation set...')
    p_xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
    p_cat_val = cat_model.predict_proba(X_val_s)[:, 1]
    alpha, val_loss = find_optimal_blend_weight(p_xgb_val, p_cat_val, y_val)
    print(f'  Optimal alpha (XGBoost weight): {alpha:.3f}')
    print(f'  CatBoost weight: {1 - alpha:.3f}')
    print(f'  Validation log-loss: {val_loss:.4f}')

    # === STEP 6: Evaluate on TRAIN, VALIDATION, and TEST sets ===
    print('\n[6/7] Evaluating on Train / Validation / Test sets...')
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        log_loss, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
    )

    def metrics(name, split, y_true, y_pred, y_prob):
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

    # Get probabilities for all three splits
    p_xgb_train = xgb_model.predict_proba(X_train_s)[:, 1]
    p_cat_train = cat_model.predict_proba(X_train_s)[:, 1]
    p_hybrid_train = alpha * p_xgb_train + (1 - alpha) * p_cat_train

    p_xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
    p_cat_val = cat_model.predict_proba(X_val_s)[:, 1]
    p_hybrid_val = alpha * p_xgb_val + (1 - alpha) * p_cat_val

    p_xgb_test = xgb_model.predict_proba(X_test_s)[:, 1]
    p_cat_test = cat_model.predict_proba(X_test_s)[:, 1]
    p_hybrid_test = alpha * p_xgb_test + (1 - alpha) * p_cat_test

    pred_xgb = (p_xgb_test >= 0.5).astype(int)
    pred_cat = (p_cat_test >= 0.5).astype(int)
    pred_hybrid = (p_hybrid_test >= 0.5).astype(int)

    # Compute metrics for all (model, split) combinations
    all_results = []
    for split, (yt, p_x, p_c, p_h) in [
        ('train', (y_train, p_xgb_train, p_cat_train, p_hybrid_train)),
        ('val',   (y_val,   p_xgb_val,   p_cat_val,   p_hybrid_val)),
        ('test',  (y_test,  p_xgb_test,  p_cat_test,  p_hybrid_test)),
    ]:
        all_results.append(metrics('XGBoost', split, yt, (p_x >= 0.5).astype(int), p_x))
        all_results.append(metrics('CatBoost', split, yt, (p_c >= 0.5).astype(int), p_c))
        all_results.append(metrics('Hybrid (XGB+Cat)', split, yt, (p_h >= 0.5).astype(int), p_h))

    # Backward-compat: keep test-only metrics for legacy code paths
    m_xgb = next(r for r in all_results if r['model'] == 'XGBoost' and r['split'] == 'test')
    m_cat = next(r for r in all_results if r['model'] == 'CatBoost' and r['split'] == 'test')
    m_hybrid = next(r for r in all_results if r['model'] == 'Hybrid (XGB+Cat)' and r['split'] == 'test')

    # Print Train / Val / Test tables
    for split_name in ['train', 'val', 'test']:
        title = {'train': 'TRAIN SET', 'val': 'VALIDATION SET', 'test': 'TEST SET'}[split_name]
        rows = [r for r in all_results if r['split'] == split_name]
        print(f'\n  ===== {title} (n = {rows[0]["n_samples"]}) =====')
        print(f'  {"Model":<22s}  {"Acc":>8s}  {"BalAcc":>8s}  {"Prec":>8s}  {"Recall":>8s}  '
              f'{"F1":>8s}  {"AUC":>8s}  {"LogLoss":>8s}  {"MCC":>8s}')
        print('  ' + '-' * 100)
        for r in rows:
            print(f'  {r["model"]:<22s}  {r["accuracy"]:>8.4f}  {r["balanced_accuracy"]:>8.4f}  '
                  f'{r["precision"]:>8.4f}  {r["recall"]:>8.4f}  {r["f1"]:>8.4f}  '
                  f'{r["roc_auc"]:>8.4f}  {r["log_loss"]:>8.4f}  {r["mcc"]:>8.4f}')

    # === Save tabular results CSV (submission-ready) ===
    results_df = pd.DataFrame(all_results)
    # Order columns logically
    col_order = ['model', 'split', 'n_samples', 'accuracy', 'balanced_accuracy',
                 'precision', 'recall', 'f1', 'roc_auc', 'log_loss', 'mcc',
                 'tn', 'fp', 'fn', 'tp']
    results_df = results_df[col_order]
    results_df.to_csv('data/processed/hybrid_train_test_results.csv', index=False)
    print('\n  Saved data/processed/hybrid_train_test_results.csv')

    # === STEP 7: Monte Carlo uncertainty ===
    print('\n[7/7] Monte Carlo uncertainty estimation...')
    mean_pred, std_pred, ci_low, ci_high = monte_carlo_uncertainty(
        X_train_s, y_train, X_all_s, alpha, n_runs=20, sample_frac=0.8
    )

    # === Generate predictions for ALL companies (not just test set) ===
    p_xgb_all = xgb_model.predict_proba(X_all_s)[:, 1]
    p_cat_all = cat_model.predict_proba(X_all_s)[:, 1]
    p_hybrid_all = alpha * p_xgb_all + (1 - alpha) * p_cat_all

    # === Save outputs ===
    print('\n  Saving outputs...')
    predictions_df = pd.DataFrame({
        'company_name': fm['company_name'].values,
        'sector': fm.get('sector', pd.Series(['']*len(fm))).values,
        'xgboost_prob': p_xgb_all,
        'catboost_prob': p_cat_all,
        'hybrid_prob': p_hybrid_all,
        'hybrid_mc_mean': mean_pred,
        'hybrid_mc_std': std_pred,
        'hybrid_ci_low': ci_low,
        'hybrid_ci_high': ci_high,
        'uncertainty': std_pred,                       # Same as std_pred, easier name
        'predicted_class': (p_hybrid_all >= 0.5).astype(int),
        'gw_label_proxy': y,
        'gw_proxy_score': proxy_score,
    })
    predictions_df.to_csv('data/processed/hybrid_predictions.csv', index=False)
    print(f'  Saved data/processed/hybrid_predictions.csv  ({predictions_df.shape})')

    # === Save metrics ===
    metrics_dict = {
        'model_type': 'Hybrid XGBoost-CatBoost with Monte Carlo',
        'n_features': len(feature_cols),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'class_balance': {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        'optimal_alpha_xgboost': float(alpha),
        'optimal_alpha_catboost': float(1 - alpha),
        'validation_logloss': float(val_loss),
        'monte_carlo_runs': 20,
        'monte_carlo_sample_frac': 0.8,
        'metrics': {
            'xgboost': m_xgb,
            'catboost': m_cat,
            'hybrid': m_hybrid,
        },
    }
    with open('data/processed/hybrid_model_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print('  Saved data/processed/hybrid_model_metrics.json')

    # === Hybrid feature importance (averaged from XGB and CAT) ===
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'xgb_importance': xgb_model.feature_importances_,
    })
    cat_importance = pd.DataFrame({
        'feature': feature_cols,
        'cat_importance': cat_model.get_feature_importance(),
    })
    fi = xgb_importance.merge(cat_importance, on='feature')

    # Normalize each before averaging
    fi['xgb_norm'] = fi['xgb_importance'] / (fi['xgb_importance'].max() + 1e-10)
    fi['cat_norm'] = fi['cat_importance'] / (fi['cat_importance'].max() + 1e-10)
    fi['hybrid_importance'] = alpha * fi['xgb_norm'] + (1 - alpha) * fi['cat_norm']
    fi = fi.sort_values('hybrid_importance', ascending=False).reset_index(drop=True)
    fi[['feature', 'xgb_importance', 'cat_importance', 'hybrid_importance']].to_csv(
        'data/processed/hybrid_feature_importance.csv', index=False
    )
    print(f'  Saved data/processed/hybrid_feature_importance.csv  ({len(fi)} features)')

    # === Display summary ===
    print('\n  --- TOP 15 HYBRID FEATURES ---')
    new_feature_set = {
        'aggregate_esg_nlp_score', 'regulatory_readiness_score', 'policy_esg_gap',
        'narrative_credibility_index', 'commitment_credibility_score',
        'multi_signal_greenwashing_score', 'temporal_greenwashing_signal',
        'news_greenwashing_signal', 'sec_climate_alignment', 'tcfd_alignment',
        'paris_agreement_alignment', 'eu_taxonomy_alignment', 'sdg_alignment',
        'gri_standards_alignment', 'past_achievement_density', 'factual_intent_density',
    }
    for i, row in fi.head(15).iterrows():
        is_new = '  (NEW)' if row['feature'] in new_feature_set else ''
        print(f'  #{i + 1:>2d}  {row["feature"]:<45s}  {row["hybrid_importance"]:.4f}{is_new}')

    # === Top high-uncertainty companies ===
    print('\n  --- TOP 10 HIGHEST UNCERTAINTY COMPANIES ---')
    high_unc = predictions_df.nlargest(10, 'uncertainty')[
        ['company_name', 'hybrid_prob', 'hybrid_mc_mean', 'uncertainty', 'hybrid_ci_low', 'hybrid_ci_high']
    ]
    for _, row in high_unc.iterrows():
        print(f'  {row["company_name"][:35]:<35s}  '
              f'p={row["hybrid_prob"]:.3f}  +/-{row["uncertainty"]:.3f}  '
              f'CI=[{row["hybrid_ci_low"]:.3f}, {row["hybrid_ci_high"]:.3f}]')

    elapsed = time.time() - start_time
    print(f'\n  Total time: {elapsed:.1f} seconds')
    print('=' * 70)


if __name__ == '__main__':
    main()
