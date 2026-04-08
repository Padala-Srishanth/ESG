"""
================================================================================
REGENERATE SHAP VALUES ON CURRENT FEATURE MATRIX
================================================================================
This script regenerates SHAP analysis to include all 91 NLP features
(Categories 7-10) that were added to feature_engineering_nlp.py.

Steps:
    1. Load current feature matrix (480 x 213)
    2. Build proxy labels using same logic as model_training.py
    3. Train Gradient Boosting model on the FULL feature set
    4. Compute REAL SHAP values using TreeExplainer
    5. Save SHAP feature importance (mean |SHAP|)
    6. Save per-company SHAP values for ALL companies (used by dashboard)

Outputs:
    - data/processed/shap_feature_importance.csv  (replaces stale data)
    - data/processed/shap_values_all.csv          (per-company SHAP)
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    print('=' * 70)
    print('  SHAP REGENERATION ON CURRENT FEATURE SET')
    print('=' * 70)

    # === STEP 1: Load current feature matrix ===
    fm = pd.read_csv('data/processed/feature_matrix.csv')
    print(f'\n[1/6] Loaded feature matrix: {fm.shape}')

    # === STEP 2: Build proxy labels (same as model_training.py) ===
    print('\n[2/6] Building proxy labels...')
    ind1 = (fm['esg_controversy_divergence'] > fm['esg_controversy_divergence'].quantile(0.75)).astype(int)
    ind2 = (fm['greenwashing_signal_score'] > fm['greenwashing_signal_score'].quantile(0.75)).astype(int)
    ind3 = fm.get('risk_controversy_mismatch', pd.Series([0] * len(fm))).fillna(0).astype(int)
    ind4 = (fm['controversy_risk_ratio'] > fm['controversy_risk_ratio'].quantile(0.75)).astype(int)
    ind5 = (fm['combined_anomaly_score'] > fm['combined_anomaly_score'].quantile(0.75)).astype(int)

    proxy_score = ind1 + ind2 + ind3 + ind4 + ind5
    gw_label = (proxy_score >= 2).astype(int)

    print(f'  Proxy score distribution: {dict(proxy_score.value_counts().sort_index())}')
    print(f'  Binary label: {dict(gw_label.value_counts())}')

    # === STEP 3: Prepare features ===
    exclude = {
        'gw_indicator_1', 'gw_indicator_2', 'gw_indicator_3', 'gw_indicator_4', 'gw_indicator_5',
        'gw_proxy_score', 'gw_label_binary', 'gw_label_multiclass',
        'symbol', 'company_name', 'sector', 'industry', 'description', 'source',
        'esg_controversy_segment', 'sector_risk_segment',
        'ESG_Risk_Level', 'Controversy_Level',
    }

    feature_cols = [c for c in fm.select_dtypes(include=[np.number]).columns if c not in exclude]
    X = fm[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = gw_label.values
    print(f'\n[3/6] Feature matrix shape: {X.shape}  ({len(feature_cols)} numeric features)')

    # Verify new features are included
    new_feats_check = [
        'aggregate_esg_nlp_score', 'regulatory_readiness_score', 'policy_esg_gap',
        'narrative_credibility_index', 'commitment_credibility_score',
        'multi_signal_greenwashing_score', 'temporal_greenwashing_signal',
        'news_greenwashing_signal',
    ]
    new_in_X = sum(1 for f in new_feats_check if f in X.columns)
    print(f'  Sample new features included: {new_in_X}/{len(new_feats_check)}')

    # === STEP 4: Train Gradient Boosting model ===
    print('\n[4/6] Training Gradient Boosting on full 213-feature set...')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f'  Train accuracy: {train_acc:.4f}')
    print(f'  Test accuracy:  {test_acc:.4f}')

    # === STEP 5: Compute SHAP values ===
    print('\n[5/6] Computing SHAP values with TreeExplainer...')
    import shap

    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        shap_arr = np.array(shap_values)

    print(f'  SHAP values shape: {shap_arr.shape}')

    # === STEP 6: Save SHAP-based feature importance ===
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': mean_abs_shap,
        'model': 'gradient_boosting',
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

    shap_importance.to_csv('data/processed/shap_feature_importance.csv', index=False)
    print('\n[6/6] Saved data/processed/shap_feature_importance.csv')

    # === Compute SHAP for ALL companies (not just test set) ===
    print('  Computing SHAP for ALL 480 companies...')
    X_all_scaled = scaler.transform(X)
    X_all_df = pd.DataFrame(X_all_scaled, columns=feature_cols)
    shap_all = explainer.shap_values(X_all_df)
    shap_all_arr = (
        np.array(shap_all[1] if isinstance(shap_all, list) and len(shap_all) > 1 else shap_all)
    )

    shap_all_df = pd.DataFrame(shap_all_arr, columns=feature_cols)
    shap_all_df.insert(0, 'company_name', fm['company_name'].values)
    shap_all_df.to_csv('data/processed/shap_values_all.csv', index=False)
    print(f'  Saved data/processed/shap_values_all.csv  ({shap_all_df.shape})')

    # === Display top features (highlight NEW features) ===
    new_feature_set = {
        'paris_agreement_alignment', 'eu_taxonomy_alignment', 'sec_climate_alignment',
        'tcfd_alignment', 'sdg_alignment', 'gri_standards_alignment',
        'regulatory_breadth_index', 'total_policy_density', 'policy_specificity_score',
        'policy_esg_gap', 'framework_consistency_score', 'regulatory_readiness_score',
        'promotional_intent_density', 'defensive_intent_density', 'factual_intent_density',
        'strategic_intent_density', 'narrative_credibility_index', 'promotional_dominance_score',
        'intent_diversity_score', 'defensive_to_factual_ratio', 'sentiment_intent_divergence',
        'news_greenwashing_signal', 'past_achievement_density', 'present_action_density',
        'specific_future_density', 'vague_future_density', 'temporal_balance_score',
        'commitment_credibility_score', 'temporal_specificity_ratio', 'progress_to_promise_ratio',
        'year_mention_density', 'temporal_greenwashing_signal',
        'policy_sentiment_interaction', 'readability_greenwashing_interaction',
        'temporal_policy_interaction', 'vocabulary_intent_interaction',
        'evidence_readability_interaction', 'claim_credibility_ratio',
        'multi_signal_greenwashing_score', 'esg_linguistic_credibility_index',
        'credibility_confidence_interval', 'policy_temporal_alignment',
        'narrative_consistency_score', 'aggregate_esg_nlp_score',
    }

    print('\n  --- TOP 25 FEATURES BY ACTUAL SHAP IMPORTANCE ---')
    for i, row in shap_importance.head(25).iterrows():
        is_new = '  (NEW Cat 7-10)' if row['feature'] in new_feature_set else ''
        rank = i + 1
        feat = row['feature']
        imp = row['shap_importance']
        print(f'  #{rank:>2d}  {feat:<50s} {imp:.6f}{is_new}')

    top50 = set(shap_importance.head(50)['feature'].tolist())
    top100 = set(shap_importance.head(100)['feature'].tolist())
    print(f'\n  New features in TOP 50:  {len(top50 & new_feature_set)}/50')
    print(f'  New features in TOP 100: {len(top100 & new_feature_set)}/100')

    # === Save model metadata ===
    import json
    metadata = {
        'model_type': 'GradientBoostingClassifier',
        'n_features': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'positive_class_ratio': float(y.mean()),
        'feature_list': feature_cols,
    }
    with open('data/processed/shap_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print('\n  Saved data/processed/shap_model_metadata.json')

    print('\n  RESULT: SHAP regeneration COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
