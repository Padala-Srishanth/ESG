"""
================================================================================
INTEGRATE NEW ESG DATASET INTO GREENWASHING DETECTION PROJECT
================================================================================
Author  : ML Engineering Pipeline
Project : ESG Greenwashing Detection via NLP & Machine Learning

Purpose:
    Integrate a new dataset of 722 companies (data/data.csv) into the existing
    480-company pipeline (S&P 500 + NIFTY 50). The new dataset has ESG grades
    and scores but NO text descriptions, so NLP features must be imputed using
    population medians from the original 480 companies.

Strategy (Option B):
    1. Normalize new dataset scores to match existing Sustainalytics scale (0-50)
    2. Map grades to controversy proxy levels
    3. Remove overlapping companies (keep existing versions with text)
    4. Merge: 480 existing + new unique = expanded dataset
    5. Run full feature engineering pipeline
    6. Impute NLP features for new companies with median from original 480
    7. Train all models on expanded dataset
    8. Compare metrics old vs new
    9. Run risk scoring on expanded dataset
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  WARNING: XGBoost not installed. Skipping XGBoost model.")

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================

def load_datasets():
    """Load existing company profiles and new ESG dataset."""
    print("=" * 70)
    print("  STEP 1: Loading Datasets")
    print("=" * 70)

    # Load existing 480 companies
    existing_path = os.path.join(PROCESSED_DIR, "company_profiles.csv")
    df_existing = pd.read_csv(existing_path)
    print(f"  Existing company profiles: {df_existing.shape[0]} companies, {df_existing.shape[1]} cols")
    print(f"    Columns: {list(df_existing.columns)}")

    # Load new 722-company dataset
    new_path = os.path.join(DATA_DIR, "data.csv")
    df_new = pd.read_csv(new_path)
    print(f"  New ESG dataset: {df_new.shape[0]} companies, {df_new.shape[1]} cols")
    print(f"    Columns: {list(df_new.columns)}")

    return df_existing, df_new


# ============================================================================
# STEP 2: NORMALIZE NEW ESG SCORES TO SUSTAINALYTICS SCALE (0-50)
# ============================================================================

def normalize_new_scores(df_new):
    """
    Normalize new dataset ESG scores to match existing Sustainalytics scale (0-50).

    New dataset ranges:
        environment_score: 200-719
        social_score: 160-667
        governance_score: 75-475
        total_score: 600-1536

    Target range: 0-50 (Sustainalytics ESG risk score scale)
    """
    print("\n" + "=" * 70)
    print("  STEP 2: Normalizing New ESG Scores to Sustainalytics Scale (0-50)")
    print("=" * 70)

    df = df_new.copy()

    # Define source ranges for each score component
    score_configs = {
        'environment_score': {'min': 200, 'max': 719},
        'social_score': {'min': 160, 'max': 667},
        'governance_score': {'min': 75, 'max': 475},
        'total_score': {'min': 600, 'max': 1536},
    }

    target_min, target_max = 0, 50  # Sustainalytics scale

    for col, cfg in score_configs.items():
        src_min, src_max = cfg['min'], cfg['max']
        # MinMax normalization: (x - min) / (max - min) * target_range + target_min
        # IMPORTANT: In Sustainalytics, HIGHER score = HIGHER risk
        # In the new dataset, HIGHER score = BETTER ESG performance
        # So we INVERT: a high new score should map to a LOW risk score
        df[f'{col}_normalized'] = target_max - (
            (df[col] - src_min) / (src_max - src_min) * (target_max - target_min)
        )
        # Clip to valid range
        df[f'{col}_normalized'] = df[f'{col}_normalized'].clip(target_min, target_max)

        print(f"  {col}: [{src_min}-{src_max}] -> [{target_min}-{target_max}] (inverted)")
        print(f"    Before: mean={df[col].mean():.1f}, range=[{df[col].min()}, {df[col].max()}]")
        print(f"    After:  mean={df[f'{col}_normalized'].mean():.2f}, "
              f"range=[{df[f'{col}_normalized'].min():.2f}, {df[f'{col}_normalized'].max():.2f}]")

    return df


# ============================================================================
# STEP 3: MAP GRADES TO CONTROVERSY LEVELS
# ============================================================================

def map_grades_to_levels(df_new):
    """
    Map ESG grades to controversy proxy levels.

    Grade mapping (higher grade = lower controversy):
        A   -> 1 (Low controversy)
        BBB -> 2 (Moderate)
        BB  -> 3 (Significant)
        B   -> 4 (High)

    Also map total_grade to ESG risk level:
        A   -> 1 (Low risk)
        BBB -> 2 (Medium risk)
        BB  -> 3 (High risk)
        B   -> 4 (Severe risk - mapped to existing scale 0-4)
    """
    print("\n" + "=" * 70)
    print("  STEP 3: Mapping Grades to Controversy & Risk Levels")
    print("=" * 70)

    df = df_new.copy()

    # Controversy level from total_grade (inverted: good grade = low controversy)
    controversy_map = {
        'A': 1,    # Best grade -> lowest controversy
        'AA': 1,   # Handle AA if present
        'BBB': 2,  # Good grade -> moderate controversy
        'BB': 3,   # Average grade -> significant controversy
        'B': 4,    # Poor grade -> high controversy
    }
    df['controversy_proxy'] = df['total_grade'].map(controversy_map).fillna(2)

    # ESG risk level from total_grade (good grade = low risk)
    risk_level_map = {
        'A': 1,    # Low risk
        'AA': 0,   # Negligible risk
        'BBB': 2,  # Medium risk
        'BB': 3,   # High risk
        'B': 4,    # Severe risk
    }
    df['esg_risk_level_proxy'] = df['total_grade'].map(risk_level_map).fillna(2)

    print(f"  Controversy proxy distribution:")
    for level, count in df['controversy_proxy'].value_counts().sort_index().items():
        print(f"    Level {int(level)}: {count} companies")

    print(f"  ESG risk level distribution:")
    for level, count in df['esg_risk_level_proxy'].value_counts().sort_index().items():
        print(f"    Level {int(level)}: {count} companies")

    return df


# ============================================================================
# STEP 4: CREATE UNIFIED COMPANY PROFILES
# ============================================================================

def create_expanded_profiles(df_existing, df_new_processed):
    """
    Create unified company profiles by merging existing and new datasets.

    Existing columns: symbol, company_name, sector, industry, description,
        total_esg_risk_score, env_risk_score, gov_risk_score, social_risk_score,
        controversy_score, ESG_Risk_Level_Encoded, Controversy_Level_Encoded, source

    Steps:
        1. Remove overlapping companies from new dataset (keep existing with text)
        2. Map new dataset columns to match existing schema
        3. Concatenate
    """
    print("\n" + "=" * 70)
    print("  STEP 4: Creating Expanded Company Profiles")
    print("=" * 70)

    # Identify overlapping companies by ticker/symbol
    existing_symbols = set(df_existing['symbol'].str.strip().str.upper())
    new_tickers = set(df_new_processed['ticker'].str.strip().str.upper())
    overlap = existing_symbols & new_tickers

    print(f"  Existing companies: {len(df_existing)}")
    print(f"  New dataset companies: {len(df_new_processed)}")
    print(f"  Overlapping tickers: {len(overlap)}")

    # Remove overlapping companies from new dataset (keep existing versions with text)
    df_new_unique = df_new_processed[
        ~df_new_processed['ticker'].str.strip().str.upper().isin(overlap)
    ].copy()
    print(f"  New unique companies (after removing overlap): {len(df_new_unique)}")

    # Map new dataset to match existing schema
    new_profiles = pd.DataFrame()
    new_profiles['symbol'] = df_new_unique['ticker'].str.strip().str.upper()
    new_profiles['company_name'] = df_new_unique['name'].str.strip().str.upper()
    new_profiles['sector'] = np.nan  # New dataset does not have sector
    new_profiles['industry'] = df_new_unique['industry'].fillna('Unknown')
    new_profiles['description'] = np.nan  # NO TEXT DESCRIPTIONS in new dataset
    new_profiles['total_esg_risk_score'] = df_new_unique['total_score_normalized'].values
    new_profiles['env_risk_score'] = df_new_unique['environment_score_normalized'].values
    new_profiles['gov_risk_score'] = df_new_unique['governance_score_normalized'].values
    new_profiles['social_risk_score'] = df_new_unique['social_score_normalized'].values
    new_profiles['controversy_score'] = df_new_unique['controversy_proxy'].values
    new_profiles['ESG_Risk_Level_Encoded'] = df_new_unique['esg_risk_level_proxy'].values
    new_profiles['Controversy_Level_Encoded'] = df_new_unique['controversy_proxy'].values
    new_profiles['source'] = 'NEW_ESG_DATA'

    # Fill missing sector using industry-to-sector mapping from existing data
    # Build a simple industry->sector lookup from existing data
    sector_lookup = df_existing.dropna(subset=['industry', 'sector']).drop_duplicates('industry')
    sector_map = dict(zip(sector_lookup['industry'].str.upper(), sector_lookup['sector']))

    new_profiles['sector'] = new_profiles['industry'].str.upper().map(sector_map)
    new_profiles['sector'] = new_profiles['sector'].fillna('Unknown')

    mapped_count = (new_profiles['sector'] != 'Unknown').sum()
    print(f"  Sectors mapped from industry: {mapped_count}/{len(new_profiles)}")

    # Concatenate existing + new
    df_expanded = pd.concat([df_existing, new_profiles], ignore_index=True)

    # Fill missing numeric values with median
    numeric_cols = df_expanded.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_expanded[col].median()
        n_null = df_expanded[col].isnull().sum()
        if n_null > 0:
            df_expanded[col] = df_expanded[col].fillna(median_val)
            print(f"  Filled {n_null} NaN in '{col}' with median ({median_val:.2f})")

    print(f"\n  EXPANDED PROFILES: {len(df_expanded)} companies "
          f"({len(df_existing)} existing + {len(new_profiles)} new)")

    return df_expanded, len(df_existing)


# ============================================================================
# STEP 5-6: FEATURE ENGINEERING WITH NLP MEDIAN IMPUTATION
# ============================================================================

def run_feature_engineering(df_expanded, n_original):
    """
    Run the full feature engineering pipeline on the expanded dataset.

    Key challenge: New companies have no text descriptions, so NLP features
    will be NaN/0 for them. After running the pipeline, we impute those
    NLP features with the MEDIAN values from the original 480 companies.

    This is superior to filling with zeros because:
    - Zeros would create a systematic bias (all new companies look like they
      have zero linguistic risk, which biases the model)
    - Median imputation assumes new companies are "average" in text features,
      which is a neutral assumption that does not distort the model
    """
    print("\n" + "=" * 70)
    print("  STEP 5: Running Feature Engineering Pipeline")
    print("=" * 70)

    # Import feature engineering modules
    sys.path.insert(0, BASE_DIR)
    from feature_engineering_numerical import NumericalFeatureEngineer
    from feature_engineering_nlp import NLPFeatureEngineer
    from feature_engineering_categorical import CategoricalFeatureEngineer

    df = df_expanded.copy()

    # Ensure description column exists and handle missing text
    if 'description' not in df.columns:
        df['description'] = ''
    df['description'] = df['description'].fillna('')

    # ---- Numerical Feature Engineering ----
    print("\n  Running Numerical Feature Engineering...")
    num_eng = NumericalFeatureEngineer()
    cols_before = df.shape[1]
    df = num_eng.engineer_all_numerical_features(df)
    print(f"  Numerical features added: {df.shape[1] - cols_before}")

    # ---- NLP Feature Engineering ----
    print("\n  Running NLP Feature Engineering...")
    nlp_eng = NLPFeatureEngineer()
    cols_before = df.shape[1]
    df = nlp_eng.engineer_all_nlp_features(df)
    nlp_features_added = df.shape[1] - cols_before
    print(f"  NLP features added: {nlp_features_added}")

    # ---- CRITICAL: Impute NLP features for new companies ----
    # Identify NLP feature columns (all columns added by the NLP module)
    nlp_feature_cols = [
        'text_polarity', 'text_positive_ratio', 'text_negative_ratio',
        'text_sentiment_strength', 'text_pos_neg_ratio',
        'avg_sentence_length', 'syllable_ratio',
        'flesch_reading_ease', 'gunning_fog_index',
        'total_word_count', 'unique_word_count',
        'lexical_diversity', 'hapax_legomena_ratio',
        'env_keyword_count', 'env_keyword_density',
        'social_keyword_count', 'social_keyword_density',
        'gov_keyword_count', 'gov_keyword_density',
        'total_esg_keyword_count', 'total_esg_keyword_density',
        'esg_keyword_balance', 'dominant_keyword_pillar',
        'vague_language_count', 'vague_language_density',
        'hedge_language_count', 'hedge_language_density',
        'superlative_count', 'superlative_density',
        'concrete_evidence_count', 'concrete_evidence_density',
        'greenwashing_signal_score', 'vague_to_concrete_ratio',
        'sentence_count', 'sentence_length_variance',
        'short_sentence_ratio', 'long_sentence_ratio',
    ]
    # Filter to columns that actually exist
    nlp_feature_cols = [c for c in nlp_feature_cols if c in df.columns]

    print(f"\n  STEP 6: Imputing NLP Features for New Companies (median from original {n_original})")
    print(f"  NLP feature columns to impute: {len(nlp_feature_cols)}")

    # Compute medians from the original companies (first n_original rows)
    original_subset = df.iloc[:n_original]
    nlp_medians = {}
    for col in nlp_feature_cols:
        median_val = original_subset[col].median()
        nlp_medians[col] = median_val

    # Impute NLP features for new companies (rows after n_original)
    # New companies have empty descriptions so NLP features are 0 or NaN
    new_company_mask = df.index >= n_original
    imputed_count = 0
    for col in nlp_feature_cols:
        if col in df.columns:
            df.loc[new_company_mask, col] = nlp_medians[col]
            imputed_count += 1

    print(f"  Imputed {imputed_count} NLP features for {new_company_mask.sum()} new companies")
    print(f"  Sample NLP medians used:")
    for col in list(nlp_medians.keys())[:5]:
        print(f"    {col}: {nlp_medians[col]:.4f}")

    # ---- Categorical Feature Engineering ----
    print("\n  Running Categorical Feature Engineering...")
    cat_eng = CategoricalFeatureEngineer()
    cols_before = df.shape[1]
    df = cat_eng.engineer_all_categorical_features(df)
    print(f"  Categorical features added: {df.shape[1] - cols_before}")

    return df, nlp_medians


# ============================================================================
# STEP 7: FEATURE QUALITY CHECKS
# ============================================================================

def run_quality_checks(df):
    """Run feature quality checks: handle inf, NaN, constants, duplicates."""
    print("\n" + "=" * 70)
    print("  STEP 7: Feature Quality Checks")
    print("=" * 70)

    cols_before = df.shape[1]

    # Replace infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  Infinite values found: {inf_count}")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_count = df[numeric_cols].isnull().sum().sum()
    print(f"  NaN values found: {nan_count}")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Remove constant features
    constant_features = [col for col in numeric_cols if df[col].nunique() <= 1]
    if constant_features:
        print(f"  Constant features removed: {len(constant_features)}")
        for c in constant_features:
            print(f"    -> {c}")
        df = df.drop(columns=constant_features)
    else:
        print("  Constant features: None found")

    # Remove duplicate columns
    numeric_df = df.select_dtypes(include=[np.number])
    cols_list = list(numeric_df.columns)
    duplicate_cols = []
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            if cols_list[j] not in duplicate_cols:
                if numeric_df[cols_list[i]].equals(numeric_df[cols_list[j]]):
                    duplicate_cols.append(cols_list[j])
    if duplicate_cols:
        print(f"  Duplicate features removed: {len(duplicate_cols)}")
        df = df.drop(columns=duplicate_cols)
    else:
        print("  Duplicate features: None found")

    print(f"  Features after quality checks: {df.shape[1]} (removed {cols_before - df.shape[1]})")
    return df


# ============================================================================
# STEP 8: SAVE EXPANDED FEATURE MATRIX
# ============================================================================

def save_expanded_data(df_expanded_profiles, df_feature_matrix):
    """Save expanded company profiles and feature matrix."""
    print("\n" + "=" * 70)
    print("  STEP 8: Saving Expanded Datasets")
    print("=" * 70)

    # Save expanded company profiles
    profiles_path = os.path.join(PROCESSED_DIR, "company_profiles_expanded.csv")
    df_expanded_profiles.to_csv(profiles_path, index=False)
    print(f"  Saved: company_profiles_expanded.csv ({df_expanded_profiles.shape})")

    # Save expanded feature matrix
    matrix_path = os.path.join(PROCESSED_DIR, "feature_matrix_expanded.csv")
    df_feature_matrix.to_csv(matrix_path, index=False)
    print(f"  Saved: feature_matrix_expanded.csv ({df_feature_matrix.shape})")

    return profiles_path, matrix_path


# ============================================================================
# STEP 9: CONSTRUCT PROXY LABELS (same logic as model_training.py)
# ============================================================================

def construct_proxy_labels(df):
    """
    Construct greenwashing risk labels from engineered features.

    5 indicators at 75th percentile threshold:
        1. esg_controversy_divergence > 75th pctl
        2. greenwashing_signal_score > 75th pctl
        3. risk_controversy_mismatch == 1
        4. controversy_risk_ratio > 75th pctl
        5. combined_anomaly_score > 75th pctl

    Binary label: score >= 2 out of 5 = potential greenwashing
    """
    print("\n" + "=" * 70)
    print("  STEP 9: Constructing Proxy Greenwashing Labels")
    print("=" * 70)

    df = df.copy()

    # Indicator 1: ESG-Controversy Divergence
    if 'esg_controversy_divergence' in df.columns:
        threshold_1 = df['esg_controversy_divergence'].quantile(0.75)
        df['gw_indicator_1'] = (df['esg_controversy_divergence'] > threshold_1).astype(int)
        print(f"  Indicator 1 (ESG-Controversy Divergence > {threshold_1:.3f}): "
              f"{df['gw_indicator_1'].sum()} flagged")
    else:
        df['gw_indicator_1'] = 0
        print("  Indicator 1: esg_controversy_divergence not found, defaulting to 0")

    # Indicator 2: Greenwashing Linguistic Score
    if 'greenwashing_signal_score' in df.columns:
        threshold_2 = df['greenwashing_signal_score'].quantile(0.75)
        df['gw_indicator_2'] = (df['greenwashing_signal_score'] > threshold_2).astype(int)
        print(f"  Indicator 2 (GW Linguistic Score > {threshold_2:.4f}): "
              f"{df['gw_indicator_2'].sum()} flagged")
    else:
        df['gw_indicator_2'] = 0
        print("  Indicator 2: greenwashing_signal_score not found, defaulting to 0")

    # Indicator 3: Risk-Controversy Mismatch
    if 'risk_controversy_mismatch' in df.columns:
        df['gw_indicator_3'] = df['risk_controversy_mismatch'].astype(int)
        print(f"  Indicator 3 (Risk-Controversy Mismatch): "
              f"{df['gw_indicator_3'].sum()} flagged")
    else:
        df['gw_indicator_3'] = 0
        print("  Indicator 3: risk_controversy_mismatch not found, defaulting to 0")

    # Indicator 4: High Controversy-to-Risk Ratio
    if 'controversy_risk_ratio' in df.columns:
        threshold_4 = df['controversy_risk_ratio'].quantile(0.75)
        df['gw_indicator_4'] = (df['controversy_risk_ratio'] > threshold_4).astype(int)
        print(f"  Indicator 4 (Controversy-Risk Ratio > {threshold_4:.4f}): "
              f"{df['gw_indicator_4'].sum()} flagged")
    else:
        df['gw_indicator_4'] = 0
        print("  Indicator 4: controversy_risk_ratio not found, defaulting to 0")

    # Indicator 5: Combined Anomaly Score
    if 'combined_anomaly_score' in df.columns:
        threshold_5 = df['combined_anomaly_score'].quantile(0.75)
        df['gw_indicator_5'] = (df['combined_anomaly_score'] > threshold_5).astype(int)
        print(f"  Indicator 5 (Combined Anomaly > {threshold_5:.4f}): "
              f"{df['gw_indicator_5'].sum()} flagged")
    else:
        df['gw_indicator_5'] = 0
        print("  Indicator 5: combined_anomaly_score not found, defaulting to 0")

    # Aggregate: Total proxy score (0-5)
    df['gw_proxy_score'] = (
        df['gw_indicator_1'] + df['gw_indicator_2'] + df['gw_indicator_3']
        + df['gw_indicator_4'] + df['gw_indicator_5']
    )

    # Binary label: score >= 2 = potential greenwashing
    df['gw_label_binary'] = (df['gw_proxy_score'] >= 2).astype(int)

    # Multi-class label
    df['gw_label_multiclass'] = pd.cut(
        df['gw_proxy_score'], bins=[-1, 1, 3, 5], labels=[0, 1, 2],
    ).astype(int)

    # Print summary
    print(f"\n  --- PROXY LABEL SUMMARY ---")
    print(f"  GW Proxy Score distribution:")
    for score in range(6):
        count = (df['gw_proxy_score'] == score).sum()
        pct = count / len(df) * 100
        bar = '#' * int(pct)
        print(f"    Score {score}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  Binary Label (gw_label_binary):")
    print(f"    0 (Not Greenwashing): {(df['gw_label_binary'] == 0).sum():4d} "
          f"({(df['gw_label_binary'] == 0).mean()*100:.1f}%)")
    print(f"    1 (Greenwashing):     {(df['gw_label_binary'] == 1).sum():4d} "
          f"({(df['gw_label_binary'] == 1).mean()*100:.1f}%)")

    return df


# ============================================================================
# STEP 10: PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(df, target_col='gw_label_binary', test_size=0.2, random_state=42):
    """Prepare feature matrix and target for model training."""
    print(f"\n{'=' * 70}")
    print(f"  STEP 10: Preparing Training Data (target: {target_col})")
    print(f"{'=' * 70}")

    # Columns to exclude (leakage + identifiers)
    exclude_columns = {
        'gw_indicator_1', 'gw_indicator_2', 'gw_indicator_3',
        'gw_indicator_4', 'gw_indicator_5',
        'gw_proxy_score', 'gw_label_binary', 'gw_label_multiclass',
        'symbol', 'company_name', 'sector', 'industry',
        'description', 'source',
        'esg_controversy_segment', 'sector_risk_segment',
    }

    # Select numeric features not in exclusion list
    feature_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude_columns
    ]

    X = df[feature_columns].copy()
    y = df[target_col].copy()

    # Handle inf and NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {dict(y.value_counts().sort_index())}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_columns, index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler


# ============================================================================
# STEP 11: MODEL CONFIGURATIONS
# ============================================================================

def get_model_configs():
    """Define all models with hyperparameter grids for GridSearchCV."""
    print(f"\n{'=' * 70}")
    print("  STEP 11: Configuring Models and Hyperparameter Grids")
    print(f"{'=' * 70}")

    models = {
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8],
            },
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(
                random_state=42, class_weight='balanced', n_jobs=-1,
            ),
            'param_grid': {
                'n_estimators': [200, 300],
                'max_depth': [5, 7, 10],
                'min_samples_split': [5, 10],
            },
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000,
            ),
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
            },
        },
        'SVM': {
            'estimator': SVC(
                random_state=42, class_weight='balanced', probability=True,
            ),
            'param_grid': {
                'C': [1.0, 10.0],
                'kernel': ['rbf'],
            },
        },
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        models['XGBoost'] = {
            'estimator': XGBClassifier(
                random_state=42, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1,
            ),
            'param_grid': {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
            },
        }

    for name, config in models.items():
        grid_size = 1
        for param, values in config['param_grid'].items():
            grid_size *= len(values)
        print(f"  {name:25s}: {grid_size:4d} hyperparameter combinations")

    return models


# ============================================================================
# STEP 12: TRAIN MODELS
# ============================================================================

def train_models(X_train, y_train, models, cv_folds=5):
    """Train all models with GridSearchCV and 5-fold stratified CV."""
    print(f"\n{'=' * 70}")
    print("  STEP 12: Training Models with Cross-Validation")
    print(f"{'=' * 70}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained_models = {}

    for name, config in models.items():
        print(f"\n  --- Training: {name} ---")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['param_grid'],
            cv=cv, scoring='f1_weighted',
            n_jobs=-1, verbose=0, refit=True,
        )
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time

        trained_models[name] = {
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'training_time': elapsed,
        }

        print(f"    Best CV F1 (weighted): {grid_search.best_score_:.4f}")
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    Training time: {elapsed:.2f}s")

    return trained_models


# ============================================================================
# STEP 13: TRAIN ISOLATION FOREST
# ============================================================================

def train_isolation_forest(X_train, contamination=0.15):
    """Train Isolation Forest for unsupervised anomaly detection."""
    print(f"\n  --- Training: Isolation Forest (Unsupervised) ---")
    start_time = time.time()

    iso_forest = IsolationForest(
        n_estimators=300, contamination=contamination,
        max_samples='auto', random_state=42, n_jobs=-1,
    )
    iso_forest.fit(X_train)

    train_predictions = iso_forest.predict(X_train)
    train_anomaly_labels = (train_predictions == -1).astype(int)
    elapsed = time.time() - start_time

    print(f"    Contamination: {contamination:.2%}")
    print(f"    Anomalies detected (train): {train_anomaly_labels.sum()} / {len(train_anomaly_labels)}")
    print(f"    Training time: {elapsed:.2f}s")

    return {
        'model': iso_forest,
        'train_predictions': train_anomaly_labels,
        'anomaly_scores': iso_forest.decision_function(X_train),
        'training_time': elapsed,
    }


# ============================================================================
# STEP 14: EVALUATE MODELS
# ============================================================================

def evaluate_models(trained_models, X_test, y_test, feature_names):
    """Evaluate all models on test set."""
    print(f"\n{'=' * 70}")
    print("  STEP 14: Evaluating Models on Test Set")
    print(f"{'=' * 70}")

    evaluation_results = {}

    for name, model_info in trained_models.items():
        model = model_info['best_estimator']
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        roc_auc = 0.0
        if y_prob is not None:
            try:
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:
                roc_auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.Series(
                np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_),
                index=feature_names
            ).sort_values(ascending=False)

        evaluation_results[name] = {
            'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc,
            'confusion_matrix': cm, 'classification_report': report,
            'feature_importance': feature_importance,
            'y_pred': y_pred, 'y_prob': y_prob,
        }

        print(f"\n  --- {name} ---")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {roc_auc:.4f}")

    # Leaderboard
    print(f"\n{'=' * 70}")
    print("  MODEL LEADERBOARD (sorted by F1 Score)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1':>10s} {'ROC-AUC':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    sorted_models = sorted(
        evaluation_results.items(), key=lambda x: x[1]['f1_score'], reverse=True,
    )
    for name, metrics in sorted_models:
        print(f"  {name:<25s} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
              f"{metrics['roc_auc']:>10.4f}")

    best_name = sorted_models[0][0]
    print(f"\n  BEST MODEL: {best_name} (F1 = {sorted_models[0][1]['f1_score']:.4f})")

    return evaluation_results


# ============================================================================
# STEP 15: SAVE MODEL RESULTS
# ============================================================================

def save_model_results(trained_models, evaluation_results, df_with_labels, feature_names):
    """Save model metrics, predictions, and feature importance."""
    print(f"\n{'=' * 70}")
    print("  STEP 15: Saving Model Results")
    print(f"{'=' * 70}")

    # Save model metrics
    metrics_rows = []
    for name, metrics in evaluation_results.items():
        metrics_rows.append({
            'model': name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'best_cv_score': trained_models[name]['best_cv_score'],
            'training_time_sec': trained_models[name]['training_time'],
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values('f1_score', ascending=False)
    metrics_path = os.path.join(PROCESSED_DIR, "model_metrics_expanded.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: model_metrics_expanded.csv ({len(metrics_df)} models)")

    # Save feature importance for each model
    all_importance_rows = []
    for name, metrics in evaluation_results.items():
        if metrics['feature_importance'] is not None:
            fi_df = metrics['feature_importance'].reset_index()
            fi_df.columns = ['feature', 'importance']
            fi_df['model'] = name
            all_importance_rows.append(fi_df)

    if all_importance_rows:
        fi_combined = pd.concat(all_importance_rows, ignore_index=True)
        fi_path = os.path.join(PROCESSED_DIR, "feature_importance_expanded.csv")
        fi_combined.to_csv(fi_path, index=False)
        print(f"  Saved: feature_importance_expanded.csv")

    # Save predictions
    pred_cols = ['company_name', 'sector', 'total_esg_risk_score',
                 'controversy_score', 'gw_proxy_score', 'gw_label_binary']
    pred_cols = [c for c in pred_cols if c in df_with_labels.columns]
    pred_df = df_with_labels[pred_cols].copy()
    pred_path = os.path.join(PROCESSED_DIR, "predictions_expanded.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: predictions_expanded.csv ({len(pred_df)} companies)")

    return metrics_df


# ============================================================================
# STEP 16: COMPARE OLD VS NEW METRICS
# ============================================================================

def compare_metrics():
    """Compare old (480-company) metrics with new (expanded) metrics."""
    print(f"\n{'=' * 70}")
    print("  STEP 16: Comparing Old vs New Model Metrics")
    print(f"{'=' * 70}")

    # Load old metrics
    old_path = os.path.join(PROCESSED_DIR, "model_metrics.csv")
    if not os.path.exists(old_path):
        print("  WARNING: Old model_metrics.csv not found. Skipping comparison.")
        return

    old_metrics = pd.read_csv(old_path)
    new_metrics = pd.read_csv(os.path.join(PROCESSED_DIR, "model_metrics_expanded.csv"))

    print(f"\n  {'Model':<25s} {'Old F1':>10s} {'New F1':>10s} {'Delta':>10s} "
          f"{'Old AUC':>10s} {'New AUC':>10s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for _, new_row in new_metrics.iterrows():
        model_name = new_row['model']
        old_row = old_metrics[old_metrics['model'] == model_name]

        if len(old_row) > 0:
            old_f1 = old_row.iloc[0]['f1_score']
            new_f1 = new_row['f1_score']
            delta_f1 = new_f1 - old_f1

            old_auc = old_row.iloc[0]['roc_auc']
            new_auc = new_row['roc_auc']
            delta_auc = new_auc - old_auc

            delta_f1_str = f"{'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}"
            delta_auc_str = f"{'+' if delta_auc >= 0 else ''}{delta_auc:.4f}"

            print(f"  {model_name:<25s} {old_f1:>10.4f} {new_f1:>10.4f} {delta_f1_str:>10s} "
                  f"{old_auc:>10.4f} {new_auc:>10.4f} {delta_auc_str:>10s}")
        else:
            print(f"  {model_name:<25s} {'N/A':>10s} {new_row['f1_score']:>10.4f} {'N/A':>10s} "
                  f"{'N/A':>10s} {new_row['roc_auc']:>10.4f} {'N/A':>10s}")


# ============================================================================
# STEP 17: RISK SCORING
# ============================================================================

def run_risk_scoring(df):
    """
    Run the 5-component weighted risk scoring formula.

    Components (same as risk_scoring.py):
        1. Proxy score (40%): gw_proxy_score / 5 * 100
        2. Linguistic GW score (15%): greenwashing_signal_score * 100
        3. Controversy-ESG divergence (15%): MinMax scaled to 0-100
        4. Inverted credibility (15%): (1 - claim_credibility_score) * 100
        5. Controversy-risk ratio (15%): MinMax scaled to 0-100
    """
    print(f"\n{'=' * 70}")
    print("  STEP 17: Computing Greenwashing Risk Scores")
    print(f"{'=' * 70}")

    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Component 1: Proxy Score (0-5 -> 0-100)
    if 'gw_proxy_score' in df.columns:
        df['comp_proxy'] = (df['gw_proxy_score'] / 5.0 * 100).clip(0, 100)
    else:
        df['comp_proxy'] = 50.0

    # Component 2: Linguistic GW Score
    if 'greenwashing_signal_score' in df.columns:
        df['comp_linguistic'] = (df['greenwashing_signal_score'] * 100).clip(0, 100)
    else:
        df['comp_linguistic'] = 50.0

    # Component 3: Controversy-ESG Divergence
    if 'esg_controversy_divergence' in df.columns:
        divergence = df[['esg_controversy_divergence']].copy()
        df['comp_divergence'] = scaler.fit_transform(divergence).flatten()
    else:
        df['comp_divergence'] = 50.0

    # Component 4: Inverted Claim Credibility
    if 'claim_credibility_score' in df.columns:
        df['comp_credibility_inv'] = ((1.0 - df['claim_credibility_score']) * 100).clip(0, 100)
    else:
        df['comp_credibility_inv'] = 50.0

    # Component 5: Controversy-Risk Ratio
    if 'controversy_risk_ratio' in df.columns:
        ratio = df[['controversy_risk_ratio']].copy()
        df['comp_controversy_ratio'] = scaler.fit_transform(ratio).flatten()
    else:
        df['comp_controversy_ratio'] = 50.0

    # Final weighted score
    df['risk_score'] = (
        0.40 * df['comp_proxy']
        + 0.15 * df['comp_linguistic']
        + 0.15 * df['comp_divergence']
        + 0.15 * df['comp_credibility_inv']
        + 0.15 * df['comp_controversy_ratio']
    ).round(2).clip(0, 100)

    # Risk tiers
    df['risk_tier'] = pd.cut(
        df['risk_score'], bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'],
        include_lowest=True,
    )

    # Print statistics
    print(f"\n  Risk Score Statistics:")
    print(f"    Mean:   {df['risk_score'].mean():.2f}")
    print(f"    Median: {df['risk_score'].median():.2f}")
    print(f"    Std:    {df['risk_score'].std():.2f}")
    print(f"    Min:    {df['risk_score'].min():.2f}")
    print(f"    Max:    {df['risk_score'].max():.2f}")

    print(f"\n  Risk Tier Distribution:")
    tier_counts = df['risk_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        bar = '#' * int(pct / 2)
        print(f"    {tier:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Save ranked output
    output_cols = [
        'company_name', 'sector', 'risk_score', 'risk_tier',
        'total_esg_risk_score', 'controversy_score', 'source',
    ]
    optional = ['gw_proxy_score', 'comp_proxy', 'comp_linguistic',
                'comp_divergence', 'comp_credibility_inv', 'comp_controversy_ratio']
    for col in optional:
        if col in df.columns:
            output_cols.append(col)
    output_cols = [c for c in output_cols if c in df.columns]

    ranked_df = df[output_cols].copy()
    ranked_df = ranked_df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    ranked_df.index += 1
    ranked_df.index.name = 'rank'

    risk_path = os.path.join(PROCESSED_DIR, "greenwashing_risk_scores_expanded.csv")
    ranked_df.to_csv(risk_path)
    print(f"\n  Saved: greenwashing_risk_scores_expanded.csv ({len(ranked_df)} companies)")

    # Print top 10
    print(f"\n  TOP 10 HIGHEST RISK COMPANIES:")
    print(f"  {'Rank':<6s} {'Company':<40s} {'Score':>7s} {'Tier':<18s} {'Source':<15s}")
    print(f"  {'-'*6} {'-'*40} {'-'*7} {'-'*18} {'-'*15}")
    for rank, row in ranked_df.head(10).iterrows():
        print(f"  {rank:<6d} {str(row['company_name'])[:39]:<40s} "
              f"{row['risk_score']:>7.2f} {str(row['risk_tier']):<18s} "
              f"{str(row.get('source', 'N/A'))[:14]:<15s}")

    return ranked_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete integration pipeline."""
    print("\n" + "#" * 70)
    print("  ESG GREENWASHING DETECTION - NEW DATASET INTEGRATION")
    print("#" * 70)

    pipeline_start = time.time()

    # ---- Step 1: Load datasets ----
    df_existing, df_new = load_datasets()

    # ---- Step 2: Normalize new ESG scores ----
    df_new = normalize_new_scores(df_new)

    # ---- Step 3: Map grades to levels ----
    df_new = map_grades_to_levels(df_new)

    # ---- Step 4: Create expanded company profiles ----
    df_expanded, n_original = create_expanded_profiles(df_existing, df_new)

    # ---- Save expanded profiles ----
    profiles_path = os.path.join(PROCESSED_DIR, "company_profiles_expanded.csv")
    df_expanded.to_csv(profiles_path, index=False)
    print(f"\n  Saved: company_profiles_expanded.csv ({df_expanded.shape})")

    # ---- Steps 5-6: Feature engineering with NLP imputation ----
    df_features, nlp_medians = run_feature_engineering(df_expanded, n_original)

    # ---- Step 7: Feature quality checks ----
    df_features = run_quality_checks(df_features)

    # ---- Step 8: Save expanded feature matrix ----
    matrix_path = os.path.join(PROCESSED_DIR, "feature_matrix_expanded.csv")
    df_features.to_csv(matrix_path, index=False)
    print(f"\n  Saved: feature_matrix_expanded.csv ({df_features.shape})")

    # ---- Step 9: Construct proxy labels ----
    df_features = construct_proxy_labels(df_features)

    # ---- Step 10: Prepare training data ----
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_training_data(
        df_features, target_col='gw_label_binary', test_size=0.2,
    )

    # ---- Step 11: Configure models ----
    models = get_model_configs()

    # ---- Step 12: Train supervised models ----
    trained_models = train_models(X_train, y_train, models, cv_folds=3)

    # ---- Step 13: Train Isolation Forest ----
    iso_results = train_isolation_forest(X_train, contamination=0.15)

    # ---- Step 14: Evaluate all models ----
    evaluation_results = evaluate_models(trained_models, X_test, y_test, feature_names)

    # ---- Step 15: Save model results ----
    metrics_df = save_model_results(trained_models, evaluation_results, df_features, feature_names)

    # ---- Step 16: Compare old vs new metrics ----
    compare_metrics()

    # ---- Step 17: Risk scoring ----
    ranked_df = run_risk_scoring(df_features)

    # ---- Final summary ----
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    print(f"\n{'#' * 70}")
    print("  INTEGRATION PIPELINE COMPLETE")
    print(f"{'#' * 70}")
    print(f"  Total pipeline time: {total_time:.2f} seconds")
    print(f"  Original dataset: 480 companies")
    print(f"  Expanded dataset: {len(df_features)} companies")
    print(f"  Feature matrix shape: {df_features.shape}")
    print(f"  Models trained: {len(trained_models)} supervised + 1 unsupervised")
    print(f"  Best model: {metrics_df.iloc[0]['model']} (F1 = {metrics_df.iloc[0]['f1_score']:.4f})")
    print(f"\n  Output files:")
    print(f"    - company_profiles_expanded.csv")
    print(f"    - feature_matrix_expanded.csv")
    print(f"    - model_metrics_expanded.csv")
    print(f"    - predictions_expanded.csv")
    print(f"    - feature_importance_expanded.csv")
    print(f"    - greenwashing_risk_scores_expanded.csv")
    print(f"\n  Results saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
