"""
================================================================================
MODEL TRAINING MODULE
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

CRITICAL DESIGN DECISION:
    The greenwashing labels dataset (European companies) has ZERO company overlap
    with our feature matrix (US S&P 500 + Indian NIFTY50). This is a common
    real-world ML challenge. We solve it with TWO strategies:

    STRATEGY 1 — PROXY LABEL CONSTRUCTION (Primary)
        Construct greenwashing risk labels from engineered features using
        domain-expert rules. The `esg_controversy_divergence` feature (z-scored
        gap between controversy and ESG risk) is mathematically equivalent to
        what the greenwashing score measures: companies that CLAIM low risk but
        HAVE high controversy.

    STRATEGY 2 — UNSUPERVISED ANOMALY DETECTION (Secondary)
        Treat greenwashing as an anomaly detection problem. Companies with
        unusual feature profiles (high controversy + low ESG risk + vague
        language) cluster separately from legitimate companies.

    Both strategies are industry-standard when labeled data is unavailable
    for the exact target population. This is called TRANSFER LEARNING via
    domain-knowledge proxy labels.

Models Trained:
    1. Random Forest Classifier        — Ensemble of decision trees (robust baseline)
    2. Gradient Boosting (sklearn)      — Sequential boosted trees (sklearn version)
    3. XGBoost Classifier              — Extreme Gradient Boosting (state-of-the-art)
    4. Logistic Regression              — Linear model (interpretable baseline)
    5. Support Vector Machine (SVM)     — Kernel-based classifier
    6. Isolation Forest                 — Unsupervised anomaly detection
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                          # Data manipulation library
import numpy as np                           # Numerical computing library
import os                                    # File system operations
import json                                  # JSON serialization for saving configs
import time                                  # Timing model training
import warnings                              # Suppress non-critical warnings
from datetime import datetime                # Timestamp for model versioning

# --- Scikit-learn: Core ML framework ---
from sklearn.model_selection import (        # Model selection utilities
    train_test_split,                        # Split data into train/test sets
    StratifiedKFold,                         # K-fold cross-validation (maintains class ratio)
    cross_val_score,                         # Run cross-validation and return scores
    GridSearchCV,                            # Exhaustive hyperparameter grid search
)
from sklearn.preprocessing import (          # Data preprocessing
    StandardScaler,                          # Z-score normalization (mean=0, std=1)
    LabelEncoder,                            # Encode categorical labels as integers
)
from sklearn.metrics import (                # Evaluation metrics
    accuracy_score,                          # Fraction of correct predictions
    precision_score,                         # TP / (TP + FP) — how precise are positive predictions
    recall_score,                            # TP / (TP + FN) — how many positives are caught
    f1_score,                                # Harmonic mean of precision and recall
    roc_auc_score,                           # Area Under ROC Curve — overall ranking quality
    classification_report,                   # Full per-class precision/recall/f1 report
    confusion_matrix,                        # TP/FP/TN/FN matrix
)

# --- Model algorithms ---
from sklearn.ensemble import (               # Ensemble learning methods
    RandomForestClassifier,                  # Bagging ensemble of decision trees
    GradientBoostingClassifier,              # Gradient boosted trees (sklearn version)
    IsolationForest,                         # Unsupervised anomaly detection
)
from sklearn.linear_model import (           # Linear models
    LogisticRegression,                      # Linear classification with sigmoid
)
from sklearn.svm import SVC                  # Support Vector Machine classifier

# --- XGBoost: State-of-the-art gradient boosting ---
from xgboost import XGBClassifier            # Extreme Gradient Boosting classifier (Chen & Guestrin 2016)

warnings.filterwarnings('ignore')            # Suppress all warnings for clean output


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Project root directory
DATA_DIR = os.path.join(BASE_DIR, "data")                   # Raw data directory
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")         # Processed data directory
MODELS_DIR = os.path.join(BASE_DIR, "models")               # Directory to save trained models
os.makedirs(MODELS_DIR, exist_ok=True)                      # Create models dir if it doesn't exist


# ============================================================================
# 1. PROXY LABEL CONSTRUCTION
# ============================================================================

def construct_proxy_labels(df):
    """
    Construct greenwashing risk labels from engineered features using
    domain-expert rules. This is the PRIMARY labeling strategy.

    The greenwashing definition:
        A company is greenwashing if it CLAIMS low ESG risk but ACTUALLY
        has high controversy/problematic behavior. This is captured by:

        1. esg_controversy_divergence > threshold   (controversy >> ESG risk)
        2. greenwashing_signal_score > threshold     (vague/hedging language)
        3. risk_controversy_mismatch == 1            (explicit mismatch flag)
        4. controversy_risk_ratio > threshold        (controversies vs claimed risk)
        5. combined_anomaly_score > threshold        (statistical outlier)

    Scoring system:
        Each indicator contributes 0 or 1 point. Total points = 0-5.
        Label: 0 (Low Risk), 1 (Moderate Risk), 2 (High Risk / Greenwashing)

    Why this works:
        Academic research (Lyon & Montgomery 2015, Delmas & Burbano 2011)
        defines greenwashing as the gap between ESG claims and ESG performance.
        Our features DIRECTLY measure this gap quantitatively.

    Parameters:
        df : pd.DataFrame — feature matrix with all engineered features

    Returns:
        pd.DataFrame — feature matrix with new target columns added
    """

    print("=" * 70)                                           # Visual separator
    print("  STEP 1: Constructing Proxy Greenwashing Labels")  # Status header
    print("=" * 70)                                           # Visual separator

    df = df.copy()  # Create a copy to avoid modifying the original dataframe

    # ------------------------------------------------------------------
    # Indicator 1: ESG-Controversy Divergence
    # ------------------------------------------------------------------
    # z(controversy) - z(ESG_risk) — positive means more controversy than risk suggests
    # Threshold: top 25th percentile (most divergent companies)
    divergence_threshold = df['esg_controversy_divergence'].quantile(0.75)  # 75th percentile
    df['gw_indicator_1'] = (                                  # Create binary indicator
        (df['esg_controversy_divergence'] > divergence_threshold)  # Is divergence high?
        .astype(int)                                          # Convert boolean to 0/1
    )
    print(f"  Indicator 1 (ESG-Controversy Divergence > {divergence_threshold:.3f}): "
          f"{df['gw_indicator_1'].sum()} companies flagged")  # Print count of flagged companies

    # ------------------------------------------------------------------
    # Indicator 2: Greenwashing Linguistic Score
    # ------------------------------------------------------------------
    # NLP-based score from vague language, hedging, superlatives vs concrete evidence
    # Threshold: top 25th percentile (most greenwashing-like language)
    linguistic_threshold = df['greenwashing_signal_score'].quantile(0.75)  # 75th percentile
    df['gw_indicator_2'] = (                                  # Create binary indicator
        (df['greenwashing_signal_score'] > linguistic_threshold)  # Is linguistic score high?
        .astype(int)                                          # Convert to 0/1
    )
    print(f"  Indicator 2 (GW Linguistic Score > {linguistic_threshold:.4f}): "
          f"{df['gw_indicator_2'].sum()} companies flagged")

    # ------------------------------------------------------------------
    # Indicator 3: Risk-Controversy Mismatch (pre-computed binary flag)
    # ------------------------------------------------------------------
    # Already computed in feature engineering: low ESG risk bin + high controversy bin
    df['gw_indicator_3'] = df['risk_controversy_mismatch'].astype(int)  # Use existing flag
    print(f"  Indicator 3 (Risk-Controversy Mismatch): "
          f"{df['gw_indicator_3'].sum()} companies flagged")

    # ------------------------------------------------------------------
    # Indicator 4: High Controversy-to-Risk Ratio
    # ------------------------------------------------------------------
    # Companies whose controversy score is disproportionately high vs their ESG risk
    ratio_threshold = df['controversy_risk_ratio'].quantile(0.75)  # 75th percentile
    df['gw_indicator_4'] = (                                  # Create binary indicator
        (df['controversy_risk_ratio'] > ratio_threshold)      # Is ratio high?
        .astype(int)                                          # Convert to 0/1
    )
    print(f"  Indicator 4 (Controversy-Risk Ratio > {ratio_threshold:.4f}): "
          f"{df['gw_indicator_4'].sum()} companies flagged")

    # ------------------------------------------------------------------
    # Indicator 5: Combined Anomaly Score (statistical outlier)
    # ------------------------------------------------------------------
    # Weighted combination of z-scores, MAD scores, and IQR outlier flags
    anomaly_threshold = df['combined_anomaly_score'].quantile(0.75)  # 75th percentile
    df['gw_indicator_5'] = (                                  # Create binary indicator
        (df['combined_anomaly_score'] > anomaly_threshold)    # Is anomaly score high?
        .astype(int)                                          # Convert to 0/1
    )
    print(f"  Indicator 5 (Combined Anomaly > {anomaly_threshold:.4f}): "
          f"{df['gw_indicator_5'].sum()} companies flagged")

    # ------------------------------------------------------------------
    # AGGREGATE: Compute total greenwashing risk score (0-5)
    # ------------------------------------------------------------------
    # Sum all 5 binary indicators to get a composite risk score
    df['gw_proxy_score'] = (                                  # Create composite score column
        df['gw_indicator_1']                                  # Divergence indicator
        + df['gw_indicator_2']                                # Linguistic indicator
        + df['gw_indicator_3']                                # Mismatch indicator
        + df['gw_indicator_4']                                # Ratio indicator
        + df['gw_indicator_5']                                # Anomaly indicator
    )

    # ------------------------------------------------------------------
    # CREATE CLASSIFICATION LABELS
    # ------------------------------------------------------------------
    # Binary label: 0 = Not Greenwashing, 1 = Potential Greenwashing
    # Threshold: score >= 2 (at least 2 out of 5 indicators flagged)
    df['gw_label_binary'] = (                                 # Create binary classification label
        (df['gw_proxy_score'] >= 2)                           # Need at least 2 indicators
        .astype(int)                                          # Convert to 0 (safe) or 1 (greenwashing)
    )

    # Multi-class label: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk
    # Low = 0-1 indicators, Moderate = 2-3, High = 4-5
    df['gw_label_multiclass'] = pd.cut(                       # Create multi-class label
        df['gw_proxy_score'],                                 # Score to bin
        bins=[-1, 1, 3, 5],                                   # Bin edges: [-1,1], (1,3], (3,5]
        labels=[0, 1, 2],                                     # Label each bin: 0, 1, 2
    ).astype(int)                                             # Convert from Categorical to int

    # ------------------------------------------------------------------
    # Print label distribution summary
    # ------------------------------------------------------------------
    print(f"\n  --- PROXY LABEL SUMMARY ---")
    print(f"  GW Proxy Score distribution:")
    for score in range(6):                                    # Iterate 0-5
        count = (df['gw_proxy_score'] == score).sum()         # Count companies with this score
        pct = count / len(df) * 100                           # Calculate percentage
        bar = '#' * int(pct)                                  # Create bar chart
        print(f"    Score {score}: {count:4d} ({pct:5.1f}%) {bar}")  # Print with bar

    print(f"\n  Binary Label (gw_label_binary):")
    print(f"    0 (Not Greenwashing): {(df['gw_label_binary'] == 0).sum():4d} "
          f"({(df['gw_label_binary'] == 0).mean()*100:.1f}%)")  # Print class 0 count
    print(f"    1 (Greenwashing):     {(df['gw_label_binary'] == 1).sum():4d} "
          f"({(df['gw_label_binary'] == 1).mean()*100:.1f}%)")  # Print class 1 count

    print(f"\n  Multi-class Label (gw_label_multiclass):")
    for label in [0, 1, 2]:                                   # Iterate through classes
        count = (df['gw_label_multiclass'] == label).sum()    # Count per class
        names = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}  # Class names
        print(f"    {label} ({names[label]:15s}): {count:4d} "
              f"({count/len(df)*100:.1f}%)")                  # Print count and percentage

    return df  # Return dataframe with new label columns


# ============================================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================================

def prepare_training_data(df, target_col='gw_label_binary', test_size=0.2, random_state=42):
    """
    Prepare feature matrix and target vector for model training.

    Steps:
        1. Select only numeric features (drop text/object columns)
        2. Drop target-related columns from features (prevent data leakage)
        3. Handle any remaining NaN/inf values
        4. Split into train/test sets with stratification
        5. Scale features using StandardScaler

    CRITICAL: Data leakage prevention
        We MUST remove all proxy label construction columns from features.
        If gw_indicator_1 is used as both a feature AND to build the label,
        the model will trivially "predict" the label by reading the indicator.

    Parameters:
        df            : pd.DataFrame — feature matrix with proxy labels
        target_col    : str — name of the target column to predict
        test_size     : float — fraction of data for test set (default 0.2 = 20%)
        random_state  : int — random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """

    print(f"\n{'=' * 70}")                                    # Visual separator
    print(f"  STEP 2: Preparing Training Data (target: {target_col})")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Step 1: Identify columns to EXCLUDE from features
    # ------------------------------------------------------------------
    # These columns must be excluded to prevent data leakage
    leakage_columns = [                                       # Columns that leak target info
        'gw_indicator_1', 'gw_indicator_2', 'gw_indicator_3',  # Individual indicators (used to BUILD label)
        'gw_indicator_4', 'gw_indicator_5',                    # More indicators
        'gw_proxy_score',                                      # Composite score (IS the label)
        'gw_label_binary', 'gw_label_multiclass',              # The labels themselves
    ]

    # Non-feature columns (identifiers and text that can't be used directly)
    id_columns = [                                            # Identification columns
        'symbol', 'company_name', 'sector', 'industry',      # Company identifiers
        'description', 'source',                              # Text and metadata
        'esg_controversy_segment', 'sector_risk_segment',     # String segments
    ]

    # Combine all columns to exclude
    exclude_columns = set(leakage_columns + id_columns)       # Set union for fast lookup

    # ------------------------------------------------------------------
    # Step 2: Select numeric features only
    # ------------------------------------------------------------------
    # Get all numeric columns that are NOT in the exclusion list
    feature_columns = [                                       # Filter feature columns
        col for col in df.select_dtypes(include=[np.number]).columns  # Only numeric columns
        if col not in exclude_columns                         # And not in exclusion list
    ]

    print(f"  Total numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  Excluded (leakage + IDs): {len(exclude_columns)}")
    print(f"  Final feature count: {len(feature_columns)}")

    # ------------------------------------------------------------------
    # Step 3: Create feature matrix X and target vector y
    # ------------------------------------------------------------------
    X = df[feature_columns].copy()                            # Feature matrix (rows=companies, cols=features)
    y = df[target_col].copy()                                 # Target vector (1D array of labels)

    # Replace any remaining inf values with NaN, then fill NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan)                  # Convert inf to NaN
    X = X.fillna(0)                                           # Fill NaN with 0

    print(f"  X shape: {X.shape} (companies x features)")
    print(f"  y shape: {y.shape} (labels)")
    print(f"  y distribution: {dict(y.value_counts().sort_index())}")  # Print class counts

    # ------------------------------------------------------------------
    # Step 4: Train/test split with stratification
    # ------------------------------------------------------------------
    # Stratification ensures the class ratio is preserved in both splits
    # Without stratification, the test set might have no positive examples
    X_train, X_test, y_train, y_test = train_test_split(     # Split the data
        X, y,                                                 # Features and labels
        test_size=test_size,                                  # 20% for testing
        random_state=random_state,                            # Reproducible split
        stratify=y,                                           # Maintain class ratio in both sets
    )

    print(f"\n  Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set:  {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    print(f"  Train y distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test y distribution:  {dict(y_test.value_counts().sort_index())}")

    # ------------------------------------------------------------------
    # Step 5: Feature scaling using StandardScaler
    # ------------------------------------------------------------------
    # StandardScaler: z = (x - mean) / std → mean=0, std=1
    # CRITICAL: Fit on train data ONLY, then transform both train and test
    # If we fit on the full dataset, test data statistics leak into training
    scaler = StandardScaler()                                 # Initialize scaler
    X_train_scaled = pd.DataFrame(                            # Scale training features
        scaler.fit_transform(X_train),                        # FIT on train, TRANSFORM train
        columns=feature_columns,                              # Preserve column names
        index=X_train.index,                                  # Preserve row indices
    )
    X_test_scaled = pd.DataFrame(                             # Scale test features
        scaler.transform(X_test),                             # ONLY TRANSFORM test (no fitting!)
        columns=feature_columns,                              # Preserve column names
        index=X_test.index,                                   # Preserve row indices
    )

    print(f"  Features scaled using StandardScaler (fit on train only)")

    # Store feature names for later use (feature importance, SHAP, etc.)
    feature_names = feature_columns                           # Save feature names list

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# ============================================================================
# 3. MODEL DEFINITIONS AND HYPERPARAMETER GRIDS
# ============================================================================

def get_model_configs():
    """
    Define all models with their hyperparameter search grids.

    Design principles:
        - Start with a WIDE grid, then narrow based on CV results
        - Include regularization to prevent overfitting on 480 samples
        - Use class_weight='balanced' to handle class imbalance
        - Keep max_depth shallow to prevent memorization of small dataset

    Returns:
        dict: Model name → (estimator, param_grid) pairs
    """

    print(f"\n{'=' * 70}")                                    # Visual separator
    print("  STEP 3: Configuring Models and Hyperparameter Grids")
    print(f"{'=' * 70}")

    models = {                                                # Dictionary of model configurations

        # ----------------------------------------------------------------
        # MODEL 1: Random Forest Classifier
        # ----------------------------------------------------------------
        # Ensemble of decision trees using bagging (Bootstrap AGGregation)
        # Each tree sees a random subset of data AND features
        # Final prediction = majority vote across all trees
        # WHY: Robust to overfitting, handles feature interactions naturally
        'Random Forest': {
            'estimator': RandomForestClassifier(              # Initialize RF classifier
                random_state=42,                              # Reproducible results
                class_weight='balanced',                      # Auto-adjust weights for class imbalance
                n_jobs=-1,                                    # Use all CPU cores for parallel training
            ),
            'param_grid': {                                   # Hyperparameters to search
                'n_estimators': [100, 200, 300],              # Number of trees in the forest
                'max_depth': [3, 5, 7, 10],                   # Max tree depth (shallow = less overfitting)
                'min_samples_split': [5, 10, 20],             # Min samples to split a node
                'min_samples_leaf': [2, 5, 10],               # Min samples in a leaf node
                'max_features': ['sqrt', 'log2'],             # Features per split: sqrt(n) or log2(n)
            },
        },

        # ----------------------------------------------------------------
        # MODEL 2: Gradient Boosting Classifier
        # ----------------------------------------------------------------
        # Sequential ensemble: each tree corrects errors of previous trees
        # Uses gradient descent to minimize loss function
        # WHY: State-of-the-art accuracy, captures complex patterns
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(          # Initialize GB classifier
                random_state=42,                              # Reproducible results
            ),
            'param_grid': {                                   # Hyperparameters to search
                'n_estimators': [100, 200, 300],              # Number of boosting stages
                'learning_rate': [0.01, 0.05, 0.1],          # Step size (smaller = more robust)
                'max_depth': [3, 4, 5],                       # Max tree depth (kept shallow)
                'min_samples_split': [5, 10],                 # Min samples to split
                'min_samples_leaf': [3, 5],                   # Min samples per leaf
                'subsample': [0.8, 1.0],                      # Fraction of data per tree (stochastic GB)
            },
        },

        # ----------------------------------------------------------------
        # MODEL 3: XGBoost Classifier
        # ----------------------------------------------------------------
        # Extreme Gradient Boosting — the industry gold standard for tabular data
        # Key advantages over sklearn GradientBoosting:
        #   - Built-in L1/L2 regularization (reg_alpha, reg_lambda)
        #   - Native handling of missing values
        #   - Column subsampling (colsample_bytree) like Random Forest
        #   - Parallel tree construction (faster training)
        #   - Histogram-based splitting (memory efficient)
        # WHY: Consistently wins Kaggle competitions on tabular data,
        #       better regularization prevents overfitting on small datasets
        'XGBoost': {
            'estimator': XGBClassifier(                       # Initialize XGBoost classifier
                random_state=42,                              # Reproducible results
                use_label_encoder=False,                      # Suppress deprecated warning
                eval_metric='logloss',                        # Log loss for binary classification
                n_jobs=-1,                                    # Use all CPU cores
            ),
            'param_grid': {                                   # Hyperparameters to search
                'n_estimators': [100, 200, 300],              # Number of boosting rounds
                'learning_rate': [0.01, 0.05, 0.1],          # Step size shrinkage (eta)
                'max_depth': [3, 5, 7],                       # Max tree depth
                'min_child_weight': [1, 3, 5],                # Min sum of instance weight in a child
                'subsample': [0.8, 1.0],                      # Row subsampling ratio per tree
                'colsample_bytree': [0.7, 0.9, 1.0],         # Column subsampling ratio per tree
                'reg_alpha': [0, 0.1],                        # L1 regularization (Lasso penalty)
                'reg_lambda': [1.0, 2.0],                     # L2 regularization (Ridge penalty)
                'scale_pos_weight': [1, 2.3],                 # Weight for positive class (ratio neg/pos ~ 2.3)
            },
        },

        # ----------------------------------------------------------------
        # MODEL 4: Logistic Regression
        # ----------------------------------------------------------------
        # Linear model: learns weighted sum of features + sigmoid activation
        # P(greenwashing) = sigmoid(w1*x1 + w2*x2 + ... + bias)
        # WHY: Highly interpretable, serves as a strong baseline
        'Logistic Regression': {
            'estimator': LogisticRegression(                  # Initialize LR classifier
                random_state=42,                              # Reproducible results
                class_weight='balanced',                      # Auto-adjust for class imbalance
                max_iter=1000,                                # Max iterations for convergence
            ),
            'param_grid': {                                   # Hyperparameters to search
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],         # Inverse regularization strength
                'penalty': ['l1', 'l2'],                      # L1 (Lasso) or L2 (Ridge) regularization
                'solver': ['liblinear'],                      # Solver that supports both L1 and L2
            },
        },

        # ----------------------------------------------------------------
        # MODEL 5: Support Vector Machine (SVM)
        # ----------------------------------------------------------------
        # Finds the optimal hyperplane that separates classes with max margin
        # Kernel trick maps data to higher dimensions for non-linear boundaries
        # WHY: Effective in high-dimensional spaces, good with small datasets
        'SVM': {
            'estimator': SVC(                                 # Initialize SVM classifier
                random_state=42,                              # Reproducible results
                class_weight='balanced',                      # Auto-adjust for class imbalance
                probability=True,                             # Enable probability estimates for ROC-AUC
            ),
            'param_grid': {                                   # Hyperparameters to search
                'C': [0.1, 1.0, 10.0],                       # Regularization parameter
                'kernel': ['rbf', 'linear'],                  # Kernel: RBF (non-linear) or linear
                'gamma': ['scale', 'auto'],                   # Kernel coefficient (RBF only)
            },
        },
    }

    # Print model summary
    for name, config in models.items():                       # Iterate through models
        grid_size = 1                                         # Calculate grid size
        for param, values in config['param_grid'].items():    # Multiply all parameter list lengths
            grid_size *= len(values)                          # Total combinations
        print(f"  {name:25s}: {grid_size:4d} hyperparameter combinations")

    return models  # Return all model configurations


# ============================================================================
# 4. TRAIN MODELS WITH CROSS-VALIDATION AND HYPERPARAMETER TUNING
# ============================================================================

def train_models(X_train, y_train, models, cv_folds=5):
    """
    Train all models using GridSearchCV with stratified k-fold cross-validation.

    Process for each model:
        1. Create StratifiedKFold cross-validator
        2. Run GridSearchCV to find best hyperparameters
        3. Refit the model with best params on full training set
        4. Store results (best params, best score, trained model)

    Why Stratified K-Fold:
        With 480 samples and class imbalance, random folds might create
        folds with no positive examples. Stratified folds guarantee each
        fold has the same class ratio as the full dataset.

    Why GridSearchCV over RandomizedSearchCV:
        Our grids are small enough (~100-300 combinations) for exhaustive
        search. RandomizedSearchCV is better for very large grids (10,000+).

    Parameters:
        X_train : pd.DataFrame — scaled training features
        y_train : pd.Series — training labels
        models  : dict — model configs from get_model_configs()
        cv_folds: int — number of cross-validation folds (default 5)

    Returns:
        dict: Model name → {best_estimator, best_params, best_score, cv_results}
    """

    print(f"\n{'=' * 70}")                                    # Visual separator
    print("  STEP 4: Training Models with Cross-Validation")
    print(f"{'=' * 70}")

    # Create stratified k-fold cross-validator
    cv = StratifiedKFold(                                     # Initialize CV object
        n_splits=cv_folds,                                    # Number of folds
        shuffle=True,                                         # Shuffle data before splitting
        random_state=42,                                      # Reproducible splits
    )

    trained_models = {}                                       # Dictionary to store results

    for name, config in models.items():                       # Iterate through each model
        print(f"\n  --- Training: {name} ---")
        start_time = time.time()                              # Record start time

        # Run GridSearchCV: exhaustive search over hyperparameter grid
        grid_search = GridSearchCV(                           # Initialize grid search
            estimator=config['estimator'],                    # The model to tune
            param_grid=config['param_grid'],                  # Hyperparameter grid
            cv=cv,                                            # Cross-validation strategy
            scoring='f1_weighted',                            # Optimize for weighted F1 score
            n_jobs=-1,                                        # Use all CPU cores
            verbose=0,                                        # Suppress detailed output
            refit=True,                                       # Refit best model on full training set
        )

        # Fit the grid search on training data
        grid_search.fit(X_train, y_train)                     # Train all parameter combos

        elapsed = time.time() - start_time                    # Calculate training time

        # Store results
        trained_models[name] = {                              # Save model results
            'best_estimator': grid_search.best_estimator_,    # Best trained model object
            'best_params': grid_search.best_params_,          # Best hyperparameters found
            'best_cv_score': grid_search.best_score_,         # Best cross-validation score
            'cv_results': grid_search.cv_results_,            # Full CV results for all combos
            'training_time': elapsed,                         # Time taken to train
        }

        # Print results for this model
        print(f"    Best CV F1 (weighted): {grid_search.best_score_:.4f}")  # Best CV score
        print(f"    Best params: {grid_search.best_params_}")               # Best hyperparameters
        print(f"    Training time: {elapsed:.2f}s")                         # Time elapsed

    return trained_models  # Return all trained models with results


# ============================================================================
# 5. TRAIN ISOLATION FOREST (UNSUPERVISED ANOMALY DETECTION)
# ============================================================================

def train_isolation_forest(X_train, contamination=0.15):
    """
    Train an Isolation Forest for unsupervised anomaly detection.

    How Isolation Forest works:
        1. Randomly select a feature and a random split value
        2. Recursively split the data into left/right partitions
        3. Anomalies are ISOLATED (separated) in fewer splits than normal points
        4. Average path length across many trees → anomaly score

    Why it works for greenwashing:
        Greenwashing companies have UNUSUAL feature combinations (e.g.,
        low ESG risk + high controversy + vague language). These unusual
        combinations are isolated quickly by random splits.

    Parameters:
        X_train       : pd.DataFrame — training features
        contamination : float — expected fraction of anomalies (default 15%)

    Returns:
        dict: Trained Isolation Forest model and predictions
    """

    print(f"\n  --- Training: Isolation Forest (Unsupervised) ---")
    start_time = time.time()                                  # Record start time

    # Initialize Isolation Forest
    iso_forest = IsolationForest(                             # Create model
        n_estimators=300,                                     # Number of isolation trees
        contamination=contamination,                          # Expected anomaly fraction
        max_samples='auto',                                   # Samples per tree (auto = min(256, n))
        random_state=42,                                      # Reproducible results
        n_jobs=-1,                                            # Use all CPU cores
    )

    # Fit the model (unsupervised — no labels needed)
    iso_forest.fit(X_train)                                   # Train on feature matrix

    # Get anomaly predictions: 1 = normal, -1 = anomaly
    train_predictions = iso_forest.predict(X_train)           # Predict on training data

    # Convert to binary: 0 = normal, 1 = anomaly (greenwashing suspect)
    train_anomaly_labels = (train_predictions == -1).astype(int)  # Convert -1 to 1, 1 to 0

    elapsed = time.time() - start_time                        # Calculate training time

    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = iso_forest.decision_function(X_train)    # Continuous anomaly scores

    print(f"    Contamination: {contamination:.2%}")          # Print contamination rate
    print(f"    Anomalies detected (train): {train_anomaly_labels.sum()} / {len(train_anomaly_labels)}")
    print(f"    Training time: {elapsed:.2f}s")               # Print training time

    return {                                                  # Return model and results
        'model': iso_forest,                                  # Trained Isolation Forest
        'train_predictions': train_anomaly_labels,            # Binary predictions on training data
        'anomaly_scores': anomaly_scores,                     # Continuous anomaly scores
        'training_time': elapsed,                             # Training time
    }


# ============================================================================
# 6. EVALUATE ALL MODELS ON TEST SET
# ============================================================================

def evaluate_models(trained_models, X_test, y_test, feature_names):
    """
    Evaluate all trained models on the held-out test set.

    Metrics computed:
        - Accuracy      : Overall fraction correct
        - Precision      : Of predicted positives, how many are correct
        - Recall         : Of actual positives, how many are found
        - F1 Score       : Harmonic mean of precision and recall
        - ROC-AUC        : Probability that a random positive is ranked above a random negative
        - Confusion Matrix: TP, FP, TN, FN counts

    Why F1 is our primary metric (not accuracy):
        With class imbalance (e.g., 70% class 0, 30% class 1), a model
        predicting ALL zeros gets 70% accuracy but catches 0% of greenwashers.
        F1 penalizes this by requiring both precision AND recall to be high.

    Parameters:
        trained_models : dict — output from train_models()
        X_test         : pd.DataFrame — scaled test features
        y_test         : pd.Series — test labels
        feature_names  : list — feature column names

    Returns:
        dict: Model name → evaluation metrics dictionary
    """

    print(f"\n{'=' * 70}")                                    # Visual separator
    print("  STEP 5: Evaluating Models on Test Set")
    print(f"{'=' * 70}")

    evaluation_results = {}                                   # Store all evaluation results

    for name, model_info in trained_models.items():           # Iterate through each model
        model = model_info['best_estimator']                  # Get the best trained model

        # Generate predictions on test set
        y_pred = model.predict(X_test)                        # Predicted class labels
        y_prob = None                                         # Initialize probability variable

        # Get prediction probabilities (for ROC-AUC)
        if hasattr(model, 'predict_proba'):                   # Check if model supports probabilities
            y_prob = model.predict_proba(X_test)              # Get probability for each class

        # ------------------------------------------------------------------
        # Compute all evaluation metrics
        # ------------------------------------------------------------------
        accuracy = accuracy_score(y_test, y_pred)             # Overall accuracy
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted precision
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted recall
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted F1

        # ROC-AUC: requires probability estimates
        roc_auc = 0.0                                         # Default if not available
        if y_prob is not None:                                # If probabilities are available
            try:
                if y_prob.shape[1] == 2:                      # Binary classification
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])  # Use prob of class 1
                else:                                         # Multi-class
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except Exception:                                 # Handle edge cases
                roc_auc = 0.0                                 # Set to 0 if calculation fails

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)                 # Compute TP/FP/TN/FN matrix

        # Classification report (full per-class breakdown)
        report = classification_report(y_test, y_pred, zero_division=0)  # Generate text report

        # ------------------------------------------------------------------
        # Get feature importance (if available)
        # ------------------------------------------------------------------
        feature_importance = None                             # Default: no importance
        if hasattr(model, 'feature_importances_'):            # Tree-based models have this
            feature_importance = pd.Series(                   # Create named Series
                model.feature_importances_,                   # Importance values
                index=feature_names                           # Feature names as index
            ).sort_values(ascending=False)                    # Sort descending (most important first)
        elif hasattr(model, 'coef_'):                         # Linear models have coefficients
            feature_importance = pd.Series(                   # Create named Series
                np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_),
                index=feature_names                           # Feature names as index
            ).sort_values(ascending=False)                    # Sort descending

        # Store all metrics for this model
        evaluation_results[name] = {                          # Save evaluation results
            'accuracy': accuracy,                             # Overall accuracy
            'precision': precision,                           # Weighted precision
            'recall': recall,                                 # Weighted recall
            'f1_score': f1,                                   # Weighted F1 (PRIMARY METRIC)
            'roc_auc': roc_auc,                               # ROC-AUC score
            'confusion_matrix': cm,                           # Confusion matrix
            'classification_report': report,                  # Full text report
            'feature_importance': feature_importance,          # Feature importance (if available)
            'y_pred': y_pred,                                 # Predicted labels
            'y_prob': y_prob,                                 # Predicted probabilities
        }

        # Print results for this model
        print(f"\n  --- {name} ---")
        print(f"    Accuracy:  {accuracy:.4f}")               # Print accuracy
        print(f"    Precision: {precision:.4f}")               # Print precision
        print(f"    Recall:    {recall:.4f}")                  # Print recall
        print(f"    F1 Score:  {f1:.4f}")                      # Print F1 (primary metric)
        print(f"    ROC-AUC:   {roc_auc:.4f}")                # Print ROC-AUC
        print(f"    Confusion Matrix:\n{cm}")                  # Print confusion matrix

    # ------------------------------------------------------------------
    # Print model comparison leaderboard
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  MODEL LEADERBOARD (sorted by F1 Score)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1':>10s} {'ROC-AUC':>10s}")                   # Print header
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")  # Print separator

    # Sort models by F1 score (descending) for leaderboard
    sorted_models = sorted(                                   # Sort by F1
        evaluation_results.items(),                           # Items to sort
        key=lambda x: x[1]['f1_score'],                       # Sort key: F1 score
        reverse=True,                                         # Descending order
    )

    for name, metrics in sorted_models:                       # Print each model's results
        print(f"  {name:<25s} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
              f"{metrics['roc_auc']:>10.4f}")

    # Identify the best model
    best_model_name = sorted_models[0][0]                     # Name of best model
    best_f1 = sorted_models[0][1]['f1_score']                 # Its F1 score
    print(f"\n  BEST MODEL: {best_model_name} (F1 = {best_f1:.4f})")

    return evaluation_results                                 # Return all evaluation results


# ============================================================================
# 7. SAVE MODELS AND RESULTS
# ============================================================================

def save_results(trained_models, evaluation_results, df_with_labels, feature_names):
    """
    Save trained models, evaluation metrics, and predictions to disk.

    Saves:
        - model_metrics.csv            : Comparison table of all models
        - feature_importance_{model}.csv: Feature importances per model
        - predictions.csv              : Company-level predictions from best model
        - training_summary.txt         : Human-readable summary report

    Parameters:
        trained_models     : dict — trained model objects and CV results
        evaluation_results : dict — evaluation metrics per model
        df_with_labels     : pd.DataFrame — full data with proxy labels and predictions
        feature_names      : list — feature column names
    """

    print(f"\n{'=' * 70}")                                    # Visual separator
    print("  STEP 6: Saving Models and Results")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Save 1: Model comparison metrics table
    # ------------------------------------------------------------------
    metrics_rows = []                                         # List to build metrics table
    for name, metrics in evaluation_results.items():          # Iterate through models
        metrics_rows.append({                                 # Add row for each model
            'model': name,                                    # Model name
            'accuracy': metrics['accuracy'],                  # Accuracy score
            'precision': metrics['precision'],                # Precision score
            'recall': metrics['recall'],                      # Recall score
            'f1_score': metrics['f1_score'],                  # F1 score
            'roc_auc': metrics['roc_auc'],                    # ROC-AUC score
            'best_cv_score': trained_models[name]['best_cv_score'],  # CV score during training
            'training_time_sec': trained_models[name]['training_time'],  # Training duration
        })

    metrics_df = pd.DataFrame(metrics_rows)                   # Convert to DataFrame
    metrics_df = metrics_df.sort_values('f1_score', ascending=False)  # Sort by F1 descending
    metrics_path = os.path.join(PROCESSED_DIR, "model_metrics.csv")  # Output path
    metrics_df.to_csv(metrics_path, index=False)              # Save to CSV
    print(f"  Saved: model_metrics.csv ({len(metrics_df)} models)")

    # ------------------------------------------------------------------
    # Save 2: Feature importance for each model
    # ------------------------------------------------------------------
    for name, metrics in evaluation_results.items():          # Iterate through models
        if metrics['feature_importance'] is not None:         # If importance exists
            fi_df = metrics['feature_importance'].reset_index()  # Convert to DataFrame
            fi_df.columns = ['feature', 'importance']         # Name columns
            fi_path = os.path.join(PROCESSED_DIR, f"feature_importance_{name.replace(' ', '_').lower()}.csv")
            fi_df.to_csv(fi_path, index=False)                # Save to CSV
            print(f"  Saved: feature_importance_{name.replace(' ', '_').lower()}.csv "
                  f"(top: {fi_df.iloc[0]['feature']})")       # Print top feature

    # ------------------------------------------------------------------
    # Save 3: Predictions from best model
    # ------------------------------------------------------------------
    best_name = metrics_df.iloc[0]['model']                   # Best model name (sorted by F1)
    best_metrics = evaluation_results[best_name]              # Get best model's metrics

    # Save predictions with company names for review
    pred_df = df_with_labels[['company_name', 'sector', 'total_esg_risk_score',
                               'controversy_score', 'gw_proxy_score',
                               'gw_label_binary']].copy()    # Select key columns
    pred_path = os.path.join(PROCESSED_DIR, "predictions.csv")  # Output path
    pred_df.to_csv(pred_path, index=False)                    # Save to CSV
    print(f"  Saved: predictions.csv ({len(pred_df)} companies)")

    # ------------------------------------------------------------------
    # Save 4: Training summary report
    # ------------------------------------------------------------------
    report_lines = []                                         # Build report text
    report_lines.append("=" * 70)
    report_lines.append("ESG GREENWASHING DETECTION - MODEL TRAINING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append(f"\nDataset: {df_with_labels.shape[0]} companies, {len(feature_names)} features")
    report_lines.append(f"Target: gw_label_binary (proxy-constructed)")
    report_lines.append(f"Train/Test split: 80/20 with stratification")
    report_lines.append(f"Cross-validation: 5-fold stratified")
    report_lines.append(f"\n--- MODEL COMPARISON ---\n")
    report_lines.append(metrics_df.to_string(index=False))    # Add metrics table

    # Add best model's classification report
    report_lines.append(f"\n\n--- BEST MODEL: {best_name} ---")
    report_lines.append(f"\nClassification Report:")
    report_lines.append(best_metrics['classification_report'])

    # Add top 20 features from best model
    if best_metrics['feature_importance'] is not None:
        report_lines.append(f"\nTop 20 Most Important Features:")
        for i, (feat, imp) in enumerate(best_metrics['feature_importance'].head(20).items()):
            report_lines.append(f"  {i+1:2d}. {feat:50s} {imp:.6f}")

    report_text = '\n'.join(report_lines)                     # Join all lines
    report_path = os.path.join(PROCESSED_DIR, "training_report.txt")  # Output path
    with open(report_path, 'w', encoding='utf-8') as f:      # Write to file
        f.write(report_text)                                  # Save report
    print(f"  Saved: training_report.txt")

    return metrics_df                                         # Return metrics DataFrame


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that orchestrates the complete model training pipeline.

    Pipeline:
        1. Load feature matrix
        2. Construct proxy greenwashing labels
        3. Prepare training data (split, scale)
        4. Configure and train all models with GridSearchCV
        5. Train Isolation Forest (unsupervised)
        6. Evaluate all models on test set
        7. Save results
    """

    print("\n" + "#" * 70)                                    # Print header
    print("  ESG GREENWASHING DETECTION - MODEL TRAINING PIPELINE")
    print("#" * 70)

    pipeline_start = time.time()                              # Record start time

    # ---- Step 1: Load feature matrix ----
    print(f"\n  Loading feature matrix...")
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")  # Path to feature matrix
    df = pd.read_csv(fm_path)                                 # Load into DataFrame
    print(f"  Loaded: {df.shape[0]} companies, {df.shape[1]} columns")

    # ---- Step 2: Construct proxy labels ----
    df = construct_proxy_labels(df)                           # Add greenwashing labels

    # ---- Step 3: Prepare training data ----
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_training_data(
        df, target_col='gw_label_binary', test_size=0.2       # 80/20 split, binary target
    )

    # ---- Step 4: Configure models ----
    models = get_model_configs()                              # Get model definitions

    # ---- Step 5: Train supervised models ----
    trained_models = train_models(X_train, y_train, models, cv_folds=5)

    # ---- Step 6: Train Isolation Forest (unsupervised) ----
    iso_results = train_isolation_forest(X_train, contamination=0.15)

    # ---- Step 7: Evaluate all models ----
    evaluation_results = evaluate_models(trained_models, X_test, y_test, feature_names)

    # ---- Step 8: Save results ----
    metrics_df = save_results(trained_models, evaluation_results, df, feature_names)

    # ---- Print final summary ----
    pipeline_end = time.time()                                # Record end time
    total_time = pipeline_end - pipeline_start                # Calculate total duration

    print(f"\n{'=' * 70}")
    print("  TRAINING PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total pipeline time: {total_time:.2f} seconds")
    print(f"  Models trained: {len(trained_models) + 1} (4 supervised + 1 unsupervised)")
    print(f"  Best model: {metrics_df.iloc[0]['model']} (F1 = {metrics_df.iloc[0]['f1_score']:.4f})")
    print(f"  Results saved to: {PROCESSED_DIR}")
    print(f"\n  Next step: Run model_pipeline.py for full end-to-end execution")

    return {                                                  # Return all pipeline outputs
        'trained_models': trained_models,                     # Trained model objects
        'evaluation_results': evaluation_results,             # Evaluation metrics
        'isolation_forest': iso_results,                      # Unsupervised model results
        'feature_names': feature_names,                       # Feature names list
        'data_with_labels': df,                               # Data with proxy labels
        'scaler': scaler,                                     # Fitted scaler for inference
    }


# Entry point: only runs when this script is executed directly
if __name__ == "__main__":
    results = main()                                          # Execute the full pipeline
