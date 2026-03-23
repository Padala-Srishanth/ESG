"""
================================================================================
MASTER ML PIPELINE — SINGLE COMMAND END-TO-END EXECUTION
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

This is the ONE COMMAND that runs the ENTIRE project from raw data to
final predictions, evaluation plots, and SHAP explanations.

    python model_pipeline.py

Pipeline Phases:
    Phase 1: Data Preprocessing    → Clean 4 raw datasets into unified profiles
    Phase 2: NLP Text Analysis     → Sentiment, claims, keyword extraction
    Phase 3: Feature Engineering   → 121 numerical + NLP + categorical features
    Phase 4: Model Training        → 4 supervised + 1 unsupervised model
    Phase 5: Model Evaluation      → ROC, PR, confusion matrix, comparison plots
    Phase 6: SHAP Explainability   → Why the model flags specific companies
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os                                    # File system operations
import sys                                   # System-level operations
import time                                  # Timing each phase
import pandas as pd                          # Data manipulation
import numpy as np                           # Numerical computing
import warnings                              # Warning suppression
from datetime import datetime                # Timestamp for reports

warnings.filterwarnings('ignore')            # Suppress all warnings for clean output

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Project root directory
DATA_DIR = os.path.join(BASE_DIR, "data")                   # Raw data directory
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")         # Processed outputs directory
PLOTS_DIR = os.path.join(BASE_DIR, "plots")                 # Visualization outputs directory
MODELS_DIR = os.path.join(BASE_DIR, "models")               # Saved models directory

# Create all output directories
for d in [PROCESSED_DIR, PLOTS_DIR, MODELS_DIR]:            # Iterate output dirs
    os.makedirs(d, exist_ok=True)                            # Create if not exists


# ============================================================================
# PHASE 1: DATA PREPROCESSING
# ============================================================================

def run_phase_1():
    """
    Phase 1: Load, clean, transform, and merge all 4 raw datasets.

    Input:  4 raw files in data/
    Output: 5 cleaned CSVs in data/processed/
    """

    print("\n" + "=" * 70)                                    # Phase header
    print("  PHASE 1: DATA PREPROCESSING")                    # Phase title
    print("=" * 70)

    start = time.time()                                       # Record start time

    # Import the preprocessing module
    from data_preprocessing import main as run_preprocessing  # Import main function

    # Execute the full preprocessing pipeline
    preprocessing_results = run_preprocessing()               # Run and capture results

    elapsed = time.time() - start                             # Calculate duration
    print(f"\n  Phase 1 completed in {elapsed:.2f} seconds")  # Print timing

    return preprocessing_results                              # Return all cleaned datasets


# ============================================================================
# PHASE 2: NLP TEXT ANALYSIS
# ============================================================================

def run_phase_2():
    """
    Phase 2: Apply NLP techniques to company description text.

    Input:  Cleaned datasets from Phase 1
    Output: NLP-enriched datasets + extracted ESG claims
    """

    print("\n" + "=" * 70)
    print("  PHASE 2: NLP TEXT ANALYSIS")
    print("=" * 70)

    start = time.time()

    # Import the NLP pipeline module
    from nlp_pipeline import main as run_nlp                  # Import main function

    # Execute the full NLP pipeline
    nlp_results = run_nlp()                                   # Run sentiment + claims + keywords

    elapsed = time.time() - start
    print(f"\n  Phase 2 completed in {elapsed:.2f} seconds")

    return nlp_results


# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================

def run_phase_3():
    """
    Phase 3: Engineer 121+ features from numerical, NLP, and categorical data.

    Input:  Company profiles from Phase 1
    Output: feature_matrix.csv (480 x 169)
    """

    print("\n" + "=" * 70)
    print("  PHASE 3: FEATURE ENGINEERING")
    print("=" * 70)

    start = time.time()

    # Import the feature engineering pipeline class
    from feature_engineering_pipeline import FeatureEngineeringPipeline  # Import pipeline class

    # Create pipeline instance and execute
    pipeline = FeatureEngineeringPipeline(data_dir='data/processed')  # Initialize with data path
    feature_matrix = pipeline.run_full_pipeline()             # Run all feature engineering steps
    feature_results = {'feature_matrix': feature_matrix}      # Wrap result in dict

    elapsed = time.time() - start
    print(f"\n  Phase 3 completed in {elapsed:.2f} seconds")

    return feature_results


# ============================================================================
# PHASE 4: MODEL TRAINING
# ============================================================================

def run_phase_4():
    """
    Phase 4: Train 5 ML models with cross-validation and hyperparameter tuning.

    Input:  feature_matrix.csv from Phase 3
    Output: Trained models, predictions, metrics
    """

    print("\n" + "=" * 70)
    print("  PHASE 4: MODEL TRAINING")
    print("=" * 70)

    start = time.time()

    # Import model training module
    from model_training import (                              # Import required functions
        construct_proxy_labels,                               # Build greenwashing labels
        prepare_training_data,                                # Split and scale data
        get_model_configs,                                    # Define model configurations
        train_models,                                         # Train with GridSearchCV
        train_isolation_forest,                               # Unsupervised anomaly detection
        evaluate_models,                                      # Evaluate on test set
        save_results,                                         # Save outputs
    )

    # Step 4.1: Load feature matrix
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")  # Path to features
    df = pd.read_csv(fm_path)                                 # Load into DataFrame
    print(f"  Loaded feature matrix: {df.shape}")

    # Step 4.2: Construct proxy labels
    df = construct_proxy_labels(df)                           # Add greenwashing labels

    # Step 4.3: Prepare training data
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_training_data(
        df, target_col='gw_label_binary', test_size=0.2
    )

    # Step 4.4: Configure and train models
    models = get_model_configs()                              # Get model configurations
    trained_models = train_models(X_train, y_train, models, cv_folds=5)

    # Step 4.5: Train Isolation Forest
    iso_results = train_isolation_forest(X_train, contamination=0.15)

    # Step 4.6: Evaluate all models
    # Add y_test to evaluation results so evaluation module can access it
    evaluation_results = evaluate_models(trained_models, X_test, y_test, feature_names)

    # Attach y_test and y_prob to results for ROC/PR curves
    for name in evaluation_results:                           # Iterate models
        evaluation_results[name]['y_test'] = y_test           # Attach true labels

    # Step 4.7: Save results
    metrics_df = save_results(trained_models, evaluation_results, df, feature_names)

    elapsed = time.time() - start
    print(f"\n  Phase 4 completed in {elapsed:.2f} seconds")

    return {                                                  # Return all Phase 4 outputs
        'trained_models': trained_models,
        'evaluation_results': evaluation_results,
        'isolation_forest': iso_results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'data_with_labels': df,
        'metrics_df': metrics_df,
    }


# ============================================================================
# PHASE 5: MODEL EVALUATION (PLOTS)
# ============================================================================

def run_phase_5(phase4_results):
    """
    Phase 5: Generate evaluation plots — ROC, PR, confusion matrix, comparison.

    Input:  Evaluation results from Phase 4
    Output: 5 plot images in plots/
    """

    print("\n" + "=" * 70)
    print("  PHASE 5: MODEL EVALUATION & VISUALIZATION")
    print("=" * 70)

    start = time.time()

    # Import evaluation module
    from model_evaluation import run_evaluation               # Import evaluation runner

    # Run all evaluation plots and report
    eval_outputs = run_evaluation(phase4_results['evaluation_results'])

    elapsed = time.time() - start
    print(f"\n  Phase 5 completed in {elapsed:.2f} seconds")

    return eval_outputs


# ============================================================================
# PHASE 6: SHAP EXPLAINABILITY
# ============================================================================

def run_phase_6(phase4_results):
    """
    Phase 6: Generate SHAP explanations for model predictions.

    Input:  Best trained model and test data from Phase 4
    Output: SHAP plots + company-level explanations
    """

    print("\n" + "=" * 70)
    print("  PHASE 6: SHAP EXPLAINABILITY")
    print("=" * 70)

    start = time.time()

    # Import explainability module
    from model_explainability import run_explainability       # Import SHAP runner

    # Find the best model (highest F1 on test set)
    best_name = max(                                          # Model with best F1
        phase4_results['evaluation_results'],                 # Search space
        key=lambda x: phase4_results['evaluation_results'][x]['f1_score']
    )
    best_model = phase4_results['trained_models'][best_name]['best_estimator']
    print(f"  Using best model: {best_name}")

    # Run SHAP analysis on best model
    shap_outputs = run_explainability(                        # Execute SHAP pipeline
        best_model=best_model,                                # Best trained model
        X_train=phase4_results['X_train'],                    # Training data (for background)
        X_test=phase4_results['X_test'],                      # Test data (to explain)
        y_test=phase4_results['y_test'],                      # True labels
        feature_names=phase4_results['feature_names'],        # Feature names
        df_with_labels=phase4_results['data_with_labels'],    # Full data for company names
    )

    elapsed = time.time() - start
    print(f"\n  Phase 6 completed in {elapsed:.2f} seconds")

    return shap_outputs


# ============================================================================
# GENERATE FINAL PIPELINE REPORT
# ============================================================================

def generate_final_report(phase4_results, total_time):
    """
    Generate the final comprehensive pipeline report.

    Parameters:
        phase4_results : dict — all Phase 4 outputs
        total_time     : float — total pipeline execution time
    """

    lines = []                                                # Build report lines
    lines.append("=" * 70)
    lines.append("ESG GREENWASHING DETECTION — FINAL PIPELINE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Pipeline summary
    lines.append(f"\nTotal execution time: {total_time:.2f} seconds")
    lines.append(f"Companies analyzed: {len(phase4_results['data_with_labels'])}")
    lines.append(f"Features engineered: {len(phase4_results['feature_names'])}")
    lines.append(f"Models trained: {len(phase4_results['trained_models'])} supervised + 1 unsupervised")

    # Model leaderboard
    lines.append(f"\n{'='*70}")
    lines.append("MODEL LEADERBOARD")
    lines.append(f"{'='*70}\n")
    lines.append(f"{'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} "
                 f"{'Recall':>10s} {'F1':>10s} {'ROC-AUC':>10s}")
    lines.append("-" * 75)

    sorted_models = sorted(                                   # Sort by F1 descending
        phase4_results['evaluation_results'].items(),
        key=lambda x: x[1]['f1_score'], reverse=True
    )
    for name, m in sorted_models:
        lines.append(f"{name:<25s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                     f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}")

    best_name = sorted_models[0][0]
    best_f1 = sorted_models[0][1]['f1_score']
    lines.append(f"\nBEST MODEL: {best_name} (F1 = {best_f1:.4f})")

    # Greenwashing label summary
    df = phase4_results['data_with_labels']
    lines.append(f"\n{'='*70}")
    lines.append("GREENWASHING RISK DISTRIBUTION")
    lines.append(f"{'='*70}\n")
    for score in range(6):
        count = (df['gw_proxy_score'] == score).sum()
        pct = count / len(df) * 100
        lines.append(f"  Score {score}: {count:4d} ({pct:5.1f}%)")

    lines.append(f"\n  Flagged as greenwashing (score >= 2): "
                 f"{(df['gw_label_binary'] == 1).sum()} / {len(df)} companies")

    # Top features
    best_metrics = phase4_results['evaluation_results'][best_name]
    if best_metrics.get('feature_importance') is not None:
        lines.append(f"\n{'='*70}")
        lines.append(f"TOP 15 PREDICTIVE FEATURES ({best_name})")
        lines.append(f"{'='*70}\n")
        for i, (feat, imp) in enumerate(best_metrics['feature_importance'].head(15).items(), 1):
            lines.append(f"  {i:2d}. {feat:50s} {imp:.6f}")

    # Output files listing
    lines.append(f"\n{'='*70}")
    lines.append("OUTPUT FILES")
    lines.append(f"{'='*70}\n")
    lines.append("  data/processed/")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        lines.append(f"    {f:<45s} {size_kb:>8.1f} KB")

    if os.path.exists(PLOTS_DIR):
        lines.append("\n  plots/")
        for f in sorted(os.listdir(PLOTS_DIR)):
            fpath = os.path.join(PLOTS_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            lines.append(f"    {f:<45s} {size_kb:>8.1f} KB")

    report_text = '\n'.join(lines)

    # Save report
    report_path = os.path.join(BASE_DIR, "PIPELINE_REPORT.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  Final report saved to: PIPELINE_REPORT.txt")
    return report_text


# ============================================================================
# MAIN: MASTER PIPELINE EXECUTION
# ============================================================================

def main():
    """
    Execute the COMPLETE end-to-end ESG Greenwashing Detection pipeline.

    One command runs everything:
        python model_pipeline.py
    """

    # ======================================================================
    # PIPELINE HEADER
    # ======================================================================
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   ESG GREENWASHING DETECTION — MASTER PIPELINE                    #")
    print("#   End-to-End: Raw Data -> Predictions -> Explanations              #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    pipeline_start = time.time()                              # Record overall start time

    # ======================================================================
    # PHASE 1: DATA PREPROCESSING
    # ======================================================================
    phase1_results = run_phase_1()                            # Clean and merge datasets

    # ======================================================================
    # PHASE 2: NLP TEXT ANALYSIS
    # ======================================================================
    phase2_results = run_phase_2()                            # Sentiment + claims + keywords

    # ======================================================================
    # PHASE 3: FEATURE ENGINEERING
    # ======================================================================
    phase3_results = run_phase_3()                            # Engineer 121+ features

    # ======================================================================
    # PHASE 4: MODEL TRAINING
    # ======================================================================
    phase4_results = run_phase_4()                            # Train 5 models + evaluate

    # ======================================================================
    # PHASE 5: MODEL EVALUATION (PLOTS)
    # ======================================================================
    phase5_results = run_phase_5(phase4_results)              # Generate evaluation plots

    # ======================================================================
    # PHASE 6: SHAP EXPLAINABILITY
    # ======================================================================
    phase6_results = run_phase_6(phase4_results)              # SHAP analysis + explanations

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    pipeline_end = time.time()                                # Record end time
    total_time = pipeline_end - pipeline_start                # Total duration

    report = generate_final_report(phase4_results, total_time)  # Generate final report

    # ======================================================================
    # COMPLETION SUMMARY
    # ======================================================================
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   PIPELINE COMPLETE                                                #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    best_name = max(phase4_results['evaluation_results'],
                    key=lambda x: phase4_results['evaluation_results'][x]['f1_score'])
    best_f1 = phase4_results['evaluation_results'][best_name]['f1_score']

    print(f"""
    Summary:
    --------
    Total time:          {total_time:.2f} seconds
    Companies analyzed:  {len(phase4_results['data_with_labels'])}
    Features engineered: {len(phase4_results['feature_names'])}
    Models trained:      {len(phase4_results['trained_models'])} supervised + 1 unsupervised
    Best model:          {best_name} (F1 = {best_f1:.4f})
    Greenwashing flagged: {(phase4_results['data_with_labels']['gw_label_binary'] == 1).sum()} / {len(phase4_results['data_with_labels'])} companies

    Output directories:
    - data/processed/    -- Cleaned data, features, predictions, metrics
    - plots/             -- ROC curves, confusion matrices, SHAP plots
    - PIPELINE_REPORT.txt -- Full pipeline execution report

    To run: python model_pipeline.py
    """)

    return {                                                  # Return all results
        'phase1': phase1_results,
        'phase2': phase2_results,
        'phase3': phase3_results,
        'phase4': phase4_results,
        'phase5': phase5_results,
        'phase6': phase6_results,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = main()                                          # Execute the full pipeline
