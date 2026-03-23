"""
================================================================================
MODEL EXPLAINABILITY MODULE (SHAP)
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

This module provides model interpretability using SHAP (SHapley Additive
exPlanations) — the gold standard for explaining ML model predictions.

SHAP answers: "WHY did the model flag this company as greenwashing?"

Key outputs:
    1. Global Feature Importance (SHAP summary plot)
    2. Per-Company Explanations (SHAP waterfall/force plots)
    3. Feature Interaction Effects (SHAP dependence plots)
    4. Company-Level Risk Reports (text-based explanations)

SHAP Theory (Nobel Prize in Economics, Lloyd Shapley 1953):
    SHAP values measure each feature's contribution to a prediction
    by computing the MARGINAL CONTRIBUTION of that feature across
    all possible feature subsets. It satisfies 4 axioms:
        - Efficiency: contributions sum to prediction
        - Symmetry: features with equal impact get equal values
        - Dummy: irrelevant features get zero value
        - Additivity: linear combination of explanations
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                          # Data manipulation
import numpy as np                           # Numerical computing
import os                                    # File system operations
import warnings                              # Warning suppression
import matplotlib                            # Matplotlib backend config
matplotlib.use('Agg')                        # Non-interactive backend (saves to file)
import matplotlib.pyplot as plt              # Plotting interface

warnings.filterwarnings('ignore')            # Suppress warnings

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Project root
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # Processed data dir
PLOTS_DIR = os.path.join(BASE_DIR, "plots")                 # Plots output dir
os.makedirs(PLOTS_DIR, exist_ok=True)                       # Ensure plots dir exists


# ============================================================================
# 1. COMPUTE SHAP VALUES
# ============================================================================

def compute_shap_values(model, X_train, X_test, feature_names, model_name="Model"):
    """
    Compute SHAP values for the given model and test data.

    SHAP value for feature j, company i:
        shap_value[i][j] = contribution of feature j to company i's prediction
        Positive SHAP = pushes prediction TOWARD greenwashing
        Negative SHAP = pushes prediction AWAY from greenwashing

    For tree-based models (RF, GBM): uses TreeExplainer (exact, fast)
    For linear models (LR): uses LinearExplainer (exact)
    For others (SVM): uses KernelExplainer (approximate, slower)

    Parameters:
        model          : trained sklearn model
        X_train        : pd.DataFrame — training features (for background data)
        X_test         : pd.DataFrame — test features to explain
        feature_names  : list — feature column names
        model_name     : str — name of the model (for logging)

    Returns:
        dict — SHAP values, explainer object, and expected value
    """

    print(f"\n  Computing SHAP values for {model_name}...")   # Status message

    try:
        import shap                                           # Import SHAP library
    except ImportError:                                       # SHAP not installed
        print("    WARNING: shap library not installed. Skipping SHAP analysis.")
        print("    Install with: pip install shap")
        return None                                           # Return None if SHAP unavailable

    explainer = None                                          # Initialize explainer variable
    shap_values = None                                        # Initialize SHAP values variable

    # --- Select the right explainer based on model type ---

    model_type = type(model).__name__                         # Get model class name as string

    if model_type in ['RandomForestClassifier',               # Tree-based models
                      'GradientBoostingClassifier',
                      'XGBClassifier',
                      'LGBMClassifier']:
        # TreeExplainer: EXACT SHAP values for tree ensembles (O(TLD) complexity)
        # Much faster than KernelExplainer and gives exact results
        print(f"    Using TreeExplainer (exact) for {model_type}")
        explainer = shap.TreeExplainer(model)                 # Create tree explainer
        shap_values = explainer.shap_values(X_test)           # Compute SHAP values

        # For binary classification, TreeExplainer returns list of [class_0, class_1]
        if isinstance(shap_values, list):                     # If list of arrays
            shap_values = shap_values[1]                      # Take class 1 (greenwashing)

    elif model_type in ['LogisticRegression']:                # Linear models
        # LinearExplainer: EXACT for linear models
        print(f"    Using LinearExplainer (exact) for {model_type}")
        explainer = shap.LinearExplainer(                     # Create linear explainer
            model,                                            # The linear model
            X_train,                                          # Background data for expected value
        )
        shap_values = explainer.shap_values(X_test)           # Compute SHAP values

    else:                                                     # SVM, KNN, etc.
        # KernelExplainer: Model-agnostic (works with ANY model)
        # Slower but universally applicable
        print(f"    Using KernelExplainer (approximate) for {model_type}")
        # Use a small background sample for efficiency (100 samples)
        background = shap.sample(X_train, min(100, len(X_train)))  # Sample background
        explainer = shap.KernelExplainer(                     # Create kernel explainer
            model.predict_proba,                              # Model prediction function
            background,                                       # Background dataset
        )
        shap_values = explainer.shap_values(                  # Compute SHAP values
            X_test.iloc[:50],                                 # Limit to 50 samples (speed)
            nsamples=100,                                     # Number of samples for approximation
        )
        if isinstance(shap_values, list):                     # Handle multi-class output
            shap_values = shap_values[1]                      # Take class 1

    # Get expected value (base prediction before any features are considered)
    if hasattr(explainer, 'expected_value'):                   # If explainer has expected value
        expected_value = explainer.expected_value              # Get it
        if isinstance(expected_value, (list, np.ndarray)):    # If array (multi-class)
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        expected_value = 0.0                                  # Default

    print(f"    SHAP values shape: {np.array(shap_values).shape}")  # Print shape
    print(f"    Expected value (base prediction): {expected_value:.4f}")

    return {                                                  # Return all SHAP results
        'shap_values': shap_values,                           # SHAP values array
        'explainer': explainer,                               # Explainer object
        'expected_value': expected_value,                     # Base prediction
        'X_test': X_test,                                     # Test data used
        'feature_names': feature_names,                       # Feature names
    }


# ============================================================================
# 2. SHAP SUMMARY PLOT (GLOBAL FEATURE IMPORTANCE)
# ============================================================================

def plot_shap_summary(shap_results, top_n=20, save_path=None):
    """
    Create SHAP summary plot showing global feature importance.

    The summary plot shows:
        - Each dot = one company
        - X-axis = SHAP value (impact on greenwashing prediction)
        - Y-axis = features (sorted by importance)
        - Color = feature value (red=high, blue=low)

    How to read:
        If "controversy_risk_ratio" has red dots on the RIGHT:
        → HIGH controversy_risk_ratio INCREASES greenwashing prediction
        This makes domain sense: high controversy relative to low ESG risk = greenwashing

    Parameters:
        shap_results : dict — from compute_shap_values()
        top_n        : int — number of top features to show
        save_path    : str — path to save plot

    Returns:
        str — path to saved plot
    """

    print(f"  Plotting SHAP Summary (top {top_n} features)...")

    if shap_results is None:                                  # Skip if SHAP not available
        print("    Skipping: No SHAP results available.")
        return None

    try:
        import shap                                           # Import SHAP
    except ImportError:
        return None

    shap_values = shap_results['shap_values']                 # Extract SHAP values
    X_test = shap_results['X_test']                           # Extract test data

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))           # Create figure

    # Create SHAP summary beeswarm plot
    shap.summary_plot(                                        # SHAP's built-in summary plot
        shap_values,                                          # SHAP values
        X_test,                                               # Feature values (for coloring)
        max_display=top_n,                                    # Show top N features
        show=False,                                           # Don't display (we'll save instead)
        plot_size=None,                                       # Use our figure size
    )

    plt.title(f'SHAP Feature Importance — Top {top_n} Greenwashing Predictors',
              fontsize=14, fontweight='bold', pad=20)         # Add title

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')      # Save plot
    plt.close('all')                                          # Close all figures

    print(f"    Saved: {save_path}")
    return save_path


# ============================================================================
# 3. SHAP BAR PLOT (MEAN ABSOLUTE IMPORTANCE)
# ============================================================================

def plot_shap_bar(shap_results, top_n=20, save_path=None):
    """
    Create SHAP bar plot showing mean absolute SHAP values per feature.

    This is a simpler view than the summary plot — just shows which
    features have the BIGGEST impact on predictions, regardless of direction.

    Parameters:
        shap_results : dict — from compute_shap_values()
        top_n        : int — number of features to show
        save_path    : str — path to save plot

    Returns:
        str — path to saved plot
    """

    print(f"  Plotting SHAP Bar Chart (top {top_n})...")

    if shap_results is None:
        print("    Skipping: No SHAP results available.")
        return None

    shap_values = shap_results['shap_values']                 # SHAP values
    feature_names = shap_results['feature_names']             # Feature names

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)          # Average |SHAP| across all companies
    importance_df = pd.DataFrame({                            # Create DataFrame
        'feature': feature_names,                             # Feature names
        'mean_abs_shap': mean_abs_shap,                       # Mean |SHAP| values
    }).sort_values('mean_abs_shap', ascending=False)          # Sort descending

    top = importance_df.head(top_n)                           # Get top N features

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))           # Create figure

    ax.barh(                                                  # Horizontal bar chart
        range(len(top)),                                      # Y positions
        top['mean_abs_shap'].values[::-1],                    # Values (reversed for top-down)
        color='#e74c3c',                                      # Red bars
        edgecolor='#c0392b',                                  # Darker red edge
        alpha=0.85,                                           # Slight transparency
    )
    ax.set_yticks(range(len(top)))                            # Y-tick positions
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=10)  # Feature names
    ax.set_xlabel('Mean |SHAP Value|', fontsize=13)           # X-axis label
    ax.set_title(f'SHAP Feature Importance — Mean |SHAP| (Top {top_n})',
                 fontsize=15, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)                       # Vertical grid

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "shap_bar.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved: {save_path}")
    return save_path


# ============================================================================
# 4. COMPANY-LEVEL EXPLANATIONS
# ============================================================================

def explain_top_companies(shap_results, df_with_labels, n_companies=10, save_path=None):
    """
    Generate text-based explanations for the top flagged companies.

    For each flagged company, explains:
        - Which features contributed MOST to the greenwashing flag
        - The direction (positive = increases risk, negative = decreases risk)
        - The actual feature values for context

    This is what an ESG analyst would use to investigate flagged companies.

    Parameters:
        shap_results    : dict — SHAP values and test data
        df_with_labels  : pd.DataFrame — full dataset with company names and labels
        n_companies     : int — number of top companies to explain
        save_path       : str — path to save explanations

    Returns:
        str — explanation text
    """

    print(f"\n  Generating explanations for top {n_companies} flagged companies...")

    if shap_results is None:
        print("    Skipping: No SHAP results available.")
        return "SHAP analysis not available."

    shap_values = shap_results['shap_values']                 # SHAP values array
    X_test = shap_results['X_test']                           # Test features
    feature_names = shap_results['feature_names']             # Feature names

    # Sum SHAP values per company to get overall greenwashing push
    company_shap_sum = np.sum(shap_values, axis=1)            # Sum across features per row

    # Get indices of companies with highest SHAP sum (most pushed toward greenwashing)
    top_indices = np.argsort(company_shap_sum)[::-1][:n_companies]  # Top N indices

    lines = []                                                # Build explanation text
    lines.append("=" * 70)
    lines.append("COMPANY-LEVEL GREENWASHING EXPLANATIONS (SHAP)")
    lines.append("=" * 70)

    for rank, idx in enumerate(top_indices, 1):               # Iterate top companies

        # Get company info from original DataFrame using test set index
        test_idx = X_test.index[idx]                          # Map to original DataFrame index
        if test_idx < len(df_with_labels):                    # Bounds check
            company_name = df_with_labels.iloc[test_idx].get('company_name', f'Company_{test_idx}')
            sector = df_with_labels.iloc[test_idx].get('sector', 'Unknown')
            gw_score = df_with_labels.iloc[test_idx].get('gw_proxy_score', 'N/A')
        else:
            company_name = f'Company_{test_idx}'
            sector = 'Unknown'
            gw_score = 'N/A'

        lines.append(f"\n{'─'*70}")
        lines.append(f"  RANK #{rank}: {company_name}")
        lines.append(f"  Sector: {sector} | GW Proxy Score: {gw_score}/5")
        lines.append(f"  Total SHAP Push: {company_shap_sum[idx]:+.4f}")
        lines.append(f"{'─'*70}")

        # Get this company's SHAP values and sort by absolute value
        company_shap = shap_values[idx]                       # SHAP values for this company
        feature_contributions = pd.DataFrame({                # Build contribution table
            'feature': feature_names,                         # Feature names
            'shap_value': company_shap,                       # SHAP values
            'feature_value': X_test.iloc[idx].values,         # Actual feature values
        })
        feature_contributions['abs_shap'] = np.abs(           # Absolute SHAP for sorting
            feature_contributions['shap_value']
        )
        feature_contributions = feature_contributions.sort_values(  # Sort by importance
            'abs_shap', ascending=False
        )

        # Print top 10 contributing features for this company
        lines.append(f"\n  Top 10 Contributing Factors:")
        lines.append(f"  {'Feature':<45s} {'SHAP':>10s} {'Value':>10s} {'Direction':<15s}")
        lines.append(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*15}")

        for _, row in feature_contributions.head(10).iterrows():
            direction = "INCREASES risk" if row['shap_value'] > 0 else "DECREASES risk"
            arrow = ">>>" if abs(row['shap_value']) > 0.1 else ">>" if abs(row['shap_value']) > 0.05 else ">"
            lines.append(f"  {row['feature']:<45s} {row['shap_value']:>+10.4f} "
                         f"{row['feature_value']:>10.4f} {arrow} {direction}")

    explanation_text = '\n'.join(lines)                        # Join all lines

    if save_path is None:
        save_path = os.path.join(PROCESSED_DIR, "shap_explanations.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(explanation_text)

    print(f"    Saved: {save_path}")
    return explanation_text


# ============================================================================
# 5. SHAP DEPENDENCE PLOT (TOP FEATURE)
# ============================================================================

def plot_shap_dependence(shap_results, feature_name=None, save_path=None):
    """
    Create SHAP dependence plot for the most important feature.

    Dependence plot shows how a single feature's value affects the prediction:
        - X-axis: Feature value
        - Y-axis: SHAP value (impact on prediction)
        - Color: Interaction feature (auto-selected)

    Parameters:
        shap_results  : dict — SHAP values and test data
        feature_name  : str — feature to plot (None = auto-select most important)
        save_path     : str — path to save plot

    Returns:
        str — path to saved plot
    """

    print("  Plotting SHAP Dependence Plot...")

    if shap_results is None:
        print("    Skipping: No SHAP results available.")
        return None

    try:
        import shap
    except ImportError:
        return None

    shap_values = shap_results['shap_values']
    X_test = shap_results['X_test']
    feature_names = shap_results['feature_names']

    # Auto-select the most important feature if not specified
    if feature_name is None:
        mean_abs = np.abs(shap_values).mean(axis=0)           # Mean |SHAP| per feature
        feature_name = feature_names[np.argmax(mean_abs)]     # Feature with highest mean |SHAP|

    if feature_name not in X_test.columns:                    # Check feature exists
        print(f"    WARNING: Feature '{feature_name}' not found. Skipping.")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    shap.dependence_plot(                                     # SHAP's dependence plot
        feature_name,                                         # Feature to analyze
        shap_values,                                          # SHAP values
        X_test,                                               # Feature values
        show=False,                                           # Don't display (save instead)
        ax=ax,                                                # Plot on our axis
    )

    ax.set_title(f'SHAP Dependence: {feature_name}', fontsize=14, fontweight='bold')

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "shap_dependence.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved: {save_path}")
    return save_path


# ============================================================================
# MAIN: Run full SHAP explainability pipeline
# ============================================================================

def run_explainability(best_model, X_train, X_test, y_test, feature_names, df_with_labels):
    """
    Run the complete SHAP explainability pipeline.

    Parameters:
        best_model     : trained sklearn model (best performing)
        X_train        : pd.DataFrame — training features
        X_test         : pd.DataFrame — test features
        y_test         : pd.Series — test labels
        feature_names  : list — feature names
        df_with_labels : pd.DataFrame — full dataset with company names

    Returns:
        dict — all SHAP outputs (values, plots, explanations)
    """

    print("\n" + "=" * 70)
    print("  SHAP EXPLAINABILITY PIPELINE")
    print("=" * 70)

    model_name = type(best_model).__name__                    # Get model class name
    outputs = {}                                              # Track all outputs

    # Step 1: Compute SHAP values
    shap_results = compute_shap_values(                       # Compute SHAP
        best_model, X_train, X_test, feature_names, model_name
    )

    if shap_results is None:                                  # SHAP failed
        print("  SHAP analysis could not be completed.")
        return outputs

    # Step 2: Summary plot (beeswarm)
    outputs['summary_plot'] = plot_shap_summary(shap_results, top_n=20)

    # Step 3: Bar plot (mean absolute)
    outputs['bar_plot'] = plot_shap_bar(shap_results, top_n=20)

    # Step 4: Dependence plot (top feature)
    outputs['dependence_plot'] = plot_shap_dependence(shap_results)

    # Step 5: Company-level explanations
    outputs['explanations'] = explain_top_companies(          # Generate text explanations
        shap_results, df_with_labels, n_companies=10
    )

    outputs['shap_results'] = shap_results                    # Store raw SHAP results

    print(f"\n  SHAP analysis complete. Outputs saved to: {PLOTS_DIR}")
    return outputs


# Entry point
if __name__ == "__main__":
    print("Run this module via model_pipeline.py for full SHAP analysis.")
