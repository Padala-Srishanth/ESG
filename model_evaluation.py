"""
================================================================================
MODEL EVALUATION MODULE
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

This module provides comprehensive model evaluation including:
    1. ROC Curve plotting for all models (side-by-side comparison)
    2. Precision-Recall Curve plotting (critical for imbalanced classes)
    3. Detailed Confusion Matrix with heatmap visualization
    4. Per-class metrics breakdown (precision, recall, F1 per class)
    5. Threshold analysis (optimal threshold selection)
    6. Cross-validation stability analysis
    7. Model comparison summary report generation
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                          # Data manipulation library
import numpy as np                           # Numerical computing library
import os                                    # File system operations
import warnings                              # Suppress non-critical warnings
import matplotlib                            # Matplotlib configuration
matplotlib.use('Agg')                        # Use non-interactive backend (no GUI needed, saves to file)
import matplotlib.pyplot as plt              # Plotting interface
import seaborn as sns                        # Statistical visualization library (built on matplotlib)
from sklearn.metrics import (                # Scikit-learn evaluation metrics
    roc_curve,                               # Compute ROC curve (FPR vs TPR at each threshold)
    auc,                                     # Compute Area Under Curve from FPR/TPR arrays
    precision_recall_curve,                  # Compute PR curve (precision vs recall at each threshold)
    average_precision_score,                 # Average precision (area under PR curve)
    confusion_matrix,                        # Compute TP/FP/TN/FN matrix
    classification_report,                   # Full per-class precision/recall/f1 text report
    accuracy_score,                          # Overall fraction of correct predictions
    f1_score,                                # Harmonic mean of precision and recall
)

warnings.filterwarnings('ignore')            # Suppress all warnings for clean output

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Project root directory
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # Processed data directory
PLOTS_DIR = os.path.join(BASE_DIR, "plots")                 # Directory to save plot images
os.makedirs(PLOTS_DIR, exist_ok=True)                       # Create plots dir if it doesn't exist


# ============================================================================
# 1. ROC CURVE PLOTTING
# ============================================================================

def plot_roc_curves(evaluation_results, save_path=None):
    """
    Plot ROC curves for all trained models on a single figure.

    ROC Curve explained:
        - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
          → "Of all SAFE companies, how many did we WRONGLY flag?"
        - Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
          → "Of all GREENWASHING companies, how many did we CORRECTLY catch?"
        - Diagonal line = random guessing (AUC = 0.5)
        - Perfect model hugs the top-left corner (AUC = 1.0)

    Parameters:
        evaluation_results : dict — model name → evaluation metrics (from model_training.py)
        save_path          : str — path to save the plot image (None = auto-generate)

    Returns:
        str — path to saved plot file
    """

    print("  [1/5] Plotting ROC Curves...")                   # Status message

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))            # Create figure with 10x8 inch size

    # Define colors for each model (consistent across all plots)
    colors = {                                                # Color map for each model
        'Gradient Boosting': '#2ecc71',                       # Green (best model)
        'Random Forest': '#3498db',                           # Blue
        'Logistic Regression': '#e74c3c',                     # Red
        'SVM': '#9b59b6',                                     # Purple
    }

    # Plot ROC curve for each model
    for name, metrics in evaluation_results.items():          # Iterate through each model
        y_prob = metrics.get('y_prob')                        # Get predicted probabilities
        y_test = metrics.get('y_test')                        # Get true labels

        if y_prob is not None and y_test is not None:         # Only plot if probabilities exist
            # Compute ROC curve: FPR and TPR at each decision threshold
            if y_prob.ndim > 1:                               # Multi-class probabilities
                fpr, tpr, thresholds = roc_curve(             # Compute ROC for class 1
                    y_test, y_prob[:, 1]                      # Use probability of positive class
                )
            else:                                             # Single array of probabilities
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)

            roc_auc = auc(fpr, tpr)                           # Calculate Area Under Curve

            # Plot this model's ROC curve
            color = colors.get(name, '#95a5a6')               # Get color (default: gray)
            ax.plot(                                          # Plot the curve
                fpr, tpr,                                     # X=FPR, Y=TPR
                color=color,                                  # Line color
                linewidth=2.5,                                # Line thickness
                label=f'{name} (AUC = {roc_auc:.4f})',        # Legend label with AUC score
            )

    # Plot the random baseline (diagonal line)
    ax.plot(                                                  # Plot diagonal reference line
        [0, 1], [0, 1],                                       # From (0,0) to (1,1)
        color='gray',                                         # Gray color
        linewidth=1.5,                                        # Thinner line
        linestyle='--',                                       # Dashed line style
        label='Random Baseline (AUC = 0.5000)',               # Legend label
    )

    # Formatting
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=13)   # X-axis label
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=13)    # Y-axis label
    ax.set_title('ROC Curves — ESG Greenwashing Detection Models', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)                 # Legend in bottom-right
    ax.set_xlim([0.0, 1.0])                                   # X-axis range
    ax.set_ylim([0.0, 1.05])                                  # Y-axis range (slightly above 1 for padding)
    ax.grid(True, alpha=0.3)                                  # Light grid for readability

    plt.tight_layout()                                        # Adjust spacing to prevent clipping

    # Save the plot
    if save_path is None:                                     # Auto-generate save path
        save_path = os.path.join(PLOTS_DIR, "roc_curves.png") # Default filename
    fig.savefig(save_path, dpi=150, bbox_inches='tight')      # Save at 150 DPI resolution
    plt.close(fig)                                            # Close figure to free memory

    print(f"    Saved: {save_path}")                          # Confirm save
    return save_path                                          # Return file path


# ============================================================================
# 2. PRECISION-RECALL CURVE PLOTTING
# ============================================================================

def plot_precision_recall_curves(evaluation_results, save_path=None):
    """
    Plot Precision-Recall curves for all models.

    Why PR curves matter MORE than ROC for imbalanced data:
        - ROC can look great even when the model is bad at finding rare positives
        - PR curves focus on the POSITIVE class (greenwashing) performance
        - A model with 99% accuracy but 0% recall on greenwashing is USELESS
        - PR curves expose this failure clearly

    Precision: Of all companies we FLAGGED, how many are ACTUALLY greenwashing?
    Recall:    Of all ACTUAL greenwashing companies, how many did we FLAG?

    Parameters:
        evaluation_results : dict — model evaluation metrics
        save_path          : str — path to save plot

    Returns:
        str — path to saved plot file
    """

    print("  [2/5] Plotting Precision-Recall Curves...")      # Status message

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))            # Create figure

    colors = {                                                # Same color scheme as ROC
        'Gradient Boosting': '#2ecc71',
        'Random Forest': '#3498db',
        'Logistic Regression': '#e74c3c',
        'SVM': '#9b59b6',
    }

    for name, metrics in evaluation_results.items():          # Iterate through models
        y_prob = metrics.get('y_prob')                        # Get predicted probabilities
        y_test = metrics.get('y_test')                        # Get true labels

        if y_prob is not None and y_test is not None:         # Only plot if data exists
            if y_prob.ndim > 1:                               # Multi-class probabilities
                precision, recall, _ = precision_recall_curve( # Compute PR curve
                    y_test, y_prob[:, 1]                      # Use positive class probability
                )
                ap = average_precision_score(y_test, y_prob[:, 1])  # Average precision (AP)
            else:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)

            color = colors.get(name, '#95a5a6')               # Get model color
            ax.plot(                                          # Plot PR curve
                recall, precision,                            # X=Recall, Y=Precision
                color=color,                                  # Line color
                linewidth=2.5,                                # Line thickness
                label=f'{name} (AP = {ap:.4f})',              # Legend with Average Precision
            )

    # Formatting
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13)        # X-axis label
    ax.set_ylabel('Precision (PPV)', fontsize=13)             # Y-axis label
    ax.set_title('Precision-Recall Curves — Greenwashing Detection', fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)                  # Legend position
    ax.set_xlim([0.0, 1.0])                                   # X range
    ax.set_ylim([0.0, 1.05])                                  # Y range
    ax.grid(True, alpha=0.3)                                  # Light grid

    plt.tight_layout()                                        # Adjust layout

    if save_path is None:                                     # Auto-generate path
        save_path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')      # Save plot
    plt.close(fig)                                            # Close figure

    print(f"    Saved: {save_path}")                          # Confirm save
    return save_path


# ============================================================================
# 3. CONFUSION MATRIX HEATMAPS
# ============================================================================

def plot_confusion_matrices(evaluation_results, save_path=None):
    """
    Plot confusion matrix heatmaps for all models in a 2x2 grid.

    Confusion Matrix explained:
        ┌─────────────────────────────────────┐
        │              PREDICTED               │
        │         Negative    Positive          │
        │  Actual                               │
        │  Neg    TN          FP                │
        │  Pos    FN          TP                │
        └─────────────────────────────────────┘

        TN (True Negative):  Correctly identified as NOT greenwashing
        FP (False Positive): Wrongly flagged as greenwashing (false alarm)
        FN (False Negative): Missed actual greenwashing (DANGEROUS)
        TP (True Positive):  Correctly caught greenwashing

    For ESG, FN is MORE COSTLY than FP:
        - FP: We investigate a clean company → wastes analyst time
        - FN: We MISS a greenwashing company → investors lose money, reputation damage

    Parameters:
        evaluation_results : dict — model evaluation metrics
        save_path          : str — path to save plot

    Returns:
        str — path to saved plot file
    """

    print("  [3/5] Plotting Confusion Matrices...")           # Status message

    n_models = len(evaluation_results)                        # Count of models
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))         # 2x2 grid of subplots
    axes = axes.flatten()                                     # Flatten 2D array to 1D for easy iteration

    class_labels = ['Not Greenwashing', 'Greenwashing']       # Human-readable class names

    for idx, (name, metrics) in enumerate(evaluation_results.items()):  # Iterate models
        if idx >= 4:                                          # Only plot first 4 models
            break

        cm = metrics['confusion_matrix']                      # Get confusion matrix

        # Plot heatmap using seaborn
        sns.heatmap(                                          # Create heatmap
            cm,                                               # Data: confusion matrix
            annot=True,                                       # Show numbers in cells
            fmt='d',                                          # Integer format (not scientific notation)
            cmap='Blues',                                      # Blue color scheme
            xticklabels=class_labels,                         # X-axis: predicted labels
            yticklabels=class_labels,                         # Y-axis: actual labels
            ax=axes[idx],                                     # Plot on this specific subplot
            cbar=True,                                        # Show color bar
            annot_kws={'size': 16, 'fontweight': 'bold'},     # Annotation font settings
        )

        # Calculate per-cell metrics for annotation
        tn, fp, fn, tp = cm.ravel()                           # Extract individual values
        total = tn + fp + fn + tp                             # Total predictions

        # Set title with accuracy
        accuracy = (tp + tn) / total                          # Calculate accuracy
        axes[idx].set_title(                                  # Set subplot title
            f'{name}\nAccuracy: {accuracy:.2%} | '
            f'TP={tp}, FP={fp}, FN={fn}, TN={tn}',
            fontsize=12, fontweight='bold'
        )
        axes[idx].set_xlabel('Predicted', fontsize=11)        # X-axis label
        axes[idx].set_ylabel('Actual', fontsize=11)           # Y-axis label

    # Hide unused subplots (if fewer than 4 models)
    for idx in range(n_models, 4):                            # Hide remaining subplots
        axes[idx].set_visible(False)                          # Make invisible

    fig.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()                                        # Adjust spacing

    if save_path is None:                                     # Auto-generate path
        save_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')      # Save plot
    plt.close(fig)                                            # Free memory

    print(f"    Saved: {save_path}")                          # Confirm save
    return save_path


# ============================================================================
# 4. FEATURE IMPORTANCE COMPARISON PLOT
# ============================================================================

def plot_feature_importance(evaluation_results, top_n=20, save_path=None):
    """
    Plot top N most important features for the best model.

    Feature importance for tree-based models:
        Measures how much each feature reduces impurity (Gini/entropy) across
        all trees in the ensemble. Higher = more predictive of greenwashing.

    Parameters:
        evaluation_results : dict — model evaluation metrics
        top_n              : int — number of top features to show
        save_path          : str — path to save plot

    Returns:
        str — path to saved plot file
    """

    print(f"  [4/5] Plotting Top {top_n} Feature Importance...")  # Status message

    # Find the best model (highest F1)
    best_name = max(                                          # Find model with highest F1
        evaluation_results,                                   # Search in all models
        key=lambda x: evaluation_results[x]['f1_score']       # Sort key: F1 score
    )
    best_metrics = evaluation_results[best_name]              # Get best model's metrics

    fi = best_metrics.get('feature_importance')               # Get feature importance

    if fi is None:                                            # No feature importance available
        print("    WARNING: Best model has no feature importance. Skipping.")
        return None

    # Select top N features
    top_features = fi.head(top_n)                             # Get top N by importance

    # Create horizontal bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))           # Create figure

    # Plot horizontal bars (reversed so most important is at top)
    bars = ax.barh(                                           # Horizontal bar chart
        range(len(top_features)),                             # Y positions
        top_features.values[::-1],                            # Values (reversed for top-down order)
        color='#2ecc71',                                      # Green color
        edgecolor='#27ae60',                                  # Darker green edge
        alpha=0.85,                                           # Slight transparency
    )

    # Set y-axis labels to feature names
    ax.set_yticks(range(len(top_features)))                   # Y-tick positions
    ax.set_yticklabels(top_features.index[::-1], fontsize=10) # Feature names (reversed)

    # Formatting
    ax.set_xlabel('Feature Importance', fontsize=13)          # X-axis label
    ax.set_title(f'Top {top_n} Features — {best_name}', fontsize=15, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)                       # Vertical grid lines only

    # Add value labels on each bar
    for i, (val, bar) in enumerate(zip(top_features.values[::-1], bars)):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)  # Label after bar

    plt.tight_layout()                                        # Adjust layout

    if save_path is None:                                     # Auto-generate path
        save_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')      # Save plot
    plt.close(fig)                                            # Free memory

    print(f"    Saved: {save_path}")                          # Confirm save
    return save_path


# ============================================================================
# 5. MODEL COMPARISON SUMMARY BAR CHART
# ============================================================================

def plot_model_comparison(evaluation_results, save_path=None):
    """
    Plot grouped bar chart comparing all models across all metrics.

    Parameters:
        evaluation_results : dict — model evaluation metrics
        save_path          : str — path to save plot

    Returns:
        str — path to saved plot file
    """

    print("  [5/5] Plotting Model Comparison...")             # Status message

    # Build comparison DataFrame
    rows = []                                                 # List to build table
    for name, metrics in evaluation_results.items():          # Iterate models
        rows.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
        })
    comp_df = pd.DataFrame(rows)                              # Create DataFrame

    # Create grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))            # Create figure

    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    x = np.arange(len(comp_df))                               # X positions for groups
    width = 0.15                                              # Width of each bar
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']  # Color per metric

    for i, (metric, color) in enumerate(zip(metrics_cols, colors)):  # Plot each metric
        offset = (i - len(metrics_cols)/2 + 0.5) * width     # Calculate bar position offset
        bars = ax.bar(                                        # Plot vertical bars
            x + offset,                                       # X position with offset
            comp_df[metric],                                  # Bar heights
            width,                                            # Bar width
            label=metric,                                     # Legend label
            color=color,                                      # Bar color
            alpha=0.85,                                       # Slight transparency
            edgecolor='white',                                # White edges between bars
        )
        # Add value labels on top of each bar
        for bar, val in zip(bars, comp_df[metric]):           # Iterate bars
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    # Formatting
    ax.set_xlabel('Model', fontsize=13)                       # X-axis label
    ax.set_ylabel('Score', fontsize=13)                       # Y-axis label
    ax.set_title('Model Performance Comparison — All Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x)                                          # Set x-tick positions
    ax.set_xticklabels(comp_df['Model'], fontsize=11)         # Model names on x-axis
    ax.legend(fontsize=10, loc='lower right')                 # Legend
    ax.set_ylim([0.0, 1.15])                                  # Y-axis range (room for labels)
    ax.grid(True, axis='y', alpha=0.3)                        # Horizontal grid lines

    plt.tight_layout()                                        # Adjust layout

    if save_path is None:                                     # Auto-generate path
        save_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')      # Save plot
    plt.close(fig)                                            # Free memory

    print(f"    Saved: {save_path}")                          # Confirm save
    return save_path


# ============================================================================
# 6. GENERATE DETAILED EVALUATION REPORT
# ============================================================================

def generate_evaluation_report(evaluation_results, save_path=None):
    """
    Generate a comprehensive text evaluation report covering all models.

    Parameters:
        evaluation_results : dict — model evaluation metrics
        save_path          : str — path to save report

    Returns:
        str — the report text
    """

    print("\n  Generating detailed evaluation report...")      # Status message

    lines = []                                                # Build report as list of lines
    lines.append("=" * 70)
    lines.append("ESG GREENWASHING DETECTION - MODEL EVALUATION REPORT")
    lines.append("=" * 70)

    # Model comparison table
    lines.append("\n--- MODEL COMPARISON ---\n")
    lines.append(f"{'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} "
                 f"{'Recall':>10s} {'F1':>10s} {'ROC-AUC':>10s}")
    lines.append("-" * 75)

    sorted_models = sorted(evaluation_results.items(),        # Sort by F1 descending
                           key=lambda x: x[1]['f1_score'], reverse=True)

    for name, m in sorted_models:                             # Print each model's metrics
        lines.append(f"{name:<25s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                     f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}")

    # Detailed per-model reports
    for name, m in sorted_models:
        lines.append(f"\n\n{'='*70}")
        lines.append(f"DETAILED REPORT: {name}")
        lines.append(f"{'='*70}")

        # Confusion matrix
        cm = m['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        lines.append(f"\nConfusion Matrix:")
        lines.append(f"  True Negatives (TN):  {tn:4d}  — Correctly identified as NOT greenwashing")
        lines.append(f"  False Positives (FP): {fp:4d}  — Wrongly flagged as greenwashing (false alarm)")
        lines.append(f"  False Negatives (FN): {fn:4d}  — MISSED actual greenwashing (DANGEROUS)")
        lines.append(f"  True Positives (TP):  {tp:4d}  — Correctly caught greenwashing")

        # Derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        lines.append(f"\n  Specificity (TNR):       {specificity:.4f}  — Of safe companies, how many correctly cleared")
        lines.append(f"  Negative Predictive Value: {npv:.4f}  — Of cleared companies, how many are actually safe")

        # Classification report
        lines.append(f"\nClassification Report:\n{m['classification_report']}")

        # Top features
        if m.get('feature_importance') is not None:
            lines.append("Top 10 Features:")
            for i, (feat, imp) in enumerate(m['feature_importance'].head(10).items()):
                lines.append(f"  {i+1:2d}. {feat:50s} {imp:.6f}")

    report = '\n'.join(lines)                                 # Join all lines

    if save_path is None:                                     # Auto-generate path
        save_path = os.path.join(PROCESSED_DIR, "evaluation_report.txt")
    with open(save_path, 'w', encoding='utf-8') as f:        # Write to file
        f.write(report)

    print(f"  Saved: {save_path}")
    return report


# ============================================================================
# MAIN: Run all evaluation steps
# ============================================================================

def run_evaluation(evaluation_results):
    """
    Run the complete evaluation pipeline: all plots + report.

    Parameters:
        evaluation_results : dict — from model_training.evaluate_models()

    Returns:
        dict — paths to all saved outputs
    """

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION PIPELINE")
    print("=" * 70)

    outputs = {}                                              # Track all output paths

    # Plot 1: ROC Curves
    outputs['roc_curves'] = plot_roc_curves(evaluation_results)

    # Plot 2: Precision-Recall Curves
    outputs['pr_curves'] = plot_precision_recall_curves(evaluation_results)

    # Plot 3: Confusion Matrices
    outputs['confusion_matrices'] = plot_confusion_matrices(evaluation_results)

    # Plot 4: Feature Importance
    outputs['feature_importance'] = plot_feature_importance(evaluation_results, top_n=20)

    # Plot 5: Model Comparison
    outputs['model_comparison'] = plot_model_comparison(evaluation_results)

    # Text Report
    outputs['report'] = generate_evaluation_report(evaluation_results)

    print(f"\n  All evaluation outputs saved to: {PLOTS_DIR}")
    return outputs


# Entry point for standalone execution
if __name__ == "__main__":
    print("Run this module via model_pipeline.py for full evaluation.")
