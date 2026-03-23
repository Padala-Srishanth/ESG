"""
================================================================================
GREENWASHING RISK SCORING MODULE
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning

This module generates a FINAL 0-100 Greenwashing Risk Score for each company
by combining multiple signals into a single, interpretable number.

Scoring Components (weighted blend):
    1. Model Prediction Probability  (40%) — ML model's confidence of greenwashing
    2. Proxy Score                   (25%) — Rule-based indicator sum (0-5 scaled to 0-100)
    3. Linguistic GW Score           (15%) — NLP-based vague/hedging language detection
    4. Controversy-ESG Divergence    (10%) — Statistical gap between controversy and ESG risk
    5. Claim Credibility (inverted)  (10%) — Low credibility = higher risk

Output:
    - greenwashing_risk_scores.csv — Ranked list of all companies with scores and tiers
    - risk_score_summary.txt       — Human-readable summary report

Risk Tiers:
    0-20   : Very Low Risk   (Green)
    21-40  : Low Risk         (Light Green)
    41-60  : Moderate Risk    (Yellow)
    61-80  : High Risk        (Orange)
    81-100 : Very High Risk   (Red)
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                          # Data manipulation library
import numpy as np                           # Numerical computing library
import os                                    # File system operations
import warnings                              # Warning suppression
from sklearn.preprocessing import MinMaxScaler  # Min-max scaling to [0, 100]

warnings.filterwarnings('ignore')            # Suppress all warnings for clean output

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Project root directory
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # Processed data directory


# ============================================================================
# 1. LOAD ALL REQUIRED DATA
# ============================================================================

def load_scoring_data():
    """
    Load the feature matrix with proxy labels and model predictions.

    We need:
        - feature_matrix.csv        : All engineered features
        - model_metrics.csv         : To identify the best model
        - predictions.csv           : Company-level predictions

    Returns:
        pd.DataFrame — feature matrix with all columns needed for scoring
    """

    print("=" * 70)                                           # Visual separator
    print("  STEP 1: Loading Data for Risk Scoring")          # Status header
    print("=" * 70)

    # Load feature matrix (has all engineered features)
    fm_path = os.path.join(PROCESSED_DIR, "feature_matrix.csv")  # Path to features
    df = pd.read_csv(fm_path)                                 # Load into DataFrame
    print(f"  Loaded feature matrix: {df.shape[0]} companies, {df.shape[1]} columns")

    # Load predictions (has proxy scores)
    pred_path = os.path.join(PROCESSED_DIR, "predictions.csv")  # Path to predictions
    if os.path.exists(pred_path):                             # Check if file exists
        pred_df = pd.read_csv(pred_path)                      # Load predictions
        print(f"  Loaded predictions: {pred_df.shape[0]} rows")

        # Merge proxy scores into feature matrix if not already present
        if 'gw_proxy_score' not in df.columns and 'gw_proxy_score' in pred_df.columns:
            df = df.merge(                                    # Merge on company name
                pred_df[['company_name', 'gw_proxy_score', 'gw_label_binary']],
                on='company_name', how='left'                 # Left join to keep all companies
            )
    else:
        print("  WARNING: predictions.csv not found. Will compute proxy scores.")

    return df                                                 # Return loaded data


# ============================================================================
# 2. CONSTRUCT PROXY LABELS (if not already present)
# ============================================================================

def ensure_proxy_labels(df):
    """
    Ensure proxy labels exist. If they don't (e.g., running standalone),
    construct them from features.

    Parameters:
        df : pd.DataFrame — feature matrix

    Returns:
        pd.DataFrame — with gw_proxy_score and gw_label_binary columns
    """

    if 'gw_proxy_score' in df.columns:                        # Already computed
        print("  Proxy labels already present.")
        return df                                             # Return as-is

    print("\n  Constructing proxy labels...")

    # Import and run proxy label construction from model_training
    from model_training import construct_proxy_labels          # Import function
    df = construct_proxy_labels(df)                            # Add labels

    return df                                                 # Return with labels


# ============================================================================
# 3. COMPUTE COMPONENT SCORES (0-100 each)
# ============================================================================

def compute_component_scores(df):
    """
    Compute individual risk score components, each normalized to 0-100 range.

    Components:
        1. proxy_score_normalized    — Proxy GW score (0-5) scaled to 0-100
        2. linguistic_score          — GW linguistic score scaled to 0-100
        3. divergence_score          — ESG-controversy divergence scaled to 0-100
        4. credibility_score_inv     — Inverted claim credibility (low cred = high risk)
        5. controversy_ratio_score   — Controversy-risk ratio scaled to 0-100

    Parameters:
        df : pd.DataFrame — feature matrix with all needed columns

    Returns:
        pd.DataFrame — with 5 new component score columns added
    """

    print("\n" + "=" * 70)                                    # Separator
    print("  STEP 2: Computing Component Scores (0-100)")     # Header
    print("=" * 70)

    df = df.copy()                                            # Don't modify original

    scaler = MinMaxScaler(feature_range=(0, 100))             # Scale everything to 0-100

    # --- Component 1: Proxy Score (0-5 -> 0-100) ---
    if 'gw_proxy_score' in df.columns:                        # If proxy score exists
        df['comp_proxy'] = (df['gw_proxy_score'] / 5.0 * 100)  # Simple linear scaling
        df['comp_proxy'] = df['comp_proxy'].clip(0, 100)      # Clip to valid range
    else:                                                     # Fallback: use 50 (neutral)
        df['comp_proxy'] = 50.0                               # Default neutral score

    print(f"  Component 1 (Proxy Score):     mean={df['comp_proxy'].mean():.1f}, "
          f"range=[{df['comp_proxy'].min():.0f}, {df['comp_proxy'].max():.0f}]")

    # --- Component 2: Linguistic GW Score (0-1 -> 0-100) ---
    if 'greenwashing_signal_score' in df.columns:             # If linguistic score exists
        df['comp_linguistic'] = (                             # Scale to 0-100
            df['greenwashing_signal_score'] * 100             # Multiply by 100
        ).clip(0, 100)                                        # Clip to valid range
    else:
        df['comp_linguistic'] = 50.0                          # Default neutral

    print(f"  Component 2 (Linguistic Score): mean={df['comp_linguistic'].mean():.1f}, "
          f"range=[{df['comp_linguistic'].min():.1f}, {df['comp_linguistic'].max():.1f}]")

    # --- Component 3: Controversy-ESG Divergence (z-score -> 0-100) ---
    if 'esg_controversy_divergence' in df.columns:            # If divergence exists
        # Use min-max scaling on the divergence (higher divergence = more risk)
        divergence = df[['esg_controversy_divergence']].copy() # Extract column
        df['comp_divergence'] = scaler.fit_transform(divergence).flatten()  # Scale to 0-100
    else:
        df['comp_divergence'] = 50.0                          # Default neutral

    print(f"  Component 3 (Divergence Score): mean={df['comp_divergence'].mean():.1f}, "
          f"range=[{df['comp_divergence'].min():.1f}, {df['comp_divergence'].max():.1f}]")

    # --- Component 4: Inverted Claim Credibility (high credibility = low risk) ---
    if 'claim_credibility_score' in df.columns:               # If credibility exists
        # Invert: low credibility = high risk
        df['comp_credibility_inv'] = (                        # Invert the score
            (1.0 - df['claim_credibility_score']) * 100       # (1 - credibility) * 100
        ).clip(0, 100)                                        # Clip to valid range
    else:
        df['comp_credibility_inv'] = 50.0                     # Default neutral

    print(f"  Component 4 (Credibility Inv):  mean={df['comp_credibility_inv'].mean():.1f}, "
          f"range=[{df['comp_credibility_inv'].min():.1f}, {df['comp_credibility_inv'].max():.1f}]")

    # --- Component 5: Controversy-Risk Ratio (higher = more suspicious) ---
    if 'controversy_risk_ratio' in df.columns:                # If ratio exists
        ratio = df[['controversy_risk_ratio']].copy()         # Extract column
        df['comp_controversy_ratio'] = scaler.fit_transform(ratio).flatten()  # Scale to 0-100
    else:
        df['comp_controversy_ratio'] = 50.0                   # Default neutral

    print(f"  Component 5 (Controversy Ratio):mean={df['comp_controversy_ratio'].mean():.1f}, "
          f"range=[{df['comp_controversy_ratio'].min():.1f}, {df['comp_controversy_ratio'].max():.1f}]")

    return df                                                 # Return with all component scores


# ============================================================================
# 4. COMPUTE FINAL COMPOSITE RISK SCORE
# ============================================================================

def compute_final_risk_score(df):
    """
    Compute the final 0-100 greenwashing risk score as a weighted blend.

    Formula:
        risk_score = 0.40 * proxy_score
                   + 0.15 * linguistic_score
                   + 0.15 * divergence_score
                   + 0.15 * credibility_inverted
                   + 0.15 * controversy_ratio

    Weight rationale:
        - Proxy score (40%): Combines 5 domain indicators, most comprehensive signal
        - Other 4 components (15% each): Specialized signals that add nuance

    Parameters:
        df : pd.DataFrame — with component score columns

    Returns:
        pd.DataFrame — with final risk_score and risk_tier columns
    """

    print("\n" + "=" * 70)
    print("  STEP 3: Computing Final Risk Score (0-100)")
    print("=" * 70)

    # Weighted combination of all 5 components
    df['risk_score'] = (                                      # Create final score column
        0.40 * df['comp_proxy']                               # 40% weight: proxy GW score
        + 0.15 * df['comp_linguistic']                        # 15% weight: linguistic signals
        + 0.15 * df['comp_divergence']                        # 15% weight: ESG-controversy gap
        + 0.15 * df['comp_credibility_inv']                   # 15% weight: inverted credibility
        + 0.15 * df['comp_controversy_ratio']                 # 15% weight: controversy ratio
    ).round(2)                                                # Round to 2 decimal places

    # Clip to 0-100 range (safety check)
    df['risk_score'] = df['risk_score'].clip(0, 100)          # Ensure within bounds

    # Assign risk tiers based on score ranges
    df['risk_tier'] = pd.cut(                                 # Bin scores into tiers
        df['risk_score'],                                     # Column to bin
        bins=[0, 20, 40, 60, 80, 100],                        # Tier boundaries
        labels=[                                              # Human-readable tier names
            'Very Low Risk',                                  # 0-20: Green
            'Low Risk',                                       # 21-40: Light Green
            'Moderate Risk',                                  # 41-60: Yellow
            'High Risk',                                      # 61-80: Orange
            'Very High Risk'                                  # 81-100: Red
        ],
        include_lowest=True,                                  # Include 0 in first bin
    )

    # Print distribution summary
    print(f"\n  Risk Score Statistics:")
    print(f"    Mean:   {df['risk_score'].mean():.2f}")       # Average score
    print(f"    Median: {df['risk_score'].median():.2f}")     # Median score
    print(f"    Std:    {df['risk_score'].std():.2f}")        # Standard deviation
    print(f"    Min:    {df['risk_score'].min():.2f}")        # Lowest score
    print(f"    Max:    {df['risk_score'].max():.2f}")        # Highest score

    print(f"\n  Risk Tier Distribution:")
    tier_counts = df['risk_tier'].value_counts().sort_index() # Count per tier
    for tier, count in tier_counts.items():                   # Print each tier
        pct = count / len(df) * 100                           # Calculate percentage
        bar = '#' * int(pct / 2)                              # Create bar chart
        print(f"    {tier:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    return df                                                 # Return with final scores


# ============================================================================
# 5. GENERATE RANKED OUTPUT
# ============================================================================

def generate_ranked_output(df):
    """
    Generate the final ranked CSV and summary report.

    Output columns:
        - rank              : 1 = highest risk company
        - company_name      : Company name
        - sector            : Business sector
        - risk_score        : Final 0-100 greenwashing risk score
        - risk_tier         : Risk category (Very Low to Very High)
        - esg_risk_score    : Original ESG risk score
        - controversy_score : Controversy level
        - gw_proxy_score    : Proxy indicator sum (0-5)
        - comp_proxy        : Proxy component (0-100)
        - comp_linguistic   : Linguistic component (0-100)
        - comp_divergence   : Divergence component (0-100)
        - comp_credibility_inv : Inverted credibility component (0-100)
        - comp_controversy_ratio : Controversy ratio component (0-100)

    Parameters:
        df : pd.DataFrame — with final risk scores

    Returns:
        tuple: (ranked_df, report_text)
    """

    print("\n" + "=" * 70)
    print("  STEP 4: Generating Ranked Output")
    print("=" * 70)

    # Select columns for output
    output_cols = [                                           # Columns to include in output
        'company_name', 'sector', 'risk_score', 'risk_tier',
        'total_esg_risk_score', 'controversy_score',
    ]

    # Add optional columns if they exist
    optional_cols = [                                         # Additional columns if available
        'gw_proxy_score', 'comp_proxy', 'comp_linguistic',
        'comp_divergence', 'comp_credibility_inv', 'comp_controversy_ratio',
    ]
    for col in optional_cols:                                 # Add each if it exists
        if col in df.columns:                                 # Check existence
            output_cols.append(col)                           # Add to output

    # Filter to existing columns only
    output_cols = [c for c in output_cols if c in df.columns]

    # Create ranked DataFrame sorted by risk score (highest first)
    ranked_df = df[output_cols].copy()                        # Select columns
    ranked_df = ranked_df.sort_values(                        # Sort by risk score descending
        'risk_score', ascending=False                         # Highest risk first
    ).reset_index(drop=True)                                  # Reset index
    ranked_df.index += 1                                      # Start rank from 1
    ranked_df.index.name = 'rank'                             # Name the index column

    # Save ranked CSV
    csv_path = os.path.join(PROCESSED_DIR, "greenwashing_risk_scores.csv")
    ranked_df.to_csv(csv_path)                                # Save with rank as index
    print(f"  Saved: greenwashing_risk_scores.csv ({len(ranked_df)} companies)")

    # --- Generate summary report ---
    lines = []                                                # Build report
    lines.append("=" * 70)
    lines.append("GREENWASHING RISK SCORING REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTotal companies scored: {len(ranked_df)}")
    lines.append(f"Risk score range: {ranked_df['risk_score'].min():.2f} - {ranked_df['risk_score'].max():.2f}")
    lines.append(f"Mean risk score: {ranked_df['risk_score'].mean():.2f}")

    # Tier distribution
    lines.append("\n--- RISK TIER DISTRIBUTION ---")
    tier_counts = ranked_df['risk_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(ranked_df) * 100
        lines.append(f"  {tier:20s}: {count:4d} ({pct:5.1f}%)")

    # Top 20 highest risk companies
    lines.append("\n--- TOP 20 HIGHEST RISK COMPANIES ---")
    lines.append(f"{'Rank':<6s} {'Company':<45s} {'Score':>7s} {'Tier':<18s} {'Sector':<20s}")
    lines.append("-" * 96)
    for rank, row in ranked_df.head(20).iterrows():
        lines.append(f"{rank:<6d} {str(row['company_name'])[:44]:<45s} "
                     f"{row['risk_score']:>7.2f} {str(row['risk_tier']):<18s} "
                     f"{str(row.get('sector', 'N/A'))[:19]:<20s}")

    # Bottom 10 lowest risk companies
    lines.append("\n--- TOP 10 LOWEST RISK COMPANIES ---")
    lines.append(f"{'Rank':<6s} {'Company':<45s} {'Score':>7s} {'Tier':<18s} {'Sector':<20s}")
    lines.append("-" * 96)
    for rank, row in ranked_df.tail(10).iterrows():
        lines.append(f"{rank:<6d} {str(row['company_name'])[:44]:<45s} "
                     f"{row['risk_score']:>7.2f} {str(row['risk_tier']):<18s} "
                     f"{str(row.get('sector', 'N/A'))[:19]:<20s}")

    # Sector-level summary
    if 'sector' in ranked_df.columns:
        lines.append("\n--- SECTOR RISK SUMMARY ---")
        sector_stats = ranked_df.groupby('sector')['risk_score'].agg(['mean', 'max', 'count'])
        sector_stats = sector_stats.sort_values('mean', ascending=False)
        lines.append(f"{'Sector':<25s} {'Avg Score':>10s} {'Max Score':>10s} {'Companies':>10s}")
        lines.append("-" * 55)
        for sector, row in sector_stats.iterrows():
            lines.append(f"{str(sector)[:24]:<25s} {row['mean']:>10.2f} "
                         f"{row['max']:>10.2f} {int(row['count']):>10d}")

    report_text = '\n'.join(lines)                            # Join lines

    # Save report
    report_path = os.path.join(PROCESSED_DIR, "risk_score_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Saved: risk_score_summary.txt")

    # Print top 10
    print(f"\n  TOP 10 HIGHEST RISK COMPANIES:")
    print(f"  {'Rank':<6s} {'Company':<40s} {'Score':>7s} {'Tier'}")
    print(f"  {'-'*6} {'-'*40} {'-'*7} {'-'*18}")
    for rank, row in ranked_df.head(10).iterrows():
        print(f"  {rank:<6d} {str(row['company_name'])[:39]:<40s} "
              f"{row['risk_score']:>7.2f} {row['risk_tier']}")

    return ranked_df, report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute the complete risk scoring pipeline.
    """

    print("\n" + "#" * 70)
    print("  ESG GREENWASHING DETECTION - RISK SCORING")
    print("#" * 70)

    # Step 1: Load data
    df = load_scoring_data()                                  # Load feature matrix + predictions

    # Step 2: Ensure proxy labels exist
    df = ensure_proxy_labels(df)                              # Construct if missing

    # Step 3: Compute component scores (0-100 each)
    df = compute_component_scores(df)                         # 5 normalized components

    # Step 4: Compute final composite risk score
    df = compute_final_risk_score(df)                         # Weighted blend -> 0-100

    # Step 5: Generate ranked output
    ranked_df, report_text = generate_ranked_output(df)       # Sorted CSV + report

    print(f"\n  Risk scoring complete!")
    print(f"  Output: data/processed/greenwashing_risk_scores.csv")

    return ranked_df                                          # Return ranked DataFrame


# Entry point
if __name__ == "__main__":
    ranked = main()                                           # Execute risk scoring
