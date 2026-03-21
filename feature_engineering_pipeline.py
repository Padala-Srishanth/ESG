"""
================================================================================
FEATURE ENGINEERING - MASTER ORCHESTRATION PIPELINE
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning
Purpose : Orchestrate all 3 feature engineering modules (Numerical, NLP,
          Categorical) into a unified pipeline that produces the final
          ML-ready feature matrix.

Pipeline Architecture:
    ┌────────────────────────┐
    │   Raw Processed Data   │  ← company_profiles.csv (480 companies)
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Numerical Features    │  -> 36 features (ratios, stats, anomalies)
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  NLP Text Features     │  -> 47 features (sentiment, readability, GW signals)
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Categorical Features  │  -> 31 features (encodings, bins, profiles)
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Feature Selection     │  -> Remove constant/duplicate/correlated features
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Final Feature Matrix  │  -> Saved to data/processed/feature_matrix.csv
    └────────────────────────┘

Output:
    - feature_matrix.csv        : Complete feature matrix for ML training
    - feature_registry.csv      : Metadata about every feature (name, category, description)
    - feature_summary_report.txt: Human-readable summary of feature engineering results
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                  # Core data manipulation library
import numpy as np                   # Numerical computing for arrays
import os                            # Operating system interface (file paths)
import json                          # JSON serialization for feature metadata
from datetime import datetime        # Date/time utilities for timestamps
import warnings                      # Python warnings control
warnings.filterwarnings('ignore')    # Suppress non-critical warnings

# Import our custom feature engineering modules
from feature_engineering_numerical import NumericalFeatureEngineer      # Numerical features
from feature_engineering_nlp import NLPFeatureEngineer                  # NLP text features
from feature_engineering_categorical import CategoricalFeatureEngineer  # Categorical features


# ============================================================================
# CLASS: FeatureEngineeringPipeline
# ============================================================================

class FeatureEngineeringPipeline:
    """
    Master orchestration pipeline for all feature engineering modules.

    This class:
        1. Loads the preprocessed data
        2. Runs Numerical -> NLP -> Categorical feature engineering in sequence
        3. Performs feature quality checks (constants, duplicates, correlations)
        4. Produces the final ML-ready feature matrix
        5. Generates comprehensive reporting on all features created

    Usage:
        pipeline = FeatureEngineeringPipeline()
        feature_matrix = pipeline.run_full_pipeline()
    """

    def __init__(self, data_dir='data/processed'):
        """
        Initialize the pipeline with data directory and module instances.

        Parameters:
            data_dir : str — path to the directory containing processed data files

        Attributes:
            data_dir         : str — directory with input/output data
            numerical_eng    : NumericalFeatureEngineer — numerical feature module
            nlp_eng          : NLPFeatureEngineer — NLP text feature module
            categorical_eng  : CategoricalFeatureEngineer — categorical feature module
            pipeline_stats   : dict — tracks pipeline execution statistics
        """

        self.data_dir = data_dir                                  # Store data directory path
        self.numerical_eng = NumericalFeatureEngineer()            # Initialize numerical module
        self.nlp_eng = NLPFeatureEngineer()                       # Initialize NLP module
        self.categorical_eng = CategoricalFeatureEngineer()        # Initialize categorical module
        self.pipeline_stats = {}                                  # Empty stats dictionary

    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================

    def load_data(self):
        """
        Load the preprocessed company profiles dataset.

        This is the primary dataset containing 480 companies with:
        - Company identifiers (symbol, name, sector, industry)
        - Text descriptions (for NLP features)
        - ESG risk scores (env, social, gov, total)
        - Controversy scores
        - Pre-encoded risk levels

        Returns:
            pd.DataFrame — loaded company profiles data
        """

        # Print loading header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 1: LOADING DATA")                           # Step title
        print("=" * 70)                                           # Visual separator

        # Construct full file path to company profiles
        file_path = os.path.join(self.data_dir, 'company_profiles.csv')  # Build path

        # Load the CSV file into a pandas DataFrame
        print(f"  Loading: {file_path}")                          # Log file being loaded
        df = pd.read_csv(file_path)                               # Read CSV into DataFrame

        # Record initial data statistics
        self.pipeline_stats['initial_rows'] = len(df)             # Store row count
        self.pipeline_stats['initial_cols'] = df.shape[1]         # Store column count
        self.pipeline_stats['companies'] = df['company_name'].nunique()  # Unique companies

        # Print data summary
        print(f"  Shape: {df.shape}")                             # Print dimensions
        print(f"  Companies: {self.pipeline_stats['companies']}") # Print company count
        print(f"  Columns: {list(df.columns)}")                   # Print column names

        # Validate required columns exist before proceeding
        required_columns = [                                      # Columns we absolutely need
            'company_name',                                       # Company identifier
            'total_esg_risk_score',                               # Total ESG risk
            'env_risk_score',                                     # Environmental pillar
            'social_risk_score',                                  # Social pillar
            'gov_risk_score',                                     # Governance pillar
            'controversy_score',                                  # Controversy level
            'description'                                         # Text for NLP
        ]

        # Check each required column exists
        missing = [                                               # Find missing columns
            col for col in required_columns                       # For each required column
            if col not in df.columns                              # If not in dataframe
        ]

        if missing:                                               # If any are missing
            raise ValueError(                                     # Raise error with details
                f"Missing required columns: {missing}"            # List missing columns
            )

        print("  All required columns validated successfully")   # Confirmation message
        return df                                                 # Return loaded dataframe

    # ========================================================================
    # STEP 2: NUMERICAL FEATURE ENGINEERING
    # ========================================================================

    def run_numerical_features(self, df):
        """
        Execute the numerical feature engineering module.

        Creates 36+ features from ESG scores including:
        - Pillar ratios and imbalance metrics
        - Risk decomposition and residuals
        - Statistical moments (variance, skewness, CV)
        - Interaction terms (pillar x controversy)
        - Anomaly detection features (z-scores, MAD, IQR outliers)
        - Sector-relative benchmarks

        Parameters:
            df : pd.DataFrame — company data with ESG scores

        Returns:
            pd.DataFrame — with numerical features added
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 2: NUMERICAL FEATURE ENGINEERING")          # Step title
        print("=" * 70)                                           # Visual separator

        # Record column count before engineering (to measure additions)
        cols_before = df.shape[1]                                 # Count columns before

        # Run the numerical feature engineering pipeline
        df = self.numerical_eng.engineer_all_numerical_features(df)  # Execute pipeline

        # Record statistics
        cols_after = df.shape[1]                                  # Count columns after
        self.pipeline_stats['numerical_features'] = cols_after - cols_before  # Features added

        # Print summary
        print(f"\n  Numerical features added: {cols_after - cols_before}")  # Summary

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # STEP 3: NLP FEATURE ENGINEERING
    # ========================================================================

    def run_nlp_features(self, df):
        """
        Execute the NLP text feature engineering module.

        Extracts 47 features from company description text including:
        - Sentiment polarity and strength
        - Readability (Flesch, Gunning Fog)
        - Vocabulary richness (lexical diversity, hapax)
        - ESG keyword density per pillar
        - Greenwashing linguistic signals (vague, hedge, superlative)
        - Document structure metrics

        Parameters:
            df : pd.DataFrame — company data with 'description' column

        Returns:
            pd.DataFrame — with NLP text features added
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 3: NLP FEATURE ENGINEERING")                # Step title
        print("=" * 70)                                           # Visual separator

        # Record column count before
        cols_before = df.shape[1]                                 # Count before

        # Run the NLP feature engineering pipeline
        df = self.nlp_eng.engineer_all_nlp_features(df)           # Execute pipeline

        # Record statistics
        cols_after = df.shape[1]                                  # Count after
        self.pipeline_stats['nlp_features'] = cols_after - cols_before  # Features added

        # Print summary
        print(f"\n  NLP features added: {cols_after - cols_before}")  # Summary

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # STEP 4: CATEGORICAL FEATURE ENGINEERING
    # ========================================================================

    def run_categorical_features(self, df):
        """
        Execute the categorical feature engineering module.

        Creates 31 features from categorical variables including:
        - Frequency encodings (sector, industry)
        - Risk-based binning (domain-aligned tiers)
        - Cross-feature derived categoricals
        - Material ESG issue encoding
        - Sector risk profiles

        Parameters:
            df : pd.DataFrame — company data with categorical columns

        Returns:
            pd.DataFrame — with categorical features added
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 4: CATEGORICAL FEATURE ENGINEERING")        # Step title
        print("=" * 70)                                           # Visual separator

        # Record column count before
        cols_before = df.shape[1]                                 # Count before

        # Run the categorical feature engineering pipeline
        df = self.categorical_eng.engineer_all_categorical_features(df)  # Execute

        # Record statistics
        cols_after = df.shape[1]                                  # Count after
        self.pipeline_stats['categorical_features'] = cols_after - cols_before

        # Print summary
        print(f"\n  Categorical features added: {cols_after - cols_before}")

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # STEP 5: FEATURE QUALITY CHECKS
    # ========================================================================

    def run_feature_quality_checks(self, df):
        """
        Perform quality checks on the engineered feature matrix.

        Checks and remediations:
            1. Constant features     -> Features with zero variance (useless)
            2. Duplicate features    -> Identical columns (redundant)
            3. Highly correlated     -> Features with |correlation| > 0.98
            4. Missing values        -> Fill remaining NaN with 0
            5. Infinite values       -> Replace inf/-inf with NaN, then fill

        These checks ensure the feature matrix is clean, non-redundant,
        and ready for ML model training without numerical issues.

        Parameters:
            df : pd.DataFrame — feature-engineered dataframe

        Returns:
            pd.DataFrame — cleaned feature matrix
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 5: FEATURE QUALITY CHECKS")                # Step title
        print("=" * 70)                                           # Visual separator

        # Record column count before quality checks
        cols_before = df.shape[1]                                 # Count before

        # ------------------------------------------------------------------
        # Check 1: Replace infinite values with NaN
        # ------------------------------------------------------------------
        # Some calculations (e.g., division by near-zero) can produce infinity
        # inf values crash most ML algorithms, so replace with NaN first
        inf_count = np.isinf(                                     # Check for infinity
            df.select_dtypes(include=[np.number])                 # Only numeric columns
        ).sum().sum()                                             # Total count of inf values

        print(f"  Infinite values found: {inf_count}")            # Report inf count

        # Replace inf and -inf with NaN across all numeric columns
        df = df.replace([np.inf, -np.inf], np.nan)                # Replace infinities

        # ------------------------------------------------------------------
        # Check 2: Fill remaining missing values with 0
        # ------------------------------------------------------------------
        # Count NaN values before filling
        nan_before = df.select_dtypes(include=[np.number]).isnull().sum().sum()
        print(f"  NaN values found: {nan_before}")                # Report NaN count

        # Fill NaN with 0 for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns  # Get numeric columns
        df[numeric_cols] = df[numeric_cols].fillna(0)             # Fill NaN with 0

        # ------------------------------------------------------------------
        # Check 3: Identify and remove constant features
        # ------------------------------------------------------------------
        # A constant feature has the same value for every row — zero information
        constant_features = []                                    # List of constant features
        for col in numeric_cols:                                  # Check each numeric column
            if df[col].nunique() <= 1:                            # If only 0 or 1 unique values
                constant_features.append(col)                     # Mark as constant

        if constant_features:                                     # If any constants found
            print(f"  Constant features removed: {len(constant_features)}")
            print(f"    -> {constant_features}")                   # List them
            df = df.drop(columns=constant_features)               # Remove constant columns
        else:                                                     # No constants found
            print("  Constant features: None found")              # Report clean

        # ------------------------------------------------------------------
        # Check 4: Identify duplicate columns
        # ------------------------------------------------------------------
        # Two columns with identical values are redundant — keep only one
        duplicate_cols = []                                        # List of duplicate columns
        numeric_df = df.select_dtypes(include=[np.number])        # Only numeric columns
        cols_list = list(numeric_df.columns)                      # Column name list

        for i in range(len(cols_list)):                            # Outer loop
            for j in range(i + 1, len(cols_list)):                # Inner loop (pairs)
                if cols_list[j] not in duplicate_cols:             # Skip already flagged
                    # Check if two columns are identical
                    if numeric_df[cols_list[i]].equals(           # Compare column i
                        numeric_df[cols_list[j]]):                # With column j
                        duplicate_cols.append(cols_list[j])       # Mark j as duplicate

        if duplicate_cols:                                        # If duplicates found
            print(f"  Duplicate features removed: {len(duplicate_cols)}")
            df = df.drop(columns=duplicate_cols)                  # Remove duplicates
        else:                                                     # No duplicates found
            print("  Duplicate features: None found")             # Report clean

        # ------------------------------------------------------------------
        # Check 5: Report highly correlated features (but keep them)
        # ------------------------------------------------------------------
        # High correlation (|r| > 0.98) means features are nearly redundant
        # We report but don't remove — let the model/feature selection decide
        numeric_df = df.select_dtypes(include=[np.number])        # Refresh numeric columns
        if len(numeric_df.columns) > 1:                           # Need at least 2 columns
            corr_matrix = numeric_df.corr().abs()                 # Absolute correlation matrix
            # Get upper triangle (avoid counting pairs twice)
            upper_tri = corr_matrix.where(                        # Upper triangle only
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            # Find pairs with correlation > 0.98
            high_corr_pairs = []                                  # List of high-corr pairs
            for col in upper_tri.columns:                         # For each column
                correlated = upper_tri.index[                     # Find rows where
                    upper_tri[col] > 0.98                         # Correlation > 0.98
                ].tolist()                                        # Convert to list
                for corr_col in correlated:                       # For each correlated column
                    high_corr_pairs.append(                       # Store the pair
                        (col, corr_col,                           # Column names
                         corr_matrix.loc[corr_col, col])          # Correlation value
                    )

            if high_corr_pairs:                                   # If high correlations exist
                print(f"  Highly correlated pairs (|r| > 0.98): {len(high_corr_pairs)}")
                for c1, c2, r in high_corr_pairs[:5]:             # Show top 5 pairs
                    print(f"    -> {c1} <-> {c2}: r={r:.4f}")       # Print pair and correlation
            else:                                                 # No high correlations
                print("  Highly correlated pairs: None found")    # Report clean

        # ------------------------------------------------------------------
        # Record quality check statistics
        # ------------------------------------------------------------------
        cols_after = df.shape[1]                                  # Count after checks
        self.pipeline_stats['features_removed'] = cols_before - cols_after  # Removed count
        self.pipeline_stats['inf_values_fixed'] = inf_count       # Inf fixes
        self.pipeline_stats['nan_values_filled'] = nan_before     # NaN fills

        print(f"\n  Features after quality checks: {cols_after}")  # Final count
        return df                                                 # Return cleaned dataframe

    # ========================================================================
    # STEP 6: GENERATE FEATURE REGISTRY
    # ========================================================================

    def generate_feature_registry(self, df):
        """
        Create a comprehensive registry of all features with metadata.

        The registry serves as documentation for:
        - What each feature measures
        - Which module created it
        - What data type it is
        - Basic statistics (mean, std, min, max)

        This is critical for model interpretability and feature debugging.

        Parameters:
            df : pd.DataFrame — final feature matrix

        Returns:
            pd.DataFrame — feature registry with metadata
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 6: GENERATING FEATURE REGISTRY")            # Step title
        print("=" * 70)                                           # Visual separator

        # Initialize registry entries list
        registry_entries = []                                     # List to hold feature info

        # Collect features from each module's registry
        module_registries = {                                     # Map module -> registry
            'numerical': self.numerical_eng.feature_registry,     # Numerical features
            'nlp': self.nlp_eng.feature_registry,                 # NLP features
            'categorical': self.categorical_eng.feature_registry  # Categorical features
        }

        # Iterate through each module and its feature categories
        for module_name, categories in module_registries.items():  # Each module
            for category_name, info in categories.items():        # Each category
                for feature_name in info['features']:             # Each feature
                    # Build entry for this feature
                    entry = {                                     # Feature metadata dict
                        'feature_name': feature_name,             # Name of the feature
                        'module': module_name,                    # Which module created it
                        'category': category_name,                # Sub-category within module
                    }

                    # Add statistics if the feature exists in the dataframe
                    if feature_name in df.columns:                # Check exists
                        col = df[feature_name]                    # Get column data
                        if pd.api.types.is_numeric_dtype(col):    # If numeric
                            entry['dtype'] = 'numeric'            # Mark as numeric
                            entry['mean'] = round(col.mean(), 4)  # Mean value
                            entry['std'] = round(col.std(), 4)    # Standard deviation
                            entry['min'] = round(col.min(), 4)    # Minimum value
                            entry['max'] = round(col.max(), 4)    # Maximum value
                            entry['null_count'] = int(col.isnull().sum())  # Missing count
                        else:                                     # Non-numeric column
                            entry['dtype'] = str(col.dtype)       # Actual dtype
                            entry['mean'] = None                  # N/A for non-numeric
                            entry['std'] = None                   # N/A
                            entry['min'] = None                   # N/A
                            entry['max'] = None                   # N/A
                            entry['null_count'] = int(col.isnull().sum())
                    else:                                         # Feature not in dataframe
                        entry['dtype'] = 'removed'                # Marked as removed
                        entry['mean'] = None                      # N/A
                        entry['std'] = None                       # N/A
                        entry['min'] = None                       # N/A
                        entry['max'] = None                       # N/A
                        entry['null_count'] = None                # N/A

                    registry_entries.append(entry)                # Add to registry list

        # Convert to DataFrame for easy viewing and saving
        registry_df = pd.DataFrame(registry_entries)              # Create registry DataFrame

        # Print registry summary
        print(f"  Total features registered: {len(registry_entries)}")
        print(f"  By module:")                                    # Per-module breakdown
        for module in ['numerical', 'nlp', 'categorical']:        # Each module
            count = len([e for e in registry_entries if e['module'] == module])
            print(f"    - {module}: {count} features")            # Module feature count

        return registry_df                                        # Return registry DataFrame

    # ========================================================================
    # STEP 7: SAVE OUTPUTS
    # ========================================================================

    def save_outputs(self, df, registry_df):
        """
        Save the final feature matrix, registry, and summary report to disk.

        Output Files:
            - feature_matrix.csv      : The ML-ready feature matrix (main output)
            - feature_registry.csv    : Metadata for every engineered feature
            - pipeline_summary.txt    : Human-readable pipeline execution report

        Parameters:
            df          : pd.DataFrame — final feature matrix
            registry_df : pd.DataFrame — feature registry with metadata
        """

        # Print step header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  STEP 7: SAVING OUTPUTS")                        # Step title
        print("=" * 70)                                           # Visual separator

        # ------------------------------------------------------------------
        # Save 1: Feature Matrix (main ML training data)
        # ------------------------------------------------------------------
        matrix_path = os.path.join(self.data_dir, 'feature_matrix.csv')  # Build path
        df.to_csv(matrix_path, index=False)                       # Save to CSV without index
        print(f"  Saved feature matrix: {matrix_path}")           # Confirm save
        print(f"    Shape: {df.shape}")                           # Print dimensions

        # ------------------------------------------------------------------
        # Save 2: Feature Registry (feature documentation)
        # ------------------------------------------------------------------
        registry_path = os.path.join(self.data_dir, 'feature_registry.csv')  # Build path
        registry_df.to_csv(registry_path, index=False)            # Save registry
        print(f"  Saved feature registry: {registry_path}")       # Confirm

        # ------------------------------------------------------------------
        # Save 3: Pipeline Summary Report (human-readable text)
        # ------------------------------------------------------------------
        report_path = os.path.join(self.data_dir, 'pipeline_summary.txt')  # Build path

        # Build the summary report text
        report_lines = [                                          # List of report lines
            "=" * 70,                                             # Separator
            "FEATURE ENGINEERING PIPELINE - EXECUTION SUMMARY",   # Title
            "=" * 70,                                             # Separator
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",  # Execution time
            f"",                                                  # Blank line
            f"INPUT DATA:",                                       # Section header
            f"  Initial rows: {self.pipeline_stats.get('initial_rows', 'N/A')}",
            f"  Initial columns: {self.pipeline_stats.get('initial_cols', 'N/A')}",
            f"  Companies: {self.pipeline_stats.get('companies', 'N/A')}",
            f"",                                                  # Blank line
            f"FEATURES ENGINEERED:",                              # Section header
            f"  Numerical features: {self.pipeline_stats.get('numerical_features', 'N/A')}",
            f"  NLP features: {self.pipeline_stats.get('nlp_features', 'N/A')}",
            f"  Categorical features: {self.pipeline_stats.get('categorical_features', 'N/A')}",
            f"",                                                  # Blank line
            f"QUALITY CHECKS:",                                   # Section header
            f"  Features removed: {self.pipeline_stats.get('features_removed', 'N/A')}",
            f"  Infinite values fixed: {self.pipeline_stats.get('inf_values_fixed', 'N/A')}",
            f"  NaN values filled: {self.pipeline_stats.get('nan_values_filled', 'N/A')}",
            f"",                                                  # Blank line
            f"FINAL OUTPUT:",                                     # Section header
            f"  Feature matrix shape: {df.shape}",                # Final dimensions
            f"  Total features: {df.select_dtypes(include=[np.number]).shape[1]}",
            "=" * 70                                              # Separator
        ]

        # Write report to text file
        with open(report_path, 'w') as f:                         # Open file for writing
            f.write('\n'.join(report_lines))                      # Write all lines

        print(f"  Saved pipeline summary: {report_path}")         # Confirm

    # ========================================================================
    # MASTER PIPELINE EXECUTION
    # ========================================================================

    def run_full_pipeline(self):
        """
        Execute the COMPLETE feature engineering pipeline end-to-end.

        This is the main entry point. It chains all 7 steps in order:
            1. Load Data
            2. Numerical Feature Engineering (36+ features)
            3. NLP Feature Engineering (47 features)
            4. Categorical Feature Engineering (31+ features)
            5. Feature Quality Checks (clean, deduplicate)
            6. Generate Feature Registry (documentation)
            7. Save Outputs (matrix, registry, report)

        Returns:
            pd.DataFrame — the final ML-ready feature matrix
        """

        # Print grand pipeline header
        print("\n")                                               # Blank line
        print("#" * 70)                                           # Heavy separator
        print("#  FEATURE ENGINEERING MASTER PIPELINE")           # Grand title
        print("#  ESG Greenwashing Detection Project")            # Project name
        print("#" * 70)                                           # Heavy separator

        # Record start time for execution duration tracking
        start_time = datetime.now()                               # Capture start time

        # ------------------------------------------------------------------
        # Execute all 7 pipeline steps in sequence
        # ------------------------------------------------------------------
        df = self.load_data()                                     # Step 1: Load data
        df = self.run_numerical_features(df)                      # Step 2: Numerical features
        df = self.run_nlp_features(df)                            # Step 3: NLP features
        df = self.run_categorical_features(df)                    # Step 4: Categorical features
        df = self.run_feature_quality_checks(df)                  # Step 5: Quality checks
        registry_df = self.generate_feature_registry(df)          # Step 6: Feature registry
        self.save_outputs(df, registry_df)                        # Step 7: Save outputs

        # ------------------------------------------------------------------
        # Print final execution summary
        # ------------------------------------------------------------------
        end_time = datetime.now()                                 # Capture end time
        duration = (end_time - start_time).total_seconds()        # Calculate duration

        print("\n")                                               # Blank line
        print("#" * 70)                                           # Heavy separator
        print("#  PIPELINE COMPLETE")                             # Completion title
        print(f"#  Execution time: {duration:.2f} seconds")       # Duration
        print(f"#  Final feature matrix: {df.shape[0]} companies x {df.shape[1]} features")
        print(f"#  Numerical: {self.pipeline_stats.get('numerical_features', 0)} features")
        print(f"#  NLP:       {self.pipeline_stats.get('nlp_features', 0)} features")
        print(f"#  Categorical: {self.pipeline_stats.get('categorical_features', 0)} features")
        print("#" * 70)                                           # Heavy separator

        return df                                                 # Return final feature matrix


# ============================================================================
# STANDALONE EXECUTION (when run as: python feature_engineering_pipeline.py)
# ============================================================================

if __name__ == "__main__":                                        # Only run if executed directly

    # Create the pipeline instance with default data directory
    pipeline = FeatureEngineeringPipeline(                        # Initialize pipeline
        data_dir='data/processed'                                 # Path to processed data
    )

    # Execute the full pipeline and get the feature matrix
    feature_matrix = pipeline.run_full_pipeline()                 # Run all 7 steps

    # ------------------------------------------------------------------
    # Display final diagnostics
    # ------------------------------------------------------------------
    print("\n  FINAL DIAGNOSTICS:")                               # Diagnostics header
    print(f"  Shape: {feature_matrix.shape}")                     # Final dimensions

    # Show numeric column count
    numeric_count = feature_matrix.select_dtypes(                 # Count numeric columns
        include=[np.number]                                       # Only numeric types
    ).shape[1]                                                    # Column count
    print(f"  Numeric features: {numeric_count}")                 # Print count

    # Show non-numeric column count
    non_numeric_count = feature_matrix.select_dtypes(             # Count non-numeric columns
        exclude=[np.number]                                       # Exclude numeric types
    ).shape[1]                                                    # Column count
    print(f"  Non-numeric columns: {non_numeric_count}")          # Print count

    # Show data types distribution
    print(f"\n  Data type distribution:")                         # Header
    print(feature_matrix.dtypes.value_counts().to_string())       # Print dtype counts

    # Show sample of the first 5 rows and key features
    key_features = [                                              # Select key features
        'company_name',                                           # Identifier
        'total_esg_risk_score',                                   # Raw ESG score
        'pillar_imbalance_score',                                 # Numerical: pillar balance
        'esg_controversy_divergence',                             # Numerical: key GW signal
        'greenwashing_signal_score',                              # NLP: linguistic GW score
        'vague_to_concrete_ratio',                                # NLP: vagueness metric
        'risk_controversy_mismatch',                              # Categorical: mismatch flag
        'esg_performance_tier'                                    # Categorical: overall tier
    ]
    # Filter to only columns that exist
    key_features = [c for c in key_features if c in feature_matrix.columns]

    print(f"\n  Key features sample (top 5):")                    # Header
    print(feature_matrix[key_features].head().to_string())        # Print sample

    print("\n  Feature engineering pipeline completed successfully!")  # Final message
