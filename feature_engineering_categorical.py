"""
================================================================================
FEATURE ENGINEERING - CATEGORICAL FEATURES MODULE
================================================================================
Project : ESG Greenwashing Detection via NLP & Machine Learning
Purpose : Transform categorical variables (sector, industry, risk levels) into
          ML-ready encodings using multiple strategies: ordinal, frequency,
          target-mean, binary binning, and cross-category derived features.

Design Philosophy:
    Categorical features require careful encoding because:
    1. Tree models (XGBoost) handle ordinal encoding natively
    2. Linear models need one-hot or target encoding
    3. High-cardinality categoricals (123 industries) need smart grouping
    4. Domain knowledge enables meaningful bin boundaries

    We generate MULTIPLE encoding variants so downstream models can pick
    the most effective representation during feature selection.
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                  # Core data manipulation library
import numpy as np                   # Numerical computing for arrays
from collections import Counter     # Counting utility for frequency encoding
import warnings                      # Warnings control module
warnings.filterwarnings('ignore')    # Suppress non-critical warnings


# ============================================================================
# CLASS: CategoricalFeatureEngineer
# ============================================================================

class CategoricalFeatureEngineer:
    """
    Transform categorical variables into ML-ready numerical features.

    Implements 5 encoding strategies:
        1. Frequency Encoding     — replace category with its occurrence count
        2. Risk-Based Binning     — domain-driven discretization of scores
        3. Cross-Feature Derivation — combine categoricals to create new features
        4. Material ESG Issues    — encode ESG issue types as numerical features
        5. Sector Risk Profiling  — aggregate sector-level risk characteristics

    Each strategy has specific advantages for different ML model types.
    """

    def __init__(self):
        """
        Initialize CategoricalFeatureEngineer with encoding maps and registries.

        Attributes:
            feature_registry    : dict — tracks all generated features with metadata
            encoding_maps       : dict — stores fitted encoding mappings for reuse
            sector_risk_profiles: dict — caches sector-level risk statistics
        """

        self.feature_registry = {}                                # Track generated features
        self.encoding_maps = {}                                   # Store encoding mappings
        self.sector_risk_profiles = {}                            # Cache sector risk profiles

    # ========================================================================
    # CATEGORY 1: FREQUENCY ENCODING
    # ========================================================================

    def create_frequency_encodings(self, df):
        """
        Replace categorical values with their frequency counts (or proportions).

        Rationale:
            Frequency encoding captures "how common is this category?"
            - Common sectors (Technology, Financial) have more data → more reliable scores
            - Rare industries may have unusual ESG patterns due to small sample size
            - Frequency can serve as a proxy for market representation

        Advantages over one-hot encoding:
            - Creates only 1 column per feature (vs N columns for N categories)
            - Handles unseen categories gracefully (default to 0)
            - Preserves ordinal information (more frequent = higher value)

        Features Created:
            - sector_frequency          : How many companies share this sector
            - sector_proportion         : Sector frequency / total companies
            - industry_frequency        : How many companies share this industry
            - industry_proportion       : Industry frequency / total companies

        Parameters:
            df : pd.DataFrame — must contain 'sector' and/or 'industry' columns

        Returns:
            pd.DataFrame — with frequency-encoded columns added
        """

        # Print section header
        print("    [1/5] Creating frequency encodings...")         # Status message

        # ------------------------------------------------------------------
        # Sector frequency encoding
        # ------------------------------------------------------------------
        if 'sector' in df.columns:                                # Check if sector exists
            # Count how many companies are in each sector
            sector_counts = df['sector'].value_counts()           # Series: sector → count
            total_companies = len(df)                             # Total number of companies

            # Map each company's sector to its frequency count
            # Example: 'Technology' appears 67 times → all tech companies get 67
            df['sector_frequency'] = (                            # Create frequency column
                df['sector'].map(sector_counts)                   # Map sector name to count
            )

            # Convert to proportion (0 to 1) — what fraction of all companies
            df['sector_proportion'] = (                           # Create proportion column
                df['sector_frequency'] / total_companies          # Frequency / total
            )

            # Store encoding map for potential reuse on new data
            self.encoding_maps['sector_freq'] = (                 # Save mapping
                sector_counts.to_dict()                           # Convert to dictionary
            )

        # ------------------------------------------------------------------
        # Industry frequency encoding
        # ------------------------------------------------------------------
        if 'industry' in df.columns:                              # Check if industry exists
            # Count how many companies are in each industry
            industry_counts = df['industry'].value_counts()       # Series: industry → count

            # Map each company's industry to its frequency count
            df['industry_frequency'] = (                          # Create frequency column
                df['industry'].map(industry_counts)               # Map industry to count
            )

            # Convert to proportion
            df['industry_proportion'] = (                         # Create proportion column
                df['industry_frequency'] / total_companies        # Frequency / total
            )

            # Store encoding map
            self.encoding_maps['industry_freq'] = (               # Save mapping
                industry_counts.to_dict()                         # Convert to dictionary
            )

        # Register features
        self.feature_registry['frequency_encoding'] = {           # Store metadata
            'count': 4,                                           # Feature count
            'features': ['sector_frequency', 'sector_proportion',  # Feature list
                        'industry_frequency', 'industry_proportion']
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 2: RISK-BASED BINNING
    # ========================================================================

    def create_risk_bins(self, df):
        """
        Discretize continuous ESG scores into meaningful risk categories.

        Rationale:
            Continuous ESG scores can be noisy — small differences (e.g., 22.1 vs 22.3)
            are not meaningful. Binning groups companies into risk tiers that align
            with industry practice (Sustainalytics uses 5 risk categories).

            Bins are based on DOMAIN KNOWLEDGE, not arbitrary quantiles:
            - Negligible Risk : 0-10   (minimal ESG concerns)
            - Low Risk        : 10-20  (minor ESG concerns)
            - Medium Risk     : 20-30  (moderate ESG concerns)
            - High Risk       : 30-40  (significant ESG concerns)
            - Severe Risk     : 40+    (critical ESG concerns)

        Features Created:
            - esg_risk_bin              : Risk tier (0=negligible to 4=severe)
            - controversy_bin           : Controversy tier (0=none to 4=severe)
            - env_risk_bin              : Environmental risk tier
            - social_risk_bin           : Social risk tier
            - gov_risk_bin              : Governance risk tier
            - cross_risk_bin            : Combined risk tier (sum of pillar bins)
            - high_risk_flag            : Binary flag for high/severe risk
            - high_controversy_flag     : Binary flag for high controversy

        Parameters:
            df : pd.DataFrame — must contain ESG risk score columns

        Returns:
            pd.DataFrame — with risk bin columns added
        """

        # Print section header
        print("    [2/5] Creating risk-based bins...")             # Status message

        # ------------------------------------------------------------------
        # Step 1: Total ESG risk binning (Sustainalytics-aligned tiers)
        # ------------------------------------------------------------------
        # Define bin boundaries based on Sustainalytics risk categories
        esg_bins = [0, 10, 20, 30, 40, float('inf')]              # Bin edges
        esg_labels = [0, 1, 2, 3, 4]                              # Encoded labels (0-4)

        # pd.cut assigns each value to a bin based on the boundaries
        # right=False means bins are [left, right) — left-inclusive, right-exclusive
        df['esg_risk_bin'] = pd.cut(                              # Create bin column
            df['total_esg_risk_score'],                           # Values to bin
            bins=esg_bins,                                        # Bin boundaries
            labels=esg_labels,                                    # Assign integer labels
            right=False                                           # Left-inclusive bins
        ).astype(float).fillna(2)                                  # Convert to float, fill NaN

        # ------------------------------------------------------------------
        # Step 2: Controversy binning (1-5 scale → 0-4 encoded tiers)
        # ------------------------------------------------------------------
        # Controversy scores are already on a discrete 1-5 scale
        # We map them to risk tiers: 1→0 (none), 2→1 (low), ..., 5→4 (severe)
        df['controversy_bin'] = (                                 # Create bin column
            (df['controversy_score'] - 1)                         # Shift from 1-5 to 0-4
            .clip(0, 4)                                           # Ensure within bounds
        )

        # ------------------------------------------------------------------
        # Step 3: Individual pillar risk binning
        # ------------------------------------------------------------------
        # Environmental risk bins (same structure, different column)
        pillar_bins = [0, 3, 6, 10, 15, float('inf')]             # Pillar-scale boundaries
        pillar_labels = [0, 1, 2, 3, 4]                           # 0=negligible to 4=severe

        # Bin environmental risk scores
        df['env_risk_bin'] = pd.cut(                              # Create env bin
            df['env_risk_score'],                                 # Environmental scores
            bins=pillar_bins,                                     # Pillar boundaries
            labels=pillar_labels,                                 # Integer labels
            right=False                                           # Left-inclusive
        ).astype(float).fillna(2)                                  # Fill NaN with medium

        # Bin social risk scores
        df['social_risk_bin'] = pd.cut(                           # Create social bin
            df['social_risk_score'],                              # Social scores
            bins=pillar_bins,                                     # Pillar boundaries
            labels=pillar_labels,                                 # Integer labels
            right=False                                           # Left-inclusive
        ).astype(float).fillna(2)                                  # Fill NaN with medium

        # Bin governance risk scores
        df['gov_risk_bin'] = pd.cut(                              # Create gov bin
            df['gov_risk_score'],                                 # Governance scores
            bins=pillar_bins,                                     # Pillar boundaries
            labels=pillar_labels,                                 # Integer labels
            right=False                                           # Left-inclusive
        ).astype(float).fillna(2)                                  # Fill NaN with medium

        # ------------------------------------------------------------------
        # Step 4: Cross-pillar risk bin (sum of individual bins)
        # ------------------------------------------------------------------
        # Combined risk bin = env_bin + social_bin + gov_bin
        # Range: 0 (all negligible) to 12 (all severe)
        # Higher = more overall risk across all ESG dimensions
        df['cross_risk_bin'] = (                                  # Create combined bin
            df['env_risk_bin']                                    # Environmental tier
            + df['social_risk_bin']                               # Plus social tier
            + df['gov_risk_bin']                                  # Plus governance tier
        )

        # ------------------------------------------------------------------
        # Step 5: Binary high-risk flags
        # ------------------------------------------------------------------
        # Binary flag: is the company in high or severe risk category?
        # 1 = high/severe risk (bin 3 or 4), 0 = lower risk
        df['high_risk_flag'] = (                                  # Create binary flag
            (df['esg_risk_bin'] >= 3)                             # True if bin is 3 or 4
            .astype(int)                                          # Convert to 0/1
        )

        # Binary flag: does the company have high controversy?
        # Controversy score >= 4 = high or severe controversy
        df['high_controversy_flag'] = (                           # Create binary flag
            (df['controversy_score'] >= 4)                        # True if controversy >= 4
            .astype(int)                                          # Convert to 0/1
        )

        # Register features
        self.feature_registry['risk_bins'] = {                    # Store metadata
            'count': 8,                                           # Feature count
            'features': [                                         # Feature list
                'esg_risk_bin', 'controversy_bin',
                'env_risk_bin', 'social_risk_bin', 'gov_risk_bin',
                'cross_risk_bin', 'high_risk_flag', 'high_controversy_flag'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 3: CROSS-FEATURE DERIVED CATEGORICALS
    # ========================================================================

    def create_cross_feature_derivations(self, df):
        """
        Derive new categorical features by combining existing features.

        Rationale:
            Some patterns only emerge when you COMBINE features:
            - A "High ESG + High Controversy" company is very different from
              a "High ESG + Low Controversy" company
            - Sector × Risk Level captures industry-specific risk profiles
            - These combinations create natural "segments" of companies

        Features Created:
            - esg_controversy_segment    : Combined ESG risk × controversy level
            - risk_controversy_mismatch  : Flag when risk and controversy disagree
            - sector_risk_segment        : Sector grouped by risk level
            - esg_performance_tier       : Overall ESG performance classification

        Parameters:
            df : pd.DataFrame — must contain risk bins and categoricals

        Returns:
            pd.DataFrame — with derived categorical features added
        """

        # Print section header
        print("    [3/5] Creating cross-feature derived categoricals...")  # Status

        # ------------------------------------------------------------------
        # Step 1: ESG × Controversy segment
        # ------------------------------------------------------------------
        # Create a combined segment by pairing ESG risk bin with controversy bin
        # This creates natural company "archetypes" like:
        #   - 2_0 = Medium ESG risk + No controversy (normal company)
        #   - 1_3 = Low ESG risk + High controversy (GREENWASHING SUSPECT)
        #   - 3_3 = High ESG risk + High controversy (known bad actor)
        df['esg_controversy_segment'] = (                         # Create segment column
            df['esg_risk_bin'].astype(int).astype(str)            # Convert ESG bin to string
            + '_'                                                 # Separator
            + df['controversy_bin'].astype(int).astype(str)       # Controversy bin as string
        )

        # Encode the segment as a numerical hash for ML models
        # factorize() assigns a unique integer to each unique string
        df['esg_controversy_segment_encoded'] = (                 # Create encoded column
            pd.factorize(df['esg_controversy_segment'])[0]        # Integer encoding
        )

        # ------------------------------------------------------------------
        # Step 2: Risk-controversy mismatch flag
        # ------------------------------------------------------------------
        # CRITICAL GREENWASHING INDICATOR:
        # If ESG risk is LOW (bin 0 or 1) but controversy is HIGH (bin 3 or 4),
        # it means the company CLAIMS low risk but has high controversy
        # This mismatch is the textbook definition of greenwashing
        df['risk_controversy_mismatch'] = (                       # Create mismatch flag
            ((df['esg_risk_bin'] <= 1) &                          # Low ESG risk
             (df['controversy_bin'] >= 3))                        # AND high controversy
            .astype(int)                                          # Convert to 0/1
        )

        # Also create a continuous mismatch score (not just binary)
        # Larger gap between controversy and risk = stronger mismatch signal
        df['risk_controversy_gap'] = (                            # Create gap column
            df['controversy_bin']                                 # Controversy level
            - df['esg_risk_bin']                                  # Minus ESG risk level
        )

        # ------------------------------------------------------------------
        # Step 3: Sector × Risk Level segment (if sector exists)
        # ------------------------------------------------------------------
        if 'sector' in df.columns:                                # Check sector exists
            # Combine sector name with ESG risk bin
            # Example: "Technology_1" = tech company with low risk
            df['sector_risk_segment'] = (                         # Create segment column
                df['sector']                                      # Sector name
                + '_'                                             # Separator
                + df['esg_risk_bin'].astype(int).astype(str)      # Risk bin as string
            )

            # Encode as integer for ML models
            df['sector_risk_segment_encoded'] = (                 # Create encoded column
                pd.factorize(df['sector_risk_segment'])[0]        # Integer encoding
            )

        # ------------------------------------------------------------------
        # Step 4: Overall ESG performance tier
        # ------------------------------------------------------------------
        # Classify companies into 4 performance tiers based on combined signals
        # Uses ESG score percentile and controversy level together
        conditions = [                                            # Define conditions list
            # Tier 3 (Best): Low risk AND low controversy
            (df['esg_risk_bin'] <= 1) & (df['controversy_bin'] <= 1),  # Condition 1

            # Tier 2 (Good): Medium risk OR moderate controversy
            (df['esg_risk_bin'] <= 2) & (df['controversy_bin'] <= 2),  # Condition 2

            # Tier 1 (Concerning): High risk OR high controversy
            (df['esg_risk_bin'] <= 3) | (df['controversy_bin'] <= 3),  # Condition 3
        ]

        choices = [3, 2, 1]                                       # Tier labels (3=best, 1=concern)

        # np.select applies conditions in order, first match wins
        # Default 0 = worst tier (severe risk AND severe controversy)
        df['esg_performance_tier'] = np.select(                   # Create tier column
            conditions,                                           # List of conditions
            choices,                                              # Corresponding tier values
            default=0                                             # Default for no match
        )

        # Register features
        self.feature_registry['cross_feature'] = {                # Store metadata
            'count': 6,                                           # Feature count
            'features': [                                         # Feature list
                'esg_controversy_segment_encoded',
                'risk_controversy_mismatch', 'risk_controversy_gap',
                'sector_risk_segment_encoded',
                'esg_performance_tier', 'esg_controversy_segment'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 4: MATERIAL ESG ISSUE ENCODING
    # ========================================================================

    def encode_material_esg_issues(self, df):
        """
        Encode Material ESG Issues into numerical features.

        Rationale:
            The NIFTY50 dataset contains 3 "Material ESG Issues" per company
            (e.g., "Business Ethics", "Carbon", "Human Capital").
            These represent the most significant ESG concerns for each company.

            Encoding them captures:
            - WHICH issues a company faces (one-hot per issue type)
            - HOW MANY unique issue types exist (diversity of concerns)
            - Whether certain high-risk issues are present (binary flags)

        Features Created:
            - has_material_issues       : Binary flag if material issues exist
            - material_issue_count      : Number of non-null material issues (0-3)
            - has_carbon_issue          : Binary flag for carbon-related issues
            - has_ethics_issue          : Binary flag for ethics-related issues
            - has_human_capital_issue   : Binary flag for human capital issues
            - has_governance_issue      : Binary flag for governance-related issues

        Parameters:
            df : pd.DataFrame — may contain Material ESG Issue columns

        Returns:
            pd.DataFrame — with encoded material issue features added
        """

        # Print section header
        print("    [4/5] Encoding material ESG issues...")         # Status message

        # Check which Material ESG Issue columns exist in the dataframe
        issue_columns = [                                         # Possible issue column names
            'Material ESG Issues 1',                              # First material issue
            'Material ESG Issues 2',                              # Second material issue
            'Material ESG Issues 3'                               # Third material issue
        ]

        # Filter to only columns that actually exist
        existing_issue_cols = [                                   # Find existing columns
            col for col in issue_columns                          # For each possible column
            if col in df.columns                                  # If it exists in dataframe
        ]

        if len(existing_issue_cols) == 0:                         # No issue columns found
            # Create default features with zeros for datasets without material issues
            print("      No material ESG issue columns found, creating defaults...")
            df['has_material_issues'] = 0                         # Default: no issues
            df['material_issue_count'] = 0                        # Default: zero count
            df['has_carbon_issue'] = 0                            # Default: no carbon issue
            df['has_ethics_issue'] = 0                             # Default: no ethics issue
            df['has_human_capital_issue'] = 0                     # Default: no HC issue
            df['has_governance_issue'] = 0                         # Default: no gov issue

        else:                                                     # Issue columns exist
            # Has material issues: 1 if ANY material issue column has data
            df['has_material_issues'] = (                         # Create flag
                df[existing_issue_cols]                            # Select issue columns
                .notna()                                          # Check if not null
                .any(axis=1)                                      # True if any column has data
                .astype(int)                                      # Convert to 0/1
            )

            # Material issue count: how many of the 3 issue slots are filled
            df['material_issue_count'] = (                        # Create count column
                df[existing_issue_cols]                            # Select issue columns
                .notna()                                          # Check each for non-null
                .sum(axis=1)                                      # Count non-null per row
            )

            # Concatenate all issue text into a single string for keyword matching
            df['_all_issues_text'] = (                            # Temporary combined column
                df[existing_issue_cols]                            # Select issue columns
                .fillna('')                                       # Replace NaN with empty string
                .apply(                                           # Apply function row-wise
                    lambda row: ' '.join(row).lower(),            # Join all issues, lowercase
                    axis=1                                        # Apply across columns
                )
            )

            # Binary flag: does this company have carbon-related material issues?
            # Carbon issues include: Carbon, Climate, GHG, Emissions
            df['has_carbon_issue'] = (                            # Create carbon flag
                df['_all_issues_text']                            # Combined issues text
                .str.contains(                                    # Search for carbon keywords
                    'carbon|climate|ghg|emission',                # Regex pattern
                    regex=True                                    # Enable regex matching
                )
                .astype(int)                                      # Convert to 0/1
            )

            # Binary flag: does this company have ethics-related material issues?
            df['has_ethics_issue'] = (                             # Create ethics flag
                df['_all_issues_text']                            # Combined issues text
                .str.contains(                                    # Search for ethics keywords
                    'ethic|corruption|bribery|fraud',             # Regex pattern
                    regex=True                                    # Enable regex
                )
                .astype(int)                                      # Convert to 0/1
            )

            # Binary flag: human capital issues (labor, safety, workforce)
            df['has_human_capital_issue'] = (                     # Create HC flag
                df['_all_issues_text']                            # Combined issues text
                .str.contains(                                    # Search for HC keywords
                    'human capital|labor|labour|workforce|safety|health',  # Pattern
                    regex=True                                    # Enable regex
                )
                .astype(int)                                      # Convert to 0/1
            )

            # Binary flag: governance-related issues
            df['has_governance_issue'] = (                         # Create governance flag
                df['_all_issues_text']                            # Combined issues text
                .str.contains(                                    # Search for gov keywords
                    'governance|board|compliance|oversight',      # Regex pattern
                    regex=True                                    # Enable regex
                )
                .astype(int)                                      # Convert to 0/1
            )

            # Drop the temporary combined text column
            df = df.drop(columns=['_all_issues_text'])            # Remove temp column

        # Register features
        self.feature_registry['material_issues'] = {              # Store metadata
            'count': 6,                                           # Feature count
            'features': [                                         # Feature list
                'has_material_issues', 'material_issue_count',
                'has_carbon_issue', 'has_ethics_issue',
                'has_human_capital_issue', 'has_governance_issue'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 5: SECTOR RISK PROFILING
    # ========================================================================

    def create_sector_risk_profiles(self, df):
        """
        Create sector-level risk profile features for each company.

        Rationale:
            Different sectors have inherently different ESG risk profiles:
            - Energy/Mining: high environmental risk by nature
            - Financial Services: high governance risk, low environmental
            - Technology: moderate social risk (privacy, labor), low environmental

            Sector risk profiles capture "how risky is this sector typically?"
            which helps the model understand if a company's risk is normal
            or unusual for its industry.

        Features Created:
            - sector_avg_esg            : Average ESG score in this sector
            - sector_avg_controversy    : Average controversy in this sector
            - sector_max_esg            : Highest ESG score in this sector
            - sector_min_esg            : Lowest ESG score in this sector
            - sector_esg_spread         : Range (max - min) within sector
            - sector_company_count      : Number of companies in this sector
            - sector_high_risk_ratio    : Fraction of sector in high risk

        Parameters:
            df : pd.DataFrame — must contain 'sector' and ESG score columns

        Returns:
            pd.DataFrame — with sector risk profile features added
        """

        # Print section header
        print("    [5/5] Creating sector risk profiles...")        # Status message

        # Check if sector column exists
        if 'sector' not in df.columns:                            # Guard clause
            print("      WARNING: 'sector' column not found, skipping")
            # Create zero-filled defaults
            for col in ['sector_avg_esg', 'sector_avg_controversy',
                       'sector_max_esg', 'sector_min_esg',
                       'sector_esg_spread', 'sector_company_count',
                       'sector_high_risk_ratio']:                 # For each expected column
                df[col] = 0                                       # Fill with zeros
            return df                                             # Return with defaults

        # ------------------------------------------------------------------
        # Step 1: Compute sector-level aggregate statistics
        # ------------------------------------------------------------------
        # Group by sector and compute multiple aggregations at once
        sector_profiles = df.groupby('sector').agg(               # Group by sector
            sector_avg_esg=('total_esg_risk_score', 'mean'),      # Mean ESG per sector
            sector_avg_controversy=('controversy_score', 'mean'), # Mean controversy per sector
            sector_max_esg=('total_esg_risk_score', 'max'),       # Max ESG per sector
            sector_min_esg=('total_esg_risk_score', 'min'),       # Min ESG per sector
            sector_company_count=('total_esg_risk_score', 'count')  # Company count per sector
        ).reset_index()                                           # Reset index to column

        # Calculate ESG spread within sector (max - min)
        sector_profiles['sector_esg_spread'] = (                  # Create spread column
            sector_profiles['sector_max_esg']                     # Maximum ESG in sector
            - sector_profiles['sector_min_esg']                   # Minus minimum ESG
        )

        # ------------------------------------------------------------------
        # Step 2: Calculate high-risk ratio per sector
        # ------------------------------------------------------------------
        # What fraction of companies in this sector are high risk?
        if 'high_risk_flag' in df.columns:                        # Check flag exists
            high_risk_by_sector = (                               # Calculate per sector
                df.groupby('sector')['high_risk_flag']            # Group by sector
                .mean()                                           # Mean of binary = proportion
                .reset_index()                                    # Reset index
                .rename(columns={'high_risk_flag': 'sector_high_risk_ratio'})  # Rename
            )
            # Merge high-risk ratio into sector profiles
            sector_profiles = sector_profiles.merge(              # Merge operation
                high_risk_by_sector,                              # Right table
                on='sector',                                      # Join key
                how='left'                                        # Left join
            )
        else:                                                     # Flag doesn't exist
            sector_profiles['sector_high_risk_ratio'] = 0         # Default to 0

        # Cache sector profiles for potential later use
        self.sector_risk_profiles = sector_profiles.copy()        # Store copy

        # ------------------------------------------------------------------
        # Step 3: Merge sector profiles back into company-level dataframe
        # ------------------------------------------------------------------
        # Drop any existing sector profile columns to avoid duplicates
        existing_profile_cols = [                                 # List of profile columns
            col for col in sector_profiles.columns                # For each profile column
            if col != 'sector' and col in df.columns              # If it already exists in df
        ]
        if existing_profile_cols:                                 # If there are duplicates
            df = df.drop(columns=existing_profile_cols)           # Drop them first

        # Left join: attach sector profile to each company
        df = df.merge(                                            # Merge operation
            sector_profiles,                                      # Right table: profiles
            on='sector',                                          # Join key: sector name
            how='left'                                            # Left join: keep all companies
        )

        # Register features
        self.feature_registry['sector_profiles'] = {              # Store metadata
            'count': 7,                                           # Feature count
            'features': [                                         # Feature list
                'sector_avg_esg', 'sector_avg_controversy',
                'sector_max_esg', 'sector_min_esg',
                'sector_esg_spread', 'sector_company_count',
                'sector_high_risk_ratio'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # MASTER EXECUTION METHOD
    # ========================================================================

    def engineer_all_categorical_features(self, df):
        """
        Execute the complete categorical feature engineering pipeline.

        Chains all 5 categorical encoding strategies in dependency order.

        Pipeline Order:
            1. Frequency Encodings      → 4 features  (independent)
            2. Risk-Based Binning       → 8 features  (independent)
            3. Cross-Feature Derivation → 6 features  (depends on step 2)
            4. Material ESG Issues      → 6 features  (independent)
            5. Sector Risk Profiles     → 7 features  (depends on step 2)

        Parameters:
            df : pd.DataFrame — company data with categorical columns

        Returns:
            pd.DataFrame — with all 31 categorical features added
        """

        # Print pipeline header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  CATEGORICAL FEATURE ENGINEERING PIPELINE")       # Pipeline title
        print("=" * 70)                                           # Visual separator

        # Execute each encoding step in dependency order
        df = self.create_frequency_encodings(df)                  # Step 1: Frequencies (4)
        df = self.create_risk_bins(df)                            # Step 2: Risk bins (8)
        df = self.create_cross_feature_derivations(df)            # Step 3: Cross-feature (6)
        df = self.encode_material_esg_issues(df)                  # Step 4: Material issues (6)
        df = self.create_sector_risk_profiles(df)                 # Step 5: Sector profiles (7)

        # Print summary report
        total_features = sum(                                     # Count total features
            info['count'] for info in self.feature_registry.values()
        )
        print(f"\n    TOTAL CATEGORICAL FEATURES ENGINEERED: {total_features}")
        for category, info in self.feature_registry.items():      # Per-category breakdown
            print(f"      - {category}: {info['count']} features")
        print("=" * 70)                                           # Visual separator

        return df                                                 # Return fully enriched df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":                                        # Only run if executed directly

    # Print script header
    print("=" * 70)                                               # Visual separator
    print("  CATEGORICAL FEATURE ENGINEERING - STANDALONE TEST")  # Title
    print("=" * 70)                                               # Visual separator

    # Load company profiles data
    DATA_PATH = "data/processed/company_profiles.csv"             # Input path
    print(f"\n  Loading data from: {DATA_PATH}")                  # Log
    df = pd.read_csv(DATA_PATH)                                   # Read CSV
    print(f"  Initial shape: {df.shape}")                         # Print dimensions

    # Initialize and run
    cat_engineer = CategoricalFeatureEngineer()                   # Create instance
    df_cat = cat_engineer.engineer_all_categorical_features(df)   # Run pipeline

    # Display results
    print(f"\n  Final shape: {df_cat.shape}")                     # Print dimensions
    print(f"  Categorical features added: {df_cat.shape[1] - 13}")

    # Show sample
    key_cat = [                                                   # Key features to display
        'company_name', 'sector',                                 # Identifiers
        'sector_frequency', 'esg_risk_bin',                       # Encodings
        'risk_controversy_mismatch', 'esg_performance_tier',      # Derived
        'sector_avg_esg', 'sector_high_risk_ratio'                # Profiles
    ]
    key_cat = [c for c in key_cat if c in df_cat.columns]         # Filter existing
    print(f"\n  Sample (top 10):")                                 # Header
    print(df_cat[key_cat].head(10).to_string())                   # Print sample

    # Save
    OUTPUT_PATH = "data/processed/categorical_features.csv"       # Output path
    df_cat.to_csv(OUTPUT_PATH, index=False)                       # Save
    print(f"\n  Saved to: {OUTPUT_PATH}")                         # Confirm
    print("=" * 70)                                               # Separator
