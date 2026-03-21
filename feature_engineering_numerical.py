"""
================================================================================
FEATURE ENGINEERING - NUMERICAL FEATURES MODULE
================================================================================
Design Philosophy:
    - Domain-driven feature creation (not blind polynomial expansion)
    - Every feature has a greenwashing detection rationale documented
    - Features grouped by: (1) ESG Pillar Ratios, (2) Risk Decomposition,
      (3) Statistical Moments, (4) Interaction Terms, (5) Anomaly Indicators

Theory:
    Greenwashing companies exhibit specific numerical patterns:
    - Disproportionate ESG pillar scores (e.g., high governance, low environmental)
    - High controversy relative to claimed ESG performance
    - Extreme outlier behavior compared to sector peers
    - Temporal volatility in ESG scores (frequent revisions)
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                  # Core data manipulation library for tabular data
import numpy as np                   # Numerical computing library for array operations
from scipy import stats              # Statistical functions (z-score, skewness, kurtosis)
from sklearn.preprocessing import (  # Scikit-learn preprocessing utilities
    StandardScaler,                  # Zero-mean, unit-variance normalization
    MinMaxScaler,                    # Min-max normalization to [0, 1] range
    RobustScaler,                    # Median/IQR-based scaling (robust to outliers)
    PowerTransformer                 # Yeo-Johnson transform for Gaussian-like distributions
)
import warnings                      # Python warnings control module
warnings.filterwarnings('ignore')    # Suppress non-critical warnings for clean output


# ============================================================================
# CLASS: NumericalFeatureEngineer
# ============================================================================

class NumericalFeatureEngineer:
    """
    Comprehensive numerical feature engineering for ESG greenwashing detection.

    This class implements 5 categories of numerical features:
        1. ESG Pillar Ratio Features — relative balance between E, S, G scores
        2. Risk Decomposition Features — break down total risk into components
        3. Statistical Moment Features — variance, skewness, kurtosis of scores
        4. Interaction Features — cross-feature products and ratios
        5. Anomaly Detection Features — z-scores, outlier flags, deviations

    Each feature is designed based on domain knowledge of how greenwashing
    manifests in quantitative ESG data.
    """

    def __init__(self):
        """
        Initialize the NumericalFeatureEngineer with empty storage.

        Attributes:
            feature_registry : dict — tracks all generated features with metadata
            scalers          : dict — stores fitted scaler objects for each method
            sector_stats     : dict — caches sector-level statistics for peer comparison
        """
        self.feature_registry = {}    # Dictionary to track every feature we create
        self.scalers = {}             # Dictionary to store fitted scaler objects
        self.sector_stats = {}        # Cache for sector-level aggregate statistics

    # ========================================================================
    # CATEGORY 1: ESG PILLAR RATIO FEATURES
    # ========================================================================

    def create_esg_pillar_ratios(self, df):
        """
        Create ratio features between Environmental, Social, and Governance pillars.

        Rationale:
            Greenwashing companies often show IMBALANCED pillar scores.
            Example: A company might invest heavily in governance (board structure)
            while neglecting actual environmental action. Ratios capture this
            imbalance more effectively than raw scores alone.

        Features Created:
            - env_to_total_ratio        : Environmental contribution to total ESG
            - social_to_total_ratio     : Social contribution to total ESG
            - gov_to_total_ratio        : Governance contribution to total ESG
            - env_gov_ratio             : Environmental vs Governance balance
            - env_social_ratio          : Environmental vs Social balance
            - social_gov_ratio          : Social vs Governance balance
            - pillar_imbalance_score    : Standard deviation across 3 pillar ratios
            - dominant_pillar_strength  : How much the strongest pillar dominates
            - weakest_pillar_weakness   : How much the weakest pillar lags behind

        Parameters:
            df : pd.DataFrame — must contain 'env_risk_score', 'social_risk_score',
                                'gov_risk_score', 'total_esg_risk_score' columns

        Returns:
            pd.DataFrame — input dataframe with new pillar ratio columns added
        """

        # Print section header for console logging during pipeline execution
        print("    [1/5] Engineering ESG pillar ratio features...")  # Status message

        # ------------------------------------------------------------------
        # Step 1: Calculate each pillar's proportion of total ESG risk
        # ------------------------------------------------------------------
        # Environmental risk as fraction of total — shows how much of the
        # company's ESG risk comes from environmental factors
        # Adding 1e-8 (0.00000001) to denominator prevents division by zero
        df['env_to_total_ratio'] = (                              # Create new column
            df['env_risk_score']                                  # Numerator: environmental score
            / (df['total_esg_risk_score'] + 1e-8)                 # Denominator: total score + epsilon
        )

        # Social risk as fraction of total — indicates social risk contribution
        df['social_to_total_ratio'] = (                           # Create new column
            df['social_risk_score']                               # Numerator: social score
            / (df['total_esg_risk_score'] + 1e-8)                 # Denominator: total score + epsilon
        )

        # Governance risk as fraction of total — indicates governance risk share
        df['gov_to_total_ratio'] = (                              # Create new column
            df['gov_risk_score']                                  # Numerator: governance score
            / (df['total_esg_risk_score'] + 1e-8)                 # Denominator: total score + epsilon
        )

        # ------------------------------------------------------------------
        # Step 2: Calculate pairwise pillar ratios (inter-pillar balance)
        # ------------------------------------------------------------------
        # Environmental vs Governance — high ratio means env risk >> gov risk
        # This is a KEY greenwashing indicator: good governance but poor env action
        df['env_gov_ratio'] = (                                   # Create ratio column
            df['env_risk_score']                                  # Numerator: environmental
            / (df['gov_risk_score'] + 1e-8)                       # Denominator: governance + epsilon
        )

        # Environmental vs Social — captures if env and social efforts are aligned
        df['env_social_ratio'] = (                                # Create ratio column
            df['env_risk_score']                                  # Numerator: environmental
            / (df['social_risk_score'] + 1e-8)                    # Denominator: social + epsilon
        )

        # Social vs Governance — captures social-governance alignment
        df['social_gov_ratio'] = (                                # Create ratio column
            df['social_risk_score']                               # Numerator: social
            / (df['gov_risk_score'] + 1e-8)                       # Denominator: governance + epsilon
        )

        # ------------------------------------------------------------------
        # Step 3: Calculate pillar imbalance metrics
        # ------------------------------------------------------------------
        # Stack the three pillar ratios into a temporary array for stats
        pillar_ratios = df[['env_to_total_ratio',                 # Select the 3 ratio columns
                            'social_to_total_ratio',              # we just created
                            'gov_to_total_ratio']]                # as a sub-dataframe

        # Pillar imbalance = standard deviation across the 3 ratios (row-wise)
        # High std = one pillar dominates, others are weak = greenwashing signal
        df['pillar_imbalance_score'] = pillar_ratios.std(axis=1)  # Row-wise std deviation

        # Dominant pillar strength = max ratio value across the 3 pillars
        # If max is close to 1.0, almost ALL risk comes from one pillar
        df['dominant_pillar_strength'] = pillar_ratios.max(axis=1)  # Row-wise maximum

        # Weakest pillar weakness = min ratio value across the 3 pillars
        # If min is close to 0.0, company is neglecting an entire ESG dimension
        df['weakest_pillar_weakness'] = pillar_ratios.min(axis=1)   # Row-wise minimum

        # Register all created features in our tracking dictionary
        self.feature_registry['pillar_ratios'] = {                # Store metadata
            'count': 9,                                           # Number of features created
            'features': [                                         # List of feature names
                'env_to_total_ratio', 'social_to_total_ratio',
                'gov_to_total_ratio', 'env_gov_ratio',
                'env_social_ratio', 'social_gov_ratio',
                'pillar_imbalance_score', 'dominant_pillar_strength',
                'weakest_pillar_weakness'
            ]
        }

        # Return the dataframe with 9 new pillar ratio features added
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 2: RISK DECOMPOSITION FEATURES
    # ========================================================================

    def create_risk_decomposition_features(self, df):
        """
        Decompose ESG risk scores into residual, relative, and gap-based features.

        Rationale:
            Raw ESG scores alone don't capture HOW risk is distributed.
            A total score of 25 could mean {E:8, S:8, G:9} (balanced) or
            {E:20, S:3, G:2} (extremely skewed). Risk decomposition reveals
            these hidden patterns that correlate with greenwashing behavior.

        Features Created:
            - esg_risk_residual         : Unexplained risk (total - sum of pillars)
            - pillar_max_min_gap        : Spread between best and worst pillar
            - pillar_range_normalized   : Gap normalized by total score
            - controversy_risk_ratio    : Controversies relative to ESG score
            - controversy_adjusted_risk : Risk score amplified by controversy level
            - risk_concentration_index  : Herfindahl-like concentration of risk

        Parameters:
            df : pd.DataFrame — must contain ESG pillar scores and controversy scores

        Returns:
            pd.DataFrame — input dataframe with risk decomposition features added
        """

        # Print section header for progress tracking
        print("    [2/5] Engineering risk decomposition features...")  # Status message

        # ------------------------------------------------------------------
        # Step 1: Calculate ESG risk residual
        # ------------------------------------------------------------------
        # Sum the three individual pillar scores
        pillar_sum = (                                            # Calculate component sum
            df['env_risk_score']                                  # Environmental component
            + df['social_risk_score']                             # Plus social component
            + df['gov_risk_score']                                # Plus governance component
        )

        # Residual = total score minus sum of pillars
        # Non-zero residual means there are interaction effects or hidden factors
        # that the individual pillars don't explain — potential red flag
        df['esg_risk_residual'] = (                               # Create residual column
            df['total_esg_risk_score'] - pillar_sum               # Total minus component sum
        )

        # ------------------------------------------------------------------
        # Step 2: Calculate pillar gap and range features
        # ------------------------------------------------------------------
        # Stack pillar scores for row-wise min/max operations
        pillars = df[['env_risk_score',                           # Select environmental score
                      'social_risk_score',                        # Select social score
                      'gov_risk_score']]                          # Select governance score

        # Max-min gap across pillars — large gap = uneven ESG performance
        df['pillar_max_min_gap'] = (                              # Create gap column
            pillars.max(axis=1)                                   # Row-wise maximum pillar
            - pillars.min(axis=1)                                 # Minus row-wise minimum pillar
        )

        # Normalize the gap by total score — makes it comparable across companies
        # A gap of 5 means more when total is 10 than when total is 50
        df['pillar_range_normalized'] = (                         # Create normalized gap
            df['pillar_max_min_gap']                              # Raw gap value
            / (df['total_esg_risk_score'] + 1e-8)                 # Divided by total + epsilon
        )

        # ------------------------------------------------------------------
        # Step 3: Controversy-adjusted risk features
        # ------------------------------------------------------------------
        # Controversy-to-risk ratio: how controversial is the company
        # RELATIVE to its declared ESG risk score
        # High ratio = company has MORE controversies than its ESG score suggests
        # This is a STRONG greenwashing indicator: claiming low risk but causing harm
        df['controversy_risk_ratio'] = (                          # Create ratio column
            df['controversy_score']                               # Numerator: controversy level
            / (df['total_esg_risk_score'] + 1e-8)                 # Denominator: total risk + epsilon
        )

        # Controversy-adjusted risk: multiply risk by controversy
        # This amplifies risk score for companies with high controversy
        # Low ESG risk + High controversy = high adjusted risk = greenwashing
        df['controversy_adjusted_risk'] = (                       # Create adjusted score
            df['total_esg_risk_score']                            # Base risk score
            * (1 + df['controversy_score'])                       # Amplified by (1 + controversy)
        )

        # ------------------------------------------------------------------
        # Step 4: Risk concentration index (Herfindahl-Hirschman inspired)
        # ------------------------------------------------------------------
        # HHI measures concentration: sum of squared shares
        # If risk is concentrated in one pillar, HHI is high (~1.0)
        # If risk is evenly spread across pillars, HHI is low (~0.33)
        # Concentrated risk suggests company focuses on some areas, ignores others
        df['risk_concentration_index'] = (                        # Create HHI column
            df['env_to_total_ratio'] ** 2                         # Squared env share
            + df['social_to_total_ratio'] ** 2                    # Plus squared social share
            + df['gov_to_total_ratio'] ** 2                       # Plus squared gov share
        )

        # Register features in tracking dictionary
        self.feature_registry['risk_decomposition'] = {           # Store metadata
            'count': 6,                                           # Number of features
            'features': [                                         # Feature name list
                'esg_risk_residual', 'pillar_max_min_gap',
                'pillar_range_normalized', 'controversy_risk_ratio',
                'controversy_adjusted_risk', 'risk_concentration_index'
            ]
        }

        # Return dataframe with 6 new risk decomposition features
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 3: STATISTICAL MOMENT FEATURES
    # ========================================================================

    def create_statistical_features(self, df):
        """
        Generate statistical moment features from ESG score distributions.

        Rationale:
            Higher-order statistics (variance, skewness, kurtosis) reveal
            distributional properties that simple means and ratios miss.
            - Skewness: Are scores asymmetrically distributed? (tail risk)
            - Kurtosis: Are there extreme outlier scores? (heavy tails)
            - Coefficient of Variation: Relative variability (size-independent)

        Features Created:
            - esg_pillar_mean           : Average across 3 pillar scores
            - esg_pillar_variance       : Variance across 3 pillar scores
            - esg_pillar_skewness       : Skewness across 3 pillar scores
            - esg_pillar_cv             : Coefficient of variation (std/mean)
            - esg_score_zscore          : Z-score of total ESG vs population mean
            - controversy_zscore        : Z-score of controversy vs population mean
            - esg_controversy_divergence: Standardized gap between ESG and controversy

        Parameters:
            df : pd.DataFrame — must contain ESG pillar scores

        Returns:
            pd.DataFrame — input dataframe with statistical features added
        """

        # Print section header for progress tracking
        print("    [3/5] Engineering statistical moment features...")  # Status msg

        # ------------------------------------------------------------------
        # Step 1: Row-wise statistics across the 3 ESG pillars
        # ------------------------------------------------------------------
        # Select the three pillar score columns for row-wise calculations
        pillars = df[['env_risk_score',                           # Environmental pillar
                      'social_risk_score',                        # Social pillar
                      'gov_risk_score']]                          # Governance pillar

        # Mean across pillars — simple average of E, S, G for each company
        df['esg_pillar_mean'] = pillars.mean(axis=1)              # Row-wise mean

        # Variance across pillars — how spread out the 3 pillar scores are
        # High variance = pillars are very different from each other
        df['esg_pillar_variance'] = pillars.var(axis=1)           # Row-wise variance

        # Skewness across pillars — asymmetry of the 3 scores
        # Positive skew = one pillar score is much higher than the other two
        # Negative skew = one pillar score is much lower than the other two
        df['esg_pillar_skewness'] = pillars.skew(axis=1)          # Row-wise skewness

        # Coefficient of Variation (CV) = std / mean — relative variability
        # CV is scale-independent, so comparable across companies with different scales
        # High CV = inconsistent performance across pillars = potential greenwashing
        df['esg_pillar_cv'] = (                                   # Create CV column
            pillars.std(axis=1)                                   # Row-wise standard deviation
            / (pillars.mean(axis=1) + 1e-8)                       # Divided by row-wise mean
        )

        # ------------------------------------------------------------------
        # Step 2: Population-level z-scores
        # ------------------------------------------------------------------
        # Z-score of total ESG risk: how far is this company from the population mean
        # Z = (x - mean) / std — measures number of standard deviations from mean
        # Companies with extreme z-scores (|z| > 2) are unusual = worth investigating
        population_esg_mean = df['total_esg_risk_score'].mean()   # Population mean ESG
        population_esg_std = df['total_esg_risk_score'].std()     # Population std ESG

        df['esg_score_zscore'] = (                                # Create z-score column
            (df['total_esg_risk_score'] - population_esg_mean)    # x minus population mean
            / (population_esg_std + 1e-8)                         # Divided by population std
        )

        # Z-score of controversy: how controversial is this company vs peers
        population_cont_mean = df['controversy_score'].mean()     # Population mean controversy
        population_cont_std = df['controversy_score'].std()       # Population std controversy

        df['controversy_zscore'] = (                              # Create z-score column
            (df['controversy_score'] - population_cont_mean)      # x minus population mean
            / (population_cont_std + 1e-8)                        # Divided by population std
        )

        # ------------------------------------------------------------------
        # Step 3: ESG-Controversy divergence
        # ------------------------------------------------------------------
        # Divergence = z(controversy) - z(ESG_risk)
        # If a company has LOW ESG risk (negative z) but HIGH controversy (positive z),
        # the divergence is LARGE and POSITIVE = strong greenwashing signal
        # Logic: "They CLAIM low risk but ARE controversial"
        df['esg_controversy_divergence'] = (                      # Create divergence column
            df['controversy_zscore']                              # How controversial (standardized)
            - df['esg_score_zscore']                              # Minus how risky (standardized)
        )

        # Register features in tracking dictionary
        self.feature_registry['statistical_moments'] = {          # Store metadata
            'count': 7,                                           # Number of features
            'features': [                                         # Feature name list
                'esg_pillar_mean', 'esg_pillar_variance',
                'esg_pillar_skewness', 'esg_pillar_cv',
                'esg_score_zscore', 'controversy_zscore',
                'esg_controversy_divergence'
            ]
        }

        # Return dataframe with 7 new statistical features
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 4: INTERACTION FEATURES
    # ========================================================================

    def create_interaction_features(self, df):
        """
        Generate cross-feature interaction terms that capture non-linear relationships.

        Rationale:
            Greenwashing is often detectable not from single features but from
            COMBINATIONS of features. For example, a company with:
            - High ESG score + High controversy = suspicious (greenwashing)
            - High ESG score + Low controversy = legitimate
            - Low ESG score + High controversy = known bad actor (not greenwashing)

            Interaction terms let linear models capture these non-linear patterns
            without requiring complex architectures.

        Features Created:
            - env_controversy_interaction    : Environmental x Controversy
            - social_controversy_interaction : Social x Controversy
            - gov_controversy_interaction    : Governance x Controversy
            - esg_squared                    : Total ESG score squared (quadratic)
            - controversy_squared            : Controversy score squared (quadratic)
            - imbalance_controversy_interact : Pillar imbalance x Controversy
            - risk_per_pillar_point          : Total risk efficiency per pillar unit
            - env_dominance_flag             : Binary flag if env dominates risk profile

        Parameters:
            df : pd.DataFrame — must contain ESG scores and previously engineered features

        Returns:
            pd.DataFrame — input dataframe with interaction features added
        """

        # Print section header
        print("    [4/5] Engineering interaction features...")     # Status message

        # ------------------------------------------------------------------
        # Step 1: Pillar x Controversy interactions
        # ------------------------------------------------------------------
        # Environmental score multiplied by controversy score
        # High product = high env risk AND high controversy = double red flag
        df['env_controversy_interaction'] = (                     # Create interaction column
            df['env_risk_score']                                  # Environmental risk score
            * df['controversy_score']                             # Times controversy score
        )

        # Social score multiplied by controversy score
        # Captures companies with social risk that also have controversies
        df['social_controversy_interaction'] = (                  # Create interaction column
            df['social_risk_score']                               # Social risk score
            * df['controversy_score']                             # Times controversy score
        )

        # Governance score multiplied by controversy score
        # Good governance + high controversy = possible governance failure
        df['gov_controversy_interaction'] = (                     # Create interaction column
            df['gov_risk_score']                                  # Governance risk score
            * df['controversy_score']                             # Times controversy score
        )

        # ------------------------------------------------------------------
        # Step 2: Quadratic (squared) features
        # ------------------------------------------------------------------
        # Squared ESG score — captures non-linear relationship between
        # ESG risk and greenwashing (effect may accelerate at extremes)
        df['esg_squared'] = (                                     # Create squared column
            df['total_esg_risk_score'] ** 2                       # ESG score raised to power 2
        )

        # Squared controversy — extreme controversy has disproportionate impact
        # A company with controversy=4 is more than twice as concerning as controversy=2
        df['controversy_squared'] = (                             # Create squared column
            df['controversy_score'] ** 2                          # Controversy raised to power 2
        )

        # ------------------------------------------------------------------
        # Step 3: Compound interaction features
        # ------------------------------------------------------------------
        # Pillar imbalance x Controversy — companies with BOTH uneven ESG
        # profiles AND high controversy are the most suspicious
        # This is a CRITICAL greenwashing detector
        df['imbalance_controversy_interact'] = (                  # Create compound interaction
            df['pillar_imbalance_score']                          # How uneven are pillars
            * df['controversy_score']                             # Times controversy level
        )

        # Risk efficiency: total risk divided by number of pillar-score points
        # Measures how "efficiently" risk translates into pillar scores
        # Anomalous efficiency = risk is not being captured properly
        df['risk_per_pillar_point'] = (                           # Create efficiency metric
            df['total_esg_risk_score']                            # Total risk
            / (df['esg_pillar_mean'] * 3 + 1e-8)                  # Divided by sum of pillar means
        )

        # ------------------------------------------------------------------
        # Step 4: Dominance flag features
        # ------------------------------------------------------------------
        # Binary flag: is environmental risk the dominant (highest) pillar?
        # If env is dominant, the company's risk profile is environmentally heavy
        # 1 = environmental dominates, 0 = social or governance dominates
        df['env_dominance_flag'] = (                              # Create binary flag
            (df['env_risk_score'] ==                              # Check if env score equals
             df[['env_risk_score',                                # the maximum across
                 'social_risk_score',                             # all three
                 'gov_risk_score']].max(axis=1))                  # pillar scores
            .astype(int)                                          # Convert boolean to integer (0/1)
        )

        # Register features in tracking dictionary
        self.feature_registry['interactions'] = {                 # Store metadata
            'count': 8,                                           # Number of features
            'features': [                                         # Feature name list
                'env_controversy_interaction', 'social_controversy_interaction',
                'gov_controversy_interaction', 'esg_squared',
                'controversy_squared', 'imbalance_controversy_interact',
                'risk_per_pillar_point', 'env_dominance_flag'
            ]
        }

        # Return dataframe with 8 new interaction features
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 5: ANOMALY DETECTION FEATURES
    # ========================================================================

    def create_anomaly_features(self, df):
        """
        Generate anomaly and outlier detection features for greenwashing identification.

        Rationale:
            Greenwashing companies are statistical OUTLIERS by definition — they
            deviate from normal ESG behavior patterns. Anomaly features quantify
            how "unusual" each company is compared to its peers, using multiple
            statistical methods to ensure robust detection.

        Features Created:
            - esg_iqr_outlier           : Binary flag if ESG score is IQR outlier
            - controversy_iqr_outlier   : Binary flag if controversy is IQR outlier
            - esg_mad_score             : Median Absolute Deviation score for ESG
            - controversy_mad_score     : MAD score for controversy
            - combined_anomaly_score    : Weighted combination of all anomaly signals
            - esg_percentile_rank       : Percentile rank (0-1) of ESG score

        Parameters:
            df : pd.DataFrame — must contain total_esg_risk_score and controversy_score

        Returns:
            pd.DataFrame — input dataframe with anomaly detection features added
        """

        # Print section header
        print("    [5/5] Engineering anomaly detection features...")  # Status msg

        # ------------------------------------------------------------------
        # Step 1: IQR-based outlier detection
        # ------------------------------------------------------------------
        # Calculate Inter-Quartile Range for ESG risk score
        # IQR = Q3 - Q1 (the middle 50% spread)
        esg_q1 = df['total_esg_risk_score'].quantile(0.25)        # 25th percentile
        esg_q3 = df['total_esg_risk_score'].quantile(0.75)        # 75th percentile
        esg_iqr = esg_q3 - esg_q1                                 # Inter-quartile range

        # IQR outlier: value is below Q1-1.5*IQR or above Q3+1.5*IQR
        # This is the classic box-plot outlier detection method
        df['esg_iqr_outlier'] = (                                 # Create outlier flag
            ((df['total_esg_risk_score'] < (esg_q1 - 1.5 * esg_iqr))   # Below lower fence
             | (df['total_esg_risk_score'] > (esg_q3 + 1.5 * esg_iqr)))  # OR above upper fence
            .astype(int)                                          # Convert boolean to 0/1
        )

        # IQR outlier detection for controversy score
        cont_q1 = df['controversy_score'].quantile(0.25)          # 25th percentile
        cont_q3 = df['controversy_score'].quantile(0.75)          # 75th percentile
        cont_iqr = cont_q3 - cont_q1                              # Inter-quartile range

        # Flag if controversy score is an IQR outlier
        # Edge case: if IQR is 0 (all values same), use abs deviation > 0
        if cont_iqr > 0:                                          # Normal case: IQR exists
            df['controversy_iqr_outlier'] = (                     # Create outlier flag
                ((df['controversy_score'] < (cont_q1 - 1.5 * cont_iqr))   # Below lower fence
                 | (df['controversy_score'] > (cont_q3 + 1.5 * cont_iqr)))  # OR above upper
                .astype(int)                                      # Convert to 0/1
            )
        else:                                                     # Edge case: IQR is zero
            # Use absolute deviation from median instead
            cont_median = df['controversy_score'].median()        # Calculate median
            df['controversy_iqr_outlier'] = (                     # Create outlier flag
                (df['controversy_score']                          # Check if controversy
                 != cont_median)                                  # differs from median
                .astype(int)                                      # Convert to 0/1
            )

        # ------------------------------------------------------------------
        # Step 2: Median Absolute Deviation (MAD) score
        # ------------------------------------------------------------------
        # MAD is more robust than z-score because it uses median instead of mean
        # Less sensitive to extreme outliers that could skew the mean
        # MAD = median(|x_i - median(x)|)

        # MAD score for ESG risk
        esg_median = df['total_esg_risk_score'].median()          # Population median
        esg_mad = np.median(                                      # Median of absolute deviations
            np.abs(df['total_esg_risk_score'] - esg_median)       # |x - median| for each company
        )

        # MAD score = |x - median| / MAD — similar to z-score but robust
        # Score > 3 = extreme outlier, > 2 = moderate outlier
        df['esg_mad_score'] = (                                   # Create MAD score column
            np.abs(df['total_esg_risk_score'] - esg_median)       # Absolute deviation from median
            / (esg_mad + 1e-8)                                    # Divided by MAD + epsilon
        )

        # MAD score for controversy
        cont_median = df['controversy_score'].median()            # Population median
        cont_mad = np.median(                                     # Median of absolute deviations
            np.abs(df['controversy_score'] - cont_median)         # |x - median| for each company
        )

        df['controversy_mad_score'] = (                           # Create MAD score column
            np.abs(df['controversy_score'] - cont_median)         # Absolute deviation from median
            / (cont_mad + 1e-8)                                   # Divided by MAD + epsilon
        )

        # ------------------------------------------------------------------
        # Step 3: Combined anomaly score
        # ------------------------------------------------------------------
        # Weighted combination of all anomaly signals into a single score
        # Weights reflect domain importance:
        #   - ESG z-score: 25% (standardized distance from mean)
        #   - Controversy z-score: 25% (standardized controversy)
        #   - ESG MAD: 20% (robust distance measure)
        #   - Controversy MAD: 20% (robust controversy measure)
        #   - IQR outliers: 10% (binary extreme flags)
        df['combined_anomaly_score'] = (                          # Create combined score
            0.25 * df['esg_score_zscore'].abs()                   # 25% weight: |ESG z-score|
            + 0.25 * df['controversy_zscore'].abs()               # 25% weight: |controversy z|
            + 0.20 * df['esg_mad_score']                          # 20% weight: ESG MAD score
            + 0.20 * df['controversy_mad_score']                  # 20% weight: controversy MAD
            + 0.10 * (df['esg_iqr_outlier']                      # 10% weight: combined IQR flags
                      + df['controversy_iqr_outlier'])            # (sum of both outlier flags)
        )

        # ------------------------------------------------------------------
        # Step 4: Percentile rank feature
        # ------------------------------------------------------------------
        # Rank each company by ESG score as a percentile (0 to 1)
        # rank(pct=True) assigns percentile ranks — 0.0 = lowest, 1.0 = highest
        df['esg_percentile_rank'] = (                             # Create percentile column
            df['total_esg_risk_score']                            # Take ESG risk score
            .rank(pct=True)                                       # Convert to percentile rank
        )

        # Register features in tracking dictionary
        self.feature_registry['anomaly_detection'] = {            # Store metadata
            'count': 6,                                           # Number of features
            'features': [                                         # Feature name list
                'esg_iqr_outlier', 'controversy_iqr_outlier',
                'esg_mad_score', 'controversy_mad_score',
                'combined_anomaly_score', 'esg_percentile_rank'
            ]
        }

        # Return dataframe with 6 new anomaly detection features
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # SCALING METHODS
    # ========================================================================

    def apply_scaling(self, df, numerical_columns, method='robust'):
        """
        Apply feature scaling to numerical columns using the specified method.

        Rationale:
            Different ML algorithms have different scaling requirements:
            - Tree-based models (XGBoost, Random Forest): scaling NOT needed
            - Linear models (Logistic Regression, SVM): scaling IS critical
            - Neural networks: scaling improves convergence speed

            We support 4 methods and save scaled versions as NEW columns
            (original values preserved for tree-based models).

        Scaling Methods:
            'standard' : StandardScaler — z = (x - mean) / std
                         Best for: normally distributed features
            'minmax'   : MinMaxScaler — z = (x - min) / (max - min)
                         Best for: bounded features, neural networks
            'robust'   : RobustScaler — z = (x - median) / IQR
                         Best for: data with outliers (our default)
            'power'    : PowerTransformer — Yeo-Johnson transform
                         Best for: highly skewed features

        Parameters:
            df                : pd.DataFrame — dataframe with numerical columns
            numerical_columns : list — column names to scale
            method            : str — scaling method ('standard'/'minmax'/'robust'/'power')

        Returns:
            pd.DataFrame — dataframe with new scaled columns (suffix: _scaled)
        """

        # Print scaling method being applied
        print(f"    Applying {method} scaling to {len(numerical_columns)} features...")

        # Filter to only columns that actually exist in the dataframe
        # This prevents KeyError if some columns were not created
        valid_columns = [                                         # Filter valid columns
            col for col in numerical_columns                      # For each requested column
            if col in df.columns                                  # Only if it exists in df
        ]

        # Select the appropriate scaler based on the method parameter
        if method == 'standard':                                  # Standard z-score scaling
            scaler = StandardScaler()                             # Initialize StandardScaler
        elif method == 'minmax':                                  # Min-max [0,1] scaling
            scaler = MinMaxScaler()                               # Initialize MinMaxScaler
        elif method == 'robust':                                  # Robust IQR-based scaling
            scaler = RobustScaler()                               # Initialize RobustScaler
        elif method == 'power':                                   # Yeo-Johnson power transform
            scaler = PowerTransformer(method='yeo-johnson')       # Initialize PowerTransformer
        else:                                                     # Unknown method specified
            raise ValueError(f"Unknown scaling method: {method}") # Raise informative error

        # Fit the scaler on the data and transform in one step
        # fit_transform: learns parameters (mean, std, etc.) AND applies transform
        scaled_values = scaler.fit_transform(                     # Fit and transform
            df[valid_columns].fillna(0)                           # Fill NaN with 0 before scaling
        )

        # Create new column names with '_scaled' suffix
        scaled_column_names = [                                   # Generate new column names
            f"{col}_scaled" for col in valid_columns              # Append '_scaled' to each name
        ]

        # Add scaled columns to the dataframe (originals preserved)
        df[scaled_column_names] = scaled_values                   # Assign scaled values

        # Store the fitted scaler for potential inverse transform later
        self.scalers[method] = {                                  # Save scaler metadata
            'scaler': scaler,                                     # The fitted scaler object
            'columns': valid_columns,                             # Which columns were scaled
            'scaled_columns': scaled_column_names                 # Names of new scaled columns
        }

        # Return dataframe with both original and scaled columns
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # SECTOR-RELATIVE FEATURES
    # ========================================================================

    def create_sector_relative_features(self, df):
        """
        Create features that measure each company relative to its sector peers.

        Rationale:
            A company with ESG risk score = 30 might be:
            - GREAT in the Energy sector (where average is 35)
            - TERRIBLE in the Technology sector (where average is 15)

            Absolute scores are misleading without sector context.
            Sector-relative features normalize performance against relevant peers,
            which is exactly how ESG analysts evaluate companies.

        Features Created:
            - esg_sector_mean          : Average ESG score in this company's sector
            - esg_sector_std           : Standard deviation of ESG in sector
            - esg_vs_sector_mean       : Company score minus sector average
            - esg_sector_zscore        : Z-score within sector
            - controversy_vs_sector    : Company controversy minus sector average
            - sector_rank              : Rank within sector (1 = best)
            - sector_percentile        : Percentile within sector

        Parameters:
            df : pd.DataFrame — must contain 'sector' and ESG score columns

        Returns:
            pd.DataFrame — input dataframe with sector-relative features added
        """

        # Print section header
        print("    [Bonus] Engineering sector-relative features...")  # Status msg

        # Check if sector column exists — if not, skip this step
        if 'sector' not in df.columns:                            # Guard clause
            print("    WARNING: 'sector' column not found, skipping sector features")
            return df                                             # Return unchanged

        # ------------------------------------------------------------------
        # Step 1: Calculate sector-level aggregate statistics
        # ------------------------------------------------------------------
        # Group by sector and calculate mean and std of total ESG risk score
        sector_stats = df.groupby('sector').agg(                  # Group by sector
            esg_sector_mean=('total_esg_risk_score', 'mean'),     # Sector average ESG
            esg_sector_std=('total_esg_risk_score', 'std'),       # Sector ESG spread
            controversy_sector_mean=('controversy_score', 'mean') # Sector avg controversy
        ).reset_index()                                           # Reset index to column

        # Fill NaN std (sectors with only 1 company) with 0
        sector_stats['esg_sector_std'] = (                        # Handle single-company sectors
            sector_stats['esg_sector_std'].fillna(0)              # Replace NaN with 0
        )

        # Cache sector stats for potential later use
        self.sector_stats = sector_stats.copy()                   # Store copy in instance

        # ------------------------------------------------------------------
        # Step 2: Merge sector stats back into company-level dataframe
        # ------------------------------------------------------------------
        # Left join: keep all companies, attach their sector's stats
        df = df.merge(                                            # Merge operation
            sector_stats,                                         # Right table: sector stats
            on='sector',                                          # Join key: sector name
            how='left'                                            # Left join: keep all companies
        )

        # ------------------------------------------------------------------
        # Step 3: Calculate relative performance features
        # ------------------------------------------------------------------
        # How much does this company deviate from its sector average?
        # Positive = worse than sector average (higher risk)
        # Negative = better than sector average (lower risk)
        df['esg_vs_sector_mean'] = (                              # Create deviation column
            df['total_esg_risk_score']                            # Company's ESG score
            - df['esg_sector_mean']                               # Minus sector average
        )

        # Z-score within sector — standardized deviation from sector mean
        # How many sector-standard-deviations is this company from sector average?
        df['esg_sector_zscore'] = (                               # Create sector z-score
            df['esg_vs_sector_mean']                              # Deviation from sector mean
            / (df['esg_sector_std'] + 1e-8)                       # Divided by sector std
        )

        # Controversy relative to sector — same logic for controversy
        df['controversy_vs_sector'] = (                           # Create deviation column
            df['controversy_score']                               # Company's controversy
            - df['controversy_sector_mean']                       # Minus sector average
        )

        # ------------------------------------------------------------------
        # Step 4: Within-sector ranking
        # ------------------------------------------------------------------
        # Rank within sector: 1 = lowest risk (best), N = highest risk (worst)
        df['sector_rank'] = (                                     # Create rank column
            df.groupby('sector')['total_esg_risk_score']          # Group by sector, use ESG
            .rank(method='min')                                   # Assign ranks (min method)
        )

        # Percentile within sector (0 to 1) — normalized rank
        df['sector_percentile'] = (                               # Create percentile column
            df.groupby('sector')['total_esg_risk_score']          # Group by sector, use ESG
            .rank(pct=True)                                       # Percentile rank
        )

        # Register features in tracking dictionary
        self.feature_registry['sector_relative'] = {              # Store metadata
            'count': 7,                                           # Number of features
            'features': [                                         # Feature name list
                'esg_sector_mean', 'esg_sector_std',
                'esg_vs_sector_mean', 'esg_sector_zscore',
                'controversy_vs_sector', 'sector_rank',
                'sector_percentile'
            ]
        }

        # Return dataframe with 7 new sector-relative features
        return df                                                 # Return enriched dataframe

    # ========================================================================
    # MASTER EXECUTION METHOD
    # ========================================================================

    def engineer_all_numerical_features(self, df):
        """
        Execute the COMPLETE numerical feature engineering pipeline.

        This is the main entry point that chains all 5 feature categories
        plus sector-relative features and optional scaling.

        Pipeline Order (dependency-aware):
            1. Pillar Ratios        — creates ratios needed by later steps
            2. Risk Decomposition   — uses ratios from step 1
            3. Statistical Moments  — uses z-scores needed by step 5
            4. Interaction Features — uses features from steps 1-3
            5. Anomaly Detection    — uses z-scores from step 3
            6. Sector-Relative      — independent, uses raw scores
            7. Scaling (optional)   — applied last, after all features exist

        Parameters:
            df : pd.DataFrame — company profiles with ESG scores

        Returns:
            pd.DataFrame — fully feature-engineered dataframe
        """

        # Print pipeline header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  NUMERICAL FEATURE ENGINEERING PIPELINE")         # Pipeline title
        print("=" * 70)                                           # Visual separator

        # Execute each feature engineering step in dependency order
        df = self.create_esg_pillar_ratios(df)                    # Step 1: Pillar ratios
        df = self.create_risk_decomposition_features(df)          # Step 2: Risk decomposition
        df = self.create_statistical_features(df)                 # Step 3: Statistical moments
        df = self.create_interaction_features(df)                 # Step 4: Interactions
        df = self.create_anomaly_features(df)                     # Step 5: Anomaly detection
        df = self.create_sector_relative_features(df)             # Step 6: Sector-relative

        # Collect all engineered numerical feature names for scaling
        all_numerical_features = []                               # Initialize empty list
        for category, info in self.feature_registry.items():      # Loop through categories
            all_numerical_features.extend(info['features'])       # Add features from each

        # Apply robust scaling (our default — handles outliers well)
        df = self.apply_scaling(                                  # Apply scaling step
            df,                                                   # Dataframe to scale
            all_numerical_features,                               # Columns to scale
            method='robust'                                       # Use robust (IQR-based) scaler
        )

        # Print summary report
        total_features = sum(                                     # Count total features
            info['count']                                         # Sum counts from each category
            for info in self.feature_registry.values()            # Iterate all categories
        )
        print(f"\n    TOTAL NUMERICAL FEATURES ENGINEERED: {total_features}")  # Print total
        print(f"    + {len(all_numerical_features)} scaled versions")  # Print scaled count
        for category, info in self.feature_registry.items():      # Print per-category summary
            print(f"      - {category}: {info['count']} features")  # Category name and count
        print("=" * 70)                                           # Visual separator

        # Return the fully engineered dataframe
        return df                                                 # Return final dataframe


# ============================================================================
# STANDALONE EXECUTION (when run as: python feature_engineering_numerical.py)
# ============================================================================

if __name__ == "__main__":                                        # Only run if executed directly

    # Print script header
    print("=" * 70)                                               # Visual separator
    print("  NUMERICAL FEATURE ENGINEERING - STANDALONE TEST")    # Script title
    print("=" * 70)                                               # Visual separator

    # ------------------------------------------------------------------
    # Step 1: Load the company profiles dataset
    # ------------------------------------------------------------------
    # This is the main dataset with ESG scores for 480 companies
    DATA_PATH = "data/processed/company_profiles.csv"             # Path to processed data
    print(f"\n  Loading data from: {DATA_PATH}")                  # Log file path

    df = pd.read_csv(DATA_PATH)                                   # Read CSV into dataframe
    print(f"  Initial shape: {df.shape}")                         # Print initial dimensions
    print(f"  Columns: {list(df.columns)}")                       # Print column names

    # ------------------------------------------------------------------
    # Step 2: Initialize and run the feature engineer
    # ------------------------------------------------------------------
    engineer = NumericalFeatureEngineer()                          # Create instance
    df_engineered = engineer.engineer_all_numerical_features(df)   # Run full pipeline

    # ------------------------------------------------------------------
    # Step 3: Display results
    # ------------------------------------------------------------------
    print(f"\n  Final shape: {df_engineered.shape}")              # Print final dimensions
    print(f"  New columns added: {df_engineered.shape[1] - 13}") # Count new columns

    # Show sample of key engineered features
    key_features = [                                              # Select key features to show
        'company_name',                                           # Company identifier
        'pillar_imbalance_score',                                 # ESG balance indicator
        'controversy_risk_ratio',                                 # Controversy vs risk
        'esg_controversy_divergence',                             # Key GW signal
        'combined_anomaly_score',                                 # Overall anomaly
        'esg_sector_zscore'                                       # Sector-relative position
    ]
    # Filter to only existing columns
    key_features = [c for c in key_features if c in df_engineered.columns]

    print(f"\n  Sample of key features (top 10):")                # Header
    print(df_engineered[key_features].head(10).to_string())       # Print sample

    # ------------------------------------------------------------------
    # Step 4: Save engineered features to disk
    # ------------------------------------------------------------------
    OUTPUT_PATH = "data/processed/numerical_features.csv"         # Output file path
    df_engineered.to_csv(OUTPUT_PATH, index=False)                # Save to CSV
    print(f"\n  Saved engineered features to: {OUTPUT_PATH}")     # Confirm save
    print("=" * 70)                                               # Visual separator
