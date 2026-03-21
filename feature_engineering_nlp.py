"""
================================================================================
FEATURE ENGINEERING - NLP TEXT FEATURES MODULE
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning
Purpose : Extract rich numerical feature vectors from company description text
          using multiple NLP techniques — sentiment, readability, vocabulary
          complexity, ESG keyword density, and linguistic greenwashing signals.

Design Philosophy:
    Text is the PRIMARY evidence of greenwashing. Companies reveal their
    true intentions through language patterns. This module converts raw
    text descriptions into 50+ quantitative features that ML models can
    consume. Every feature is designed with a specific greenwashing
    detection rationale.

Feature Categories:
    1. Sentiment Features       — VADER polarity, subjectivity, confidence
    2. Readability Features     — Flesch, Gunning Fog, syllable complexity
    3. Vocabulary Features      — lexical diversity, word frequencies, n-grams
    4. ESG Keyword Features     — domain-specific term density per pillar
    5. Greenwashing Linguistic   — vagueness, hedging, superlatives, future tense
    6. Document Structure       — sentence count, paragraph length, section balance
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                  # Core data manipulation library
import numpy as np                   # Numerical computing for arrays
import re                            # Regular expressions for pattern matching
from collections import Counter     # Dictionary subclass for counting elements
import warnings                      # Python warnings control
warnings.filterwarnings('ignore')    # Suppress non-critical warnings


# ============================================================================
# CLASS: NLPFeatureEngineer
# ============================================================================

class NLPFeatureEngineer:
    """
    Extract NLP-based numerical features from company description text.

    This class transforms free-text company descriptions into a structured
    feature matrix suitable for ML model training. It implements 6 categories
    of text features specifically designed for greenwashing detection.

    The key insight: greenwashing companies use systematically different
    LANGUAGE PATTERNS compared to genuinely sustainable companies.
    """

    def __init__(self):
        """
        Initialize NLPFeatureEngineer with ESG lexicons and pattern libraries.

        Sets up:
            - ESG keyword dictionaries (environmental, social, governance)
            - Greenwashing signal patterns (vague, hedge, superlative, future)
            - Concrete evidence patterns (numbers, certifications, standards)
            - Feature registry for tracking all created features
        """

        # Feature tracking registry — stores metadata about all generated features
        self.feature_registry = {}                                # Empty dict for feature tracking

        # ==================================================================
        # ESG KEYWORD LEXICONS (domain-specific vocabularies)
        # ==================================================================
        # These word lists are curated from ESG reporting standards (GRI, SASB, TCFD)
        # and academic literature on greenwashing detection

        # Environmental keywords — terms related to environmental performance
        self.env_keywords = [                                     # Environmental lexicon
            'carbon', 'emission', 'emissions', 'climate',        # Carbon/climate terms
            'renewable', 'solar', 'wind', 'energy',              # Energy terms
            'sustainable', 'sustainability', 'green',            # Sustainability terms
            'biodiversity', 'ecosystem', 'conservation',         # Ecology terms
            'pollution', 'waste', 'recycling', 'recycle',        # Pollution/waste terms
            'water', 'deforestation', 'reforestation',           # Resource terms
            'footprint', 'net-zero', 'net zero', 'decarbonize',  # Target terms
            'circular economy', 'clean energy', 'ghg',           # Technical terms
            'scope 1', 'scope 2', 'scope 3', 'paris agreement'  # Framework terms
        ]

        # Social keywords — terms related to social responsibility
        self.social_keywords = [                                  # Social lexicon
            'employee', 'employees', 'workforce', 'worker',     # Workforce terms
            'diversity', 'inclusion', 'equity', 'gender',        # DEI terms
            'safety', 'health', 'wellbeing', 'well-being',      # Health/safety terms
            'community', 'communities', 'stakeholder',           # Community terms
            'human rights', 'labor', 'labour', 'wage',           # Rights terms
            'training', 'education', 'development',              # Development terms
            'supply chain', 'child labor', 'forced labor',       # Supply chain terms
            'privacy', 'data protection', 'accessibility'        # Modern social terms
        ]

        # Governance keywords — terms related to corporate governance
        self.gov_keywords = [                                     # Governance lexicon
            'board', 'director', 'directors', 'governance',      # Board terms
            'compliance', 'regulation', 'regulatory',            # Compliance terms
            'transparency', 'disclosure', 'accountability',      # Transparency terms
            'ethics', 'ethical', 'integrity', 'corruption',      # Ethics terms
            'audit', 'risk management', 'oversight',             # Oversight terms
            'shareholder', 'stakeholder', 'fiduciary',           # Stakeholder terms
            'independent', 'compensation', 'executive pay',      # Structure terms
            'whistleblower', 'anti-bribery', 'tax'               # Policy terms
        ]

        # ==================================================================
        # GREENWASHING LINGUISTIC PATTERNS
        # ==================================================================
        # Based on academic research: Lyon & Montgomery (2015), Delmas & Burbano (2011)

        # Vague/ambiguous language — sounds impressive but lacks specifics
        self.vague_patterns = [                                   # Vagueness lexicon
            'committed to', 'dedicated to', 'striving for',     # Commitment without proof
            'working towards', 'moving towards', 'journey',     # Process without endpoint
            'mindful', 'conscious', 'aware', 'exploring',       # Awareness without action
            'eco-friendly', 'environmentally friendly',          # Undefined friendliness
            'planet-positive', 'doing our part',                 # Unmeasurable claims
            'making a difference', 'better future',              # Aspirational fluff
            'responsible', 'responsibly', 'thoughtful',          # Vague responsibility
            'holistic approach', 'comprehensive strategy'        # Abstract strategy
        ]

        # Hedging language — weakens commitments with qualifiers
        self.hedge_patterns = [                                   # Hedging lexicon
            'may', 'might', 'could', 'potentially',             # Possibility hedges
            'where feasible', 'where possible', 'where practical',  # Conditional hedges
            'aim to', 'aspire to', 'hope to', 'intend to',     # Intent hedges
            'seek to', 'endeavor', 'strive', 'attempt',         # Effort hedges
            'approximately', 'roughly', 'about', 'around',      # Precision hedges
            'subject to', 'contingent upon', 'depending on'     # Conditional hedges
        ]

        # Superlative/exaggerated language — overclaims performance
        self.superlative_patterns = [                             # Superlative lexicon
            'world-class', 'best-in-class', 'industry-leading', # Superlative claims
            'pioneering', 'revolutionary', 'transformative',     # Innovation overclaims
            'unparalleled', 'unmatched', 'unprecedented',        # Uniqueness claims
            'cutting-edge', 'state-of-the-art', 'leading',      # Technology claims
            'first-of-its-kind', 'groundbreaking', 'trailblazing'  # Priority claims
        ]

        # Future tense / promise language — commitments without current proof
        self.future_patterns = [                                  # Future tense lexicon
            'will', 'by 2030', 'by 2040', 'by 2050',            # Dated promises
            'plan to', 'planning to', 'going to',               # Future plans
            'target', 'goal', 'ambition', 'pledge',             # Target language
            'commitment', 'promise', 'roadmap', 'vision',       # Vision language
            'upcoming', 'forthcoming', 'expected to'             # Future orientation
        ]

        # Concrete evidence language — verifiable, specific claims
        self.concrete_patterns = [                                # Concrete evidence lexicon
            'reduced by', 'increased by', 'decreased by',       # Quantified changes
            'percent', '%', 'tonnes', 'tons', 'kwh',            # Units of measurement
            'iso 14001', 'iso 26000', 'gri', 'sasb', 'tcfd',   # Standards/frameworks
            'verified', 'certified', 'audited', 'third-party',  # Verification terms
            'measured', 'reported', 'disclosed', 'published',   # Reporting terms
            'baseline', 'benchmark', 'metric', 'kpi'            # Measurement terms
        ]

    # ========================================================================
    # CATEGORY 1: SENTIMENT FEATURES
    # ========================================================================

    def extract_sentiment_features(self, df, text_column='description'):
        """
        Extract sentiment polarity and related features from text.

        Uses a simplified VADER-like approach with ESG-domain word scoring.
        Measures: overall polarity, positivity ratio, negativity ratio,
        and sentiment strength.

        Rationale:
            Greenwashing text tends to be OVERLY POSITIVE — more positive
            than genuine sustainability reports which also discuss challenges.

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — name of the column with text data

        Returns:
            pd.DataFrame — with sentiment feature columns added
        """

        # Print section header
        print("    [1/6] Extracting sentiment features from text...")  # Status

        # Define positive and negative word lists for ESG context
        positive_words = {                                        # Set of positive words
            'good', 'great', 'excellent', 'outstanding', 'strong',     # General positive
            'improved', 'growth', 'success', 'innovative', 'leading',  # Performance positive
            'sustainable', 'responsible', 'committed', 'achievement',  # ESG positive
            'progress', 'advanced', 'efficient', 'clean', 'safe',     # Improvement positive
            'reduced', 'renewable', 'protection', 'diverse', 'ethical' # Action positive
        }

        negative_words = {                                        # Set of negative words
            'risk', 'loss', 'decline', 'fail', 'poor',               # General negative
            'violation', 'penalty', 'fine', 'lawsuit', 'scandal',     # Legal negative
            'pollution', 'contamination', 'spill', 'accident',        # Environmental negative
            'unsafe', 'hazardous', 'toxic', 'controversy',            # Safety negative
            'corruption', 'fraud', 'breach', 'negligence', 'harm'     # Governance negative
        }

        # Initialize result lists for each sentiment feature
        polarity_scores = []                                      # Overall sentiment score
        positive_ratios = []                                      # Fraction of positive words
        negative_ratios = []                                      # Fraction of negative words
        sentiment_strengths = []                                  # Absolute sentiment magnitude

        # Process each company's text description row by row
        for idx, row in df.iterrows():                            # Iterate through dataframe rows
            text = str(row.get(text_column, ''))                  # Get text, default to empty
            text_lower = text.lower()                             # Convert to lowercase
            words = re.findall(r'\b[a-z]+\b', text_lower)        # Extract all words (letters only)

            if len(words) == 0:                                   # Guard: empty text
                polarity_scores.append(0.0)                       # Neutral polarity
                positive_ratios.append(0.0)                       # No positive words
                negative_ratios.append(0.0)                       # No negative words
                sentiment_strengths.append(0.0)                   # No sentiment
                continue                                          # Skip to next row

            # Count positive and negative word occurrences
            pos_count = sum(1 for w in words if w in positive_words)  # Count positives
            neg_count = sum(1 for w in words if w in negative_words)  # Count negatives
            total_words = len(words)                              # Total word count

            # Calculate polarity: (positive - negative) / total words
            # Range: -1 (all negative) to +1 (all positive)
            polarity = (pos_count - neg_count) / total_words      # Net sentiment per word
            polarity_scores.append(polarity)                      # Store polarity

            # Positive ratio: fraction of words that are positive
            positive_ratios.append(pos_count / total_words)       # Store positive ratio

            # Negative ratio: fraction of words that are negative
            negative_ratios.append(neg_count / total_words)       # Store negative ratio

            # Sentiment strength: absolute value of polarity (intensity regardless of direction)
            sentiment_strengths.append(abs(polarity))             # Store strength

        # Assign computed features to dataframe columns
        df['text_polarity'] = polarity_scores                     # Overall sentiment direction
        df['text_positive_ratio'] = positive_ratios               # Fraction of positive words
        df['text_negative_ratio'] = negative_ratios               # Fraction of negative words
        df['text_sentiment_strength'] = sentiment_strengths       # Sentiment intensity

        # Derived feature: positive-to-negative ratio
        # Greenwashing text often has very high pos/neg ratio (overly positive)
        df['text_pos_neg_ratio'] = (                              # Create ratio column
            (df['text_positive_ratio'] + 1e-8)                    # Positive ratio + epsilon
            / (df['text_negative_ratio'] + 1e-8)                  # Divided by negative + epsilon
        )

        # Register created features
        self.feature_registry['sentiment'] = {                    # Store metadata
            'count': 5,                                           # Number of features
            'features': ['text_polarity', 'text_positive_ratio',  # Feature list
                        'text_negative_ratio', 'text_sentiment_strength',
                        'text_pos_neg_ratio']
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 2: READABILITY FEATURES
    # ========================================================================

    def extract_readability_features(self, df, text_column='description'):
        """
        Calculate text readability and complexity metrics.

        Rationale:
            Research shows greenwashing text is often:
            - MORE complex (Gunning Fog Index) — obscures with jargon
            - LONGER sentences — buries information in complex structures
            - HIGHER syllable count — uses fancy words instead of clear ones

            Genuinely sustainable companies tend to write clearer, more
            direct reports because they have nothing to hide.

        Features Created:
            - avg_word_length           : Average characters per word
            - avg_sentence_length       : Average words per sentence
            - syllable_ratio            : Estimated complex word fraction
            - flesch_reading_ease       : Flesch readability (higher = easier)
            - gunning_fog_index         : Gunning Fog (higher = harder to read)
            - long_word_ratio           : Fraction of words with 6+ characters

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — column name with text data

        Returns:
            pd.DataFrame — with readability features added
        """

        # Print section header
        print("    [2/6] Extracting readability features...")      # Status message

        # Initialize lists to store feature values for each company
        avg_word_lengths = []                                     # Average word length (chars)
        avg_sentence_lengths = []                                 # Average sentence length (words)
        syllable_ratios = []                                      # Complex word fraction
        flesch_scores = []                                        # Flesch readability score
        fog_indices = []                                          # Gunning Fog index
        long_word_ratios = []                                     # Fraction of long words

        # Process each company's description text
        for idx, row in df.iterrows():                            # Iterate rows
            text = str(row.get(text_column, ''))                  # Get text safely
            words = re.findall(r'\b[a-z]+\b', text.lower())      # Extract lowercase words

            # Split text into sentences using common sentence delimiters
            sentences = re.split(r'[.!?]+', text)                 # Split on . ! ?
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]  # Filter empties

            if len(words) == 0:                                   # Guard: no words found
                avg_word_lengths.append(0.0)                      # Zero for all metrics
                avg_sentence_lengths.append(0.0)                  # Zero
                syllable_ratios.append(0.0)                       # Zero
                flesch_scores.append(0.0)                         # Zero
                fog_indices.append(0.0)                           # Zero
                long_word_ratios.append(0.0)                      # Zero
                continue                                          # Skip to next row

            # ------------------------------------------------------------------
            # Metric 1: Average word length (characters per word)
            # ------------------------------------------------------------------
            word_lengths = [len(w) for w in words]                # Length of each word
            avg_wl = np.mean(word_lengths)                        # Average word length
            avg_word_lengths.append(avg_wl)                       # Store result

            # ------------------------------------------------------------------
            # Metric 2: Average sentence length (words per sentence)
            # ------------------------------------------------------------------
            num_sentences = max(len(sentences), 1)                # At least 1 sentence
            avg_sl = len(words) / num_sentences                   # Words divided by sentences
            avg_sentence_lengths.append(avg_sl)                   # Store result

            # ------------------------------------------------------------------
            # Metric 3: Syllable ratio (complex word estimation)
            # ------------------------------------------------------------------
            # Approximate syllable count: count vowel groups in each word
            # Words with 3+ syllables are considered "complex" (per Gunning Fog)
            complex_words = 0                                     # Counter for complex words
            for w in words:                                       # Loop through each word
                # Count vowel groups (consecutive vowels = 1 syllable)
                syllables = len(re.findall(r'[aeiouy]+', w))      # Count vowel groups
                if syllables >= 3:                                 # 3+ syllables = complex
                    complex_words += 1                            # Increment counter

            syllable_ratio = complex_words / len(words)           # Fraction of complex words
            syllable_ratios.append(syllable_ratio)                # Store result

            # ------------------------------------------------------------------
            # Metric 4: Flesch Reading Ease Score
            # ------------------------------------------------------------------
            # Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
            # Higher = easier to read (60-70 is standard, <30 = very difficult)
            total_syllables = sum(                                # Sum syllables across all words
                max(len(re.findall(r'[aeiouy]+', w)), 1)          # At least 1 syllable per word
                for w in words                                    # For each word
            )
            syllables_per_word = total_syllables / len(words)     # Average syllables per word
            flesch = (                                            # Calculate Flesch score
                206.835                                           # Base constant
                - 1.015 * avg_sl                                  # Subtract sentence length penalty
                - 84.6 * syllables_per_word                       # Subtract syllable penalty
            )
            flesch_scores.append(flesch)                           # Store result

            # ------------------------------------------------------------------
            # Metric 5: Gunning Fog Index
            # ------------------------------------------------------------------
            # Formula: 0.4 * (avg_sentence_length + 100 * complex_word_ratio)
            # Estimates years of education needed to understand the text
            # > 12 = college level, > 17 = graduate level (red flag for public docs)
            fog = 0.4 * (avg_sl + 100 * syllable_ratio)           # Calculate Fog index
            fog_indices.append(fog)                               # Store result

            # ------------------------------------------------------------------
            # Metric 6: Long word ratio
            # ------------------------------------------------------------------
            # Fraction of words with 6+ characters — another complexity proxy
            long_words = sum(1 for w in words if len(w) >= 6)     # Count long words
            long_word_ratios.append(long_words / len(words))      # Store ratio

        # Assign all feature lists to dataframe columns
        df['avg_word_length'] = avg_word_lengths                  # Average word length
        df['avg_sentence_length'] = avg_sentence_lengths          # Average sentence length
        df['syllable_ratio'] = syllable_ratios                    # Complex word fraction
        df['flesch_reading_ease'] = flesch_scores                 # Flesch readability
        df['gunning_fog_index'] = fog_indices                     # Gunning Fog index
        df['long_word_ratio'] = long_word_ratios                  # Long word fraction

        # Register features
        self.feature_registry['readability'] = {                  # Store metadata
            'count': 6,                                           # Number of features
            'features': ['avg_word_length', 'avg_sentence_length',  # List
                        'syllable_ratio', 'flesch_reading_ease',
                        'gunning_fog_index', 'long_word_ratio']
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 3: VOCABULARY RICHNESS FEATURES
    # ========================================================================

    def extract_vocabulary_features(self, df, text_column='description'):
        """
        Measure vocabulary richness, diversity, and lexical patterns.

        Rationale:
            Greenwashing text often shows:
            - LOW lexical diversity — repeats same buzzwords
            - HIGH hapax legomena ratio — uses rare words once for impression
            - SPECIFIC word frequency patterns — overuses certain ESG terms

        Features Created:
            - total_word_count          : Total number of words
            - unique_word_count         : Number of distinct words
            - lexical_diversity         : unique / total (type-token ratio)
            - hapax_legomena_ratio      : Words appearing exactly once / total
            - top_word_concentration    : Fraction of text from top 10 words
            - avg_word_frequency        : Average times each word appears

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — column name with text data

        Returns:
            pd.DataFrame — with vocabulary richness features added
        """

        # Print section header
        print("    [3/6] Extracting vocabulary richness features...")  # Status

        # Initialize feature storage lists
        total_words_list = []                                     # Total word counts
        unique_words_list = []                                    # Unique word counts
        lexical_diversities = []                                  # Type-token ratios
        hapax_ratios = []                                         # Single-occurrence word ratios
        top_word_concentrations = []                              # Top-10 word dominance
        avg_frequencies = []                                      # Average word frequency

        # Process each row's text
        for idx, row in df.iterrows():                            # Iterate rows
            text = str(row.get(text_column, ''))                  # Get text safely
            words = re.findall(r'\b[a-z]+\b', text.lower())      # Extract words

            if len(words) == 0:                                   # Guard: empty text
                total_words_list.append(0)                        # Zero for all metrics
                unique_words_list.append(0)                       # Zero
                lexical_diversities.append(0.0)                   # Zero
                hapax_ratios.append(0.0)                          # Zero
                top_word_concentrations.append(0.0)               # Zero
                avg_frequencies.append(0.0)                       # Zero
                continue                                          # Skip

            # Count word frequencies using Counter (efficient counting)
            word_freq = Counter(words)                            # {word: count, ...}
            total = len(words)                                    # Total word count
            unique = len(word_freq)                               # Unique word count

            total_words_list.append(total)                        # Store total count
            unique_words_list.append(unique)                      # Store unique count

            # Lexical diversity (Type-Token Ratio) = unique / total
            # High TTR = diverse vocabulary, Low TTR = repetitive
            lexical_diversities.append(unique / total)            # Store TTR

            # Hapax legomena = words that appear exactly once
            # High hapax ratio suggests varied language (or jargon-heavy text)
            hapax_count = sum(                                    # Count words appearing once
                1 for word, count in word_freq.items()            # For each word
                if count == 1                                     # If it appears exactly once
            )
            hapax_ratios.append(hapax_count / total)              # Store hapax ratio

            # Top-10 word concentration = fraction of text from 10 most common words
            # High concentration = text relies heavily on a few key terms
            top_10 = word_freq.most_common(10)                    # Get 10 most frequent words
            top_10_count = sum(c for _, c in top_10)              # Sum their frequencies
            top_word_concentrations.append(top_10_count / total)  # Store concentration

            # Average word frequency = mean occurrence count per unique word
            avg_freq = total / unique                             # Total / unique words
            avg_frequencies.append(avg_freq)                      # Store average

        # Assign features to dataframe columns
        df['total_word_count'] = total_words_list                 # Total words in description
        df['unique_word_count'] = unique_words_list               # Unique words
        df['lexical_diversity'] = lexical_diversities             # Type-token ratio
        df['hapax_legomena_ratio'] = hapax_ratios                 # Single-occurrence word ratio
        df['top_word_concentration'] = top_word_concentrations    # Top-10 dominance
        df['avg_word_frequency'] = avg_frequencies                # Average frequency

        # Register features
        self.feature_registry['vocabulary'] = {                   # Store metadata
            'count': 6,                                           # Feature count
            'features': ['total_word_count', 'unique_word_count',  # List
                        'lexical_diversity', 'hapax_legomena_ratio',
                        'top_word_concentration', 'avg_word_frequency']
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 4: ESG KEYWORD DENSITY FEATURES
    # ========================================================================

    def extract_esg_keyword_features(self, df, text_column='description'):
        """
        Calculate ESG-domain keyword density for each pillar (E, S, G).

        Rationale:
            Keyword density reveals:
            - Which ESG pillar a company emphasizes in its communication
            - Whether keyword density matches actual ESG performance
            - Potential "keyword stuffing" (high keyword density but low scores)

            A company with HIGH environmental keyword density but LOW env score
            is a strong greenwashing signal — they TALK green but don't ACT green.

        Features Created:
            - env_keyword_count          : Count of environmental keywords
            - env_keyword_density        : Env keywords / total words
            - social_keyword_count       : Count of social keywords
            - social_keyword_density     : Social keywords / total words
            - gov_keyword_count          : Count of governance keywords
            - gov_keyword_density        : Gov keywords / total words
            - total_esg_keyword_count    : Combined E+S+G keyword count
            - total_esg_keyword_density  : Combined keywords / total words
            - esg_keyword_balance        : Std dev of pillar keyword densities
            - dominant_keyword_pillar    : Which pillar has most keywords (encoded)

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — column name with text data

        Returns:
            pd.DataFrame — with ESG keyword features added
        """

        # Print section header
        print("    [4/6] Extracting ESG keyword density features...")  # Status

        # Initialize storage lists for all 10 features
        env_counts = []                                           # Environmental keyword counts
        env_densities = []                                        # Environmental keyword density
        social_counts = []                                        # Social keyword counts
        social_densities = []                                     # Social keyword density
        gov_counts = []                                           # Governance keyword counts
        gov_densities = []                                        # Governance keyword density
        total_esg_counts = []                                     # Combined ESG counts
        total_esg_densities = []                                  # Combined ESG density
        esg_balances = []                                         # Balance across pillars
        dominant_pillars = []                                     # Which pillar dominates

        # Process each row
        for idx, row in df.iterrows():                            # Iterate rows
            text = str(row.get(text_column, '')).lower()          # Get text in lowercase
            words = re.findall(r'\b[a-z]+\b', text)              # Extract words
            total = max(len(words), 1)                            # Total words (min 1)

            # ------------------------------------------------------------------
            # Count keyword occurrences for each ESG pillar
            # ------------------------------------------------------------------
            # Environmental keywords: count how many times each env term appears
            env_count = sum(                                      # Sum all env keyword matches
                text.count(keyword)                               # Count occurrences of keyword
                for keyword in self.env_keywords                  # For each environmental keyword
            )

            # Social keywords: count social term occurrences
            social_count = sum(                                   # Sum all social keyword matches
                text.count(keyword)                               # Count occurrences of keyword
                for keyword in self.social_keywords               # For each social keyword
            )

            # Governance keywords: count governance term occurrences
            gov_count = sum(                                      # Sum all gov keyword matches
                text.count(keyword)                               # Count occurrences of keyword
                for keyword in self.gov_keywords                  # For each governance keyword
            )

            # ------------------------------------------------------------------
            # Calculate densities (normalized by text length)
            # ------------------------------------------------------------------
            env_density = env_count / total                       # Env keywords per word
            social_density = social_count / total                 # Social keywords per word
            gov_density = gov_count / total                       # Gov keywords per word
            total_esg = env_count + social_count + gov_count      # Combined keyword count
            total_density = total_esg / total                     # Combined density

            # ------------------------------------------------------------------
            # Calculate cross-pillar balance and dominance
            # ------------------------------------------------------------------
            # Keyword balance = std deviation across 3 pillar densities
            # High std = company talks about one pillar much more than others
            densities = [env_density, social_density, gov_density] # Three pillar densities
            balance = np.std(densities)                           # Std dev (imbalance metric)

            # Determine dominant pillar (which pillar has highest keyword density)
            # Encoded as: 0=Environmental, 1=Social, 2=Governance
            pillar_idx = np.argmax(densities)                     # Index of maximum density

            # Store all computed values
            env_counts.append(env_count)                          # Store env count
            env_densities.append(env_density)                     # Store env density
            social_counts.append(social_count)                    # Store social count
            social_densities.append(social_density)               # Store social density
            gov_counts.append(gov_count)                          # Store gov count
            gov_densities.append(gov_density)                     # Store gov density
            total_esg_counts.append(total_esg)                    # Store total count
            total_esg_densities.append(total_density)             # Store total density
            esg_balances.append(balance)                          # Store balance score
            dominant_pillars.append(pillar_idx)                   # Store dominant pillar

        # Assign all features to dataframe columns
        df['env_keyword_count'] = env_counts                      # Environmental keyword count
        df['env_keyword_density'] = env_densities                 # Environmental keyword density
        df['social_keyword_count'] = social_counts                # Social keyword count
        df['social_keyword_density'] = social_densities           # Social keyword density
        df['gov_keyword_count'] = gov_counts                      # Governance keyword count
        df['gov_keyword_density'] = gov_densities                 # Governance keyword density
        df['total_esg_keyword_count'] = total_esg_counts          # Combined ESG keyword count
        df['total_esg_keyword_density'] = total_esg_densities     # Combined ESG density
        df['esg_keyword_balance'] = esg_balances                  # Pillar keyword balance
        df['dominant_keyword_pillar'] = dominant_pillars           # Dominant pillar (0/1/2)

        # Register features
        self.feature_registry['esg_keywords'] = {                 # Store metadata
            'count': 10,                                          # Feature count
            'features': [                                         # Feature list
                'env_keyword_count', 'env_keyword_density',
                'social_keyword_count', 'social_keyword_density',
                'gov_keyword_count', 'gov_keyword_density',
                'total_esg_keyword_count', 'total_esg_keyword_density',
                'esg_keyword_balance', 'dominant_keyword_pillar'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 5: GREENWASHING LINGUISTIC SIGNAL FEATURES
    # ========================================================================

    def extract_greenwashing_linguistic_features(self, df, text_column='description'):
        """
        Detect specific linguistic patterns associated with greenwashing.

        This is the MOST IMPORTANT feature category for our task.
        Based on peer-reviewed research on corporate greenwashing linguistics.

        Rationale:
            5 linguistic dimensions separate greenwashing from genuine ESG:
            1. Vagueness  — lacks specific details or measurements
            2. Hedging    — qualifies commitments with "may", "could", etc.
            3. Superlatives — exaggerated claims without evidence
            4. Future focus — promises without current performance proof
            5. Concreteness — verifiable data, standards, certifications

            Formula: GW_signal = (vague + hedge + superlative + future) - 2*concrete

        Features Created:
            - vague_language_count       : Count of vague/ambiguous phrases
            - vague_language_density     : Vague phrases per word
            - hedge_language_count       : Count of hedging phrases
            - hedge_language_density     : Hedging phrases per word
            - superlative_count          : Count of superlative/exaggerated phrases
            - superlative_density        : Superlatives per word
            - future_language_count      : Count of future-tense/promise phrases
            - future_language_density    : Future phrases per word
            - concrete_evidence_count    : Count of verifiable evidence markers
            - concrete_evidence_density  : Evidence markers per word
            - greenwashing_signal_score  : Combined GW linguistic score (0-1)
            - vague_to_concrete_ratio    : Ratio of vague to concrete language

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — column name with text data

        Returns:
            pd.DataFrame — with greenwashing linguistic features added
        """

        # Print section header
        print("    [5/6] Extracting greenwashing linguistic signals...")  # Status

        # Initialize storage for all 12 features
        vague_counts = []                                         # Vague language counts
        vague_densities = []                                      # Vague language density
        hedge_counts = []                                         # Hedge language counts
        hedge_densities = []                                      # Hedge language density
        superlative_counts = []                                   # Superlative counts
        superlative_densities = []                                # Superlative density
        future_counts = []                                        # Future language counts
        future_densities = []                                     # Future language density
        concrete_counts = []                                      # Concrete evidence counts
        concrete_densities = []                                   # Concrete evidence density
        gw_signal_scores = []                                     # Combined GW signal
        vague_concrete_ratios = []                                # Vague-to-concrete ratio

        # Process each company's text
        for idx, row in df.iterrows():                            # Iterate rows
            text = str(row.get(text_column, '')).lower()          # Get lowercase text
            words = re.findall(r'\b[a-z]+\b', text)              # Extract words
            total = max(len(words), 1)                            # Total words (min 1)

            # Count occurrences of each pattern category
            # Each pattern is a phrase — count how many times it appears in text
            vague = sum(text.count(p) for p in self.vague_patterns)          # Vague count
            hedge = sum(text.count(p) for p in self.hedge_patterns)          # Hedge count
            superlative = sum(text.count(p) for p in self.superlative_patterns)  # Superlative count
            future = sum(text.count(p) for p in self.future_patterns)        # Future count
            concrete = sum(text.count(p) for p in self.concrete_patterns)    # Concrete count

            # Calculate densities (normalized by word count)
            vague_d = vague / total                               # Vague density
            hedge_d = hedge / total                               # Hedge density
            superlative_d = superlative / total                   # Superlative density
            future_d = future / total                             # Future density
            concrete_d = concrete / total                         # Concrete density

            # ------------------------------------------------------------------
            # GREENWASHING SIGNAL SCORE (core formula)
            # ------------------------------------------------------------------
            # Positive signals (increase GW score):
            #   - Vague language (weight 1.0)
            #   - Hedging language (weight 1.0)
            #   - Superlative language (weight 1.0)
            #   - Future promises (weight 1.0)
            # Negative signals (decrease GW score):
            #   - Concrete evidence (weight -2.0, double penalty)
            #
            # Raw signal = sum of positive - 2 * concrete
            # Then normalized to [0, 1] using sigmoid-like transformation
            raw_signal = (                                        # Raw GW signal
                vague_d + hedge_d + superlative_d + future_d      # Sum of GW signals
                - 2.0 * concrete_d                                # Minus double-weighted evidence
            )
            # Normalize using sigmoid: 1 / (1 + exp(-10*x))
            # Maps any real number to [0, 1] range
            # Factor of 10 controls steepness of the sigmoid curve
            gw_score = 1.0 / (1.0 + np.exp(-10 * raw_signal))    # Sigmoid normalization

            # Vague-to-concrete ratio: how much vague language per unit of evidence
            # High ratio = mostly vague, Low ratio = mostly concrete
            vtc_ratio = (vague + hedge + 1e-8) / (concrete + 1e-8)  # Vague/concrete ratio

            # Store all values
            vague_counts.append(vague)                            # Store vague count
            vague_densities.append(vague_d)                       # Store vague density
            hedge_counts.append(hedge)                            # Store hedge count
            hedge_densities.append(hedge_d)                       # Store hedge density
            superlative_counts.append(superlative)                # Store superlative count
            superlative_densities.append(superlative_d)           # Store superlative density
            future_counts.append(future)                          # Store future count
            future_densities.append(future_d)                     # Store future density
            concrete_counts.append(concrete)                      # Store concrete count
            concrete_densities.append(concrete_d)                 # Store concrete density
            gw_signal_scores.append(gw_score)                     # Store GW signal score
            vague_concrete_ratios.append(vtc_ratio)               # Store V/C ratio

        # Assign features to dataframe
        df['vague_language_count'] = vague_counts                 # Vague phrase count
        df['vague_language_density'] = vague_densities            # Vague phrases per word
        df['hedge_language_count'] = hedge_counts                 # Hedge phrase count
        df['hedge_language_density'] = hedge_densities            # Hedge phrases per word
        df['superlative_count'] = superlative_counts              # Superlative phrase count
        df['superlative_density'] = superlative_densities         # Superlatives per word
        df['future_language_count'] = future_counts               # Future phrase count
        df['future_language_density'] = future_densities          # Future phrases per word
        df['concrete_evidence_count'] = concrete_counts           # Concrete evidence count
        df['concrete_evidence_density'] = concrete_densities      # Evidence per word
        df['greenwashing_signal_score'] = gw_signal_scores        # Combined GW score (0-1)
        df['vague_to_concrete_ratio'] = vague_concrete_ratios     # Vague/concrete ratio

        # Register features
        self.feature_registry['greenwashing_linguistic'] = {      # Store metadata
            'count': 12,                                          # Feature count
            'features': [                                         # Feature list
                'vague_language_count', 'vague_language_density',
                'hedge_language_count', 'hedge_language_density',
                'superlative_count', 'superlative_density',
                'future_language_count', 'future_language_density',
                'concrete_evidence_count', 'concrete_evidence_density',
                'greenwashing_signal_score', 'vague_to_concrete_ratio'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # CATEGORY 6: DOCUMENT STRUCTURE FEATURES
    # ========================================================================

    def extract_document_structure_features(self, df, text_column='description'):
        """
        Analyze document structure — sentence patterns, paragraph structure,
        section balance, and information density.

        Rationale:
            Document structure reveals writing strategy:
            - Short, punchy sentences = marketing copy (potential greenwashing)
            - Long, complex sentences = legal/technical writing
            - Sentence length variance = mixed writing style (possible inconsistency)
            - Information density = how much substance per unit of text

        Features Created:
            - sentence_count            : Total number of sentences
            - sentence_length_variance  : Variance in sentence lengths
            - short_sentence_ratio      : Fraction of sentences < 10 words
            - long_sentence_ratio       : Fraction of sentences > 30 words
            - question_mark_count       : Number of rhetorical questions
            - exclamation_count         : Number of exclamations (marketing signal)
            - number_density            : Fraction of tokens that are numbers
            - capitalized_word_ratio    : Fraction of words that start uppercase

        Parameters:
            df          : pd.DataFrame — contains text descriptions
            text_column : str — column name with text data

        Returns:
            pd.DataFrame — with document structure features added
        """

        # Print section header
        print("    [6/6] Extracting document structure features...")  # Status

        # Initialize storage lists
        sentence_counts = []                                      # Sentence count per doc
        sent_len_variances = []                                   # Sentence length variance
        short_sent_ratios = []                                    # Short sentence fraction
        long_sent_ratios = []                                     # Long sentence fraction
        question_counts = []                                      # Question mark count
        exclamation_counts = []                                   # Exclamation mark count
        number_densities = []                                     # Numeric token fraction
        cap_word_ratios = []                                      # Capitalized word fraction

        # Process each row
        for idx, row in df.iterrows():                            # Iterate rows
            text = str(row.get(text_column, ''))                  # Get text safely

            # Split into sentences
            sentences = re.split(r'[.!?]+', text)                 # Split on sentence endings
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]  # Filter empty

            num_sentences = max(len(sentences), 1)                # At least 1 sentence
            sentence_counts.append(num_sentences)                 # Store sentence count

            # Calculate sentence lengths (words per sentence)
            sent_lengths = []                                     # List of sentence lengths
            for sent in sentences:                                # For each sentence
                words_in_sent = len(sent.split())                 # Count words in sentence
                sent_lengths.append(words_in_sent)                # Store length

            if len(sent_lengths) == 0:                            # Guard: no sentences
                sent_lengths = [0]                                # Default to [0]

            # Sentence length variance — high variance = inconsistent writing style
            sent_len_variances.append(np.var(sent_lengths))       # Variance of lengths

            # Short sentence ratio — sentences with fewer than 10 words
            # Marketing and greenwashing text often uses short punchy sentences
            short = sum(1 for l in sent_lengths if l < 10)        # Count short sentences
            short_sent_ratios.append(short / num_sentences)       # Store ratio

            # Long sentence ratio — sentences with more than 30 words
            # Legal/compliance text tends to have very long sentences
            long = sum(1 for l in sent_lengths if l > 30)         # Count long sentences
            long_sent_ratios.append(long / num_sentences)         # Store ratio

            # Question mark count — rhetorical questions are a persuasion technique
            question_counts.append(text.count('?'))               # Count question marks

            # Exclamation count — exclamations suggest marketing/promotional tone
            exclamation_counts.append(text.count('!'))            # Count exclamation marks

            # Number density — fraction of tokens that are numeric
            # Higher number density = more quantitative (more concrete evidence)
            all_tokens = text.split()                             # Split into all tokens
            if len(all_tokens) > 0:                               # Guard: non-empty
                num_count = sum(                                  # Count numeric tokens
                    1 for t in all_tokens                         # For each token
                    if re.search(r'\d', t)                        # If it contains a digit
                )
                number_densities.append(num_count / len(all_tokens))  # Store density
            else:                                                 # Empty text
                number_densities.append(0.0)                      # Zero density

            # Capitalized word ratio — many capitals = proper nouns or emphasis
            if len(all_tokens) > 0:                               # Guard: non-empty
                cap_words = sum(                                  # Count capitalized words
                    1 for t in all_tokens                         # For each token
                    if t and t[0].isupper()                       # If first char is uppercase
                )
                cap_word_ratios.append(cap_words / len(all_tokens))  # Store ratio
            else:                                                 # Empty text
                cap_word_ratios.append(0.0)                       # Zero ratio

        # Assign features to dataframe
        df['sentence_count'] = sentence_counts                    # Total sentences
        df['sentence_length_variance'] = sent_len_variances       # Sentence length variability
        df['short_sentence_ratio'] = short_sent_ratios            # Short sentence fraction
        df['long_sentence_ratio'] = long_sent_ratios              # Long sentence fraction
        df['question_mark_count'] = question_counts               # Rhetorical questions
        df['exclamation_count'] = exclamation_counts              # Exclamation marks
        df['number_density'] = number_densities                   # Numeric token density
        df['capitalized_word_ratio'] = cap_word_ratios            # Capitalized word fraction

        # Register features
        self.feature_registry['document_structure'] = {           # Store metadata
            'count': 8,                                           # Feature count
            'features': [                                         # Feature list
                'sentence_count', 'sentence_length_variance',
                'short_sentence_ratio', 'long_sentence_ratio',
                'question_mark_count', 'exclamation_count',
                'number_density', 'capitalized_word_ratio'
            ]
        }

        return df                                                 # Return enriched dataframe

    # ========================================================================
    # MASTER EXECUTION METHOD
    # ========================================================================

    def engineer_all_nlp_features(self, df, text_column='description'):
        """
        Execute the complete NLP feature engineering pipeline.

        Chains all 6 NLP feature categories in sequence to extract
        a comprehensive text-based feature matrix from company descriptions.

        Pipeline Order:
            1. Sentiment Features       → 5 features
            2. Readability Features     → 6 features
            3. Vocabulary Features      → 6 features
            4. ESG Keyword Features     → 10 features
            5. Greenwashing Linguistic  → 12 features (MOST IMPORTANT)
            6. Document Structure       → 8 features

        Parameters:
            df          : pd.DataFrame — company data with text descriptions
            text_column : str — column containing text (default: 'description')

        Returns:
            pd.DataFrame — with all 47 NLP features added
        """

        # Print pipeline header
        print("\n" + "=" * 70)                                    # Visual separator
        print("  NLP FEATURE ENGINEERING PIPELINE")               # Pipeline title
        print("=" * 70)                                           # Visual separator

        # Execute each feature extraction step in sequence
        df = self.extract_sentiment_features(df, text_column)     # Step 1: Sentiment (5)
        df = self.extract_readability_features(df, text_column)   # Step 2: Readability (6)
        df = self.extract_vocabulary_features(df, text_column)    # Step 3: Vocabulary (6)
        df = self.extract_esg_keyword_features(df, text_column)   # Step 4: ESG keywords (10)
        df = self.extract_greenwashing_linguistic_features(        # Step 5: GW linguistic (12)
            df, text_column)
        df = self.extract_document_structure_features(            # Step 6: Doc structure (8)
            df, text_column)

        # Print summary report
        total_features = sum(                                     # Count total features
            info['count'] for info in self.feature_registry.values()  # Sum per category
        )
        print(f"\n    TOTAL NLP FEATURES ENGINEERED: {total_features}")  # Print total
        for category, info in self.feature_registry.items():      # Per-category summary
            print(f"      - {category}: {info['count']} features")  # Category count
        print("=" * 70)                                           # Visual separator

        return df                                                 # Return fully enriched df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":                                        # Only run if executed directly

    # Print script header
    print("=" * 70)                                               # Visual separator
    print("  NLP FEATURE ENGINEERING - STANDALONE TEST")          # Script title
    print("=" * 70)                                               # Visual separator

    # Load company profiles data
    DATA_PATH = "data/processed/company_profiles.csv"             # Input file path
    print(f"\n  Loading data from: {DATA_PATH}")                  # Log path
    df = pd.read_csv(DATA_PATH)                                   # Read CSV
    print(f"  Initial shape: {df.shape}")                         # Print dimensions

    # Initialize and run NLP feature engineer
    nlp_engineer = NLPFeatureEngineer()                            # Create instance
    df_nlp = nlp_engineer.engineer_all_nlp_features(df)           # Run full pipeline

    # Display results
    print(f"\n  Final shape: {df_nlp.shape}")                     # Print final dimensions
    print(f"  NLP features added: {df_nlp.shape[1] - 13}")       # Count new columns

    # Show sample of key NLP features
    key_nlp = [                                                   # Key features to display
        'company_name',                                           # Identifier
        'text_polarity',                                          # Sentiment direction
        'flesch_reading_ease',                                    # Readability
        'lexical_diversity',                                      # Vocabulary richness
        'total_esg_keyword_density',                              # ESG keyword density
        'greenwashing_signal_score',                              # GW linguistic signal
        'vague_to_concrete_ratio'                                 # Vague vs concrete
    ]
    key_nlp = [c for c in key_nlp if c in df_nlp.columns]        # Filter existing

    print(f"\n  Sample of key NLP features (top 10):")            # Header
    print(df_nlp[key_nlp].head(10).to_string())                  # Print sample

    # Save NLP features
    OUTPUT_PATH = "data/processed/nlp_features.csv"               # Output path
    df_nlp.to_csv(OUTPUT_PATH, index=False)                       # Save to CSV
    print(f"\n  Saved NLP features to: {OUTPUT_PATH}")            # Confirm
    print("=" * 70)                                               # Visual separator
