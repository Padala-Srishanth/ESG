"""
================================================================================
FEATURE ENGINEERING - NLP TEXT FEATURES MODULE (ENHANCED)
================================================================================
Author  : ML Engineering Scientist (20+ years industry experience)
Project : ESG Greenwashing Detection via NLP & Machine Learning
Purpose : Extract rich numerical feature vectors from company description text
          using multiple NLP techniques — sentiment, readability, vocabulary
          complexity, ESG keyword density, linguistic greenwashing signals,
          government policy compliance, news intent analysis, temporal
          linguistics, and cross-feature aggregate scoring.

Design Philosophy:
    Text is the PRIMARY evidence of greenwashing. Companies reveal their
    true intentions through language patterns. This module converts raw
    text descriptions into 90+ quantitative features that ML models can
    consume. Every feature is designed with a specific greenwashing
    detection rationale.

    NOVELTY: Categories 7-10 introduce government policy benchmarking,
    news intent classification, temporal tense analysis, and a novel
    cross-feature aggregate ESG credibility index — features not found
    in standard ESG NLP pipelines. These are designed to maximize SHAP
    interpretability and provide actionable policy-gap insights.

Feature Categories:
    1. Sentiment Features            — VADER polarity, subjectivity, confidence
    2. Readability Features          — Flesch, Gunning Fog, syllable complexity
    3. Vocabulary Features           — lexical diversity, word frequencies, n-grams
    4. ESG Keyword Features          — domain-specific term density per pillar
    5. Greenwashing Linguistic       — vagueness, hedging, superlatives, future tense
    6. Document Structure            — sentence count, paragraph length, section balance
    7. Government Policy Compliance  — alignment to Paris, EU Taxonomy, TCFD, SDGs, SEC
    8. News Intent & Narrative       — promotional vs factual, defensive, crisis patterns
    9. Temporal Linguistic Signals   — tense ratios, commitment horizons, progress tracking
   10. Cross-Feature Aggregate Score — ESG credibility index, SHAP-optimized interactions
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

        # ==================================================================
        # GOVERNMENT POLICY & REGULATORY FRAMEWORK LEXICONS
        # ==================================================================
        # Curated from major global ESG regulatory frameworks and standards
        # Used to measure company alignment with government policy expectations

        # Paris Agreement & Climate Policy keywords
        self.paris_agreement_keywords = [
            'paris agreement', 'paris accord', 'nationally determined',
            'ndc', '1.5 degree', '1.5°c', '2 degree', '2°c',
            'cop26', 'cop27', 'cop28', 'unfccc', 'kyoto',
            'carbon neutral', 'carbon neutrality', 'net zero', 'net-zero',
            'climate action', 'climate target', 'science-based target',
            'sbti', 'science based targets initiative',
            'carbon budget', 'carbon pricing', 'carbon tax',
            'emission reduction', 'emissions reduction',
            'decarbonization', 'decarbonisation', 'just transition'
        ]

        # EU Taxonomy & European Regulation keywords
        self.eu_taxonomy_keywords = [
            'eu taxonomy', 'european taxonomy', 'taxonomy regulation',
            'sustainable finance', 'sfdr', 'csrd', 'nfrd',
            'corporate sustainability reporting', 'double materiality',
            'taxonomy alignment', 'taxonomy eligible',
            'substantial contribution', 'do no significant harm', 'dnsh',
            'technical screening criteria', 'minimum safeguards',
            'green bond', 'green bond standard', 'ecolabel',
            'eu green deal', 'european green deal', 'fit for 55',
            'cbam', 'carbon border', 'ets', 'emissions trading'
        ]

        # SEC Climate Disclosure & US Regulation keywords
        # Broadened to include terms found in business descriptions
        self.sec_climate_keywords = [
            'sec climate', 'sec disclosure', 'climate disclosure',
            'climate risk disclosure', 'material climate risk',
            'regulation s-k', 'form 10-k climate', 'proxy statement',
            'fiduciary duty', 'shareholder proposal',
            'dodd-frank', 'esg disclosure', 'mandatory disclosure',
            'climate litigation', 'stranded asset', 'stranded assets',
            'physical risk', 'transition risk', 'climate scenario',
            'stress test', 'climate stress',
            'securities', 'regulatory', 'regulation', 'compliance',
            'oversight', 'material', 'disclosure'
        ]

        # TCFD (Task Force on Climate-related Financial Disclosures)
        # Broadened to include governance and risk management terms
        self.tcfd_keywords = [
            'tcfd', 'task force on climate', 'climate-related financial',
            'governance of climate', 'climate strategy',
            'climate risk management', 'climate metrics',
            'climate targets', 'scenario analysis',
            'climate governance', 'board oversight climate',
            'climate opportunity', 'climate resilience',
            'risk management', 'board', 'governance', 'climate',
            'reporting', 'audit', 'assurance'
        ]

        # UN Sustainable Development Goals (SDGs)
        self.sdg_keywords = [
            'sdg', 'sustainable development goal', 'sdgs',
            'no poverty', 'zero hunger', 'good health',
            'quality education', 'gender equality', 'clean water',
            'affordable energy', 'decent work', 'industry innovation',
            'reduced inequalities', 'sustainable cities',
            'responsible consumption', 'climate action',
            'life below water', 'life on land', 'peace justice',
            'partnerships for the goals', 'un global compact',
            'ungc', 'principles for responsible investment', 'pri'
        ]

        # GRI Standards (Global Reporting Initiative)
        # Broadened to include reporting and stakeholder terms
        self.gri_standards_keywords = [
            'gri standards', 'gri reporting', 'gri framework',
            'materiality assessment', 'stakeholder engagement',
            'gri 300', 'gri 400', 'gri 200',
            'universal standards', 'topic standards',
            'reporting boundary', 'reporting period',
            'assurance statement', 'external assurance',
            'integrated reporting', 'iirc', 'value reporting',
            'materiality', 'stakeholder', 'reporting',
            'transparency', 'accountability'
        ]

        # ==================================================================
        # NEWS INTENT & NARRATIVE PATTERN LEXICONS
        # ==================================================================
        # Patterns for classifying the communicative intent of company text

        # Promotional / Marketing language patterns
        self.promotional_patterns = [
            'proud to announce', 'excited to', 'thrilled to',
            'delighted to', 'pleased to announce', 'honored to',
            'award-winning', 'recognized as', 'named as',
            'industry leader', 'market leader', 'global leader',
            'best practice', 'gold standard', 'flagship',
            'showcase', 'highlight', 'celebrate', 'milestone'
        ]

        # Defensive / Crisis-response language patterns
        self.defensive_patterns = [
            'we deny', 'we reject', 'allegations', 'allegation',
            'unfounded', 'misleading', 'inaccurate', 'taken out of context',
            'we disagree', 'we dispute', 'contrary to reports',
            'clarify', 'clarification', 'set the record straight',
            'regret', 'apologize', 'apology', 'deeply sorry',
            'remediation', 'remedial', 'corrective action',
            'investigation', 'under review', 'compliance review'
        ]

        # Factual / Data-driven language patterns
        # Adapted for corporate descriptions: includes business metrics,
        # quantitative terms, and operational specifics found in SEC filings
        self.factual_patterns = [
            'according to', 'data shows', 'evidence indicates',
            'research demonstrates', 'study found', 'analysis reveals',
            'year-over-year', 'quarter-over-quarter', 'compared to',
            'increased from', 'decreased from', 'grew by',
            'declined by', 'remained stable', 'fluctuated',
            'approximately', 'million', 'billion', 'revenue',
            'segments', 'operates', 'headquartered', 'founded',
            'subsidiaries', 'customers', 'employs', 'portfolio',
            'generated', 'reported', 'fiscal', 'annual'
        ]

        # Forward-looking / Strategic language patterns
        # Adapted for corporate descriptions: includes business strategy,
        # market positioning, and operational planning terms
        self.strategic_patterns = [
            'strategic priority', 'long-term strategy', 'roadmap',
            'action plan', 'implementation plan', 'phased approach',
            'interim target', 'short-term goal', 'medium-term',
            'milestone', 'deliverable', 'key performance indicator',
            'accountability', 'governance structure', 'oversight mechanism',
            'review mechanism', 'progress report', 'annual review',
            'strategic', 'strategy', 'market', 'platform',
            'investment', 'acquisition', 'diversified', 'scale',
            'partnership', 'innovation', 'pipeline', 'expansion'
        ]

        # ==================================================================
        # TEMPORAL LINGUISTIC PATTERNS
        # ==================================================================
        # Patterns for detecting temporal orientation in text

        # Past tense / Achievement language
        # Adapted for corporate descriptions: includes company founding,
        # historical operations, and established business activities
        self.past_achievement_patterns = [
            'achieved', 'accomplished', 'completed', 'delivered',
            'implemented', 'established', 'launched', 'deployed',
            'reduced', 'eliminated', 'resolved', 'addressed',
            'invested', 'donated', 'contributed', 'partnered',
            'last year', 'previous year', 'in 2020', 'in 2021',
            'in 2022', 'in 2023', 'in 2024', 'in 2025',
            'since 2015', 'since 2018', 'since 2020',
            'historically', 'over the past', 'in recent years',
            'founded', 'built', 'developed', 'created',
            'expanded', 'acquired', 'generated', 'grew'
        ]

        # Present tense / Current action language
        self.present_action_patterns = [
            'currently', 'now', 'today', 'this year',
            'ongoing', 'in progress', 'underway', 'active',
            'we are', 'we have', 'we operate', 'we maintain',
            'continuously', 'regularly', 'consistently',
            'at present', 'as of', 'real-time', 'day-to-day'
        ]

        # Specific future timeline patterns (verifiable commitments)
        # Includes fiscal/financial reporting terms common in SEC filings
        self.specific_future_patterns = [
            'by 2025', 'by 2026', 'by 2027', 'by 2028',
            'by 2029', 'by 2030', 'by 2035', 'by 2040',
            'by 2045', 'by 2050', 'within 5 years',
            'within 3 years', 'within 2 years', 'next year',
            'q1', 'q2', 'q3', 'q4', 'fiscal year',
            'first half', 'second half', 'by end of',
            'fiscal', 'annual', 'quarterly', 'pipeline',
            'backlog', 'contracted', 'scheduled'
        ]

        # Vague future patterns (unverifiable promises)
        self.vague_future_patterns = [
            'someday', 'eventually', 'in the future',
            'in due course', 'when possible', 'as soon as',
            'in the coming years', 'over time', 'gradually',
            'step by step', 'moving forward', 'going forward',
            'down the road', 'in the long run', 'one day'
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
    # CATEGORY 7: GOVERNMENT POLICY COMPLIANCE FEATURES
    # ========================================================================

    def extract_government_policy_features(self, df, text_column='description'):
        """
        Measure company text alignment against major government ESG policies.

        Rationale:
            Government policies set the STANDARD for ESG performance. Companies
            that reference specific frameworks (Paris Agreement, EU Taxonomy,
            TCFD, SDGs) demonstrate awareness and commitment. Companies that
            use ESG language WITHOUT referencing any regulatory framework are
            more likely greenwashing — they adopt the vocabulary without the
            accountability structure.

            NOVEL INSIGHT: The ratio of policy-aligned language to total ESG
            language reveals whether a company's ESG claims are grounded in
            recognized standards or are self-defined (higher greenwashing risk).

        Mathematical Framework:
            Policy Alignment Score (PAS):
                PAS_i = sum(policy_keyword_hits_i) / max(total_esg_keywords, 1)
                for each framework i in {Paris, EU, SEC, TCFD, SDG, GRI}

            Regulatory Breadth Index (RBI):
                RBI = count(frameworks_mentioned > 0) / total_frameworks
                Range: [0, 1], where 1 = mentions all 6 frameworks

            Policy Gap Score (PGS):
                PGS = sigmoid(esg_keyword_density - policy_keyword_density)
                High PGS = lots of ESG talk but no policy grounding

        Features Created (12):
            - paris_agreement_alignment     : Paris Agreement keyword density
            - eu_taxonomy_alignment         : EU Taxonomy keyword density
            - sec_climate_alignment         : SEC Climate Disclosure keyword density
            - tcfd_alignment                : TCFD framework keyword density
            - sdg_alignment                 : UN SDG keyword density
            - gri_standards_alignment       : GRI Standards keyword density
            - regulatory_breadth_index      : Fraction of frameworks mentioned
            - total_policy_density          : Combined policy keyword density
            - policy_specificity_score      : Specific vs generic policy references
            - policy_esg_gap               : Gap between ESG talk and policy grounding
            - framework_consistency_score   : Consistency across mentioned frameworks
            - regulatory_readiness_score    : Composite regulatory preparedness (0-1)
        """

        print("    [7/10] Extracting government policy compliance features...")

        # Store all 6 framework lexicons for iteration
        policy_frameworks = {
            'paris': self.paris_agreement_keywords,
            'eu_taxonomy': self.eu_taxonomy_keywords,
            'sec_climate': self.sec_climate_keywords,
            'tcfd': self.tcfd_keywords,
            'sdg': self.sdg_keywords,
            'gri': self.gri_standards_keywords
        }

        # Initialize storage for all 12 features
        paris_scores = []
        eu_scores = []
        sec_scores = []
        tcfd_scores = []
        sdg_scores = []
        gri_scores = []
        breadth_indices = []
        total_policy_densities = []
        specificity_scores = []
        policy_gaps = []
        consistency_scores = []
        readiness_scores = []

        for idx, row in df.iterrows():
            text = str(row.get(text_column, '')).lower()
            words = re.findall(r'\b[a-z]+\b', text)
            total_words = max(len(words), 1)

            # ------------------------------------------------------------------
            # Count keyword hits per framework
            # ------------------------------------------------------------------
            framework_hits = {}
            framework_densities = {}
            for name, keywords in policy_frameworks.items():
                hits = sum(text.count(kw) for kw in keywords)
                framework_hits[name] = hits
                framework_densities[name] = hits / total_words

            paris_scores.append(framework_densities['paris'])
            eu_scores.append(framework_densities['eu_taxonomy'])
            sec_scores.append(framework_densities['sec_climate'])
            tcfd_scores.append(framework_densities['tcfd'])
            sdg_scores.append(framework_densities['sdg'])
            gri_scores.append(framework_densities['gri'])

            # ------------------------------------------------------------------
            # Regulatory Breadth Index: fraction of frameworks mentioned
            # ------------------------------------------------------------------
            frameworks_mentioned = sum(
                1 for hits in framework_hits.values() if hits > 0
            )
            rbi = frameworks_mentioned / len(policy_frameworks)
            breadth_indices.append(rbi)

            # ------------------------------------------------------------------
            # Total policy keyword density
            # ------------------------------------------------------------------
            total_policy_hits = sum(framework_hits.values())
            total_policy_density = total_policy_hits / total_words
            total_policy_densities.append(total_policy_density)

            # ------------------------------------------------------------------
            # Policy Specificity Score
            # Specific references (named frameworks) vs generic ESG language
            # High specificity = more credible ESG claims
            # ------------------------------------------------------------------
            # Count all ESG keywords (from Category 4 lexicons)
            all_esg_hits = (
                sum(text.count(kw) for kw in self.env_keywords)
                + sum(text.count(kw) for kw in self.social_keywords)
                + sum(text.count(kw) for kw in self.gov_keywords)
            )
            specificity = (
                total_policy_hits / max(all_esg_hits, 1)
            )
            # Cap at 1.0 (policy refs can't exceed ESG refs meaningfully)
            specificity_scores.append(min(specificity, 1.0))

            # ------------------------------------------------------------------
            # Policy-ESG Gap Score
            # Measures: company talks ESG but doesn't ground in policy
            # Formula: sigmoid(esg_density - policy_density)
            # High gap = greenwashing signal (ESG buzzwords without framework)
            # ------------------------------------------------------------------
            esg_density = all_esg_hits / total_words
            raw_gap = esg_density - total_policy_density
            policy_gap = 1.0 / (1.0 + np.exp(-15 * raw_gap))
            policy_gaps.append(policy_gap)

            # ------------------------------------------------------------------
            # Framework Consistency Score
            # If a company mentions multiple frameworks, are they balanced?
            # Low std across mentioned frameworks = consistent engagement
            # ------------------------------------------------------------------
            mentioned_densities = [
                d for d in framework_densities.values() if d > 0
            ]
            if len(mentioned_densities) >= 2:
                consistency = 1.0 - min(
                    np.std(mentioned_densities) / (np.mean(mentioned_densities) + 1e-8),
                    1.0
                )
            elif len(mentioned_densities) == 1:
                consistency = 0.5                                    # Single framework = partial
            else:
                consistency = 0.0                                    # No frameworks = no consistency
            consistency_scores.append(consistency)

            # ------------------------------------------------------------------
            # Regulatory Readiness Score (Composite)
            # Weighted combination of breadth, specificity, and consistency
            # Formula: 0.35*RBI + 0.30*specificity + 0.20*consistency + 0.15*(1-gap)
            # ------------------------------------------------------------------
            readiness = (
                0.35 * rbi
                + 0.30 * min(specificity, 1.0)
                + 0.20 * consistency
                + 0.15 * (1.0 - policy_gap)
            )
            readiness_scores.append(readiness)

        # Assign all features to dataframe
        df['paris_agreement_alignment'] = paris_scores
        df['eu_taxonomy_alignment'] = eu_scores
        df['sec_climate_alignment'] = sec_scores
        df['tcfd_alignment'] = tcfd_scores
        df['sdg_alignment'] = sdg_scores
        df['gri_standards_alignment'] = gri_scores
        df['regulatory_breadth_index'] = breadth_indices
        df['total_policy_density'] = total_policy_densities
        df['policy_specificity_score'] = specificity_scores
        df['policy_esg_gap'] = policy_gaps
        df['framework_consistency_score'] = consistency_scores
        df['regulatory_readiness_score'] = readiness_scores

        # Register features
        self.feature_registry['government_policy'] = {
            'count': 12,
            'features': [
                'paris_agreement_alignment', 'eu_taxonomy_alignment',
                'sec_climate_alignment', 'tcfd_alignment',
                'sdg_alignment', 'gri_standards_alignment',
                'regulatory_breadth_index', 'total_policy_density',
                'policy_specificity_score', 'policy_esg_gap',
                'framework_consistency_score', 'regulatory_readiness_score'
            ]
        }

        return df

    # ========================================================================
    # CATEGORY 8: NEWS INTENT & NARRATIVE ANALYSIS FEATURES
    # ========================================================================

    def extract_news_intent_features(self, df, text_column='description'):
        """
        Classify the communicative intent and narrative strategy in company text.

        Rationale:
            Company communications serve different purposes — promotional,
            defensive, factual, or strategic. Greenwashing companies
            disproportionately use PROMOTIONAL language and avoid FACTUAL
            or DEFENSIVE disclosures. By classifying intent, we create
            features that help SHAP explain WHY a company is flagged.

            NEWS INSIGHT: When a company's text is mostly promotional but
            their controversy score is high, the intent-reality divergence
            is a powerful greenwashing signal. This cross-referencing with
            existing features creates SHAP-interpretable interactions.

        Mathematical Framework:
            Intent Classification Vector (ICV):
                ICV = [promotional_d, defensive_d, factual_d, strategic_d]
                where _d = density (hits / total_words)

            Narrative Credibility Index (NCI):
                NCI = (factual_d + strategic_d) / (promotional_d + defensive_d + eps)
                High NCI = substantive communication
                Low NCI = marketing-heavy or crisis-driven communication

            Promotional Dominance Score (PDS):
                PDS = promotional_d / (sum(ICV) + eps)
                Range: [0, 1], where 1 = purely promotional

        Features Created (10):
            - promotional_intent_density    : Marketing/PR language density
            - defensive_intent_density      : Crisis/denial language density
            - factual_intent_density        : Data-driven language density
            - strategic_intent_density      : Strategic planning language density
            - narrative_credibility_index   : Factual+strategic vs promotional+defensive
            - promotional_dominance_score   : How promotional the overall text is
            - intent_diversity_score        : Shannon entropy across 4 intent types
            - defensive_to_factual_ratio    : Denial language vs evidence language
            - sentiment_intent_divergence   : Gap between positive sentiment and factual content
            - news_greenwashing_signal      : Combined news-based greenwashing indicator
        """

        print("    [8/10] Extracting news intent & narrative features...")

        # Initialize storage for all 10 features
        promotional_densities = []
        defensive_densities = []
        factual_densities = []
        strategic_densities = []
        credibility_indices = []
        dominance_scores = []
        diversity_scores = []
        defensive_factual_ratios = []
        sentiment_intent_gaps = []
        news_gw_signals = []

        for idx, row in df.iterrows():
            text = str(row.get(text_column, '')).lower()
            words = re.findall(r'\b[a-z]+\b', text)
            total_words = max(len(words), 1)

            # Count hits for each intent category
            promo_hits = sum(text.count(p) for p in self.promotional_patterns)
            defense_hits = sum(text.count(p) for p in self.defensive_patterns)
            factual_hits = sum(text.count(p) for p in self.factual_patterns)
            strategic_hits = sum(text.count(p) for p in self.strategic_patterns)

            # Calculate densities
            promo_d = promo_hits / total_words
            defense_d = defense_hits / total_words
            factual_d = factual_hits / total_words
            strategic_d = strategic_hits / total_words

            promotional_densities.append(promo_d)
            defensive_densities.append(defense_d)
            factual_densities.append(factual_d)
            strategic_densities.append(strategic_d)

            # ------------------------------------------------------------------
            # Narrative Credibility Index (NCI)
            # Substantive (factual + strategic) vs superficial (promo + defensive)
            # ------------------------------------------------------------------
            substantive = factual_d + strategic_d
            superficial = promo_d + defense_d
            nci = substantive / (superficial + 1e-8)
            # Normalize to [0, 1] with sigmoid
            nci_normalized = 1.0 / (1.0 + np.exp(-2 * (nci - 1.0)))
            credibility_indices.append(nci_normalized)

            # ------------------------------------------------------------------
            # Promotional Dominance Score
            # What fraction of all intent signals is promotional?
            # ------------------------------------------------------------------
            total_intent = promo_d + defense_d + factual_d + strategic_d
            pds = promo_d / (total_intent + 1e-8)
            dominance_scores.append(pds)

            # ------------------------------------------------------------------
            # Intent Diversity Score (Shannon Entropy)
            # High entropy = balanced communication across intent types
            # Low entropy = dominated by single intent (suspicious if promotional)
            # ------------------------------------------------------------------
            intent_vec = [promo_d, defense_d, factual_d, strategic_d]
            intent_sum = sum(intent_vec) + 1e-8
            probs = [v / intent_sum for v in intent_vec]
            entropy = -sum(
                p * np.log2(p + 1e-10) for p in probs
            )
            # Normalize by max entropy (log2(4) = 2.0)
            diversity_scores.append(entropy / 2.0)

            # ------------------------------------------------------------------
            # Defensive-to-Factual Ratio
            # High ratio = more denial than evidence (crisis mode without data)
            # Normalized with sigmoid to bound output in [0, 1]
            # ------------------------------------------------------------------
            dtf_raw = (defense_d + 1e-8) / (factual_d + 1e-8)
            dtf_ratio = 1.0 / (1.0 + np.exp(-2 * (dtf_raw - 1.0)))
            defensive_factual_ratios.append(dtf_ratio)

            # ------------------------------------------------------------------
            # Sentiment-Intent Divergence
            # If text is very positive (from Category 1) but low on factual
            # content, this divergence signals potential greenwashing.
            # Uses polarity from already-computed sentiment features.
            # ------------------------------------------------------------------
            polarity = row.get('text_polarity', 0.0)
            if pd.isna(polarity):
                polarity = 0.0
            # Divergence = high positivity combined with low factual density
            sid = max(polarity, 0) * (1.0 - min(factual_d * 50, 1.0))
            sentiment_intent_gaps.append(sid)

            # ------------------------------------------------------------------
            # News Greenwashing Signal (Composite)
            # Combines promotional dominance, low credibility, low factual
            # Formula: 0.4*pds + 0.3*(1-nci_norm) + 0.3*sid
            # ------------------------------------------------------------------
            news_gw = (
                0.4 * pds
                + 0.3 * (1.0 - nci_normalized)
                + 0.3 * sid
            )
            news_gw_signals.append(news_gw)

        # Assign all features to dataframe
        df['promotional_intent_density'] = promotional_densities
        df['defensive_intent_density'] = defensive_densities
        df['factual_intent_density'] = factual_densities
        df['strategic_intent_density'] = strategic_densities
        df['narrative_credibility_index'] = credibility_indices
        df['promotional_dominance_score'] = dominance_scores
        df['intent_diversity_score'] = diversity_scores
        df['defensive_to_factual_ratio'] = defensive_factual_ratios
        df['sentiment_intent_divergence'] = sentiment_intent_gaps
        df['news_greenwashing_signal'] = news_gw_signals

        # Register features
        self.feature_registry['news_intent'] = {
            'count': 10,
            'features': [
                'promotional_intent_density', 'defensive_intent_density',
                'factual_intent_density', 'strategic_intent_density',
                'narrative_credibility_index', 'promotional_dominance_score',
                'intent_diversity_score', 'defensive_to_factual_ratio',
                'sentiment_intent_divergence', 'news_greenwashing_signal'
            ]
        }

        return df

    # ========================================================================
    # CATEGORY 9: TEMPORAL LINGUISTIC & TIME SERIES FEATURES
    # ========================================================================

    def extract_temporal_linguistic_features(self, df, text_column='description'):
        """
        Analyze temporal orientation and commitment horizon in company text.

        Rationale:
            TIME is a critical dimension of greenwashing. Companies that:
            - Overuse FUTURE TENSE without PAST ACHIEVEMENTS = promise without proof
            - Use VAGUE timelines ("eventually") vs SPECIFIC dates ("by 2030")
            - Show heavy PRESENT language without concrete PAST results

            This temporal analysis creates features that improve time-series
            ESG modeling by capturing the linguistic dimension of temporal
            commitment — a novel signal not in standard ESG feature sets.

        Mathematical Framework:
            Temporal Orientation Vector (TOV):
                TOV = [past_d, present_d, specific_future_d, vague_future_d]

            Commitment Credibility Score (CCS):
                CCS = (past_d + specific_future_d) / (vague_future_d + past_d + eps)
                High CCS = verifiable track record + specific targets
                Low CCS = vague promises without evidence of past delivery

            Temporal Specificity Ratio (TSR):
                TSR = specific_future_d / (specific_future_d + vague_future_d + eps)
                Range: [0, 1], where 1 = all future refs are specific

            Progress-to-Promise Ratio (PPR):
                PPR = past_achievement_d / (future_total_d + eps)
                High PPR = company delivers more than it promises
                Low PPR = company promises more than it delivers (greenwashing)

        Features Created (10):
            - past_achievement_density      : Past tense achievement language density
            - present_action_density        : Present tense action language density
            - specific_future_density       : Specific dated future commitments density
            - vague_future_density          : Vague undated future promises density
            - temporal_balance_score        : Balance across past/present/future
            - commitment_credibility_score  : Past track record + specific targets
            - temporal_specificity_ratio    : Specific vs vague future references
            - progress_to_promise_ratio     : Achievements vs promises ratio
            - year_mention_density          : Density of specific year references
            - temporal_greenwashing_signal  : Composite temporal GW indicator
        """

        print("    [9/10] Extracting temporal linguistic features...")

        # Initialize storage for all 10 features
        past_densities = []
        present_densities = []
        specific_future_densities = []
        vague_future_densities = []
        temporal_balances = []
        credibility_scores = []
        specificity_ratios = []
        progress_promise_ratios = []
        year_densities = []
        temporal_gw_signals = []

        for idx, row in df.iterrows():
            text = str(row.get(text_column, '')).lower()
            words = re.findall(r'\b[a-z]+\b', text)
            total_words = max(len(words), 1)

            # Count hits for each temporal category
            past_hits = sum(text.count(p) for p in self.past_achievement_patterns)
            present_hits = sum(text.count(p) for p in self.present_action_patterns)
            spec_future_hits = sum(text.count(p) for p in self.specific_future_patterns)
            vague_future_hits = sum(text.count(p) for p in self.vague_future_patterns)

            # Calculate densities
            past_d = past_hits / total_words
            present_d = present_hits / total_words
            spec_future_d = spec_future_hits / total_words
            vague_future_d = vague_future_hits / total_words

            past_densities.append(past_d)
            present_densities.append(present_d)
            specific_future_densities.append(spec_future_d)
            vague_future_densities.append(vague_future_d)

            # ------------------------------------------------------------------
            # Temporal Balance Score
            # Shannon entropy across 4 temporal orientations
            # Balanced temporal communication = higher credibility
            # ------------------------------------------------------------------
            temp_vec = [past_d, present_d, spec_future_d, vague_future_d]
            temp_sum = sum(temp_vec) + 1e-8
            temp_probs = [v / temp_sum for v in temp_vec]
            temp_entropy = -sum(
                p * np.log2(p + 1e-10) for p in temp_probs
            )
            temporal_balances.append(temp_entropy / 2.0)

            # ------------------------------------------------------------------
            # Commitment Credibility Score (CCS)
            # Companies with past achievements + specific future targets = credible
            # ------------------------------------------------------------------
            credible = past_d + spec_future_d
            incredible = vague_future_d + 1e-8
            ccs_raw = credible / (credible + incredible)
            credibility_scores.append(ccs_raw)

            # ------------------------------------------------------------------
            # Temporal Specificity Ratio (TSR)
            # Of all future-oriented language, how much is specific vs vague?
            # ------------------------------------------------------------------
            future_total = spec_future_d + vague_future_d
            tsr = spec_future_d / (future_total + 1e-8)
            specificity_ratios.append(tsr)

            # ------------------------------------------------------------------
            # Progress-to-Promise Ratio (PPR)
            # Past achievements / future promises
            # High = delivers more than promises; Low = promises more than delivers
            # ------------------------------------------------------------------
            all_future_d = spec_future_d + vague_future_d
            ppr = past_d / (all_future_d + 1e-8)
            # Normalize with sigmoid centered at 1.0 (balanced)
            ppr_normalized = 1.0 / (1.0 + np.exp(-2 * (ppr - 1.0)))
            progress_promise_ratios.append(ppr_normalized)

            # ------------------------------------------------------------------
            # Year Mention Density
            # Count specific year references (2020-2060) — concrete temporal anchors
            # ------------------------------------------------------------------
            year_mentions = len(re.findall(
                r'\b(19|20)\d{2}\b', text
            ))
            year_density = year_mentions / total_words
            year_densities.append(year_density)

            # ------------------------------------------------------------------
            # Temporal Greenwashing Signal (Composite)
            # High vague future + low past achievement + low specificity = greenwashing
            # Formula: 0.35*(1-TSR) + 0.30*(1-PPR_norm) + 0.20*vague_future_d*100
            #          + 0.15*(1-temporal_balance)
            # ------------------------------------------------------------------
            temporal_gw = (
                0.35 * (1.0 - tsr)
                + 0.30 * (1.0 - ppr_normalized)
                + 0.20 * min(vague_future_d * 100, 1.0)
                + 0.15 * (1.0 - temp_entropy / 2.0)
            )
            temporal_gw_signals.append(temporal_gw)

        # Assign all features to dataframe
        df['past_achievement_density'] = past_densities
        df['present_action_density'] = present_densities
        df['specific_future_density'] = specific_future_densities
        df['vague_future_density'] = vague_future_densities
        df['temporal_balance_score'] = temporal_balances
        df['commitment_credibility_score'] = credibility_scores
        df['temporal_specificity_ratio'] = specificity_ratios
        df['progress_to_promise_ratio'] = progress_promise_ratios
        df['year_mention_density'] = year_densities
        df['temporal_greenwashing_signal'] = temporal_gw_signals

        # Register features
        self.feature_registry['temporal_linguistic'] = {
            'count': 10,
            'features': [
                'past_achievement_density', 'present_action_density',
                'specific_future_density', 'vague_future_density',
                'temporal_balance_score', 'commitment_credibility_score',
                'temporal_specificity_ratio', 'progress_to_promise_ratio',
                'year_mention_density', 'temporal_greenwashing_signal'
            ]
        }

        return df

    # ========================================================================
    # CATEGORY 10: CROSS-FEATURE AGGREGATE ESG SCORE
    # ========================================================================

    def extract_aggregate_esg_score_features(self, df, text_column='description'):  # noqa: ARG002
        """
        Create cross-feature interaction terms and a novel aggregate ESG
        credibility index that synthesizes signals from ALL previous categories.

        Rationale:
            Individual NLP features capture isolated signals. But greenwashing
            is a MULTI-DIMENSIONAL phenomenon — it manifests simultaneously
            across sentiment, readability, policy alignment, temporal patterns,
            and narrative intent. This category creates interaction features
            and a composite score that SHAP can decompose to explain exactly
            WHICH dimensions drive a company's greenwashing risk.

        Mathematical Framework:
            ESG Linguistic Credibility Index (ELCI):
                ELCI = w1*S + w2*P + w3*T + w4*N + w5*R + w6*(1-GW)

                Where:
                    S = policy_specificity_score (Category 7)
                    P = regulatory_readiness_score (Category 7)
                    T = commitment_credibility_score (Category 9)
                    N = narrative_credibility_index (Category 8)
                    R = concrete_evidence_density (Category 5), normalized
                    GW = greenwashing_signal_score (Category 5)

                Weights: w1=0.20, w2=0.20, w3=0.20, w4=0.15, w5=0.15, w6=0.10
                Range: [0, 1], where 1 = maximum linguistic credibility

            SHAP Interaction Features:
                - policy_x_sentiment: policy alignment * text polarity
                - readability_x_greenwashing: complexity * GW signal
                - temporal_x_policy: temporal credibility * policy readiness
                These multiplicative interactions create non-linear decision
                boundaries that tree-based models (GB, XGBoost) exploit well.

        Features Created (12):
            - policy_sentiment_interaction   : Policy alignment * sentiment polarity
            - readability_greenwashing_interaction : Complexity * GW signal
            - temporal_policy_interaction    : Temporal credibility * policy readiness
            - vocabulary_intent_interaction  : Lexical diversity * factual intent
            - evidence_readability_interaction : Concrete evidence * reading ease
            - claim_credibility_ratio        : Concrete claims / (vague + hedge + promo)
            - multi_signal_greenwashing_score : Ensemble of all GW signals
            - esg_linguistic_credibility_index : Master composite score (ELCI)
            - credibility_confidence_interval : Uncertainty estimate of ELCI
            - policy_temporal_alignment      : Do policy refs match temporal commitments?
            - narrative_consistency_score     : Consistency across all NLP dimensions
            - aggregate_esg_nlp_score        : Final 0-100 aggregate ESG NLP score
        """

        print("    [10/10] Computing cross-feature aggregate ESG score...")

        # Initialize storage for all 12 features
        policy_sentiment_interactions = []
        readability_gw_interactions = []
        temporal_policy_interactions = []
        vocab_intent_interactions = []
        evidence_readability_interactions = []
        claim_credibility_ratios = []
        multi_signal_gw_scores = []
        elci_scores = []
        confidence_intervals = []
        policy_temporal_alignments = []
        narrative_consistencies = []
        aggregate_scores = []

        for idx, row in df.iterrows():
            # Safely retrieve features from prior categories with defaults
            def safe_get(col, default=0.0):
                val = row.get(col, default)
                if pd.isna(val):
                    return default
                return float(val)

            # ------------------------------------------------------------------
            # Retrieve prior category features
            # ------------------------------------------------------------------
            # Category 1: Sentiment
            polarity = safe_get('text_polarity')

            # Category 2: Readability
            flesch = safe_get('flesch_reading_ease')

            # Category 3: Vocabulary
            lexical_div = safe_get('lexical_diversity')

            # Category 5: Greenwashing signals
            gw_score = safe_get('greenwashing_signal_score', 0.5)
            concrete_density = safe_get('concrete_evidence_density')
            vague_density = safe_get('vague_language_density')
            hedge_density = safe_get('hedge_language_density')

            # Category 7: Policy
            policy_specificity = safe_get('policy_specificity_score')
            reg_readiness = safe_get('regulatory_readiness_score')
            policy_gap = safe_get('policy_esg_gap', 0.5)
            total_policy = safe_get('total_policy_density')

            # Category 8: News intent
            narrative_cred = safe_get('narrative_credibility_index', 0.5)
            promo_dominance = safe_get('promotional_dominance_score')
            factual_density = safe_get('factual_intent_density')
            news_gw = safe_get('news_greenwashing_signal', 0.5)

            # Category 9: Temporal
            commit_cred = safe_get('commitment_credibility_score', 0.5)
            temporal_gw = safe_get('temporal_greenwashing_signal', 0.5)
            ppr = safe_get('progress_to_promise_ratio', 0.5)

            # ------------------------------------------------------------------
            # SHAP Interaction Feature 1: Policy * Sentiment
            # If company is positive AND policy-aligned → credible
            # If company is positive but NOT policy-aligned → greenwashing
            # ------------------------------------------------------------------
            policy_sent = policy_specificity * max(polarity, 0)
            policy_sentiment_interactions.append(policy_sent)

            # ------------------------------------------------------------------
            # SHAP Interaction Feature 2: Readability * Greenwashing
            # Complex text (low Flesch) combined with high GW signal = obfuscation
            # ------------------------------------------------------------------
            # Normalize Flesch to [0, 1] where 0=very hard, 1=very easy
            flesch_norm = max(min(flesch / 100.0, 1.0), 0.0)
            read_gw = (1.0 - flesch_norm) * gw_score
            readability_gw_interactions.append(read_gw)

            # ------------------------------------------------------------------
            # SHAP Interaction Feature 3: Temporal * Policy
            # Specific future targets + policy grounding = credible transition plan
            # ------------------------------------------------------------------
            temp_policy = commit_cred * reg_readiness
            temporal_policy_interactions.append(temp_policy)

            # ------------------------------------------------------------------
            # SHAP Interaction Feature 4: Vocabulary * Intent
            # Rich vocabulary + factual intent = substantive communication
            # ------------------------------------------------------------------
            vocab_intent = lexical_div * min(factual_density * 100, 1.0)
            vocab_intent_interactions.append(vocab_intent)

            # ------------------------------------------------------------------
            # SHAP Interaction Feature 5: Evidence * Readability
            # Concrete evidence in readable text = transparent communication
            # ------------------------------------------------------------------
            evidence_read = min(concrete_density * 100, 1.0) * flesch_norm
            evidence_readability_interactions.append(evidence_read)

            # ------------------------------------------------------------------
            # Claim Credibility Ratio
            # Concrete claims / (vague + hedge + promotional)
            # ------------------------------------------------------------------
            credible_signals = concrete_density + total_policy
            incredible_signals = vague_density + hedge_density + promo_dominance * 0.01
            ccr = credible_signals / (incredible_signals + 1e-8)
            # Normalize with sigmoid
            ccr_norm = 1.0 / (1.0 + np.exp(-5 * (ccr - 1.0)))
            claim_credibility_ratios.append(ccr_norm)

            # ------------------------------------------------------------------
            # Multi-Signal Greenwashing Score
            # Ensemble of all 3 greenwashing signals (Cat 5, 8, 9)
            # Weighted average with policy gap as bonus signal
            # ------------------------------------------------------------------
            multi_gw = (
                0.30 * gw_score                                  # Linguistic GW (Cat 5)
                + 0.25 * news_gw                                 # News intent GW (Cat 8)
                + 0.25 * temporal_gw                             # Temporal GW (Cat 9)
                + 0.20 * policy_gap                              # Policy-ESG gap (Cat 7)
            )
            multi_signal_gw_scores.append(multi_gw)

            # ------------------------------------------------------------------
            # ESG Linguistic Credibility Index (ELCI) — THE MASTER SCORE
            # Synthesizes positive signals into a single credibility measure
            # ------------------------------------------------------------------
            elci = (
                0.20 * policy_specificity                        # Policy grounding
                + 0.20 * reg_readiness                           # Regulatory preparedness
                + 0.20 * commit_cred                             # Temporal credibility
                + 0.15 * narrative_cred                          # Narrative substance
                + 0.15 * min(concrete_density * 100, 1.0)       # Evidence density
                + 0.10 * (1.0 - gw_score)                       # Inverse GW signal
            )
            elci_scores.append(elci)

            # ------------------------------------------------------------------
            # Credibility Confidence Interval
            # Measures agreement across sub-scores — high variance = uncertain
            # ------------------------------------------------------------------
            sub_scores = [
                policy_specificity, reg_readiness, commit_cred,
                narrative_cred, min(concrete_density * 100, 1.0),
                1.0 - gw_score
            ]
            confidence = 1.0 - min(np.std(sub_scores) * 2, 1.0)
            confidence_intervals.append(confidence)

            # ------------------------------------------------------------------
            # Policy-Temporal Alignment
            # Does the company's policy framework engagement match its
            # temporal commitment pattern? Both high = genuine transition plan
            # ------------------------------------------------------------------
            pta = (reg_readiness + commit_cred) / 2.0
            policy_temporal_alignments.append(pta)

            # ------------------------------------------------------------------
            # Narrative Consistency Score
            # Measures whether ALL NLP dimensions tell the same story
            # Low consistency = mixed signals (possible greenwashing)
            # ------------------------------------------------------------------
            # Collect normalized signals: higher = more credible
            credibility_signals = [
                1.0 - gw_score,                                  # Low GW signal
                narrative_cred,                                  # High narrative credibility
                commit_cred,                                     # High temporal credibility
                policy_specificity,                              # High policy alignment
                1.0 - promo_dominance,                          # Low promotional dominance
                ppr                                              # High progress/promise
            ]
            signal_mean = np.mean(credibility_signals)
            signal_std = np.std(credibility_signals)
            narrative_cons = signal_mean * (1.0 - min(signal_std, 1.0))
            narrative_consistencies.append(narrative_cons)

            # ------------------------------------------------------------------
            # AGGREGATE ESG NLP SCORE (0-100)
            # Final composite score combining credibility and risk
            # Formula: 100 * (0.5 * ELCI + 0.3 * (1-multi_GW) + 0.2 * consistency)
            # 100 = perfect ESG linguistic credibility
            # 0 = maximum greenwashing risk
            # ------------------------------------------------------------------
            aggregate = 100.0 * (
                0.50 * elci
                + 0.30 * (1.0 - multi_gw)
                + 0.20 * narrative_cons
            )
            aggregate = max(0.0, min(100.0, aggregate))          # Clamp to [0, 100]
            aggregate_scores.append(aggregate)

        # Assign all features to dataframe
        df['policy_sentiment_interaction'] = policy_sentiment_interactions
        df['readability_greenwashing_interaction'] = readability_gw_interactions
        df['temporal_policy_interaction'] = temporal_policy_interactions
        df['vocabulary_intent_interaction'] = vocab_intent_interactions
        df['evidence_readability_interaction'] = evidence_readability_interactions
        df['claim_credibility_ratio'] = claim_credibility_ratios
        df['multi_signal_greenwashing_score'] = multi_signal_gw_scores
        df['esg_linguistic_credibility_index'] = elci_scores
        df['credibility_confidence_interval'] = confidence_intervals
        df['policy_temporal_alignment'] = policy_temporal_alignments
        df['narrative_consistency_score'] = narrative_consistencies
        df['aggregate_esg_nlp_score'] = aggregate_scores

        # Register features
        self.feature_registry['aggregate_esg_score'] = {
            'count': 12,
            'features': [
                'policy_sentiment_interaction',
                'readability_greenwashing_interaction',
                'temporal_policy_interaction',
                'vocabulary_intent_interaction',
                'evidence_readability_interaction',
                'claim_credibility_ratio',
                'multi_signal_greenwashing_score',
                'esg_linguistic_credibility_index',
                'credibility_confidence_interval',
                'policy_temporal_alignment',
                'narrative_consistency_score',
                'aggregate_esg_nlp_score'
            ]
        }

        return df

    # ========================================================================
    # MASTER EXECUTION METHOD
    # ========================================================================

    def engineer_all_nlp_features(self, df, text_column='description'):
        """
        Execute the complete NLP feature engineering pipeline (ENHANCED).

        Chains all 10 NLP feature categories in sequence to extract
        a comprehensive text-based feature matrix from company descriptions.

        Pipeline Order:
            1. Sentiment Features            → 5 features
            2. Readability Features          → 6 features
            3. Vocabulary Features           → 6 features
            4. ESG Keyword Features          → 10 features
            5. Greenwashing Linguistic       → 12 features
            6. Document Structure            → 8 features
            7. Government Policy Compliance  → 12 features  (NEW)
            8. News Intent & Narrative       → 10 features  (NEW)
            9. Temporal Linguistic Signals   → 10 features  (NEW)
           10. Cross-Feature Aggregate Score → 12 features  (NEW)

        IMPORTANT: Categories 7-9 are independent and can run in any order.
        Category 10 MUST run last as it synthesizes features from all others.

        Parameters:
            df          : pd.DataFrame — company data with text descriptions
            text_column : str — column containing text (default: 'description')

        Returns:
            pd.DataFrame — with all 91 NLP features added
        """

        # Print pipeline header
        print("\n" + "=" * 70)
        print("  NLP FEATURE ENGINEERING PIPELINE (ENHANCED)")
        print("  10 Categories | 91 Features | Policy + News + Temporal + Aggregate")
        print("=" * 70)

        # Phase 1: Core NLP features (Categories 1-6, original pipeline)
        df = self.extract_sentiment_features(df, text_column)     # Step 1: Sentiment (5)
        df = self.extract_readability_features(df, text_column)   # Step 2: Readability (6)
        df = self.extract_vocabulary_features(df, text_column)    # Step 3: Vocabulary (6)
        df = self.extract_esg_keyword_features(df, text_column)   # Step 4: ESG keywords (10)
        df = self.extract_greenwashing_linguistic_features(        # Step 5: GW linguistic (12)
            df, text_column)
        df = self.extract_document_structure_features(            # Step 6: Doc structure (8)
            df, text_column)

        # Phase 2: Advanced features (Categories 7-9, independent of each other)
        df = self.extract_government_policy_features(             # Step 7: Gov policy (12)
            df, text_column)
        df = self.extract_news_intent_features(                   # Step 8: News intent (10)
            df, text_column)
        df = self.extract_temporal_linguistic_features(           # Step 9: Temporal (10)
            df, text_column)

        # Phase 3: Aggregate score (Category 10, depends on ALL prior categories)
        df = self.extract_aggregate_esg_score_features(           # Step 10: Aggregate (12)
            df, text_column)

        # Print summary report
        total_features = sum(
            info['count'] for info in self.feature_registry.values()
        )
        print(f"\n    TOTAL NLP FEATURES ENGINEERED: {total_features}")
        print("    " + "-" * 50)
        for category, info in self.feature_registry.items():
            print(f"      {category:.<35s} {info['count']:>3d} features")
        print("    " + "-" * 50)
        print(f"    {'TOTAL':.<35s} {total_features:>3d} features")
        print("=" * 70)

        return df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":                                        # Only run if executed directly

    # Print script header
    print("=" * 70)
    print("  NLP FEATURE ENGINEERING - STANDALONE TEST (ENHANCED)")
    print("  91 Features | 10 Categories | Gov Policy + News + Temporal + Aggregate")
    print("=" * 70)

    # Load company profiles data
    DATA_PATH = "data/processed/company_profiles.csv"
    print(f"\n  Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Initial shape: {df.shape}")

    # Initialize and run NLP feature engineer
    nlp_engineer = NLPFeatureEngineer()
    df_nlp = nlp_engineer.engineer_all_nlp_features(df)

    # Display results
    print(f"\n  Final shape: {df_nlp.shape}")
    print(f"  NLP features added: {df_nlp.shape[1] - 13}")

    # Show sample of ORIGINAL key NLP features
    key_original = [
        'company_name',
        'text_polarity',
        'flesch_reading_ease',
        'lexical_diversity',
        'total_esg_keyword_density',
        'greenwashing_signal_score',
        'vague_to_concrete_ratio'
    ]
    key_original = [c for c in key_original if c in df_nlp.columns]
    print(f"\n  Original NLP features (top 5):")
    print(df_nlp[key_original].head(5).to_string())

    # Show sample of NEW enhanced features
    key_new = [
        'company_name',
        'regulatory_readiness_score',                             # Gov policy (Cat 7)
        'policy_esg_gap',                                         # Gov policy gap (Cat 7)
        'narrative_credibility_index',                            # News intent (Cat 8)
        'news_greenwashing_signal',                               # News GW signal (Cat 8)
        'commitment_credibility_score',                           # Temporal (Cat 9)
        'temporal_greenwashing_signal',                           # Temporal GW (Cat 9)
        'esg_linguistic_credibility_index',                       # Aggregate ELCI (Cat 10)
        'multi_signal_greenwashing_score',                        # Multi-signal GW (Cat 10)
        'aggregate_esg_nlp_score'                                 # Final 0-100 score (Cat 10)
    ]
    key_new = [c for c in key_new if c in df_nlp.columns]
    print(f"\n  NEW Enhanced features (top 5):")
    print(df_nlp[key_new].head(5).to_string())

    # Print feature category summary
    print("\n  Feature Category Breakdown:")
    print("  " + "-" * 55)
    for category, info in nlp_engineer.feature_registry.items():
        print(f"    {category:.<40s} {info['count']:>3d} features")
    total = sum(v['count'] for v in nlp_engineer.feature_registry.values())
    print("  " + "-" * 55)
    print(f"    {'TOTAL':.<40s} {total:>3d} features")

    # Save NLP features
    OUTPUT_PATH = "data/processed/nlp_features.csv"
    df_nlp.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved NLP features to: {OUTPUT_PATH}")
    print("=" * 70)
