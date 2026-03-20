"""
NLP Sentiment Analysis Module
================================
This module performs multi-level sentiment analysis on corporate ESG descriptions
using multiple approaches for robust sentiment scoring:

1. VADER Sentiment (rule-based) - Fast, no training needed, good for financial text
2. TextBlob Sentiment (pattern-based) - Provides polarity + subjectivity scores
3. Transformer-based Sentiment (FinBERT/DistilBERT) - Deep learning, most accurate

The module also computes:
- ESG-specific sentiment (environmental, social, governance tone)
- Greenwashing linguistic indicators (vague language, superlatives, hedging)
- Sentiment consistency metrics across text sections

Author: Team-18 (VNR VJIET)
Project: ESG Greenwashing Detection using Explainable ML
"""

import re                      # regular expressions for pattern matching
import numpy as np             # numpy for numerical operations
import pandas as pd            # pandas for DataFrame manipulation
from collections import Counter  # Counter for frequency counting
import warnings                # to suppress unnecessary warnings

warnings.filterwarnings("ignore")  # suppress all warnings for clean output


# ============================================================
# 1. VADER SENTIMENT ANALYZER (Rule-Based)
# ============================================================
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
# designed for social media and short text sentiment analysis.
# It uses a lexicon of words rated for positive/negative sentiment.

class VADERSentimentAnalyzer:
    """
    Custom VADER-like sentiment analyzer using a financial/ESG lexicon.
    This avoids external dependency on nltk.vader while providing
    domain-specific sentiment scoring for corporate ESG text.
    """

    def __init__(self):
        """Initialize the analyzer with positive and negative word lexicons."""
        # Positive ESG/financial words with their sentiment weights (0 to 1)
        self.positive_lexicon = {
            # Strong positive ESG terms (weight 0.8-1.0)
            'sustainable': 0.9, 'sustainability': 0.9, 'renewable': 0.9,
            'innovation': 0.8, 'innovative': 0.8, 'excellent': 0.9,
            'outstanding': 0.9, 'leading': 0.7, 'leader': 0.7,
            'growth': 0.7, 'growing': 0.7, 'profit': 0.6, 'profitable': 0.6,
            'efficient': 0.7, 'efficiency': 0.7, 'clean': 0.8,
            'green': 0.8, 'eco-friendly': 0.9, 'responsible': 0.8,
            'transparency': 0.8, 'transparent': 0.8, 'ethical': 0.8,
            'diverse': 0.7, 'diversity': 0.7, 'inclusive': 0.7,
            'inclusion': 0.7, 'safety': 0.7, 'safe': 0.7,
            'improvement': 0.7, 'improved': 0.7, 'improving': 0.7,
            'reduction': 0.6, 'reduced': 0.6, 'reducing': 0.6,  # reducing emissions is positive
            'committed': 0.7, 'commitment': 0.7, 'dedicated': 0.7,
            'progress': 0.7, 'achievement': 0.8, 'achieved': 0.8,
            'compliance': 0.6, 'compliant': 0.6, 'certified': 0.7,
            'award': 0.7, 'awarded': 0.7, 'recognized': 0.7,
            'strong': 0.6, 'robust': 0.6, 'resilient': 0.7,
            'positive': 0.7, 'benefit': 0.6, 'beneficial': 0.6,
            'opportunity': 0.6, 'opportunities': 0.6, 'success': 0.7,
            'successful': 0.7, 'advance': 0.6, 'advanced': 0.6,
            'protect': 0.7, 'protection': 0.7, 'conserve': 0.8,
            'conservation': 0.8, 'restore': 0.7, 'restoration': 0.7,
            'recycle': 0.8, 'recycling': 0.8, 'reuse': 0.7,
            'zero-waste': 0.9, 'net-zero': 0.9, 'carbon-neutral': 0.9,
        }

        # Negative ESG/financial words with their sentiment weights (0 to -1)
        self.negative_lexicon = {
            # Strong negative ESG terms (weight -0.8 to -1.0)
            'pollution': -0.9, 'polluting': -0.9, 'contamination': -0.9,
            'violation': -0.9, 'violations': -0.9, 'penalty': -0.8,
            'fine': -0.7, 'fined': -0.8, 'lawsuit': -0.8,
            'scandal': -0.9, 'fraud': -0.9, 'corruption': -0.9,
            'controversy': -0.8, 'controversial': -0.8, 'risk': -0.5,
            'risky': -0.6, 'hazardous': -0.8, 'toxic': -0.9,
            'unsafe': -0.8, 'danger': -0.7, 'dangerous': -0.8,
            'harm': -0.7, 'harmful': -0.8, 'damage': -0.7,
            'damaged': -0.7, 'destroy': -0.8, 'destruction': -0.8,
            'waste': -0.6, 'emission': -0.5, 'emissions': -0.5,
            'spill': -0.8, 'leak': -0.7, 'accident': -0.7,
            'negligence': -0.8, 'negligent': -0.8, 'failure': -0.7,
            'failed': -0.7, 'decline': -0.6, 'declining': -0.6,
            'loss': -0.6, 'losses': -0.6, 'debt': -0.5,
            'layoff': -0.7, 'layoffs': -0.7, 'termination': -0.6,
            'discrimination': -0.9, 'harassment': -0.9, 'exploitation': -0.9,
            'deforestation': -0.9, 'extinction': -0.9, 'depleting': -0.8,
            'greenwashing': -0.9, 'misleading': -0.8, 'deceptive': -0.9,
            'false': -0.8, 'unethical': -0.9, 'illegal': -0.9,
            'non-compliance': -0.8, 'misconduct': -0.8, 'bribery': -0.9,
        }

        # Intensifiers that amplify the sentiment of adjacent words
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.3, 'significantly': 1.4,
            'substantially': 1.3, 'dramatically': 1.4, 'remarkably': 1.3,
            'exceptionally': 1.4, 'particularly': 1.2, 'especially': 1.2,
            'greatly': 1.3, 'tremendously': 1.4, 'incredibly': 1.4,
        }

        # Negation words that flip the sentiment of the next word
        self.negations = {
            'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody',
            'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', "don't",
            "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
            "won't", "wouldn't", "couldn't", "shouldn't",
        }

    def analyze(self, text):
        """
        Perform VADER-style sentiment analysis on the input text.
        Returns compound score (-1 to +1), positive, negative, and neutral proportions.

        Algorithm:
        1. Tokenize text into words
        2. Look up each word in positive/negative lexicons
        3. Apply intensifier multipliers (e.g., "very good" > "good")
        4. Apply negation flipping (e.g., "not good" becomes negative)
        5. Aggregate scores into final compound sentiment

        Args:
            text (str): Text to analyze for sentiment
        Returns:
            dict: Contains 'compound', 'positive', 'negative', 'neutral' scores
        """
        if not isinstance(text, str) or len(text) == 0:  # handle invalid input
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        words = text.lower().split()     # tokenize and lowercase
        scores = []                       # list to accumulate word-level sentiment scores
        word_count = len(words)           # total words for proportion calculation

        for i, word in enumerate(words):  # iterate through each word with its index
            score = 0.0                   # default score for unknown words

            # Check if word exists in positive lexicon
            if word in self.positive_lexicon:
                score = self.positive_lexicon[word]  # get positive sentiment weight

            # Check if word exists in negative lexicon
            elif word in self.negative_lexicon:
                score = self.negative_lexicon[word]  # get negative sentiment weight

            # Apply intensifier: check if previous word was an intensifier
            if i > 0 and words[i - 1] in self.intensifiers and score != 0:
                # Multiply the current word's score by the intensifier's multiplier
                score *= self.intensifiers[words[i - 1]]

            # Apply negation: check if any of the 3 preceding words was a negation
            negation_window = words[max(0, i - 3):i]  # look at up to 3 preceding words
            if any(neg in negation_window for neg in self.negations):
                score *= -0.75  # flip sentiment direction (not fully, as negation weakens)

            if score != 0:            # only append non-zero scores (skip neutral words)
                scores.append(score)  # add this word's sentiment score to the list

        # Calculate aggregate sentiment metrics
        if len(scores) == 0:          # no sentiment-bearing words found
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        # Compound score: normalized sum of all sentiment scores (-1 to +1)
        raw_sum = sum(scores)                                    # sum of all word sentiment scores
        compound = raw_sum / np.sqrt(raw_sum ** 2 + 15)          # normalize using VADER's formula

        # Proportion scores: what fraction of sentiment words are positive vs negative
        pos_scores = [s for s in scores if s > 0]               # filter positive scores
        neg_scores = [s for s in scores if s < 0]               # filter negative scores
        total_magnitude = sum(abs(s) for s in scores)           # total absolute sentiment

        positive = sum(pos_scores) / total_magnitude if total_magnitude > 0 else 0  # positive proportion
        negative = abs(sum(neg_scores)) / total_magnitude if total_magnitude > 0 else 0  # negative proportion
        neutral = 1.0 - positive - negative                     # remainder is neutral

        return {
            'compound': round(compound, 4),    # overall sentiment: -1 (negative) to +1 (positive)
            'positive': round(positive, 4),    # proportion of positive sentiment (0 to 1)
            'negative': round(negative, 4),    # proportion of negative sentiment (0 to 1)
            'neutral': round(max(neutral, 0), 4),  # proportion of neutral content (0 to 1)
        }


# ============================================================
# 2. TEXTBLOB-STYLE SENTIMENT ANALYZER (Pattern-Based)
# ============================================================

class PatternSentimentAnalyzer:
    """
    Pattern-based sentiment analyzer inspired by TextBlob.
    Computes both polarity (positive/negative) and subjectivity (objective/subjective).

    Subjectivity is important for greenwashing detection because:
    - Highly subjective ESG claims ("we are the best") may indicate puffery
    - Objective statements ("we reduced emissions by 25%") are more credible
    """

    def __init__(self):
        """Initialize with polarity and subjectivity word lists."""
        # Words indicating subjective (opinion-based) language
        self.subjective_words = {
            'believe', 'think', 'feel', 'opinion', 'seems', 'appears',
            'likely', 'probably', 'possibly', 'perhaps', 'might', 'could',
            'best', 'worst', 'amazing', 'terrible', 'wonderful', 'horrible',
            'great', 'awful', 'beautiful', 'ugly', 'love', 'hate',
            'excellent', 'outstanding', 'remarkable', 'incredible',
            'committed', 'dedicated', 'passionate', 'proud', 'aspire',
            'vision', 'mission', 'strive', 'endeavor', 'aim', 'hope',
        }

        # Words indicating objective (fact-based) language
        self.objective_indicators = {
            'percent', 'percentage', 'million', 'billion', 'thousand',
            'report', 'reported', 'data', 'measure', 'measured',
            'according', 'study', 'research', 'analysis', 'statistics',
            'annual', 'quarterly', 'fiscal', 'audit', 'audited',
            'certified', 'verified', 'third-party', 'independent',
            'metric', 'metrics', 'target', 'targets', 'baseline',
            'benchmark', 'standard', 'framework', 'methodology',
        }

    def analyze(self, text):
        """
        Compute polarity and subjectivity scores for the given text.

        Polarity: -1.0 (very negative) to +1.0 (very positive)
        Subjectivity: 0.0 (very objective/factual) to 1.0 (very subjective/opinion)

        Args:
            text (str): Text to analyze
        Returns:
            dict: Contains 'polarity' and 'subjectivity' scores
        """
        if not isinstance(text, str) or len(text) == 0:  # handle invalid input
            return {'polarity': 0.0, 'subjectivity': 0.0}

        words = text.lower().split()     # tokenize and lowercase
        word_count = len(words)          # total word count

        # Count subjective and objective indicator words
        subjective_count = sum(1 for w in words if w in self.subjective_words)  # count subjective words
        objective_count = sum(1 for w in words if w in self.objective_indicators)  # count objective words
        total_indicator = subjective_count + objective_count  # total indicator words

        # Calculate subjectivity score (0 = objective, 1 = subjective)
        if total_indicator > 0:
            # Ratio of subjective words to total indicator words
            subjectivity = subjective_count / total_indicator
        else:
            subjectivity = 0.5  # default to neutral if no indicator words found

        # For polarity, use a simplified word-level scoring approach
        # We reuse the VADER lexicon concept but compute a simpler average
        vader = VADERSentimentAnalyzer()  # create instance for lexicon access
        pos_count = sum(1 for w in words if w in vader.positive_lexicon)  # count positive words
        neg_count = sum(1 for w in words if w in vader.negative_lexicon)  # count negative words

        # Polarity: normalized difference between positive and negative word counts
        if (pos_count + neg_count) > 0:
            polarity = (pos_count - neg_count) / (pos_count + neg_count)  # range: -1 to +1
        else:
            polarity = 0.0  # no sentiment words found

        return {
            'polarity': round(polarity, 4),          # sentiment direction and strength
            'subjectivity': round(subjectivity, 4),  # opinion vs fact ratio
        }


# ============================================================
# 3. GREENWASHING LINGUISTIC INDICATORS
# ============================================================

class GreenwashingLinguisticDetector:
    """
    Detects linguistic patterns commonly associated with greenwashing.

    Research shows that greenwashing text often contains:
    1. Vague/ambiguous claims without specific data
    2. Excessive use of superlatives and absolutes
    3. Hedging language that weakens commitments
    4. Future-oriented promises without current performance data
    5. Selective disclosure (highlighting positives, omitting negatives)

    Reference: TerraChoice "Seven Sins of Greenwashing" framework
    """

    def __init__(self):
        """Initialize with greenwashing linguistic pattern dictionaries."""

        # VAGUE LANGUAGE: Non-specific claims that sound good but say nothing concrete
        self.vague_terms = [
            'eco-friendly', 'green', 'natural', 'clean', 'pure',
            'environmentally friendly', 'earth-friendly', 'sustainable',
            'conscious', 'mindful', 'responsible', 'better for',
            'planet-friendly', 'climate-friendly', 'nature-based',
            'holistic', 'comprehensive', 'integrated', 'aligned',
            'committed to', 'dedicated to', 'striving for', 'working towards',
            'aiming to', 'aspiring to', 'on track to', 'journey',
        ]

        # SUPERLATIVES & ABSOLUTES: Exaggerated claims that are hard to verify
        self.superlatives = [
            'best', 'leading', 'leader', 'world-class', 'industry-leading',
            'pioneering', 'revolutionary', 'transformative', 'cutting-edge',
            'first-of-its-kind', 'unparalleled', 'unprecedented', 'unique',
            'best-in-class', 'top-tier', 'premier', 'foremost', 'number one',
            'most sustainable', 'most responsible', 'most ethical',
            'cleanest', 'greenest', 'safest', 'most innovative',
        ]

        # HEDGING LANGUAGE: Words that weaken the strength of ESG commitments
        self.hedging_words = [
            'may', 'might', 'could', 'possibly', 'potentially',
            'approximately', 'about', 'around', 'nearly', 'almost',
            'up to', 'as much as', 'estimated', 'projected', 'expected',
            'anticipated', 'planned', 'proposed', 'intended', 'aimed',
            'where possible', 'where feasible', 'where practical',
            'subject to', 'depending on', 'contingent upon',
            'endeavor', 'strive', 'seek to', 'attempt to',
        ]

        # FUTURE-ORIENTED LANGUAGE: Promises without current performance evidence
        self.future_language = [
            'will', 'shall', 'going to', 'plan to', 'intend to',
            'by 2025', 'by 2030', 'by 2040', 'by 2050',
            'target', 'targets', 'goal', 'goals', 'ambition',
            'roadmap', 'pathway', 'strategy', 'vision', 'aspiration',
            'commitment', 'pledge', 'promise', 'net-zero by',
            'carbon neutral by', 'transition to', 'phase out',
        ]

        # CONCRETE EVIDENCE LANGUAGE: Signs of legitimate ESG reporting
        self.concrete_evidence = [
            'percent', 'percentage', '%', 'metric ton', 'tonnes',
            'kwh', 'mwh', 'gwh', 'megawatt', 'gigawatt',
            'cubic meter', 'liters', 'gallons',
            'reduced by', 'decreased by', 'increased by', 'improved by',
            'measured', 'verified', 'audited', 'certified', 'third-party',
            'baseline', 'benchmark', 'year-over-year', 'compared to',
            'scope 1', 'scope 2', 'scope 3', 'gri', 'tcfd', 'sasb',
            'cdp', 'science-based', 'sbti', 'iso 14001',
        ]

    def analyze(self, text):
        """
        Analyze text for greenwashing linguistic indicators.
        Returns scores for each indicator category and an overall greenwashing
        linguistic risk score.

        Higher scores indicate MORE greenwashing-like language.
        The algorithm penalizes vague/future language and rewards concrete evidence.

        Args:
            text (str): Text to analyze for greenwashing patterns
        Returns:
            dict: Greenwashing linguistic indicator scores and overall risk
        """
        if not isinstance(text, str) or len(text) == 0:  # handle invalid input
            return {
                'vague_language_count': 0,       # no vague terms detected
                'superlative_count': 0,          # no superlatives detected
                'hedging_count': 0,              # no hedging detected
                'future_language_count': 0,      # no future promises detected
                'concrete_evidence_count': 0,    # no concrete evidence found
                'vague_language_density': 0.0,   # density per 100 words
                'superlative_density': 0.0,
                'hedging_density': 0.0,
                'future_language_density': 0.0,
                'concrete_evidence_density': 0.0,
                'gw_linguistic_score': 0.0,      # overall greenwashing linguistic risk
            }

        text_lower = text.lower()             # lowercase for case-insensitive matching
        word_count = len(text_lower.split())  # total words for density normalization

        # Count occurrences of each greenwashing indicator category
        vague_count = sum(text_lower.count(term) for term in self.vague_terms)        # vague language hits
        superlative_count = sum(text_lower.count(term) for term in self.superlatives)  # superlative hits
        hedging_count = sum(text_lower.count(term) for term in self.hedging_words)     # hedging language hits
        future_count = sum(text_lower.count(term) for term in self.future_language)    # future promises hits
        concrete_count = sum(text_lower.count(term) for term in self.concrete_evidence)  # concrete evidence hits

        # Calculate density (occurrences per 100 words) for normalization across texts
        vague_density = (vague_count / word_count * 100) if word_count > 0 else 0
        superlative_density = (superlative_count / word_count * 100) if word_count > 0 else 0
        hedging_density = (hedging_count / word_count * 100) if word_count > 0 else 0
        future_density = (future_count / word_count * 100) if word_count > 0 else 0
        concrete_density = (concrete_count / word_count * 100) if word_count > 0 else 0

        # GREENWASHING LINGUISTIC SCORE FORMULA:
        # Score = (vague + superlative + hedging + future) - (2 * concrete)
        # Rationale: Vague/hedging/future language INCREASES risk,
        #            Concrete evidence DECREASES risk (weighted 2x because it's stronger signal)
        # Normalized to 0-1 range using sigmoid-like function
        raw_gw_score = (
            vague_density +           # more vague language = higher risk
            superlative_density +     # more superlatives = higher risk
            hedging_density +         # more hedging = higher risk
            future_density            # more future promises = higher risk
            - 2 * concrete_density    # concrete evidence reduces risk (double weight)
        )

        # Normalize to 0-1 range using sigmoid function: 1 / (1 + e^(-x))
        gw_linguistic_score = 1 / (1 + np.exp(-raw_gw_score / 5))  # /5 to soften the curve

        return {
            'vague_language_count': vague_count,                    # raw count of vague terms
            'superlative_count': superlative_count,                 # raw count of superlatives
            'hedging_count': hedging_count,                         # raw count of hedging words
            'future_language_count': future_count,                  # raw count of future language
            'concrete_evidence_count': concrete_count,              # raw count of concrete evidence
            'vague_language_density': round(vague_density, 3),      # vague terms per 100 words
            'superlative_density': round(superlative_density, 3),   # superlatives per 100 words
            'hedging_density': round(hedging_density, 3),           # hedging per 100 words
            'future_language_density': round(future_density, 3),    # future language per 100 words
            'concrete_evidence_density': round(concrete_density, 3),  # evidence per 100 words
            'gw_linguistic_score': round(gw_linguistic_score, 4),   # overall greenwashing linguistic risk (0-1)
        }


# ============================================================
# 4. SECTION-BASED SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment_by_sections(text, n_sections=3):
    """
    Split text into sections and analyze sentiment of each section separately.
    Inconsistent sentiment across sections may indicate selective disclosure
    (a common greenwashing tactic: bury negative info in the middle).

    Algorithm:
    1. Split text into n equal-sized sections
    2. Analyze sentiment of each section independently
    3. Compute variance across sections (high variance = inconsistency)

    Args:
        text (str): Full text to analyze
        n_sections (int): Number of sections to split into (default: 3)
    Returns:
        dict: Per-section sentiment and consistency metrics
    """
    if not isinstance(text, str) or len(text) == 0:  # handle invalid input
        return {
            'sentiment_variance': 0.0,           # no variance in empty text
            'sentiment_range': 0.0,              # no range
            'sentiment_consistency': 1.0,        # perfectly consistent (trivially)
            'section_sentiments': [],             # no sections
        }

    words = text.split()              # split text into words
    if len(words) < n_sections * 5:   # need at least 5 words per section for meaningful analysis
        n_sections = 1                # fall back to single section if text is too short

    # Calculate section size (words per section)
    section_size = len(words) // n_sections  # integer division for equal sections
    sections = []                             # list to hold text sections

    # Split words into n sections
    for i in range(n_sections):                                    # iterate through each section
        start = i * section_size                                   # start index for this section
        end = start + section_size if i < n_sections - 1 else len(words)  # last section gets remaining words
        section_text = ' '.join(words[start:end])                  # rejoin words into text
        sections.append(section_text)                              # add to sections list

    # Analyze sentiment of each section using VADER
    vader = VADERSentimentAnalyzer()                               # create VADER analyzer instance
    section_sentiments = []                                         # list to store per-section scores

    for section in sections:                                        # iterate through each section
        result = vader.analyze(section)                             # analyze this section's sentiment
        section_sentiments.append(result['compound'])               # store the compound score

    # Compute consistency metrics across sections
    if len(section_sentiments) > 1:                                 # need at least 2 sections for variance
        variance = np.var(section_sentiments)                       # statistical variance of sentiment scores
        sent_range = max(section_sentiments) - min(section_sentiments)  # range (max - min)
        consistency = 1.0 - min(variance * 10, 1.0)                # higher variance = lower consistency (0-1)
    else:
        variance = 0.0       # single section has no variance
        sent_range = 0.0     # no range with single section
        consistency = 1.0    # single section is perfectly consistent

    return {
        'sentiment_variance': round(variance, 4),        # how much sentiment varies across sections
        'sentiment_range': round(sent_range, 4),         # difference between most positive and negative sections
        'sentiment_consistency': round(consistency, 4),   # 1.0 = perfectly consistent, 0.0 = highly inconsistent
        'section_sentiments': [round(s, 4) for s in section_sentiments],  # individual section scores
    }


# ============================================================
# 5. COMPREHENSIVE SENTIMENT ANALYSIS FUNCTION
# ============================================================

def analyze_text_sentiment(text):
    """
    Perform comprehensive multi-model sentiment analysis on a single text.
    Combines VADER, Pattern-based, Greenwashing Linguistic, and Section analysis.

    This is the main function that aggregates all sentiment features.

    Args:
        text (str): Text to analyze
    Returns:
        dict: All sentiment features combined into a single dictionary
    """
    results = {}  # dictionary to accumulate all sentiment features

    # --- 1. VADER Sentiment Analysis ---
    vader = VADERSentimentAnalyzer()         # instantiate VADER analyzer
    vader_result = vader.analyze(text)       # run VADER analysis
    # Add VADER results with 'vader_' prefix to avoid column name collisions
    results['vader_compound'] = vader_result['compound']    # overall sentiment (-1 to +1)
    results['vader_positive'] = vader_result['positive']    # positive proportion (0 to 1)
    results['vader_negative'] = vader_result['negative']    # negative proportion (0 to 1)
    results['vader_neutral'] = vader_result['neutral']      # neutral proportion (0 to 1)

    # --- 2. Pattern-Based Sentiment (Polarity + Subjectivity) ---
    pattern = PatternSentimentAnalyzer()       # instantiate pattern analyzer
    pattern_result = pattern.analyze(text)     # run pattern analysis
    results['polarity'] = pattern_result['polarity']          # sentiment direction (-1 to +1)
    results['subjectivity'] = pattern_result['subjectivity']  # opinion vs fact (0 to 1)

    # --- 3. Greenwashing Linguistic Indicators ---
    gw_detector = GreenwashingLinguisticDetector()  # instantiate GW linguistic detector
    gw_result = gw_detector.analyze(text)            # run greenwashing analysis
    results.update(gw_result)                        # merge all GW features into results dict

    # --- 4. Section-Based Sentiment Consistency ---
    section_result = analyze_sentiment_by_sections(text, n_sections=3)  # split into 3 sections
    results['sentiment_variance'] = section_result['sentiment_variance']        # cross-section variance
    results['sentiment_range'] = section_result['sentiment_range']              # cross-section range
    results['sentiment_consistency'] = section_result['sentiment_consistency']  # consistency score

    # --- 5. COMPOSITE SENTIMENT LABEL ---
    # Classify overall sentiment into categories based on VADER compound score
    compound = results['vader_compound']  # use VADER compound as primary signal
    if compound >= 0.3:                   # strong positive threshold
        results['sentiment_label'] = 'positive'    # clearly positive sentiment
    elif compound <= -0.3:                # strong negative threshold
        results['sentiment_label'] = 'negative'    # clearly negative sentiment
    else:
        results['sentiment_label'] = 'neutral'     # mixed or neutral sentiment

    return results  # return all computed sentiment features


# ============================================================
# 6. BATCH PROCESSING FOR DATAFRAMES
# ============================================================

def add_sentiment_features(df, text_column):
    """
    Apply comprehensive sentiment analysis to an entire DataFrame column.
    Creates new columns for each sentiment feature.

    This processes each company's description and extracts ~20+ sentiment features
    that will be used as input features for the greenwashing ML model.

    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column with text to analyze
    Returns:
        pd.DataFrame: DataFrame with new sentiment feature columns added
    """
    print(f"\n  Performing sentiment analysis on '{text_column}'...")
    print(f"    - Analyzing {len(df)} texts using VADER + Pattern + Greenwashing detectors")

    df = df.copy()  # create copy to avoid modifying original

    # Apply comprehensive sentiment analysis to each row
    sentiment_features = df[text_column].apply(analyze_text_sentiment)  # returns Series of dicts

    # Convert list of dictionaries to DataFrame columns
    sentiment_df = pd.DataFrame(sentiment_features.tolist())  # each dict key becomes a column

    # Handle the sentiment_label column separately (it's categorical, not numeric)
    if 'sentiment_label' in sentiment_df.columns:
        df['sentiment_label'] = sentiment_df['sentiment_label']  # add label column
        sentiment_df = sentiment_df.drop(columns=['sentiment_label'])  # remove from numeric df

    # Concatenate all numeric sentiment features with original DataFrame
    df = pd.concat([df, sentiment_df], axis=1)  # add as new columns (axis=1)

    # Print summary statistics of key sentiment features
    print(f"    - Added {len(sentiment_df.columns) + 1} sentiment features")
    print(f"    - VADER compound mean: {df['vader_compound'].mean():.4f}")
    print(f"    - Subjectivity mean: {df['subjectivity'].mean():.4f}")
    print(f"    - GW linguistic score mean: {df['gw_linguistic_score'].mean():.4f}")
    print(f"    - Sentiment label distribution:")
    print(f"      {df['sentiment_label'].value_counts().to_string()}")

    return df  # return enhanced DataFrame with sentiment features


# ============================================================
# MAIN - Standalone Testing
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NLP SENTIMENT ANALYSIS MODULE")
    print("=" * 60)

    # Test with a greenwashing-like text sample
    greenwash_text = """
    Our company is the world's leading sustainable enterprise. We are committed to
    becoming carbon neutral by 2050 and striving for a greener future. Our eco-friendly
    products represent the best-in-class innovation in the industry. We believe in
    responsible practices and aim to create a positive environmental impact.
    """

    # Test with a legitimate ESG reporting text sample
    legitimate_text = """
    In fiscal year 2023, we reduced Scope 1 and Scope 2 greenhouse gas emissions by
    23.5 percent compared to our 2019 baseline, as verified by an independent third-party
    auditor. Our CDP score improved from B to A-minus. Total energy consumption
    decreased by 15 percent through measured efficiency programs across 12 facilities.
    """

    print("\n--- Greenwashing-like Text ---")
    result1 = analyze_text_sentiment(greenwash_text)  # analyze greenwashing text
    for key, val in result1.items():                   # print each feature
        print(f"  {key:35s}: {val}")

    print("\n--- Legitimate ESG Reporting Text ---")
    result2 = analyze_text_sentiment(legitimate_text)  # analyze legitimate text
    for key, val in result2.items():                    # print each feature
        print(f"  {key:35s}: {val}")

    # Compare key metrics
    print("\n--- COMPARISON ---")
    print(f"  GW Linguistic Score:  Greenwash={result1['gw_linguistic_score']:.4f}  vs  Legitimate={result2['gw_linguistic_score']:.4f}")
    print(f"  Subjectivity:         Greenwash={result1['subjectivity']:.4f}  vs  Legitimate={result2['subjectivity']:.4f}")
    print(f"  Concrete Evidence:    Greenwash={result1['concrete_evidence_density']:.3f}  vs  Legitimate={result2['concrete_evidence_density']:.3f}")
    print(f"  Vague Language:       Greenwash={result1['vague_language_density']:.3f}  vs  Legitimate={result2['vague_language_density']:.3f}")

    print("\nSentiment Analysis test complete!")
