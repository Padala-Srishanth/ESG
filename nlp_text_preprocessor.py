"""
NLP Text Preprocessor Module
==============================
This module provides comprehensive text cleaning and preprocessing utilities
specifically designed for ESG (Environmental, Social, Governance) corporate text data.

It handles:
- Raw text cleaning (HTML removal, special characters, normalization)
- Tokenization and lemmatization using spaCy
- Stopword removal with domain-specific customization
- N-gram extraction for ESG-specific phrases
- Text statistics computation (readability, complexity metrics)

Author: Team-18 (VNR VJIET)
Project: ESG Greenwashing Detection using Explainable ML
"""

import re                      # regular expressions for pattern matching in text
import string                  # provides string constants like punctuation characters
import numpy as np             # numpy for numerical computations
import pandas as pd            # pandas for DataFrame operations
from collections import Counter  # Counter for counting word frequencies


# ============================================================
# 1. TEXT CLEANING FUNCTIONS
# ============================================================

def remove_html_tags(text):
    """
    Remove HTML tags from text using regex.
    Corporate reports often contain HTML artifacts from web scraping.

    Args:
        text (str): Raw text that may contain HTML tags
    Returns:
        str: Text with all HTML tags removed
    """
    if not isinstance(text, str):    # check if input is a valid string
        return ""                     # return empty string for non-string inputs (NaN, None)
    clean = re.compile('<.*?>')      # compile regex pattern to match any HTML tag (<...>)
    return re.sub(clean, '', text)   # substitute all HTML tags with empty string


def remove_urls(text):
    """
    Remove URLs (http, https, www) from text.
    Company descriptions often contain website links that add noise.

    Args:
        text (str): Text that may contain URLs
    Returns:
        str: Text with all URLs removed
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    # regex pattern matches http/https URLs and www. URLs
    url_pattern = r'https?://\S+|www\.\S+'  # \S+ matches any non-whitespace characters after the prefix
    return re.sub(url_pattern, '', text)     # replace all matched URLs with empty string


def remove_special_characters(text):
    """
    Remove special characters but keep essential punctuation (periods, commas).
    Preserves sentence structure while removing noise characters.

    Args:
        text (str): Text with potential special characters
    Returns:
        str: Cleaned text with only alphanumeric chars and basic punctuation
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    # keep letters, numbers, spaces, periods, commas, and hyphens
    pattern = r'[^a-zA-Z0-9\s\.\,\-]'  # caret ^ inside brackets means "NOT these characters"
    return re.sub(pattern, '', text)     # remove everything that doesn't match the allowed characters


def normalize_whitespace(text):
    """
    Replace multiple consecutive whitespace characters with a single space.
    Also strips leading and trailing whitespace.

    Args:
        text (str): Text with potential extra whitespace
    Returns:
        str: Text with normalized single spaces
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    return re.sub(r'\s+', ' ', text).strip()  # replace 2+ spaces/tabs/newlines with single space, then strip edges


def convert_to_lowercase(text):
    """
    Convert all text to lowercase for consistent processing.
    Essential for matching ESG keywords regardless of case.

    Args:
        text (str): Text in mixed case
    Returns:
        str: Text in all lowercase
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    return text.lower()              # convert entire string to lowercase


def remove_numbers(text):
    """
    Remove standalone numbers from text while keeping numbers within words.
    Removes financial figures that don't carry ESG semantic meaning.

    Args:
        text (str): Text containing numbers
    Returns:
        str: Text with standalone numbers removed
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    # \b means word boundary - only removes numbers that are standalone words
    return re.sub(r'\b\d+\.?\d*\b', '', text)  # matches integers and decimals


# ============================================================
# 2. MASTER TEXT CLEANING PIPELINE
# ============================================================

def clean_text(text):
    """
    Apply the full text cleaning pipeline in the correct order.
    Order matters: HTML removal first, then URLs, then special chars, etc.

    Pipeline order:
    1. Remove HTML tags (from web-scraped data)
    2. Remove URLs (website links)
    3. Convert to lowercase (for consistent matching)
    4. Remove special characters (noise reduction)
    5. Remove standalone numbers (financial figures)
    6. Normalize whitespace (clean formatting)

    Args:
        text (str): Raw text from corporate descriptions or reports
    Returns:
        str: Fully cleaned and normalized text ready for NLP processing
    """
    if not isinstance(text, str):        # guard against non-string inputs
        return ""
    text = remove_html_tags(text)        # step 1: strip any HTML tags
    text = remove_urls(text)             # step 2: remove web URLs
    text = convert_to_lowercase(text)    # step 3: lowercase everything
    text = remove_special_characters(text)  # step 4: remove special chars
    text = remove_numbers(text)          # step 5: remove standalone numbers
    text = normalize_whitespace(text)    # step 6: fix whitespace issues
    return text                          # return the fully cleaned text


# ============================================================
# 3. STOPWORD REMOVAL
# ============================================================

# Standard English stopwords list (most common words that don't carry meaning)
STANDARD_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'also', 'company', 'companies', 'inc', 'ltd', 'limited', 'corporation',
    'corp', 'group', 'llc', 'plc',  # corporate suffixes that don't carry ESG meaning
}

# ESG-specific terms that should NOT be removed (they carry domain meaning)
ESG_PRESERVE_WORDS = {
    'carbon', 'emission', 'emissions', 'climate', 'renewable', 'sustainability',
    'sustainable', 'green', 'environmental', 'social', 'governance', 'esg',
    'waste', 'pollution', 'biodiversity', 'diversity', 'inclusion', 'safety',
    'ethics', 'ethical', 'transparency', 'compliance', 'risk', 'water',
    'energy', 'solar', 'wind', 'recycling', 'circular', 'net-zero', 'netzero',
    'decarbonization', 'footprint', 'stakeholder', 'community', 'human rights',
    'labor', 'supply chain', 'corruption', 'bribery', 'whistleblower',
    'controversy', 'greenwashing', 'offset', 'neutral', 'positive', 'negative',
}


def remove_stopwords(text, preserve_esg_terms=True):
    """
    Remove stopwords from text while optionally preserving ESG domain terms.
    This ensures that important ESG-related words are never accidentally removed.

    Args:
        text (str): Cleaned text with stopwords
        preserve_esg_terms (bool): If True, ESG-specific words are never removed
    Returns:
        str: Text with stopwords removed
    """
    if not isinstance(text, str):    # guard against non-string inputs
        return ""
    words = text.split()             # split text into individual words

    # Build the effective stopword set
    stopwords = STANDARD_STOPWORDS.copy()  # start with standard stopwords
    if preserve_esg_terms:                  # if ESG preservation is enabled
        stopwords -= ESG_PRESERVE_WORDS     # remove ESG terms from stopwords so they're kept

    # Filter out stopwords, keeping only meaningful words
    filtered_words = [word for word in words if word not in stopwords]  # list comprehension filter
    return ' '.join(filtered_words)  # rejoin words into a single string


# ============================================================
# 4. TEXT STATISTICS & COMPLEXITY METRICS
# ============================================================

def compute_text_statistics(text):
    """
    Compute various text statistics that can serve as features for ML models.
    These metrics capture writing complexity, which may correlate with greenwashing
    (e.g., overly complex language to obscure actual ESG performance).

    Metrics computed:
    - word_count: Total number of words
    - char_count: Total number of characters
    - avg_word_length: Average word length (longer = more complex vocabulary)
    - sentence_count: Number of sentences
    - avg_sentence_length: Average words per sentence (longer = harder to read)
    - unique_word_ratio: Ratio of unique words to total (lexical diversity)
    - long_word_ratio: Proportion of words with 8+ characters (vocabulary complexity)

    Args:
        text (str): Input text to analyze
    Returns:
        dict: Dictionary containing all computed text statistics
    """
    if not isinstance(text, str) or len(text) == 0:  # handle empty or non-string input
        return {
            'word_count': 0,              # no words in empty text
            'char_count': 0,              # no characters
            'avg_word_length': 0,         # no average to compute
            'sentence_count': 0,          # no sentences
            'avg_sentence_length': 0,     # no average sentence length
            'unique_word_ratio': 0,       # no diversity to measure
            'long_word_ratio': 0,         # no long words
        }

    words = text.split()                  # split text into words
    sentences = re.split(r'[.!?]+', text)  # split text at sentence boundaries
    sentences = [s.strip() for s in sentences if s.strip()]  # remove empty sentences

    word_count = len(words)               # total number of words
    char_count = len(text)                # total number of characters including spaces
    unique_words = set(words)             # set of unique words (duplicates removed)

    # Calculate average word length: sum of all word lengths divided by count
    avg_word_length = np.mean([len(w) for w in words]) if words else 0

    # Count sentences (split by period, exclamation, question mark)
    sentence_count = len(sentences) if sentences else 1  # at least 1 sentence

    # Average sentence length: total words divided by number of sentences
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Unique word ratio: proportion of vocabulary that is unique (lexical diversity)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

    # Long word ratio: proportion of words with 8 or more characters
    long_words = [w for w in words if len(w) >= 8]  # filter words by length
    long_word_ratio = len(long_words) / word_count if word_count > 0 else 0

    return {
        'word_count': word_count,                        # total words in text
        'char_count': char_count,                        # total characters in text
        'avg_word_length': round(avg_word_length, 3),    # average length of each word
        'sentence_count': sentence_count,                # number of sentences detected
        'avg_sentence_length': round(avg_sentence_length, 3),  # average words per sentence
        'unique_word_ratio': round(unique_word_ratio, 3),      # lexical diversity score
        'long_word_ratio': round(long_word_ratio, 3),          # vocabulary complexity score
    }


# ============================================================
# 5. BATCH PROCESSING FOR DATAFRAMES
# ============================================================

def preprocess_dataframe_text(df, text_column, output_prefix="clean"):
    """
    Apply the full text preprocessing pipeline to a DataFrame column.
    Creates new columns for cleaned text and text statistics.

    This function:
    1. Cleans the raw text (HTML, URLs, special chars, etc.)
    2. Removes stopwords while preserving ESG terms
    3. Computes text statistics as new feature columns

    Args:
        df (pd.DataFrame): Input DataFrame containing text data
        text_column (str): Name of the column containing raw text
        output_prefix (str): Prefix for new column names (default: "clean")
    Returns:
        pd.DataFrame: DataFrame with new cleaned text and statistics columns
    """
    print(f"  Preprocessing text column: '{text_column}'")  # log which column is being processed

    df = df.copy()  # create a copy to avoid modifying the original DataFrame

    # Step 1: Apply full text cleaning pipeline to each row
    print(f"    - Cleaning text (removing HTML, URLs, special chars)...")
    df[f'{output_prefix}_text'] = df[text_column].apply(clean_text)  # apply clean_text to each row

    # Step 2: Remove stopwords while preserving ESG-specific terms
    print(f"    - Removing stopwords (preserving ESG domain terms)...")
    df[f'{output_prefix}_text_no_stopwords'] = df[f'{output_prefix}_text'].apply(
        lambda x: remove_stopwords(x, preserve_esg_terms=True)  # lambda applies function to each row
    )

    # Step 3: Compute text statistics for each description
    print(f"    - Computing text statistics (word count, complexity metrics)...")
    stats = df[f'{output_prefix}_text'].apply(compute_text_statistics)  # returns a Series of dicts
    stats_df = pd.DataFrame(stats.tolist())  # convert list of dicts into DataFrame columns

    # Rename statistics columns with the output prefix to avoid name collisions
    stats_df.columns = [f'{output_prefix}_{col}' for col in stats_df.columns]

    # Step 4: Concatenate the statistics columns to the original DataFrame
    df = pd.concat([df, stats_df], axis=1)  # axis=1 means add as new columns (not rows)

    print(f"    - Added {len(stats_df.columns)} text statistic features")
    print(f"    - Total columns now: {df.shape[1]}")

    return df  # return the enhanced DataFrame


# ============================================================
# 6. ESG KEYWORD FREQUENCY ANALYSIS
# ============================================================

# Comprehensive ESG keyword dictionary organized by E, S, G pillars
ESG_KEYWORD_DICT = {
    # Environmental keywords - related to planet and environmental impact
    'environmental': [
        'carbon', 'emission', 'emissions', 'climate', 'renewable', 'solar',
        'wind', 'energy', 'green', 'sustainable', 'sustainability', 'pollution',
        'waste', 'recycling', 'biodiversity', 'deforestation', 'water',
        'fossil', 'fuel', 'net-zero', 'netzero', 'decarbonization',
        'footprint', 'circular', 'eco', 'organic', 'clean', 'conservation',
        'environment', 'environmental', 'greenhouse', 'ghg', 'methane',
        'ozone', 'toxic', 'hazardous', 'contamination', 'reforestation',
    ],
    # Social keywords - related to people and community impact
    'social': [
        'diversity', 'inclusion', 'equity', 'safety', 'health', 'community',
        'employee', 'employees', 'labor', 'labour', 'human rights', 'rights',
        'welfare', 'wellbeing', 'training', 'education', 'philanthropy',
        'volunteer', 'stakeholder', 'indigenous', 'gender', 'equality',
        'discrimination', 'harassment', 'child labor', 'forced labor',
        'privacy', 'data protection', 'customer', 'satisfaction', 'fair',
        'living wage', 'benefits', 'workplace', 'occupational',
    ],
    # Governance keywords - related to corporate leadership and ethics
    'governance': [
        'governance', 'ethics', 'ethical', 'compliance', 'transparency',
        'accountability', 'board', 'director', 'directors', 'independent',
        'audit', 'corruption', 'bribery', 'whistleblower', 'regulation',
        'regulatory', 'oversight', 'shareholder', 'fiduciary', 'integrity',
        'misconduct', 'fraud', 'conflict of interest', 'insider', 'lobbying',
        'political', 'tax', 'executive', 'compensation', 'remuneration',
        'voting', 'proxy', 'disclosure', 'reporting', 'material',
    ],
}


def compute_esg_keyword_frequencies(text):
    """
    Count occurrences of ESG keywords in text, organized by E, S, G pillars.
    Returns both raw counts and density (counts per 100 words) for each pillar.

    Higher keyword density in a specific pillar may indicate emphasis on that
    ESG dimension. Greenwashing companies may have disproportionately high
    environmental keyword density compared to actual performance.

    Args:
        text (str): Cleaned text to analyze for ESG keywords
    Returns:
        dict: Dictionary with counts and densities for each ESG pillar
    """
    if not isinstance(text, str) or len(text) == 0:  # handle empty/invalid input
        return {
            'env_keyword_count': 0,      # no environmental keywords found
            'social_keyword_count': 0,   # no social keywords found
            'gov_keyword_count': 0,      # no governance keywords found
            'env_keyword_density': 0.0,  # density = count per 100 words
            'social_keyword_density': 0.0,
            'gov_keyword_density': 0.0,
            'total_esg_keyword_count': 0,   # sum of all ESG keywords
            'esg_keyword_density': 0.0,     # overall ESG density
            'dominant_esg_pillar': 'none',  # which pillar has the most keywords
        }

    text_lower = text.lower()             # convert to lowercase for case-insensitive matching
    word_count = len(text_lower.split())  # count total words for density calculation

    results = {}      # dictionary to store results
    pillar_counts = {}  # store counts per pillar for comparison

    # Iterate through each ESG pillar and count keyword matches
    for pillar, keywords in ESG_KEYWORD_DICT.items():  # pillar = 'environmental', 'social', 'governance'
        count = 0                         # initialize counter for this pillar
        for keyword in keywords:          # check each keyword in the pillar's list
            # Count non-overlapping occurrences of the keyword in the text
            count += text_lower.count(keyword.lower())  # case-insensitive count
        pillar_counts[pillar] = count     # store the count for this pillar

        # Create short prefix names for the output dictionary
        prefix = pillar[:3] if pillar != 'social' else 'social'  # 'env', 'social', 'gov'
        results[f'{prefix}_keyword_count'] = count  # raw count of keywords

        # Calculate density: keywords per 100 words (normalized by text length)
        density = (count / word_count * 100) if word_count > 0 else 0.0
        results[f'{prefix}_keyword_density'] = round(density, 3)  # round to 3 decimal places

    # Calculate total ESG keywords across all three pillars
    total_count = sum(pillar_counts.values())          # sum of all pillar counts
    results['total_esg_keyword_count'] = total_count   # store total count
    results['esg_keyword_density'] = round(            # overall ESG keyword density
        (total_count / word_count * 100) if word_count > 0 else 0.0, 3
    )

    # Determine which ESG pillar is most dominant in the text
    if total_count > 0:
        # Find the pillar with the highest keyword count
        dominant = max(pillar_counts, key=pillar_counts.get)  # get key with max value
        results['dominant_esg_pillar'] = dominant              # store the dominant pillar name
    else:
        results['dominant_esg_pillar'] = 'none'  # no ESG keywords found at all

    return results  # return all keyword frequency metrics


def add_esg_keyword_features(df, text_column):
    """
    Add ESG keyword frequency features to a DataFrame.
    Applies compute_esg_keyword_frequencies to each row.

    Args:
        df (pd.DataFrame): Input DataFrame with text data
        text_column (str): Column name containing text to analyze
    Returns:
        pd.DataFrame: DataFrame with new ESG keyword feature columns
    """
    print(f"  Computing ESG keyword frequencies from '{text_column}'...")

    df = df.copy()  # avoid modifying original DataFrame

    # Apply keyword frequency computation to each row's text
    keyword_features = df[text_column].apply(compute_esg_keyword_frequencies)  # returns Series of dicts
    keyword_df = pd.DataFrame(keyword_features.tolist())  # convert to DataFrame

    # Concatenate the new keyword features with the original DataFrame
    df = pd.concat([df, keyword_df], axis=1)  # add as new columns

    print(f"    - Added {len(keyword_df.columns)} ESG keyword features")
    return df  # return enhanced DataFrame


# ============================================================
# MAIN - Standalone testing
# ============================================================
if __name__ == "__main__":
    # Test the text preprocessor with a sample ESG description
    print("=" * 60)
    print("TESTING NLP TEXT PREPROCESSOR")
    print("=" * 60)

    sample_text = """
    <p>Acme Corp Inc. is committed to <b>sustainability</b> and reducing carbon emissions
    by 50% by 2030. Visit https://www.acmecorp.com for more details.
    The company has invested $500 million in renewable energy projects
    and achieved a 25% reduction in greenhouse gas emissions.
    Our diversity & inclusion programs have expanded to cover 100% of employees.</p>
    """

    print(f"\n--- Original Text ---\n{sample_text}")                    # show raw input
    cleaned = clean_text(sample_text)                                    # apply cleaning pipeline
    print(f"\n--- Cleaned Text ---\n{cleaned}")                         # show cleaned output
    no_stop = remove_stopwords(cleaned)                                  # remove stopwords
    print(f"\n--- After Stopword Removal ---\n{no_stop}")               # show without stopwords
    stats = compute_text_statistics(cleaned)                             # compute text stats
    print(f"\n--- Text Statistics ---\n{stats}")                        # show statistics
    esg_freq = compute_esg_keyword_frequencies(cleaned)                  # compute ESG keywords
    print(f"\n--- ESG Keyword Frequencies ---\n{esg_freq}")             # show ESG keyword metrics
    print("\nText Preprocessor test complete!")
