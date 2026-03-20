"""
NLP Master Pipeline
=====================
This is the master orchestration script that runs the complete NLP pipeline
for ESG Greenwashing Detection. It chains together:

1. Text Preprocessing  (nlp_text_preprocessor.py)
2. Sentiment Analysis  (nlp_sentiment_analysis.py)
3. ESG Claim Extraction (nlp_esg_claim_extraction.py)

Input:  Cleaned datasets from data/processed/ (output of data_preprocessing.py)
Output: NLP-enriched datasets in data/processed/ with all text features

Pipeline Flow:
    Raw Text → Clean Text → Text Stats → Sentiment Scores → ESG Keywords
    → Claim Extraction → Claim Metrics → Final NLP Feature Set

Author: Team-18 (VNR VJIET)
Project: ESG Greenwashing Detection using Explainable ML
"""

import os                      # os module for file and directory path operations
import time                    # time module for measuring execution duration
import pandas as pd            # pandas for DataFrame operations
import numpy as np             # numpy for numerical operations
import warnings                # warnings module to suppress unnecessary warnings

# Import our custom NLP modules from the same project directory
from nlp_text_preprocessor import (
    preprocess_dataframe_text,     # function to clean text in a DataFrame column
    add_esg_keyword_features,      # function to compute ESG keyword frequencies
)
from nlp_sentiment_analysis import (
    add_sentiment_features,        # function to perform multi-model sentiment analysis
)
from nlp_esg_claim_extraction import (
    extract_claims_from_dataframe,  # function to extract and score ESG claims
    generate_claim_report,          # function to generate human-readable claim report
)

warnings.filterwarnings("ignore")  # suppress all warnings for clean output


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # directory where this script lives
DATA_DIR = os.path.join(BASE_DIR, "data")                   # raw data directory
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")         # processed data directory
os.makedirs(PROCESSED_DIR, exist_ok=True)                   # create output dir if it doesn't exist


# ============================================================
# 1. LOAD PROCESSED DATASETS
# ============================================================
def load_processed_data():
    """
    Load the cleaned datasets from the preprocessing step.
    These datasets contain company descriptions that will be analyzed by NLP.

    Returns:
        tuple: (sp500_df, nifty50_df, company_profiles_df)
    """
    print("=" * 70)
    print("STEP 1: Loading Processed Datasets")
    print("=" * 70)

    # Load S&P 500 ESG dataset (430 companies with descriptions)
    sp500_path = os.path.join(PROCESSED_DIR, "sp500_esg_cleaned.csv")  # construct file path
    df_sp500 = pd.read_csv(sp500_path)                                  # read CSV into DataFrame
    print(f"  Loaded S&P 500 ESG data:    {df_sp500.shape[0]} rows, {df_sp500.shape[1]} cols")

    # Load NIFTY50 ESG dataset (50 Indian companies with descriptions)
    nifty50_path = os.path.join(PROCESSED_DIR, "nifty50_esg_cleaned.csv")
    df_nifty50 = pd.read_csv(nifty50_path)
    print(f"  Loaded NIFTY50 ESG data:    {df_nifty50.shape[0]} rows, {df_nifty50.shape[1]} cols")

    # Load unified company profiles (S&P500 + NIFTY50 merged)
    profiles_path = os.path.join(PROCESSED_DIR, "company_profiles.csv")
    df_profiles = pd.read_csv(profiles_path)
    print(f"  Loaded Company Profiles:    {df_profiles.shape[0]} rows, {df_profiles.shape[1]} cols")

    # Verify that description column exists and has data
    for name, df in [("SP500", df_sp500), ("NIFTY50", df_nifty50), ("Profiles", df_profiles)]:
        if 'description' in df.columns:                                   # check column exists
            non_null = df['description'].notna().sum()                    # count non-null descriptions
            print(f"    {name} has {non_null}/{len(df)} non-null descriptions")
        else:
            print(f"    WARNING: {name} has no 'description' column!")    # warn if column is missing

    return df_sp500, df_nifty50, df_profiles  # return all three DataFrames


# ============================================================
# 2. RUN TEXT PREPROCESSING
# ============================================================
def run_text_preprocessing(df, dataset_name):
    """
    Apply text preprocessing to a dataset's description column.
    Cleans text, removes stopwords, and computes text statistics.

    Args:
        df (pd.DataFrame): DataFrame with 'description' column
        dataset_name (str): Name of the dataset (for logging)
    Returns:
        pd.DataFrame: DataFrame with cleaned text and text statistic columns
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Text Preprocessing - {dataset_name}")
    print(f"{'=' * 70}")

    start_time = time.time()  # record start time for performance measurement

    # Check if description column exists and has non-null values
    if 'description' not in df.columns:              # guard against missing column
        print(f"  WARNING: No 'description' column found in {dataset_name}. Skipping.")
        return df  # return unchanged DataFrame

    # Count and handle missing descriptions
    null_count = df['description'].isnull().sum()    # count NaN descriptions
    if null_count > 0:
        print(f"  Found {null_count} null descriptions - filling with empty string")
        df['description'] = df['description'].fillna('')  # replace NaN with empty string

    # Apply the full text preprocessing pipeline
    # This adds: clean_text, clean_text_no_stopwords, and 7 text statistic columns
    df = preprocess_dataframe_text(df, text_column='description', output_prefix='desc')

    elapsed = time.time() - start_time  # calculate elapsed time
    print(f"  Preprocessing completed in {elapsed:.2f} seconds")
    print(f"  DataFrame shape: {df.shape}")

    return df  # return enhanced DataFrame


# ============================================================
# 3. RUN ESG KEYWORD ANALYSIS
# ============================================================
def run_esg_keyword_analysis(df, dataset_name):
    """
    Compute ESG keyword frequencies for each company description.
    Counts Environmental, Social, and Governance keywords and their densities.

    Args:
        df (pd.DataFrame): DataFrame with cleaned text column
        dataset_name (str): Name of the dataset (for logging)
    Returns:
        pd.DataFrame: DataFrame with ESG keyword feature columns
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 3: ESG Keyword Analysis - {dataset_name}")
    print(f"{'=' * 70}")

    start_time = time.time()  # record start time

    # Use the cleaned text (with stopwords removed) for keyword analysis
    # This gives more accurate keyword density since common words are removed
    text_col = 'desc_text_no_stopwords' if 'desc_text_no_stopwords' in df.columns else 'description'

    # Apply ESG keyword frequency computation
    df = add_esg_keyword_features(df, text_column=text_col)

    elapsed = time.time() - start_time  # calculate elapsed time
    print(f"  Keyword analysis completed in {elapsed:.2f} seconds")

    return df  # return DataFrame with keyword features


# ============================================================
# 4. RUN SENTIMENT ANALYSIS
# ============================================================
def run_sentiment_analysis(df, dataset_name):
    """
    Perform multi-model sentiment analysis on company descriptions.
    Uses VADER, Pattern-based, Greenwashing Linguistic, and Section analysis.

    Args:
        df (pd.DataFrame): DataFrame with cleaned text column
        dataset_name (str): Name of the dataset (for logging)
    Returns:
        pd.DataFrame: DataFrame with sentiment feature columns
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 4: Sentiment Analysis - {dataset_name}")
    print(f"{'=' * 70}")

    start_time = time.time()  # record start time

    # Use the cleaned text (lowercase, no HTML/URLs) for sentiment analysis
    text_col = 'desc_text' if 'desc_text' in df.columns else 'description'

    # Apply comprehensive sentiment analysis (VADER + Pattern + GW Linguistic + Sections)
    df = add_sentiment_features(df, text_column=text_col)

    elapsed = time.time() - start_time  # calculate elapsed time
    print(f"  Sentiment analysis completed in {elapsed:.2f} seconds")

    return df  # return DataFrame with sentiment features


# ============================================================
# 5. RUN ESG CLAIM EXTRACTION
# ============================================================
def run_claim_extraction(df, dataset_name):
    """
    Extract ESG claims from company descriptions and compute claim metrics.
    Identifies specific ESG claims, classifies them by pillar, and scores strength.

    Args:
        df (pd.DataFrame): DataFrame with description column
        dataset_name (str): Name of the dataset (for logging)
    Returns:
        tuple: (enhanced_df with claim metrics, detailed_claims_df)
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 5: ESG Claim Extraction - {dataset_name}")
    print(f"{'=' * 70}")

    start_time = time.time()  # record start time

    # Use original description (not cleaned) for claim extraction
    # because patterns need original casing and punctuation
    df, claims_df = extract_claims_from_dataframe(df, text_column='description')

    elapsed = time.time() - start_time  # calculate elapsed time
    print(f"  Claim extraction completed in {elapsed:.2f} seconds")

    return df, claims_df  # return enhanced DataFrame and detailed claims


# ============================================================
# 6. SAVE NLP-ENRICHED DATASETS
# ============================================================
def save_nlp_results(df_sp500, df_nifty50, df_profiles,
                     claims_sp500, claims_nifty50):
    """
    Save all NLP-enriched datasets and claim details to CSV files.

    Args:
        df_sp500: S&P 500 DataFrame with all NLP features
        df_nifty50: NIFTY50 DataFrame with all NLP features
        df_profiles: Company profiles DataFrame with all NLP features
        claims_sp500: Detailed claims from S&P 500 companies
        claims_nifty50: Detailed claims from NIFTY50 companies
    """
    print(f"\n{'=' * 70}")
    print(f"STEP 6: Saving NLP-Enriched Datasets")
    print(f"{'=' * 70}")

    # Dictionary mapping filenames to DataFrames for batch saving
    datasets = {
        "sp500_nlp_features.csv": df_sp500,           # S&P 500 with all NLP features
        "nifty50_nlp_features.csv": df_nifty50,       # NIFTY50 with all NLP features
        "company_profiles_nlp.csv": df_profiles,       # Unified profiles with NLP features
    }

    # Save main datasets
    for filename, df in datasets.items():                              # iterate through each dataset
        path = os.path.join(PROCESSED_DIR, filename)                   # construct full file path
        df.to_csv(path, index=False)                                   # save to CSV without row index
        print(f"  Saved: {filename:40s} ({df.shape[0]:>5} rows, {df.shape[1]:>3} cols)")

    # Save detailed claims DataFrames (every individual claim as a row)
    if not claims_sp500.empty:                                          # only save if claims exist
        claims_sp500_path = os.path.join(PROCESSED_DIR, "claims_sp500_detailed.csv")
        claims_sp500.to_csv(claims_sp500_path, index=False)            # save S&P 500 claims
        print(f"  Saved: {'claims_sp500_detailed.csv':40s} ({len(claims_sp500):>5} claims)")

    if not claims_nifty50.empty:                                        # only save if claims exist
        claims_nifty50_path = os.path.join(PROCESSED_DIR, "claims_nifty50_detailed.csv")
        claims_nifty50.to_csv(claims_nifty50_path, index=False)        # save NIFTY50 claims
        print(f"  Saved: {'claims_nifty50_detailed.csv':40s} ({len(claims_nifty50):>5} claims)")

    # Save combined claims for complete analysis
    all_claims = pd.concat([claims_sp500, claims_nifty50], ignore_index=True)  # merge all claims
    if not all_claims.empty:
        all_claims_path = os.path.join(PROCESSED_DIR, "all_esg_claims.csv")
        all_claims.to_csv(all_claims_path, index=False)                # save all claims combined
        print(f"  Saved: {'all_esg_claims.csv':40s} ({len(all_claims):>5} claims)")

    return all_claims  # return combined claims DataFrame


# ============================================================
# 7. PRINT NLP PIPELINE SUMMARY
# ============================================================
def print_nlp_summary(df_sp500, df_nifty50, df_profiles, all_claims):
    """
    Print a comprehensive summary of the NLP pipeline results.
    Shows feature counts, key statistics, and data quality metrics.

    Args:
        df_sp500: S&P 500 DataFrame with NLP features
        df_nifty50: NIFTY50 DataFrame with NLP features
        df_profiles: Company profiles with NLP features
        all_claims: All extracted ESG claims
    """
    print(f"\n{'=' * 70}")
    print("NLP PIPELINE SUMMARY")
    print(f"{'=' * 70}")

    # Dataset dimensions after NLP enrichment
    print(f"""
    Dataset                          Rows    Cols    NLP Features Added
    -------                          ----    ----    ------------------
    S&P 500 (NLP-enriched)           {df_sp500.shape[0]:<8}{df_sp500.shape[1]:<8}{df_sp500.shape[1] - 14}
    NIFTY50 (NLP-enriched)           {df_nifty50.shape[0]:<8}{df_nifty50.shape[1]:<8}{df_nifty50.shape[1] - 16}
    Company Profiles (NLP-enriched)  {df_profiles.shape[0]:<8}{df_profiles.shape[1]:<8}{df_profiles.shape[1] - 13}
    ESG Claims Extracted             {len(all_claims):<8}{'--':8s}--
    """)

    # Key NLP feature categories
    print("  NLP Feature Categories:")
    print("    1. Text Statistics     : word_count, char_count, avg_word_length, etc.")
    print("    2. ESG Keywords        : env/social/gov keyword counts and densities")
    print("    3. VADER Sentiment     : compound, positive, negative, neutral scores")
    print("    4. Pattern Sentiment   : polarity, subjectivity scores")
    print("    5. GW Linguistics      : vague language, superlatives, hedging densities")
    print("    6. Section Consistency : sentiment variance, range, consistency")
    print("    7. ESG Claims          : claim counts, strength, credibility, verification")

    # Key statistics for greenwashing detection
    if 'gw_linguistic_score' in df_profiles.columns:
        print(f"\n  Key Greenwashing Indicators (Company Profiles):")
        print(f"    GW Linguistic Score - Mean: {df_profiles['gw_linguistic_score'].mean():.4f}, "
              f"Std: {df_profiles['gw_linguistic_score'].std():.4f}")
    if 'subjectivity' in df_profiles.columns:
        print(f"    Subjectivity        - Mean: {df_profiles['subjectivity'].mean():.4f}, "
              f"Std: {df_profiles['subjectivity'].std():.4f}")
    if 'claim_credibility_score' in df_profiles.columns:
        print(f"    Claim Credibility   - Mean: {df_profiles['claim_credibility_score'].mean():.4f}, "
              f"Std: {df_profiles['claim_credibility_score'].std():.4f}")

    print(f"\n  Output directory: {PROCESSED_DIR}")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """
    Main function that orchestrates the complete NLP pipeline.
    Runs all NLP processing steps in sequence on all datasets.
    """
    print("\n" + "#" * 70)
    print("  ESG GREENWASHING DETECTION - NLP PIPELINE")
    print("#" * 70)

    pipeline_start = time.time()  # record overall pipeline start time

    # ---- STEP 1: Load processed datasets ----
    df_sp500, df_nifty50, df_profiles = load_processed_data()

    # ---- STEP 2: Text Preprocessing (cleaning + statistics) ----
    df_sp500 = run_text_preprocessing(df_sp500, "S&P 500")       # preprocess S&P 500 descriptions
    df_nifty50 = run_text_preprocessing(df_nifty50, "NIFTY50")   # preprocess NIFTY50 descriptions
    df_profiles = run_text_preprocessing(df_profiles, "Company Profiles")  # preprocess profiles

    # ---- STEP 3: ESG Keyword Analysis ----
    df_sp500 = run_esg_keyword_analysis(df_sp500, "S&P 500")     # compute keyword frequencies for S&P500
    df_nifty50 = run_esg_keyword_analysis(df_nifty50, "NIFTY50")  # compute keywords for NIFTY50
    df_profiles = run_esg_keyword_analysis(df_profiles, "Company Profiles")  # compute for profiles

    # ---- STEP 4: Sentiment Analysis ----
    df_sp500 = run_sentiment_analysis(df_sp500, "S&P 500")       # run sentiment on S&P 500
    df_nifty50 = run_sentiment_analysis(df_nifty50, "NIFTY50")   # run sentiment on NIFTY50
    df_profiles = run_sentiment_analysis(df_profiles, "Company Profiles")  # run sentiment on profiles

    # ---- STEP 5: ESG Claim Extraction ----
    df_sp500, claims_sp500 = run_claim_extraction(df_sp500, "S&P 500")     # extract claims from S&P500
    df_nifty50, claims_nifty50 = run_claim_extraction(df_nifty50, "NIFTY50")  # extract from NIFTY50
    df_profiles, claims_profiles = run_claim_extraction(df_profiles, "Company Profiles")  # extract from profiles

    # ---- STEP 6: Save all NLP-enriched datasets ----
    all_claims = save_nlp_results(df_sp500, df_nifty50, df_profiles,
                                  claims_sp500, claims_nifty50)

    # ---- Generate and save claim analysis report ----
    if not all_claims.empty:
        report = generate_claim_report(all_claims, top_n=5)  # generate top 5 examples report
        report_path = os.path.join(PROCESSED_DIR, "esg_claim_report.txt")  # save path
        with open(report_path, 'w', encoding='utf-8') as f:               # open file for writing
            f.write(report)                                                 # write report to file
        print(f"\n  Claim report saved to: esg_claim_report.txt")

    # ---- STEP 7: Print Summary ----
    print_nlp_summary(df_sp500, df_nifty50, df_profiles, all_claims)

    pipeline_end = time.time()  # record pipeline end time
    total_time = pipeline_end - pipeline_start  # calculate total duration
    print(f"\n  Total NLP pipeline time: {total_time:.2f} seconds")
    print(f"\nNLP Pipeline complete!\n")

    # Return all enriched datasets for use by downstream ML models
    return {
        "sp500_nlp": df_sp500,                # S&P 500 with all NLP features
        "nifty50_nlp": df_nifty50,            # NIFTY50 with all NLP features
        "profiles_nlp": df_profiles,          # company profiles with all NLP features
        "all_claims": all_claims,             # all individual ESG claims extracted
    }


# Entry point: runs when script is executed directly
if __name__ == "__main__":
    results = main()  # execute the full NLP pipeline and store results
