"""
ESG Greenwashing Detection - Data Preprocessing Pipeline
=========================================================
This script loads, cleans, transforms, and merges all 4 datasets into a single
analysis-ready dataset for greenwashing detection using ML models.

Datasets:
1. Greenwashing_Score_Data.xlsx   - Greenwashing scores (target variable)
2. SP 500 ESG Risk Ratings.csv    - S&P 500 ESG risk ratings + descriptions
3. company_esg_financial_dataset.csv - ESG + financial metrics (11K rows)
4. final_data.csv                 - NIFTY50 ESG risk data with descriptions
"""

import pandas as pd       # pandas library for data manipulation and analysis
import numpy as np        # numpy library for numerical operations
import os                 # os module for file path and directory operations
import warnings           # warnings module to suppress unnecessary warnings

warnings.filterwarnings("ignore")  # suppress all warning messages for clean output

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # get the directory where this script is located
DATA_DIR = os.path.join(BASE_DIR, "data")              # path to the raw data folder
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")    # path to save cleaned/processed output files
os.makedirs(PROCESSED_DIR, exist_ok=True)              # create processed folder if it doesn't exist


# ============================================================
# 1. LOAD RAW DATASETS
# ============================================================
def load_datasets():
    """Load all 4 raw datasets from the data folder."""
    print("=" * 60)                    # print a separator line for readability
    print("STEP 1: Loading Raw Datasets")
    print("=" * 60)

    # Dataset 1: Greenwashing scores - this is our TARGET variable for ML models
    df_gw = pd.read_excel(os.path.join(DATA_DIR, "Greenwashing_Score_Data.xlsx"))  # read Excel file into DataFrame
    print(f"  [1] Greenwashing Scores     : {df_gw.shape[0]:>6} rows, {df_gw.shape[1]:>2} cols")  # print row and column count

    # Dataset 2: S&P 500 ESG Risk Ratings - contains ESG scores for 500 US companies
    df_sp500 = pd.read_csv(os.path.join(DATA_DIR, "SP 500 ESG Risk Ratings.csv"))  # read CSV file into DataFrame
    print(f"  [2] S&P 500 ESG Ratings     : {df_sp500.shape[0]:>6} rows, {df_sp500.shape[1]:>2} cols")

    # Dataset 3: Company ESG Financial - ESG scores with financial metrics over multiple years
    df_esg_fin = pd.read_csv(os.path.join(DATA_DIR, "company_esg_financial_dataset.csv"))  # read CSV file
    print(f"  [3] ESG Financial Dataset   : {df_esg_fin.shape[0]:>6} rows, {df_esg_fin.shape[1]:>2} cols")

    # Dataset 4: NIFTY50 ESG Data - ESG risk data for top 50 Indian companies
    df_nifty = pd.read_csv(os.path.join(DATA_DIR, "final_data.csv"))  # read CSV file
    print(f"  [4] NIFTY50 ESG Data        : {df_nifty.shape[0]:>6} rows, {df_nifty.shape[1]:>2} cols")

    return df_gw, df_sp500, df_esg_fin, df_nifty  # return all 4 DataFrames as a tuple


# ============================================================
# 2. CLEAN GREENWASHING DATASET
# ============================================================
def clean_greenwashing(df_gw):
    """Clean the greenwashing scores dataset (target variable)."""
    print("\n" + "=" * 60)                    # print separator with newline
    print("STEP 2: Cleaning Greenwashing Dataset")
    print("=" * 60)

    df = df_gw.copy()  # create a copy to avoid modifying the original DataFrame

    # Standardize company names: remove leading/trailing spaces and convert to uppercase
    df["COMPANY_NAME"] = df["COMPANY_NAME"].str.strip().str.upper()

    # Remove duplicate rows where same company appears multiple times for the same year
    before = len(df)  # store row count before deduplication
    df = df.drop_duplicates(subset=["COMPANY_NAME", "YEAR"], keep="first")  # keep first occurrence only
    print(f"  Removed {before - len(df)} duplicate rows")  # print how many duplicates were removed

    # Create binary greenwashing label: 1 if GW_SCORE >= 0.5 (greenwashing), 0 otherwise
    df["GW_LABEL"] = (df["GW_SCORE"] >= 0.5).astype(int)  # convert boolean to integer (0 or 1)

    # Create greenwashing risk categories by binning the continuous GW_SCORE into 4 groups
    df["GW_RISK_CATEGORY"] = pd.cut(
        df["GW_SCORE"],                                        # column to bin
        bins=[0, 0.25, 0.5, 0.75, 1.0],                       # bin edges: 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
        labels=["Low", "Medium", "High", "Very High"],         # human-readable labels for each bin
        include_lowest=True,                                   # include the lowest edge (0) in the first bin
    )

    # Print dataset statistics for verification
    print(f"  Unique companies: {df['COMPANY_NAME'].nunique()}")       # count of distinct companies
    print(f"  Year range: {df['YEAR'].min()} - {df['YEAR'].max()}")   # earliest and latest year
    print(f"  GW_LABEL distribution:\n{df['GW_LABEL'].value_counts().to_string()}")          # count of 0s and 1s
    print(f"  GW_RISK_CATEGORY distribution:\n{df['GW_RISK_CATEGORY'].value_counts().to_string()}")  # count per category

    return df  # return the cleaned DataFrame


# ============================================================
# 3. CLEAN S&P 500 ESG RISK RATINGS
# ============================================================
def clean_sp500(df_sp500):
    """Clean the S&P 500 ESG risk ratings dataset."""
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning S&P 500 ESG Risk Ratings")
    print("=" * 60)

    df = df_sp500.copy()  # create a copy to avoid modifying original data

    # Drop rows where the main ESG score column is missing (no useful data)
    before = len(df)  # store count before dropping
    df = df.dropna(subset=["Total ESG Risk score"])  # remove rows with NaN in ESG score
    print(f"  Dropped {before - len(df)} rows with no ESG scores")  # print count of dropped rows

    # Standardize company names: strip whitespace and convert to uppercase for consistent matching
    df["Name"] = df["Name"].str.strip().str.upper()

    # Clean 'Full Time Employees' column: remove commas (e.g., "3,157" -> "3157") and convert to number
    df["Full Time Employees"] = (
        df["Full Time Employees"]
        .astype(str)                          # convert to string first to handle mixed types
        .str.replace(",", "", regex=False)    # remove comma separators from numbers
        .str.strip()                          # remove any leading/trailing whitespace
    )
    df["Full Time Employees"] = pd.to_numeric(df["Full Time Employees"], errors="coerce")  # convert to numeric, NaN for invalid values

    # Encode ESG Risk Level as ordinal numbers (0=best to 4=worst) for ML models
    risk_level_map = {
        "Negligible": 0,  # lowest risk
        "Low": 1,         # low risk
        "Medium": 2,      # moderate risk
        "High": 3,        # high risk
        "Severe": 4,      # highest risk
    }
    df["ESG_Risk_Level_Encoded"] = df["ESG Risk Level"].map(risk_level_map)  # map text labels to numeric values

    # Encode Controversy Level as ordinal numbers (0=no controversy to 5=severe)
    controversy_map = {
        "None Controversy Level": 0,          # no controversies reported
        "Low Controversy Level": 1,           # minor controversies
        "Moderate Controversy Level": 2,      # moderate controversies
        "Significant Controversy Level": 3,   # significant controversies
        "High Controversy Level": 4,          # high-profile controversies
        "Severe Controversy Level": 5,        # severe/critical controversies
    }
    df["Controversy_Level_Encoded"] = df["Controversy Level"].map(controversy_map)  # map text to numbers

    # Extract numeric percentile value from text like "50th percentile" -> 50.0
    df["ESG_Risk_Percentile_Num"] = (
        df["ESG Risk Percentile"]
        .astype(str)                              # convert to string
        .str.extract(r"(\d+)", expand=False)      # extract first number using regex
        .astype(float)                            # convert extracted string to float
    )

    # Rename columns to lowercase with underscores for consistency across all datasets
    df = df.rename(columns={
        "Symbol": "symbol",                           # stock ticker symbol
        "Name": "company_name",                       # company name
        "Sector": "sector",                           # business sector (e.g., Technology)
        "Industry": "industry",                       # specific industry (e.g., Solar)
        "Description": "description",                 # company description text
        "Total ESG Risk score": "total_esg_risk_score",  # overall ESG risk score
        "Environment Risk Score": "env_risk_score",      # environmental risk component
        "Governance Risk Score": "gov_risk_score",       # governance risk component
        "Social Risk Score": "social_risk_score",        # social risk component
        "Controversy Score": "controversy_score",        # numerical controversy score
        "Full Time Employees": "employees",              # number of employees
    })

    # Select only the relevant columns, dropping Address and other unnecessary fields
    cols_keep = [
        "symbol", "company_name", "sector", "industry", "description",    # identification columns
        "employees", "total_esg_risk_score", "env_risk_score",            # numeric score columns
        "gov_risk_score", "social_risk_score", "controversy_score",       # risk score columns
        "ESG_Risk_Level_Encoded", "Controversy_Level_Encoded",            # encoded categorical columns
        "ESG_Risk_Percentile_Num",                                        # percentile column
    ]
    df = df[cols_keep]  # filter to keep only selected columns

    # Fill remaining missing numeric values with the median of each column
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # get all numeric column names
    for col in numeric_cols:                      # iterate through each numeric column
        median_val = df[col].median()             # calculate the median value
        n_filled = df[col].isnull().sum()         # count how many NaN values exist
        if n_filled > 0:                          # only fill if there are missing values
            df[col] = df[col].fillna(median_val)  # replace NaN with median
            print(f"  Filled {n_filled} NaNs in '{col}' with median ({median_val:.2f})")  # log the action

    print(f"  Final shape: {df.shape}")  # print final row x column dimensions
    return df  # return cleaned DataFrame


# ============================================================
# 4. CLEAN COMPANY ESG FINANCIAL DATASET
# ============================================================
def clean_esg_financial(df_esg_fin):
    """Clean the ESG financial dataset."""
    print("\n" + "=" * 60)
    print("STEP 4: Cleaning ESG Financial Dataset")
    print("=" * 60)

    df = df_esg_fin.copy()  # create a copy to preserve original data

    # Standardize company names to uppercase for consistent matching
    df["CompanyName"] = df["CompanyName"].str.strip().str.upper()

    # Fill missing GrowthRate values with 0.0 (first year of each company has no prior year to compare)
    n_null = df["GrowthRate"].isnull().sum()       # count missing values
    df["GrowthRate"] = df["GrowthRate"].fillna(0.0)  # replace NaN with 0.0
    print(f"  Filled {n_null} NaN GrowthRate values with 0.0")

    # Remove duplicate rows (same company + same year)
    before = len(df)  # store count before removing duplicates
    df = df.drop_duplicates(subset=["CompanyName", "Year"], keep="first")  # keep first occurrence
    print(f"  Removed {before - len(df)} duplicate rows")

    # FEATURE ENGINEERING: ESG component gap = difference between highest and lowest E/S/G sub-scores
    # A large gap indicates inconsistency across ESG dimensions (potential greenwashing signal)
    df["ESG_Component_Gap"] = (
        df[["ESG_Environmental", "ESG_Social", "ESG_Governance"]].max(axis=1)  # max of 3 sub-scores per row
        - df[["ESG_Environmental", "ESG_Social", "ESG_Governance"]].min(axis=1)  # minus min of 3 sub-scores
    )

    # FEATURE ENGINEERING: Carbon intensity = carbon emissions divided by revenue
    # Measures how much CO2 a company produces per unit of revenue (lower is better)
    df["Carbon_Intensity"] = df["CarbonEmissions"] / df["Revenue"].replace(0, np.nan)  # avoid division by zero
    df["Carbon_Intensity"] = df["Carbon_Intensity"].fillna(0)  # fill any resulting NaN with 0

    # FEATURE ENGINEERING: Energy intensity = energy consumption divided by revenue
    # Measures energy efficiency relative to company size
    df["Energy_Intensity"] = df["EnergyConsumption"] / df["Revenue"].replace(0, np.nan)  # avoid division by zero
    df["Energy_Intensity"] = df["Energy_Intensity"].fillna(0)  # fill NaN with 0

    # FEATURE ENGINEERING: Water intensity = water usage divided by revenue
    # Measures water efficiency relative to company size
    df["Water_Intensity"] = df["WaterUsage"] / df["Revenue"].replace(0, np.nan)  # avoid division by zero
    df["Water_Intensity"] = df["Water_Intensity"].fillna(0)  # fill NaN with 0

    # Rename all columns to lowercase with underscores for consistency
    df = df.rename(columns={
        "CompanyID": "company_id",                   # unique identifier for each company
        "CompanyName": "company_name",               # company name
        "Industry": "industry",                      # industry sector
        "Region": "region",                          # geographic region
        "Year": "year",                              # fiscal year
        "Revenue": "revenue",                        # annual revenue
        "ProfitMargin": "profit_margin",             # profit margin percentage
        "MarketCap": "market_cap",                   # market capitalization
        "GrowthRate": "growth_rate",                 # year-over-year growth rate
        "ESG_Overall": "esg_overall",                # overall ESG score (0-100)
        "ESG_Environmental": "esg_environmental",    # environmental sub-score
        "ESG_Social": "esg_social",                  # social sub-score
        "ESG_Governance": "esg_governance",          # governance sub-score
        "CarbonEmissions": "carbon_emissions",       # total carbon emissions
        "WaterUsage": "water_usage",                 # total water usage
        "EnergyConsumption": "energy_consumption",   # total energy consumption
        "ESG_Component_Gap": "esg_component_gap",    # gap between max and min E/S/G scores
        "Carbon_Intensity": "carbon_intensity",      # carbon emissions per unit revenue
        "Energy_Intensity": "energy_intensity",      # energy usage per unit revenue
        "Water_Intensity": "water_intensity",        # water usage per unit revenue
    })

    print(f"  Final shape: {df.shape}")  # print dimensions of cleaned dataset
    print(f"  New features added: esg_component_gap, carbon_intensity, energy_intensity, water_intensity")
    return df  # return cleaned DataFrame with new features


# ============================================================
# 5. CLEAN NIFTY50 ESG DATASET
# ============================================================
def clean_nifty50(df_nifty):
    """Clean the NIFTY50 ESG dataset."""
    print("\n" + "=" * 60)
    print("STEP 5: Cleaning NIFTY50 Dataset")
    print("=" * 60)

    df = df_nifty.copy()  # create a copy to preserve original data

    # Drop the empty unnamed column (artifact from CSV export with trailing comma)
    if "Unnamed: 13" in df.columns:                # check if the empty column exists
        df = df.drop(columns=["Unnamed: 13"])      # remove it

    # Standardize company names to uppercase for consistent matching
    df["company"] = df["company"].str.strip().str.upper()

    # Encode ESG Risk Level as ordinal numbers (0=Negligible to 4=Severe)
    risk_level_map = {
        "Negligible": 0, "Low": 1, "Medium": 2, "High": 3, "Severe": 4,  # ordinal mapping
    }
    df["esg_risk_level_encoded"] = df["esg_risk_level"].map(risk_level_map)  # apply mapping

    # Encode Controversy Level as ordinal numbers (0=None to 5=Severe)
    controversy_map = {
        "None Controversy Level": 0,          # no controversies
        "Low Controversy Level": 1,           # minor controversies
        "Moderate Controversy Level": 2,      # moderate controversies
        "Significant Controversy Level": 3,   # significant controversies
        "High Controversy Level": 4,          # high controversies
        "Severe Controversy Level": 5,        # most severe controversies
    }
    df["controversy_level_encoded"] = df["Controversy Level"].map(controversy_map)  # apply mapping

    # Encode ESG Risk Exposure level (Low=0, Medium=1, High=2)
    exposure_map = {"Low": 0, "Medium": 1, "High": 2}  # ordinal mapping for exposure
    df["esg_risk_exposure_encoded"] = df["esg_risk_exposure"].map(exposure_map)  # apply mapping

    # Encode ESG Risk Management quality (Weak=0, Average=1, Strong=2)
    mgmt_map = {"Weak": 0, "Average": 1, "Strong": 2}  # ordinal mapping for management quality
    df["esg_risk_management_encoded"] = df["esg_risk_management"].map(mgmt_map)  # apply mapping

    # FEATURE: Calculate how much the predicted future ESG score deviates from current score
    # Positive means ESG risk is expected to increase (worsening), negative means improving
    df["esg_score_deviation"] = df["predicted_future_esg_score"] - df["esg_risk_score_2024"]

    # Rename columns to lowercase with underscores for consistency
    df = df.rename(columns={
        "Symbol": "symbol",              # stock ticker symbol
        "company": "company_name",       # company name
        "Sector": "sector",             # business sector
        "Industry": "industry",          # specific industry
        "Description": "description",    # company description text for NLP
    })

    # Select only the relevant columns for further analysis
    cols_keep = [
        "symbol", "company_name", "sector", "industry", "description",              # identification columns
        "esg_risk_score_2024", "predicted_future_esg_score",                         # ESG score columns
        "esg_risk_level_encoded", "controversy_level_encoded",                       # encoded risk levels
        "esg_risk_exposure_encoded", "esg_risk_management_encoded",                  # encoded exposure & management
        "controversy_score", "esg_score_deviation",                                  # numeric scores
        "Material ESG Issues 1", "Material ESG Issues 2", "Material ESG Issues 3",  # top 3 material ESG issues
    ]
    df = df[cols_keep]  # filter to keep only selected columns

    print(f"  Final shape: {df.shape}")  # print final dimensions
    return df  # return cleaned DataFrame


# ============================================================
# 6. COMBINE SP500 + NIFTY50 INTO UNIFIED COMPANY PROFILE
# ============================================================
def create_company_profiles(df_sp500_clean, df_nifty_clean):
    """Merge S&P 500 and NIFTY50 into a unified company profile dataset."""
    print("\n" + "=" * 60)
    print("STEP 6: Creating Unified Company Profiles (S&P 500 + NIFTY50)")
    print("=" * 60)

    # Align NIFTY50 column names to match S&P 500 column naming structure
    nifty_aligned = df_nifty_clean.rename(columns={
        "esg_risk_score_2024": "total_esg_risk_score",             # rename to match S&P 500's column name
        "controversy_level_encoded": "Controversy_Level_Encoded",  # match S&P 500 encoding column name
        "esg_risk_level_encoded": "ESG_Risk_Level_Encoded",        # match S&P 500 encoding column name
    })

    # Extract relevant columns from S&P 500 dataset for the unified profile
    sp500_subset = df_sp500_clean[[
        "symbol", "company_name", "sector", "industry", "description",  # company identification
        "total_esg_risk_score", "env_risk_score", "gov_risk_score",     # ESG risk scores
        "social_risk_score", "controversy_score",                        # controversy data
        "ESG_Risk_Level_Encoded", "Controversy_Level_Encoded",           # encoded categories
    ]].copy()  # .copy() to avoid SettingWithCopyWarning
    sp500_subset["source"] = "SP500"  # add source column to track data origin

    # Extract relevant columns from NIFTY50 dataset for the unified profile
    nifty_subset = nifty_aligned[[
        "symbol", "company_name", "sector", "industry", "description",  # company identification
        "total_esg_risk_score", "controversy_score",                     # available scores
        "ESG_Risk_Level_Encoded", "Controversy_Level_Encoded",           # encoded categories
    ]].copy()  # .copy() to avoid SettingWithCopyWarning
    nifty_subset["source"] = "NIFTY50"  # add source column to track data origin

    # Concatenate both datasets vertically (stack rows) into one unified DataFrame
    company_profiles = pd.concat([sp500_subset, nifty_subset], ignore_index=True)  # reset index after concat

    # Fill any missing numeric values with the median of that column
    numeric_cols = company_profiles.select_dtypes(include=[np.number]).columns  # get all numeric columns
    for col in numeric_cols:  # iterate over each numeric column
        company_profiles[col] = company_profiles[col].fillna(company_profiles[col].median())  # fill NaN with median

    # Print summary statistics
    print(f"  S&P 500 companies: {len(sp500_subset)}")      # count of S&P 500 companies
    print(f"  NIFTY50 companies: {len(nifty_subset)}")      # count of NIFTY50 companies
    print(f"  Combined profiles: {len(company_profiles)}")  # total combined count

    return company_profiles  # return the unified company profiles DataFrame


# ============================================================
# 7. FEATURE ENGINEERING ON ESG FINANCIAL DATA
# ============================================================
def engineer_esg_financial_features(df_esg_fin_clean):
    """Create aggregated features from the ESG financial time-series data."""
    print("\n" + "=" * 60)
    print("STEP 7: Engineering Features from ESG Financial Data")
    print("=" * 60)

    df = df_esg_fin_clean.copy()  # create a copy to avoid modifying original

    # Define aggregation functions to compute per-company statistics from time-series data
    agg_funcs = {
        "revenue": ["mean", "std"],              # average and variability of revenue over years
        "profit_margin": ["mean", "std"],        # average and variability of profit margin
        "market_cap": ["mean"],                  # average market capitalization
        "growth_rate": ["mean"],                 # average year-over-year growth rate
        "esg_overall": ["mean", "std", "min", "max"],  # ESG score statistics (mean, spread, range)
        "esg_environmental": ["mean", "std"],    # environmental score mean and variability
        "esg_social": ["mean", "std"],           # social score mean and variability
        "esg_governance": ["mean", "std"],       # governance score mean and variability
        "carbon_emissions": ["mean", "std"],     # carbon emissions mean and variability
        "carbon_intensity": ["mean"],            # average carbon intensity over years
        "energy_intensity": ["mean"],            # average energy intensity over years
        "water_intensity": ["mean"],             # average water intensity over years
        "esg_component_gap": ["mean", "max"],    # average and worst ESG component gap
    }

    # Group by company name and compute all aggregation functions
    df_agg = df.groupby("company_name").agg(agg_funcs)  # apply aggregation per company
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns]  # flatten multi-level column names (e.g., "revenue_mean")
    df_agg = df_agg.reset_index()  # convert company_name from index back to a regular column

    # Get the most recent (latest) year's data for each company
    df_latest = df.sort_values("year").groupby("company_name").last().reset_index()  # sort by year, take last row per company
    df_latest = df_latest.rename(columns={
        "esg_overall": "esg_overall_latest",           # rename to indicate this is the most recent value
        "esg_environmental": "esg_env_latest",         # latest environmental score
        "esg_social": "esg_social_latest",             # latest social score
        "esg_governance": "esg_gov_latest",            # latest governance score
        "carbon_emissions": "carbon_emissions_latest", # latest carbon emissions
    })
    # Select only the columns we need from the latest year data
    df_latest = df_latest[["company_name", "industry", "region", "year",
                           "esg_overall_latest", "esg_env_latest",
                           "esg_social_latest", "esg_gov_latest",
                           "carbon_emissions_latest"]]

    # Merge the aggregated statistics with the latest year data on company name
    df_features = pd.merge(df_agg, df_latest, on="company_name", how="left")  # left join to keep all aggregated companies

    # FEATURE: ESG trend = latest ESG score minus historical average
    # Positive value means the company's ESG has improved over time
    df_features["esg_trend"] = df_features["esg_overall_latest"] - df_features["esg_overall_mean"]

    # FEATURE: ESG volatility = coefficient of variation (std / mean)
    # High volatility may indicate inconsistent ESG behavior (potential greenwashing signal)
    df_features["esg_volatility"] = (
        df_features["esg_overall_std"] / df_features["esg_overall_mean"].replace(0, np.nan)  # avoid division by zero
    ).fillna(0)  # fill NaN (from zero mean) with 0

    # Fill any remaining NaN values with 0 to ensure no missing data in final output
    df_features = df_features.fillna(0)

    print(f"  Aggregated features per company: {df_features.shape[1]} columns")  # print number of features created
    print(f"  Total companies: {df_features.shape[0]}")  # print number of companies

    return df_features  # return the feature-engineered DataFrame


# ============================================================
# 8. SAVE ALL PROCESSED DATASETS
# ============================================================
def save_processed(df_gw_clean, df_sp500_clean, df_esg_fin_features,
                   df_nifty_clean, company_profiles):
    """Save all cleaned and processed datasets to CSV files."""
    print("\n" + "=" * 60)
    print("STEP 8: Saving Processed Datasets")
    print("=" * 60)

    # Dictionary mapping output filenames to their corresponding DataFrames
    datasets = {
        "greenwashing_cleaned.csv": df_gw_clean,        # cleaned greenwashing target data
        "sp500_esg_cleaned.csv": df_sp500_clean,         # cleaned S&P 500 ESG data
        "esg_financial_features.csv": df_esg_fin_features,  # aggregated ESG financial features
        "nifty50_esg_cleaned.csv": df_nifty_clean,       # cleaned NIFTY50 ESG data
        "company_profiles.csv": company_profiles,         # unified company profiles (S&P500 + NIFTY50)
    }

    # Iterate through each dataset and save as CSV
    for filename, df in datasets.items():                         # loop through each filename-DataFrame pair
        path = os.path.join(PROCESSED_DIR, filename)              # construct full file path
        df.to_csv(path, index=False)                              # save to CSV without row index
        print(f"  Saved: {filename:40s} ({df.shape[0]:>6} rows, {df.shape[1]:>3} cols)")  # print confirmation

    return datasets  # return the dictionary of saved datasets


# ============================================================
# 9. GENERATE PREPROCESSING SUMMARY REPORT
# ============================================================
def print_summary(df_gw_clean, df_sp500_clean, df_esg_fin_features,
                  df_nifty_clean, company_profiles):
    """Print a summary of all preprocessed datasets."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    # Print a formatted table showing all datasets with their dimensions and status
    print(f"""
    Dataset                     Rows    Cols    Status
    -------                     ----    ----    ------
    Greenwashing (target)       {df_gw_clean.shape[0]:<8}{df_gw_clean.shape[1]:<8}Ready
    S&P 500 ESG Profiles        {df_sp500_clean.shape[0]:<8}{df_sp500_clean.shape[1]:<8}Ready
    ESG Financial Features      {df_esg_fin_features.shape[0]:<8}{df_esg_fin_features.shape[1]:<8}Ready
    NIFTY50 ESG Profiles        {df_nifty_clean.shape[0]:<8}{df_nifty_clean.shape[1]:<8}Ready
    Company Profiles (merged)   {company_profiles.shape[0]:<8}{company_profiles.shape[1]:<8}Ready

    Target variable: GW_SCORE (continuous 0-1) & GW_LABEL (binary 0/1)
    Output directory: {PROCESSED_DIR}
    """)


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main function that orchestrates the entire preprocessing pipeline."""
    print("\n" + "#" * 60)                         # print header separator
    print("  ESG GREENWASHING DETECTION - DATA PREPROCESSING")  # print pipeline title
    print("#" * 60)

    # STEP 1: Load all 4 raw datasets from the data folder
    df_gw, df_sp500, df_esg_fin, df_nifty = load_datasets()

    # STEP 2-5: Clean each dataset individually (handle missing values, duplicates, encoding)
    df_gw_clean = clean_greenwashing(df_gw)          # clean greenwashing target dataset
    df_sp500_clean = clean_sp500(df_sp500)            # clean S&P 500 ESG ratings
    df_esg_fin_clean = clean_esg_financial(df_esg_fin)  # clean ESG financial dataset
    df_nifty_clean = clean_nifty50(df_nifty)          # clean NIFTY50 ESG dataset

    # STEP 6: Merge S&P 500 and NIFTY50 into unified company profiles
    company_profiles = create_company_profiles(df_sp500_clean, df_nifty_clean)

    # STEP 7: Engineer aggregated features from ESG financial time-series data
    df_esg_fin_features = engineer_esg_financial_features(df_esg_fin_clean)

    # STEP 8: Save all processed datasets to CSV files in data/processed/
    save_processed(df_gw_clean, df_sp500_clean, df_esg_fin_features,
                   df_nifty_clean, company_profiles)

    # STEP 9: Print final summary report showing all dataset dimensions
    print_summary(df_gw_clean, df_sp500_clean, df_esg_fin_features,
                  df_nifty_clean, company_profiles)

    print("Preprocessing complete!\n")  # print completion message

    # Return all processed datasets as a dictionary for use in other scripts
    return {
        "greenwashing": df_gw_clean,           # cleaned greenwashing data with labels
        "sp500": df_sp500_clean,               # cleaned S&P 500 ESG profiles
        "esg_financial": df_esg_fin_features,  # aggregated ESG financial features
        "nifty50": df_nifty_clean,             # cleaned NIFTY50 ESG profiles
        "company_profiles": company_profiles,  # unified company profiles
    }


# Entry point: only runs when this script is executed directly (not when imported)
if __name__ == "__main__":
    main()  # execute the full preprocessing pipeline
