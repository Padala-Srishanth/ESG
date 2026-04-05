# ESG Greenwashing Detection using NLP & Machine Learning

A comprehensive machine learning pipeline that detects corporate greenwashing by analyzing ESG (Environmental, Social, Governance) scores, financial metrics, and company description text using Natural Language Processing (NLP) techniques.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Implementation Plan](#implementation-plan)
4. [Project Architecture](#project-architecture)
5. [Detailed File Breakdowns](#detailed-file-breakdowns)
6. [Datasets](#datasets)
7. [Feature Engineering Summary](#feature-engineering-summary)
8. [New Features (Advanced Dashboard Modules)](#new-features-advanced-dashboard-modules)
    - [Company Search & Analysis (with Reason Engine)](#1-company-search--analysis-with-greenwashing-reason-engine)
    - [Real-Time Intelligence](#2-real-time-intelligence)
    - [ESG Report Analyzer (AI)](#3-esg-report-analyzer-ai-powered)
    - [Report Generator](#4-report-generator)
    - [Advanced Explainability](#5-advanced-explainability-beyond-shap)
    - [Company Comparison](#6-company-comparison)
    - [Time-Series Risk Tracking](#7-time-series-risk-tracking)
    - [Greenwashing Reason Engine](#8-greenwashing-reason-engine-embedded-in-company-search)
9. [How to Run](#how-to-run)
10. [Dependencies](#dependencies)

---

## Project Overview

Greenwashing is the practice of making misleading claims about a company's environmental or social practices. This project builds an end-to-end ML pipeline that:

- Ingests ESG scores, financial data, and company descriptions from multiple sources
- Cleans and preprocesses raw data into analysis-ready formats
- Applies NLP techniques (sentiment analysis, readability metrics, greenwashing linguistic signals) to company text
- Engineers 121+ domain-driven features across numerical, NLP, and categorical categories
- Produces a final ML-ready feature matrix (480 companies x 169 columns) for greenwashing classification

---

## Problem Statement

Traditional ESG ratings can be manipulated. Companies may:
- Use vague language ("committed to sustainability") without concrete actions
- Show imbalanced ESG profiles (high governance scores but poor environmental performance)
- Have high controversy levels despite claiming low ESG risk

This project combines quantitative ESG analysis with NLP-based text analysis to detect these patterns and flag potential greenwashing.

---

## Implementation Plan

The project follows a 7-phase pipeline:

```
Phase 1: Data Collection
    Download ESG datasets from Kaggle (S&P 500, NIFTY 50, Greenwashing, Financial)

Phase 2: Data Preprocessing
    Clean, normalize, merge datasets into unified company profiles

Phase 3: NLP Text Analysis
    Sentiment analysis, ESG claim extraction from company descriptions

Phase 4: Feature Engineering
    Engineer 121 features across numerical, NLP, and categorical categories

Phase 5: Model Training
    Train 6 ML models: XGBoost, Gradient Boosting, Random Forest, Logistic Regression,
    SVM, Isolation Forest with 5-fold cross-validation and GridSearchCV

Phase 6: Evaluation & Explainability
    ROC curves, PR curves, confusion matrices, SHAP explanations

Phase 7: Risk Scoring & Dashboard
    0-100 greenwashing risk score per company + interactive Streamlit dashboard
```

### Current Progress

| Phase | Status | Output |
|-------|--------|--------|
| Phase 1: Data Collection | Complete | 4 raw datasets in `data/` |
| Phase 2: Data Preprocessing | Complete | 5 cleaned files in `data/processed/` |
| Phase 3: NLP Text Analysis | Complete | NLP features + ESG claims extracted |
| Phase 4: Feature Engineering | Complete | 480 x 169 feature matrix |
| Phase 5: Model Training | Complete | 6 models trained, best F1 = 0.9682 |
| Phase 6: Evaluation & Explainability | Complete | 8 plots + SHAP explanations |
| Phase 7: Risk Scoring & Dashboard | Complete | 480 companies scored 0-100, Streamlit UI |

---

## Project Architecture

```
ESG/
|
|-- README.md                              # Project documentation (this file)
|-- PIPELINE_REPORT.txt                    # Full pipeline execution report
|-- hamara mini...pdf                      # Project specification document
|
|-- data/                                  # Raw datasets (downloaded from Kaggle)
|   |-- Greenwashing_Score_Data.xlsx       # Corporate greenwashing accusations (595 records)
|   |-- SP 500 ESG Risk Ratings.csv        # S&P 500 ESG risk ratings (430 companies)
|   |-- company_esg_financial_dataset.csv  # ESG & financial performance (1000 records)
|   |-- final_data.csv                     # NIFTY 50 ESG score data (50 companies)
|   |
|   |-- processed/                         # Cleaned and engineered data (25 files)
|       |-- company_profiles.csv           # Merged company profiles (480 companies)
|       |-- feature_matrix.csv             # FINAL ML-ready feature matrix (480 x 169)
|       |-- greenwashing_risk_scores.csv   # Final 0-100 risk scores (ranked)
|       |-- model_metrics.csv              # All model comparison metrics
|       |-- predictions.csv                # Company-level model predictions
|       |-- shap_explanations.txt          # SHAP company-level explanations
|       |-- training_report.txt            # Model training summary
|       |-- evaluation_report.txt          # Model evaluation details
|       |-- risk_score_summary.txt         # Risk scoring summary
|       |-- all_esg_claims.csv             # Extracted ESG claims from descriptions
|       |-- feature_registry.csv           # Metadata for all 121 engineered features
|       |-- feature_importance_*.csv       # Feature importances per model
|       |-- (+ 13 more intermediate files)
|
|-- plots/                                 # Visualization outputs (8 plots)
|   |-- roc_curves.png                     # ROC curves for all models
|   |-- precision_recall_curves.png        # Precision-Recall curves
|   |-- confusion_matrices.png             # Confusion matrix heatmaps (2x2 grid)
|   |-- model_comparison.png               # Grouped bar chart of all metrics
|   |-- feature_importance.png             # Top 20 features bar chart
|   |-- shap_summary.png                   # SHAP beeswarm plot
|   |-- shap_bar.png                       # SHAP mean absolute importance
|   |-- shap_dependence.png                # SHAP dependence plot (top feature)
|
|-- data_preprocessing.py                  # Phase 2: Data cleaning and merging
|-- nlp_text_preprocessor.py               # Phase 3: Text cleaning and tokenization
|-- nlp_sentiment_analysis.py              # Phase 3: VADER + pattern sentiment analysis
|-- nlp_esg_claim_extraction.py            # Phase 3: ESG claim detection and classification
|-- nlp_pipeline.py                        # Phase 3: Master NLP orchestration pipeline
|-- feature_engineering_numerical.py       # Phase 4: Numerical feature engineering (43 features)
|-- feature_engineering_nlp.py             # Phase 4: NLP text feature engineering (47 features)
|-- feature_engineering_categorical.py     # Phase 4: Categorical feature engineering (31 features)
|-- feature_engineering_pipeline.py        # Phase 4: Master feature engineering pipeline
|-- model_training.py                      # Phase 5: 6-model training with GridSearchCV
|-- model_evaluation.py                    # Phase 6: ROC, PR, confusion matrix plots
|-- model_explainability.py                # Phase 6: SHAP values and explanations
|-- risk_scoring.py                        # Phase 7: 0-100 greenwashing risk score
|-- streamlit_dashboard.py                 # Phase 7: Interactive Streamlit web dashboard
|-- model_pipeline.py                      # Master pipeline (runs everything end-to-end)
```

---

## Detailed File Breakdowns

---

### `data_preprocessing.py` -- Complete Breakdown

This file is the entry point of the entire pipeline. It takes 4 raw messy datasets and transforms them into clean, analysis-ready CSV files.

#### The Core Idea

You start with 4 completely different datasets:
```
Greenwashing_Score_Data.xlsx       → 595 rows, 4 cols (target variable)
SP 500 ESG Risk Ratings.csv        → 430 rows, mixed formats
company_esg_financial_dataset.csv  → 11,000+ rows, time-series data
final_data.csv                     → 50 rows, NIFTY50 companies
```

These datasets have different column names, different formats, missing values, and duplicates. This script standardizes everything into one unified format.

#### Step 1: Load Raw Datasets -- Lines 33-55

**What it does:** Reads all 4 raw files from the `data/` folder using pandas `read_csv` and `read_excel`.

**Why it matters:** Each dataset uses different file formats (CSV vs XLSX) and has different structures. This step validates that all files exist and reports their dimensions.

#### Step 2: Clean Greenwashing Dataset -- Lines 61-94

**What it does:**
- Standardizes company names to UPPERCASE (so "Apple Inc" and "APPLE INC" match)
- Removes duplicate rows (same company + same year)
- Creates binary label: `GW_LABEL = 1` if GW_SCORE >= 0.5, else 0
- Creates risk categories by binning: Low (0-0.25), Medium (0.25-0.5), High (0.5-0.75), Very High (0.75-1.0)

**Why it matters:** The greenwashing score is our TARGET variable. The binary label enables classification models, and risk categories enable stratified analysis.

#### Step 3: Clean S&P 500 ESG Ratings -- Lines 100-189

**What it does:**
- Drops rows with missing ESG scores (no usable data)
- Cleans employee count: removes commas ("3,157" -> 3157) and converts to numbers
- Encodes ESG Risk Level as ordinal: Negligible=0, Low=1, Medium=2, High=3, Severe=4
- Encodes Controversy Level: None=0, Low=1, Moderate=2, Significant=3, High=4, Severe=5
- Extracts numeric percentile from text: "50th percentile" -> 50.0
- Renames all columns to lowercase_underscore format
- Fills remaining missing numeric values with column median

**Why ordinal encoding matters:**
```
"Negligible" and "Severe" have an inherent ORDER (Negligible < Severe)
ML models need numbers, not text, so we map: Negligible=0 ... Severe=4
This preserves the ordering relationship that one-hot encoding would lose
```

#### Step 4: Clean ESG Financial Dataset -- Lines 195-264

**What it does:**
- Fills missing GrowthRate with 0.0 (first year has no prior year comparison)
- Removes duplicate company+year rows
- Engineers 3 new intensity features:
  - `Carbon_Intensity = CarbonEmissions / Revenue` (CO2 per dollar of revenue)
  - `Energy_Intensity = EnergyConsumption / Revenue` (energy per dollar)
  - `Water_Intensity = WaterUsage / Revenue` (water per dollar)
- Engineers `ESG_Component_Gap = max(E,S,G) - min(E,S,G)` (pillar imbalance)
- Renames all columns to lowercase_underscore format

**Why intensity features matter:**
```
Company A: Carbon = 1,000,000 tonnes, Revenue = $50 billion → Intensity = 0.00002
Company B: Carbon = 100,000 tonnes, Revenue = $1 billion    → Intensity = 0.0001

Company A emits MORE total carbon but is 5x MORE EFFICIENT per dollar
Raw emissions are misleading without size normalization
```

#### Step 5: Clean NIFTY50 Dataset -- Lines 270-335

**What it does:**
- Drops empty `Unnamed: 13` column (CSV export artifact)
- Encodes ESG Risk Level, Controversy Level (same ordinal scheme as S&P 500)
- Encodes ESG Risk Exposure: Low=0, Medium=1, High=2
- Encodes ESG Risk Management quality: Weak=0, Average=1, Strong=2
- Computes `esg_score_deviation = predicted_future - current_2024` (is ESG getting worse?)
- Preserves Material ESG Issues 1, 2, 3 columns (for categorical feature engineering later)

#### Step 6: Create Unified Company Profiles -- Lines 341-384

**What it does:**
- Aligns NIFTY50 column names to match S&P 500 naming convention
- Adds a `source` column ("SP500" or "NIFTY50") to track data origin
- Concatenates both datasets vertically: 430 + 50 = 480 companies
- Fills any remaining NaN with column median

**Why we merge:** The ML model trains on ALL 480 companies together. More data = better generalization. The `source` column lets us analyze regional differences later.

#### Step 7: Engineer Financial Features -- Lines 390-454

**What it does:** Transforms the 11,000-row time-series financial dataset into 1 row per company using aggregation:

| Aggregation | Columns | What It Captures |
|-------------|---------|-----------------|
| mean | revenue, profit_margin, ESG scores | Average performance over time |
| std | revenue, profit_margin, ESG scores | Stability/volatility over time |
| min, max | esg_overall | Best and worst ESG performance |
| last | esg_overall, carbon_emissions | Most recent values |

Then creates 2 derived features:
- `esg_trend = latest_ESG - mean_ESG` (positive = improving over time)
- `esg_volatility = std_ESG / mean_ESG` (coefficient of variation = consistency)

**Why volatility matters for greenwashing:**
```
Company A: ESG scores = [60, 62, 58, 61, 59] → volatility = 0.03 (stable, consistent)
Company B: ESG scores = [40, 70, 30, 80, 55] → volatility = 0.35 (erratic, suspicious)

High volatility = company's ESG performance fluctuates wildly = unreliable
```

#### Step 8-9: Save & Report -- Lines 460-556

Saves 5 cleaned files to `data/processed/` and prints a summary table.

**Input:** 4 raw datasets totaling ~12,000 rows
**Output:** 5 cleaned files, the main one being `company_profiles.csv` (480 x 13)

---

### `nlp_text_preprocessor.py` -- Complete Breakdown

This file provides the text cleaning foundation that all NLP analysis builds on. Raw company descriptions contain noise (HTML, URLs, special characters) that must be removed before analysis.

#### The Core Idea

Raw company descriptions look like this:
```
"<p>Apple Inc.&reg; designs, manufactures... visit https://apple.com for more info!!!</p>"
```

After cleaning:
```
"apple inc designs manufactures"
```

Clean text gives accurate word counts, keyword matches, and sentiment scores.

#### Text Cleaning Functions -- Lines 29-127

6 individual cleaning functions, each targeting a specific type of noise:

| Function | Line | What It Removes | Regex Used |
|----------|------|-----------------|------------|
| `remove_html_tags()` | 29 | `<p>`, `<br>`, `</div>` etc. | `<.*?>` |
| `remove_urls()` | 45 | `https://...` and `www.` links | `https?://\S+\|www\.\S+` |
| `remove_special_characters()` | 62 | `@`, `#`, `&reg;`, etc. | `[^a-zA-Z0-9\s\.\,\-]` |
| `normalize_whitespace()` | 79 | Multiple spaces/tabs/newlines | `\s+` |
| `convert_to_lowercase()` | 94 | Uppercase letters | `.lower()` |
| `remove_numbers()` | 109 | All digits (optional) | `\d+` |

The master `clean_text()` function (Line 129) chains these in sequence.

#### Stopword Removal -- Lines 163-195

**What it does:** Removes common English words that carry no meaning:

```python
STANDARD_STOPWORDS = {'the', 'is', 'at', 'which', 'on', 'a', 'an', ...}  # 120+ words
```

**But preserves ESG-critical words:**
```python
ESG_PRESERVE_WORDS = {'not', 'no', 'nor', 'never', 'against', 'without', ...}
```

**Why "not" is preserved:** "Not sustainable" means the OPPOSITE of "sustainable". If we remove "not", the sentence becomes "sustainable" -- completely wrong.

#### Text Statistics -- Lines 227-295

Computes 7 metrics per company description:

| Metric | Formula | Why It Matters |
|--------|---------|---------------|
| `word_count` | len(words) | Text length normalization |
| `char_count` | len(text) | Character-level length |
| `avg_word_length` | total_chars / word_count | Complexity indicator |
| `sentence_count` | count of `.!?` | Document structure |
| `avg_sentence_length` | words / sentences | Readability proxy |
| `unique_word_ratio` | unique / total | Lexical diversity |
| `stopword_ratio` | stopwords / total | Content density |

#### ESG Keyword Analysis -- Lines 350-475

Curated lexicons of ESG terms across 3 pillars:

```python
ESG_KEYWORD_DICT = {
    'environmental': ['carbon', 'emission', 'renewable', 'solar', 'wind', ...],  # 25+ terms
    'social':        ['employee', 'safety', 'diversity', 'community', ...],       # 25+ terms
    'governance':    ['board', 'compliance', 'transparency', 'ethics', ...],       # 25+ terms
}
```

For each company, counts keywords per pillar and computes density (keywords / total words).

**Input:** Raw text strings
**Output:** Cleaned text + 7 text statistics + 6 keyword features per company

---

### `nlp_sentiment_analysis.py` -- Complete Breakdown

This file performs multi-method sentiment analysis using 4 distinct techniques, producing ~20 sentiment features per company.

#### The Core Idea

Greenwashing text has specific sentiment patterns:
- **Overly positive** (more positive than genuine sustainability reports)
- **Highly subjective** (opinions instead of facts)
- **Uses specific greenwashing language** (vague, hedging, superlatives)
- **Inconsistent tone** (positive at start, negative buried in middle)

4 analyzers detect these patterns from different angles.

#### Analyzer 1: VADER Sentiment -- Lines 36-184

**What it is:** A rule-based sentiment analyzer using a dictionary of 100+ ESG-specific words rated for positive/negative sentiment.

**How the algorithm works (Line 114-184):**
```
Step 1: Tokenize text into words
Step 2: Look up each word in positive/negative lexicons
Step 3: Apply intensifier multipliers
Step 4: Apply negation flipping
Step 5: Aggregate into compound score (-1 to +1)
```

**The lexicons (Lines 46-96):**

| Word | Score | Reason |
|------|-------|--------|
| `sustainable` | +0.9 | Strong positive ESG term |
| `innovation` | +0.8 | Positive performance indicator |
| `pollution` | -0.9 | Strong negative environmental term |
| `greenwashing` | -0.9 | Direct negative ESG term |

**Intensifiers (Lines 99-104):** Words that amplify sentiment:
```
"very sustainable"   → 0.9 x 1.3 = 1.17 (stronger positive)
"extremely toxic"    → -0.9 x 1.5 = -1.35 (stronger negative)
```

**Negations (Lines 107-112):** Words that flip sentiment:
```
"not sustainable"    → 0.9 x -0.75 = -0.675 (flipped to negative)
```
The -0.75 multiplier (not -1.0) reflects that negation weakens but doesn't fully reverse sentiment.

**Compound score normalization (Line 168):**
```
compound = raw_sum / sqrt(raw_sum^2 + 15)
```
This is VADER's original normalization formula. The constant 15 controls sensitivity. Output is always between -1 and +1.

#### Analyzer 2: Pattern Sentiment / Subjectivity -- Lines 191-270

**What it is:** Measures two dimensions -- polarity AND subjectivity.

**Subjectivity (Lines 204-223):**
```python
subjective_words = {'believe', 'think', 'feel', 'best', 'committed', 'passionate', ...}
objective_indicators = {'percent', 'million', 'audited', 'certified', 'metric', ...}

subjectivity = subjective_count / (subjective_count + objective_count)
```

**Why subjectivity matters for greenwashing:**
```
"We believe we are the most sustainable company"  → subjectivity = 1.0 (pure opinion)
"We reduced emissions by 25% per ISO 14001 audit" → subjectivity = 0.0 (pure fact)
```

Greenwashing text is HIGHLY SUBJECTIVE. Legitimate reports are OBJECTIVE.

#### Analyzer 3: Greenwashing Linguistic Detector -- Lines 277-422

**What it is:** Detects 5 specific language patterns from academic greenwashing research.

**The 5 pattern categories (Lines 295-346):**

| Category | Count | Example Terms | Signal |
|----------|-------|---------------|--------|
| Vague Language | 23 | "eco-friendly", "journey", "committed to" | Sounds good, says nothing |
| Superlatives | 17 | "world-class", "pioneering", "unparalleled" | Exaggerated, unverifiable |
| Hedging | 22 | "may", "where feasible", "endeavor" | Weakens commitments |
| Future Language | 17 | "by 2050", "plan to", "roadmap" | Promise without proof |
| Concrete Evidence | 24 | "reduced by", "%", "ISO 14001", "scope 1" | Verifiable, specific |

**The Greenwashing Score Formula (Lines 394-408):**
```
raw_score = vague_density + superlative_density + hedging_density + future_density
            - 2 x concrete_evidence_density

gw_linguistic_score = 1 / (1 + exp(-raw_score / 5))    # sigmoid normalization
```

**Why concrete evidence gets 2x weight:** One verifiable fact outweighs multiple vague claims. A company saying "reduced emissions by 25% (verified by Bureau Veritas)" is far more credible than 10 sentences of "we are committed to sustainability."

**The sigmoid function maps any score to [0, 1]:**
```
raw = -5  →  sigmoid = 0.27  (more concrete than vague = LOW greenwashing)
raw =  0  →  sigmoid = 0.50  (balanced)
raw = +5  →  sigmoid = 0.73  (more vague than concrete = HIGH greenwashing)
```

#### Analyzer 4: Section-Based Sentiment -- Lines 429-492

**What it does:** Splits text into 3 equal sections (beginning, middle, end) and checks if sentiment is consistent.

**Why it matters:**
```
Section 1: "We are leaders in sustainability"        → +0.8 (positive)
Section 2: "However, emissions increased by 40%"     → -0.6 (negative)
Section 3: "We pledge to do better by 2050"           → +0.5 (positive)

Variance = HIGH → Company buries bad news in the middle = greenwashing tactic
```

**Consistency formula (Line 481):**
```
consistency = 1.0 - min(variance x 10, 1.0)

Low variance  → consistency near 1.0 (honest, uniform tone)
High variance → consistency near 0.0 (hiding information)
```

#### Comprehensive Function -- Lines 499-549

The `analyze_text_sentiment()` function chains all 4 analyzers and produces ~20 features per text:

```
VADER:    compound, positive, negative, neutral (4 features)
Pattern:  polarity, subjectivity (2 features)
GW Ling:  5 counts + 5 densities + gw_score (11 features)
Section:  variance, range, consistency (3 features)
Label:    positive/negative/neutral (1 feature)
Total:    ~21 features per company
```

**Input:** Company description text
**Output:** Dictionary of ~21 sentiment features

---

### `nlp_esg_claim_extraction.py` -- Complete Breakdown

This file extracts individual ESG claims from company descriptions, classifies them, and scores their credibility.

#### The Core Idea

A company description contains multiple ESG claims mixed together:
```
"We reduced emissions by 35%... Our eco-friendly products... We plan to be carbon neutral by 2050..."
```

This module finds each claim, asks 4 questions about it:
1. **Which ESG pillar?** (Environmental, Social, or Governance)
2. **How strong?** (vague "committed to" vs specific "reduced by 35%")
3. **Verified?** (does it reference ISO, GRI, third-party audit?)
4. **Past or future?** (proven "achieved" vs promised "plan to")

#### Sentence Tokenizer -- Lines 32-72

**What it does:** Splits text into individual sentences while handling abbreviations.

**The problem:** Naive splitting on "." breaks on abbreviations:
```
"Acme Corp. reduced emissions." → ["Acme Corp", " reduced emissions"]  (WRONG)
```

**The solution (Lines 49-60):** Temporarily replace abbreviation periods with a placeholder:
```
"Acme Corp." → "Acme Corp<PERIOD>"  (protect from splitting)
Split on sentence boundaries
"Acme Corp<PERIOD>" → "Acme Corp."  (restore period)
```

#### ESG Claim Detection Patterns -- Lines 82-202

20 regex patterns organized by ESG pillar:

**Environmental Patterns (7):**

| Pattern Name | Regex | What It Catches |
|-------------|-------|-----------------|
| `emission_reduction` | `(reduc\w+\|cut\w*).{0,40}(emission\|carbon\|ghg)` | "reduced carbon emissions" |
| `renewable_energy` | `(renewable\|solar\|wind).{0,30}(energy\|power)` | "solar energy usage" |
| `carbon_neutral` | `(carbon\s*neutral\|net[\s-]zero)` | "net-zero" targets |
| `waste_management` | `(recycl\w+\|zero[\s-]waste\|circular)` | "zero-waste" claims |
| `water_conservation` | `(water).{0,30}(conserv\|reduc\|efficien)` | "water conservation" |
| `biodiversity` | `(biodiversity\|ecosystem\|habitat)` | Ecosystem claims |
| `env_certification` | `(iso\s*14001\|leed\|energy\s*star)` | Environmental certs |

**Social Patterns (6):** diversity_inclusion, employee_safety, community_impact, human_rights, employee_development, data_privacy

**Governance Patterns (6):** board_independence, ethics_compliance, transparency_reporting, risk_management, executive_compensation, whistleblower

**How the regex works (example):**
```
Pattern: (reduc\w+|cut\w*).{0,40}(emission|carbon|ghg)

(reduc\w+|cut\w*)  → matches "reduced", "reducing", "reduction", "cut", "cutting"
.{0,40}            → allows up to 40 characters between the two parts
(emission|carbon)  → matches "emission", "carbon", "ghg"

Matches: "reduced our carbon" ✓
Matches: "significant reduction in greenhouse gas emissions" ✓
Misses:  "carbon was reduced by us in many different ways across the organization" ✗ (gap > 40 chars)
```

#### Claim Strength Scoring -- Lines 318-360

Scores each claim from 0 (vague) to 1 (highly specific) using 5 criteria:

| Criterion | Score | What It Looks For |
|-----------|-------|-------------------|
| Quantitative data | +0.30 | Numbers, percentages, units (%, tonnes, kwh) |
| Specific timeframe | +0.20 | Year references (2023), "fiscal year", "quarterly" |
| Third-party verification | +0.20 | "verified", "audited", "ISO", "GRI", "TCFD" |
| Action verbs | +0.15 | "achieved", "implemented", "deployed", "measured" |
| Comparisons | +0.15 | "compared to", "baseline", "year-over-year" |

**Maximum possible score: 1.0** (all 5 criteria met)

**Example:**
```
"Reduced emissions by 35% in 2023 vs baseline, verified by Bureau Veritas"
  → quantitative (35%): +0.30
  → timeframe (2023): +0.20
  → verified (Bureau Veritas): +0.20
  → action verb (reduced): +0.15
  → comparison (vs baseline): +0.15
  → TOTAL: 1.00 (maximum credibility)

"We are committed to becoming more sustainable"
  → No numbers, no date, no verification, no action verb, no comparison
  → TOTAL: 0.00 (no credibility)
```

#### Temporal Classification -- Lines 375-395

Classifies each claim as past (proven) vs future (promised):

```python
past_indicators = ['achieved', 'reduced', 'completed', 'implemented', 'last year', ...]
future_indicators = ['will', 'plan to', 'by 2030', 'committed to', 'roadmap', ...]
```

**Why this matters for greenwashing:**
```
Past claims:   "We ACHIEVED a 25% reduction" → Proven, verifiable
Future claims: "We PLAN TO be carbon neutral by 2050" → Unverifiable promise

Companies heavy on future promises and light on past achievements = greenwashing signal
```

#### Aggregate Claim Metrics -- Lines 402-504

Computes 15 per-company metrics from all extracted claims:

| Metric | Formula | Greenwashing Signal |
|--------|---------|-------------------|
| `total_claims` | count of all claims | More claims = more to scrutinize |
| `env_claims` / `social_claims` / `gov_claims` | per-pillar counts | Imbalanced = suspicious |
| `avg_claim_strength` | mean(all strengths) | Low = mostly vague claims |
| `verified_claim_ratio` | verified / total | Low = unverified claims |
| `past_claim_ratio` | past / total | Low = no proven track record |
| `future_claim_ratio` | future / total | High = lots of promises |
| `strong_claim_ratio` | (strength >= 0.5) / total | Low = mostly weak claims |
| `weak_claim_ratio` | (strength < 0.3) / total | High = mostly vague claims |
| `claim_pillar_diversity` | distinct pillars (0-3) | Low = only covers one ESG area |
| `claim_credibility_score` | weighted composite | **THE KEY METRIC** |

**Credibility Score Formula (Lines 480-486):**
```
credibility = 0.30 x verified_ratio       # 30% weight on third-party verification
            + 0.25 x past_ratio            # 25% weight on proven past performance
            + 0.25 x avg_strength          # 25% weight on claim specificity
            + 0.10 x credential_ratio      # 10% weight on certifications
            + 0.10 x (pillar_diversity/3)  # 10% weight on balanced ESG coverage
```

**Interpretation:**
```
credibility = 0.85  → Company backs up claims with data, verification, past results
credibility = 0.15  → Company makes vague, unverified, future-only promises
```

**Input:** Company description text
**Output:** List of individual claims + 15 aggregate metrics per company

---

### `nlp_pipeline.py` -- Complete Breakdown

This file is the master orchestrator that chains the 3 NLP modules together into a single pipeline.

#### The Core Idea

Instead of running 3 separate scripts manually:
```
python nlp_text_preprocessor.py    # Step 1
python nlp_sentiment_analysis.py   # Step 2
python nlp_esg_claim_extraction.py # Step 3
```

This pipeline runs everything in one command and handles data flow between steps.

#### Pipeline Flow -- Lines 340-400

```
main()
  |
  |-- Step 1: load_processed_data()          # Load 3 datasets (SP500, NIFTY50, Profiles)
  |
  |-- Step 2: run_text_preprocessing()       # Clean text for each dataset (x3)
  |     Calls: preprocess_dataframe_text()   # From nlp_text_preprocessor.py
  |     Adds: clean_text, text_stats         # 7+ text statistic columns
  |
  |-- Step 3: run_esg_keyword_analysis()     # Count ESG keywords for each dataset (x3)
  |     Calls: add_esg_keyword_features()    # From nlp_text_preprocessor.py
  |     Adds: env/social/gov keyword counts  # 6 keyword feature columns
  |
  |-- Step 4: run_sentiment_analysis()       # Full sentiment analysis for each dataset (x3)
  |     Calls: add_sentiment_features()      # From nlp_sentiment_analysis.py
  |     Adds: VADER + Pattern + GW + Section # ~21 sentiment feature columns
  |
  |-- Step 5: run_claim_extraction()         # Extract ESG claims for each dataset (x3)
  |     Calls: extract_claims_from_dataframe() # From nlp_esg_claim_extraction.py
  |     Adds: claim counts, strength, cred   # 15 claim metric columns
  |
  |-- Step 6: save_nlp_results()             # Save all enriched datasets
  |     Saves: sp500_nlp_features.csv        # 430 rows with all NLP features
  |     Saves: nifty50_nlp_features.csv      # 50 rows with all NLP features
  |     Saves: company_profiles_nlp.csv      # 480 rows with all NLP features
  |     Saves: all_esg_claims.csv            # Every individual claim extracted
  |     Saves: esg_claim_report.txt          # Human-readable claim summary
  |
  |-- Step 7: print_nlp_summary()            # Print results summary
```

**Key design decision (Line 222-223):** Claim extraction uses the ORIGINAL description text (not cleaned), because regex patterns need original casing and punctuation to match correctly.

**Input:** 3 cleaned datasets from Phase 2
**Output:** 5 NLP-enriched files + claim report

---

### `feature_engineering_numerical.py` -- Complete Breakdown

This file transforms raw ESG scores (5 numbers per company) into 43 engineered numerical features that expose hidden greenwashing patterns.

#### The Core Idea

You start with just 5 raw numbers per company:
```
total_esg_risk_score = 25.3
env_risk_score       = 5.2
social_risk_score    = 8.1
gov_risk_score       = 7.4
controversy_score    = 2
```

But these raw numbers don't tell the full story. A greenwashing company can have a "decent" ESG score while hiding imbalances. This module creates 43 features that expose those hidden patterns across 5 categories.

#### Category 1: ESG Pillar Ratios (9 features) -- Lines 76-173

**Question it answers:** "Is the company's ESG performance balanced across E, S, G -- or lopsided?"

```
Company A:  E=33%, S=33%, G=33%  →  Balanced (legitimate)
Company B:  E=5%,  S=10%, G=85%  →  Lopsided (suspicious)
```

Company B invested in governance (board structure, policies) but ignored actual environmental action -- a classic greenwashing tactic.

| Feature | Formula | What It Catches |
|---------|---------|-----------------|
| `env_to_total_ratio` | env / total | How much risk is environmental |
| `social_to_total_ratio` | social / total | How much risk is social |
| `gov_to_total_ratio` | gov / total | How much risk is governance |
| `env_gov_ratio` | env / gov | Environmental vs governance balance |
| `env_social_ratio` | env / social | Environmental vs social balance |
| `social_gov_ratio` | social / gov | Social vs governance balance |
| `pillar_imbalance_score` | std(3 ratios) | **KEY**: How uneven are the pillars |
| `dominant_pillar_strength` | max(3 ratios) | How much the strongest pillar dominates |
| `weakest_pillar_weakness` | min(3 ratios) | How neglected is the weakest pillar |

#### Category 2: Risk Decomposition (6 features) -- Lines 178-268

**Question it answers:** "Does the company's controversy level match its declared ESG risk?"

The key feature is `controversy_risk_ratio`:
```
Company X: ESG_risk = 10 (low), Controversy = 5 (severe) → ratio = 0.50 (SUSPICIOUS)
Company Y: ESG_risk = 40 (high), Controversy = 2 (low)   → ratio = 0.05 (expected)
```

Also includes the Herfindahl-Hirschman Index (HHI) for risk concentration:
```
HHI = (env_share)^2 + (social_share)^2 + (gov_share)^2
Perfect balance:    (0.33)^2 + (0.33)^2 + (0.33)^2 = 0.33
Total concentration: (1.0)^2 + (0.0)^2 + (0.0)^2   = 1.00
```

#### Category 3: Statistical Moments (7 features) -- Lines 273-370

**Question it answers:** "How statistically unusual is this company compared to all others?"

The #1 most important feature is `esg_controversy_divergence`:
```
z-score tells you: "How far from normal is this value?"

Company A:  z(ESG_risk) = -1.5  (claims MUCH LOWER risk than average)
            z(controversy) = +2.0  (has MUCH HIGHER controversy than average)
            divergence = 2.0 - (-1.5) = 3.5  → STRONG GREENWASHING SIGNAL

Company B:  z(ESG_risk) = +1.0  z(controversy) = +1.0
            divergence = 0.0  → CONSISTENT (not greenwashing)
```

#### Category 4: Interaction Features (8 features) -- Lines 375-514

**Question it answers:** "What patterns emerge when we COMBINE features?"

```
High ESG score + Low controversy  = Legitimate (good)
High ESG score + High controversy = Greenwashing (bad!)
Low ESG score  + High controversy = Known bad actor (different problem)
```

The COMBINATION matters. Key feature: `imbalance_controversy_interact`:
```
Company A: imbalance=0.3, controversy=4 → 0.3 x 4 = 1.2 (RED FLAG)
Company B: imbalance=0.3, controversy=1 → 0.3 x 1 = 0.3 (mild concern)
```
Only fires strongly when BOTH imbalance AND controversy are high.

#### Category 5: Anomaly Detection (6 features) -- Lines 519-624

Uses 3 different outlier detection methods:
- **IQR (box plot):** Outlier if value < Q1-1.5xIQR or > Q3+1.5xIQR
- **MAD (Median Absolute Deviation):** More robust than z-score because median is immune to outliers
- **Combined score:** Weighted blend of all signals

#### Sector-Relative Features (7 features) -- Lines 629-730

```
ESG score = 30:
  In Technology sector (avg 15) → TERRIBLE (z = +2.0)
  In Energy sector (avg 35)     → GOOD (z = -0.7)
```

#### Scaling -- Lines 643-689

Robust Scaling: `z = (x - median) / IQR`. Creates `_scaled` versions of all 43 features.

**Input:** 5 raw ESG numbers per company
**Output:** 43 features + 43 scaled versions = 86 new columns

---

### `feature_engineering_nlp.py` -- Complete Breakdown

This file extracts 47 numerical features from company description text across 6 categories.

#### The Core Idea

Text is the PRIMARY evidence of greenwashing. Companies reveal their true intentions through language patterns. This module converts raw text descriptions into 47 quantitative features.

#### Category 1: Sentiment Features (5 features) -- Lines 172-252

| Feature | Range | Greenwashing Signal |
|---------|-------|-------------------|
| `text_polarity` | -1 to +1 | Overly positive = suspicious |
| `text_positive_ratio` | 0 to 1 | High positive ratio = marketing tone |
| `text_negative_ratio` | 0 to 1 | Very low = avoids discussing problems |
| `text_sentiment_strength` | 0 to 1 | Very high = emotionally charged (not factual) |
| `text_pos_neg_ratio` | 0 to inf | Very high = one-sided positive spin |

#### Category 2: Readability Features (6 features) -- Lines 277-417

| Feature | What It Measures | Greenwashing Signal |
|---------|-----------------|-------------------|
| `avg_word_length` | Characters per word | Longer = more jargon/complexity |
| `avg_sentence_length` | Words per sentence | Longer = buries information |
| `syllable_ratio` | Complex words (3+ syllables) / total | Higher = harder to understand |
| `flesch_reading_ease` | Readability (higher = easier) | Very low = obscures with complexity |
| `gunning_fog_index` | Years of education needed | > 17 = graduate level (red flag) |
| `long_word_ratio` | Words with 6+ characters / total | Higher = more complex vocabulary |

**Flesch formula (Line 393):**
```
Flesch = 206.835 - 1.015 x (words/sentences) - 84.6 x (syllables/words)
Score 60-70 = standard    Score < 30 = very difficult to read
```

#### Category 3: Vocabulary Richness (6 features) -- Lines 419-520

| Feature | Formula | Greenwashing Signal |
|---------|---------|-------------------|
| `lexical_diversity` | unique / total (Type-Token Ratio) | Low = repeats same buzzwords |
| `hapax_legomena_ratio` | words-appearing-once / total | High = uses rare words for impression |
| `top_word_concentration` | top-10 words / total | High = text dominated by few key terms |

#### Category 4: ESG Keyword Density (10 features) -- Lines 522-630

Counts Environmental, Social, and Governance keywords using curated lexicons from GRI/SASB/TCFD standards (Lines 77-111).

**Key insight:**
```
Company A: env_keyword_density = 0.08, env_risk_score = 2.0 (low)
  → HIGH keyword usage + LOW actual performance = GREENWASHING
  → They TALK green but don't ACT green

Company B: env_keyword_density = 0.02, env_risk_score = 2.0 (low)
  → LOW keyword usage + LOW performance = consistent (just not green-focused)
```

#### Category 5: Greenwashing Linguistic Signals (12 features) -- Lines 632-770

**THE MOST IMPORTANT NLP CATEGORY.** Uses 5 linguistic pattern lexicons (Lines 119-166):

- **Vague patterns:** 20 phrases like "committed to", "journey", "eco-friendly"
- **Hedge patterns:** 18 phrases like "may", "where feasible", "aim to"
- **Superlative patterns:** 15 phrases like "world-class", "pioneering", "unprecedented"
- **Future patterns:** 15 phrases like "by 2050", "plan to", "roadmap"
- **Concrete patterns:** 18 phrases like "reduced by", "%", "ISO 14001", "scope 1"

**The Greenwashing Signal Score (Lines 736-742):**
```
raw_signal = vague_density + hedge_density + superlative_density + future_density
             - 2.0 x concrete_density

gw_score = 1 / (1 + exp(-10 x raw_signal))    # sigmoid to [0, 1]
```

The `vague_to_concrete_ratio` captures: "How much fluff per unit of evidence?"

#### Category 6: Document Structure (8 features) -- Lines 772-870

| Feature | What It Catches |
|---------|---------------|
| `sentence_count` | Document length |
| `sentence_length_variance` | Inconsistent writing = mixed content |
| `short_sentence_ratio` | Marketing copy uses short punchy sentences |
| `long_sentence_ratio` | Legal/technical text has very long sentences |
| `question_mark_count` | Rhetorical questions = persuasion technique |
| `exclamation_count` | Exclamations = promotional marketing tone |
| `number_density` | More numbers = more quantitative evidence |
| `capitalized_word_ratio` | Excessive caps = emphasis/marketing |

**Input:** Company description text
**Output:** 47 NLP features per company

---

### `feature_engineering_categorical.py` -- Complete Breakdown

This file transforms categorical variables (sector, industry, risk levels) into 31 ML-ready numerical features.

#### The Core Idea

ML models need numbers, not text. "Technology" and "Energy" are categories that must be encoded. But HOW you encode them matters:

```
One-hot encoding: Technology → [1,0,0,0,0,0,0,0,0,0,0]  (creates 11 columns for 11 sectors)
Frequency encoding: Technology → 67  (just 1 column: how many companies in this sector)
```

This module uses 5 smart encoding strategies instead of brute-force one-hot.

#### Category 1: Frequency Encoding (4 features) -- Lines 70-155

Replaces each category with how often it appears:
```
Technology → 67 (67 tech companies)
Energy → 25 (25 energy companies)
```

**Why this works:** Sectors with more companies have more reliable statistics. Rare sectors (5 companies) have noisy ESG scores.

#### Category 2: Risk-Based Binning (8 features) -- Lines 160-295

Discretizes continuous scores into domain-aligned tiers:

```
ESG Score 0-10   → Bin 0 (Negligible)
ESG Score 10-20  → Bin 1 (Low)
ESG Score 20-30  → Bin 2 (Medium)
ESG Score 30-40  → Bin 3 (High)
ESG Score 40+    → Bin 4 (Severe)
```

Also creates binary flags:
- `high_risk_flag = 1` if ESG risk bin >= 3
- `high_controversy_flag = 1` if controversy score >= 4

#### Category 3: Cross-Feature Derivations (6 features) -- Lines 298-420

**The critical greenwashing detector -- `risk_controversy_mismatch` (Lines 345-350):**
```
IF esg_risk_bin <= 1 (Low risk) AND controversy_bin >= 3 (High controversy):
    risk_controversy_mismatch = 1  → GREENWASHING SUSPECT
ELSE:
    risk_controversy_mismatch = 0  → Consistent
```

This is the TEXTBOOK DEFINITION of greenwashing: claiming low risk while having high controversy.

Also creates ESG x Controversy segments (natural company "archetypes"):
```
Segment "1_0" = Low ESG risk + No controversy (normal company)
Segment "1_3" = Low ESG risk + High controversy (GREENWASHING)
Segment "3_3" = High ESG risk + High controversy (known bad actor)
```

#### Category 4: Material ESG Issues (6 features) -- Lines 425-563

Encodes Material ESG Issues from the NIFTY50 dataset into binary flags:
```
has_carbon_issue = 1 if text contains "carbon|climate|ghg|emission"
has_ethics_issue = 1 if text contains "ethic|corruption|bribery|fraud"
has_human_capital_issue = 1 if text contains "human capital|labor|safety"
has_governance_issue = 1 if text contains "governance|board|compliance"
```

#### Category 5: Sector Risk Profiles (7 features) -- Lines 566-680

Computes sector-level aggregate statistics and attaches them to each company:
```
sector_avg_esg = mean ESG score for all companies in this sector
sector_esg_spread = max - min ESG within the sector
sector_high_risk_ratio = fraction of sector companies in high risk
```

**Input:** Company profiles with sector, industry, risk columns
**Output:** 31 categorical features

---

### `feature_engineering_pipeline.py` -- Complete Breakdown

This file is the master orchestrator that chains all 3 feature engineering modules into one pipeline.

#### The Pipeline Flow -- Lines 603-657

```
Step 1: Load Data
  → company_profiles.csv (480 companies, 13 columns)

Step 2: Numerical Feature Engineering (+87 columns)
  → Pillar ratios, risk decomposition, statistics, interactions, anomalies, sector-relative

Step 3: NLP Feature Engineering (+47 columns)
  → Sentiment, readability, vocabulary, ESG keywords, GW linguistic, doc structure

Step 4: Categorical Feature Engineering (+32 columns)
  → Frequency encoding, risk bins, cross-features, material issues, sector profiles

Step 5: Feature Quality Checks
  → Remove constant features (7 removed: zero information)
  → Remove duplicate columns (3 removed: identical values)
  → Report highly correlated pairs (66 pairs with |r| > 0.98)
  → Replace inf with NaN, fill NaN with 0

Step 6: Generate Feature Registry
  → Metadata for all 121 features (name, module, category, stats)

Step 7: Save Outputs
  → feature_matrix.csv (480 x 169) -- THE FINAL ML-READY DATASET
  → feature_registry.csv (121 features with metadata)
  → pipeline_summary.txt (execution report)
```

**Execution time:** Under 1 second for 480 companies.

**Input:** `company_profiles.csv` (480 x 13)
**Output:** `feature_matrix.csv` (480 x 169)

---

## Datasets

### Raw Datasets (from Kaggle)

| Dataset | File | Records | Key Columns |
|---------|------|---------|-------------|
| Corporate Greenwashing Accusations | `Greenwashing_Score_Data.xlsx` | 595 | Company name, year, GW score (0-1) |
| S&P 500 ESG Risk Ratings | `SP 500 ESG Risk Ratings.csv` | 430 | ESG scores, env/social/gov risk, controversy |
| ESG & Financial Performance | `company_esg_financial_dataset.csv` | 1000 | ESG scores, financial metrics, carbon emissions |
| NIFTY 50 ESG Score Data | `final_data.csv` | 50 | ESG risk, material issues, controversy levels |

### Processed Datasets

| File | Shape | Description |
|------|-------|-------------|
| `company_profiles.csv` | 480 x 13 | Merged S&P 500 + NIFTY 50 company data |
| `feature_matrix.csv` | 480 x 169 | Final ML-ready feature matrix |
| `feature_registry.csv` | 121 x 9 | Metadata for all engineered features |
| `greenwashing_cleaned.csv` | 595 x 6 | Cleaned greenwashing labels |
| `esg_financial_features.csv` | 1000 x 34 | Cleaned financial features |

---

## Feature Engineering Summary

### Feature Counts by Module

| Module | Category | Features | Description |
|--------|----------|----------|-------------|
| Numerical | Pillar Ratios | 9 | E/S/G balance and imbalance metrics |
| Numerical | Risk Decomposition | 6 | Residuals, gaps, controversy ratios |
| Numerical | Statistical Moments | 7 | Variance, skewness, z-scores, CV |
| Numerical | Interaction Terms | 8 | Cross-feature products and squared terms |
| Numerical | Anomaly Detection | 6 | IQR outliers, MAD scores, combined anomaly |
| Numerical | Sector-Relative | 7 | Benchmarks against sector peers |
| NLP | Sentiment | 5 | Polarity, positivity, negativity ratios |
| NLP | Readability | 6 | Flesch, Gunning Fog, word complexity |
| NLP | Vocabulary | 6 | Lexical diversity, hapax, concentration |
| NLP | ESG Keywords | 10 | Per-pillar keyword density and balance |
| NLP | GW Linguistic | 12 | Vague, hedge, superlative, future, concrete |
| NLP | Document Structure | 8 | Sentence patterns, number density |
| Categorical | Frequency Encoding | 4 | Sector/industry frequency and proportion |
| Categorical | Risk Bins | 8 | Domain-aligned risk tier discretization |
| Categorical | Cross-Feature | 6 | ESG x controversy segments, mismatch flags |
| Categorical | Material Issues | 6 | ESG issue type binary flags |
| Categorical | Sector Profiles | 7 | Sector-level risk statistics |
| **Total** | | **121** | **+ 43 scaled versions + 5 original columns** |

### Top 5 Most Important Greenwashing Detection Features

1. **`esg_controversy_divergence`** (Numerical) -- z(controversy) minus z(ESG_risk). Large positive value = "claims low risk but is highly controversial."

2. **`greenwashing_signal_score`** (NLP) -- Sigmoid-normalized score combining vague language, hedging, superlatives, and future promises minus concrete evidence.

3. **`risk_controversy_mismatch`** (Categorical) -- Binary flag: ESG risk bin is low but controversy bin is high. Textbook greenwashing definition.

4. **`controversy_risk_ratio`** (Numerical) -- Controversy score divided by total ESG risk. High ratio = more controversies than the ESG score suggests.

5. **`vague_to_concrete_ratio`** (NLP) -- Ratio of vague/hedging language to concrete evidence. High ratio = more fluff than facts.

---

## Model Training Results

### Models Trained

| # | Model | Type | Library | Description |
|---|-------|------|---------|-------------|
| 1 | Random Forest | Supervised | scikit-learn | Bagging ensemble of decision trees |
| 2 | Gradient Boosting | Supervised | scikit-learn | Sequential boosted trees (sklearn) |
| 3 | XGBoost | Supervised | xgboost | Extreme Gradient Boosting (state-of-the-art) |
| 4 | Logistic Regression | Supervised | scikit-learn | Linear model with L1/L2 regularization |
| 5 | SVM | Supervised | scikit-learn | Support Vector Machine with RBF kernel |
| 6 | Isolation Forest | Unsupervised | scikit-learn | Anomaly detection (no labels needed) |

### Model Leaderboard (sorted by F1 Score)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Gradient Boosting | 0.9688 | 0.9701 | 0.9688 | 0.9682 | 0.9979 |
| Random Forest | 0.8958 | 0.8946 | 0.8958 | 0.8947 | 0.9449 |
| Logistic Regression | 0.8646 | 0.8661 | 0.8646 | 0.8652 | 0.9305 |
| SVM | 0.8438 | 0.8496 | 0.8438 | 0.8458 | 0.9202 |

### Training Configuration

- **Cross-validation:** 5-fold Stratified K-Fold
- **Hyperparameter tuning:** GridSearchCV (exhaustive search)
- **Train/Test split:** 80/20 with stratification
- **Primary metric:** Weighted F1 Score
- **Class imbalance handling:** `class_weight='balanced'` + `scale_pos_weight`

### Top 15 Predictive Features (Gradient Boosting)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `controversy_risk_ratio` | 0.2763 | Numerical |
| 2 | `controversy_risk_ratio_scaled` | 0.2570 | Numerical (scaled) |
| 3 | `esg_controversy_divergence_scaled` | 0.1500 | Numerical (scaled) |
| 4 | `esg_controversy_divergence` | 0.0901 | Numerical |
| 5 | `greenwashing_signal_score` | 0.0413 | NLP |
| 6 | `sector_esg_spread` | 0.0197 | Categorical |
| 7 | `controversy_adjusted_risk` | 0.0176 | Numerical |
| 8 | `esg_sector_mean_scaled` | 0.0166 | Numerical (scaled) |
| 9 | `esg_sector_mean` | 0.0120 | Numerical |
| 10 | `avg_word_length` | 0.0105 | NLP |
| 11 | `esg_mad_score_scaled` | 0.0104 | Numerical (scaled) |
| 12 | `controversy_adjusted_risk_scaled` | 0.0072 | Numerical (scaled) |
| 13 | `esg_performance_tier` | 0.0065 | Categorical |
| 14 | `number_density` | 0.0061 | NLP |
| 15 | `total_esg_keyword_count` | 0.0057 | NLP |

### Proxy Label Construction

Since the greenwashing labels dataset (European companies) has zero company overlap with our feature matrix (US S&P 500 + Indian NIFTY50), we construct proxy labels using 5 domain-expert indicators:

1. **ESG-Controversy Divergence** -- z(controversy) - z(ESG_risk) > 75th percentile
2. **Greenwashing Linguistic Score** -- NLP vague/hedging score > 75th percentile
3. **Risk-Controversy Mismatch** -- Low ESG risk bin + High controversy bin
4. **Controversy-Risk Ratio** -- controversy/ESG_risk > 75th percentile
5. **Combined Anomaly Score** -- Statistical outlier score > 75th percentile

Binary label: `gw_label_binary = 1` if >= 2 out of 5 indicators are flagged.

Result: 145/480 companies (30.2%) flagged as potential greenwashing.

---

## Risk Scoring

Each company receives a **0-100 Greenwashing Risk Score** combining 5 weighted components:

| Component | Weight | Source | What It Measures |
|-----------|--------|--------|-----------------|
| Proxy Score | 40% | model_training.py | Sum of 5 domain indicators (0-5 scaled to 0-100) |
| Linguistic Score | 15% | nlp_sentiment_analysis.py | Vague/hedging language density |
| Divergence Score | 15% | feature_engineering_numerical.py | ESG-controversy statistical gap |
| Credibility (inverted) | 15% | nlp_esg_claim_extraction.py | Low claim credibility = higher risk |
| Controversy Ratio | 15% | feature_engineering_numerical.py | Controversy relative to ESG score |

### Risk Tiers

| Tier | Score Range | Companies | Percentage |
|------|------------|-----------|------------|
| Very Low Risk | 0-20 | 26 | 5.4% |
| Low Risk | 21-40 | 317 | 66.0% |
| Moderate Risk | 41-60 | 128 | 26.7% |
| High Risk | 61-80 | 9 | 1.9% |
| Very High Risk | 81-100 | 0 | 0.0% |

### Top 10 Highest Risk Companies

| Rank | Company | Score | Tier | Sector |
|------|---------|-------|------|--------|
| 1 | Adani Ports | 76.57 | High Risk | Industrials |
| 2 | Mastercard | 71.36 | High Risk | Financial Services |
| 3 | Target Corporation | 68.86 | High Risk | Consumer Cyclical |
| 4 | Paramount Global | 68.53 | High Risk | Communication Services |
| 5 | Alphabet (Google) | 65.98 | High Risk | Technology |
| 6 | Hasbro | 63.45 | High Risk | Consumer Cyclical |
| 7 | Cencora | 61.92 | High Risk | Healthcare |
| 8 | Thermo Fisher | 61.31 | High Risk | Healthcare |
| 9 | Johnson Controls | 60.11 | High Risk | Industrials |
| 10 | Cardinal Health | 59.60 | Moderate Risk | Healthcare |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) answers: **"WHY did the model flag this company?"**

For each flagged company, SHAP shows:
- Which features pushed the prediction TOWARD greenwashing (positive SHAP)
- Which features pushed AWAY from greenwashing (negative SHAP)
- The magnitude of each feature's contribution

### Outputs Generated

| File | Description |
|------|-------------|
| `plots/shap_summary.png` | Beeswarm plot: all features x all companies |
| `plots/shap_bar.png` | Mean absolute SHAP values (global importance) |
| `plots/shap_dependence.png` | Top feature: how its value affects prediction |
| `data/processed/shap_explanations.txt` | Text explanations for top 10 flagged companies |

---

## Streamlit Dashboard

Interactive web UI with 5 pages, all charts built with **Plotly** (zoomable, hoverable, clickable):

| Page | What It Shows |
|------|--------------|
| **Risk Score Dashboard** | Histogram, pie chart, sector comparison, bubble scatter, ranked table |
| **Model Performance** | Grouped bar chart, radar comparison, training time, metrics table |
| **Feature Importance** | Interactive bar chart, treemap by category, cumulative importance curve |
| **Company Deep Dive** | Component breakdown, gauge meter, radar profile, sector peer comparison |
| **Company Search & Analysis** | Text search, 8-section report, NLP analysis, indicator table, sector ranking |
| **Real-Time Intelligence** | Live news feed, sentiment timeline, risk alerts, portfolio monitor |
| **ESG Report Analyzer (AI)** | PDF upload, Gemini LLM claim extraction, greenwashing highlighting |
| **Report Generator** | Single/batch/portfolio HTML reports, downloadable, professional formatting |
| **Advanced Explainability** | Counterfactual explanations, what-if sliders, sensitivity curves |
| **SHAP Explanations** | Feature impact bars, distribution explorer, correlation heatmap |

---

## New Features (Advanced Dashboard Modules)

The dashboard has been extended with **8 advanced pages** beyond the original 5 core pages. These modules add real-time intelligence, LLM-powered analysis, report generation, advanced explainability, company comparison, time-series forecasting, and a greenwashing reason engine. The dashboard now has **13 pages total** serving as a complete ESG greenwashing detection platform.

**Quick Reference -- All Dashboard Pages:**

| # | Page | API/AI Used? | Dataset Used |
|---|------|-------------|--------------|
| 1 | Risk Score Dashboard | No | `greenwashing_risk_scores.csv` |
| 2 | Model Performance | No | `model_metrics.csv`, `predictions.csv` |
| 3 | Feature Importance | No | `feature_importance_gradient_boosting.csv` |
| 4 | Company Deep Dive | No | `greenwashing_risk_scores.csv`, `feature_matrix.csv` |
| 5 | Company Search & Analysis | No | `greenwashing_risk_scores.csv`, `feature_matrix.csv`, `predictions.csv` |
| 6 | Real-Time Intelligence | Google News RSS (free, no key) | Live news + `greenwashing_risk_scores.csv` |
| 7 | ESG Report Analyzer (AI) | Google Gemini 2.0 Flash (free key) | User-uploaded PDF + `greenwashing_risk_scores.csv` |
| 8 | Report Generator | No | `greenwashing_risk_scores.csv`, `feature_matrix.csv`, `predictions.csv` |
| 9 | Advanced Explainability | No | `feature_matrix.csv` (trains model live) |
| 10 | Company Comparison | No | `greenwashing_risk_scores.csv`, `feature_matrix.csv` |
| 11 | Time-Series Risk Tracking | No | `company_esg_financial_dataset.csv` (11K rows, 11 years) |
| 12 | SHAP Explanations | No | `feature_importance_*.csv`, `feature_matrix.csv`, `shap_explanations.txt` |

---

### 1. Company Search & Analysis (with Greenwashing Reason Engine)

**In Brief:** Select any company from a searchable dropdown and get a complete 9-section greenwashing analysis report with a plain-English "Reason Engine" explaining WHY the company received its rating.

**Why it matters:** The core pipeline produces scores and numbers, but stakeholders need to understand the WHY. This page translates raw ML output into actionable, human-readable insights that a non-technical investor or compliance officer can understand.

**Datasets Used:**
| Dataset | File | What It Provides |
|---------|------|------------------|
| Risk Scores | `data/processed/greenwashing_risk_scores.csv` | Risk score (0-100), tier, sector, component scores |
| Feature Matrix | `data/processed/feature_matrix.csv` | All 161 features per company (ESG, NLP, categorical) |
| Predictions | `data/processed/predictions.csv` | Proxy score (0-5), binary label, model prediction |

**Implementation -- Searchable Dropdown:**
```python
st.selectbox(
    "Search & Select Company",
    options=company_names,     # All 480 companies
    index=None,                # Nothing selected by default
    placeholder="Start typing to search..."
)
```
Streamlit's `selectbox` with `index=None` creates a native searchable dropdown -- type "Tes" and it filters to "TESLA INC". No custom JavaScript needed.

**9 Report Sections:**

| # | Section | Content | Data Source |
|---|---------|---------|-------------|
| 1 | Risk Overview | 5 KPI cards (score, tier, ESG risk, controversy, rank) + color-coded verdict | `risk_scores` |
| 2 | **Reason Engine (NEW)** | Plain-English bullet points explaining WHY this rating | `feature_matrix` + `predictions` |
| 3 | Risk Breakdown | Bar chart of 5 weighted components + gauge chart | `risk_scores` |
| 4 | ESG Pillar Analysis | Company vs Population bar chart + radar chart + imbalance warning | `feature_matrix` |
| 5 | NLP & Linguistic Analysis | 10 NLP features compared to population avg + linguistic red flag detection | `feature_matrix` |
| 6 | Key Greenwashing Indicators | 6 indicators vs 75th percentile thresholds (HIGH/Normal status) | `feature_matrix` |
| 7 | Sector Peer Comparison | Ranked bar chart within sector + sector rank | `risk_scores` |
| 8 | Model Prediction Summary | Proxy score (0-5), binary label, flagged/not-flagged verdict | `predictions` |
| 9 | Full Feature Profile | Expandable table with all 161 features | `feature_matrix` |

**Greenwashing Reason Engine -- In Depth:**

The Reason Engine is the key differentiator. It does NOT use any API or AI -- it's purely rule-based logic using the project's pre-computed features. It runs **9 checks** on the company's actual data and converts each into a plain-English reason with severity level:

| # | Check | What It Reads | Logic | Example Output |
|---|-------|---------------|-------|----------------|
| 1 | ESG-Controversy Divergence | `esg_controversy_divergence` column | If value > 75th percentile of 480 companies | "ESG risk is low (25.2) but controversy is high (4.0) -- classic divergence" |
| 2 | Controversy-Risk Ratio | `controversy_risk_ratio` column | If value > 75th percentile | "Controversy-Risk Ratio (0.159) is in the top 8%" |
| 3 | Linguistic Greenwashing | `greenwashing_signal_score`, `vague_language_count`, `concrete_evidence_count`, `hedge_language_count`, `superlative_count`, `future_language_count` | If GW signal > 75th pct OR vague > concrete | "Uses 5 vague terms but only 1 concrete evidence point" |
| 4 | Pillar Imbalance | `env_risk_score`, `social_risk_score`, `gov_risk_score`, `pillar_imbalance_score` | If imbalance > 1.5x population average | "Governance (3.8) inconsistent with Environmental (12.1)" |
| 5 | Risk-Controversy Mismatch | `risk_controversy_mismatch` (binary 0/1) | If flag = 1 | "Risk-Controversy MISMATCH detected" |
| 6 | Statistical Anomaly | `combined_anomaly_score` | If value > 75th percentile | "Statistical anomaly detected (score = 2.31)" |
| 7 | Sentiment Puffery | `text_polarity` | If value in top 15% | "Overly positive sentiment in corporate text" |
| 8 | Sector Ranking | `risk_score` grouped by sector | If ranked #1-3 in sector OR > sector_avg + 10 | "Ranked #3 highest risk out of 25 in Energy sector" |
| 9 | Proxy Score | `gw_proxy_score` (0-5) | If >= 3 indicators triggered | "Triggered 4 out of 5 greenwashing indicators" |

**Each reason is classified as:**
- `CRITICAL` (red) -- Strong greenwashing signal, immediate concern
- `WARNING` (orange) -- Moderate risk, warrants investigation
- `GOOD` (green) -- Positive signal, no concern on this metric

**Display format (dark-themed box):**
```
🚨 ADANI ENTERPRISES -- HIGH RISK
   3 critical | 2 warnings | 1 positive signals

🚨 ESG risk is low (25.2) but controversy is high (4.0)    [CRITICAL]
   ESG-Controversy Divergence = 1.035 (higher than 89% of companies)

🚨 Controversy-Risk Ratio (0.159) is in the top 8%         [CRITICAL]
   This is the #1 predictive feature in the model

⚠️ More vague promises (5) than concrete evidence (1)      [WARNING]
   GW signal = 0.523. Hedge words: 3, Superlatives: 2

✅ ESG pillars are balanced (E=12.1, S=8.5, G=3.8)         [GOOD]
   Imbalance = 2.89 (avg = 2.89). No sign of selective reporting.
```

**Implementation code pattern:**
```python
# Each check follows this pattern:
value = company_fm.get('esg_controversy_divergence', 0)
threshold = fm['esg_controversy_divergence'].quantile(0.75)

if value > threshold:
    reasons.append({
        'icon': '🚨', 'severity': 'critical',
        'text': f'ESG risk is low ({esg_val}) but controversy is high ({controversy_val})',
        'detail': f'Divergence = {value:.3f} (higher than {percentile}% of companies)',
        'tag': 'CRITICAL',
    })

# Reasons are sorted: critical first, then warning, then good
# Rendered as HTML with custom CSS styling
```

---

### 2. Real-Time Intelligence

**In Brief:** Monitor live ESG news sentiment for any company and compute real-time risk adjustments. Functions like a Bloomberg/AI-powered terminal with breaking alerts.

**Why it matters:** The base pipeline is static -- trained once on historical data. Real-time news can change a company's risk overnight (e.g., an oil spill, a fraud scandal). This page bridges the gap between static ML scores and dynamic real-world events.

**Datasets Used:**
| Dataset | Source | What It Provides |
|---------|--------|------------------|
| Live News | Google News RSS (free, no API key) | Latest ESG-related headlines per company |
| Base Risk Scores | `data/processed/greenwashing_risk_scores.csv` | Starting risk score to adjust |
| Sentiment Lexicon | Built-in (35+ positive, 40+ negative ESG words) | Word-level sentiment weights |

**Data Source -- Google News RSS:**
```
https://news.google.com/rss/search?q={company_name}+ESG+sustainability&hl=en&gl=US
```
- Free, no API key, no rate limits
- Uses Python's built-in `urllib.request` + `xml.etree.ElementTree`
- Fetches up to 15 latest headlines with title, source, date, URL
- Falls back to realistic simulated headlines if network is unavailable (for demo/presentation)

**Live Sentiment Scoring -- Implementation:**
```python
class _LiveSentimentAnalyzer:
    POSITIVE = {'sustainable': 0.9, 'certified': 0.7, 'renewable': 0.9, ...}  # 35+ words
    NEGATIVE = {'scandal': -0.9, 'violation': -0.9, 'fraud': -0.9, ...}       # 40+ words

    def score(text):
        # Tokenize headline, match against lexicon
        raw = sum(matched_scores)
        return raw / sqrt(raw² + 15)  # VADER normalization, range -1 to +1
```

**Risk Delta Calculation:**
```python
recency_weight = exp(-age_hours / 72)           # Recent news = more weight (72h half-life)
weighted_sentiment = sentiment * recency_weight   # Time-weighted sentiment
risk_delta = -avg_weighted_sentiment * 15         # Convert to risk adjustment (-15 to +15)
adjusted_risk = clamp(base_risk + risk_delta, 0, 100)
```

**5 Dashboard Sections:**

| Section | Content | Interactive? |
|---------|---------|-------------|
| Breaking Risk Alerts | Color-coded alert banners: RED (risk jump >= 5), ORANGE (moderate), GREEN (improvement) | Auto-generated |
| Live Risk Gauge | Side-by-side gauge charts: Base (model) vs Adjusted (live) | Plotly gauge |
| News Sentiment Timeline | Scatter plot of sentiment over time + histogram + pie breakdown | Hover, zoom |
| Live News Feed | Color-coded cards per article with source, age, sentiment score | Scrollable |
| Portfolio Risk Monitor | Select multiple companies, scan all at once, Bloomberg-style ticker | Multi-select + button |

**Example output:**
```
ALERT: Tesla risk jumped from 32 → 47 (+15.0) due to recent negative ESG news coverage

▲ TESLA       32 → 47 (+15.0) [ALERT]
▲ EXXON       45 → 52 (+7.0)  [WATCH]
▼ APPLE       28 → 22 (-6.0)  [IMPROVED]
— MICROSOFT   18 → 19 (+1.0)  [STABLE]
```

---

### 3. ESG Report Analyzer (AI-Powered)

**In Brief:** Upload any PDF ESG/sustainability report and get an AI-powered greenwashing analysis with claim-by-claim color-coded assessment. The only page that uses an external LLM.

**Why it matters:** Companies publish annual ESG/sustainability reports to showcase their environmental and social efforts. These reports are the primary vehicle for greenwashing -- they contain carefully crafted language designed to appear more sustainable than reality. This tool lets you upload any such report and get an instant greenwashing analysis.

**Datasets Used:**
| Dataset | Source | What It Provides |
|---------|--------|------------------|
| User-uploaded PDF | Any ESG report | Raw text content of the report |
| Existing ML Data | `data/processed/greenwashing_risk_scores.csv` | Cross-check if company is in our 480-company database |
| Gemini LLM | Google Gemini 2.0 Flash API | AI-powered claim extraction and analysis |

**LLM: Google Gemini 2.0 Flash**
- Free tier: 15 requests/minute, 1,500 requests/day, $0 cost forever
- No billing or credit card required
- API key stored in `.env` file (auto-loaded, git-ignored)

**How to get the API key:**
1. Go to https://aistudio.google.com/apikey
2. Click "Create API Key" (select any Google Cloud project)
3. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

**End-to-End Pipeline:**
```
PDF Upload → PyPDF2 text extraction (seek(0) + PdfReader)
    → Text validation (min 50 chars, reject scanned/image PDFs)
    → Prompt construction (report text + company data if available)
    → Gemini API call (temperature=0.3, max_tokens=8000)
    → JSON response parsing (with regex fallback for malformed JSON)
    → Display results with color-coded claim cards
```

**Gemini Prompt Design:**
```
Role: "Expert ESG analyst specializing in greenwashing detection"
Task: Analyze report text and return structured JSON with:
  - company_name, overall_risk (HIGH/MEDIUM/LOW), overall_score (0-100)
  - claims[] with: claim_text, category, risk_level, issue, explanation, evidence_type
  - red_flags[], positive_signals[], recommendations[]
Rules:
  1. Extract 8-15 specific claims
  2. Check for: vague language, missing data, future-only promises, superlatives
  3. Cross-reference with company data if available
  4. Temperature: 0.3 (factual, not creative)
```

**Claim Classification:**
| Risk Level | Color | Evidence Type Examples |
|------------|-------|----------------------|
| HIGH | Red border | Vague Promise, Contradicts Data |
| MEDIUM | Yellow border | Unverified, Missing Data |
| LOW | Green border | Credible (verified, measurable) |

**Cross-Check Feature:**
If the uploaded report belongs to a company in our 480-company database, the analyzer compares:
- Report's AI-assessed risk score vs our ML model's risk score
- Flags significant discrepancies (e.g., "Report claims low risk but our model scores 72/100")

**Fallback Mode (no API key):**
Uses the project's own NLP rules (regex pattern matching):
- 12 vague patterns ("committed to", "striving for", etc.)
- 4 superlative patterns ("industry-leading", etc.)
- 4 future patterns ("will", "by 2030", etc.)
- 4 concrete patterns (percentages, ISO, verified, etc.)

**Export Options:**
- Download full analysis as JSON
- Download claims table as CSV

---

### 4. Report Generator

**In Brief:** Generate professional, downloadable HTML reports for any single company, batch of companies, or the entire 480-company portfolio. Reports are styled with CSS, printable to PDF, and include all analysis sections.

**Why it matters:** Stakeholders (investors, compliance officers, ESG analysts) need formal documentation they can share, archive, and reference. This feature turns the dashboard's interactive analysis into a portable, printable report format.

**Datasets Used:**
| Dataset | File | What It Provides |
|---------|------|------------------|
| Risk Scores | `data/processed/greenwashing_risk_scores.csv` | Risk score, tier, components, sector |
| Feature Matrix | `data/processed/feature_matrix.csv` | ESG pillars, NLP features, indicators |
| Predictions | `data/processed/predictions.csv` | Proxy score, binary label |
| Feature Importance | `data/processed/feature_importance_gradient_boosting.csv` | Top features (passed to function) |

**3 Report Modes:**

| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| Single Company | Select 1 company | Full 7-section HTML report + summary CSV | Deep dive on one company |
| Batch | Select multiple companies | Individual report per company + batch summary CSV | Compare a shortlist |
| Full Portfolio | All 480 companies | Portfolio overview with sector breakdown | Executive summary |

**Single Company Report -- 7 Sections:**
1. **Executive Summary** -- Cover page with gradient styling, risk score, tier, verdict, model prediction
2. **Risk Score Breakdown** -- 5 components with visual progress bars (40% proxy, 15% linguistic, 15% divergence, 15% credibility, 15% controversy ratio)
3. **ESG Pillar Analysis** -- E/S/G score cards, pillar imbalance, controversy divergence, mismatch flag
4. **NLP & Linguistic Analysis** -- 4 KPI cards + 7-row linguistic features table with risk implications
5. **Key Greenwashing Indicators** -- 5 indicators with values, thresholds, and HIGH/Normal status
6. **Sector Peer Comparison** -- Rank within sector, company vs sector average
7. **Methodology** -- Data sources, company count, feature count, model performance, 7-phase pipeline

**Implementation:**
- Pure HTML+CSS (no JavaScript, no external dependencies)
- Professional styling: gradient cover page, CSS grid KPI cards, color-coded tables, progress bars
- Generated using Python f-strings with the company's actual data values
- Previewed in dashboard using `st.components.v1.html(html, height=800, scrolling=True)`
- Downloaded via `st.download_button` with `mime="text/html"`
- HTML can be opened in any browser and printed to PDF via Ctrl+P

**Full Portfolio Report includes:**
- 4 KPI cards: Total companies, High Risk, Moderate, Low Risk counts
- Sector summary table with `groupby().agg()`: companies, avg risk, max risk, high risk count
- Top 20 highest risk companies table sorted by risk score

---

### 5. Advanced Explainability (Beyond SHAP)

**In Brief:** Three tabs providing counterfactual explanations ("what must change to flip the prediction"), interactive what-if sliders (live prediction updates), and feature sensitivity curves (how much each feature matters).

**Why it matters:** Standard SHAP shows which features are important globally, but doesn't answer actionable questions like "what should this company DO to reduce its risk?" or "how sensitive is this specific prediction to controversy?" This page answers those questions with interactive tools.

**Datasets Used:**
| Dataset | File | What It Provides |
|---------|------|------------------|
| Feature Matrix | `data/processed/feature_matrix.csv` | All 161 features for all 480 companies |
| Feature Importance | `data/processed/feature_importance_gradient_boosting.csv` | Top features to analyze |

**Note:** This page does NOT use pre-saved models. It trains a **lightweight Gradient Boosting model** on startup using the same proxy label construction as `model_training.py`. The model is cached with `@st.cache_resource` so it only trains once per session.

**Model Training (happens once, cached):**
```python
# Same proxy labels as model_training.py:
# 5 indicators, each triggered at 75th percentile
# Label = 1 if proxy_score >= 2
model = GradientBoostingClassifier(n_estimators=200, max_depth=5, lr=0.1)
model.fit(StandardScaler().fit_transform(X), y)
# Cached -- all subsequent predictions use this model
```

**Tab 1: Counterfactual Explanations**

*Question: "What is the smallest change to ONE feature that would flip this company's prediction?"*

**Algorithm:**
```
For each of the top 15 important features:
    1. Get current value for this company
    2. If FLAGGED → search toward population minimum
       If NOT FLAGGED → search toward population maximum
    3. Generate 30 test values from current → extreme
    4. For each test value:
       - Replace ONLY this feature in the 161-feature vector
       - Scale → predict with cached model
       - If prediction FLIPS → record as counterfactual, stop
    5. Compute change in std deviations for comparison
Sort by |change_in_std| → smallest = easiest to achieve
```

**Example output:**
```
EASIEST PATH TO LOW RISK:
If controversy_risk_ratio decreased from 0.4200 to 0.1800 ↓
→ Risk becomes NOT FLAGGED (probability: 32.1%)
Change: -0.2400 (-1.2 std deviations)
```

**Tab 2: What-If Sensitivity Sliders**

*Question: "If I change feature values right now, what happens?"*

- User selects up to 8 features, gets interactive sliders (range: 1st-99th percentile)
- Every slider drag triggers: copy vector → replace feature → scale → predict → display
- Shows Before vs After gauges, probability shift, "PREDICTION FLIPPED!" alert
- All computation is instant (<1ms per prediction)

**Tab 3: Feature Sensitivity Analysis**

*Question: "How sensitive is this prediction to each feature across its full range?"*

- Sweeps each selected feature across 50 values (2nd-98th percentile)
- Plots sensitivity curve: X = feature value, Y = GW probability
- Red star = current position, dashed line at 50% = decision boundary
- Summary table: sensitivity range, can flip?, probability at extremes
- Steep curve = critical feature; flat curve = doesn't matter for this company

---

### 6. Company Comparison

**In Brief:** Side-by-side comparison of 2-3 companies plus industry average across all metrics: risk scores, ESG pillars, risk components, NLP features, greenwashing indicators, and sector positioning.

**Why it matters:** Investors comparing ESG performance across companies need a single view showing how companies stack up. A company might look good in isolation but poor relative to peers. This page provides that comparative context.

**Datasets Used:**
| Dataset | File | What It Provides |
|---------|------|------------------|
| Risk Scores | `data/processed/greenwashing_risk_scores.csv` | Risk score, tier, components for all 480 companies |
| Feature Matrix | `data/processed/feature_matrix.csv` | ESG pillars, NLP features, indicators for all companies |

**8 Comparison Sections:**

| # | Section | What It Shows | Chart Type |
|---|---------|---------------|------------|
| 1 | Head-to-Head KPIs | Side-by-side metric cards with LOWEST/HIGHEST RISK badges | KPI cards |
| 2 | Risk Score Comparison | Companies + industry avg + population avg | Grouped bar chart |
| 3 | ESG Pillar Radar | Overlapping E/S/G/Controversy profiles (normalized 0-1) | Radar chart |
| 4 | Risk Component Breakdown | 5 risk components per company | Grouped bar chart |
| 5 | NLP Linguistic Comparison | 9 NLP metrics (GW signal, vague, concrete, hedge, etc.) | Grouped bar + table |
| 6 | Key Indicators Heatmap | 6 greenwashing indicators color-coded | Plotly heatmap + status table |
| 7 | Sector Peer Positioning | All 480 companies with selected ones highlighted | Scatter plot |
| 8 | Complete Comparison Table | 17 metrics side-by-side + industry average | Data table + CSV export |

**Implementation Details:**
- Industry average is auto-computed from the most common sector among selected companies using `df[df['sector'] == sector].mean()`
- Radar chart normalizes all values to 0-1 using `(value - min) / (max - min)` for fair comparison
- Heatmap uses `pd.DataFrame.pivot()` to reshape indicator data into a matrix
- LOWEST RISK / HIGHEST RISK badges auto-assigned by comparing risk scores
- All 17 comparison metrics exportable as CSV via `st.download_button`

**Metrics Compared:**
Risk Score, Total ESG Risk, Controversy, Risk Tier, Proxy Score, Environmental Risk, Social Risk, Governance Risk, Pillar Imbalance, Controversy-Risk Ratio, ESG-Controversy Divergence, GW Linguistic Score, Vague Language Count, Concrete Evidence Count, Text Sentiment, Readability, Anomaly Score

---

### 7. Time-Series Risk Tracking

**In Brief:** Historical ESG trends from 2015-2025 with interactive timeline, before-vs-after controversy analysis, carbon/environmental trends, trend forecasting with confidence intervals, and industry benchmark comparison.

**Why it matters:** A static snapshot doesn't tell you if a company is improving or deteriorating. Time-series analysis reveals trends, identifies turning points (like a controversy event), and forecasts future risk direction. This transforms the project from a one-time assessment into a dynamic monitoring tool.

**Datasets Used:**
| Dataset | File | Size | What It Provides |
|---------|------|------|------------------|
| ESG Financial Time-Series | `data/company_esg_financial_dataset.csv` | 11,000 rows (1,000 companies x 11 years) | ESG_Overall, ESG_Environmental, ESG_Social, ESG_Governance, CarbonEmissions, WaterUsage, EnergyConsumption, Revenue, ProfitMargin, GrowthRate |
| Risk Scores | `data/processed/greenwashing_risk_scores.csv` | 480 rows | Current risk scores for reference |

**Time range:** 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

**Industries in time-series data:** Healthcare (1,331 rows), Manufacturing (1,287), Transportation (1,287), Consumer Goods (1,276), Finance (1,243), Energy (1,188), Utilities (1,177), Retail (1,166), Technology (1,045)

**Risk Score Computation Per Year (from real data):**
```python
esg_risk_proxy     = 100 - ESG_Overall                    # Lower ESG = higher risk
carbon_intensity   = CarbonEmissions / Revenue             # Emissions per dollar
controversy_proxy  = MinMax_normalize(carbon_intensity) * 5  # Scale to 0-5
pillar_imbalance   = std(Environmental, Social, Governance)  # Row-wise std
divergence         = controversy_proxy - (esg_risk_proxy / 20)

risk_score = 35% * esg_risk_proxy
           + 25% * (controversy_proxy * 20)
           + 20% * (pillar_imbalance * 3)
           + 20% * (divergence.clip(0) * 15)
# Normalized to 0-100 using MinMax scaling
```

**7 Dashboard Sections:**

| # | Section | Content | Chart Type |
|---|---------|---------|------------|
| 1 | Risk Score Over Time | 11-year risk timeline + forecast with 95% confidence band + risk tier backgrounds | Line + area |
| 2 | ESG Pillar Trends | E/S/G/Overall scores over time + pillar imbalance area chart | Multi-line + area |
| 3 | Before vs After Controversy | Slider to pick pivot year → grouped bar showing all metrics before/after + waterfall chart of % changes | Bar + waterfall |
| 4 | Environmental Impact | Carbon emissions bars, carbon intensity line, energy + water area charts | Bar + line + area |
| 5 | Trend Prediction | Linear regression forecast + confidence intervals + multi-metric forecast | Line + confidence band |
| 6 | Industry Trend Comparison | Company vs industry average ESG over time + gap analysis (green=above, red=below) | Line + bar |
| 7 | Data Table + Export | Full time-series data table + CSV download | Table + download |

**Forecasting -- Implementation:**
```python
from sklearn.linear_model import LinearRegression

X = years[2015..2025].reshape(-1, 1)    # 11 real data points
y = risk_scores[2015..2025]             # Computed from actual ESG data

model = LinearRegression().fit(X, y)
future_risk = model.predict([[2026], [2027], [2028]])

# 95% confidence interval
residuals = y - model.predict(X)
std_err = std(residuals)
upper_bound = future_risk + 1.96 * std_err
lower_bound = future_risk - 1.96 * std_err
```

**Before vs After Analysis:**
- User selects any year (2016-2024) as a "controversy event" using a slider
- System splits data into Before (< pivot year) and After (>= pivot year)
- Computes mean of all metrics in each period
- Displays grouped bar chart (before=blue, after=red) + waterfall chart of % changes
- Auto-generates narrative: "Risk increased by 12.3 points after 2020. ESG performance deteriorated."

**Industry Benchmark:**
- Computes industry average using `ts_data[ts_data['Industry'] == company_industry].groupby('Year').mean()`
- Plots company line vs industry average line
- Gap analysis: bar chart showing per-year delta (green = above industry avg, red = below)

---

### 8. Greenwashing Reason Engine (Embedded in Company Search)

**In Brief:** A rule-based system that converts raw ML output into plain-English bullet-point explanations. Appears automatically in the Company Search & Analysis page, right after the verdict banner.

**Why it matters:** A risk score of "67.3" means nothing to a non-technical user. The Reason Engine translates it into: "HIGH RISK because: ESG claims low risk but controversy is high, uses excessive vague language, ranked #3 worst in sector." This is the bridge between data science and business value.

**Datasets Used:**
| Dataset | File | Columns Used |
|---------|------|-------------|
| Feature Matrix | `data/processed/feature_matrix.csv` | `esg_controversy_divergence`, `controversy_risk_ratio`, `greenwashing_signal_score`, `vague_language_count`, `concrete_evidence_count`, `hedge_language_count`, `superlative_count`, `future_language_count`, `pillar_imbalance_score`, `env_risk_score`, `social_risk_score`, `gov_risk_score`, `risk_controversy_mismatch`, `combined_anomaly_score`, `text_polarity` |
| Risk Scores | `data/processed/greenwashing_risk_scores.csv` | `risk_score`, `sector`, `total_esg_risk_score`, `controversy_score` |
| Predictions | `data/processed/predictions.csv` | `gw_proxy_score`, `gw_label_binary` |

**NO API. NO AI. 100% rule-based on pre-computed features.**

**9 Checks Performed:**

| # | Check Name | Feature(s) Used | Threshold | CRITICAL If | WARNING If | GOOD If |
|---|-----------|-----------------|-----------|-------------|------------|---------|
| 1 | ESG-Controversy Divergence | `esg_controversy_divergence` | 75th percentile of 480 companies | Value > 75th pct | Value > median | Value <= median |
| 2 | Controversy-Risk Ratio | `controversy_risk_ratio` | 75th percentile | Value > 75th pct | Value > median | (not shown) |
| 3 | Linguistic Greenwashing | `greenwashing_signal_score`, `vague_language_count`, `concrete_evidence_count`, `hedge_language_count`, `superlative_count`, `future_language_count` | GW signal > 75th pct | GW signal > 75th pct | vague > concrete AND vague > 2 | concrete > vague AND concrete > 1 |
| 4 | Pillar Imbalance | `env_risk_score`, `social_risk_score`, `gov_risk_score`, `pillar_imbalance_score` | 1.5x population mean | (not used as critical) | Imbalance > 1.5x mean | Imbalance <= 1.5x mean |
| 5 | Risk-Controversy Mismatch | `risk_controversy_mismatch` | Binary flag = 1 | Flag = 1 | (not used) | (not shown) |
| 6 | Statistical Anomaly | `combined_anomaly_score` | 75th percentile | (not used as critical) | Value > 75th pct | (not shown) |
| 7 | Sentiment Puffery | `text_polarity` | 85th percentile | (not used as critical) | Value > 85th pct | (not shown) |
| 8 | Sector Ranking | `risk_score` grouped by `sector` | Top 3 in sector | Rank <= 3 AND sector >= 5 companies | Score > sector_avg + 10 | Score < sector_avg - 10 |
| 9 | Proxy Score | `gw_proxy_score` | >= 3 out of 5 | Score >= 3 | Score = 2 | (not shown) |

**Each reason includes:**
- **Icon** (🚨/⚠️/✅) for quick visual scanning
- **Plain-English text** with actual numbers from the company's data
- **Severity tag** (CRITICAL/WARNING/GOOD) as a colored badge
- **Detail line** in gray with technical specifics and percentile rank

**Sorting:** Reasons are sorted by severity (critical first → warning → good) so the most important concerns appear at the top.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scipy scikit-learn openpyxl xgboost matplotlib seaborn shap plotly streamlit google-generativeai PyPDF2 requests
```

### Option 1: Run Everything (Single Command)

```bash
python model_pipeline.py
```

This runs all 6 phases end-to-end (~5-10 minutes).

### Option 2: Run Step-by-Step

```bash
# Phase 1: Data Preprocessing
python data_preprocessing.py

# Phase 2: NLP Analysis
python nlp_pipeline.py

# Phase 3: Feature Engineering
python feature_engineering_pipeline.py

# Phase 4: Model Training (includes proxy labels + 6 models)
python model_training.py

# Phase 5: Risk Scoring
python risk_scoring.py

# Phase 6: Launch Dashboard
python -m streamlit run streamlit_dashboard.py
```

### Setup Gemini API Key (for ESG Report Analyzer)

```bash
# Create .env file in project root
echo GEMINI_API_KEY=your_api_key_here > .env
```
Get free key at: https://aistudio.google.com/apikey (no billing required)

### Quick Verification

```python
import pandas as pd

# Check feature matrix
fm = pd.read_csv('data/processed/feature_matrix.csv')
print(f"Feature matrix: {fm.shape}")  # Expected: (480, 169)

# Check risk scores
rs = pd.read_csv('data/processed/greenwashing_risk_scores.csv')
print(f"Risk scores: {rs.shape}")     # Expected: (480, 12+)
print(rs[['company_name', 'risk_score', 'risk_tier']].head(10))

# Check model metrics
mm = pd.read_csv('data/processed/model_metrics.csv')
print(mm[['model', 'f1_score', 'roc_auc']])
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.5 | Data manipulation and analysis |
| numpy | >= 1.23 | Numerical computing |
| scipy | >= 1.9 | Statistical functions (z-score, skewness) |
| scikit-learn | >= 1.1 | ML models, preprocessing, metrics |
| xgboost | >= 1.7 | Extreme Gradient Boosting classifier |
| matplotlib | >= 3.6 | Static plot generation |
| seaborn | >= 0.12 | Statistical visualization (confusion matrices) |
| shap | >= 0.42 | Model explainability (SHAP values) |
| plotly | >= 5.0 | Interactive charts (dashboard) |
| streamlit | >= 1.25 | Web dashboard framework |
| openpyxl | >= 3.0 | Reading Excel (.xlsx) files |
| google-generativeai | >= 0.5 | Google Gemini LLM API (ESG Report Analyzer) |
| PyPDF2 | >= 3.0 | PDF text extraction (report upload) |
| requests | >= 2.28 | HTTP requests (news fetching) |

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 15 |
| Total lines of code | ~15,000+ |
| Processed output files | 25 |
| Visualization plots | 8 static + all interactive Plotly charts |
| Companies analyzed | 480 |
| Features engineered | 121 (+43 scaled = 161 total) |
| ML models trained | 6 (5 supervised + 1 unsupervised) |
| Best model F1 score | 0.9682 |
| Best model ROC-AUC | 0.9979 |
| Greenwashing flagged | 145/480 (30.2%) |
| High risk companies | 9/480 (1.9%) |
| Pipeline execution time | ~5 minutes |
| Dashboard pages | 10 (5 core + 5 advanced) |
| LLM integration | Google Gemini 2.0 Flash (free tier) |
| Real-time data source | Google News RSS (free, no API key) |
| Report formats | HTML (downloadable, printable to PDF) |
