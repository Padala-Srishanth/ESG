"""
NLP ESG Claim Extraction Module
==================================
This module extracts and classifies ESG-related claims from corporate text data.

Key capabilities:
1. ESG Claim Detection   - Identifies sentences that contain ESG-related claims
2. Claim Classification  - Classifies claims into E (Environmental), S (Social), G (Governance)
3. Claim Strength Scoring - Rates claim strength (vague vs specific/quantified)
4. Commitment Extraction  - Detects forward-looking commitments and targets
5. Claim Credibility Assessment - Evaluates if claims have supporting evidence

Research basis:
- TerraChoice "Seven Sins of Greenwashing" (2010)
- Lyon & Montgomery "The Means and End of Greenwash" (2015)
- GRI Standards for Sustainability Reporting

Author: Team-18 (VNR VJIET)
Project: ESG Greenwashing Detection using Explainable ML
"""

import re                      # regular expressions for text pattern matching
import numpy as np             # numpy for numerical computations
import pandas as pd            # pandas for DataFrame operations
from collections import Counter  # Counter for frequency analysis


# ============================================================
# 1. SENTENCE TOKENIZER
# ============================================================

def split_into_sentences(text):
    """
    Split text into individual sentences using regex-based rules.
    Handles abbreviations (e.g., Inc., Ltd., Dr.) to avoid false splits.

    We need sentence-level splitting because ESG claims are typically
    expressed as individual sentences within longer descriptions.

    Args:
        text (str): Full text to split into sentences
    Returns:
        list: List of individual sentence strings
    """
    if not isinstance(text, str) or len(text) == 0:  # handle invalid input
        return []                                      # return empty list

    # Common abbreviations that contain periods but aren't sentence endings
    abbreviations = [
        'inc', 'ltd', 'corp', 'llc', 'plc', 'co', 'dr', 'mr', 'mrs',
        'ms', 'prof', 'jr', 'sr', 'vs', 'etc', 'approx', 'dept',
        'est', 'govt', 'no', 'vol', 'st', 'ave', 'blvd',
    ]

    # Temporarily replace abbreviation periods with a placeholder
    text_modified = text  # work on a copy
    for abbr in abbreviations:                                      # iterate through each abbreviation
        # Pattern matches the abbreviation followed by a period (case-insensitive)
        pattern = re.compile(r'\b' + abbr + r'\.', re.IGNORECASE)  # \b = word boundary
        text_modified = pattern.sub(abbr + '<PERIOD>', text_modified)  # replace period with placeholder

    # Split on sentence-ending punctuation followed by whitespace and uppercase letter
    # This regex matches: period/exclamation/question + space + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_modified)  # lookbehind and lookahead assertions

    # Restore the abbreviation periods from placeholders
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]  # put periods back

    # Filter out very short fragments (less than 10 characters are likely noise)
    sentences = [s for s in sentences if len(s) >= 10]  # keep only meaningful sentences

    return sentences  # return list of clean sentences


# ============================================================
# 2. ESG CLAIM DETECTION PATTERNS
# ============================================================

# Regex patterns that identify ESG-related claims in corporate text
# Each pattern is a tuple: (pattern_name, regex_pattern, esg_pillar)

ESG_CLAIM_PATTERNS = {
    # ---- ENVIRONMENTAL CLAIMS ----
    'emission_reduction': {
        'pattern': r'(reduc\w+|cut\w*|lower\w*|decreas\w+).{0,40}(emission|carbon|greenhouse|ghg|co2|methane)',
        'pillar': 'Environmental',                    # E pillar
        'claim_type': 'performance',                  # reporting past performance
        'description': 'Claim about reducing emissions',
    },
    'renewable_energy': {
        'pattern': r'(renewable|solar|wind|hydro|geothermal|biomass).{0,30}(energy|power|electricity|source)',
        'pillar': 'Environmental',
        'claim_type': 'performance',
        'description': 'Claim about renewable energy usage',
    },
    'carbon_neutral': {
        'pattern': r'(carbon\s*(neutral|negative|zero|free)|net[\s-]zero|climate\s*neutral)',
        'pillar': 'Environmental',
        'claim_type': 'target',                       # forward-looking target
        'description': 'Carbon neutrality or net-zero claim',
    },
    'waste_management': {
        'pattern': r'(recycl\w+|zero[\s-]waste|circular|waste\s*(reduc|manag|divert|minimi))',
        'pillar': 'Environmental',
        'claim_type': 'performance',
        'description': 'Waste management or recycling claim',
    },
    'water_conservation': {
        'pattern': r'(water).{0,30}(conserv|reduc|efficien|recycl|reuse|harvest|treatment)',
        'pillar': 'Environmental',
        'claim_type': 'performance',
        'description': 'Water conservation or efficiency claim',
    },
    'biodiversity': {
        'pattern': r'(biodiversity|ecosystem|habitat|species|wildlife|conservation|reforestation)',
        'pillar': 'Environmental',
        'claim_type': 'performance',
        'description': 'Biodiversity or ecosystem claim',
    },
    'env_certification': {
        'pattern': r'(iso\s*14001|leed|energy\s*star|green\s*building|epa|certified\s*green)',
        'pillar': 'Environmental',
        'claim_type': 'credential',                   # third-party certification
        'description': 'Environmental certification claim',
    },

    # ---- SOCIAL CLAIMS ----
    'diversity_inclusion': {
        'pattern': r'(divers\w+|inclusi\w+|equity|equal\w*).{0,30}(program|initiative|policy|workforce|board)',
        'pillar': 'Social',                           # S pillar
        'claim_type': 'performance',
        'description': 'Diversity and inclusion claim',
    },
    'employee_safety': {
        'pattern': r'(employee|worker|workplace|occupational).{0,30}(safety|health|wellbeing|welfare|injury)',
        'pillar': 'Social',
        'claim_type': 'performance',
        'description': 'Employee safety or health claim',
    },
    'community_impact': {
        'pattern': r'(communit\w+|philanthropi\w+|volunteer\w*|donat\w+|charit\w+).{0,30}(impact|support|program|invest)',
        'pillar': 'Social',
        'claim_type': 'performance',
        'description': 'Community engagement or philanthropy claim',
    },
    'human_rights': {
        'pattern': r'(human\s*rights|forced\s*labor|child\s*labor|modern\s*slavery|fair\s*trade|living\s*wage)',
        'pillar': 'Social',
        'claim_type': 'performance',
        'description': 'Human rights or labor practices claim',
    },
    'employee_development': {
        'pattern': r'(training|development|education|skill\w*|learning).{0,30}(program|employee|workforce|invest)',
        'pillar': 'Social',
        'claim_type': 'performance',
        'description': 'Employee training and development claim',
    },
    'data_privacy': {
        'pattern': r'(data\s*(privacy|protection|security)|gdpr|cyber\s*security|information\s*security)',
        'pillar': 'Social',
        'claim_type': 'performance',
        'description': 'Data privacy and security claim',
    },

    # ---- GOVERNANCE CLAIMS ----
    'board_independence': {
        'pattern': r'(independent\s*(director|board|member)|board\s*(diversity|independence|oversight))',
        'pillar': 'Governance',                       # G pillar
        'claim_type': 'performance',
        'description': 'Board independence or diversity claim',
    },
    'ethics_compliance': {
        'pattern': r'(ethic\w+|compliance|anti[\s-]corruption|anti[\s-]bribery|code\s*of\s*conduct)',
        'pillar': 'Governance',
        'claim_type': 'performance',
        'description': 'Ethics and compliance program claim',
    },
    'transparency_reporting': {
        'pattern': r'(transparen\w+|disclos\w+|report\w+).{0,30}(esg|sustainability|annual|framework|gri|tcfd|sasb)',
        'pillar': 'Governance',
        'claim_type': 'credential',
        'description': 'Transparency and reporting framework claim',
    },
    'risk_management': {
        'pattern': r'(risk\s*(management|assessment|oversight|mitigation)|enterprise\s*risk)',
        'pillar': 'Governance',
        'claim_type': 'performance',
        'description': 'Risk management practices claim',
    },
    'executive_compensation': {
        'pattern': r'(executive|ceo|management).{0,30}(compensation|pay|remuneration|incentive).{0,30}(esg|sustainability|performance)',
        'pillar': 'Governance',
        'claim_type': 'performance',
        'description': 'ESG-linked executive compensation claim',
    },
    'whistleblower': {
        'pattern': r'(whistleblower|hotline|reporting\s*mechanism|speak[\s-]up|grievance)',
        'pillar': 'Governance',
        'claim_type': 'performance',
        'description': 'Whistleblower or reporting mechanism claim',
    },
}


# ============================================================
# 3. ESG CLAIM EXTRACTOR CLASS
# ============================================================

class ESGClaimExtractor:
    """
    Extracts and classifies ESG claims from corporate text data.

    The extractor:
    1. Splits text into sentences
    2. Matches each sentence against ESG claim patterns
    3. Scores claim strength (vague=low, quantified=high)
    4. Assesses claim credibility (evidence, third-party verification)
    5. Produces per-company aggregate claim metrics
    """

    def __init__(self):
        """Initialize the ESG claim extractor with pattern definitions."""
        self.patterns = ESG_CLAIM_PATTERNS  # load the predefined ESG patterns

        # Quantitative indicator patterns: numbers, percentages, units
        # Claims with quantitative data are stronger and more credible
        self.quantitative_patterns = [
            r'\d+\.?\d*\s*%',                    # percentage (e.g., "25%", "3.5%")
            r'\d+\.?\d*\s*(million|billion|trillion)',  # large numbers with units
            r'\d+\.?\d*\s*(ton|tonne|kg|lb)',     # weight units (emissions)
            r'\d+\.?\d*\s*(kwh|mwh|gwh|twh)',     # energy units
            r'\d+\.?\d*\s*(liter|gallon|cubic)',   # volume units (water)
            r'\$\d+',                              # dollar amounts
            r'(scope\s*[123])',                    # GHG protocol scope references
            r'(20\d{2})',                          # year references (2000-2099)
        ]

        # Third-party verification indicators: suggests claim is independently verified
        self.verification_indicators = [
            'verified', 'audited', 'certified', 'third-party', 'independent',
            'accredited', 'validated', 'assured', 'reviewed by',
            'iso', 'gri', 'tcfd', 'sasb', 'cdp', 'sbti', 'ungc',
            'science-based', 'b corp', 'leed', 'energy star',
        ]

        # Temporal indicators: past (proven) vs future (promised)
        self.past_indicators = [
            'achieved', 'reduced', 'improved', 'completed', 'implemented',
            'reached', 'attained', 'delivered', 'accomplished', 'demonstrated',
            'resulted in', 'led to', 'generated', 'saved', 'prevented',
            'last year', 'in 2023', 'in 2022', 'fiscal year', 'year-over-year',
        ]
        self.future_indicators = [
            'will', 'plan to', 'intend to', 'aim to', 'target',
            'by 2025', 'by 2030', 'by 2040', 'by 2050', 'goal',
            'committed to', 'pledge', 'ambition', 'roadmap', 'aspire',
            'future', 'upcoming', 'next year', 'going forward',
        ]

    def extract_claims(self, text):
        """
        Extract all ESG claims from a text and return detailed claim information.

        For each detected claim, records:
        - The matched sentence text
        - Which ESG pillar it belongs to (E, S, or G)
        - The claim type (performance, target, credential)
        - Claim strength score (how specific/quantified the claim is)
        - Whether it has third-party verification
        - Whether it's past performance or future promise

        Args:
            text (str): Corporate description or report text
        Returns:
            list: List of dictionaries, each representing one extracted claim
        """
        if not isinstance(text, str) or len(text) == 0:  # handle invalid input
            return []  # no claims in empty text

        sentences = split_into_sentences(text)  # split text into individual sentences
        claims = []  # list to accumulate extracted claims

        # Check each sentence against all ESG claim patterns
        for sentence in sentences:                         # iterate through each sentence
            sentence_lower = sentence.lower()              # lowercase for case-insensitive matching

            for pattern_name, pattern_info in self.patterns.items():  # iterate through each ESG pattern
                # Try to match the regex pattern against this sentence
                match = re.search(pattern_info['pattern'], sentence_lower, re.IGNORECASE)

                if match:  # if the pattern matched this sentence
                    # Score the claim strength (how specific/quantified it is)
                    strength = self._score_claim_strength(sentence_lower)

                    # Check for third-party verification
                    has_verification = self._check_verification(sentence_lower)

                    # Determine temporal orientation (past achievement vs future promise)
                    temporal = self._classify_temporal(sentence_lower)

                    # Build the claim record
                    claim = {
                        'sentence': sentence.strip(),            # the original sentence text
                        'pattern_name': pattern_name,            # which pattern matched
                        'pillar': pattern_info['pillar'],        # E, S, or G classification
                        'claim_type': pattern_info['claim_type'],  # performance/target/credential
                        'description': pattern_info['description'],  # human-readable description
                        'matched_text': match.group(),           # the specific text that matched
                        'strength_score': strength,              # 0-1 strength score
                        'has_verification': has_verification,    # boolean: third-party verified?
                        'temporal': temporal,                     # 'past', 'future', or 'unspecified'
                    }
                    claims.append(claim)  # add claim to results list
                    break  # move to next sentence (avoid double-counting same sentence)

        return claims  # return all extracted claims

    def _score_claim_strength(self, text):
        """
        Score the strength of an ESG claim from 0 (very vague) to 1 (very specific).

        Strength scoring criteria:
        - Quantitative data (numbers, percentages) = +0.3
        - Specific timeframe = +0.2
        - Third-party reference = +0.2
        - Action verbs (achieved, implemented) = +0.15
        - Comparison/benchmark = +0.15

        Args:
            text (str): Lowercased sentence text
        Returns:
            float: Strength score between 0.0 and 1.0
        """
        score = 0.0  # start with zero strength

        # Check for quantitative data (strongest evidence of specificity)
        for pattern in self.quantitative_patterns:       # iterate through quantitative patterns
            if re.search(pattern, text, re.IGNORECASE):  # if any quantitative pattern matches
                score += 0.3                              # add 0.3 for quantitative evidence
                break                                     # only count once even if multiple matches

        # Check for specific timeframe references
        if re.search(r'(20\d{2}|fiscal year|quarter|annual|year-over-year)', text):
            score += 0.2  # add 0.2 for temporal specificity

        # Check for third-party verification references
        if any(indicator in text for indicator in self.verification_indicators):
            score += 0.2  # add 0.2 for verification evidence

        # Check for strong action verbs (indicate completed actions, not just plans)
        action_verbs = ['achieved', 'implemented', 'completed', 'reduced', 'invested',
                        'installed', 'launched', 'deployed', 'measured', 'reported']
        if any(verb in text for verb in action_verbs):
            score += 0.15  # add 0.15 for concrete action language

        # Check for comparison or benchmark references
        if re.search(r'(compared to|baseline|benchmark|year-over-year|vs|versus)', text):
            score += 0.15  # add 0.15 for comparative context

        return min(score, 1.0)  # cap at 1.0 maximum

    def _check_verification(self, text):
        """
        Check if a claim references third-party verification or certification.
        Verified claims are more credible and less likely to be greenwashing.

        Args:
            text (str): Lowercased sentence text
        Returns:
            bool: True if third-party verification is referenced
        """
        # Return True if any verification indicator is found in the text
        return any(indicator in text for indicator in self.verification_indicators)

    def _classify_temporal(self, text):
        """
        Classify whether a claim refers to past performance or future promises.
        Greenwashing often relies on future promises without current evidence.

        Args:
            text (str): Lowercased sentence text
        Returns:
            str: 'past' (proven), 'future' (promised), or 'unspecified'
        """
        has_past = any(indicator in text for indicator in self.past_indicators)      # check for past language
        has_future = any(indicator in text for indicator in self.future_indicators)  # check for future language

        if has_past and not has_future:     # only past indicators found
            return 'past'                    # claim refers to proven past performance
        elif has_future and not has_past:   # only future indicators found
            return 'future'                  # claim is a forward-looking promise
        elif has_past and has_future:       # both found (e.g., "achieved X, target Y by 2030")
            return 'past'                    # give benefit of doubt to past evidence
        else:
            return 'unspecified'             # no temporal orientation detected


# ============================================================
# 4. AGGREGATE CLAIM METRICS PER COMPANY
# ============================================================

def compute_claim_metrics(claims):
    """
    Compute aggregate metrics from a list of extracted claims for one company.
    These metrics become features for the ML greenwashing detection model.

    Metrics include:
    - Total claim count and per-pillar counts
    - Average claim strength
    - Ratio of verified vs unverified claims
    - Ratio of past vs future claims
    - Claim diversity (how many pillars are covered)
    - Greenwashing risk indicators

    Args:
        claims (list): List of claim dictionaries from ESGClaimExtractor
    Returns:
        dict: Aggregated claim metrics as a flat dictionary
    """
    if not claims or len(claims) == 0:  # handle empty claims list
        return {
            'total_claims': 0,                    # no claims found
            'env_claims': 0,                      # no environmental claims
            'social_claims': 0,                   # no social claims
            'gov_claims': 0,                      # no governance claims
            'avg_claim_strength': 0.0,            # no strength to average
            'verified_claim_ratio': 0.0,          # no verified claims
            'past_claim_ratio': 0.0,              # no past-oriented claims
            'future_claim_ratio': 0.0,            # no future-oriented claims
            'performance_claim_ratio': 0.0,       # no performance claims
            'target_claim_ratio': 0.0,            # no target claims
            'credential_claim_ratio': 0.0,        # no credential claims
            'claim_pillar_diversity': 0,          # no pillar coverage
            'strong_claim_ratio': 0.0,            # no strong claims
            'weak_claim_ratio': 0.0,              # no weak claims
            'claim_credibility_score': 0.0,       # overall credibility
        }

    total = len(claims)  # total number of claims extracted

    # --- Per-pillar claim counts ---
    pillar_counts = Counter(c['pillar'] for c in claims)  # count claims per ESG pillar
    env_claims = pillar_counts.get('Environmental', 0)     # environmental claim count
    social_claims = pillar_counts.get('Social', 0)         # social claim count
    gov_claims = pillar_counts.get('Governance', 0)        # governance claim count

    # --- Average claim strength ---
    strengths = [c['strength_score'] for c in claims]      # extract all strength scores
    avg_strength = np.mean(strengths)                       # compute mean strength

    # --- Verification ratio: what fraction of claims reference third-party verification ---
    verified_count = sum(1 for c in claims if c['has_verification'])  # count verified claims
    verified_ratio = verified_count / total                            # proportion verified

    # --- Temporal ratios: past achievement vs future promise ---
    past_count = sum(1 for c in claims if c['temporal'] == 'past')      # count past claims
    future_count = sum(1 for c in claims if c['temporal'] == 'future')  # count future claims
    past_ratio = past_count / total                                      # proportion past
    future_ratio = future_count / total                                  # proportion future

    # --- Claim type ratios ---
    type_counts = Counter(c['claim_type'] for c in claims)  # count by claim type
    performance_ratio = type_counts.get('performance', 0) / total  # actual performance claims
    target_ratio = type_counts.get('target', 0) / total            # target/goal claims
    credential_ratio = type_counts.get('credential', 0) / total   # certification claims

    # --- Pillar diversity: how many distinct ESG pillars have claims (0-3) ---
    pillar_diversity = len(pillar_counts)  # number of distinct pillars covered

    # --- Strong vs weak claim ratios ---
    strong_claims = sum(1 for c in claims if c['strength_score'] >= 0.5)  # claims with strength >= 0.5
    weak_claims = sum(1 for c in claims if c['strength_score'] < 0.3)     # claims with strength < 0.3
    strong_ratio = strong_claims / total  # proportion of strong claims
    weak_ratio = weak_claims / total      # proportion of weak claims

    # --- OVERALL CLAIM CREDIBILITY SCORE ---
    # Formula: weights verified claims, past performance, and strong evidence
    # High credibility = more verified, past-oriented, strong claims
    # Low credibility = more unverified, future-oriented, weak claims (greenwashing signal)
    credibility = (
        0.30 * verified_ratio +      # 30% weight on third-party verification
        0.25 * past_ratio +          # 25% weight on past performance evidence
        0.25 * avg_strength +        # 25% weight on claim specificity/strength
        0.10 * credential_ratio +    # 10% weight on certifications
        0.10 * (pillar_diversity / 3)  # 10% weight on balanced ESG coverage (normalized to 0-1)
    )

    return {
        'total_claims': total,                                          # total number of ESG claims extracted
        'env_claims': env_claims,                                       # environmental claims count
        'social_claims': social_claims,                                 # social claims count
        'gov_claims': gov_claims,                                       # governance claims count
        'avg_claim_strength': round(avg_strength, 4),                   # average specificity of claims
        'verified_claim_ratio': round(verified_ratio, 4),               # fraction with 3rd-party verification
        'past_claim_ratio': round(past_ratio, 4),                       # fraction referring to past achievements
        'future_claim_ratio': round(future_ratio, 4),                   # fraction referring to future promises
        'performance_claim_ratio': round(performance_ratio, 4),         # fraction about actual performance
        'target_claim_ratio': round(target_ratio, 4),                   # fraction about targets/goals
        'credential_claim_ratio': round(credential_ratio, 4),           # fraction about certifications
        'claim_pillar_diversity': pillar_diversity,                     # number of ESG pillars covered (0-3)
        'strong_claim_ratio': round(strong_ratio, 4),                   # fraction of strong/specific claims
        'weak_claim_ratio': round(weak_ratio, 4),                       # fraction of weak/vague claims
        'claim_credibility_score': round(credibility, 4),               # overall credibility score (0-1)
    }


# ============================================================
# 5. BATCH PROCESSING FOR DATAFRAMES
# ============================================================

def extract_claims_from_dataframe(df, text_column):
    """
    Apply ESG claim extraction to an entire DataFrame column.
    Extracts claims from each row's text and computes aggregate metrics.

    This is the main function for integrating claim extraction into the pipeline.

    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column with text to analyze
    Returns:
        tuple: (enhanced_df, all_claims_df)
            - enhanced_df: Original DataFrame with new claim metric columns
            - all_claims_df: Detailed DataFrame of every individual claim extracted
    """
    print(f"\n  Extracting ESG claims from '{text_column}'...")
    print(f"    - Processing {len(df)} company descriptions...")

    df = df.copy()  # create copy to avoid modifying original

    extractor = ESGClaimExtractor()  # instantiate the claim extractor

    all_claims = []         # list to store all individual claims across all companies
    all_metrics = []        # list to store per-company aggregate metrics

    # Process each row in the DataFrame
    for idx, row in df.iterrows():                         # iterate through each row
        text = row[text_column]                             # get the text from this row
        claims = extractor.extract_claims(text)             # extract ESG claims from this text

        # Add company identifier to each claim for traceability
        for claim in claims:                                # iterate through extracted claims
            claim['company_index'] = idx                    # record which company this claim belongs to
            if 'company_name' in df.columns:                # if company name column exists
                claim['company_name'] = row['company_name']  # add company name for readability
            all_claims.append(claim)                         # add to master claims list

        # Compute aggregate metrics for this company
        metrics = compute_claim_metrics(claims)              # aggregate claims into metrics
        all_metrics.append(metrics)                          # add to metrics list

    # Convert metrics list to DataFrame and concatenate with original
    metrics_df = pd.DataFrame(all_metrics)                   # each dict becomes a row
    df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)  # add metrics as new columns

    # Create detailed claims DataFrame (every individual claim as a row)
    claims_df = pd.DataFrame(all_claims) if all_claims else pd.DataFrame()

    # Print extraction summary statistics
    print(f"    - Total ESG claims extracted: {len(all_claims)}")
    if len(all_claims) > 0:
        pillar_dist = Counter(c['pillar'] for c in all_claims)  # count claims per pillar
        print(f"    - Environmental claims: {pillar_dist.get('Environmental', 0)}")
        print(f"    - Social claims: {pillar_dist.get('Social', 0)}")
        print(f"    - Governance claims: {pillar_dist.get('Governance', 0)}")
        print(f"    - Avg claim strength: {metrics_df['avg_claim_strength'].mean():.4f}")
        print(f"    - Avg credibility score: {metrics_df['claim_credibility_score'].mean():.4f}")
    print(f"    - Added {len(metrics_df.columns)} claim feature columns")

    return df, claims_df  # return enhanced DataFrame and detailed claims


# ============================================================
# 6. CLAIM ANALYSIS REPORT GENERATOR
# ============================================================

def generate_claim_report(claims_df, top_n=10):
    """
    Generate a human-readable report summarizing extracted ESG claims.
    Useful for manual review and validation of the extraction process.

    Args:
        claims_df (pd.DataFrame): DataFrame of individual extracted claims
        top_n (int): Number of top examples to show per category
    Returns:
        str: Formatted text report
    """
    if claims_df.empty:                                     # handle empty claims DataFrame
        return "No ESG claims were extracted from the data."

    report = []                                             # list to accumulate report lines
    report.append("=" * 70)
    report.append("ESG CLAIM EXTRACTION REPORT")
    report.append("=" * 70)

    # Overall statistics
    total = len(claims_df)                                  # total claims extracted
    report.append(f"\nTotal claims extracted: {total}")

    # Per-pillar breakdown
    pillar_counts = claims_df['pillar'].value_counts()      # count by pillar
    report.append("\n--- Claims by ESG Pillar ---")
    for pillar, count in pillar_counts.items():             # iterate through pillars
        pct = count / total * 100                            # calculate percentage
        report.append(f"  {pillar:20s}: {count:4d} ({pct:.1f}%)")  # formatted output

    # Per-type breakdown
    type_counts = claims_df['claim_type'].value_counts()    # count by claim type
    report.append("\n--- Claims by Type ---")
    for ctype, count in type_counts.items():                # iterate through types
        pct = count / total * 100
        report.append(f"  {ctype:20s}: {count:4d} ({pct:.1f}%)")

    # Temporal orientation breakdown
    temporal_counts = claims_df['temporal'].value_counts()  # count by temporal orientation
    report.append("\n--- Temporal Orientation ---")
    for temp, count in temporal_counts.items():             # iterate through temporal categories
        pct = count / total * 100
        report.append(f"  {temp:20s}: {count:4d} ({pct:.1f}%)")

    # Strength distribution
    strengths = claims_df['strength_score']                 # extract strength scores
    report.append(f"\n--- Claim Strength Statistics ---")
    report.append(f"  Mean:   {strengths.mean():.4f}")      # average strength
    report.append(f"  Median: {strengths.median():.4f}")    # median strength
    report.append(f"  Std:    {strengths.std():.4f}")       # standard deviation
    report.append(f"  Min:    {strengths.min():.4f}")       # weakest claim
    report.append(f"  Max:    {strengths.max():.4f}")       # strongest claim

    # Top strongest claims (examples of credible ESG reporting)
    report.append(f"\n--- Top {top_n} Strongest Claims ---")
    top_strong = claims_df.nlargest(top_n, 'strength_score')  # get top N by strength
    for _, row in top_strong.iterrows():                       # iterate through top claims
        company = row.get('company_name', 'Unknown')           # get company name
        report.append(f"  [{row['pillar']}] ({row['strength_score']:.2f}) {company}")
        report.append(f"    \"{row['sentence'][:120]}...\"")   # truncate long sentences

    # Top weakest claims (examples of vague/greenwashing language)
    report.append(f"\n--- Top {top_n} Weakest Claims (Potential Greenwashing) ---")
    top_weak = claims_df.nsmallest(top_n, 'strength_score')   # get bottom N by strength
    for _, row in top_weak.iterrows():
        company = row.get('company_name', 'Unknown')
        report.append(f"  [{row['pillar']}] ({row['strength_score']:.2f}) {company}")
        report.append(f"    \"{row['sentence'][:120]}...\"")

    return '\n'.join(report)  # join all lines into a single report string


# ============================================================
# MAIN - Standalone Testing
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ESG CLAIM EXTRACTION MODULE")
    print("=" * 60)

    # Test with sample corporate descriptions
    sample_texts = [
        # Company A: Strong, verifiable ESG claims
        """Acme Corp reduced Scope 1 and 2 greenhouse gas emissions by 35% compared to
        the 2019 baseline, as verified by Bureau Veritas. The company achieved ISO 14001
        certification across all manufacturing facilities. Employee safety incidents
        decreased by 42% year-over-year. The board includes 60% independent directors
        with ESG expertise. Total renewable energy usage reached 78% of operations.""",

        # Company B: Vague, greenwashing-style claims
        """GreenWash Inc is committed to becoming the world's most sustainable company.
        We believe in creating a positive environmental impact and are working towards
        a greener future. Our eco-friendly products represent cutting-edge innovation.
        We aspire to achieve carbon neutrality and plan to reduce our environmental
        footprint. The company strives for diversity and inclusion excellence.""",
    ]

    extractor = ESGClaimExtractor()  # create extractor instance

    for i, text in enumerate(sample_texts):             # test each sample
        print(f"\n--- Company {'A (Legitimate)' if i == 0 else 'B (Greenwashing)'} ---")
        claims = extractor.extract_claims(text)          # extract claims
        metrics = compute_claim_metrics(claims)          # compute metrics

        print(f"  Total claims: {metrics['total_claims']}")
        print(f"  Avg strength: {metrics['avg_claim_strength']:.4f}")
        print(f"  Verified ratio: {metrics['verified_claim_ratio']:.4f}")
        print(f"  Past/Future ratio: {metrics['past_claim_ratio']:.2f} / {metrics['future_claim_ratio']:.2f}")
        print(f"  Credibility score: {metrics['claim_credibility_score']:.4f}")
        print(f"  Pillar diversity: {metrics['claim_pillar_diversity']}/3")

        for claim in claims:                             # print individual claims
            print(f"    [{claim['pillar']:15s}] (str={claim['strength_score']:.2f}) {claim['description']}")

    print("\nESG Claim Extraction test complete!")
