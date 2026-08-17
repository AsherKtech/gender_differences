#%%
# ==================================================================================================
# ESS PIPELINE: TEMPORAL STABILITY & CROSS-ROUND GENERALIZATION EXPERIMENT
# ==================================================================================================
#
# OBJECTIVE:
# This script tests the temporal stability and generalizability of gender response patterns across 
# European societies over time. By training on early data (Rounds 1-2) and testing on later data 
# (Round 9), and vice versa, we determine whether ML models learn timeless gender signals or 
# time-bound epoch-specific noise.
#
# CORE HYPOTHESIS: Lower model accuracy = higher gender similarity/convergence. If gender 
# differences become less predictable over time, it suggests convergence in gender roles.
#
# TWO EXPERIMENTAL DIRECTIONS:
# 1. FORWARD PREDICTABILITY (Rounds 1-2 → Rounds 1-9): Tests if historical patterns predict 
#    modern responses. High accuracy = persistent gender differences; low accuracy = convergence.
# 2. BACKWARD PREDICTABILITY (Rounds 8-9 → Rounds 1-9): Tests if modern patterns backcast 
#    historical responses. Asymmetry between forward/backward models reveals changing norms.
#
# ICELAND EXCLUSION STRATEGY:
# Small countries like Iceland have tiny sample sizes in early/late rounds. Including them in 
# train_n calculation would cap training at 80% of ~10-20 samples, crippling the classifier.
# Solution: Drop Iceland from train_n calculation to allow large countries (Germany, UK) to 
# contribute sufficient training data for learning real patterns.
#
# ==================================================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import statsmodels.formula.api as smf


#%%
# --------------------------------------------------------------------------------------------------
# STEP 1: LOAD PREPROCESSED DATA OR PROCESS FROM RAW
# --------------------------------------------------------------------------------------------------
#
# This step loads the raw ESS (European Social Survey) data file and extracts metadata.
# The Sav file contains survey responses from multiple rounds across European countries,
# including gender information and various attitudinal questions that serve as features.
#

# Define the path to the combined ESS dataset subset in SPSS (.sav) format
# This single file contains data from ESS Rounds 1-9, each with different survey designs
FILE_PATH = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')

# Load the SPSS file and extract both data (df_raw) and metadata (meta)
# user_missing=True ensures missing value codes are properly recognized
print("Step 1: Loading raw data and metadata...")
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)


#%%
# Process column names and value labels to standardize the dataset structure.
# Column names in ESS data are often cryptic (e.g., 'gndr', 'q3bapty0') and need 
# transformation into readable feature names. Value labels map numeric codes to 
# meaningful text responses (e.g., 1→'Male', 2→'Female').
#

# Extract the mapping from raw column names to human-readable labels
raw_labels = meta.column_names_to_labels

# Convert space-containing labels to underscore-separated identifiers for Python compatibility
# Example: "Respondent's gender" → "Respondent's_gender"
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}

# Extract country value labels (numeric codes mapped to country names)
# e.g., {1.0: 'Austria', 2.0: 'Belgium', ...}
cntry_val_labels = meta.variable_value_labels.get('cntry', {})

# Identify the gender column name (may be 'gndr' or 'gender' in the dataset)
gender_raw_col = next((c for c in ['gndr', 'gender'] if c in df_raw.columns), 'gndr')

# Filter to include only valid gender responses (1=Male, 2=Female; excluding other codes like 0, 9)
valid_mask = df_raw[gender_raw_col].isin([1, 2, 1.0, 2.0])
df_sub = df_raw[valid_mask].copy()


#%%
# --------------------------------------------------------------------------------------------------
# STEP 2: IDENTIFY COMMON COLUMNS & HANDLE MISSING VALUES
# --------------------------------------------------------------------------------------------------
#
# ESS datasets use specific numeric codes to indicate missing responses:
# - Code 6/66/666/6666: "Not applicable" (valid missingness, e.g., never married asking about spouse)
# - Codes 7/77/777/7777: "Don't know" responses
# - Codes 8/88/888/8888: "No answer" / skipped questions
# - Code 9/99/999/9999: Missing value (data not recorded)
#
# This step identifies which columns have sufficient valid data across country/round combinations,
# ensuring only features with meaningful responses are included in analysis.
#

# Define all missing code values used across ESS surveys
ALL_MISSING_CODES = {
    6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666',
    7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777',
    8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888',
    9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'
}

# Grouping columns: Country identifier and ESS round number (1-9)
group_cols = ['cntry', 'essround']

# All non-group columns are candidate features for analysis
candidate_cols = [c for c in df_sub.columns if c not in group_cols]


#%%
# Define validation function to check if a series has any valid (non-missing) values
def is_strictly_valid(s):
    # Returns True where the value is NOT NA AND NOT in the set of missing codes
    return s.notna() & (~s.isin(ALL_MISSING_CODES))

# For each country/round combination, identify which columns have at least one valid response
valid_per_group = df_sub.groupby(group_cols)[candidate_cols].apply(
    lambda group: group.apply(lambda col: is_strictly_valid(col).any())
)

# Retain only columns that are valid across ALL country/round combinations
# This ensures we only use features consistently available throughout the dataset
retained_cols = valid_per_group.columns[valid_per_group.all()].tolist()

# Create subset DataFrame with grouping columns and retained feature columns
df_subset = df_sub[group_cols + retained_cols].copy()

# Standardize column headers using the readable labels from metadata
df_subset.rename(columns=code_to_label, inplace=True)

# Identify the renamed versions of key grouping columns for later reference
cntry_renamed = code_to_label.get(group_cols[0], group_cols[0])
round_renamed = code_to_label.get(group_cols[1], group_cols[1])
gender_renamed = code_to_label.get(gender_raw_col, gender_raw_col)

# Map numeric country codes to actual country names using value labels
df_subset['Country'] = df_subset[cntry_renamed].astype(str).str.strip().map(cntry_val_labels).fillna(df_subset[cntry_renamed])

# Store ESS round number as a clean feature column
df_subset['ESS_round'] = df_subset[round_renamed]

# Store gender values (1=Male, 2=Female) as the target variable
df_subset['Gender'] = df_subset[gender_renamed]


#%%
# Exclude columns that are not useful for gender prediction or analysis:
# - Survey design identifiers and weights (not predictive of gender)
# - Demographic info about household members (irrelevant to respondent's gender)
# - Metadata fields (dataset title, production date, etc.)
#
# These columns would add noise without contributing meaningful signal.
#

cols_to_drop = [
    group_cols[0], group_cols[1], gender_raw_col, "Title_of_dataset", "Edition", "Production_date", 
    "Respondent's_identification_number", "Design_weight", "Post-stratification_weight_including_design_weight",
    "Population_size_weight_(must_be_combined_with_dweight_or_pspwght)", "Country_of_birth", 
    "Discrimination_of_respondent's_group:_gender", "Discrimination_of_respondent's_group:_other_grounds", 
    "Country_of_birth,_father", "Language_most_often_spoken_at_home:_first_mentioned", "Country_of_birth,_mother",
    "nan_count", "Citizenship", "Language_most_often_spoken_at_home:_second_mentioned", "Region", 
    "Gender_of_second_person_in_household", "Gender_of_third_person_in_household",
    "Gender_of_fourth_person_in_household", "Gender_of_fifth_person_in_household",
    "Gender_of_sixth_person_in_household", "Year_of_birth_of_second_person_in_household",
    "Year_of_birth_of_third_person_in_household"
]

# Drop the specified columns, ignoring any that don't exist (errors='ignore')
df_subset.drop(columns=[c for c in cols_to_drop if c in df_subset.columns], errors='ignore', inplace=True)


#%%
# --------------------------------------------------------------------------------------------------
# STEP 3: ICELAND EXCLUSION & HELPER FUNCTIONS
# --------------------------------------------------------------------------------------------------
#
# This step creates indicator flags for missingness patterns and converts all features 
# to numeric format suitable for machine learning. Many ESS questions use "Not applicable" 
# codes (6, 66, etc.) which are valid responses indicating the question doesn't apply.
# Other missing codes (7, 8, 9 series) represent actual missing data that should be handled.
#
# The transformed DataFrame includes:
# - Original feature values (numeric, with missing/NA treated as 0)
# - Binary flags indicating whether each response was "Not applicable"
# - Binary flags indicating whether each response was "Missing" or NA
#

# Define codes that mean "Not applicable" (valid missingness, not data error)
NOT_APPLICABLE_CODES = {6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666'}

# Define codes that represent actual missing data (don't know, no answer, etc.)
OTHER_MISSING_CODES = ALL_MISSING_CODES - NOT_APPLICABLE_CODES

# Identify all columns that will be used as features for prediction
base_features = [c for c in df_subset.columns if c not in ['Country', 'ESS_round', 'Gender']]

# Build transformed dictionary with Country, ESS_round, Gender, and processed feature columns
transformed = {
    'Country': df_subset['Country'], 
    'ESS_round': df_subset['ESS_round'], 
    'Gender': df_subset['Gender']
}

# Process each base feature column:
for c in base_features:
    s = df_subset[c]                    # Get the series for this feature
    is_na = s.isin(NOT_APPLICABLE_CODES)   # Flag "Not applicable" responses (6, 66, etc.)
    is_miss = s.isin(OTHER_MISSING_CODES) | s.isna()  # Flag actual missing values
    
    # Create indicator: 1 if response was "Not applicable", else 0
    transformed[f"{c}_is_na"] = is_na.astype(int)
    
    # Create indicator: 1 if response was missing/NA, else 0
    transformed[f"{c}_is_missing"] = is_miss.astype(int)
    
    # Convert to numeric, coercing errors to NaN, then set NA/missing to 0.0
    num_s = pd.to_numeric(s, errors='coerce')
    num_s[is_na | is_miss] = 0.0
    transformed[c] = num_s.fillna(0.0)

# Create final processed DataFrame from the transformed dictionary
df_processed = pd.DataFrame(transformed)

# Encode gender labels (1→Male, 2→Female) as integers for ML model compatibility
le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['Gender'].astype(str))


#%%
# Exclude Iceland from training calculations to prevent undersampling bottleneck.
# Iceland typically has very small sample sizes in ESS surveys (often < 100 respondents).
# When calculating train_n based on smallest country/round group, including Iceland
# would result in train_n being only 80% of ~20-30 samples per gender = ~10-15 rows.
# This is insufficient for training a complex model like HistGradientBoostingClassifier.
#
# By excluding Iceland from the train_n calculation, we ensure large countries like 
# Germany, UK, France, etc. can contribute their full sample sizes to the training pool.
#

# Filter out Iceland (and 'is' alias) from the dataset
df_filtered = df_processed[~df_processed['Country'].str.lower().isin(['iceland', 'is'])].copy()

# Extract feature column names for model training
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'Gender', 'target_encoded']]


#%%
# Helper function to evaluate model performance within a country/round group
def eval_group(g):
    # Calculate accuracy: proportion of correct predictions
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    
    # Calculate F1 score (harmonic mean of precision and recall)
    # Using 'binary' average for binary classification with zero_division=0 to handle edge cases
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    
    # Return summary statistics as a Series
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})

# Main experiment runner function that trains on specified rounds and tests on all rounds
def run_temporal_experiment(train_rounds, exp_name):
    print(f"\n==================================================================")
    print(f" RUNNING EXPERIMENT: {exp_name} (Training on Rounds {train_rounds})")
    print(f"==================================================================")

    # 1. Filter dataset to include only responses from training rounds
    # This creates the pool of data available for model training
    train_pool = df_filtered[df_filtered['ESS_round'].isin(train_rounds)].copy()
    
    # 2. Compute train_n threshold based on 80% of smallest country/round group in training pool
    # Group by Country and ESS_round, count respondents in each combination
    counts = train_pool.groupby(['Country', 'ESS_round']).size()
    
    # train_n = 80% of the smallest group size (most constrained by sample)
    # Example: If smallest group has 100 people, train_n = 80 per country/round
    train_n = int(counts.min() * 0.8)
    print(f"Calculated train_n (80% of min group size in rounds {train_rounds}): {train_n}")

    # 3. Sample balanced training data from each country/round combination
    # For each group, sample up to train_n rows with random_state=42 for reproducibility
    # This ensures equal representation across all country/round combinations in training
    training_data = train_pool.groupby(['Country', 'ESS_round'], group_keys=False).apply(
        lambda x: x.sample(n=min(train_n, len(x)), random_state=42), include_groups=False
    )
    
    # Test set contains all remaining data from ALL 9 rounds (not just test rounds)
    # This allows evaluation of how well the model generalizes across the entire dataset
    test_data = df_processed.drop(index=training_data.index).copy()

    # Extract feature matrix X and target vector y for training
    X_train = training_data[feature_cols]
    y_train = training_data['target_encoded']
    
    # Prepare full test set features (across all 9 rounds)
    X_test_full = test_data[feature_cols]

    # 4. Feature Selection via Baseline Model Permutation Importance
    print("Performing feature selection on training split...")
    
    # Train a baseline HistGradientBoostingClassifier on the full feature set
    baseline_hgb = HistGradientBoostingClassifier(random_state=42)
    baseline_hgb.fit(X_train, y_train)

    # Sample up to 10,000 rows for permutation importance calculation
    # Large samples provide more stable importance estimates but are computationally expensive
    X_select_sample = X_train.sample(n=min(10000, len(X_train)), random_state=42)
    
    # Extract corresponding target values for the sampled rows
    y_select_sample = y_train.loc[X_select_sample.index]

    # Calculate permutation importance: shuffle each feature and measure accuracy drop
    # Features causing larger accuracy drops when shuffled are more important
    perm_selection = permutation_importance(
        baseline_hgb, X_select_sample, y_select_sample, n_repeats=2, random_state=42, n_jobs=1
    )

    # Build DataFrame of features and their importance scores
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': perm_selection.importances_mean})
    
    # Select only features with positive importance (importance > 0)
    selected_features = imp_df[imp_df['Importance'] > 0]['Feature'].tolist()
    print(f"Retained {len(selected_features)} features with positive importance out of {len(feature_cols)}.")

    # 5. Retrain model on selected feature subset
    # Use only the important features identified in step 4 for final training
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test_full[selected_features]

    # Create and train the final classifier with selected features
    final_hgb = HistGradientBoostingClassifier(random_state=42)
    final_hgb.fit(X_train_selected, y_train)

    # 6. Evaluate accuracy across all 9 rounds
    # Make predictions on the full test set (all country/round combinations)
    test_data['y_pred'] = final_hgb.predict(X_test_selected)
    
    # Store predicted probabilities for the positive class (gender=Female)
    test_data['y_prob'] = final_hgb.predict_proba(X_test_selected)[:, 1]

    # Calculate performance metrics grouped by Country and ESS_round
    perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()
    
    # Run Mixed-Effects Model to account for country-level random effects
    # This tests whether accuracy varies significantly across rounds after controlling for country effects
    mixed_res = smf.mixedlm("accuracy ~ ESS_round", perf_df, groups=perf_df["Country"]).fit()
    print("\nMixed-Effects Regression Summary:")
    print(mixed_res.summary())

    # Return the performance DataFrame for further analysis or saving
    return perf_df


#%%
# --------------------------------------------------------------------------------------------------
# STEP 4: EXECUTE FORWARD AND BACKWARD EXPERIMENTS
# --------------------------------------------------------------------------------------------------
#
# TWO TEMPORAL DIRECTIONS FOR TESTING GENDER PATTERN STABILITY:
#
# EXPERIMENT 1: FORWARD PREDICTABILITY (Historical → Modern)
# Training: Rounds 1 & 2 (~2002-2004, early European integration era)
# Testing: All 9 rounds (Rounds 1-9, spanning ~2002-2018)
#
# What it tests:
# - Can gender patterns from the early ESS waves predict responses in later waves?
# - High accuracy in Round 9 suggests enduring, stable gender differences
# - Low accuracy in Round 9 suggests convergence or changing response patterns
#
# Research insight: If historical models maintain high accuracy over time, 
# gender differences are structurally stable. A sharp drop indicates significant 
# social change affecting gender-role expression.
#
# --------------------------------------------------------------------
#
# EXPERIMENT 2: BACKWARD PREDICTABILITY (Modern → Historical)
# Training: Rounds 8 & 9 (~2016-2018, contemporary Europe with more gender parity norms)
# Testing: All 9 rounds (Rounds 1-9, entire time series)
#
# What it tests:
# - Can modern gender patterns backcast historical responses?
# - Comparing forward vs. backward model performance reveals asymmetry
# - If backward > forward accuracy, modern norms may have shifted significantly
#
# Research insight: Asymmetry between models helps identify whether changing 
# norms are unidirectional (e.g., only convergence) or bidirectional.
#
# --------------------------------------------------------------------
#

# Forward Predictability: Train on Rounds 1 & 2 -> Test on All Rounds 1-9
print("\n" + "="*80)
print("FORWARD PREDICTABILITY EXPERIMENT")
print("="*80)
print("Training on historical data (Rounds 1-2, ~2002-2004)")
print("Testing on all 9 rounds to assess temporal generalization")
print("="*80 + "\n")

forward_perf = run_temporal_experiment(train_rounds=[1, 2], exp_name="FORWARD PREDICTABILITY")
forward_perf.to_csv("ess_temporal_forward_eval.csv", index=False)
print(" Saved forward evaluation to 'ess_temporal_forward_eval.csv'")


#%%
# Backward Predictability: Train on Rounds 8 & 9 -> Test on All Rounds 1-9
print("\n" + "="*80)
print("BACKWARD PREDICTABILITY EXPERIMENT")
print("="*80)
print("Training on modern data (Rounds 8-9, ~2016-2018)")
print("Testing on all 9 rounds to assess backcasting ability")
print("="*80 + "\n")

backward_perf = run_temporal_experiment(train_rounds=[8, 9], exp_name="BACKWARD PREDICTABILITY")
backward_perf.to_csv("ess_temporal_backward_eval.csv", index=False)
print(" Saved backward evaluation to 'ess_temporal_backward_eval.csv'")

# Note: The performance comparison between forward and backward models reveals:
# - Symmetric performance (forward ≈ backward): Stable gender patterns over time
# - Forward > Backward: Historical patterns persist; modern patterns don't backcast well
#   (suggests convergence toward modern norms)
# - Backward > Forward: Modern patterns are more predictive of past than vice versa
#   (suggests recent norm shifts that affect historical interpretation)

print("\n Execution complete!")

# Output files:
# - ess_temporal_forward_eval.csv: Performance metrics for forward prediction model
# - ess_temporal_backward_eval.csv: Performance metrics for backward prediction model
#
# Each CSV contains columns: Country, ESS_round, test_n, accuracy, f1_score
# Lower accuracy = higher gender similarity (core hypothesis of the research)