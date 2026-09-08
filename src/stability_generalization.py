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
#%%
import time
import pandas as pd
import numpy as np
import pyreadstat

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import plotly.express as px
from pyprojroot import here
import group_valid


#%%

def subsample_xy(X, y, n_max=10000, random_state=42):
    """
    Subsamples up to n_max rows from X (and the matching rows of y) for cheaper
    permutation-importance evaluation. Used both for feature selection and final export.
    """
    X_sub = X.sample(n=min(n_max, len(X)), random_state=random_state)
    y_sub = y.loc[X_sub.index]
    return X_sub, y_sub


def print_gender_distribution(series, label_for_code, code_word="code"):
    """
    Prints per-value row counts for a gender/target series, mapping each numeric code
    to a human-readable label via label_for_code(code).
    """
    counts = series.value_counts().sort_index()
    for code, count in counts.items():
        code_int = int(code)
        print(f"{label_for_code(code_int)} ({code_word} {code_int}): {count} rows")

def _missing_code_variants(digit):
    variants = set()
    for reps in range(1, 5):
        n = int(str(digit) * reps)
        variants |= {n, float(n), str(n)}
    return variants

def is_strictly_valid_str(s):
    return s.notna() & (~s.astype(str).str.strip().isin(string_missing_codes))

def eval_group(g):
    # Calculate accuracy: proportion of correct predictions
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    
    # Calculate F1 score (harmonic mean of precision and recall)
    # Using 'binary' average for binary classification with zero_division=0 to handle edge cases
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    
    # Return summary statistics as a Series
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})


def run_temporal_experiment(data, train_rounds, exp_name, target_col='target_encoded', round_col='ESS_round', country_col='Country'):
    """
    Trains a prediction model on specified rounds from a dataset and evaluates performance across remaining data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input dataset containing features, target, country, and round columns.
    train_rounds : list of int
        Round numbers to use for training.
    exp_name : str
        Descriptive name for the experiment.
    target_col : str, optional (default='target_encoded')
        Name of target column.
    round_col : str, optional (default='ESS_round')
        Name of round column.
    country_col : str, optional (default='Country')
        Name of country column.
        
    Returns:
    --------
    perf_df : pd.DataFrame
        Performance metrics grouped by country and round.
    final_hgb : HistGradientBoostingClassifier
        Trained model with selected features.
    X_train_selected : pd.DataFrame
        Feature matrix for training set (selected features only).
    y_train : pd.Series
        True labels from training set.
    X_test_selected : pd.DataFrame
        Feature matrix for test set (selected features only).
    y_test : pd.Series
        True labels from test set.
    selected_features : list of str
        List of feature names used in final model.
    """
    print(f"\n================ Running Experiment: {exp_name} ================")

    # Identify feature columns (excluding metadata/targets)
    # Updated Feature Exclusion in run_temporal_experiment
    non_feature_cols = [
        target_col, 'Gender', 'gndr', 'gender', 'target', 'target_encoded',
        round_col, country_col, 'cntry', 'round', 'ESS_round',
        'y_pred', 'y_prob', 'index'
    ]

    # Extract feature columns safely
    feature_cols = [c for c in data.columns if c not in non_feature_cols and not c.startswith('target')]

    # 1. Filter dataset to include only responses from training rounds
    train_pool = data[data[round_col].isin(train_rounds)].copy()
    
    # 2. Compute train_n threshold based on 80% of smallest country/round group in training pool
    counts = train_pool.groupby([country_col, round_col]).size()
    train_n = int(counts.min() * 0.8)
    print(f"Calculated train_n (80% of min group size in rounds {train_rounds}): {train_n}")

# 3. Sample balanced training data from each country/round combination
    training_data = train_pool.groupby([country_col, round_col], group_keys=False).apply(
        lambda x: x.sample(n=min(train_n, len(x)), random_state=42)
    )
    
    # Test set contains all remaining data from the dataset
    test_data = data.drop(index=training_data.index).copy()

    # Extract feature matrix X and target vector y for training
    X_train = training_data[feature_cols]
    y_train = training_data[target_col]
    
    # Prepare test set features
    X_test_full = test_data[feature_cols]
    y_test = test_data[target_col]

    # 4. Feature Selection via Baseline Model Permutation Importance
    print("Performing feature selection on training split...")
    
    baseline_hgb = HistGradientBoostingClassifier(random_state=42)
    baseline_hgb.fit(X_train, y_train)

    # Subsample up to 10,000 rows for permutation importance calculation
    X_select_sample = X_train.sample(n=min(10000, len(X_train)), random_state=42)
    y_select_sample = y_train.loc[X_select_sample.index]

    perm_selection = permutation_importance(
        baseline_hgb, X_select_sample, y_select_sample, n_repeats=2, random_state=42, n_jobs=-1
    )

    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': perm_selection.importances_mean})
    print(imp_df)
    # Select features with positive importance
    selected_features = imp_df[imp_df['Importance'] > 0]['Feature'].tolist()
    print(f"Retained {len(selected_features)} features with positive importance out of {len(feature_cols)}.")

    # 5. Retrain model on selected feature subset
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test_full[selected_features]

    final_hgb = HistGradientBoostingClassifier(random_state=42)
    final_hgb.fit(X_train_selected, y_train)

    # 6. Evaluate accuracy across testing rounds
    test_data['y_pred'] = final_hgb.predict(X_test_selected)
    test_data['y_prob'] = final_hgb.predict_proba(X_test_selected)[:, 1]

    # Compute performance metrics per country and round
    perf_df = test_data.groupby([country_col, round_col]).apply(eval_group, include_groups=False).reset_index()
    
    # Mixed-Effects Model
    mixed_res = smf.mixedlm(f"accuracy ~ {round_col}", perf_df, groups=perf_df[country_col]).fit()
    print("\nMixed-Effects Regression Summary:")
    print(mixed_res.summary())

    return perf_df, final_hgb, X_train_selected, y_train, X_test_selected, y_test, selected_features

def compute_permutation_importance_df(model, X, y, n_repeats=5, random_state=42):
    """
    Runs permutation importance and returns an unsorted Feature/Importance DataFrame.
    Feature names are taken from X.columns, so the caller never needs to pass them separately.
    """
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    return pd.DataFrame({'Feature': X.columns.tolist(), 'Importance': perm.importances_mean})


def export_separate_feature_importance(model, X_eval, y_eval, output_csv=None, output_html=None):
    """
    Computes permutation importance on evaluation data, exports sorted CSV of all features, 
    and saves an HTML bar plot displaying only the top 20 features sorted strictly by importance.
    """
    if output_csv is None:
        output_csv = here('data/processed/features_importances_sorted.csv')
    if output_html is None:
        output_html = here('feature_importance_plot.html')
    
    print("\nEvaluating and exporting feature importance...")  # 
    imp_df = compute_permutation_importance_df(model, X_eval, y_eval)

    imp_df['Category'] = imp_df['Feature'].apply(
        lambda name: 'Not Applicable Flag' if name.endswith('_is_na') else ('Other Missing Flag' if name.endswith('_is_missing') else 'Base Survey Question')
    )

    # Sort full dataset descending for CSV output
    imp_df_sorted = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Slice top 20 features and sort ascending for horizontal Plotly bar rendering (highest at top)
    top20_df = imp_df_sorted.head(20).sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        top20_df,
        x='Importance',
        y='Feature',
        color='Category',
        orientation='h',
        title="<b>Top 20 Retained Features Importance (Indicator Flags & Native Tree Splits)</b>",
        template='plotly_white',
        color_discrete_map={
            'Base Survey Question': '#1f77b4',
            'Not Applicable Flag': '#ff7f0e',
            'Other Missing Flag': '#d62728'
        }
    )

    # Force strict ordering on the y-axis by feature importance values regardless of category grouping
    fig_imp.update_yaxes(
        type='category', 
        categoryorder='array',
        categoryarray=top20_df['Feature'].tolist(),
        tickmode='linear', 
        dtick=1, 
        automargin=True, 
        showgrid=True, 
        gridcolor='#E5E5E5'
    )
    
    fig_imp.update_xaxes(showgrid=True, gridcolor='#E5E5E5')
    fig_imp.update_layout(
        height=700, 
        width=1150, 
        margin=dict(l=350, r=50, t=100, b=50), 
        legend=dict(title="Feature Category", y=0.1, x=0.7),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    fig_imp.write_html(here(output_html), include_plotlyjs='cdn')

    # Export all features to CSV
    imp_df_sorted[['Feature', 'Importance', 'Category']].to_csv(here(output_csv), index=False)
    print(f"✅ Full feature importances exported to '{output_csv}' and Top 20 plot saved to '{output_html}'.")
    
    return imp_df_sorted

def create_subset_summary(df, base_features, raw_df):
    """
    Creates a summary table even if grouping columns were stripped by include_groups=False.
    """
    df_temp = df.copy()

    # Reattach missing Country and ESS_round from raw dataset index if absent
    if 'Country' not in df_temp.columns or 'ESS_round' not in df_temp.columns:
        df_temp['Country'] = raw_df.loc[df_temp.index, 'Country']
        df_temp['ESS_round'] = raw_df.loc[df_temp.index, 'ESS_round']

    # Identify indicator columns
    na_cols = [f"{c}_is_na" for c in base_features if f"{c}_is_na" in df_temp.columns]
    missing_cols = [f"{c}_is_missing" for c in base_features if f"{c}_is_missing" in df_temp.columns]

    df_temp['sum_na_row'] = df_temp[na_cols].sum(axis=1) if na_cols else 0
    df_temp['sum_missing_row'] = df_temp[missing_cols].sum(axis=1) if missing_cols else 0

    grouped = df_temp.groupby(['Country', 'ESS_round'], as_index=False)

    summary_df = grouped.agg(
        total_number_rows=('sum_na_row', 'count'),
        total_na_per_group=('sum_na_row', 'sum'),
        total_missing_per_group=('sum_missing_row', 'sum')
    )

    n_features = len(base_features)
    total_cells_per_group = summary_df['total_number_rows'] * n_features

    summary_df['pct_cells_Non_applicable'] = (summary_df['total_na_per_group'] / total_cells_per_group) * 100
    summary_df['pct_cells_Missing'] = (summary_df['total_missing_per_group'] / total_cells_per_group) * 100

    return summary_df[['Country', 'ESS_round', 'total_number_rows', 'pct_cells_Non_applicable', 'pct_cells_Missing']].sort_values(['Country', 'ESS_round']).reset_index(drop=True)




# --------------------------------------------------------------------------------------------------
#%%
# ==================================================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA
# ==================================================================================================
print("Step 1: Loading raw SPSS file and metadata...")  
# Step 1: Loading raw SPSS file and metadata...
# This step loads the raw ESS (European Social Survey) data file and extracts metadata.
# The Sav file contains survey responses from multiple rounds across European countries,
# including gender information and various attitudinal questions that serve as features.
#

# Define the path to the combined ESS dataset subset in SPSS (.sav) format
# This single file contains data from ESS Rounds 1-9, each with different survey designs
# user_missing=True ensures missing value codes are properly recognized
import time
t0 = time.time()
path = here('data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(str(path), user_missing=True)
print(time.time() - t0)
print(f"Number of unique countries: {df_raw['cntry'].nunique()}")  # Number of unique countries: 38





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
print(f"Gender distribution before filtering:")  # Gender distribution before filtering:
print_gender_distribution(
    df_sub[gender_raw_col],
    lambda c: "Male" if c == 1 else ("Female" if c == 2 else "Unknown"),
)
'''
Gender distribution before filtering:
Male (code 1): 198789 rows
Female (code 2): 231749 rows
'''

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

NOT_APPLICABLE_CODES = _missing_code_variants(6)
ALL_MISSING_CODES = set().union(*(_missing_code_variants(d) for d in (6, 7, 8, 9)))

# Grouping columns used for analysis: country and survey round
group_cols = ['cntry', 'essround']

# Candidate columns are all columns except grouping columns and metadata fields
candidate_cols = [c for c in df_sub.columns if c not in group_cols]

# Remove known metadata columns that should not be included as features
metadata_leak = ['name', 'edition', 'proddate']
candidate_cols = [c for c in candidate_cols if c not in metadata_leak]

# Determine column types: numeric vs string (non-numeric)
dtypes = df_sub[candidate_cols].dtypes
string_cols = dtypes[~dtypes.apply(pd.api.types.is_numeric_dtype)].index.tolist()
numeric_cols = [c for c in candidate_cols if c not in string_cols]

print(f"{len(numeric_cols)} numeric candidate cols, {len(string_cols)} string candidate cols")  # 2340 numeric candidate cols, 23 string candidate cols
#2340 numeric candidate cols, 23 string candidate cols



#%%
# ---- Validity check: numeric columns via C++ ----
t0 = time.time()

# Create a MultiIndex from country and round to uniquely identify each group (country-round pair)
group_key = pd.MultiIndex.from_arrays([df_sub[group_cols[0]], df_sub[group_cols[1]]])
# Factorize the MultiIndex into integer IDs for efficient grouping; group_labels maps IDs back to original pairs
group_ids, group_labels = pd.factorize(group_key)

# Convert numeric candidate columns to a NumPy array of float64 (required by C++ extension)
values = df_sub[numeric_cols].to_numpy(dtype=np.float64)

# Build sorted list of missing codes as floats (excluding string versions) for C++ validation
missing_arr = np.array(
    sorted({float(c) for c in ALL_MISSING_CODES if not isinstance(c, str)}),
    dtype=np.float64
)

# Run the C++-accelerated validity check: for each group (country-round), determine whether *any* valid value exists per column
result = group_valid.group_any_valid(
    values, group_ids.astype(np.int64), missing_arr, len(group_labels)
)
# Convert result to DataFrame with rows = groups (country-round pairs) and columns = numeric candidate variables
valid_per_group_numeric = pd.DataFrame(result, columns=numeric_cols, index=group_labels)

# ---- Validity check: string columns via pandas ----
# Combine all missing codes as strings (including empty/whitespace strings)
string_missing_codes = {str(c) for c in ALL_MISSING_CODES} | {'', ' '}

# For each string column, create a boolean mask indicating valid (non-missing) entries per row
valid_mask_str = df_sub[string_cols].apply(is_strictly_valid_str)
# Group by country-round and check if *any* valid value exists per group/column
valid_per_group_str = valid_mask_str.groupby(
    [df_sub[group_cols[0]], df_sub[group_cols[1]]]
).any()
# Reindex to match the order of group_labels (ensuring alignment with numeric result)
valid_per_group_str = valid_per_group_str.reindex(group_labels)

print(f"Validation completed in {time.time() - t0:.2f} seconds")  # Validation completed in 4.10 seconds

# ---- Combine and determine retained columns ----
# Concatenate validity DataFrames side-by-side (numeric + string columns)
valid_per_group = pd.concat([valid_per_group_numeric, valid_per_group_str], axis=1)
# A column is retained only if it has *at least one valid value* in *every* country-round group
retained_cols = valid_per_group.columns[valid_per_group.all()].tolist()


#%% 
# # ==================================================================================================
# STEP 3: BUILD SUBSET, RENAME & STANDARDIZE METADATA HEADERS
# ==================================================================================================
print("Step 3: Renaming headers and dropping excluded metadata...")  # Step 3: Renaming headers and dropping excluded metadata...

# ---- Build df_subset ----
# Create a copy of the subset containing only grouping columns (Country, ESS_round) and retained feature columns
df_subset = df_sub[group_cols + retained_cols].copy()

# Rename all column names using the mapping from variable codes to human-readable labels (e.g., 'gndr' → 'Gender')
df_subset.rename(columns=code_to_label, inplace=True)

# Extract renamed versions of grouping columns for clarity and consistency
cntry_renamed = code_to_label.get(group_cols[0], group_cols[0])  # e.g., 'cntry' → 'Country'
round_renamed = code_to_label.get(group_cols[1], group_cols[1])   # e.g., 'essround' → 'ESS_round'
gender_renamed = code_to_label.get(gender_raw_col, gender_raw_col) # e.g., 'gndr' → 'Gender'

# Create standardized columns:
# - 'Country': map country codes (e.g., 1) to full names (e.g., 'Austria') using metadata value labels
# - 'ESS_round': keep numeric round identifier as-is
# - 'Gender': use the renamed gender column directly
df_subset['Country'] = df_subset[cntry_renamed].astype(str).str.strip().map(cntry_val_labels).fillna(df_subset[cntry_renamed])
df_subset['ESS_round'] = df_subset[round_renamed]
df_subset['Gender'] = df_subset[gender_renamed]



#%% ---- Drop irrelevant columns (after extracting Country/ESS_round/Gender) ----
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

existing_drops = [c for c in cols_to_drop if c in df_subset.columns]
missing_from_drop_list = [c for c in cols_to_drop if c not in df_subset.columns]
if missing_from_drop_list:
    print(f"Note: {len(missing_from_drop_list)} cols_to_drop entries not found in df_subset "
          f"(already excluded by retained_cols filter): {missing_from_drop_list}")
'''
Note: 14 cols_to_drop entries not found in df_subset (already excluded by retained_cols filter): ['cntry', 'essround', 'gndr', 'Title_of_dataset', 'Edition', 'Production_date', 'Country_of_birth', 'Country_of_birth,_father', 'Language_most_often_spoken_at_home:_first_mentioned', 'Country_of_birth,_mother', 'nan_count', 'Citizenship', 'Language_most_often_spoken_at_home:_second_mentioned', 'Region']
'''
df_subset.drop(columns=existing_drops, inplace=True)


#%%
print('\n'.join(df_subset.columns))
print("-------------------------------------------\n Number of Columns:", len(df_subset.columns))
"""
Country
ESS_round
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves
Most_people_can_be_trusted_or_you_can't_be_too_careful
Worn_or_displayed_campaign_badge/sticker_last_12_months
Boycotted_certain_products_last_12_months
Feel_closer_to_a_particular_party_than_all_other_parties
Contacted_politician_or_government_official_last_12_months
Gays_and_lesbians_free_to_live_life_as_they_wish
Government_should_reduce_differences_in_income_levels
Placement_on_left_right_scale
Taken_part_in_lawful_public_demonstration_last_12_months
How_interested_in_politics
How_close_to_party
Signed_petition_last_12_months
How_satisfied_with_the_way_democracy_works_in_country
How_satisfied_with_present_state_of_economy_in_country
State_of_education_in_country_nowadays
State_of_health_services_in_country_nowadays
How_satisfied_with_life_as_a_whole
Trust_in_the_European_Parliament
Trust_in_the_legal_system
Trust_in_the_police
Trust_in_politicians
Trust_in_country's_parliament
Trust_in_the_United_Nations
Voted_last_national_election
Worked_in_political_party_or_action_group_last_12_months
Allow_many/few_immigrants_from_poorer_countries_outside_Europe
Immigration_bad_or_good_for_country's_economy
Country's_cultural_life_undermined_or_enriched_by_immigrants
Immigrants_make_country_worse_or_better_place_to_live
Feeling_of_safety_of_walking_alone_in_local_area_after_dark
Belong_to_minority_ethnic_group_in_country
Born_in_country
Respondent_or_household_member_victim_of_burglary/assault_last_5_years
Citizen_of_country
Discrimination_of_respondent's_group:_age
Discrimination_of_respondent's_group:_don't_know
Discrimination_of_respondent's_group:_disability
Discrimination_of_respondent's_group:_ethnic_group
Member_of_a_group_discriminated_against_in_this_country
Discrimination_of_respondent's_group:_language
Discrimination_of_respondent's_group:_no_answer
Discrimination_of_respondent's_group:_not_applicable
Discrimination_of_respondent's_group:_nationality
Discrimination_of_respondent's_group:_colour_or_race
Discrimination_of_respondent's_group:_refusal
Discrimination_of_respondent's_group:_religion
Discrimination_of_respondent's_group:_sexuality
Father_born_in_country
How_happy_are_you
Subjective_general_health
Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem
Mother_born_in_country
How_often_pray_apart_from_at_religious_services
How_often_attend_religious_services_apart_from_special_occasions
How_religious_are_you
Take_part_in_social_activities_compared_to_others_of_same_age
How_often_socially_meet_with_friends,_relatives_or_colleagues
Number_of_people_living_regularly_as_member_of_household
Gender
Year_of_birth
Year_of_birth_of_fourth_person_in_household
Year_of_birth_of_fifth_person_in_household
Year_of_birth_of_sixth_person_in_household
Age_of_respondent,_calculated
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
Ever_had_children_living_in_household
Doing_last_7_days:_community_or_military_service
Partner_doing_last_7_days:_community_or_military_service
Doing_last_7_days:_don't_know
Partner_doing_last_7_days:_don't_know
Doing_last_7_days:_no_answer
Partner_doing_last_7_days:_no_answer
Partner_doing_last_7_days:_not_applicable
Doing_last_7_days:_other
Partner_doing_last_7_days:_other
Doing_last_7_days:_refusal
Partner_doing_last_7_days:_refusal
Domicile,_respondent's_description
Doing_last_7_days:_permanently_sick_or_disabled
Partner_doing_last_7_days:_permanently_sick_or_disabled
Doing_last_7_days:_education
Partner_doing_last_7_days:_education
Years_of_full-time_education_completed
Highest_level_of_education,_ES_-_ISCED
Number_of_employees_respondent_has/had
Father's_employment_status_when_respondent_14
Mother's_employment_status_when_respondent_14
Establishment_size
Doing_last_7_days:_housework,_looking_after_children,_others
Partner_doing_last_7_days:_housework,_looking_after_children,_others
Responsible_for_supervising_other_employees
Main_activity_last_7_days
Main_activity,_last_7_days._All_respondents._Post_coded
Partner's_main_activity_last_7_days
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job
Year_last_in_paid_job
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work
Doing_last_7_days:_retired
Partner_doing_last_7_days:_retired
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months
Any_period_of_unemployment_and_work_seeking_within_last_5_years
Doing_last_7_days:_unemployed,_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job
Doing_last_7_days:_unemployed,_not_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job
Hours_normally_worked_a_week_in_main_job_overtime_included,_partner
Number of Columns: 113
"""


#%%
# --------------------------------------------------------------------------------------------------
# STEP 4: ICELAND EXCLUSION & HELPER FUNCTIONS
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

# ==================================================================================================
# STEP 4: HANDLE MISSING VALUES (DUAL INDICATORS & ZERO-FILLING)
# ==================================================================================================
print("Step 4: Constructing _is_na and _is_missing indicator flags...")  # Step 4: Constructing _is_na and _is_missing indicator flags...

# NOT_APPLICABLE_CODES and ALL_MISSING_CODES were generated in Step 2; derive the remainder here.
# OTHER_MISSING_CODES = all other missing codes (refused, don't know, etc.), NaN handled separately below.
OTHER_MISSING_CODES = ALL_MISSING_CODES - NOT_APPLICABLE_CODES

# List of base feature columns (excluding grouping and target variables)
base_features = [c for c in df_subset.columns if c not in ['Country', 'ESS_round', 'Gender']]

# Initialize transformed dictionary with grouping and target columns
transformed = {
    'Country': df_subset['Country'], 
    'ESS_round': df_subset['ESS_round'], 
    'Gender': df_subset['Gender']
}

# Counters for tracking total missingness across dataset
total_na = 0
total_missing = 0

# Process each base feature column:
for c in base_features:
    s = df_subset[c]  # Extract the raw series

    # Identify "Not Applicable" responses (e.g., skipped due to skip logic)
    is_na = s.isin(NOT_APPLICABLE_CODES)

    # Identify other missing values: either explicit missing codes or actual NaNs
    is_miss = s.isin(OTHER_MISSING_CODES) | s.isna()

    # Accumulate counts for reporting
    total_na += int(is_na.sum())
    total_missing += int(is_miss.sum())

    # Create binary indicator flags:
    # - _is_na: 1 if value was "Not Applicable", else 0
    # - _is_missing: 1 if value was missing (refused/don't know/etc.) or NaN, else 0
    transformed[f"{c}_is_na"] = is_na.astype(int)
    transformed[f"{c}_is_missing"] = is_miss.astype(int)
    
    # Convert column to numeric (coerce non-numeric entries to NaN)
    num_s = pd.to_numeric(s, errors='coerce')

    # Replace all missing values (both NA and other missing) with 0.0
    num_s[is_na | is_miss] = 0.0

    # Fill any remaining NaNs (e.g., from non-numeric strings that couldn’t be coerced)
    transformed[c] = num_s.fillna(0.0)

# Report totals for transparency and debugging
print(f"Total 'Not Applicable' (NA) values across all features: {total_na}")  
# Total 'Not Applicable' (NA) values across all features: 5094341
print(f"Total other missing values (refused/don't know/etc.) + NaNs: {total_missing}") 
 # Total other missing values (refused/don't know/etc.) + NaNs: 3689290
'''
Step 4: Constructing _is_na and _is_missing indicator flags...
Total 'Not Applicable' (NA) values across all features: 5094341
Total other missing values (refused/don't know/etc.) + NaNs: 3689290
'''
# Convert transformed dictionary to DataFrame
df_processed = pd.DataFrame(transformed)

# Encode gender labels into numeric targets for classification (e.g., 'Male' → 0, 'Female' → 1)
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
# ==================================================================================================
# STEP 5: GENERATE MISSINGNESS SUMMARY TABLE PER COUNTRY/ROUND
# ==================================================================================================
print("Step 5: Generating missingness summary table per country and round...")  # Step 5: Generating missingness summary table per country and round...

# base_features (raw feature names, excluding grouping/target columns) was already computed in Step 4 —
# the indicator-flag and target columns added since then don't change that underlying set, so it's reused as-is.

# Use the helper function to generate missingness summary
summary_df = create_subset_summary(df_filtered, feature_cols, df_raw)
print("✅ Missingness summary table generated successfully!")  # ✅ Missingness summary table generated successfully!
print(summary_df.head(10))  #

'''
Step 5: Generating missingness summary table per country and round...
✅ Missingness summary table generated successfully!
   Country  ESS_round  total_number_rows  pct_cells_Non_applicable  \
0  Albania        6.0               1201                  9.133298   
1  Austria        1.0               2257                 10.930036   
2  Austria        2.0               2256                 10.736622   
3  Austria        3.0               2405                 10.874693   
4  Austria        7.0               1795                 11.099519   
5  Austria        8.0               2010                 11.436454   
6  Austria        9.0               2499                 11.068064   
7  Belgium        1.0               1870                 11.018474   
8  Belgium        2.0               1778                 11.178546   
9  Belgium        3.0               1798                 11.261503   

   pct_cells_Missing  
0           6.263720  
1           8.106497  
2           7.836074  
3           8.313362  
4           7.081286  
5           7.578019  
6           8.680927  
7           8.951386  
8           8.186420  
9           8.554454
'''

#%%
# Re-run gender distribution check at this point in script (after preprocessing steps 2–5)
print("\n" + "="*84)
print("Gender Distribution After Preprocessing Steps (Steps 2–5)")
print("="*84)

# Use df_filtered, which has already been cleaned and encoded.
# Maps back to human-readable labels using the LabelEncoder's classes_.
print_gender_distribution(
    df_filtered['target_encoded'],
    lambda c: le.inverse_transform([c])[0],
    code_word="encoded",
)
print("\n✅ Gender distribution check completed.")  # 
'''
====================================================================================
Gender Distribution After Preprocessing Steps (Steps 2–5)
====================================================================================
1.0 (encoded 0): 197286 rows
2.0 (encoded 1): 230195 rows

✅ Gender distribution check completed.
'''


#%%
# Export the summary to CSV
output_summary_csv = here("data/processed/stability_full_row_summary.csv")
summary_df.to_csv(output_summary_csv, index=False)
print(f"✅ Missingness summary table saved to '{output_summary_csv}'")  
print(f"   Summary shape: {summary_df.shape[0]} rows (country-round pairs), {summary_df.shape[1]} columns")  #    Summary shape: 228 rows (country-round pairs), 5 columns
'''
✅ Missingness summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/stability_full_row_summary.csv'
   Summary shape: 224 rows (country-round pairs), 5 columns
   '''


#%%
# Export the processed dataset to CSV
output_csv = here("data/processed/stability_processed_two_missingness_indicators.csv")
df_filtered.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_filtered.shape} to '{output_csv}'")  # ✅ Processed dataset saved with shape (430538, 334) to '/data/home/asher.katz/Projects/gender_differences/data/processed/gender_processed_two_missingness_indicators.csv'





#%%
# --------------------------------------------------------------------------------------------------
# STEP 6: EXECUTE FORWARD AND BACKWARD EXPERIMENTS
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


# Forward Predictability: Train on Rounds 1 & 2 -> Test on All Rounds 1-9
print("\n" + "="*80)
print("FORWARD PREDICTABILITY EXPERIMENT")
print("="*80)

forward_perf, hgb_model_fwd, X_train_selected_fwd, y_train_selected_fwd, X_test_selected_fwd, y_test_selected_fwd, fwd_features = run_temporal_experiment(
    data = df_filtered, 
    train_rounds=[1, 2], 
    exp_name="FORWARD PREDICTABILITY"
)

forward_perf.to_csv(here("data/processed/forward_stability_country_and_round_performance_indicators.csv"), index=False)
print(" Saved forward evaluation to data/processed/forward_stability_country_and_round_performance_indicators.csv")

# 1. Reconstruct exact training and testing subsets fed into the final HGB model
training_data = df_filtered.loc[X_train_selected_fwd.index].copy()
test_data = df_filtered.loc[X_test_selected_fwd.index].copy()

# 2. Summarize [Country, ESS_round, total_number_rows, pct_cells_Non_applicable, pct_cells_Missing] for actual training data
train_summary = create_subset_summary(training_data, fwd_features, df_filtered)

# 3. Summarize [Country, ESS_round, total_number_rows, pct_cells_Non_applicable, pct_cells_Missing] for actual testing data
test_summary = create_subset_summary(test_data, fwd_features, df_filtered)

# 4. Save summary tables to disk
output_train_csv = here("data/processed/forwards_stability_train_row_summary.csv")
output_test_csv = here("data/processed/forwards_stability_test_row_summary.csv")

train_summary.to_csv(output_train_csv, index=False)
print(f"✅ Train summary table saved to '{output_train_csv}'")
print(f"   Shape: {train_summary.shape[0]} rows (country-round pairs), {train_summary.shape[1]} columns")

test_summary.to_csv(output_test_csv, index=False)
print(f"✅ Test summary table saved to '{output_test_csv}'")
print(f"   Shape: {test_summary.shape[0]} rows (country-round pairs), {test_summary.shape[1]} columns")

'''

================================================================================
FORWARD PREDICTABILITY EXPERIMENT
================================================================================

================ Running Experiment: FORWARD PREDICTABILITY ================
Calculated train_n (80% of min group size in rounds [1, 2]): 965
Performing feature selection on training split...
                                               Feature  Importance
0    Most_people_try_to_take_advantage_of_you,_or_t...    -0.00035
1    Most_people_try_to_take_advantage_of_you,_or_t...     0.00605
2    Most_people_try_to_take_advantage_of_you,_or_t...     0.00255
3    Most_of_the_time_people_helpful_or_mostly_look...     0.00025
4    Most_of_the_time_people_helpful_or_mostly_look...     0.00085
..                                                 ...         ...
325  Partner_doing_last_7_days:_unemployed,_not_act...     0.00000
326  Partner_doing_last_7_days:_unemployed,_not_act...     0.00015
327  Hours_normally_worked_a_week_in_main_job_overt...    -0.00005
328  Hours_normally_worked_a_week_in_main_job_overt...     0.00135
329  Hours_normally_worked_a_week_in_main_job_overt...     0.03305

[330 rows x 2 columns]
Retained 132 features with positive importance out of 330.
/data/home/asher.katz/miniconda3/envs/gendEnv/lib/python3.11/site-packages/statsmodels/regression/mixed_linear_model.py:2237: ConvergenceWarning: The MLE may be on the boundary of the parameter space.
  warnings.warn(msg, ConvergenceWarning)

Mixed-Effects Regression Summary:
        Mixed Linear Model Regression Results
======================================================
Model:            MixedLM Dependent Variable: accuracy
No. Observations: 224     Method:             REML    
No. Groups:       37      Scale:              0.0004  
Min. group size:  1       Log-Likelihood:     506.1491
Max. group size:  9       Converged:          Yes     
Mean group size:  6.1                                 
------------------------------------------------------
           Coef.  Std.Err.    z    P>|z| [0.025 0.975]
------------------------------------------------------
Intercept   0.790    0.006 135.105 0.000  0.779  0.802
ESS_round  -0.007    0.001 -12.252 0.000 -0.008 -0.006
Group Var   0.001    0.014                            
======================================================

 Saved forward evaluation to data/processed/forward_stability_country_and_round_performance_indicators.csv
✅ Train summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/forwards_stability_train_row_summary.csv'
   Shape: 46 rows (country-round pairs), 5 columns
✅ Test summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/forwards_stability_test_row_summary.csv'
   Shape: 224 rows (country-round pairs), 5 columns
'''



#%%
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
# Backward Predictability: Train on Rounds 8 & 9 -> Test on All Rounds 1-9
print("\n" + "="*80)
print("BACKWARD PREDICTABILITY EXPERIMENT")
print("="*80)

backward_perf, hgb_model_bwd, X_train_selected_bwd, y_train_selected_bwd, X_test_selected_bwd, y_test_selected_bwd, bwd_features = run_temporal_experiment(
    data=df_filtered,
    train_rounds=[8, 9], 
    exp_name="BACKWARD PREDICTABILITY"
)

backward_perf.to_csv(here("data/processed/backward_stability_country_and_round_performance_indicators.csv"), index=False)
print(" Saved backward evaluation to data/processed/backward_stability_country_and_round_performance_indicators.csv")

# 1. Reconstruct exact training and testing subsets fed into the final HGB model
training_data = df_filtered.loc[X_train_selected_bwd.index].copy()
test_data = df_filtered.loc[X_test_selected_bwd.index].copy()

# 2. Summarize [Country, ESS_round, total_number_rows, pct_cells_Non_applicable, pct_cells_Missing] for actual training data
train_summary = create_subset_summary(training_data, bwd_features, df_filtered)

# 3. Summarize [Country, ESS_round, total_number_rows, pct_cells_Non_applicable, pct_cells_Missing] for actual testing data
test_summary = create_subset_summary(test_data, bwd_features, df_filtered)

# 4. Save summary tables to disk
output_train_csv = here("data/processed/backwards_stability_train_row_summary.csv")
output_test_csv = here("data/processed/backwards_stability_test_row_summary.csv")


train_summary.to_csv(output_train_csv, index=False)
print(f"✅ Train summary table saved to '{output_train_csv}'")
print(f"   Shape: {train_summary.shape[0]} rows (country-round pairs), {train_summary.shape[1]} columns")

test_summary.to_csv(output_test_csv, index=False)
print(f"✅ Test summary table saved to '{output_test_csv}'")
print(f"   Shape: {test_summary.shape[0]} rows (country-round pairs), {test_summary.shape[1]} columns")

'''
================================================================================
BACKWARD PREDICTABILITY EXPERIMENT
================================================================================

================ Running Experiment: BACKWARD PREDICTABILITY ================
Calculated train_n (80% of min group size in rounds [8, 9]): 624
Performing feature selection on training split...
                                               Feature  Importance
0    Most_people_try_to_take_advantage_of_you,_or_t...     0.00015
1    Most_people_try_to_take_advantage_of_you,_or_t...     0.00080
2    Most_people_try_to_take_advantage_of_you,_or_t...     0.00035
3    Most_of_the_time_people_helpful_or_mostly_look...     0.00005
4    Most_of_the_time_people_helpful_or_mostly_look...     0.00145
..                                                 ...         ...
325  Partner_doing_last_7_days:_unemployed,_not_act...     0.00000
326  Partner_doing_last_7_days:_unemployed,_not_act...     0.00010
327  Hours_normally_worked_a_week_in_main_job_overt...     0.00060
328  Hours_normally_worked_a_week_in_main_job_overt...     0.00190
329  Hours_normally_worked_a_week_in_main_job_overt...     0.02445

[330 rows x 2 columns]
Retained 142 features with positive importance out of 330.
/data/home/asher.katz/miniconda3/envs/gendEnv/lib/python3.11/site-packages/statsmodels/regression/mixed_linear_model.py:2237: ConvergenceWarning: The MLE may be on the boundary of the parameter space.
  warnings.warn(msg, ConvergenceWarning)

Mixed-Effects Regression Summary:
        Mixed Linear Model Regression Results
======================================================
Model:            MixedLM Dependent Variable: accuracy
No. Observations: 224     Method:             REML    
No. Groups:       37      Scale:              0.0004  
Min. group size:  1       Log-Likelihood:     516.5523
Max. group size:  9       Converged:          Yes     
Mean group size:  6.1                                 
------------------------------------------------------
           Coef.  Std.Err.    z    P>|z| [0.025 0.975]
------------------------------------------------------
Intercept   0.768    0.005 145.520 0.000  0.758  0.779
ESS_round  -0.003    0.001  -5.833 0.000 -0.004 -0.002
Group Var   0.001    0.011                            
======================================================

 Saved backward evaluation to data/processed/backward_stability_country_and_round_performance_indicators.csv
✅ Train summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/backwards_stability_train_row_summary.csv'
   Shape: 50 rows (country-round pairs), 5 columns
✅ Test summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/backwards_stability_test_row_summary.csv'
   Shape: 224 rows (country-round pairs), 5 columns
   '''





#%%
eval_X_test_selected_fwd, eval_y_test_selected_fwd = subsample_xy(X_test_selected_fwd, y_test_selected_fwd)
fwd_imp = export_separate_feature_importance(hgb_model_fwd, eval_X_test_selected_fwd, eval_y_test_selected_fwd, output_csv=here("data/processed/forward_stability_feature_importance.csv"), output_html=here("plots/forward_stability_feature_importance_top20.html"))
fwd_imp
'''
Evaluating and exporting feature importance...
✅ Full feature importances exported to '/data/home/asher.katz/Projects/gender_differences/data/processed/forward_stability_feature_importance.csv' and Top 20 plot saved to '/data/home/asher.katz/Projects/gender_differences/plots/forward_stability_feature_importance_top20.html'.
	Feature	Importance	Category
0	Partner_doing_last_7_days:_housework,_looking_...	0.04518	Base Survey Question
1	Feeling_of_safety_of_walking_alone_in_local_ar...	0.04234	Base Survey Question
2	Hours_normally_worked_a_week_in_main_job_overt...	0.02898	Base Survey Question
3	Doing_last_7_days:_housework,_looking_after_ch...	0.02876	Base Survey Question
4	Partner_doing_last_7_days:_not_applicable	0.02172	Base Survey Question
...	...	...	...
127	State_of_health_services_in_country_nowadays_i...	-0.00032	Other Missing Flag
128	Immigrants_make_country_worse_or_better_place_...	-0.00038	Base Survey Question
129	Main_activity_last_7_days	-0.00038	Base Survey Question
130	Voted_last_national_election	-0.00046	Base Survey Question
131	Main_activity_last_7_days_is_na	-0.00048	Not Applicable Flag
132 rows × 3 columns


'''

#%%
eval_X_test_selected_bwd, eval_y_test_selected_bwd = subsample_xy(X_test_selected_bwd, y_test_selected_bwd)
bwd_imp = export_separate_feature_importance(hgb_model_bwd, eval_X_test_selected_bwd, eval_y_test_selected_bwd, output_csv=here("data/processed/backward_stability_feature_importance.csv"), output_html=here("plots/backward_stability_feature_importance_top20.html"))
bwd_imp
'''
Evaluating and exporting feature importance...
✅ Full feature importances exported to '/data/home/asher.katz/Projects/gender_differences/data/processed/forward_stability_feature_importance.csv' and Top 20 plot saved to '/data/home/asher.katz/Projects/gender_differences/plots/forward_stability_feature_importance_top20.html'.
Feature	Importance	Category
0	Feeling_of_safety_of_walking_alone_in_local_ar...	0.04310	Base Survey Question
1	Partner_doing_last_7_days:_housework,_looking_...	0.04276	Base Survey Question
2	Doing_last_7_days:_housework,_looking_after_ch...	0.02900	Base Survey Question
3	Hours_normally_worked_a_week_in_main_job_overt...	0.01910	Base Survey Question
4	Partner_doing_last_7_days:_not_applicable	0.01608	Base Survey Question
...	...	...	...
137	Trust_in_the_European_Parliament	-0.00094	Base Survey Question
138	Trust_in_the_European_Parliament_is_missing	-0.00112	Other Missing Flag
139	Placement_on_left_right_scale	-0.00132	Base Survey Question
140	Most_of_the_time_people_helpful_or_mostly_look...	-0.00144	Base Survey Question
141	Most_of_the_time_people_helpful_or_mostly_look...	-0.00172	Other Missing Flag
142 rows × 3 columns
'''
