'''
ESS PIPELINE: DUAL INDICATOR FLAGS (HIST-GRADIENT BOOSTING & NATIVE SPLITS)
====================================================================================================
'''


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
# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================
def subsample_xy(X, y, n_max=10000, random_state=42):
    """
    Subsamples up to n_max rows from X (and the matching rows of y) for cheaper
    permutation-importance evaluation. Used both for feature selection and final export.
    """
    X_sub = X.sample(n=min(n_max, len(X)), random_state=random_state)
    y_sub = y.loc[X_sub.index]
    return X_sub, y_sub

def _missing_code_variants(digit):
    variants = set()
    for reps in range(1, 5):
        n = int(str(digit) * reps)
        variants |= {n, float(n), str(n)}
    return variants


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


def is_strictly_valid_str(s):
    return s.notna() & (~s.astype(str).str.strip().isin(string_missing_codes))


def print_gender_distribution(series, label_for_code, code_word="code"):
    """
    Prints per-value row counts for a gender/target series, mapping each numeric code
    to a human-readable label via label_for_code(code).
    """
    counts = series.value_counts().sort_index()
    for code, count in counts.items():
        code_int = int(code)
        print(f"{label_for_code(code_int)} ({code_word} {code_int}): {count} rows")


def eval_group(g):
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})





#%%
# ==================================================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA
# ==================================================================================================
print("Step 1: Loading raw SPSS file and metadata...")  
# Step 1: Loading raw SPSS file and metadata...

path = here('data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(str(path), user_missing=True)

print(f"Number of unique countries: {df_raw['cntry'].nunique()}")  
# Number of unique countries: 38


#%%
# change the column names to the description of the feature
raw_labels = meta.column_names_to_labels  
# Get a dictionary mapping variable names (e.g., 'gndr') to their human-readable labels (e.g., 'Gender')
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}  
# Replace spaces in labels with underscores for cleaner column naming
cntry_val_labels = meta.variable_value_labels.get('cntry', {})  
# Get value-to-label mapping for country codes (e.g., 1 → 'Austria')

# Identify the gender column — try common names first, default to 'gndr' if neither is found
gender_raw_col = next((c for c in ['gndr', 'gender'] if c in df_raw.columns), 'gndr')

# Filter rows where gender values are valid (i.e., 1 or 2 — typically male/female in ESS)
valid_mask = df_raw[gender_raw_col].isin([1, 2, 1.0, 2.0])
df_sub = df_raw[valid_mask].copy()  
# Keep only valid gender responses


#%%


#%%
print(f"Gender distribution before filtering:")  # Gender distribution before filtering:
print_gender_distribution(
    df_sub[gender_raw_col],
    lambda c: "Male" if c == 1 else ("Female" if c == 2 else "Unknown"),
)
# Male (code 1): 198789 rows
# Female (code 2): 231749 rows



#%%
# ==================================================================================================
# STEP 2: FIND COLUMNS WITH COVERAGE ACROSS ALL COUNTRY/ROUND COMBOS 
# ==================================================================================================
print("Step 2: Identifying columns present across all Country/Round pairs...")  # Step 2: Identifying columns present across all Country/Round pairs...

# Missing value codes used in ESS surveys: 6=Not Applicable, 7=Refusal, 8=Don't Know, 9=No Answer/Blank.
# Generated programmatically (int/float/str variants at 1-4 digit repetitions) so ALL_MISSING_CODES and
# NOT_APPLICABLE_CODES (defined later, in Step 4) can never drift out of sync with each other.

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
# ==================================================================================================
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

# ---- Drop irrelevant columns (after extracting Country/ESS_round/Gender) ----
# List of column names to drop — mostly metadata, identifiers, and sensitive or redundant variables
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

# Identify which columns in cols_to_drop actually exist in df_subset (some may have been filtered out earlier)
existing_drops = [c for c in cols_to_drop if c in df_subset.columns]
missing_from_drop_list = [c for c in cols_to_drop if c not in df_subset.columns]

# Report any missing entries — likely already excluded by retained_cols filter
if missing_from_drop_list:
    print(f"Note: {len(missing_from_drop_list)} cols_to_drop entries not found in df_subset "
          f"(already excluded by retained_cols filter): {missing_from_drop_list}")  # Note: 14 cols_to_drop entries not found in df_subset (already excluded by retained_cols filter): ['cntry', 'essround', 'gndr', 'Title_of_dataset', 'Edition', 'Production_date', 'Country_of_birth', 'Country_of_birth,_father', 'Language_most_often_spoken_at_home:_first_mentioned', 'Country_of_birth,_mother', 'nan_count', 'Citizenship', 'Language_most_often_spoken_at_home:_second_mentioned', 'Region']

# Drop the identified columns from df_subset
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

# Convert transformed dictionary to DataFrame
df_processed = pd.DataFrame(transformed)

# Encode gender labels into numeric targets for classification (e.g., 'Male' → 0, 'Female' → 1)
le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['Gender'].astype(str))




#%%

print('\n'.join(df_processed.columns))
print("-------------------------------------------\n Number of Columns:", len(df_processed.columns))
"""
Country
ESS_round
Gender
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair_is_na
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair_is_missing
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_na
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_missing
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves
Most_people_can_be_trusted_or_you_can't_be_too_careful_is_na
Most_people_can_be_trusted_or_you_can't_be_too_careful_is_missing
Most_people_can_be_trusted_or_you_can't_be_too_careful
Worn_or_displayed_campaign_badge/sticker_last_12_months_is_na
Worn_or_displayed_campaign_badge/sticker_last_12_months_is_missing
Worn_or_displayed_campaign_badge/sticker_last_12_months
Boycotted_certain_products_last_12_months_is_na
Boycotted_certain_products_last_12_months_is_missing
Boycotted_certain_products_last_12_months
Feel_closer_to_a_particular_party_than_all_other_parties_is_na
Feel_closer_to_a_particular_party_than_all_other_parties_is_missing
Feel_closer_to_a_particular_party_than_all_other_parties
Contacted_politician_or_government_official_last_12_months_is_na
Contacted_politician_or_government_official_last_12_months_is_missing
Contacted_politician_or_government_official_last_12_months
Gays_and_lesbians_free_to_live_life_as_they_wish_is_na
Gays_and_lesbians_free_to_live_life_as_they_wish_is_missing
Gays_and_lesbians_free_to_live_life_as_they_wish
Government_should_reduce_differences_in_income_levels_is_na
Government_should_reduce_differences_in_income_levels_is_missing
Government_should_reduce_differences_in_income_levels
Placement_on_left_right_scale_is_na
Placement_on_left_right_scale_is_missing
Placement_on_left_right_scale
Taken_part_in_lawful_public_demonstration_last_12_months_is_na
Taken_part_in_lawful_public_demonstration_last_12_months_is_missing
Taken_part_in_lawful_public_demonstration_last_12_months
How_interested_in_politics_is_na
How_interested_in_politics_is_missing
How_interested_in_politics
How_close_to_party_is_na
How_close_to_party_is_missing
How_close_to_party
Signed_petition_last_12_months_is_na
Signed_petition_last_12_months_is_missing
Signed_petition_last_12_months
How_satisfied_with_the_way_democracy_works_in_country_is_na
How_satisfied_with_the_way_democracy_works_in_country_is_missing
How_satisfied_with_the_way_democracy_works_in_country
How_satisfied_with_present_state_of_economy_in_country_is_na
How_satisfied_with_present_state_of_economy_in_country_is_missing
How_satisfied_with_present_state_of_economy_in_country
State_of_education_in_country_nowadays_is_na
State_of_education_in_country_nowadays_is_missing
State_of_education_in_country_nowadays
State_of_health_services_in_country_nowadays_is_na
State_of_health_services_in_country_nowadays_is_missing
State_of_health_services_in_country_nowadays
How_satisfied_with_life_as_a_whole_is_na
How_satisfied_with_life_as_a_whole_is_missing
How_satisfied_with_life_as_a_whole
Trust_in_the_European_Parliament_is_na
Trust_in_the_European_Parliament_is_missing
Trust_in_the_European_Parliament
Trust_in_the_legal_system_is_na
Trust_in_the_legal_system_is_missing
Trust_in_the_legal_system
Trust_in_the_police_is_na
Trust_in_the_police_is_missing
Trust_in_the_police
Trust_in_politicians_is_na
Trust_in_politicians_is_missing
Trust_in_politicians
Trust_in_country's_parliament_is_na
Trust_in_country's_parliament_is_missing
Trust_in_country's_parliament
Trust_in_the_United_Nations_is_na
Trust_in_the_United_Nations_is_missing
Trust_in_the_United_Nations
Voted_last_national_election_is_na
Voted_last_national_election_is_missing
Voted_last_national_election
Worked_in_political_party_or_action_group_last_12_months_is_na
Worked_in_political_party_or_action_group_last_12_months_is_missing
Worked_in_political_party_or_action_group_last_12_months
Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_na
Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_missing
Allow_many/few_immigrants_from_poorer_countries_outside_Europe
Immigration_bad_or_good_for_country's_economy_is_na
Immigration_bad_or_good_for_country's_economy_is_missing
Immigration_bad_or_good_for_country's_economy
Country's_cultural_life_undermined_or_enriched_by_immigrants_is_na
Country's_cultural_life_undermined_or_enriched_by_immigrants_is_missing
Country's_cultural_life_undermined_or_enriched_by_immigrants
Immigrants_make_country_worse_or_better_place_to_live_is_na
Immigrants_make_country_worse_or_better_place_to_live_is_missing
Immigrants_make_country_worse_or_better_place_to_live
Feeling_of_safety_of_walking_alone_in_local_area_after_dark_is_na
Feeling_of_safety_of_walking_alone_in_local_area_after_dark_is_missing
Feeling_of_safety_of_walking_alone_in_local_area_after_dark
Belong_to_minority_ethnic_group_in_country_is_na
Belong_to_minority_ethnic_group_in_country_is_missing
Belong_to_minority_ethnic_group_in_country
Born_in_country_is_na
Born_in_country_is_missing
Born_in_country
Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_na
Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_missing
Respondent_or_household_member_victim_of_burglary/assault_last_5_years
Citizen_of_country_is_na
Citizen_of_country_is_missing
Citizen_of_country
Discrimination_of_respondent's_group:_age_is_na
Discrimination_of_respondent's_group:_age_is_missing
Discrimination_of_respondent's_group:_age
Discrimination_of_respondent's_group:_don't_know_is_na
Discrimination_of_respondent's_group:_don't_know_is_missing
Discrimination_of_respondent's_group:_don't_know
Discrimination_of_respondent's_group:_disability_is_na
Discrimination_of_respondent's_group:_disability_is_missing
Discrimination_of_respondent's_group:_disability
Discrimination_of_respondent's_group:_ethnic_group_is_na
Discrimination_of_respondent's_group:_ethnic_group_is_missing
Discrimination_of_respondent's_group:_ethnic_group
Member_of_a_group_discriminated_against_in_this_country_is_na
Member_of_a_group_discriminated_against_in_this_country_is_missing
Member_of_a_group_discriminated_against_in_this_country
Discrimination_of_respondent's_group:_language_is_na
Discrimination_of_respondent's_group:_language_is_missing
Discrimination_of_respondent's_group:_language
Discrimination_of_respondent's_group:_no_answer_is_na
Discrimination_of_respondent's_group:_no_answer_is_missing
Discrimination_of_respondent's_group:_no_answer
Discrimination_of_respondent's_group:_not_applicable_is_na
Discrimination_of_respondent's_group:_not_applicable_is_missing
Discrimination_of_respondent's_group:_not_applicable
Discrimination_of_respondent's_group:_nationality_is_na
Discrimination_of_respondent's_group:_nationality_is_missing
Discrimination_of_respondent's_group:_nationality
Discrimination_of_respondent's_group:_colour_or_race_is_na
Discrimination_of_respondent's_group:_colour_or_race_is_missing
Discrimination_of_respondent's_group:_colour_or_race
Discrimination_of_respondent's_group:_refusal_is_na
Discrimination_of_respondent's_group:_refusal_is_missing
Discrimination_of_respondent's_group:_refusal
Discrimination_of_respondent's_group:_religion_is_na
Discrimination_of_respondent's_group:_religion_is_missing
Discrimination_of_respondent's_group:_religion
Discrimination_of_respondent's_group:_sexuality_is_na
Discrimination_of_respondent's_group:_sexuality_is_missing
Discrimination_of_respondent's_group:_sexuality
Father_born_in_country_is_na
Father_born_in_country_is_missing
Father_born_in_country
How_happy_are_you_is_na
How_happy_are_you_is_missing
How_happy_are_you
Subjective_general_health_is_na
Subjective_general_health_is_missing
Subjective_general_health
Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_na
Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_missing
Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem
Mother_born_in_country_is_na
Mother_born_in_country_is_missing
Mother_born_in_country
How_often_pray_apart_from_at_religious_services_is_na
How_often_pray_apart_from_at_religious_services_is_missing
How_often_pray_apart_from_at_religious_services
How_often_attend_religious_services_apart_from_special_occasions_is_na
How_often_attend_religious_services_apart_from_special_occasions_is_missing
How_often_attend_religious_services_apart_from_special_occasions
How_religious_are_you_is_na
How_religious_are_you_is_missing
How_religious_are_you
Take_part_in_social_activities_compared_to_others_of_same_age_is_na
Take_part_in_social_activities_compared_to_others_of_same_age_is_missing
Take_part_in_social_activities_compared_to_others_of_same_age
How_often_socially_meet_with_friends,_relatives_or_colleagues_is_na
How_often_socially_meet_with_friends,_relatives_or_colleagues_is_missing
How_often_socially_meet_with_friends,_relatives_or_colleagues
Number_of_people_living_regularly_as_member_of_household_is_na
Number_of_people_living_regularly_as_member_of_household_is_missing
Number_of_people_living_regularly_as_member_of_household
Year_of_birth_is_na
Year_of_birth_is_missing
Year_of_birth
Year_of_birth_of_fourth_person_in_household_is_na
Year_of_birth_of_fourth_person_in_household_is_missing
Year_of_birth_of_fourth_person_in_household
Year_of_birth_of_fifth_person_in_household_is_na
Year_of_birth_of_fifth_person_in_household_is_missing
Year_of_birth_of_fifth_person_in_household
Year_of_birth_of_sixth_person_in_household_is_na
Year_of_birth_of_sixth_person_in_household_is_missing
Year_of_birth_of_sixth_person_in_household
Age_of_respondent,_calculated_is_na
Age_of_respondent,_calculated_is_missing
Age_of_respondent,_calculated
Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_na
Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_missing
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
Ever_had_children_living_in_household_is_na
Ever_had_children_living_in_household_is_missing
Ever_had_children_living_in_household
Doing_last_7_days:_community_or_military_service_is_na
Doing_last_7_days:_community_or_military_service_is_missing
Doing_last_7_days:_community_or_military_service
Partner_doing_last_7_days:_community_or_military_service_is_na
Partner_doing_last_7_days:_community_or_military_service_is_missing
Partner_doing_last_7_days:_community_or_military_service
Doing_last_7_days:_don't_know_is_na
Doing_last_7_days:_don't_know_is_missing
Doing_last_7_days:_don't_know
Partner_doing_last_7_days:_don't_know_is_na
Partner_doing_last_7_days:_don't_know_is_missing
Partner_doing_last_7_days:_don't_know
Doing_last_7_days:_no_answer_is_na
Doing_last_7_days:_no_answer_is_missing
Doing_last_7_days:_no_answer
Partner_doing_last_7_days:_no_answer_is_na
Partner_doing_last_7_days:_no_answer_is_missing
Partner_doing_last_7_days:_no_answer
Partner_doing_last_7_days:_not_applicable_is_na
Partner_doing_last_7_days:_not_applicable_is_missing
Partner_doing_last_7_days:_not_applicable
Doing_last_7_days:_other_is_na
Doing_last_7_days:_other_is_missing
Doing_last_7_days:_other
Partner_doing_last_7_days:_other_is_na
Partner_doing_last_7_days:_other_is_missing
Partner_doing_last_7_days:_other
Doing_last_7_days:_refusal_is_na
Doing_last_7_days:_refusal_is_missing
Doing_last_7_days:_refusal
Partner_doing_last_7_days:_refusal_is_na
Partner_doing_last_7_days:_refusal_is_missing
Partner_doing_last_7_days:_refusal
Domicile,_respondent's_description_is_na
Domicile,_respondent's_description_is_missing
Domicile,_respondent's_description
Doing_last_7_days:_permanently_sick_or_disabled_is_na
Doing_last_7_days:_permanently_sick_or_disabled_is_missing
Doing_last_7_days:_permanently_sick_or_disabled
Partner_doing_last_7_days:_permanently_sick_or_disabled_is_na
Partner_doing_last_7_days:_permanently_sick_or_disabled_is_missing
Partner_doing_last_7_days:_permanently_sick_or_disabled
Doing_last_7_days:_education_is_na
Doing_last_7_days:_education_is_missing
Doing_last_7_days:_education
Partner_doing_last_7_days:_education_is_na
Partner_doing_last_7_days:_education_is_missing
Partner_doing_last_7_days:_education
Years_of_full-time_education_completed_is_na
Years_of_full-time_education_completed_is_missing
Years_of_full-time_education_completed
Highest_level_of_education,_ES_-_ISCED_is_na
Highest_level_of_education,_ES_-_ISCED_is_missing
Highest_level_of_education,_ES_-_ISCED
Number_of_employees_respondent_has/had_is_na
Number_of_employees_respondent_has/had_is_missing
Number_of_employees_respondent_has/had
Father's_employment_status_when_respondent_14_is_na
Father's_employment_status_when_respondent_14_is_missing
Father's_employment_status_when_respondent_14
Mother's_employment_status_when_respondent_14_is_na
Mother's_employment_status_when_respondent_14_is_missing
Mother's_employment_status_when_respondent_14
Establishment_size_is_na
Establishment_size_is_missing
Establishment_size
Doing_last_7_days:_housework,_looking_after_children,_others_is_na
Doing_last_7_days:_housework,_looking_after_children,_others_is_missing
Doing_last_7_days:_housework,_looking_after_children,_others
Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_na
Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_missing
Partner_doing_last_7_days:_housework,_looking_after_children,_others
Responsible_for_supervising_other_employees_is_na
Responsible_for_supervising_other_employees_is_missing
Responsible_for_supervising_other_employees
Main_activity_last_7_days_is_na
Main_activity_last_7_days_is_missing
Main_activity_last_7_days
Main_activity,_last_7_days._All_respondents._Post_coded_is_na
Main_activity,_last_7_days._All_respondents._Post_coded_is_missing
Main_activity,_last_7_days._All_respondents._Post_coded
Partner's_main_activity_last_7_days_is_na
Partner's_main_activity_last_7_days_is_missing
Partner's_main_activity_last_7_days
Number_of_people_responsible_for_in_job_is_na
Number_of_people_responsible_for_in_job_is_missing
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job_is_na
Ever_had_a_paid_job_is_missing
Ever_had_a_paid_job
Year_last_in_paid_job_is_na
Year_last_in_paid_job_is_missing
Year_last_in_paid_job
Doing_last_7_days:_paid_work_is_na
Doing_last_7_days:_paid_work_is_missing
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work_is_na
Partner_doing_last_7_days:_paid_work_is_missing
Partner_doing_last_7_days:_paid_work
Doing_last_7_days:_retired_is_na
Doing_last_7_days:_retired_is_missing
Doing_last_7_days:_retired
Partner_doing_last_7_days:_retired_is_na
Partner_doing_last_7_days:_retired_is_missing
Partner_doing_last_7_days:_retired
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_na
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_missing
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_na
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_missing
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months
Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_na
Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_missing
Any_period_of_unemployment_and_work_seeking_within_last_5_years
Doing_last_7_days:_unemployed,_actively_looking_for_job_is_na
Doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing
Doing_last_7_days:_unemployed,_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_na
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job
Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na
Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing
Doing_last_7_days:_unemployed,_not_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job
Hours_normally_worked_a_week_in_main_job_overtime_included,_partner_is_na
Hours_normally_worked_a_week_in_main_job_overtime_included,_partner_is_missing
Hours_normally_worked_a_week_in_main_job_overtime_included,_partner
target_encoded
-------------------------------------------
 Number of Columns: 334
"""


#%%
# ==================================================================================================
# STEP 5: GENERATE MISSINGNESS SUMMARY TABLE PER COUNTRY/ROUND
# ==================================================================================================
print("Step 5: Generating missingness summary table per country and round...")  # Step 5: Generating missingness summary table per country and round...

# base_features (raw feature names, excluding grouping/target columns) was already computed in Step 4 —
# the indicator-flag and target columns added since then don't change that underlying set, so it's reused as-is.

# Use the helper function to generate missingness summary
summary_df = create_subset_summary(df_processed, base_features, df_raw)
print("✅ Missingness summary table generated successfully!")  # ✅ Missingness summary table generated successfully!
print(summary_df.head(10))  #
#    Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  9.133298           6.263720
# 1  Austria        1.0  ...                 10.930036           8.106497
# 2  Austria        2.0  ...                 10.736622           7.836074
# 3  Austria        3.0  ...                 10.874693           8.313362
# 4  Austria        7.0  ...                 11.099519           7.081286
# 5  Austria        8.0  ...                 11.436454           7.578019
# 6  Austria        9.0  ...                 11.068064           8.680927
# 7  Belgium        1.0  ...                 11.018474           8.951386
# 8  Belgium        2.0  ...                 11.178546           8.186420
# 9  Belgium        3.0  ...                 11.261503           8.554454
#
# [10 rows x 5 columns]



#%%

# Re-run gender distribution check at this point in script (after preprocessing steps 2–5)
print("\n" + "="*84)
print("Gender Distribution After Preprocessing Steps (Steps 2–5)")
print("="*84)

# Use df_processed, which has already been cleaned and encoded.
# Maps back to human-readable labels using the LabelEncoder's classes_.
print_gender_distribution(
    df_processed['target_encoded'],
    lambda c: le.inverse_transform([c])[0],
    code_word="encoded",
)
print("\n✅ Gender distribution check completed.")  # 
# ✅ Gender distribution check completed.


#%%
# Export the summary to CSV
output_summary_csv = here("data/processed/gender_full_row_summary.csv")
summary_df.to_csv(output_summary_csv, index=False)
print(f"✅ Missingness summary table saved to '{output_summary_csv}'")  # ✅ Missingness summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/gender_full_row_summary.csv'
print(f"   Summary shape: {summary_df.shape[0]} rows (country-round pairs), {summary_df.shape[1]} columns")  #    Summary shape: 228 rows (country-round pairs), 5 columns



#%%
# Export the processed dataset to CSV
output_csv = here("data/processed/gender_processed_two_missingness_indicators.csv")
df_processed.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_processed.shape} to '{output_csv}'")  # ✅ Processed dataset saved with shape (430538, 334) to '/data/home/asher.katz/Projects/gender_differences/data/processed/gender_processed_two_missingness_indicators.csv'





#%%
# ==================================================================================================
# STEP 6: BALANCED SAMPLING & TRAIN/TEST SPLIT
# ==================================================================================================
# Perform balanced stratified train/test split by Country and ESS_round.
print("Step 6: Performing balanced stratified train/test split...")  # Step 6: Performing balanced stratified train/test split...

# Compute the minimum group size across all (Country, ESS_round) combinations — ensures equal representation
counts = df_processed.groupby(['Country', 'ESS_round']).size()
train_n = int(counts.min() * 0.8)

# For each group (Country, ESS_round), sample exactly `min(train_n, len(group))` rows at random.
# This guarantees balanced sampling across groups while preserving stratification by Country and Round.
training_data = df_processed.groupby(['Country', 'ESS_round'], group_keys=False).apply(
    lambda x: x.sample(n=min(train_n, len(x)), random_state=42), include_groups=False
)

# The test set consists of all rows *not* included in the training data (i.e., remaining observations).
test_data = df_processed.drop(index=training_data.index).copy()

# Define feature columns as all columns except grouping variables and target.
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'Gender', 'target_encoded']]

# Split into design matrix (X) and response vector (y).
X_train = training_data[feature_cols]
y_train = training_data['target_encoded']
X_test = test_data[feature_cols]
y_test = test_data['target_encoded']




#%%

# ==================================================================================================
# STEP 7: CREATE TRAIN AND TEST SUMMARY TABLES
# ==================================================================================================
print("Step 7: Creating train and test summary tables...")  # Step 7: Creating train and test summary tables...

# Create train and test summary tables
train_summary = create_subset_summary(training_data, base_features, df_processed)
test_summary = create_subset_summary(test_data, base_features, df_processed)

# Save to CSV files
output_train_csv = here("data/processed/gender_train_row_summary.csv")
output_test_csv = here("data/processed/gender_test_row_summary.csv")

train_summary.to_csv(output_train_csv, index=False)
print(f"✅ Train summary table saved to '{output_train_csv}'")  # ✅ Train summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/gender_train_row_summary.csv'
print(f"   Shape: {train_summary.shape[0]} rows (country-round pairs), {train_summary.shape[1]} columns")  #    Shape: 228 rows (country-round pairs), 5 columns

test_summary.to_csv(output_test_csv, index=False)
print(f"✅ Test summary table saved to '{output_test_csv}'")  # ✅ Test summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/gender_test_row_summary.csv'
print(f"   Shape: {test_summary.shape[0]} rows (country-round pairs), {test_summary.shape[1]} columns")  #    Shape: 228 rows (country-round pairs), 5 columns

# Optional: Display first few rows
print("\nTrain Summary Preview:")  # 
print(train_summary.head(5))  #    Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
                                # 0  Albania        6.0  ...                  9.112839           6.259968
                                # 1  Austria        1.0  ...                 10.861244           7.878788
                                # 2  Austria        2.0  ...                 10.807416           7.826954
                                # 3  Austria        3.0  ...                 10.944976           8.403110
                                # 4  Austria        7.0  ...                 11.084530           7.242823
                                # 
                                # [5 rows x 5 columns]

print("\nTest Summary Preview:")  # 
print(test_summary.head(5))  #    Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
                                # 0  Albania        6.0  ...                  9.145821           6.266016
                                # 1  Austria        1.0  ...                 10.947453           8.164151
                                # 2  Austria        2.0  ...                 10.718687           7.838384
                                # 3  Austria        3.0  ...                 10.858249           8.292364
                                # 4  Austria        7.0  ...                 11.104624           7.026275
                                # 
                                # [5 rows x 5 columns]


#%%
# ==================================================================================================
# STEP 8: MODEL TRAINING & SELECTION
# ==================================================================================================
# Fit a baseline HistGradientBoostingClassifier on the full training set.
# This model will be used to compute feature importances via permutation importance.
print("Step 8: Fitting HistGradientBoostingClassifier...")

baseline_hgb = HistGradientBoostingClassifier(random_state=42)
baseline_hgb.fit(X_train, y_train)

# To reduce computational cost of permutation importance,
# we subsample the training data (up to 10,000 rows) for evaluation.
X_select_sample, y_select_sample = subsample_xy(X_train, y_train)

# Compute permutation importance on the subsampled data:
# - Measures how much model performance drops when a feature is randomly permuted
# - Higher importance = more predictive power
importance_df = compute_permutation_importance_df(baseline_hgb, X_select_sample, y_select_sample)

# Select only features with strictly positive importance (i.e., those that contribute to performance).
selected_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()

# Identify dropped features — those with zero or negative importance.
dropped_features = importance_df[importance_df['Importance'] <= 0]['Feature'].tolist()



#%%
# Final training on selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
print('\n'.join(X_train_selected.columns))
print("-------------------------------------------\n Number of Columns:", len(X_train_selected.columns))

"""
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair_is_missing
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_missing
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves
Worn_or_displayed_campaign_badge/sticker_last_12_months
Contacted_politician_or_government_official_last_12_months
Gays_and_lesbians_free_to_live_life_as_they_wish_is_missing
Gays_and_lesbians_free_to_live_life_as_they_wish
Government_should_reduce_differences_in_income_levels
Placement_on_left_right_scale_is_missing
Placement_on_left_right_scale
Taken_part_in_lawful_public_demonstration_last_12_months
How_interested_in_politics_is_missing
How_interested_in_politics
How_close_to_party
Signed_petition_last_12_months
How_satisfied_with_the_way_democracy_works_in_country
How_satisfied_with_present_state_of_economy_in_country_is_missing
How_satisfied_with_present_state_of_economy_in_country
State_of_health_services_in_country_nowadays_is_missing
State_of_health_services_in_country_nowadays
How_satisfied_with_life_as_a_whole_is_missing
Trust_in_the_European_Parliament_is_na
Trust_in_the_European_Parliament_is_missing
Trust_in_the_European_Parliament
Trust_in_the_legal_system_is_na
Trust_in_the_legal_system_is_missing
Trust_in_the_legal_system
Trust_in_the_police_is_na
Trust_in_the_police_is_missing
Trust_in_the_police
Trust_in_politicians
Trust_in_country's_parliament_is_na
Trust_in_country's_parliament_is_missing
Trust_in_country's_parliament
Trust_in_the_United_Nations_is_missing
Voted_last_national_election
Worked_in_political_party_or_action_group_last_12_months
Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_missing
Allow_many/few_immigrants_from_poorer_countries_outside_Europe
Immigration_bad_or_good_for_country's_economy_is_na
Immigration_bad_or_good_for_country's_economy_is_missing
Immigration_bad_or_good_for_country's_economy
Country's_cultural_life_undermined_or_enriched_by_immigrants_is_missing
Country's_cultural_life_undermined_or_enriched_by_immigrants
Immigrants_make_country_worse_or_better_place_to_live_is_missing
Immigrants_make_country_worse_or_better_place_to_live
Feeling_of_safety_of_walking_alone_in_local_area_after_dark_is_missing
Feeling_of_safety_of_walking_alone_in_local_area_after_dark
Respondent_or_household_member_victim_of_burglary/assault_last_5_years
Discrimination_of_respondent's_group:_age
Discrimination_of_respondent's_group:_ethnic_group
Member_of_a_group_discriminated_against_in_this_country
Discrimination_of_respondent's_group:_not_applicable
Discrimination_of_respondent's_group:_nationality
Discrimination_of_respondent's_group:_colour_or_race
Discrimination_of_respondent's_group:_religion
Discrimination_of_respondent's_group:_sexuality
Father_born_in_country
How_happy_are_you_is_na
How_happy_are_you_is_missing
Subjective_general_health
How_often_pray_apart_from_at_religious_services_is_na
How_often_pray_apart_from_at_religious_services_is_missing
How_often_pray_apart_from_at_religious_services
How_often_attend_religious_services_apart_from_special_occasions
How_religious_are_you_is_na
How_religious_are_you_is_missing
How_religious_are_you
Take_part_in_social_activities_compared_to_others_of_same_age
How_often_socially_meet_with_friends,_relatives_or_colleagues_is_na
How_often_socially_meet_with_friends,_relatives_or_colleagues_is_missing
How_often_socially_meet_with_friends,_relatives_or_colleagues
Number_of_people_living_regularly_as_member_of_household
Year_of_birth
Year_of_birth_of_fourth_person_in_household_is_na
Year_of_birth_of_fourth_person_in_household
Year_of_birth_of_fifth_person_in_household_is_missing
Age_of_respondent,_calculated
Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_missing
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
Ever_had_children_living_in_household_is_na
Ever_had_children_living_in_household_is_missing
Ever_had_children_living_in_household
Doing_last_7_days:_community_or_military_service
Partner_doing_last_7_days:_not_applicable
Domicile,_respondent's_description
Doing_last_7_days:_permanently_sick_or_disabled
Partner_doing_last_7_days:_permanently_sick_or_disabled
Doing_last_7_days:_education
Partner_doing_last_7_days:_education
Years_of_full-time_education_completed
Highest_level_of_education,_ES_-_ISCED_is_na
Highest_level_of_education,_ES_-_ISCED_is_missing
Highest_level_of_education,_ES_-_ISCED
Number_of_employees_respondent_has/had
Father's_employment_status_when_respondent_14_is_missing
Father's_employment_status_when_respondent_14
Mother's_employment_status_when_respondent_14
Establishment_size_is_na
Establishment_size_is_missing
Establishment_size
Doing_last_7_days:_housework,_looking_after_children,_others
Partner_doing_last_7_days:_housework,_looking_after_children,_others
Main_activity_last_7_days_is_na
Main_activity_last_7_days_is_missing
Main_activity_last_7_days
Main_activity,_last_7_days._All_respondents._Post_coded_is_missing
Main_activity,_last_7_days._All_respondents._Post_coded
Partner's_main_activity_last_7_days_is_missing
Partner's_main_activity_last_7_days
Number_of_people_responsible_for_in_job_is_na
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job_is_na
Ever_had_a_paid_job
Year_last_in_paid_job
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work
Partner_doing_last_7_days:_retired
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_na
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months
Any_period_of_unemployment_and_work_seeking_within_last_5_years
Doing_last_7_days:_unemployed,_actively_looking_for_job
Hours_normally_worked_a_week_in_main_job_overtime_included,_partner
-------------------------------------------
 Number of Columns: 125
 """
#%%

hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_model.fit(X_train_selected, y_train)

test_data['y_pred'] = hgb_model.predict(X_test_selected)
test_data['y_prob'] = hgb_model.predict_proba(X_test_selected)[:, 1]


# Group test data by Country and ESS_round, then compute accuracy and F1 per group
perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()

# Save performance metrics to CSV for downstream analysis
perf_df.to_csv(here("data/processed/gender_country_and_round_performance_indicators.csv"), index=False)




#%%
# ==================================================================================================
# STEP 9: computing feature importances
# ==================================================================================================
print("\nStep 9: Running exports and building plots...")

# Feature Importance Export & Plot
X_eval, y_eval = subsample_xy(X_test_selected, y_test)

#%%
export_separate_feature_importance(hgb_model, X_eval, y_eval, output_csv=here("data/processed/gender_feature_importance.csv"), output_html=here("plots/gender_feature_importance_top20.html"))



#%%

# Calculate average accuracy across all countries for each round
avg_accuracy_by_round = perf_df.groupby('ESS_round')['accuracy'].mean().reset_index()
avg_accuracy_by_round.columns = ['ESS_round', 'average_accuracy']

# Display the results in a clean table format
print("\n==================================================")  # 
print("   AVERAGE GENDER ACCURACY ACROSS ALL COUNTRIES BY ROUND ")  #    AVERAGE GENDER ACCURACY ACROSS ALL COUNTRIES BY ROUND 
print("==================================================")  # ==================================================
for _, row in avg_accuracy_by_round.iterrows():  # Round 1: 0.7847
    print(f"Round {int(row['ESS_round'])}: {row['average_accuracy']:.4f}")  # Round 2: 0.7774
                                                            # Round 3: 0.7776
                                                            # Round 4: 0.7652
                                                            # Round 5: 0.7621
                                                            # Round 6: 0.7604
                                                            # Round 7: 0.7528
                                                            # Round 8: 0.7461
                                                            # Round 9: 0.7362

# Optionally, save to CSV for further analysis
avg_accuracy_by_round.to_csv(here("data/processed/gender_average_accuracy_by_round.csv"), index=False)




#%%
# Create a DataFrame with country-level summary: name, average accuracy, and number of rounds
country_summary = perf_df.groupby('Country').agg(
    avg_accuracy=('accuracy', 'mean'),
    num_rounds=('ESS_round', 'nunique')
).reset_index()

# Rename columns for clarity
country_summary.columns = ['Name_of_country', 'average_accuracy', 'num_rounds']

# Sort first by num_rounds (high to low), then by average_accuracy (high to low)
country_summary = country_summary.sort_values(
    by=['num_rounds', 'average_accuracy'],
    ascending=[False, False]
)
# Save to CSV
output_country_summary_csv = here("data/processed/country_rankings.csv")
country_summary.to_csv(output_country_summary_csv, index=False)
print(f"✅ Country summary saved to '{output_country_summary_csv}'")  
# ✅ Country summary saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/country_rankings.csv' 

# Display preview
print("\nCountry Summary Preview:")  
print(country_summary.head(10))  
#    Name_of_country  average_accuracy  num_rounds
# 23     Netherlands          0.795371           9
# 32           Spain          0.790778           9
# 34     Switzerland          0.787760           9
# 11         Germany          0.784323           9
# 15         Ireland          0.777064           9
# 24          Norway          0.773308           9
# 2          Belgium          0.768452           9
# 31        Slovenia          0.765818           9
# 37  United Kingdom          0.765263           9
# 9          Finland          0.762557           9
# %%
