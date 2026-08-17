'''
====================================================================================================
ESS PIPELINE: DUAL INDICATOR FLAGS (HIST-GRADIENT BOOSTING & NATIVE SPLITS)
====================================================================================================

This Python script analyzes gender differences across European countries using data from the 
European Social Survey (ESS). The pipeline:

1. Loads raw SPSS (.sav) files containing survey responses
2. Identifies columns with complete coverage across all country/round combinations
3. Renames columns to human-readable labels and standardizes metadata headers
4. Handles missing values by creating dual indicator flags:
   - "_is_na" flags: Not Applicable codes (6, 66, 666, etc.)
   - "_is_missing" flags: Other missing data codes (7, 8, 9 variants)
5. Performs balanced stratified train/test split for cross-validation
6. Trains a HistGradientBoostingClassifier to predict gender from survey responses
7. Generates interactive visualizations of accuracy trends and feature importance

The core hypothesis is that differences in how men and women answer survey questions can be 
detected by machine learning models, with lower accuracy suggesting greater gender similarity
in response patterns.
====================================================================================================
'''


#%%
# ==================================================================================================
# IMPORT STATEMENTS: Core library imports for data processing, modeling, and visualization
# ==================================================================================================

from pathlib import Path                           # Provides object-oriented filesystem paths
import pandas as pd                                # Data manipulation and analysis library (DataFrames)
import numpy as np                                 # Numerical computing with support for large arrays/matrices
import pyreadstat                                  # Library to read SPSS (.sav) files into Python

# Machine learning libraries from scikit-learn
from sklearn.ensemble import HistGradientBoostingClassifier  # High-performance gradient boosting classifier
from sklearn.preprocessing import LabelEncoder               # Converts categorical labels to numeric codes
from sklearn.metrics import (                                # Model evaluation metrics
    accuracy_score,                             # Fraction of correct predictions
    precision_recall_fscore_support,            # Precision, recall, F1 score, support per class
    roc_auc_score                               # Area Under ROC Curve for binary classification
)
from sklearn.inspection import permutation_importance  # Computes feature importance via permutation

# Statistical modeling libraries from statsmodels
import statsmodels.api as sm                        # For statistical models (OLS, GLM, etc.)
import statsmodels.formula.api as smf              # Formula interface for specifying models

# Interactive visualization library Plotly
import plotly.express as px                         # High-level interface for creating charts
import plotly.graph_objects as go                  # Low-level interface for customizing plots



#%%
# ==================================================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA
# ==================================================================================================

print("Step 1: Loading raw SPSS file and metadata...")

# Define the path to the combined ESS survey dataset saved in SPSS format
FILE_PATH = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')

# Read the SPSS file, extracting both data (df_raw) and metadata (meta)
# user_missing=True ensures missing value codes are preserved rather than converted to NaN
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)


#%%
# Count unique countries in the dataset's 'cntry' column
num_countries = df_raw['cntry'].nunique()
print(f"Number of unique countries: {num_countries}")

# Include all countries to obtain as broad a sampling of women across Europe as possible


#%%
# Extract mapping from variable/column names to their human-readable labels
raw_labels = meta.column_names_to_labels

# Create dictionary mapping original column codes to cleaned labels (spaces replaced with underscores)
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}

# Get country value labels (e.g., 'A' -> 'Austria') from metadata
cntry_val_labels = meta.variable_value_labels.get('cntry', {})

# Determine which column contains gender information by checking for 'gndr' or 'gender'
gender_raw_col = next((c for c in ['gndr', 'gender'] if c in df_raw.columns), 'gndr')

# Filter to keep only valid gender responses (Male=1, Female=2, including float variants)
valid_mask = df_raw[gender_raw_col].isin([1, 2, 1.0, 2.0])
df_sub = df_raw[valid_mask].copy()  # Create a copy of filtered DataFrame



#%%
# ==================================================================================================
# STEP 2: FIND COLUMNS WITH COVERAGE ACROSS ALL COUNTRY/ROUND COMBOS
# ==================================================================================================

print("Step 2: Identifying columns present across all Country/Round pairs...")

# Define codes that represent missing values in ESS surveys:
# - Codes 6, 66, 666, 6666: Typically "Not applicable" or "Don't know"
# - Codes 7, 77, 777, 7777: Often "Refused" or "No answer"
# - Codes 8, 88, 888, 8888: May indicate "Not applicable to this case"
# - Codes 9, 99, 999, 9999: Sometimes used for missing data
# Both numeric and string variants are included to handle type variations
ALL_MISSING_CODES = {
    6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666',
    7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777',
    8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888',
    9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'
}

# Grouping columns: country identifier and survey round number
group_cols = ['cntry', 'essround']

# Candidate columns: all columns except grouping variables (country and round)
candidate_cols = [c for c in df_sub.columns if c not in group_cols]



#%%
def is_strictly_valid(s):
    """
    Check if a Series contains any valid (non-missing) values.
    
    Args:
        s (pd.Series): A pandas Series to check
    
    Returns:
        bool: True if at least one value is not in ALL_MISSING_CODES and not NaN
    """
    return s.notna() & (~s.isin(ALL_MISSING_CODES))


# For each country/round combination, find columns that have at least one valid value
# This ensures we only include columns with complete coverage across all groups
valid_per_group = df_sub.groupby(group_cols)[candidate_cols].apply(
    lambda group: group.apply(lambda col: is_strictly_valid(col).any())
)

# Retain only columns that have valid values in ALL country/round combinations. This way, we can
# gather a training dataset with an equal amount of rows from each country/round combo, all with the same columns
retained_cols = valid_per_group.columns[valid_per_group.all()].tolist()

# Create subset DataFrame with grouping columns and retained feature columns
df_subset = df_sub[group_cols + retained_cols].copy()



#%%
# ==================================================================================================
# STEP 3: RENAME COLUMNS & STANDARDIZE METADATA HEADERS
# ==================================================================================================

print("Step 3: Renaming headers and dropping excluded metadata...")

# Rename all columns using the code_to_label mapping (e.g., 'gndr' -> 'Gender')
df_subset.rename(columns=code_to_label, inplace=True)

# Get human-readable names for grouping columns
cntry_renamed = code_to_label.get(group_cols[0], group_cols[0])  # Country name
round_renamed = code_to_label.get(group_cols[1], group_cols[1])  # ESS round name
gender_renamed = code_to_label.get(gender_raw_col, gender_raw_col)  # Gender column name

# Create standardized 'Country' column by mapping country codes to full names
df_subset['Country'] = df_subset[cntry_renamed].astype(str).str.strip().map(cntry_val_labels).fillna(df_subset[cntry_renamed])

# Store ESS round number in standardized 'ESS_round' column
df_subset['ESS_round'] = df_subset[round_renamed]

# Store gender value in standardized 'Gender' column
df_subset['Gender'] = df_subset[gender_renamed]



#%%
# Define list of columns to drop - these are metadata, identifiers, or sensitive information
# that should not be included in the analysis
cols_to_drop = [
     group_cols[0], group_cols[1], gender_raw_col,  # Original grouping column names
    "Title_of_dataset", "Edition", "Production_date", "Respondent's_identification_number", 
    "Design_weight", "Post-stratification_weight_including_design_weight",
    "Population_size_weight_(must_be_combined_with_dweight_or_pspwght)",
    "Country_of_birth", "Discrimination_of_respondent's_group:_gender",
    "Discrimination_of_respondent's_group:_other_grounds", "Country_of_birth,_father",
    "Language_most_often_spoken_at_home:_first_mentioned", "Country_of_birth,_mother",
    "nan_count", "Citizenship", "Language_most_often_spoken_at_home:_second_mentioned",
    "Region", "Gender_of_second_person_in_household", "Gender_of_third_person_in_household",
    "Gender_of_fourth_person_in_household", "Gender_of_fifth_person_in_household",
    "Gender_of_sixth_person_in_household", "Year_of_birth_of_second_person_in_household",
    "Year_of_birth_of_third_person_in_household", 
]

# Drop the specified columns from df_subset, ignoring errors if some columns don't exist
df_subset.drop(columns=[c for c in cols_to_drop if c in df_subset.columns], errors='ignore', inplace=True)



#%%
# ==================================================================================================
# STEP 4: HANDLE MISSING VALUES (DUAL INDICATORS & ZERO-FILLING)
# ==================================================================================================
print("Step 4: Constructing _is_na and _is_missing indicator flags...")

# Define NOT_APPLICABLE codes - these indicate the question doesn't apply to this respondent
NOT_APPLICABLE_CODES = {6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666'}

# OTHER_MISSING_CODES: All missing codes that are NOT "not applicable" (refused, don't know, etc.)
OTHER_MISSING_CODES = ALL_MISSING_CODES - NOT_APPLICABLE_CODES

# Get list of base feature columns for processing
base_features = [c for c in df_subset.columns if c not in ['Country', 'ESS_round', 'Gender']]

# Initialize dictionary to store transformed data with indicator flags
transformed = {
    'Country': df_subset['Country'], 
    'ESS_round': df_subset['ESS_round'], 
    'Gender': df_subset['Gender']
}

# Initialize counters for reporting missing value types
total_na = 0
total_missing = 0

# Process each base feature column
for c in base_features:
    s = df_subset[c]  # Get the Series for this column
    
    # Check if values are "Not Applicable" codes (6, 66, etc.)
    is_na = s.isin(NOT_APPLICABLE_CODES)
    
    # Check if values are other missing codes OR actual NaN values
    is_miss = s.isin(OTHER_MISSING_CODES) | s.isna()
    
    # Count occurrences for reporting
    na_count = is_na.sum()
    miss_count = is_miss.sum()
    total_na += int(na_count)
    total_missing += int(miss_count)

# Print summary of missing value types and indicator counts
print(f"Total 'Not Applicable' (NA) values across all features: {total_na}")
print(f"Total other missing values (refused/don't know/etc.) + NaNs: {total_missing}")

# Process each base feature column again to build transformed data
for c in base_features:
    s = df_subset[c]  # Get the Series for this column

    # Check if values are "Not Applicable" codes (6, 66, etc.)
    is_na = s.isin(NOT_APPLICABLE_CODES)

    # Check if values are other missing codes OR actual NaN values
    is_miss = s.isin(OTHER_MISSING_CODES) | s.isna()

    # Create indicator flags: 1 where condition is true, 0 otherwise
    transformed[f"{c}_is_na"] = is_na.astype(int)
    transformed[f"{c}_is_missing"] = is_miss.astype(int)
    
    # Convert to numeric, coercing errors to NaN
    num_s = pd.to_numeric(s, errors='coerce')
    
    # Set not applicable and missing values to 0 (for modeling)
    num_s[is_na | is_miss] = 0.0
    
    # Fill any remaining NaN with 0 and store in transformed dict
    transformed[c] = num_s.fillna(0.0)

# Create final processed DataFrame from transformed dictionary
df_processed = pd.DataFrame(transformed)

# Encode gender labels (e.g., 'Male', 'Female') to numeric codes (0, 1)
le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['Gender'].astype(str))

#%%
# second round of drops
cols_to_drop = [
"Voted_last_national_election_is_na",
"Worked_in_political_party_or_action_group_last_12_months_is_missing",
"Trust_in_the_United_Nations_is_na",
"Worked_in_political_party_or_action_group_last_12_months_is_na",
"Voted_last_national_election_is_missing",
"Discrimination_of_respondent's_group:_don't_know_is_missing",
"Discrimination_of_respondent's_group:_don't_know_is_na",
"Discrimination_of_respondent's_group:_age_is_missing",
"Discrimination_of_respondent's_group:_age_is_na",
"Feeling_of_safety_of_walking_alone_in_local_area_after_dark_is_na",
"Belong_to_minority_ethnic_group_in_country_is_na",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_na",
"Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na",
"Discrimination_of_respondent's_group:_colour_or_race_is_na",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_missing",
"Citizen_of_country_is_na",
"Citizen_of_country_is_missing",
"Discrimination_of_respondent's_group:_not_applicable_is_na",
"Number_of_people_responsible_for_in_job_is_missing",
"Ever_had_a_paid_job_is_missing",
"Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_na",
"Year_last_in_paid_job_is_na",
"Number_of_people_responsible_for_in_job_is_na",
"How_interested_in_politics_is_na",
"Feel_closer_to_a_particular_party_than_all_other_parties_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months_is_missing",
"Born_in_country_is_missing",
"Born_in_country_is_na",
"Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing",
"Immigrants_make_country_worse_or_better_place_to_live_is_na",
"Belong_to_minority_ethnic_group_in_country_is_missing",
"Discrimination_of_respondent's_group:_ethnic_group_is_missing",
"Country's_cultural_life_undermined_or_enriched_by_immigrants_is_na",
"Discrimination_of_respondent's_group:_disability_is_missing",
"Discrimination_of_respondent's_group:_disability_is_na",
"Discrimination_of_respondent's_group:_don't_know",
"Discrimination_of_respondent's_group:_ethnic_group_is_na",
"Gays_and_lesbians_free_to_live_life_as_they_wish_is_na",
"Government_should_reduce_differences_in_income_levels_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months_is_na",
"Boycotted_certain_products_last_12_months_is_missing",
"Boycotted_certain_products_last_12_months_is_na",
"Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves",
"Worn_or_displayed_campaign_badge/sticker_last_12_months_is_missing",
"Worn_or_displayed_campaign_badge/sticker_last_12_months_is_na",
"Responsible_for_supervising_other_employees_is_na",
"Responsible_for_supervising_other_employees_is_missing",
"Responsible_for_supervising_other_employees",
"Main_activity_last_7_days_is_missing",
"Discrimination_of_respondent's_group:_colour_or_race_is_missing",
"Discrimination_of_respondent's_group:_refusal",
"Discrimination_of_respondent's_group:_refusal_is_missing",
"Discrimination_of_respondent's_group:_refusal_is_na",
"Discrimination_of_respondent's_group:_religion_is_missing",
"Discrimination_of_respondent's_group:_sexuality_is_na",
"Discrimination_of_respondent's_group:_sexuality_is_missing",
"Discrimination_of_respondent's_group:_religion_is_na",
"Discrimination_of_respondent's_group:_sexuality",
"Member_of_a_group_discriminated_against_in_this_country_is_na",
"Father_born_in_country_is_missing",
"Father_born_in_country_is_na",
"Discrimination_of_respondent's_group:_not_applicable_is_missing",
"Discrimination_of_respondent's_group:_no_answer_is_missing",
"Discrimination_of_respondent's_group:_no_answer",
"Member_of_a_group_discriminated_against_in_this_country_is_missing",
"Discrimination_of_respondent's_group:_language_is_missing",
"Discrimination_of_respondent's_group:_language_is_na",
"Discrimination_of_respondent's_group:_language",
"Discrimination_of_respondent's_group:_no_answer_is_na",
"Discrimination_of_respondent's_group:_nationality_is_na",
"Discrimination_of_respondent's_group:_not_applicable",
"Discrimination_of_respondent's_group:_nationality_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_missing",
"How_often_attend_religious_services_apart_from_special_occasions_is_missing",
"Take_part_in_social_activities_compared_to_others_of_same_age_is_na",
"Mother_born_in_country_is_na",
"Mother_born_in_country_is_missing",
"Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_na",
"Doing_last_7_days:_community_or_military_service_is_na",
"Doing_last_7_days:_community_or_military_service_is_missing",
"Number_of_people_living_regularly_as_member_of_household_is_missing",
"Year_of_birth_is_na",
"Subjective_general_health_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_na",
"Year_of_birth_of_sixth_person_in_household_is_missing",
"Year_of_birth_of_fifth_person_in_household_is_na",
"Age_of_respondent,_calculated_is_na",
"Year_of_birth_of_sixth_person_in_household",
"Mother's_employment_status_when_respondent_14_is_na",
"State_of_education_in_country_nowadays_is_missing",
"State_of_health_services_in_country_nowadays_is_na",
"State_of_education_in_country_nowadays_is_na",
"Age_of_respondent,_calculated_is_missing",
"How_satisfied_with_the_way_democracy_works_in_country_is_missing",
"How_satisfied_with_the_way_democracy_works_in_country_is_na",
"Signed_petition_last_12_months_is_missing",
"Signed_petition_last_12_months_is_na",
"How_close_to_party_is_missing",
"Partner_doing_last_7_days:_no_answer",
"Take_part_in_social_activities_compared_to_others_of_same_age_is_missing",
"Subjective_general_health_is_na",
"Year_of_birth_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem",
"Number_of_people_living_regularly_as_member_of_household_is_na",
"Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_missing",
"Doing_last_7_days:_housework,_looking_after_children,_others_is_na",
"Doing_last_7_days:_housework,_looking_after_children,_others_is_missing",
"Mother's_employment_status_when_respondent_14_is_missing",
"Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_na",
"Doing_last_7_days:_don't_know_is_missing",
"Doing_last_7_days:_don't_know",
"Partner_doing_last_7_days:_don't_know_is_na",
"Partner_doing_last_7_days:_don't_know_is_missing",
"Partner_doing_last_7_days:_don't_know",
"Doing_last_7_days:_no_answer_is_na",
"Doing_last_7_days:_no_answer_is_missing",
"Doing_last_7_days:_no_answer",
"Partner_doing_last_7_days:_no_answer_is_na",
"Partner_doing_last_7_days:_no_answer_is_missing",
"Partner_doing_last_7_days:_community_or_military_service_is_missing",
"Doing_last_7_days:_don't_know_is_na",
"Partner_doing_last_7_days:_community_or_military_service_is_na",
"Domicile,_respondent's_description_is_missing",
"Domicile,_respondent's_description_is_na",
"Partner_doing_last_7_days:_community_or_military_service",
"Partner_doing_last_7_days:_other_is_na",
"Doing_last_7_days:_other",
"Doing_last_7_days:_other_is_na",
"Doing_last_7_days:_other_is_missing",
"Partner_doing_last_7_days:_not_applicable_is_na",
"Father's_employment_status_when_respondent_14_is_na",
"Number_of_employees_respondent_has/had_is_missing",
"Partner_doing_last_7_days:_not_applicable_is_missing",
"Partner_doing_last_7_days:_refusal_is_missing",
"Partner_doing_last_7_days:_refusal_is_na",
"Doing_last_7_days:_refusal",
"Doing_last_7_days:_refusal_is_missing",
"Doing_last_7_days:_refusal_is_na",
"Partner_doing_last_7_days:_other",
"Partner_doing_last_7_days:_other_is_missing",
"Partner_doing_last_7_days:_refusal",
"Partner_doing_last_7_days:_permanently_sick_or_disabled_is_na",
"Partner_doing_last_7_days:_permanently_sick_or_disabled_is_missing",
"Doing_last_7_days:_education_is_na",
"Doing_last_7_days:_education_is_missing",
"Partner_doing_last_7_days:_education_is_missing",
"Partner_doing_last_7_days:_education_is_na",
"Years_of_full-time_education_completed_is_na",
"Number_of_employees_respondent_has/had_is_na",
"Partner_doing_last_7_days:_retired_is_missing",
"Partner_doing_last_7_days:_retired_is_na",
"Doing_last_7_days:_retired_is_missing",
"Doing_last_7_days:_permanently_sick_or_disabled_is_na",
"Feel_closer_to_a_particular_party_than_all_other_parties_is_missing",
"Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing",
"Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_missing",
"Doing_last_7_days:_permanently_sick_or_disabled_is_missing",
"Doing_last_7_days:_paid_work_is_na",
"Doing_last_7_days:_paid_work_is_missing",
"Partner_doing_last_7_days:_paid_work_is_na",
"Partner_doing_last_7_days:_paid_work_is_missing",
"Doing_last_7_days:_retired_is_na",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing",
"Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_na",
"Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_missing",
"Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_missing",
"Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_na",
"Contacted_politician_or_government_official_last_12_months_is_missing",
"Contacted_politician_or_government_official_last_12_months_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months",
"Citizen_of_country",
"How_satisfied_with_life_as_a_whole_is_missing",
"How_often_pray_apart_from_at_religious_services_is_na",
"Father's_employment_status_when_respondent_14",
"Year_last_in_paid_job_is_missing",
"How_happy_are_you_is_na",
"Ever_had_a_paid_job_is_na",
"Discrimination_of_respondent's_group:_disability",
"Hours_normally_worked_a_week_in_main_job_overtime_included,_partner_is_na",
"Placement_on_left_right_scale_is_missing",
"Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_missing",
"Partner_doing_last_7_days:_education",
"Ever_had_children_living_in_household_is_missing",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job",
"Discrimination_of_respondent's_group:_ethnic_group",
"How_often_socially_meet_with_friends,_relatives_or_colleagues_is_na",
"Most_people_can_be_trusted_or_you_can't_be_too_careful",
"Father's_employment_status_when_respondent_14_is_missing",
"State_of_education_in_country_nowadays",
"Years_of_full-time_education_completed_is_missing",
"How_satisfied_with_life_as_a_whole_is_na",
"Trust_in_the_legal_system",
"Trust_in_the_legal_system_is_missing",
"Discrimination_of_respondent's_group:_colour_or_race",
"Year_of_birth_of_fourth_person_in_household_is_na",
"Discrimination_of_respondent's_group:_nationality",
"Trust_in_the_United_Nations",
"Born_in_country",
"Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_missing",
"State_of_health_services_in_country_nowadays",
"Main_activity_last_7_days",
"Year_of_birth_of_fifth_person_in_household",
"Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_missing",
"Country's_cultural_life_undermined_or_enriched_by_immigrants_is_missing",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years",
"Trust_in_the_police",
"Doing_last_7_days:_paid_work",
"How_close_to_party_is_na",
"Discrimination_of_respondent's_group:_religion",
"Trust_in_politicians_is_na",
"How_satisfied_with_present_state_of_economy_in_country",
"Main_activity,_last_7_days._All_respondents._Post_coded_is_na",
"Father_born_in_country",
"How_often_attend_religious_services_apart_from_special_occasions_is_na",
"Voted_last_national_election_is_na",
"Worked_in_political_party_or_action_group_last_12_months_is_missing",
"Trust_in_the_United_Nations_is_na",
"Worked_in_political_party_or_action_group_last_12_months_is_na",
"Voted_last_national_election_is_missing",
"Discrimination_of_respondent's_group:_don't_know_is_missing",
"Discrimination_of_respondent's_group:_don't_know_is_na",
"Discrimination_of_respondent's_group:_age_is_missing",
"Discrimination_of_respondent's_group:_age_is_na",
"Feeling_of_safety_of_walking_alone_in_local_area_after_dark_is_na",
"Belong_to_minority_ethnic_group_in_country_is_na",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_na",
"Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na",
"Discrimination_of_respondent's_group:_colour_or_race_is_na",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years_is_missing",
"Citizen_of_country_is_na",
"Citizen_of_country_is_missing",
"Discrimination_of_respondent's_group:_not_applicable_is_na",
"Number_of_people_responsible_for_in_job_is_missing",
"Ever_had_a_paid_job_is_missing",
"Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_na",
"Year_last_in_paid_job_is_na",
"Number_of_people_responsible_for_in_job_is_na",
"How_interested_in_politics_is_na",
"Feel_closer_to_a_particular_party_than_all_other_parties_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months_is_missing",
"Born_in_country_is_missing",
"Born_in_country_is_na",
"Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing",
"Immigrants_make_country_worse_or_better_place_to_live_is_na",
"Belong_to_minority_ethnic_group_in_country_is_missing",
"Discrimination_of_respondent's_group:_ethnic_group_is_missing",
"Country's_cultural_life_undermined_or_enriched_by_immigrants_is_na",
"Discrimination_of_respondent's_group:_disability_is_missing",
"Discrimination_of_respondent's_group:_disability_is_na",
"Discrimination_of_respondent's_group:_don't_know",
"Discrimination_of_respondent's_group:_ethnic_group_is_na",
"Gays_and_lesbians_free_to_live_life_as_they_wish_is_na",
"Government_should_reduce_differences_in_income_levels_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months_is_na",
"Boycotted_certain_products_last_12_months_is_missing",
"Boycotted_certain_products_last_12_months_is_na",
"Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves",
"Worn_or_displayed_campaign_badge/sticker_last_12_months_is_missing",
"Worn_or_displayed_campaign_badge/sticker_last_12_months_is_na",
"Responsible_for_supervising_other_employees_is_na",
"Responsible_for_supervising_other_employees_is_missing",
"Responsible_for_supervising_other_employees",
"Main_activity_last_7_days_is_missing",
"Discrimination_of_respondent's_group:_colour_or_race_is_missing",
"Discrimination_of_respondent's_group:_refusal",
"Discrimination_of_respondent's_group:_refusal_is_missing",
"Discrimination_of_respondent's_group:_refusal_is_na",
"Discrimination_of_respondent's_group:_religion_is_missing",
"Discrimination_of_respondent's_group:_sexuality_is_na",
"Discrimination_of_respondent's_group:_sexuality_is_missing",
"Discrimination_of_respondent's_group:_religion_is_na",
"Discrimination_of_respondent's_group:_sexuality",
"Member_of_a_group_discriminated_against_in_this_country_is_na",
"Father_born_in_country_is_missing",
"Father_born_in_country_is_na",
"Discrimination_of_respondent's_group:_not_applicable_is_missing",
"Discrimination_of_respondent's_group:_no_answer_is_missing",
"Discrimination_of_respondent's_group:_no_answer",
"Member_of_a_group_discriminated_against_in_this_country_is_missing",
"Discrimination_of_respondent's_group:_language_is_missing",
"Discrimination_of_respondent's_group:_language_is_na",
"Discrimination_of_respondent's_group:_language",
"Discrimination_of_respondent's_group:_no_answer_is_na",
"Discrimination_of_respondent's_group:_nationality_is_na",
"Discrimination_of_respondent's_group:_not_applicable",
"Discrimination_of_respondent's_group:_nationality_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_missing",
"How_often_attend_religious_services_apart_from_special_occasions_is_missing",
"Take_part_in_social_activities_compared_to_others_of_same_age_is_na",
"Mother_born_in_country_is_na",
"Mother_born_in_country_is_missing",
"Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_na",
"Doing_last_7_days:_community_or_military_service_is_na",
"Doing_last_7_days:_community_or_military_service_is_missing",
"Number_of_people_living_regularly_as_member_of_household_is_missing",
"Year_of_birth_is_na",
"Subjective_general_health_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_na",
"Year_of_birth_of_sixth_person_in_household_is_missing",
"Year_of_birth_of_fifth_person_in_household_is_na",
"Age_of_respondent,_calculated_is_na",
"Year_of_birth_of_sixth_person_in_household",
"Mother's_employment_status_when_respondent_14_is_na",
"State_of_education_in_country_nowadays_is_missing",
"State_of_health_services_in_country_nowadays_is_na",
"State_of_education_in_country_nowadays_is_na",
"Age_of_respondent,_calculated_is_missing",
"How_satisfied_with_the_way_democracy_works_in_country_is_missing",
"How_satisfied_with_the_way_democracy_works_in_country_is_na",
"Signed_petition_last_12_months_is_missing",
"Signed_petition_last_12_months_is_na",
"How_close_to_party_is_missing",
"Partner_doing_last_7_days:_no_answer",
"Take_part_in_social_activities_compared_to_others_of_same_age_is_missing",
"Subjective_general_health_is_na",
"Year_of_birth_is_missing",
"Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem",
"Number_of_people_living_regularly_as_member_of_household_is_na",
"Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_missing",
"Doing_last_7_days:_housework,_looking_after_children,_others_is_na",
"Doing_last_7_days:_housework,_looking_after_children,_others_is_missing",
"Mother's_employment_status_when_respondent_14_is_missing",
"Partner_doing_last_7_days:_housework,_looking_after_children,_others_is_na",
"Doing_last_7_days:_don't_know_is_missing",
"Doing_last_7_days:_don't_know",
"Partner_doing_last_7_days:_don't_know_is_na",
"Partner_doing_last_7_days:_don't_know_is_missing",
"Partner_doing_last_7_days:_don't_know",
"Doing_last_7_days:_no_answer_is_na",
"Doing_last_7_days:_no_answer_is_missing",
"Doing_last_7_days:_no_answer",
"Partner_doing_last_7_days:_no_answer_is_na",
"Partner_doing_last_7_days:_no_answer_is_missing",
"Partner_doing_last_7_days:_community_or_military_service_is_missing",
"Doing_last_7_days:_don't_know_is_na",
"Partner_doing_last_7_days:_community_or_military_service_is_na",
"Domicile,_respondent's_description_is_missing",
"Domicile,_respondent's_description_is_na",
"Partner_doing_last_7_days:_community_or_military_service",
"Partner_doing_last_7_days:_other_is_na",
"Doing_last_7_days:_other",
"Doing_last_7_days:_other_is_na",
"Doing_last_7_days:_other_is_missing",
"Partner_doing_last_7_days:_not_applicable_is_na",
"Father's_employment_status_when_respondent_14_is_na",
"Number_of_employees_respondent_has/had_is_missing",
"Partner_doing_last_7_days:_not_applicable_is_missing",
"Partner_doing_last_7_days:_refusal_is_missing",
"Partner_doing_last_7_days:_refusal_is_na",
"Doing_last_7_days:_refusal",
"Doing_last_7_days:_refusal_is_missing",
"Doing_last_7_days:_refusal_is_na",
"Partner_doing_last_7_days:_other",
"Partner_doing_last_7_days:_other_is_missing",
"Partner_doing_last_7_days:_refusal",
"Partner_doing_last_7_days:_permanently_sick_or_disabled_is_na",
"Partner_doing_last_7_days:_permanently_sick_or_disabled_is_missing",
"Doing_last_7_days:_education_is_na",
"Doing_last_7_days:_education_is_missing",
"Partner_doing_last_7_days:_education_is_missing",
"Partner_doing_last_7_days:_education_is_na",
"Years_of_full-time_education_completed_is_na",
"Number_of_employees_respondent_has/had_is_na",
"Partner_doing_last_7_days:_retired_is_missing",
"Partner_doing_last_7_days:_retired_is_na",
"Doing_last_7_days:_retired_is_missing",
"Doing_last_7_days:_permanently_sick_or_disabled_is_na",
"Feel_closer_to_a_particular_party_than_all_other_parties_is_missing",
"Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_missing",
"Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_missing",
"Doing_last_7_days:_permanently_sick_or_disabled_is_missing",
"Doing_last_7_days:_paid_work_is_na",
"Doing_last_7_days:_paid_work_is_missing",
"Partner_doing_last_7_days:_paid_work_is_na",
"Partner_doing_last_7_days:_paid_work_is_missing",
"Doing_last_7_days:_retired_is_na",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_not_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_actively_looking_for_job_is_na",
"Doing_last_7_days:_unemployed,_actively_looking_for_job_is_missing",
"Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_na",
"Any_period_of_unemployment_and_work_seeking_within_last_5_years_is_missing",
"Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_missing",
"Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months_is_na",
"Contacted_politician_or_government_official_last_12_months_is_missing",
"Contacted_politician_or_government_official_last_12_months_is_na",
"Taken_part_in_lawful_public_demonstration_last_12_months",
"Citizen_of_country",
"How_satisfied_with_life_as_a_whole_is_missing",
"How_often_pray_apart_from_at_religious_services_is_na",
"Father's_employment_status_when_respondent_14",
"Year_last_in_paid_job_is_missing",
"How_happy_are_you_is_na",
"Ever_had_a_paid_job_is_na",
"Discrimination_of_respondent's_group:_disability",
"Hours_normally_worked_a_week_in_main_job_overtime_included,_partner_is_na",
"Placement_on_left_right_scale_is_missing",
"Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_missing",
"Partner_doing_last_7_days:_education",
"Ever_had_children_living_in_household_is_missing",
"Partner_doing_last_7_days:_unemployed,_actively_looking_for_job",
"Discrimination_of_respondent's_group:_ethnic_group",
"How_often_socially_meet_with_friends,_relatives_or_colleagues_is_na",
"Most_people_can_be_trusted_or_you_can't_be_too_careful",
"Father's_employment_status_when_respondent_14_is_missing",
"State_of_education_in_country_nowadays",
"Years_of_full-time_education_completed_is_missing",
"How_satisfied_with_life_as_a_whole_is_na",
"Trust_in_the_legal_system",
"Trust_in_the_legal_system_is_missing",
"Discrimination_of_respondent's_group:_colour_or_race",
"Year_of_birth_of_fourth_person_in_household_is_na",
"Discrimination_of_respondent's_group:_nationality",
"Trust_in_the_United_Nations",
"Born_in_country",
"Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_missing",
"State_of_health_services_in_country_nowadays",
"Main_activity_last_7_days",
"Year_of_birth_of_fifth_person_in_household",
"Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_missing",
"Country's_cultural_life_undermined_or_enriched_by_immigrants_is_missing",
"Respondent_or_household_member_victim_of_burglary/assault_last_5_years",
"Trust_in_the_police",
"Doing_last_7_days:_paid_work",
"How_close_to_party_is_na",
"Discrimination_of_respondent's_group:_religion",
"Trust_in_politicians_is_na",
"How_satisfied_with_present_state_of_economy_in_country",
"Main_activity,_last_7_days._All_respondents._Post_coded_is_na",
"Father_born_in_country",
"How_often_attend_religious_services_apart_from_special_occasions_is_na",


]

# Drop the specified columns from df_subset, ignoring errors if some columns don't exist
df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns], errors='ignore', inplace=True)



#%%
# Save the processed dataset for future use
output_csv = "ess_processed_two_missingness_indicators.csv"
df_processed.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_processed.shape} to '{output_csv}'")


#%%
# Count and print total number of respondents in the processed dataset
num_respondents = len(df_processed)
print(f"📊 Total number of respondents in the processed dataset: {num_respondents:,}")


#%%
# =============================================================================
# STEP 5: COUNT AND REPORT NUMBER OF FEATURES (EXCLUDING METADATA & TARGET)
# =============================================================================

# Define columns to exclude from feature count:
# - 'Country', 'ESS_round', 'Gender': metadata identifiers
# - 'target_encoded': the model target variable
# - All '_is_na' and '_is_missing' columns: missingness indicators, not features themselves
exclude_cols = ['Country', 'ESS_round', 'Gender', 'target_encoded']
feature_cols = [col for col in df_processed.columns
                if col not in exclude_cols
                and not (col.endswith('_is_na') or col.endswith('_is_missing'))]

num_features = len(feature_cols)
print(f"🔢 Number of features used for modeling: {num_features:,}")
print(f"   Feature columns include: {', '.join(feature_cols[:5])}{'...' if num_features > 5 else ''}")

# Optional: Save feature list to file for reproducibility
feature_list_path = "ess_feature_list.txt"
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"📋 Feature list saved to '{feature_list_path}'")


#%%
# ==================================================================================================
# STEP 5: BALANCED SAMPLING & TRAIN/TEST SPLIT
# ==================================================================================================

print("Step 5: Performing balanced stratified train/test split...")

# Count the number of observations in each country/round combination
counts = df_processed.groupby(['Country', 'ESS_round']).size()
counts

#%%
# Calculate training size as 80% of the smallest group (ensures balance across all strata)
train_n = int(counts.min() * 0.8)
print("Number of rows from each round/country combination:", train_n)
# For each country/round group, randomly sample train_n observations (or all if fewer than train_n)
training_data = df_processed.groupby(['Country', 'ESS_round'], group_keys=False).apply(
    lambda x: x.sample(n=min(train_n, len(x)), random_state=42), include_groups=False
)

# Test data is everything not in the training set
test_data = df_processed.drop(index=training_data.index).copy()

# Define feature columns: all columns except identifiers and target. We want the model to train on the data without know where any particular respondant was from.
# Then when we validate the model, we can see the performance differences of the model on the different countries.
# The model will learn how to tell the difference between Male and female and then validate it also without country/round and then
# we can see which countries the model struggled with more by reattaching the country/round lables, and use that as a gender similiarity metric for that country
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'Gender', 'target_encoded']]

# Create training and test sets for modeling
X_train = training_data[feature_cols]  # Feature matrix (train)
y_train = training_data['target_encoded']  # Target vector (train)
X_test = test_data[feature_cols]       # Feature matrix (test)
y_test = test_data['target_encoded']   # Target vector (test)

#%%
# ==================================================================================================
# STEP 6: MODEL TRAINING (HIST-GRADIENT BOOSTING)
# ==================================================================================================
'''
We chose HGB Classifier because:
Although we didn't ultimatley did not include and NaNs in the training data, HistGradientBoostingClassifier natively handles missing values by evaluating during tree construction whether NaN observations should go to the left or right child node based on which path yields the maximum reduction in loss.
2. HGB discretizes continuous numerical features into integer-valued bins (typically up to 256 bins). This reduces the complexity of finding tree splits from $O(n \log n)$ per feature down to $O(n_{\text{bins}})$, making model fitting extremely fast even across hundreds of thousands of European Social Survey (ESS) rows and dozens of features.
3. While external libraries like LightGBM and XGBoost offer similar histogram-based gradient boosting algorithms:HGB is fully built into sklearn.ensemble, eliminating external C++/OpenMP system dependencies or third-party binary package installations across execution environments.It interfaces seamlessly with standard scikit-learn tools such as LabelEncoder, permutation_importance, and Pipeline.
4. We needed to use the same model for all of our different trials with missingness and feature changes any observed differences in accuracy trends or feature importances are strictly attributable to those changes. It works well and fast out of the box and was therefore ideal for running many trials on enourmous datasets to identify the features and handling without having to worry about setting parameters
5. The model utilizes C-level multithreading (via OpenMP) natively during fit time, delivering near-instantaneous execution speeds across large tabular datasets without requiring explicit hyperparameter tuning or custom parallel backends.

'''
print("Step 6: Fitting HistGradientBoostingClassifier...")

# Initialize the classifier with a fixed random state for reproducibility
hgb_model = HistGradientBoostingClassifier(random_state=42)

# Train the model on the training data
hgb_model.fit(X_train, y_train)

# Make predictions on test set
test_data['y_pred'] = hgb_model.predict(X_test)          # Predicted class labels
test_data['y_prob'] = hgb_model.predict_proba(X_test)[:, 1]  # Probability of positive class (Female=2)

def eval_group(g):
    """
    Evaluate model performance for a single country/round group.
    
    Args:
        g: DataFrame subset containing predictions and true labels for one group
    
    Returns:
        pd.Series with test sample size, accuracy, and F1 score
    """
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})


# Evaluate model performance for each country/round combination
perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()

# Save performance metrics to CSV
perf_df.to_csv("ess_country_and_round_performance_indicators.csv", index=False)

# Fit a mixed-effects linear model to analyze accuracy trends over time
# - Fixed effect: ESS_round (survey wave)
# - Random effect: Country (accounting for country-specific variation)
mixed_res = smf.mixedlm("accuracy ~ ESS_round", perf_df, groups=perf_df["Country"]).fit()

# Print the full statistical summary of the mixed model
print("\n==================================================")
print("     GLOBAL ACCURACY TREND OVER TIME (MIXED LM)   ")
print("==================================================")
print(mixed_res.summary())

#%%
# ==================================================================================================
# STEP 7: ENHANCED VISUALIZATIONS (FAST FEATURE IMPORTANCE)
# ==================================================================================================

print("\nStep 7: Generating plots...")


# ------------------------------------------------------------------------------
# PLOT 1: ACCURACY TREND WITH 5-GROUP CATEGORY SELECTOR
# ------------------------------------------------------------------------------

# Ensure ESS_round is treated as integer for plotting
perf_df['ESS_round'] = perf_df['ESS_round'].astype(int)
#%%

def calculate_slope(group):
    """
    Calculate the linear trend slope of accuracy over ESS rounds for a country.
    
    Args:
        group: DataFrame subset with one country's performance across rounds
    
    Returns:
        float: Slope coefficient from OLS regression
               Positive = accuracy increasing, Negative = decreasing
    """
    if len(group['ESS_round'].unique()) < 2:
        return 0.0  # Need at least 2 points to calculate slope
    
    try:
        # Add intercept term for OLS
        X = sm.add_constant(group['ESS_round'])
        
        # Fit ordinary least squares regression
        model = sm.OLS(group['accuracy'], X).fit()
        
        # Return the slope coefficient (ESS_round parameter)
        return model.params['ESS_round'] if 'ESS_round' in model.params else 0.0
    except Exception:
        return 0.0


# Calculate slope for each country to determine performance trajectory
country_slopes = perf_df.groupby('Country').apply(calculate_slope, include_groups=False).reset_index(name='slope')

# Define five categories based on the distribution of slopes
groups_order = [
    'Steepest Decrease (--)',      # Countries with most negative slope (growing gender similarity)
    'Slight Decrease (-)',         # Moderately negative slope
    'Neutral / Stable (0)',        # Near-zero slope (stable gender differences)
    'Slight Increase (+)',         # Slightly positive slope
    'Steepest Increase (++)'       # Most positive slope (growing gender divergence)
]

# Bin countries into groups based on their slope values using quantiles
slope_bins = pd.qcut(country_slopes['slope'], q=5, labels=groups_order)
country_slopes['group'] = slope_bins

# Merge country group assignments back to performance DataFrame
perf_grouped = perf_df.merge(country_slopes[['Country', 'group']], on='Country')


# Create the interactive Plotly figure for accuracy trends
fig_trend = go.Figure()

# Use a qualitative color palette for distinct countries
palette = px.colors.qualitative.Bold

# Dictionary to track which traces belong to each group (for toggling visibility)
group_trace_indices = {}
current_trace_idx = 0

# Add traces for each group in order
for grp in groups_order:
    # Get all countries assigned to this slope category
    sub_df = perf_grouped[perf_grouped['group'] == grp]
    countries_in_grp = sub_df['Country'].unique()
    
    start_idx = current_trace_idx  # Track starting index for this group
    
    # Add a line plot for each country in this group
    for i, cntry in enumerate(countries_in_grp):
        # Filter and sort data for this country
        cntry_df = sub_df[sub_df['Country'] == cntry].sort_values('ESS_round')
        
        # Use colors from palette (cycling if more countries than colors)
        color = palette[i % len(palette)]
        
        # Add scatter plot trace for this country's accuracy trend
        fig_trend.add_trace(go.Scatter(
            x=cntry_df['ESS_round'],
            y=cntry_df['accuracy'],
            mode='lines+markers',
            name=cntry,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>Country:</b> {cntry}<br><b>Round:</b> %{{x}}<br><b>Accuracy:</b> %{{y:.2%}}<extra></extra>",
            # Only first group (Steepest Decrease) is visible initially
            visible=(grp == 'Steepest Decrease (--)')
        ))
        
        current_trace_idx += 1
    
    # Record which traces belong to this group for toggling
    end_idx = current_trace_idx
    group_trace_indices[grp] = list(range(start_idx, end_idx))

# Store total number of traces created
total_traces = current_trace_idx

# Create dropdown menu buttons for each group
buttons = []
for grp in groups_order:
    # Initialize visibility mask (all False)
    visible_mask = [False] * total_traces
    
    # Set only traces belonging to this group to True
    for idx in group_trace_indices[grp]:
        visible_mask[idx] = True
        
    # Create button configuration
    buttons.append(dict(
        label=grp,
        method="update",
        args=[
            {"visible": visible_mask},
            {
                "title": f"<b>Gender Predictability Trends (Indicator Flags): Group {grp}</b><br><sup>Lower Accuracy = Higher Gender Similarity</sup>"
            }
        ]
    ))

# Update figure layout with interactive controls
fig_trend.update_layout(
    updatemenus=[dict(
        active=0,  # First group visible by default
        buttons=buttons,
        direction="down",      # Dropdown menu opens downward
        pad={"r": 10, "t": 10},  # Padding around dropdown
        showactive=True,       # Show currently selected option
        x=0.0,                 # Horizontal position (left)
        xanchor="left",
        y=1.22,                # Vertical position (above plot)
        yanchor="top"
    )],
    title="<b>Gender Predictability Trends (Indicator Flags): Group Steepest Decrease (--)</b><br><sup>Lower Accuracy = Higher Gender Similarity</sup>",
    xaxis=dict(title="ESS Survey Round", dtick=1, tickmode='linear'),
    yaxis=dict(title="Model Accuracy", tickformat=".0%", range=[0.40, 0.95]),
    height=650,
    width=1100,
    hovermode="x unified",
    margin=dict(l=80, r=120, t=120, b=80)
)


# Add horizontal reference line at 50% accuracy (random guessing threshold)
fig_trend.add_hline(y=0.50, line_dash="dash", line_color="gray", 
                    annotation_text="Random Guess (50%)", 
                    annotation_position="bottom right")
#%%
# Save the interactive plot as HTML
fig_trend.write_html("attempt5_gender_similarity_trends_indicators.html", include_plotlyjs='cdn')
print("✅ Grouped trend plot saved to 'gender_similarity_trends_indicators.html'")
#%%

# ------------------------------------------------------------------------------
# PLOT 2: FEATURE IMPORTANCE PLOT (FAST EVALUATION ON SAMPLED TEST SET)
# ------------------------------------------------------------------------------

print("Calculating fast permutation feature importances...")

# Sample up to 10,000 test rows for faster importance calculation
# This reduces computation time while maintaining reasonable accuracy
X_eval = X_test.sample(n=min(10000, len(X_test)), random_state=42)
y_eval = y_test.loc[X_eval.index]

# Compute permutation-based feature importance:
# - Randomly permute each feature and measure accuracy drop
# - Larger drop = more important feature
# - n_repeats=2: Repeat permutation twice for stability
# - n_jobs=1: Single-threaded to avoid potential lock issues
perm_imp = permutation_importance(
    hgb_model, 
    X_eval, 
    y_eval, 
    n_repeats=2, 
    random_state=42, 
    n_jobs=1  
)

# Create DataFrame of feature importances (sorted by importance descending)
imp_df = pd.DataFrame({
    'Feature': feature_cols, 
    'Importance': perm_imp.importances_mean
}).sort_values('Importance', ascending=False)


def get_feat_type(name):
    """
    Categorize features into types based on naming convention.
    
    Args:
        name: Feature column name
    
    Returns:
        str: Category label for the feature
    """
    if name.endswith('_is_na'): 
        return 'Not Applicable Flag'
    if name.endswith('_is_missing'): 
        return 'Other Missing Flag'
    return 'Base Survey Question'


# Add category labels to each feature
imp_df['Category'] = imp_df['Feature'].apply(get_feat_type)

# Get top 20 most important features, sorted ascending for horizontal bar plot
# top_20_df = imp_df.head(20).sort_values('Importance', ascending=True)
top_20_df = imp_df.sort_values('Importance', ascending=True)


# Create horizontal bar chart of top 20 features
fig_imp = px.bar(
    top_20_df, 
    x='Importance',          # X-axis: importance value
    y='Feature',             # Y-axis: feature name
    color='Category',        # Color by feature type
    orientation='h',         # Horizontal bars
    title="<b>Top 20 Features (Indicator Flags & Native Tree Splits)</b>",
    color_discrete_map={
        'Base Survey Question': '#1f77b4',     # Blue
        'Not Applicable Flag': '#ff7f0e',      # Orange
        'Other Missing Flag': '#d62728'        # Red
    }
)

# Configure Y-axis for proper category display
fig_imp.update_yaxes(
    type='category',
    autorange='reversed',    # Top feature at top of plot
    tickmode='linear',
    dtick=1,
    automargin=True          # Auto-adjust margins to fit labels
)

# Update layout with custom dimensions and margins
fig_imp.update_layout(
    height=700,              # Taller for 20 features
    width=1150,              # Wider for label space
    margin=dict(l=350, r=50, t=100, b=50),  # Large left margin for feature names
    legend=dict(title="Feature Category", y=0.1, x=0.7)  # Legend position
)
#%%

# Save the feature importance plot as HTML
fig_imp.write_html("attempt5_total_feature_importance_indicators.html", include_plotlyjs='cdn')
print("✅ Feature importance plot saved to 'feature_importance_indicators.html'")
print("\n🎉 Execution complete!")
# %%
# Get features with negative permutation importance (i.e., accuracy *increased* when permuted)
negative_imp_features = imp_df[imp_df['Importance'] < 0]['Feature'].tolist()

# Print the list of features with negative importance
print("\nFeatures with Negative Permutation Importance:")
for feat in negative_imp_features:
    print(f"'{feat}',")

# Optionally, save them to a file for later reference
with open("negative_importance_features.txt", "w") as f:
    f.write("Features with Negative Permutation Importance:\n")
    for feat in negative_imp_features:
        f.write(f"- {feat}\n")

print("\n✅ List of features with negative importance saved to 'negative_importance_features.txt'")
# %%
# Get features with permutation feature importance greater than 0
positive_imp_features = imp_df[imp_df['Importance'] > 0]['Feature'].tolist()

# Print the list of features with positive importance
print("\nFeatures with Positive Permutation Importance:")
for feat in positive_imp_features:
    print(f"'{feat}',")

# Optionally, save them to a file for later reference
with open("positive_importance_features.txt", "w") as f:
    f.write("Features with Positive Permutation Importance:\n")
    for feat in positive_imp_features:
        f.write(f"- {feat}\n")

print("\n✅ List of features with positive importance saved to 'positive_importance_features.txt'")
# %%
