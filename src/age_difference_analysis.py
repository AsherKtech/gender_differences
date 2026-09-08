#%%
import time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pyreadstat

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.inspection import permutation_importance

import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
from pyprojroot import here

import group_valid


#%%
# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================
def compute_permutation_importance_df(model, X, y, n_repeats=5, random_state=42):
    """
    Runs permutation importance, categorizes features, and returns an unsorted DataFrame.
    """
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    
    df = pd.DataFrame({'Feature': X.columns.tolist(), 'Importance': perm.importances_mean})
    
    df['Category'] = df['Feature'].apply(
        lambda name: 'Not Applicable Flag' if name.endswith('_is_na') 
        else ('Other Missing Flag' if name.endswith('_is_missing') 
        else 'Base Survey Question')
    )
    return df


def plot_and_export_feature_importance(importance_df, output_csv=None, output_html=None):
    """
    Takes a pre-computed importance DataFrame, exports a sorted CSV of all features, 
    and saves an HTML bar plot displaying only the top 20 features sorted strictly by importance.
    """
    if output_csv is None:
        output_csv = here('data/processed/features_importances_sorted.csv')
    if output_html is None:
        output_html = here('plots/feature_importance_plot.html')
    
    # Sort full dataset descending for CSV output
    imp_df_sorted = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

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

    # Force strict ordering on the y-axis by feature importance values
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
    fig_imp.write_html(output_html, include_plotlyjs='cdn')

    # Export all features to CSV
    imp_df_sorted[['Feature', 'Importance', 'Category']].to_csv(output_csv, index=False)
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


def eval_group(g):
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})


def calculate_slope(group):
    if len(group['ESS_round'].unique()) < 2:
        return 0.0
    try:
        X = sm.add_constant(group['ESS_round'])
        model = sm.OLS(group['accuracy'], X).fit()
        return model.params['ESS_round'] if 'ESS_round' in model.params else 0.0
    except Exception:
        return 0.0

def print_age_distribution(series, label_for_code=None, code_word="code"):
    """
    Prints per-value row counts for an age/target series, handling both string 
    labels and numeric codes gracefully.
    """
    counts = series.value_counts().sort_index()
    for code, count in counts.items():
        # Handle string labels vs numeric codes safely
        if isinstance(code, (int, float, np.number)):
            code_val = int(code)
            display_label = label_for_code(code_val) if label_for_code else code_val
            print(f"{display_label} ({code_word} {code_val}): {count} rows")
        else:
            display_label = label_for_code(code) if label_for_code else code
            print(f"{display_label}: {count} rows")



def subsample_xy(X, y, n_max=10000, random_state=42):
    """
    Subsamples up to n_max rows from X (and the matching rows of y) for cheaper
    permutation-importance evaluation. Used both for feature selection and final export.
    """
    X_sub = X.sample(n=min(n_max, len(X)), random_state=random_state)
    y_sub = y.loc[X_sub.index]
    return X_sub, y_sub


#%%
# ==================================================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA AND FILTER BY AGE GROUPS
# ==================================================================================================
print("Step 1: Loading raw SPSS file and metadata...")# Step 1: Loading raw SPSS file and metadata...

path = here('data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(str(path), user_missing=True)

print(f"Number of unique countries: {df_raw['cntry'].nunique()}")
# Number of unique countries: 38


#%%
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}
cntry_val_labels = meta.variable_value_labels.get('cntry', {})

age_raw_col = next((c for c in ['agea', 'age'] if c in df_raw.columns), 'agea')
df_raw['age_numeric'] = pd.to_numeric(df_raw[age_raw_col], errors='coerce')


#%%

# --- A. Check Non-NaN Counts and Raw Value Counts ---
print(f"Total non-NaN rows in 'age': {df_raw['age'].notna().sum()}")
print(f"Total rows (including NaN): {len(df_raw)}")

# --- B. Clean 'age': Replace missing codes (999, 999.0, etc.) with NaN ---
# Common ESS missing codes for age include 777 (Refusal), 888 (Don't know), 999 (Not available)
missing_codes = [777, 888, 999]
df_raw['age_clean'] = df_raw['age'].replace(missing_codes, np.nan)

# Non-NaN count after removing 999 codes
valid_age_count = df_raw['age_clean'].notna().sum()
print(f"Total valid non-NaN age rows (excluding 999s): {valid_age_count}")

print("\n=== Age Distribution (Floor/Rounded Integers) ===")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_raw['age_clean'].dropna().astype(int).value_counts().sort_index())




#%%
# --- A. Check Non-NaN Counts and Raw Value Counts ---
print(f"Total non-NaN rows in 'agea': {df_raw['agea'].notna().sum()}")
print(f"Total rows (including NaN): {len(df_raw)}")

# --- B. Clean 'age': Replace missing codes (999, 999.0, etc.) with NaN ---
# Common ESS missing codes for age include 777 (Refusal), 888 (Don't know), 999 (Not available)
missing_codes = [777, 888, 999]
df_raw['agea_clean'] = df_raw['agea'].replace(missing_codes, np.nan)

# Non-NaN count after removing 999 codes
valid_age_count = df_raw['agea_clean'].notna().sum()
print(f"Total valid non-NaN age rows (excluding 999s): {valid_age_count}")


print("\n=== Age Distribution (Floor/Rounded Integers) ===")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_raw['agea_clean'].dropna().astype(int).value_counts().sort_index())

#%%
# --- A. Check Non-NaN Counts and Raw Value Counts ---
print(f"Total non-NaN rows in 'age_numeric': {df_raw['age_numeric'].notna().sum()}")
print(f"Total rows (including NaN): {len(df_raw)}")

# --- B. Clean 'age': Replace missing codes (999, 999.0, etc.) with NaN ---
# Common ESS missing codes for age include 777 (Refusal), 888 (Don't know), 999 (Not available)
missing_codes = [777, 888, 999]
df_raw['age_numeric_clean'] = df_raw['age_numeric'].replace(missing_codes, np.nan)

# Non-NaN count after removing 999 codes
valid_age_count = df_raw['age_numeric_clean'].notna().sum()
print(f"Total valid non-NaN age rows (excluding 999s): {valid_age_count}")


print("\n=== Age Distribution (Floor/Rounded Integers) ===")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_raw['age_numeric_clean'].dropna().astype(int).value_counts().sort_index())

#%%
# ==================================================================================================
# STEP 1: SET MANUAL AGE BRACKETS FOR FULL PIPELINE TESTING
# ==================================================================================================
START_AGE = 37  # Change to test: 35, 36, 37, 38, 39, 40, 41, etc.
END_AGE   = 51  # Match with table above (e.g., 35-48, 36-50, 37-51, 41-56)

print(f"Step 1: Setting test age brackets [18-34] vs [{START_AGE}-{END_AGE}]...")
#Step 1: Setting test age brackets [18-34] vs [37-51]...
conditions = [
    (df_raw['age_numeric'] >= 18) & (df_raw['age_numeric'] <= 34),
    (df_raw['age_numeric'] >= START_AGE) & (df_raw['age_numeric'] <= END_AGE)
]
choices = [
    'Young (18-34)',
    f'Older ({START_AGE}-{END_AGE})'
]

df_raw['AgeGroup'] = np.select(conditions, choices, default=None)
df_sub = df_raw[df_raw['AgeGroup'].notna()].copy()


#%%


print("\n" + "="*84) # creates a division line via a bunch of equal signs

print("Age Distribution Before Preprocessing Steps (Steps 2–5)")
print("="*84)
# Print Age group distribution using the helper function
print(f"\nSample distribution for [{START_AGE}-{END_AGE}] before preprocessing:")
print_age_distribution(df_sub['AgeGroup'], lambda x: x)
print("\n✅ Age distribution check completed.")
# ====================================================================================
# Sample distribution for [37-51] before preprocessing:
# Older (37-51): 108929 rows
# Young (18-34): 103537 rows





#%%
# ==================================================================================================  
# STEP 2: FIND COLUMNS WITH COVERAGE ACROSS ALL COUNTRY/ROUND COMBOS 
# ==================================================================================================
print("Step 2: Identifying columns present across all Country/Round pairs...")  
ALL_MISSING_CODES = {
    6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666',
    7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777',
    8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888',
    9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'
}

group_cols = ['cntry', 'essround']
candidate_cols = [c for c in df_sub.columns if c not in group_cols + ['AgeGroup', 'age_numeric']]

metadata_leak = ['name', 'edition', 'proddate']
candidate_cols = [c for c in candidate_cols if c not in metadata_leak]

dtypes = df_sub[candidate_cols].dtypes
string_cols = dtypes[~dtypes.apply(pd.api.types.is_numeric_dtype)].index.tolist()  # # 2340 numeric candidate cols, 23 string candidate cols
numeric_cols = [c for c in candidate_cols if c not in string_cols]

print(f"{len(numeric_cols)} numeric candidate cols, {len(string_cols)} string candidate cols") 
 # 2340 numeric candidate cols, 23 string candidate cols


#%%
# ---- Validity check: numeric columns via C++ ----
t0 = time.time()

group_key = pd.MultiIndex.from_arrays([df_sub[group_cols[0]], df_sub[group_cols[1]]])
group_ids, group_labels = pd.factorize(group_key)

values = df_sub[numeric_cols].to_numpy(dtype=np.float64)
missing_arr = np.array(
    sorted({float(c) for c in ALL_MISSING_CODES if not isinstance(c, str)}),
    dtype=np.float64
)

result = group_valid.group_any_valid(
    values, group_ids.astype(np.int64), missing_arr, len(group_labels) 
)
valid_per_group_numeric = pd.DataFrame(result, columns=numeric_cols, index=group_labels)
print(time.time() - t0) 
#1.4523115158081055


#%%
# ---- Validity check: string columns via pandas ----
string_missing_codes = {str(c) for c in ALL_MISSING_CODES} | {'', ' '}

valid_mask_str = df_sub[string_cols].apply(is_strictly_valid_str)
valid_per_group_str = valid_mask_str.groupby(
    [df_sub[group_cols[0]], df_sub[group_cols[1]]]  
).any()
valid_per_group_str = valid_per_group_str.reindex(group_labels)
print(f"Validation completed in {time.time() - t0:.2f} seconds")  
# Validation completed in 1.98 seconds



#%%
# ---- Combine and determine retained columns ----
valid_per_group = pd.concat([valid_per_group_numeric, valid_per_group_str], axis=1)
retained_cols = valid_per_group.columns[valid_per_group.all()].tolist()




 
#%%
# ================================================================================================== 
# STEP 3: BUILD SUBSET, RENAME & STANDARDIZE METADATA HEADERS
# ==================================================================================================
print("Step 3: Renaming headers and dropping excluded metadata, structural proxies, and age leakage...")  

df_subset = df_sub[group_cols + ['AgeGroup'] + retained_cols].copy()

# Apply metadata column name mapping
df_subset.rename(columns=code_to_label, inplace=True)

cntry_renamed = code_to_label.get(group_cols[0], group_cols[0])
round_renamed = code_to_label.get(group_cols[1], group_cols[1])

df_subset['Country'] = df_subset[cntry_renamed].astype(str).str.strip().map(cntry_val_labels).fillna(df_subset[cntry_renamed])
df_subset['ESS_round'] = df_subset[round_renamed]

# Explicit metadata, leakage, and structural life-stage proxy drop list
cols_to_drop = [
    group_cols[0], group_cols[1], age_raw_col, "Title_of_dataset", "Edition", "Production_date",
    "Respondent's_identification_number", "Design_weight", "Post-stratification_weight_including_design_weight",
    "Population_size_weight_(must_be_combined_with_dweight_or_pspwght)", "Country_of_birth",
    "Country_of_birth,_father", "Language_most_often_spoken_at_home:_first_mentioned", "Country_of_birth,_mother",
    "nan_count", "Citizenship", "Language_most_often_spoken_at_home:_second_mentioned", "Region",
    "Year_of_birth", "Year_of_birth_of_respondent", "Age_of_respondent,_calculated", "Year_last_in_paid_job",
    "Ever_had_children_living_in_household",
    "Year_of_birth_of_second_person_in_household", "Year_of_birth_of_third_person_in_household",
    "Year_of_birth_of_fourth_person_in_household", "Year_of_birth_of_fifth_person_in_household"
]

# Dynamic pattern match for age/year, retirement status, children, and education level proxies
leakage_keywords = ['birth', 'yrbrn', 'year_last', 'retir', 'pension', 'child', 'edulv', 'eisced']

dynamic_leakage = [
    c for c in df_subset.columns 
    if any(kw in c.lower() for kw in leakage_keywords) 
    and c not in ['Production_date']
]

all_drops = list(set(cols_to_drop + dynamic_leakage))
existing_drops = [c for c in all_drops if c in df_subset.columns]  # # Dropped 17 leakage/proxy/metadata columns: ['Population_size_weight_(must_be_combined_with_dweight_or_pspwght)', 'Age_of_respondent,_calculated', 'Year_of_birth_of_sixth_person_in_household', 'Year_of_birth_of_fourth_person_in_household', 'Design_weight', 'Partner_doing_last_7_days:_housework,_looking_after_children,_others', 'Doing_last_7_days:_housework,_looking_after_children,_others', "Respondent's_identification_number", 'Partner_doing_last_7_days:_retired', 'Post-stratification_weight_including_design_weight', 'Year_of_birth_of_second_person_in_household', 'Year_of_birth', 'Year_of_birth_of_third_person_in_household', 'Year_of_birth_of_fifth_person_in_household', 'Year_last_in_paid_job', 'Doing_last_7_days:_retired', 'Ever_had_children_living_in_household']

df_subset.drop(columns=existing_drops, inplace=True)
print(f"Dropped {len(existing_drops)} leakage/proxy/metadata columns: {existing_drops}")  
#Dropped 17 leakage/proxy/metadata columns: ['Population_size_weight_(must_be_combined_with_dweight_or_pspwght)', 'Age_of_respondent,_calculated', 'Year_of_birth_of_sixth_person_in_household', 'Year_of_birth_of_fourth_person_in_household', 'Design_weight', 'Partner_doing_last_7_days:_housework,_looking_after_children,_others', 'Doing_last_7_days:_housework,_looking_after_children,_others', "Respondent's_identification_number", 'Partner_doing_last_7_days:_retired', 'Post-stratification_weight_including_design_weight', 'Year_of_birth_of_second_person_in_household', 'Year_of_birth', 'Year_of_birth_of_third_person_in_household', 'Year_of_birth_of_fifth_person_in_household', 'Year_last_in_paid_job', 'Doing_last_7_days:_retired', 'Ever_had_children_living_in_household']


#%%
print('\n'.join(df_subset.columns))
print("-------------------------------------------\n Number of Columns:", len(df_subset.columns))
"""
Country
ESS_round
AgeGroup
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
Discrimination_of_respondent's_group:_gender
Member_of_a_group_discriminated_against_in_this_country
Discrimination_of_respondent's_group:_language
Discrimination_of_respondent's_group:_no_answer
Discrimination_of_respondent's_group:_not_applicable
Discrimination_of_respondent's_group:_nationality
Discrimination_of_respondent's_group:_other_grounds
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
Gender_of_second_person_in_household
Gender_of_third_person_in_household
Gender_of_fourth_person_in_household
Gender_of_fifth_person_in_household
Gender_of_sixth_person_in_household
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
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
Responsible_for_supervising_other_employees
Main_activity_last_7_days
Main_activity,_last_7_days._All_respondents._Post_coded
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months
Any_period_of_unemployment_and_work_seeking_within_last_5_years
Doing_last_7_days:_unemployed,_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job
Doing_last_7_days:_unemployed,_not_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job
-------------------------------------------
 Number of Columns: 108
"""


#%%
# ==================================================================================================  
# STEP 4: HANDLE MISSING VALUES (DUAL INDICATORS & ZERO-FILLING)
# ==================================================================================================
print("Step 4: Constructing _is_na and _is_missing indicator flags...")  
#Step 4: Constructing _is_na and _is_missing indicator flags...

NOT_APPLICABLE_CODES = {6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666'}
OTHER_MISSING_CODES = ALL_MISSING_CODES - NOT_APPLICABLE_CODES

base_features = [c for c in df_subset.columns if c not in ['Country', 'ESS_round', 'AgeGroup']]

transformed = {
    'Country': df_subset['Country'], 
    'ESS_round': df_subset['ESS_round'], 
    'AgeGroup': df_subset['AgeGroup']
}

total_na = 0
total_missing = 0

for c in base_features:
    s = df_subset[c]
    is_na = s.isin(NOT_APPLICABLE_CODES)
    is_miss = s.isin(OTHER_MISSING_CODES) | s.isna()

    total_na += int(is_na.sum())
    total_missing += int(is_miss.sum())

    transformed[f"{c}_is_na"] = is_na.astype(int)
    transformed[f"{c}_is_missing"] = is_miss.astype(int)
    
    num_s = pd.to_numeric(s, errors='coerce')
    num_s[is_na | is_miss] = 0.0  
    transformed[c] = num_s.fillna(0.0)  

print(f"Total 'Not Applicable' (NA) values across all features: {total_na}")  
# Total 'Not Applicable' (NA) values across all features: 2018882
print(f"Total other missing values: {total_missing}")  
# Total other missing values: 1788932

df_processed = pd.DataFrame(transformed)

le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['AgeGroup'].astype(str))

#%%

print('\n'.join(df_processed.columns))
print("-------------------------------------------\n Number of Columns:", len(df_processed.columns))
"""
Country
ESS_round
AgeGroup
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
Discrimination_of_respondent's_group:_gender_is_na
Discrimination_of_respondent's_group:_gender_is_missing
Discrimination_of_respondent's_group:_gender
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
Discrimination_of_respondent's_group:_other_grounds_is_na
Discrimination_of_respondent's_group:_other_grounds_is_missing
Discrimination_of_respondent's_group:_other_grounds
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
Gender_is_na
Gender_is_missing
Gender
Gender_of_second_person_in_household_is_na
Gender_of_second_person_in_household_is_missing
Gender_of_second_person_in_household
Gender_of_third_person_in_household_is_na
Gender_of_third_person_in_household_is_missing
Gender_of_third_person_in_household
Gender_of_fourth_person_in_household_is_na
Gender_of_fourth_person_in_household_is_missing
Gender_of_fourth_person_in_household
Gender_of_fifth_person_in_household_is_na
Gender_of_fifth_person_in_household_is_missing
Gender_of_fifth_person_in_household
Gender_of_sixth_person_in_household_is_na
Gender_of_sixth_person_in_household_is_missing
Gender_of_sixth_person_in_household
Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_na
Improve_knowledge/skills:_course/lecture/conference,_last_12_months_is_missing
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
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
Responsible_for_supervising_other_employees_is_na
Responsible_for_supervising_other_employees_is_missing
Responsible_for_supervising_other_employees
Main_activity_last_7_days_is_na
Main_activity_last_7_days_is_missing
Main_activity_last_7_days
Main_activity,_last_7_days._All_respondents._Post_coded_is_na
Main_activity,_last_7_days._All_respondents._Post_coded_is_missing
Main_activity,_last_7_days._All_respondents._Post_coded
Number_of_people_responsible_for_in_job_is_na
Number_of_people_responsible_for_in_job_is_missing
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job_is_na
Ever_had_a_paid_job_is_missing
Ever_had_a_paid_job
Doing_last_7_days:_paid_work_is_na
Doing_last_7_days:_paid_work_is_missing
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work_is_na
Partner_doing_last_7_days:_paid_work_is_missing
Partner_doing_last_7_days:_paid_work
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
target_encoded
-------------------------------------------
 Number of Columns: 319
"""

#%%
# ==================================================================================================
# STEP 5: GENERATE MISSINGNESS SUMMARY TABLE PER COUNTRY/ROUND
# ==================================================================================================
print("Step 5: Generating missingness summary table per country and round...")

# base_features (raw feature names, excluding grouping/target columns) was already computed in Step 4 —
# the indicator-flag and target columns added since then don't change that underlying set, so it's reused as-is.
  # # ✅ Missingness summary table generated successfully!
# Use the helper function to generate missingness summary
summary_df = create_subset_summary(df_processed, base_features, df_raw)
print("✅ Missingness summary table generated successfully!")  
# ✅ Missingness summary table generated successfully!

print(summary_df.head(10))
# Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  6.822698           6.061677
# 1  Austria        1.0  ...                  9.233431           8.335966
# 2  Austria        2.0  ...                  9.130840           8.132187
# 3  Austria        3.0  ...                  9.244205           8.624560
# 4  Austria        7.0  ...                  9.758716           7.460672
# 5  Austria        8.0  ...                 10.303391           8.078852
# 6  Austria        9.0  ...                  9.847260           9.113935
# 7  Belgium        1.0  ...                  9.575690           8.938299
# 8  Belgium        2.0  ...                  9.581409           8.641939
# 9  Belgium        3.0  ...                  9.400631           9.195333




# Re-run Age distribution check at this point in script (after preprocessing steps 2–5)
print("\n" + "="*84) # creates a division line via a bunch of equal signs
print("Age Distribution After Preprocessing Steps (Steps 2–5)")
print("="*84)

# Use df_processed, which has already been cleaned and encoded.
# Maps back to human-readable labels using the LabelEncoder's classes_.  # 
print_age_distribution(
    df_processed['target_encoded'],
    lambda c: le.inverse_transform([c])[0],
    code_word="encoded",
)
print("\n✅ Age distribution check completed.") 
 # ====================================================================================
# Older (37-51) (encoded 0): 108929 rows
# Young (18-34) (encoded 1): 103537 rows  # 

#%%
# Export the summary to CSV
output_summary_csv = here("data/processed/age_full_row_summary.csv")
summary_df.to_csv(output_summary_csv, index=False)
# ✅ Missingness summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/age_full_row_summary.csv' 
print(f"   Summary shape: {summary_df.shape[0]} rows (country-round pairs), {summary_df.shape[1]} columns")  
# Summary shape: 228 rows (country-round pairs), 5 columns


  # Step 6: Performing balanced stratified train/test split...
#%%
# Export the processed dataset to CSV
output_csv = here("data/processed/age_processed_two_missingness_indicators.csv")
df_processed.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_processed.shape} to '{output_csv}'")  
# ✅ Processed dataset saved with shape (212466, 319) to '/data/home/asher.katz/Projects/gender_differences/data/processed/age_processed_two_missingness_indicators.csv'


#%%
# ==================================================================================================
# STEP 6: BALANCED SAMPLING & TRAIN/TEST SPLIT
# ==================================================================================================
# Perform balanced stratified train/test split by Country and ESS_round.
print("Step 6: Performing balanced stratified train/test split...") 
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
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'Age', 'AgeGroup', 'target_encoded']] 
# Split into design matrix (X) and response vector (y).
X_train = training_data[feature_cols]
y_train = training_data['target_encoded']
X_test = test_data[feature_cols]
y_test = test_data['target_encoded']


# Verify no overlap between train and test indices (data integrity check)
assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test sets must be mutually exclusive"


#%%
# ==================================================================================================
# STEP 7: CREATE TRAIN AND TEST SUMMARY TABLES
# ==================================================================================================  
print("Step 7: Creating train and test summary tables...") 
# Shape: 228 rows (country-round pairs), 5 columns

# Create train and test summary tables
train_summary = create_subset_summary(training_data, base_features, df_processed)  # # Train Summary Preview:
# Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  6.977401           6.158192
# 1  Austria        1.0  ...                  9.309927           8.329298
# 2  Austria        2.0  ...                  9.063761           8.563358
# 3  Austria        3.0  ...                  9.039548           9.180791
# 4  Austria        7.0  ...                  9.874899           7.530266
test_summary = create_subset_summary(test_data, base_features, df_processed)  # # [5 rows x 5 columns]

# Save to CSV files  # # Test Summary Preview:
# Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  6.733866           6.006257
# 1  Austria        1.0  ...                  9.215904           8.337494
# 2  Austria        2.0  ...                  9.144690           8.043161
# 3  Austria        3.0  ...                  9.284522           8.514985
# 4  Austria        7.0  ...                  9.717172           7.435786
output_train_csv = here("data/processed/age_train_row_summary.csv")  # # [5 rows x 5 columns]
output_test_csv = here("data/processed/age_test_row_summary.csv")

train_summary.to_csv(output_train_csv, index=False)
print(f"✅ Train summary table saved to '{output_train_csv}'")  
# ✅ Train summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/age_train_row_summary.csv'
print(f"   Shape: {train_summary.shape[0]} rows (country-round pairs), {train_summary.shape[1]} columns")  
# Shape: 228 rows (country-round pairs), 5 columns

test_summary.to_csv(output_test_csv, index=False)
print(f"✅ Test summary table saved to '{output_test_csv}'")  
# ✅ Test summary table saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/age_test_row_summary.csv'
print(f"   Shape: {test_summary.shape[0]} rows (country-round pairs), {test_summary.shape[1]} columns")  
# Shape: 228 rows (country-round pairs), 5 columns

# Optional: Display first few rows
print("\nTrain Summary Preview:")  
# Train Summary Preview:
# Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  6.977401           6.158192
# 1  Austria        1.0  ...                  9.309927           8.329298
# 2  Austria        2.0  ...                  9.063761           8.563358
# 3  Austria        3.0  ...                  9.039548           9.180791
# 4  Austria        7.0  ...                  9.874899           7.530266
print(train_summary.head(5))  # [5 rows x 5 columns]

print("\nTest Summary Preview:")  
# Test Summary Preview:
# Country  ESS_round  ...  pct_cells_Non_applicable  pct_cells_Missing
# 0  Albania        6.0  ...                  6.733866           6.006257
# 1  Austria        1.0  ...                  9.215904           8.337494
# 2  Austria        2.0  ...                  9.144690           8.043161
# 3  Austria        3.0  ...                  9.284522           8.514985
# 4  Austria        7.0  ...                  9.717172           7.435786
print(test_summary.head(5))  # [5 rows x 5 columns]


#%%
# ==================================================================================================
# STEP 8: FEATURE SELECTION (Baseline Model)
# ==================================================================================================
print("Step 8: Fitting Baseline HistGradientBoostingClassifier & Selecting Features...")

baseline_hgb = HistGradientBoostingClassifier(random_state=42)
baseline_hgb.fit(X_train, y_train)

X_select_sample, y_select_sample = subsample_xy(X_train, y_train)
baseline_importance_df = compute_permutation_importance_df(baseline_hgb, X_select_sample, y_select_sample)  # \nStep 9: Computing final feature importances and generating plot...

# Retain strictly positive features
selected_features = baseline_importance_df[baseline_importance_df['Importance'] > 0]['Feature'].tolist()
dropped_features = baseline_importance_df[baseline_importance_df['Importance'] <= 0]['Feature'].tolist()


#%%
# ==================================================================================================
# FINAL MODEL TRAINING & EVALUATION
# ==================================================================================================
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
print('\n'.join(X_train_selected.columns))
print("-------------------------------------------\n Number of Columns:", len(X_train_selected.columns))
"""
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair_is_na
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair_is_missing
Most_people_try_to_take_advantage_of_you,_or_try_to_be_fair
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves_is_missing
Most_of_the_time_people_helpful_or_mostly_looking_out_for_themselves
Most_people_can_be_trusted_or_you_can't_be_too_careful_is_missing
Most_people_can_be_trusted_or_you_can't_be_too_careful
Worn_or_displayed_campaign_badge/sticker_last_12_months
Boycotted_certain_products_last_12_months
Feel_closer_to_a_particular_party_than_all_other_parties_is_missing
Feel_closer_to_a_particular_party_than_all_other_parties
Contacted_politician_or_government_official_last_12_months
Gays_and_lesbians_free_to_live_life_as_they_wish_is_missing
Gays_and_lesbians_free_to_live_life_as_they_wish
Government_should_reduce_differences_in_income_levels_is_missing
Government_should_reduce_differences_in_income_levels
Placement_on_left_right_scale_is_na
Placement_on_left_right_scale_is_missing
Placement_on_left_right_scale
Taken_part_in_lawful_public_demonstration_last_12_months
How_interested_in_politics_is_missing
How_interested_in_politics
How_close_to_party_is_na
How_close_to_party_is_missing
Signed_petition_last_12_months_is_missing
How_satisfied_with_the_way_democracy_works_in_country
How_satisfied_with_present_state_of_economy_in_country_is_na
How_satisfied_with_present_state_of_economy_in_country_is_missing
How_satisfied_with_present_state_of_economy_in_country
State_of_education_in_country_nowadays_is_missing
State_of_education_in_country_nowadays
How_satisfied_with_life_as_a_whole_is_na
How_satisfied_with_life_as_a_whole_is_missing
How_satisfied_with_life_as_a_whole
Trust_in_the_European_Parliament_is_na
Trust_in_the_European_Parliament_is_missing
Trust_in_the_European_Parliament
Trust_in_the_legal_system_is_missing
Trust_in_the_police_is_missing
Trust_in_politicians
Trust_in_the_United_Nations_is_na
Trust_in_the_United_Nations_is_missing
Trust_in_the_United_Nations
Voted_last_national_election_is_missing
Voted_last_national_election
Allow_many/few_immigrants_from_poorer_countries_outside_Europe_is_missing
Allow_many/few_immigrants_from_poorer_countries_outside_Europe
Immigration_bad_or_good_for_country's_economy_is_na
Immigration_bad_or_good_for_country's_economy_is_missing
Immigration_bad_or_good_for_country's_economy
Country's_cultural_life_undermined_or_enriched_by_immigrants_is_na
Immigrants_make_country_worse_or_better_place_to_live_is_missing
Immigrants_make_country_worse_or_better_place_to_live
Feeling_of_safety_of_walking_alone_in_local_area_after_dark
Belong_to_minority_ethnic_group_in_country_is_missing
Belong_to_minority_ethnic_group_in_country
Born_in_country
Respondent_or_household_member_victim_of_burglary/assault_last_5_years
Discrimination_of_respondent's_group:_ethnic_group
Member_of_a_group_discriminated_against_in_this_country
Discrimination_of_respondent's_group:_other_grounds
Discrimination_of_respondent's_group:_religion
Father_born_in_country_is_missing
How_happy_are_you_is_missing
Subjective_general_health
Hampered_in_daily_activities_by_illness/disability/infirmity/mental_problem_is_missing
How_often_pray_apart_from_at_religious_services_is_missing
How_often_pray_apart_from_at_religious_services
How_often_attend_religious_services_apart_from_special_occasions_is_missing
How_often_attend_religious_services_apart_from_special_occasions
How_religious_are_you_is_na
How_religious_are_you_is_missing
How_religious_are_you
Take_part_in_social_activities_compared_to_others_of_same_age_is_missing
Take_part_in_social_activities_compared_to_others_of_same_age
How_often_socially_meet_with_friends,_relatives_or_colleagues_is_na
How_often_socially_meet_with_friends,_relatives_or_colleagues
Number_of_people_living_regularly_as_member_of_household
Gender
Gender_of_second_person_in_household_is_na
Gender_of_second_person_in_household
Gender_of_third_person_in_household_is_na
Gender_of_third_person_in_household
Gender_of_fourth_person_in_household_is_na
Gender_of_fourth_person_in_household_is_missing
Gender_of_fourth_person_in_household
Gender_of_fifth_person_in_household_is_na
Gender_of_fifth_person_in_household
Gender_of_sixth_person_in_household
Improve_knowledge/skills:_course/lecture/conference,_last_12_months
Doing_last_7_days:_community_or_military_service
Partner_doing_last_7_days:_don't_know
Partner_doing_last_7_days:_not_applicable
Domicile,_respondent's_description
Doing_last_7_days:_permanently_sick_or_disabled
Partner_doing_last_7_days:_permanently_sick_or_disabled
Doing_last_7_days:_education
Partner_doing_last_7_days:_education
Years_of_full-time_education_completed_is_na
Years_of_full-time_education_completed
Highest_level_of_education,_ES_-_ISCED_is_na
Highest_level_of_education,_ES_-_ISCED_is_missing
Highest_level_of_education,_ES_-_ISCED
Number_of_employees_respondent_has/had
Father's_employment_status_when_respondent_14_is_missing
Father's_employment_status_when_respondent_14
Mother's_employment_status_when_respondent_14_is_missing
Mother's_employment_status_when_respondent_14
Responsible_for_supervising_other_employees
Main_activity_last_7_days_is_na
Main_activity,_last_7_days._All_respondents._Post_coded_is_na
Main_activity,_last_7_days._All_respondents._Post_coded_is_missing
Main_activity,_last_7_days._All_respondents._Post_coded
Number_of_people_responsible_for_in_job
Ever_had_a_paid_job
Doing_last_7_days:_paid_work
Partner_doing_last_7_days:_paid_work
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more_is_na
Any_period_of_unemployment_and_work_seeking_lasted_12_months_or_more
Ever_unemployed_and_seeking_work_for_a_period_more_than_three_months
Any_period_of_unemployment_and_work_seeking_within_last_5_years
Doing_last_7_days:_unemployed,_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_actively_looking_for_job
Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job
-------------------------------------------
 Number of Columns: 124
"""

#%%


hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_model.fit(X_train_selected, y_train)

test_data['y_pred'] = hgb_model.predict(X_test_selected)
test_data['y_prob'] = hgb_model.predict_proba(X_test_selected)[:, 1]

perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()
perf_df.to_csv(here("data/processed/age_country_and_round_performance_indicators.csv"), index=False)


#%%  # # ==================================================\n# Round 1: 0.7576\n# Round 2: 0.7562\n# Round 3: 0.7577\n# Round 4: 0.7564\n# Round 5: 0.7733\n# Round 6: 0.7705\n# Round 7: 0.7701\n# Round 8: 0.7576\n# Round 9: 0.7647
# ✅ Country summary saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/country_rankings.csv'
# ==================================================================================================
# STEP 9: FINAL MODEL IMPORTANCE PLOTTING & EXPORT
# ==================================================================================================
print("\nStep 9: Computing final feature importances and generating plot...")  
X_eval, y_eval = subsample_xy(X_test_selected, y_test)



#%%
final_importance_df = compute_permutation_importance_df(hgb_model, X_eval, y_eval)
plot_and_export_feature_importance(
    final_importance_df, 
    output_csv=here("data/processed/age_feature_importance.csv"), 
    output_html=here("plots/age_feature_importance_top20.html")
)

"""
Feature	Importance	Category
0	Partner_doing_last_7_days:_not_applicable	0.05892	Base Survey Question
1	Doing_last_7_days:_education	0.01744	Base Survey Question
2	Subjective_general_health	0.01578	Base Survey Question
3	Voted_last_national_election	0.01508	Base Survey Question
4	Gender_of_fourth_person_in_household_is_na	0.01290	Not Applicable Flag
...	...	...	...
119	Gender_of_fourth_person_in_household	-0.00072	Base Survey Question
120	How_often_pray_apart_from_at_religious_service...	-0.00076	Other Missing Flag
121	Responsible_for_supervising_other_employees	-0.00080	Base Survey Question
122	Gays_and_lesbians_free_to_live_life_as_they_wish	-0.00092	Base Survey Question
123	Government_should_reduce_differences_in_income...	-0.00102	Base Survey Question

"""


#%%

# Calculate average accuracy across all countries for each round
avg_accuracy_by_round = perf_df.groupby('ESS_round')['accuracy'].mean().reset_index()
avg_accuracy_by_round.columns = ['ESS_round', 'average_accuracy']

# Display the results in a clean table format
print("\n==================================================")
print("   AVERAGE Age ACCURACY ACROSS ALL COUNTRIES BY ROUND ")
print("==================================================")  # # ✅ Country summary saved to '/data/home/asher.katz/Projects/gender_differences/data/processed/country_rankings.csv'
for _, row in avg_accuracy_by_round.iterrows():
    print(f"Round {int(row['ESS_round'])}: {row['average_accuracy']:.4f}")  
    # ==================================================\n
    # # Round 1: 0.7576
    # # Round 2: 0.7562
    #  Round 3: 0.7577
    # Round 4: 0.7564
    # Round 5: 0.7733
    #  Round 6: 0.7705
    #  Round 7: 0.7701
    #  Round 8: 0.7576
    #  Round 9: 0.7647

avg_accuracy_by_round.to_csv(here("data/processed/age_average_accuracy_by_round.csv"), index=False)




#%%
# Create a DataFrame with country-level summary: name, average accuracy, and number of rounds
#  This is primarily used for creating the custom manual country selection and color mapping for the subsequent visualization script
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
output_country_summary_csv = here("data/processed/age_country_rankings.csv")
country_summary.to_csv(output_country_summary_csv, index=False)
print(f"✅ Country summary saved to '{output_country_summary_csv}'")  
print("\nCountry Summary Preview:")  
# Country Summary Preview:
# Name_of_country  average_accuracy  num_rounds
# 31        Slovenia          0.810186           9
# 11         Germany          0.794456           9
# 32           Spain          0.786103           9
# 25          Poland          0.784272           9
# 9          Finland          0.772352           9
# 33          Sweden          0.770639           9
# 24          Norway          0.770176           9
# 2          Belgium          0.770054           9
# 13         Hungary          0.762535           9
# 10          France          0.758556           9
print(country_summary.head(10))
# %%

