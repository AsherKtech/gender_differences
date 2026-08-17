# %% [markdown]
# ## ESS Data Processing: Clean, Cross-Comparable Dataset Construction

# %% Step 1: Imports & Data Load
from pathlib import Path
import pandas as pd
import pyreadstat
import numpy as np

# Load raw SPSS file
FILE_PATH = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)

print(f"Loaded raw dataset shape: {df_raw.shape}")

# Build initial code-to-label map
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}



# %% Step 2: Identify and Filter Gender Column
# ESS standard gender variable short code is 'gndr'
gender_raw_col = next((c for c in ['gndr', 'gender'] if c in df_raw.columns), None)
gender_label_col = code_to_label.get(gender_raw_col, 'Gender')

if gender_raw_col:
    # Rename gender column in df_raw for clear tracking
    df_raw.rename(columns={gender_raw_col: gender_label_col}, inplace=True)
    
    # Keep only Male (1 / 1.0) and Female (2 / 2.0)
    valid_gender_mask = df_raw[gender_label_col].isin([1, 2, 1.0, 2.0])
    df_raw = df_raw[valid_gender_mask].copy()
    
    print(f"Filtered to rows with valid Male/Female gender. Remaining rows: {len(df_raw):,}")
else:
    print("Warning: Could not locate gender column in raw dataset.")



# %% Step 3: Filter Columns Present in ALL Country/Round Combinations
# Define all missing/NA codes (6=Not App, 7=Refusal, 8=Don't Know, 9=No Answer)
ALL_MISSING_CODES = {
    6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666',
    7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777',
    8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888',
    9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'
}

group_cols = ['cntry', 'essround']
candidate_cols = [c for c in df_raw.columns if c not in group_cols]

# A cell is strictly valid ONLY if it is not NaN and not in any missing code set
def is_strictly_valid_response(s):
    return s.notna() & (~s.isin(ALL_MISSING_CODES))

# Evaluate if each column has at least one strictly valid response per country/round group
valid_per_group = (
    df_raw.groupby(group_cols)[candidate_cols]
    .apply(lambda group: group.apply(lambda col: is_strictly_valid_response(col).any()))
)

# Retain columns present across all country/round combinations
retained_features = valid_per_group.all(axis=0)
retained_features = retained_features[retained_features].index.tolist()

df_subset = df_raw[group_cols + retained_features].copy()
print(f"Columns retained (strictly valid in all country/round pairs): {len(df_subset.columns)} / {df_raw.shape[1]}")




# %% Step 4: Transform Missing Values & Build Dual Indicator Flags
NOT_APPLICABLE_CODES = {6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666'}
OTHER_MISSING_CODES = ALL_MISSING_CODES - NOT_APPLICABLE_CODES

transformed_data = {col: df_subset[col] for col in group_cols}

for col in retained_features:
    s = df_subset[col]
    
    # 1. Identify missing masks
    is_not_app = s.isin(NOT_APPLICABLE_CODES)
    is_other_missing = s.isin(OTHER_MISSING_CODES) | s.isna()
    
    # 2. Store indicator flags (1 or 0)
    transformed_data[f"{col}_is_na"] = is_not_app.astype(int)
    transformed_data[f"{col}_is_missing"] = is_other_missing.astype(int)
    
    # 3. Clean numeric base feature (zero-out missing positions)
    numeric_s = pd.to_numeric(s, errors='coerce')
    numeric_s[is_not_app | is_other_missing] = 0.0
    transformed_data[col] = numeric_s.fillna(0.0)

df_processed = pd.DataFrame(transformed_data)
print(f"Processed dataset shape (with indicators): {df_processed.shape}")







# %% Step 5: Apply Underscored Explanatory Labels Across All Headers
extended_label_map = {}
for col in df_processed.columns:
    if col in code_to_label:
        extended_label_map[col] = code_to_label[col]
    elif col.endswith("_is_na"):
        base = col[:-6]
        if base in code_to_label:
            extended_label_map[col] = f"{code_to_label[base]}_is_na"
    elif col.endswith("_is_missing"):
        base = col[:-11]
        if base in code_to_label:
            extended_label_map[col] = f"{code_to_label[base]}_is_missing"

df_processed.rename(columns=extended_label_map, inplace=True)

# Map country codes to full country names
cntry_val_labels = meta.variable_value_labels.get('cntry', {})
cntry_col = code_to_label.get('cntry', 'cntry')

if cntry_col in df_processed.columns:
    cleaned_cntry = df_processed[cntry_col].astype(str).str.strip()
    df_processed['Country_Name'] = cleaned_cntry.map(cntry_val_labels).fillna(cleaned_cntry)




#%%
# %% Step 5.5: Drop Specific Metadata/Redundant Columns and Their Indicators
cols_to_drop = [
    "Title_of_dataset",
    "Edition",
    "Production_date",
    "Respondent's_identification_number",
    "Design_weight",
    "Post-stratification_weight_including_design_weight",
    "Population_size_weight_(must_be_combined_with_dweight_or_pspwght)",
    "Country_of_birth",
    "Discrimination_of_respondent's_group:_gender",
    "Discrimination_of_respondent's_group:_other_grounds",
    "Country_of_birth,_father",
    "Language_most_often_spoken_at_home:_first_mentioned",
    "Country_of_birth,_mother",
    "Partner_doing_last_7_days:_community_or_military_service",
    "Partner_doing_last_7_days:_don't_know",
    "Partner_doing_last_7_days:_no_answer",
    "Partner_doing_last_7_days:_not_applicable",
    "Partner_doing_last_7_days:_other",
    "Partner_doing_last_7_days:_refusal",
    "Partner_doing_last_7_days:_permanently_sick_or_disabled",
    "Partner_doing_last_7_days:_education",
    "Partner's_employment_relation",
    "Partner_doing_last_7_days:_housework,_looking_after_children,_others",
    "Partner's_main_activity_last_7_days",
    "Partner_doing_last_7_days:_paid_work",
    "Partner_doing_last_7_days:_retired",
    "Partner_doing_last_7_days:_unemployed,_actively_looking_for_job",
    "Partner_doing_last_7_days:_unemployed,_not_actively_looking_for_job",
    "nan_count", 
    "Citizenship", 
    "Language_most_often_spoken_at_home:_second_mentioned", 
    "Region", 
    "Gender_of_second_person_in_household", 
    "Gender_of_third_person_in_household", 
    "Gender_of_fourth_person_in_household",
    "Gender_of_fifth_person_in_household", 
    "Gender_of_sixth_person_in_household"
]

# Build comprehensive drop set: includes base names, short codes, and indicator suffixes
all_drop_patterns = set()

for c in cols_to_drop:
    all_drop_patterns.add(c)
    all_drop_patterns.add(f"{c}_is_na")
    all_drop_patterns.add(f"{c}_is_missing")

# Find actual columns matching any drop pattern in df_processed
actual_cols_to_drop = [col for col in df_processed.columns if col in all_drop_patterns]

# Drop from df_processed
df_processed.drop(columns=actual_cols_to_drop, errors='ignore', inplace=True)

print(f"Dropped {len(actual_cols_to_drop)} columns (including indicator flags).")
print(f"Final dataset shape before saving: {df_processed.shape}")

# %% Step 6: Save Cleaned Dataset
output_csv = "ess_processed_two_missingness_indicators.csv"
df_processed.to_csv(output_csv, index=False)
print(f"Saved final processed dataset with shape {df_processed.shape} to '{output_csv}'")



#%%

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML
df_processed = pd.read_csv("ess_processed_two_missingness_indicators.csv")
# 1. Identify base features and country/round identifiers
cntry_col = 'Country_Name' if 'Country_Name' in df_processed.columns else 'cntry'
round_col = 'essround' if 'essround' in df_processed.columns else 'ESS_round'
 
#%%

base_features = [
    c for c in df_processed.columns 
    if not c.endswith('_is_na') 
    and not c.endswith('_is_missing') 
    and c not in [cntry_col, round_col, 'cntry', 'essround', 'Country', 'ESS_round', 'nan_count']
]

# 2. Melt base features and match indicator flags
df_base_melt = df_processed.melt(
    id_vars=[cntry_col, round_col], 
    value_vars=base_features, 
    var_name='variable', 
    value_name='val'
)

na_series = [df_processed[[cntry_col, round_col, f"{feat}_is_na"]].rename(columns={f"{feat}_is_na": 'is_na'}) for feat in base_features]
missing_series = [df_processed[[cntry_col, round_col, f"{feat}_is_missing"]].rename(columns={f"{feat}_is_missing": 'is_missing'}) for feat in base_features]

df_base_melt['is_na'] = pd.concat(na_series, axis=0)['is_na'].values
df_base_melt['is_missing'] = pd.concat(missing_series, axis=0)['is_missing'].values
df_base_melt['is_valid'] = ((df_base_melt['is_na'] == 0) & (df_base_melt['is_missing'] == 0)).astype(int)

# 3. Aggregate per (Country, Feature, ESS_round)
round_stats = (
    df_base_melt.groupby([cntry_col, 'variable', round_col])
    .agg(
        valid=('is_valid', 'sum'),
        not_applicable=('is_na', 'sum'),
        other_missing=('is_missing', 'sum')
    )
    .reset_index()
)

# 4. Build hover text and metrics
def build_detailed_hover(group):
    total_valid = group['valid'].sum()
    total_not_app = group['not_applicable'].sum()
    total_missing = group['other_missing'].sum()
    
    total_responses = total_valid + total_not_app + total_missing
    valid_pct = (total_valid / total_responses * 100) if total_responses > 0 else 0.0

    round_lines = []
    for _, row in group.iterrows():
        r = int(row[round_col])
        v = int(row['valid'])
        na = int(row['not_applicable'])
        m = int(row['other_missing'])
        tot = v + na + m
        r_pct = (v / tot * 100) if tot > 0 else 0.0
        
        round_lines.append(
            f"  • <b>Round {r}:</b> {v:,}/{tot:,} Valid ({r_pct:.1f}%) | "
            f"Not App: {na:,} | Other Missing: {m:,}"
        )

    breakdown_str = "<br>".join(round_lines)
    
    hover_str = (
        f"<b>Total Sample:</b> {total_responses:,}<br>"
        f"<b>Valid Responses:</b> {total_valid:,} (<b>{valid_pct:.1f}%</b>)<br>"
        f"<b>Not Applicable (_is_na = 1):</b> {total_not_app:,}<br>"
        f"<b>Other Missing (_is_missing = 1):</b> {total_missing:,}<br><br>"
        f"<b>Per-Round Breakdown:</b><br>{breakdown_str}"
    )
    return pd.Series({'total_valid': total_valid, 'valid_pct': round(valid_pct, 2), 'hover_text': hover_str})

agg_df = round_stats.groupby([cntry_col, 'variable']).apply(build_detailed_hover, include_groups=False).reset_index()

# 5. Pivot matrices
z_count_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='total_valid').fillna(0)
z_pct_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='valid_pct').fillna(0)
hover_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='hover_text').fillna("No Data Available")

x_labels = list(z_count_matrix.columns)
y_labels = list(z_count_matrix.index)

# 6. Build Plotly Figure with Two Traces and Dropdown Toggle
fig = go.Figure()

# Trace 1: Absolute Counts (Visible by default)
fig.add_trace(go.Heatmap(
    z=z_count_matrix.values,
    x=x_labels,
    y=y_labels,
    colorscale="plasma",
    colorbar=dict(title="Valid Count"),
    customdata=hover_matrix.values,
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><b>Valid Count:</b> %{z:,}<br><br>%{customdata}<extra></extra>",
    visible=True
))

# Trace 2: Percentages (%)
fig.add_trace(go.Heatmap(
    z=z_pct_matrix.values,
    x=x_labels,
    y=y_labels,
    colorscale="plasma",
    colorbar=dict(title="Valid %"),
    customdata=hover_matrix.values,
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><b>Valid %:</b> %{z:.1f}%<br><br>%{customdata}<extra></extra>",
    visible=False
))

# Configure Dropdown Menu Buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.5,
            y=1.12,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(
                    label="Color by: Total Valid Count",
                    method="update",
                    args=[{"visible": [True, False]}, {"title": "Valid Response Counts per Country & Feature"}]
                ),
                dict(
                    label="Color by: Valid Response %",
                    method="update",
                    args=[{"visible": [False, True]}, {"title": "Valid Response Percentage (%) per Country & Feature"}]
                )
            ]
        )
    ],
    title="Valid Response Metrics per Country & Feature (Hover for Detailed Breakdown)",
    height=max(650, len(y_labels) * 25),
    width=max(1200, len(x_labels) * 20),
    margin=dict(l=150, b=200, t=100, r=50)
)

fig.update_xaxes(tickmode='linear', dtick=1, tickangle=-45, automargin=True)




#%%
# Save and render
output_filename = "country_vs_feature_toggle_heatmap.html"
fig.write_html(output_filename)
print(f"Successfully saved toggle heatmap to '{output_filename}'")
HTML(fig.to_html(include_plotlyjs='cdn'))


#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML

# 1. Master lookup dictionaries from SPSS metadata
val_labels_by_var = meta.variable_value_labels or {}
reverse_col_map = {label.replace(" ", "_"): col for col, label in meta.column_names_to_labels.items()} if hasattr(meta, 'column_names_to_labels') else {}

# 2. Identify country column & base feature columns
cntry_col = 'Country_Name' if 'Country_Name' in df_processed.columns else 'cntry'

base_features = [
    c for c in df_processed.columns 
    if not c.endswith('_is_na') 
    and not c.endswith('_is_missing') 
    and c not in [cntry_col, 'cntry', 'essround', 'Country', 'ESS_round', 'nan_count']
]

# 3. Build long-format dataset with value distributions and indicator flags
df_base_melt = df_processed.melt(
    id_vars=[cntry_col], 
    value_vars=base_features, 
    var_name='variable', 
    value_name='val'
)

na_series = [df_processed[[cntry_col, f"{feat}_is_na"]].rename(columns={f"{feat}_is_na": 'is_na'}) for feat in base_features]
missing_series = [df_processed[[cntry_col, f"{feat}_is_missing"]].rename(columns={f"{feat}_is_missing": 'is_missing'}) for feat in base_features]

df_base_melt['is_na'] = pd.concat(na_series, axis=0)['is_na'].values
df_base_melt['is_missing'] = pd.concat(missing_series, axis=0)['is_missing'].values
df_base_melt['is_valid'] = ((df_base_melt['is_na'] == 0) & (df_base_melt['is_missing'] == 0)).astype(int)

# 4. Count value frequency distributions for valid observations
valid_counts_df = (
    df_base_melt[df_base_melt['is_valid'] == 1]
    .groupby([cntry_col, 'variable', 'val'], dropna=False)
    .size()
    .reset_index(name='count')
)

# Aggregate missing indicator totals per (Country, Feature)
indicator_totals = (
    df_base_melt.groupby([cntry_col, 'variable'])
    .agg(
        total_sample=('val', 'count'),
        total_valid=('is_valid', 'sum'),
        total_not_app=('is_na', 'sum'),
        total_other_missing=('is_missing', 'sum')
    )
    .reset_index()
)

# 5. Build translated value distribution hover strings
def safe_sort_key(val):
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, str(val))

def build_labeled_hover(group):
    country_name, feature_name = group.name if isinstance(group.name, tuple) else (None, group.name)
    
    # Retrieve metadata value labels
    orig_var = reverse_col_map.get(feature_name, feature_name)
    var_val_map = val_labels_by_var.get(orig_var, {})
    
    # Extract indicator summary row
    ind_match = indicator_totals[
        (indicator_totals[cntry_col] == country_name) & 
        (indicator_totals['variable'] == feature_name)
    ]
    
    if ind_match.empty:
        return pd.Series({'total_valid': 0, 'valid_pct': 0.0, 'hover_text': 'No Data'})
        
    tot_sample = int(ind_match['total_sample'].values[0])
    tot_valid = int(ind_match['total_valid'].values[0])
    tot_na = int(ind_match['total_not_app'].values[0])
    tot_missing = int(ind_match['total_other_missing'].values[0])
    
    valid_pct = (tot_valid / tot_sample * 100) if tot_sample > 0 else 0.0

    # Build valid response value breakdown lines
    value_lines = []
    if not group.empty:
        group_sorted = group.copy()
        group_sorted['sort_key'] = group_sorted['val'].apply(safe_sort_key)
        group_sorted = group_sorted.sort_values(by='sort_key')
        
        for _, row in group_sorted.iterrows():
            val = row['val']
            cnt = int(row['count'])
            pct = (cnt / tot_sample * 100) if tot_sample > 0 else 0.0
            
            val_key = int(val) if isinstance(val, (int, float)) and float(val).is_integer() else val
            label_text = var_val_map.get(val_key, var_val_map.get(val, None))
            display_label = f"{val_key} ({label_text})" if label_text else f"{val_key}"
            
            value_lines.append(f"  • <b>{display_label}:</b> {cnt:,} ({pct:.1f}%)")

    valid_dist_str = "<br>".join(value_lines) if value_lines else "None"
    
    hover_str = (
        f"<b>Total Sample:</b> {tot_sample:,}<br>"
        f"<b>Valid Responses:</b> {tot_valid:,} (<b>{valid_pct:.1f}%</b>)<br><br>"
        f"<b>Valid Value Distribution:</b><br>{valid_dist_str}<br><br>"
        f"<b>Missingness Breakdown:</b><br>"
        f"  • <b>Not Applicable (_is_na = 1):</b> {tot_na:,}<br>"
        f"  • <b>Other Missing (_is_missing = 1):</b> {tot_missing:,}"
    )
    
    return pd.Series({'total_valid': tot_valid, 'valid_pct': round(valid_pct, 2), 'hover_text': hover_str})

agg_df = valid_counts_df.groupby([cntry_col, 'variable']).apply(build_labeled_hover, include_groups=False).reset_index()

# 6. Pivot into matrices for Plotly
z_count_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='total_valid').fillna(0)
z_pct_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='valid_pct').fillna(0)
hover_matrix = agg_df.pivot(index=cntry_col, columns='variable', values='hover_text').fillna("No Data Available")

x_labels = list(z_count_matrix.columns)
y_labels = list(z_count_matrix.index)

# 7. Render Interactive Heatmap with Dropdown Toggle
fig = go.Figure()

# Trace 1: Count
fig.add_trace(go.Heatmap(
    z=z_count_matrix.values,
    x=x_labels,
    y=y_labels,
    colorscale="Plasma",
    colorbar=dict(title="Valid Count"),
    customdata=hover_matrix.values,
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><b>Valid Count:</b> %{z:,}<br><br>%{customdata}<extra></extra>",
    visible=True
))

# Trace 2: Percentage
fig.add_trace(go.Heatmap(
    z=z_pct_matrix.values,
    x=x_labels,
    y=y_labels,
    colorscale="Plasma",
    colorbar=dict(title="Valid %"),
    customdata=hover_matrix.values,
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><b>Valid %:</b> %{z:.1f}%<br><br>%{customdata}<extra></extra>",
    visible=False
))

# Configure Dropdown Buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.5,
            y=1.12,
            xanchor="center",
            yanchor="top",
            buttons=[
                dict(
                    label="Color by: Total Valid Count",
                    method="update",
                    args=[{"visible": [True, False]}, {"title": "Survey Response Counts per Country & Feature"}]
                ),
                dict(
                    label="Color by: Valid Response %",
                    method="update",
                    args=[{"visible": [False, True]}, {"title": "Survey Response Percentage (%) per Country & Feature"}]
                )
            ]
        )
    ],
    title="Survey Response Distributions per Country & Feature (Hover for Labeled Responses & Missingness)",
    height=max(650, len(y_labels) * 25),
    width=max(1200, len(x_labels) * 20),
    margin=dict(l=150, b=200, t=100, r=50)
)

fig.update_xaxes(tickmode='linear', dtick=1, tickangle=-45, automargin=True)

# Save and render
output_filename = "country_vs_feature_response_counts_processed.html"
fig.write_html(output_filename)
print(f"Successfully saved plot to '{output_filename}'")
HTML(fig.to_html(include_plotlyjs='cdn'))



#%%
# how many real country/round combinations are there? some countries don't have certain rounds
import pandas as pd

# Unique country/round combinations
combo_df = df[['Country', 'ESS_round']].drop_duplicates().sort_values(['Country', 'ESS_round'])

# Total count
total_unique_combos = len(combo_df)
print(f"Total Unique Country/Round Combinations: {total_unique_combos}\n")

# Summary per country
country_summary = (
    combo_df.groupby('Country')['ESS_round']
    .agg(
        total_rounds='count',
        rounds_participated=lambda x: ", ".join(map(str, sorted(x.astype(int))))
    )
    .reset_index()
)

print(country_summary.to_string(index=False))





# %%
# We want to create a training dataset with a uniform number of rows from each country/round combination
# Find me the country/round with the fewest number of rows and return how many rows 80% of that country/round rows is.

# Count rows per country/round
country_round_counts = df.groupby(['Country', 'ESS_round']).size().reset_index(name='count')

# Find minimum count
min_count = country_round_counts['count'].min()
eighty_percent_min = int(min_count * 0.8)

print(f"\nMinimum row count across any Country/Round combination: {min_count}")
print(f"80% of that minimum is: {eighty_percent_min} rows")

# Optional: Show which combinations have the minimum
min_combos = country_round_counts[country_round_counts['count'] == min_count]
print("\nCountry/Round combinations with the fewest rows:")
for _, row in min_combos.iterrows():
    print(f"  - {row['Country']} (Round {int(row['ESS_round'])}): {int(row['count'])} rows")






# %%

#We want country out of the training data, but we want to be able to see how the model performs on unseen data from each country. 
# If the model performs poorly on a particular country, thats not a bad thing, it means that in that particulr country the two genders are more similiar, which is the kind of thing we are looking to know.
#  We want to use the random forrest as a sort of gender role similarity metric. 
# So the question is, how do we keep Country away from the model in training but get it later for unseen data to see how gender similiar each particular country is?

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

#%%
# 1. Clean missing target rows
clean_df = df[df[target_col].notna()].copy()

# Filter out ESS missing codes from the target itself (77, 88, 99)
clean_df = clean_df[~clean_df[target_col].isin([7, 8, 9, 66, 77, 88, 99, 666, 777, 888, 999])]

# 2. Convert string/numeric gender target to strict 0 / 1 binary encoding
le = LabelEncoder()
clean_df['target_encoded'] = le.fit_transform(clean_df[target_col].astype(str))
#%%
# 3. Feature columns
exclude_cols = ['Country', 'ESS_round', 'cntry', 'essround', target_col, 'target_encoded', 'nan_count']
feature_cols = [col for col in clean_df.columns if col not in exclude_cols]
#%%
# 4. Stratified Train / Test split by Country & ESS Round
country_round_counts = clean_df.groupby(['Country', 'ESS_round']).size()
min_rows_per_combo = country_round_counts.min()
train_rows_per_combo = int(min_rows_per_combo * 0.8)

training_data = (
    clean_df.groupby(['Country', 'ESS_round'], group_keys=False)
    .apply(lambda x: x.sample(n=min(train_rows_per_combo, len(x)), random_state=42), include_groups=False)
)

test_data = clean_df.drop(index=training_data.index).copy()

X_train = training_data[feature_cols]
y_train = training_data['target_encoded']

X_test = test_data[feature_cols]
y_test = test_data['target_encoded']
#%%
# 5. Build Pipeline with Median Imputation for ESS Missingness
clf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

clf_pipeline.fit(X_train, y_train)

# 6. Predict on Test Set
test_data['y_pred'] = clf_pipeline.predict(X_test)
test_data['y_prob'] = clf_pipeline.predict_proba(X_test)[:, 1]


#%%
# 7. Evaluate per Country
def evaluate_country_similarity(group):
    y_true = group['target_encoded']
    y_pred = group['y_pred']
    y_prob = group['y_prob']
    
    acc = accuracy_score(y_true, y_pred)
    
    # Check if both binary classes exist in this country's test subset
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan
        
    return pd.Series({
        'test_n': len(group),
        'accuracy': acc,
        'roc_auc': auc
    })

#%%
gender_similarity_by_country = (
    test_data.groupby('Country')
    .apply(evaluate_country_similarity, include_groups=False)
    .reset_index()
    .sort_values(by='accuracy', ascending=True)
)

print("\n--- Gender Similarity Ranking (Lower Accuracy = More Similar Genders) ---")
print(gender_similarity_by_country.to_string(index=False))
# %%
# Check Pearson correlation between sample size and model accuracy
corr = gender_similarity_by_country['accuracy'].corr(gender_similarity_by_country['test_n'])
print(f"Correlation between test_n and accuracy: {corr:.3f}")




# %%
# %%
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import numpy as np

# Use encoded binary target ('target_encoded') to ensure exact metric compatibility
y_true = test_data['target_encoded']
y_pred = test_data['y_pred']
y_prob = test_data['y_prob'] if 'y_prob' in test_data.columns else None

# Calculate overall point metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
auc = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else np.nan

# Map original class labels back for readability in the classification report
target_names = [str(c) for c in le.classes_] if 'le' in globals() else None

print("==================================================")
print("       OVERALL TEST DATA PERFORMANCE REPORT       ")
print("==================================================")
print(f"Total Test Samples (n) : {len(test_data):,}")
print(f"Accuracy               : {acc:.4f}")
print(f"Precision              : {prec:.4f}")
print(f"Recall                 : {rec:.4f}")
print(f"F1-Score               : {f1:.4f}")
print(f"ROC-AUC Score          : {auc:.4f}")
print("==================================================\n")

print("Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


# %%
# Compute metrics broken down by Country and ESS_round
def compute_round_metrics(group):
    y_t = group['target_encoded']
    y_p = group['y_pred']
    
    acc_val = accuracy_score(y_t, y_p)
    _, _, f1_val, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
    
    return pd.Series({
        'test_n': len(group),
        'accuracy': acc_val,
        'f1_score': f1_val
    })

country_round_perf = (
    test_data.groupby(['Country', 'ESS_round'])
    .apply(compute_round_metrics, include_groups=False)
    .reset_index()
    .sort_values(by=['Country', 'ESS_round'])
)

# Save the Country x Round metrics table to CSV
csv_filename = "ess_country_and_round_performance_nonan.csv"
country_round_perf.to_csv(csv_filename, index=False)

print(f"✅ Country/Round metrics saved to {csv_filename}")

#%%
country_round_perf

# %%
import plotly.express as px
import plotly.io as pio

# Set default template to avoid theme conflicts in VS Code/Jupyter
pio.templates.default = "plotly_white"

# Ensure ESS_round is sorted as an integer sequence
country_round_perf['ESS_round'] = country_round_perf['ESS_round'].astype(int)
country_round_perf = country_round_perf.sort_values(by=['Country', 'ESS_round'])

# Create interactive multi-line chart
fig = px.line(
    country_round_perf,
    x='ESS_round',
    y='accuracy',
    color='Country',
    markers=True,
    hover_data=['test_n', 'f1_score'],
    labels={
        'ESS_round': 'ESS Survey Round',
        'accuracy': 'Model Accuracy (Gender Predictability)',
        'Country': 'Country'
    },
    title="<b>Within-Country Gender Similarity Over Time</b><br><sup>Lower Accuracy = Higher Gender Role Similarity</sup>"
)

# Reference line at 50% (chance level / maximum similarity)
fig.add_hline(
    y=0.50, 
    line_dash="dash", 
    line_color="gray", 
    annotation_text="Random Guess (50% - Equal Similarity)", 
    annotation_position="bottom right"
)

fig.update_xaxes(dtick=1, tickmode='linear')
fig.update_yaxes(range=[0.40, 1.0], tickformat=".0%")

fig.update_layout(
    height=700,
    width=1100,
    hovermode="x unified",
    legend=dict(title="Click to Filter Countries:", y=0.5),
    margin=dict(l=80, r=150, t=100, b=80)
)

# 1. Save directly as a standalone HTML file
html_filename = "gender_similarity_trends_noNans.html"
fig.write_html(html_filename, include_plotlyjs='cdn')
print(f"✅ Interactive plot saved to {html_filename}")

# 2. Display directly in notebook environment
try:
    fig.show(renderer="notebook")
except Exception:
    fig.show()












# %%
# %%
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

#%%
# 1. ESS Missingness sets
NOT_APP = {6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666'}
REFUSAL = {7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777'}
DONT_KNOW = {8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888'}
NO_ANSWER = {9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'}

# 2. Clean Target (Gender)
target_col = 'Gender'
clean_df = df[df[target_col].notna()].copy()
clean_df = clean_df[~clean_df[target_col].isin(NOT_APP | REFUSAL | DONT_KNOW | NO_ANSWER)]

le = LabelEncoder()
clean_df['target_encoded'] = le.fit_transform(clean_df[target_col].astype(str))
#%%
# 3. Identify feature columns
exclude_cols = ['Country', 'ESS_round', 'cntry', 'essround', target_col, 'target_encoded', 'nan_count']
raw_feature_cols = [col for col in clean_df.columns if col not in exclude_cols]
#%%
# 4. Engineer Explicit Missingness Indicators
print("Engineering missingness features...")
df_processed = clean_df.copy()

for col in raw_feature_cols:
    s = df_processed[col]
    
    # Create behavioral binary indicators (1 if condition met, 0 otherwise)
    df_processed[f'{col}_is_refusal'] = s.isin(REFUSAL).astype(int)
    df_processed[f'{col}_is_dontknow'] = s.isin(DONT_KNOW).astype(int)
    
    # Convert ALL ESS missing codes in the numeric column to actual NaN
    all_missing_codes = NOT_APP | REFUSAL | DONT_KNOW | NO_ANSWER
    df_processed[col] = df_processed[col].mask(s.isin(all_missing_codes), np.nan)

# Update feature list to include new indicator columns
feature_cols = [c for c in df_processed.columns if c not in exclude_cols]

#%%
# 5. Stratified Train/Test Split
country_round_counts = df_processed.groupby(['Country', 'ESS_round']).size()
min_rows_per_combo = country_round_counts.min()
train_rows_per_combo = int(min_rows_per_combo * 0.8)

training_data = (
    df_processed.groupby(['Country', 'ESS_round'], group_keys=False)
    .apply(lambda x: x.sample(n=min(train_rows_per_combo, len(x)), random_state=42), include_groups=False)
)

test_data = df_processed.drop(index=training_data.index).copy()

X_train = training_data[feature_cols]
y_train = training_data['target_encoded']

X_test = test_data[feature_cols]
y_test = test_data['target_encoded']

#%%
# 6. Model Training
# Option A: HistGradientBoostingClassifier natively handles NaNs as a valid split direction
clf = HistGradientBoostingClassifier(random_state=42)

# Option B: Standard Random Forest with Median Imputation for the numeric NaNs
# clf = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
# ])

clf.fit(X_train, y_train)



#%%
# 7. Predict & Evaluate
test_data['y_pred'] = clf.predict(X_test)
test_data['y_prob'] = clf.predict_proba(X_test)[:, 1]

#%%
def evaluate_country_similarity(group):
    y_true = group['target_encoded']
    y_pred = group['y_pred']
    y_prob = group['y_prob']
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        
    return pd.Series({
        'test_n': len(group),
        'accuracy': acc,
        'roc_auc': auc
    })

gender_similarity_by_country = (
    test_data.groupby('Country')
    .apply(evaluate_country_similarity, include_groups=False)
    .reset_index()
    .sort_values(by='accuracy', ascending=True)
)

print("\n--- Gender Similarity Ranking (With Missingness Features) ---")
print(gender_similarity_by_country.to_string(index=False))
# %%

# Save gender similarity results to CSV
gender_similarity_by_country.to_csv("gender_similarity_ranking.csv", index=False)
print("✅ Gender similarity ranking saved to 'gender_similarity_ranking.csv'")

# %%
from sklearn.inspection import permutation_importance

# 1. Compute permutation importances on test set
result = permutation_importance(
    clf, X_test, y_test, 
    n_repeats=5, 
    random_state=42, 
    n_jobs=-1
)

# 2. Package into a Pandas Series
importances = pd.Series(result.importances_mean, index=feature_cols)

# 3. Top 20 overall features
print("--- Top 20 Most Predictive Features ---")
print(importances.nlargest(20).round(4))

# 4. Check if engineered missingness features carry signal
missingness_importances = importances[importances.index.str.contains('_is_')]
print("\n--- Top Missingness Indicators for Predicting Gender ---")
print(missingness_importances.nlargest(10).round(4))
# %%


# %%
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.inspection import permutation_importance

# 1. Extract feature importances (handles both HistGradientBoosting & RandomForest)
if hasattr(clf, 'named_steps'):
    # Pipeline + RandomForest
    rf = clf.named_steps['rf']
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    metric_label = "MDI Feature Importance (Gini)"
else:
    # HistGradientBoostingClassifier (Permutation Importance)
    print("Computing permutation importances on test set (this may take a few seconds)...")
    perm_result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(perm_result.importances_mean, index=feature_cols)
    metric_label = "Mean Accuracy Drop on Permutation"

# 2. Build plotting DataFrames
top_20_overall = (
    importances.nlargest(20)
    .reset_index()
    .rename(columns={'index': 'Feature', 0: 'Importance'})
    .sort_values(by='Importance', ascending=True) # Ascending for clean bottom-to-top horiz bar plot
)

missingness_importances = importances[importances.index.str.contains('_is_')]
top_15_missing = (
    missingness_importances.nlargest(15)
    .reset_index()
    .rename(columns={'index': 'Feature', 0: 'Importance'})
    .sort_values(by='Importance', ascending=True)
)

# 3. Plot Top 20 Overall Features
fig_overall = px.bar(
    top_20_overall,
    x='Importance',
    y='Feature',
    orientation='h',
    text_auto='.4f',
    title="<b>Top 20 Most Predictive Features for Gender</b>",
    labels={'Importance': metric_label, 'Feature': 'Survey Variable'},
    color='Importance',
    color_continuous_scale='Viridis'
)

fig_overall.update_layout(
    height=600,
    width=900,
    showlegend=False,
    xaxis_title=metric_label,
    yaxis_title="",
    margin=dict(l=150, r=50, t=80, b=50),
    template="plotly_white"
)

# Save and Show Overall Plot
fig_overall.write_html("top_20_gender_features.html")
fig_overall.show()


# 4. Plot Missingness / Non-Response Features specifically
fig_missing = px.bar(
    top_15_missing,
    x='Importance',
    y='Feature',
    orientation='h',
    text_auto='.4f',
    title="<b>Top Missingness Indicators ('Don't Know' / 'Refusal') Predicting Gender</b>",
    labels={'Importance': metric_label, 'Feature': 'Missingness Indicator'},
    color='Importance',
    color_continuous_scale='Plasma'
)

fig_missing.update_layout(
    height=550,
    width=900,
    showlegend=False,
    xaxis_title=metric_label,
    yaxis_title="",
    margin=dict(l=200, r=50, t=80, b=50),
    template="plotly_white"
)

# Save and Show Missingness Plot
fig_missing.write_html("top_missingness_gender_features.html")
fig_missing.show()
# %%
