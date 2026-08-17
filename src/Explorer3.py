# %% [markdown]
# ## ESS Data Processing: Extracting Fully Cross-Comparable Columns

# %% Cell 1: Imports & Setup
from pathlib import Path
import pandas as pd
import pyreadstat

# %% Cell 2: Load SPSS File & Metadata
FILE_PATH1 = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
FILE_PATH2 = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')

df, meta = pyreadstat.read_sav(FILE_PATH2)

# Build a clean mapping dictionary: short_code -> Label_With_Underscores
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}

print(f"Loaded raw dataset with shape: {df.shape}")


# %% Cell 3: Filter Dataset for Consistent Columns
# Step A: Subset to Rounds 1 through 9 and drop any missing country/round rows
df_sub = df[df['essround'].between(1, 9)].dropna(subset=['cntry', 'essround']).copy()

# Step B: Identify columns that have AT LEAST ONE valid response in EVERY country-round pair
group_has_data = df_sub.groupby(['cntry', 'essround']).apply(
    lambda group: group.notna().any()
)

valid_cols = group_has_data.columns[group_has_data.all()].tolist()

# Step C: Mandatory columns that MUST be in the final dataset
mandatory_cols = ['cntry', 'essround', 'gndr']

# Ensure mandatory columns are included even if they were excluded during group validation
final_cols = list(dict.fromkeys(mandatory_cols + valid_cols))  # preserves order while deduplicating

# Step D: Build the filtered DataFrame
filtered_df = df_sub[final_cols].copy()

print(f"Filtered DataFrame shape: {filtered_df.shape}")
print(f"Mandatory columns included: {[col for col in mandatory_cols if col in filtered_df.columns]}")



# %%
filtered_df.columns.tolist()


# %% Cell 4: Apply Underscored Labels to Column Headers
# Re-assert code_to_label dictionary from meta metadata
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}

# Map short codes to their underscored descriptions
filtered_df.rename(columns=code_to_label, inplace=True)

print("=== SAMPLE OF NEW COLUMN NAMES ===")
for col in list(filtered_df.columns)[:10]:
    print(col)





# %% Cell 5: Validation Checks
# 1. Check total NaN count across filtered DataFrame
total_nans = filtered_df.isna().sum().sum()
print(f"\nTotal NaN values in filtered dataset: {total_nans}")

# 2. Verify country-round pair completeness
group_valid_check = filtered_df.groupby([code_to_label['cntry'], code_to_label['essround']]).apply(
    lambda group: group.notna().any()
)
invalid_cols = group_valid_check.columns[~group_valid_check.all()].tolist()

if not invalid_cols:
    print("SUCCESS: Every column has valid data for all country-round pairs!")
else:
    print(f"WARNING: Found {len(invalid_cols)} columns missing full coverage.")






#%%
# 1. Extract the VALUE labels dictionary for the 'cntry' column
# pyreadstat stores value labels as: {'cntry': {'AT': 'Austria', 'BE': 'Belgium', ...}}
cntry_value_labels = meta.variable_value_labels.get('cntry', {})

# 2. Ensure Country values are cleaned strings
filtered_df['Country'] = filtered_df['Country'].astype(str).str.strip()

# 3. Map country codes ('AT', 'BE') to full country names ('Austria', 'Belgium')
filtered_df['Country'] = filtered_df['Country'].map(cntry_value_labels).fillna(filtered_df['Country'])

# Inspect the updated column
print(filtered_df['Country'].head(10))



# %%
# Get column names (handles both short codes and renamed underscored labels)
cntry_col = code_to_label.get('cntry', 'Country')
round_col = code_to_label.get('essround', 'ESS_round')

# Count rows for each Country/Round pair and format as a clean DataFrame
combo_counts = (
    filtered_df.groupby([cntry_col, round_col])
    .size()
    .reset_index(name='row_count')
)

# Display the counts
print(combo_counts.to_string(index=False))

# %%
# Pivot table of row counts per country per round
count_matrix = pd.crosstab(
    filtered_df[cntry_col], 
    filtered_df[round_col], 
    margins=True, 
    margins_name="Total"
)

# Fill missing combinations with 0
count_matrix.fillna(0).astype(int)


#%%
count_matrix.to_csv("ess_country_round_counts.csv")

print("Saved country-round counts to 'ess_country_round_counts.csv'")



# %%
# Get dynamic column names (handles short codes and underscored labels)
cntry_col = code_to_label.get('cntry', 'Country')
round_col = code_to_label.get('essround', 'ESS_round')

# 1. Create long-format dataset: Country | Round | Sample Size
country_round_counts = (
    filtered_df.groupby([cntry_col, round_col])
    .size()
    .reset_index(name='sample_size')
)

# 2. Sort cleanly by Country then Round
country_round_counts.sort_values(by=[cntry_col, round_col], inplace=True)

# 3. View the first 15 rows
print("=== COUNTRY / ROUND SAMPLE SIZES ===")
print(country_round_counts.head(15).to_string(index=False))

# 4. Save to CSV for easy loading into future analyses
country_round_counts.to_csv("ess_country_round_sample_sizes.csv", index=False)
print("\nSaved as 'ess_country_round_sample_sizes.csv'")




# %%
# 1. Dynamically find the column names for country and round
cntry_col = code_to_label.get('cntry', 'cntry')
round_col = code_to_label.get('essround', 'essround')

# 2. Ensure 'nan_count' exists in filtered_df (counts NaNs across columns per row)
filtered_df['nan_count'] = filtered_df.isna().sum(axis=1)

# 3. Aggregate row NaN statistics per Country and Round
row_nan_summary = (
    filtered_df.groupby([cntry_col, round_col])['nan_count']
    .agg(
        total_rows='count',
        mean_nans_per_row='mean',
        max_nans_in_a_row='max',
        rows_with_at_least_one_nan=lambda x: (x > 0).sum()
    )
    .reset_index()
)

# 4. View results
print("=== ROW NaN SUMMARY BY COUNTRY & ROUND ===")
print(row_nan_summary.head(15).to_string(index=False))

# 5. Optional: Save summary to CSV
row_nan_summary.to_csv("ess_row_nan_summary.csv", index=False)

# %% Drop any columns that are entirely empty (100% NaN)
before_count = filtered_df.shape[1]

# Drop columns where all values are NaN
filtered_df.dropna(how='all', axis=1, inplace=True)

after_count = filtered_df.shape[1]
print(f"Dropped {before_count - after_count} completely empty columns. Remaining columns: {after_count}")

# %% Find and drop duplicated columns that contain only NaNs
# Keep the first occurrence, drop subsequent duplicate empty columns
filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated(keep='first')]

# Alternatively, drop any column that is entirely NaN across all rows
filtered_df = filtered_df.dropna(how='all', axis=1)

#%%

after_count = filtered_df.shape[1]
print(f"Dropped {before_count - after_count} completely empty columns. Remaining columns: {after_count}")

# %% 1. Clean column names (strip suffixes like '.1', '.2', '[DUPLICATE]')
filtered_df.columns = (
    filtered_df.columns.str.replace(r"\.\d+$", "", regex=True)
    .str.replace(r"\s*\[DUPLICATE\]", "", regex=True)
)

# %% 2. Combine duplicate columns cleanly (pandas 2.2+ compatible)
filtered_df = (
    filtered_df.T
    .groupby(level=0)
    .bfill()
    .T
    .loc[:, ~filtered_df.columns.duplicated(keep='first')]
)

print(f"Shape after combining duplicate columns: {filtered_df.shape}")
# %% List of metadata, weight, and redundant partner/demographic columns to drop
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
]

# Drop the columns from filtered_df
filtered_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

print(f"Updated DataFrame shape after dropping columns: {filtered_df.shape}")


# %% Save filtered_df to CSV
output_path = "ess_filtered_dataset.csv"

# Save to CSV (excluding the DataFrame index)
filtered_df.to_csv(output_path, index=False)

print(f"Successfully saved filtered DataFrame with shape {filtered_df.shape} to '{output_path}'!")

print(f"Successfully saved cleaned dataset to '{output_path}'!")


# %%
# load saved  filtered_df dataset
import pandas as pd

output_path = "ess_filtered_dataset.csv"
df = pd.read_csv(output_path)
df

#%%
# show me all of the columns of the DataFrame, so that they will all display in an interactive notebook window
with pd.option_context('display.max_columns', None 
                       , 'display.max_rows', None):
    print(df.columns.tolist())



# %%
# lets drop the following columns 
#  'Gender_of_second_person_in_household', 'Gender_of_third_person_in_household', 'Gender_of_fourth_person_in_household', 'Gender_of_fifth_person_in_household', 'Gender_of_sixth_person_in_household'
df = df.drop(columns=['Gender_of_second_person_in_household', 'Gender_of_third_person_in_household', 'Gender_of_fourth_person_in_household', 'Gender_of_fifth_person_in_household', 'Gender_of_sixth_person_in_household'])





#%%
df
# %%
# create a plotly visualization where the countries are the yaxis, the columns/featutres are the xaxis and the values are the count of rows in tthat country for that column/feature

import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import HTML


# 1. Identify feature columns (excluding metadata/helper columns)
exclude_cols = ['Country', 'ESS_round', 'nan_count']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# 2. Count non-null rows per (Country, ESS_round) for every feature
# melt reshapes data so we can aggregate efficiently
df_melted = df.melt(id_vars=['Country', 'ESS_round'], value_vars=feature_cols)
df_valid = df_melted.dropna(subset=['value'])

# Group by Country, Feature, and Round
round_counts = (
    df_valid.groupby(['Country', 'variable', 'ESS_round'])
    .size()
    .reset_index(name='count')
)

# 3. Build formatted hover strings for each (Country, Feature) combination
def build_hover_text(group):
    total = group['count'].sum()
    # Format each round breakdown line
    round_lines = "<br>".join([
        f"  • Round {int(r)}: {c:,} rows" 
        for r, c in zip(group['ESS_round'], group['count'])
    ])
    hover_str = f"<b>Total Valid Rows:</b> {total:,}<br><br><b>Breakdown by Round:</b><br>{round_lines}"
    return pd.Series({'total_count': total, 'hover_text': hover_str})

# Aggregate hover text and total counts
agg_df = round_counts.groupby(['Country', 'variable']).apply(build_hover_text).reset_index()

# 4. Pivot into matrices for Plotly Heatmap
z_matrix = agg_df.pivot(index='Country', columns='variable', values='total_count').fillna(0)
hover_matrix = agg_df.pivot(index='Country', columns='variable', values='hover_text').fillna("No Valid Rows")

# 5. Create Heatmap
fig = px.imshow(
    z_matrix,
    labels=dict(x="Feature / Column", y="Country", color="Valid Row Count"),
    x=z_matrix.columns,
    y=z_matrix.index,
    color_continuous_scale="plasma",
    title="Valid Response Counts per Country and Feature (Hover for ESS Round Breakdown)"
)

# 6. Attach custom hover text matrix
fig.update_traces(
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><br>%{customdata}<extra></extra>",
    customdata=hover_matrix.values
)

# 7. Layout tweaks for readability
fig.update_layout(
    height=max(600, len(z_matrix.index) * 25),
    width=max(1000, len(z_matrix.columns) * 15),
    xaxis_tickangle=-45,
    margin=dict(l=150, b=150, t=80, r=50)
)


# Output the interactive plot directly without calling fig.show()
HTML(fig.to_html(include_plotlyjs='cdn'))




#%%
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import HTML

# 1. Exclude non-feature metadata columns
exclude_cols = ['Country', 'ESS_round', 'nan_count']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# 2. Melt to unpivot the dataframe (preserves missingness per column)
df_melted = df.melt(id_vars=['Country', 'ESS_round'], value_vars=feature_cols)

# 3. Calculate both valid and missing (NaN) rows per (Country, Feature, Round)
round_stats = (
    df_melted.groupby(['Country', 'variable', 'ESS_round'])['value']
    .agg(
        total_rows='count',                        # Total survey responses in round
        valid_rows=lambda x: x.notna().sum(),       # Non-null responses
        nan_rows=lambda x: x.isna().sum()           # Missing (NaN) responses
    )
    .reset_index()
)

# 4. Format detailed hover strings showing round-by-round NaN breakdowns
def build_detailed_hover(group):
    total_valid = group['valid_rows'].sum()
    total_nans = group['nan_rows'].sum()
    total_overall = total_valid + total_nans
    overall_nan_pct = (total_nans / total_overall * 100) if total_overall > 0 else 0

    round_lines = []
    for _, row in group.iterrows():
        r = int(row['ESS_round'])
        v = int(row['valid_rows'])
        n = int(row['nan_rows'])
        tot = v + n
        pct_nan = (n / tot * 100) if tot > 0 else 0
        round_lines.append(
            f"  • <b>Round {r}:</b> {v:,} valid | {n:,} NaNs ({pct_nan:.1f}% missing)"
        )

    breakdown_str = "<br>".join(round_lines)
    hover_str = (
        f"<b>Total Valid Rows:</b> {total_valid:,}<br>"
        f"<b>Total NaNs:</b> {total_nans:,} ({overall_nan_pct:.1f}% missing overall)<br><br>"
        f"<b>Breakdown by Round:</b><br>{breakdown_str}"
    )
    return pd.Series({'total_valid': total_valid, 'hover_text': hover_str})

# Aggregate hover text per Country & Feature
agg_df = round_stats.groupby(['Country', 'variable']).apply(build_detailed_hover).reset_index()

# 5. Pivot into matrices for the Heatmap
z_matrix = agg_df.pivot(index='Country', columns='variable', values='total_valid').fillna(0)
hover_matrix = agg_df.pivot(index='Country', columns='variable', values='hover_text').fillna("No Data Available")

# 6. Create Heatmap
fig = px.imshow(
    z_matrix,
    labels=dict(x="Feature / Column", y="Country", color="Valid Row Count"),
    x=z_matrix.columns,
    y=z_matrix.index,
    color_continuous_scale="plasma",
    title="Valid Response Counts per Country & Feature (Hover for Per-Round NaN Breakdown)"
)

# Attach hover matrix
fig.update_traces(
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><br>%{customdata}<extra></extra>",
    customdata=hover_matrix.values
)

# Force every x-axis column label to render explicitly
fig.update_xaxes(
    tickmode='linear',
    dtick=1,
    tickangle=-45,
    automargin=True
)

fig.update_layout(
    height=max(600, len(z_matrix.index) * 25),
    width=max(1200, len(z_matrix.columns) * 25),  # Generous width so labels don't crowd
    margin=dict(l=150, b=200, t=80, r=50)
)

# Explicitly use HTML display to bypass any local nbformat renderer issues
HTML(fig.to_html(include_plotlyjs='cdn'))








# %%
# Save the plot to an HTML file for offline viewing
fig.write_html("country_vs_feature_round_counts.html")




# %%
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import HTML

# 1. Master lookup dictionary from SPSS metadata
val_labels_by_var = meta.variable_value_labels or {}

# Reverse mapping to match renamed descriptions back to raw SPSS variable codes
reverse_col_map = {label.replace(" ", "_"): col for col, label in meta.column_names_to_labels.items()}

# 2. Exclude non-feature columns
exclude_cols = ['Country', 'ESS_round', 'nan_count']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# 3. Unpivot features
df_melted = df.melt(id_vars=['Country'], value_vars=feature_cols)

# 4. Count response value frequencies
value_counts_df = (
    df_melted.groupby(['Country', 'variable'], dropna=False)['value']
    .value_counts(dropna=False)
    .reset_index(name='count')
)

# Sorting key for mixed data types
def safe_sort_key(val):
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, str(val))

# 5. Build hover breakdown
def build_translated_value_hover(group):
    _, feature_name = group.name if isinstance(group.name, tuple) else (None, group['variable'].iloc[0])
    
    orig_var = reverse_col_map.get(feature_name, feature_name)
    var_val_map = val_labels_by_var.get(orig_var, {})

    total_responses = group['count'].sum()
    
    valid_group = group[group['value'].notna()].copy()
    nan_group = group[group['value'].isna()]
    
    total_valid = valid_group['count'].sum()
    total_nan = nan_group['count'].sum() if not nan_group.empty else 0
    nan_pct = (total_nan / total_responses * 100) if total_responses > 0 else 0

    valid_group['sort_key'] = valid_group['value'].apply(safe_sort_key)
    valid_sorted = valid_group.sort_values(by='sort_key')

    value_lines = []
    for _, row in valid_sorted.iterrows():
        val = row['value']
        
        val_key = int(val) if isinstance(val, (int, float)) and float(val).is_integer() else val
        label_text = var_val_map.get(val_key, var_val_map.get(val, None))
        
        display_label = f"{val_key} ({label_text})" if label_text else f"{val_key}"
            
        cnt = int(row['count'])
        pct = (cnt / total_responses * 100) if total_responses > 0 else 0
        value_lines.append(f"  • <b>{display_label}:</b> {cnt:,} ({pct:.1f}%)")

    if total_nan > 0:
        value_lines.append(f"  • <b>Missing / NaN:</b> {total_nan:,} ({nan_pct:.1f}%)")

    breakdown_str = "<br>".join(value_lines)
    hover_str = (
        f"<b>Total Sample:</b> {total_responses:,}<br>"
        f"<b>Valid Responses:</b> {total_valid:,}<br><br>"
        f"<b>Response Value Breakdown:</b><br>{breakdown_str}"
    )
    return pd.Series({'total_valid': total_valid, 'hover_text': hover_str})

# 6. Aggregate hover text
agg_df = value_counts_df.groupby(['Country', 'variable']).apply(build_translated_value_hover).reset_index()

# 7. Pivot into matrices for Plotly
z_matrix = agg_df.pivot(index='Country', columns='variable', values='total_valid').fillna(0)
hover_matrix = agg_df.pivot(index='Country', columns='variable', values='hover_text').fillna("No Data Available")

# 8. Render Heatmap
fig3 = px.imshow(
    z_matrix,
    labels=dict(x="Feature / Column", y="Country", color="Valid Response Count"),
    x=z_matrix.columns,
    y=z_matrix.index,
    color_continuous_scale="Plasma",
    title="Survey Response Counts per Country & Feature (Hover for Labeled Response Breakdown)"
)

fig3.update_traces(
    hovertemplate="<b>Country:</b> %{y}<br><b>Feature:</b> %{x}<br><br>%{customdata}<extra></extra>",
    customdata=hover_matrix.values
)

# Explicitly display every single x-axis column label
fig3.update_xaxes(
    tickmode='linear',
    dtick=1,
    tickangle=-45,
    automargin=True
)

fig3.update_layout(
    height=max(600, len(z_matrix.index) * 25),
    width=max(1200, len(z_matrix.columns) * 25),  # Generous width per feature column
    margin=dict(l=150, b=200, t=80, r=50)
)

HTML(fig3.to_html(include_plotlyjs='cdn'))








#%%
# save the plot to an HTML file
fig3.write_html('country_vs_feature_response_counts.html')

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


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

output_path = "ess_filtered_dataset.csv"
df = pd.read_csv(output_path)
# lets drop the following columns 
#  'Gender_of_second_person_in_household', 'Gender_of_third_person_in_household', 'Gender_of_fourth_person_in_household', 'Gender_of_fifth_person_in_household', 'Gender_of_sixth_person_in_household'
df = df.drop(columns=['Gender_of_second_person_in_household', 'Gender_of_third_person_in_household', 'Gender_of_fourth_person_in_household', 'Gender_of_fifth_person_in_household', 'Gender_of_sixth_person_in_household'])


# Step 1: Define target and feature columns (EXCLUDES Country)
target_col = 'Gender'



#%%
# Step 0: Ensure target variable has NO NaNs upfront
clean_df = df[df[target_col].notna()].copy()

# Step 1: Define feature columns (excluding target & metadata)
exclude_cols = [
    'Country', 
    'ESS_round', 
    'nan_count', 
    'Citizenship', 
    'Language_most_often_spoken_at_home:_second_mentioned', 
    'Region', 
    target_col
]
feature_cols = [col for col in clean_df.columns if col not in exclude_cols]

# Step 2: Determine minimum row count across clean Country/Round combos
country_round_counts = clean_df.groupby(['Country', 'ESS_round']).size()
min_rows_per_combo = country_round_counts.min()
train_rows_per_combo = int(min_rows_per_combo * 0.8)

# Step 3: Sample training data (preserving index)
training_data = (
    clean_df.groupby(['Country', 'ESS_round'], group_keys=False)
    .apply(lambda x: x.sample(n=train_rows_per_combo, random_state=42))
)

# Step 4: Drop training rows to create test set
test_data = clean_df.drop(index=training_data.index).copy()

# Step 5: Extract X and y
X_train = training_data[feature_cols]
y_train = training_data[target_col]

X_test = test_data[feature_cols]
y_test = test_data[target_col]




#%%
X_train.columns.to_list()

#%%
# Step 6: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)



# Step 7: Predict on unseen test data & store predictions in test_data
test_data['y_pred'] = rf_model.predict(X_test)

# Store predicted probabilities for female/male (useful for ROC-AUC)
if hasattr(rf_model, "predict_proba"):
    test_data['y_prob'] = rf_model.predict_proba(X_test)[:, 1]

#%%


# Step 8: Calculate gender similarity metric (model accuracy) per country
def evaluate_country_similarity(group):
    acc = accuracy_score(group[target_col], group['y_pred'])
    
    # Calculate ROC-AUC if both genders exist in the test slice
    try:
        auc = roc_auc_score(group[target_col], group['y_prob'])
    except (ValueError, KeyError):
        auc = np.nan
        
    return pd.Series({
        'test_n': len(group),
        'accuracy': acc,
        'roc_auc': auc
    })

gender_similarity_by_country = (
    test_data.groupby('Country')
    .apply(evaluate_country_similarity)
    .reset_index()
    .sort_values(by='accuracy', ascending=True)  # Lowest accuracy = highest gender similarity
)

# Display results
print("\n--- Gender Similarity Ranking (Lower Accuracy = More Similar Genders) ---")
print(gender_similarity_by_country.to_string(index=False))

# %%
# Check Pearson correlation between sample size and model accuracy
corr = gender_similarity_by_country['accuracy'].corr(gender_similarity_by_country['test_n'])
print(f"Correlation between test_n and accuracy: {corr:.3f}")




# %%
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score


# Overall metrics calculation
y_true = test_data[target_col]
y_pred = test_data['y_pred']
y_prob = test_data['y_prob'] if 'y_prob' in test_data.columns else None

# Calculate point metrics
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan

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
print(classification_report(y_true, y_pred, digits=4))



# %%
# Compute metrics broken down by Country and ESS_round
def compute_round_metrics(group):
    y_t = group[target_col]
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
    .apply(compute_round_metrics)
    .reset_index()
    .sort_values(by=['Country', 'ESS_round'])
)

# Save the Country x Round metrics table to CSV
csv_filename = "ess_country_and_round_performance.csv"
country_round_perf.to_csv(csv_filename, index=False)

print(f"✅ Country/Round metrics saved to {csv_filename}")
# %%






# %%
import plotly.express as px
import plotly.io as pio

# Optional: Set default template to avoid dark/black theme conflicts in VS Code
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
html_filename = "gender_similarity_trends.html"
fig.write_html(html_filename, include_plotlyjs='cdn')
print(f"✅ Interactive plot saved to {html_filename}")

# 2. Display directly in notebook using native Plotly renderer
# Use renderer="vscode" if running inside VS Code, or renderer="notebook" for standard Jupyter
try:
    fig.show(renderer="notebook")
except Exception:
    fig.show()
# %%

# %% Test for Longitudinal Decrease in Accuracy Across Countries
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Fit Mixed-Effects Model (Accounts for repeated measures within countries)
# ESS_round fixed effect + Country random intercept
mixed_model = smf.mixedlm("accuracy ~ ESS_round", country_round_perf, groups=country_round_perf["Country"])
mixed_res = mixed_model.fit()

print("==================================================")
print("     GLOBAL ACCURACY TREND OVER TIME (MIXED LM)   ")
print("==================================================")
print(mixed_res.summary())

round_slope = mixed_res.params['ESS_round']
p_val = mixed_res.pvalues['ESS_round']

if round_slope < 0:
    print(f"\n✅ REPLICATION SUCCESS: Overall accuracy decreases by {abs(round_slope)*100:.2f}% per round (p = {p_val:.4f}).")
else:
    print(f"\n⚠️ NO DECREASE: Overall accuracy changes by {round_slope*100:.2f}% per round (p = {p_val:.4f}).")


# %% 2. Compute Per-Country Slopes Safely
def compute_country_slope(group):
    # Need at least 2 distinct ESS rounds to calculate a slope
    if len(group['ESS_round'].unique()) < 2:
        return pd.Series({
            'slope_per_round': np.nan, 
            'p_value': np.nan, 
            'first_round_acc': group['accuracy'].iloc[0], 
            'latest_round_acc': group['accuracy'].iloc[-1],
            'total_change': 0.0
        })
    
    try:
        X = sm.add_constant(group['ESS_round'])
        y = group['accuracy']
        model = sm.OLS(y, X).fit()
        
        # Check if ESS_round is in params (handles constant/collinear cases)
        slope = model.params['ESS_round'] if 'ESS_round' in model.params else np.nan
        p_val = model.pvalues['ESS_round'] if 'ESS_round' in model.pvalues else np.nan
        
        return pd.Series({
            'slope_per_round': slope,
            'p_value': p_val,
            'first_round_acc': group['accuracy'].iloc[0],
            'latest_round_acc': group['accuracy'].iloc[-1],
            'total_change': group['accuracy'].iloc[-1] - group['accuracy'].iloc[0]
        })
    except Exception:
        return pd.Series({
            'slope_per_round': np.nan, 
            'p_value': np.nan, 
            'first_round_acc': group['accuracy'].iloc[0], 
            'latest_round_acc': group['accuracy'].iloc[-1],
            'total_change': np.nan
        })

# Compute slopes
country_slopes = (
    country_round_perf.sort_values('ESS_round')
    .groupby('Country', group_keys=False)
    .apply(compute_country_slope)
    .reset_index()
)

# Sort safely after column creation is guaranteed
country_slopes = country_slopes.sort_values(by='slope_per_round', ascending=True, na_position='last')

print("\n--- Country Slopes (Negative = Decreasing Accuracy / Higher Similarity) ---")
print(country_slopes.to_string(index=False))

# Summary metrics
valid_slopes = country_slopes.dropna(subset=['slope_per_round'])
decreasing_countries = (valid_slopes['slope_per_round'] < 0).sum()
total_countries = len(valid_slopes)

if total_countries > 0:
    print(f"\nCountries with decreasing accuracy over time: {decreasing_countries} / {total_countries} ({decreasing_countries/total_countries:.1%})")
# %% 3. Visualizing Country Slopes
fig_slopes = px.bar(
    country_slopes.dropna(),
    x='Country',
    y='slope_per_round',
    color='slope_per_round',
    color_continuous_scale='RdYlBu', # Blue = negative slope (decreasing accuracy), Red = positive
    title="<b>Change in Gender Predictability Accuracy per ESS Round</b><br><sup>Negative values indicate gender role convergence over time</sup>",
    labels={'slope_per_round': 'Accuracy Change per Round (% points)', 'Country': 'Country'}
)

fig_slopes.add_hline(y=0, line_dash="dash", line_color="black")
fig_slopes.update_layout(height=500, width=1000, yaxis_tickformat=".2%")
fig_slopes.show()

# %%
html_filename = "change_in_accuracy_attempt3.html"
fig_slopes.write_html(html_filename, include_plotlyjs='cdn')
print(f"✅ Interactive plot saved to {html_filename}")







# %% Calculate and Visualize Feature Importances
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Extract importances and map to feature names
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Add cumulative importance percentage
importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()

# 2. Categorize features (Base vs. Missingness Indicators)
def get_feature_category(feat):
    if feat.endswith('_is_na'):
        return 'Not Applicable Flag (_is_na)'
    elif feat.endswith('_is_missing'):
        return 'Other Missing Flag (_is_missing)'
    return 'Base Survey Question'

importance_df['Category'] = importance_df['Feature'].apply(get_feature_category)

# 3. Print Top 25 Most Important Features
print("==================================================")
print("       TOP 25 MOST IMPORTANT FEATURES (GINI)      ")
print("==================================================")
print(importance_df.head(25).to_string(index=False))

# %% 4. Render Interactive Horizontal Bar Chart for Top 20 Features
top_20 = importance_df.head(20).sort_values(by='Importance', ascending=True)

fig_imp = px.bar(
    top_20,
    x='Importance',
    y='Feature',
    color='Category',
    orientation='h',
    title="<b>Top 20 Features Driving Gender Predictability</b><br><sup>Measured via Gini Importance in Random Forest</sup>",
    labels={'Importance': 'Normalized Gini Importance', 'Feature': 'Survey Question / Indicator'},
    color_discrete_map={
        'Base Survey Question': '#1f77b4',
        'Not Applicable Flag (_is_na)': '#ff7f0e',
        'Other Missing Flag (_is_missing)': '#d62728'
    }
)

fig_imp.update_layout(
    height=650,
    width=1100,
    margin=dict(l=250, r=50, t=100, b=50),
    legend=dict(title="Feature Type", y=0.15, x=0.65)
)

fig_imp.show()