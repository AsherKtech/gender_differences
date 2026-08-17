#%%# ==============================================================================
# ESS PIPELINE: RAW MISSINGNESS CODES (NO IMPUTATION, NATIVE MISSING SPLITS)
# ==============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go





#%%
# ==============================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA
# ==============================================================================
print("Step 1: Loading raw SPSS file and metadata...")
FILE_PATH = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)





#%%
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}
cntry_val_labels = meta.variable_value_labels.get('cntry', {})

gender_raw_col = next((c for c in ['gndr', 'gender'] if c in df_raw.columns), 'gndr')





#%%
# Filter target upfront (keep only valid Male=1, Female=2)
valid_mask = df_raw[gender_raw_col].isin([1, 2, 1.0, 2.0])
df_sub = df_raw[valid_mask].copy()





#%%
# ==============================================================================
# STEP 2: FIND COLUMNS WITH COVERAGE ACROSS ALL COUNTRY/ROUND COMBOS
# ==============================================================================
print("Step 2: Identifying columns present across all Country/Round pairs...")
ALL_MISSING_CODES = {
    6, 66, 666, 6666, 6.0, 66.0, 666.0, 6666.0, '6', '66', '666', '6666',
    7, 77, 777, 7777, 7.0, 77.0, 777.0, 7777.0, '7', '77', '777', '7777',
    8, 88, 888, 8888, 8.0, 88.0, 888.0, 8888.0, '8', '88', '888', '8888',
    9, 99, 999, 9999, 9.0, 99.0, 999.0, 9999.0, '9', '99', '999', '9999'
}

group_cols = ['cntry', 'essround']
candidate_cols = [c for c in df_sub.columns if c not in group_cols]

def is_strictly_valid(s):
    return s.notna() & (~s.isin(ALL_MISSING_CODES))

valid_per_group = df_sub.groupby(group_cols)[candidate_cols].apply(
    lambda group: group.apply(lambda col: is_strictly_valid(col).any())
)
retained_cols = valid_per_group.columns[valid_per_group.all()].tolist()
df_subset = df_sub[group_cols + retained_cols].copy()





#%%
# ==============================================================================
# STEP 3: RENAME COLUMNS & STANDARDIZE METADATA HEADERS
# ==============================================================================
print("Step 3: Renaming headers and dropping excluded metadata...")
df_subset.rename(columns=code_to_label, inplace=True)

cntry_renamed = code_to_label.get(group_cols[0], group_cols[0])
round_renamed = code_to_label.get(group_cols[1], group_cols[1])
gender_renamed = code_to_label.get(gender_raw_col, gender_raw_col)

df_subset['Country'] = df_subset[cntry_renamed].astype(str).str.strip().map(cntry_val_labels).fillna(df_subset[cntry_renamed])
df_subset['ESS_round'] = df_subset[round_renamed]
df_subset['Gender'] = df_subset[gender_renamed]





#%%
cols_to_drop = [
     group_cols[0], group_cols[1], gender_raw_col,
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
    "Year_of_birth_of_third_person_in_household"
]
df_subset.drop(columns=[c for c in cols_to_drop if c in df_subset.columns], errors='ignore', inplace=True)





#%%
# ==============================================================================
# STEP 4: PRESERVE RAW MISSING CODES (NO IMPUTATION, CONVERT MISSING TO NA)
# ==============================================================================
print("Step 4: Mapping missing codes to NaN for native tree splits (Zero Imputation)...")
df_processed = df_subset.copy()
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'Gender']]

for c in feature_cols:
    s = df_processed[c]
    is_missing_code = s.isin(ALL_MISSING_CODES)
    num_s = pd.to_numeric(s, errors='coerce')
    num_s[is_missing_code] = np.nan
    df_processed[c] = num_s

le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['Gender'].astype(str))





#%%
output_csv = "ess_processed_raw_missingness.csv"
df_processed.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_processed.shape} to '{output_csv}'")





#%%
# ==============================================================================
# STEP 5: BALANCED SAMPLING & TRAIN/TEST SPLIT
# ==============================================================================
print("Step 5: Performing balanced stratified train/test split...")
counts = df_processed.groupby(['Country', 'ESS_round']).size()
train_n = int(counts.min() * 0.8)

training_data = df_processed.groupby(['Country', 'ESS_round'], group_keys=False).apply(
    lambda x: x.sample(n=min(train_n, len(x)), random_state=42), include_groups=False
)
test_data = df_processed.drop(index=training_data.index).copy()

X_train = training_data[feature_cols]
y_train = training_data['target_encoded']
X_test = test_data[feature_cols]
y_test = test_data['target_encoded']





#%%
# ==============================================================================
# STEP 6: MODEL TRAINING (HIST-GRADIENT BOOSTING HAS NATIVE MISSING SPLITS)
# ==============================================================================
print("Step 6: Fitting HistGradientBoostingClassifier (Native NaN splitting, No Imputation)...")
hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_model.fit(X_train, y_train)

test_data['y_pred'] = hgb_model.predict(X_test)
test_data['y_prob'] = hgb_model.predict_proba(X_test)[:, 1]

def eval_group(g):
    acc = accuracy_score(g['target_encoded'], g['y_pred'])
    f1 = precision_recall_fscore_support(g['target_encoded'], g['y_pred'], average='binary', zero_division=0)[2]
    return pd.Series({'test_n': len(g), 'accuracy': acc, 'f1_score': f1})

perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()
perf_df.to_csv("ess_country_and_round_performance_raw.csv", index=False)

mixed_res = smf.mixedlm("accuracy ~ ESS_round", perf_df, groups=perf_df["Country"]).fit()
print("\n==================================================")
print("     GLOBAL ACCURACY TREND OVER TIME (MIXED LM)   ")
print("==================================================")
print(mixed_res.summary())





#%%
# ==============================================================================
# STEP 7: ENHANCED VISUALIZATIONS
# ==============================================================================
print("\nStep 7: Generating plots...")

# ------------------------------------------------------------------------------
# 1. ACCURACY TREND PLOT WITH 5-GROUP CATEGORY SELECTOR
# ------------------------------------------------------------------------------
perf_df['ESS_round'] = perf_df['ESS_round'].astype(int)

def calculate_slope(group):
    if len(group['ESS_round'].unique()) < 2:
        return 0.0
    try:
        X = sm.add_constant(group['ESS_round'])
        model = sm.OLS(group['accuracy'], X).fit()
        return model.params['ESS_round'] if 'ESS_round' in model.params else 0.0
    except Exception:
        return 0.0

country_slopes = perf_df.groupby('Country').apply(calculate_slope, include_groups=False).reset_index(name='slope')

groups_order = [
    'Steepest Decrease (--)', 
    'Slight Decrease (-)', 
    'Neutral / Stable (0)', 
    'Slight Increase (+)', 
    'Steepest Increase (++)'
]

slope_bins = pd.qcut(country_slopes['slope'], q=5, labels=groups_order)
country_slopes['group'] = slope_bins

perf_grouped = perf_df.merge(country_slopes[['Country', 'group']], on='Country')

fig_trend = go.Figure()
palette = px.colors.qualitative.Bold

group_trace_indices = {}
current_trace_idx = 0

for grp in groups_order:
    sub_df = perf_grouped[perf_grouped['group'] == grp]
    countries_in_grp = sub_df['Country'].unique()
    
    start_idx = current_trace_idx
    for i, cntry in enumerate(countries_in_grp):
        cntry_df = sub_df[sub_df['Country'] == cntry].sort_values('ESS_round')
        color = palette[i % len(palette)]
        
        fig_trend.add_trace(go.Scatter(
            x=cntry_df['ESS_round'],
            y=cntry_df['accuracy'],
            mode='lines+markers',
            name=cntry,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>Country:</b> {cntry}<br><b>Round:</b> %{{x}}<br><b>Accuracy:</b> %{{y:.2%}}<extra></extra>",
            visible=(grp == 'Steepest Decrease (--)')
        ))
        current_trace_idx += 1
        
    end_idx = current_trace_idx
    group_trace_indices[grp] = list(range(start_idx, end_idx))

total_traces = current_trace_idx

buttons = []
for grp in groups_order:
    visible_mask = [False] * total_traces
    for idx in group_trace_indices[grp]:
        visible_mask[idx] = True
        
    buttons.append(dict(
        label=grp,
        method="update",
        args=[
            {"visible": visible_mask},
            {"title": f"<b>Gender Predictability Trends (Raw Native Splits): Group {grp}</b><br><sup>Lower Accuracy = Higher Gender Similarity</sup>"}
        ]
    ))

fig_trend.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.0,
        xanchor="left",
        y=1.22,
        yanchor="top"
    )],
    title="<b>Gender Predictability Trends (Raw Native Splits): Group Steepest Decrease (--)</b><br><sup>Lower Accuracy = Higher Gender Similarity</sup>",
    xaxis=dict(title="ESS Survey Round", dtick=1, tickmode='linear'),
    yaxis=dict(title="Model Accuracy", tickformat=".0%", range=[0.40, 0.95]),
    height=650,
    width=1100,
    hovermode="x unified",
    margin=dict(l=80, r=120, t=120, b=80)
)





#%%
fig_trend.add_hline(y=0.50, line_dash="dash", line_color="gray", annotation_text="Random Guess (50%)", annotation_position="bottom right")
fig_trend.write_html("gender_similarity_trends_raw.html", include_plotlyjs='cdn')
print("✅ Grouped trend plot saved to 'gender_similarity_trends_raw.html'")





#%%
# ------------------------------------------------------------------------------
# 2. FEATURE IMPORTANCE PLOT (FAST EVALUATION ON SAMPLED TEST SET)
# ------------------------------------------------------------------------------
print("Calculating fast permutation feature importances...")

X_eval = X_test.sample(n=min(10000, len(X_test)), random_state=42)
y_eval = y_test.loc[X_eval.index]

perm_imp = permutation_importance(
    hgb_model, 
    X_eval, 
    y_eval, 
    n_repeats=2, 
    random_state=42, 
    n_jobs=1
)

imp_df = pd.DataFrame({
    'Feature': feature_cols, 
    'Importance': perm_imp.importances_mean
}).sort_values('Importance', ascending=False)

top_20_df = imp_df.head(20).sort_values('Importance', ascending=True)

fig_imp = px.bar(
    top_20_df, 
    x='Importance', 
    y='Feature', 
    orientation='h', 
    title="<b>Top 20 Features (Raw Missingness & Native Tree Splits)</b>",
    color_discrete_sequence=['#1f77b4']
)

fig_imp.update_yaxes(
    type='category',
    autorange='reversed',
    tickmode='linear',
    dtick=1,
    automargin=True
)

fig_imp.update_layout(
    height=700,
    width=1150,
    margin=dict(l=350, r=50, t=100, b=50)
)





#%%
fig_imp.write_html("feature_importance_raw.html", include_plotlyjs='cdn')
print("✅ Feature importance plot saved to 'feature_importance_raw.html'")
print("\n🎉 Execution complete!")
# %%
