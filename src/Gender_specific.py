"""
====================================================================================================
ESS PIPELINE: AGE PREDICTABILITY CONTROL EXPERIMENT (HIST-GRADIENT BOOSTING & NATIVE SPLITS)
====================================================================================================

OVERVIEW:
This script implements a machine learning pipeline to predict age group membership (Young 18-34 vs Older 50-63) 
using ESS (European Social Survey) data. The core hypothesis is that if demographic attributes become less 
discriminative of age, the model's accuracy decreases - indicating greater cross-generational similarity in 
sociodemographic profiles.

Key Methodological Features:
- HistGradientBoostingClassifier: Native handling of missing values without imputation
- Dual Missing Indicators: _is_na (Not Applicable) and _is_missing flags for comprehensive NA tracking
- Stratified Sampling: Balanced train/test splits across Country+ESS_round combinations
- Permutation Importance: Feature selection based on actual model contribution assessment

Dataset Source: ESS1e06 through ESS9e03 (9 rounds, 38 European countries)
Target: Binary classification of age group membership

EXECUTION OUTPUT (from successful run):
========================================
Step 1: Loading raw SPSS file, setting age thresholds, and building discrete groups...
Targeting ~50% random chance: Matched Young [18-34] (N=103537) against Older [50-63]
Number of responses in Young age group [18-34]: 103537
Number of responses in Older age group [50-63]: 100427
Total number of valid age entries (18+): 430870
Filtered dataset size: 203964 rows across 38 countries.

Step 2: Identifying columns present across all Country/Round pairs...
2340 numeric candidate cols, 23 string candidate cols
Validation completed in 1.80 seconds

Step 3: Renaming headers and dropping excluded metadata, structural proxies, and age leakage...
Dropped 17 leakage/proxy/metadata columns:
- Respondent's_identification_number, Year_of_birth, Year_last_in_paid_job
- Doing_last_7_days:_housework,_looking_after_children,_others
- Age_of_respondent,_calculated, Design_weight, Post-stratification_weight...
- Year_of_birth_of_*, Partner_doing_last_7_days:*, Ever_had_children_*

Step 4: Constructing _is_na and _is_missing indicator flags...
Total 'Not Applicable' (NA) values across all features: 2138522
Total other missing values: 1708125
✅ Processed dataset saved with shape (203964, 322) to 'ess_processed_age_experiment.csv'

Verifying age group composition in df_processed...
Unique AgeGroup values found: ['Older (50-63)', 'Young (18-34)']
✅ Age group composition verified: only 'Young (18-34)' and 'Older (50-63)' present.
   - No respondents aged 40 or 70 (or any other ages outside [18,34] ∪ [50,63]).

Step 5: Performing balanced stratified train/test split...

Step 6: Fitting HistGradientBoostingClassifier for Age Prediction...
Selected 137 informative features, dropped 181 uninformative ones.

Step 7: Running exports and building plots...
✅ Feature importances exported to 'age_features_importances_sorted.csv' 
   and plot saved to 'age_total_feature_importance.html'.
✅ Unified plot saved to 'age_similarity_trends_unified.html' (21 countries included).
✅ Group-filtered trend plot saved to 'age_similarity_trends_indicators_grouped.html'.

OUTPUT FILES GENERATED:
=======================
- ess_processed_age_experiment.csv (180 MB): Main processed dataset with 322 columns
- age_features_importances_sorted.csv: Sorted feature importances from permutation analysis
- age_total_feature_importance.html: Interactive bar plot of feature importances
- age_similarity_trends_unified.html: Unified trend plot across 21 countries (≥6 rounds)
- age_similarity_trends_indicators_grouped.html: Grouped interactive plot with dropdown filters
- ess_country_and_round_performance_age_experiment.csv: Per-country performance metrics
- age_dropped_features.txt: List of 181 dropped features (Importance <= 0)
- age_retained_features.txt: List of 137 retained features (Importance > 0)

MODULE ORGANIZATION:
====================
1. HELPER FUNCTIONS (lines ~30-250):
   - export_separate_feature_importance(): Permutation importance calculation and visualization
   - create_unified_all_countries_plot(): Unified trend plot across eligible countries
   - create_grouped_trend_plot(): Grouped interactive plot with slope-based country classification
   - eval_group(): Performance evaluation helper for group-wise metrics
   - calculate_slope(): OLS slope calculation for trend analysis

2. STEP 1: DATA LOADING & AGE GROUP FILTERING (lines ~250-380):
   - Load raw SPSS file with metadata
   - Determine optimal age cutoff for balanced class distribution
   - Create AgeGroup labels: 'Young (18-34)' and 'Older (50-63)'
   
3. STEP 2: COLUMN COVERAGE VALIDATION (lines ~380-450):
   - Identify columns present across all Country/Round combinations
   - Validate numeric columns using C++ backend (group_valid)
   - Validate string columns using pandas
   - Build list of retained columns meeting coverage threshold
   
4. STEP 3: DATA SUBSETTING & RENAMING (lines ~450-520):
   - Create subset with Group_cols and retained features
   - Apply metadata column name mappings
   - Drop explicit leakage/proxy/metadata columns
   - Dynamic pattern matching for age/year/retirement keywords
   
5. STEP 4: MISSING VALUE HANDLING (lines ~520-600):
   - Construct dual missing indicators (_is_na and _is_missing)
   - Zero-fill missing values in original columns
   - Track totals: 2,138,522 NA values + 1,708,125 other missing values
   
6. STEP 5: TRAIN/TEST SPLIT (lines ~600-640):
   - Stratified sampling within Country+ESS_round combinations
   - 80% training, 20% testing with balanced representation
   - Data integrity verification for mutual exclusivity
   
7. STEP 6: MODEL TRAINING & SELECTION (lines ~640-690):
   - Initial permutation importance calculation on baseline model
   - Feature selection: retain features with positive importance
   - Retrain final model on selected features only
   - Evaluate per Country+ESS_round combination
   
8. STEP 7: VISUALIZATIONS & EXPORTS (lines ~690-720):
   - Export feature importance CSV and HTML plot
   - Generate unified trend plot for countries with ≥6 rounds
   - Generate grouped interactive plot with slope-based classification

KEY VARIABLES AND THEIR MEANINGS:
=================================
- best_xx: Optimal upper age limit for Older group (50-63 in final run)
- ALL_MISSING_CODES: Set of codes representing various missing value types
- NOT_APPLICABLE_CODES: Subset representing 'Not Applicable' responses
- X_eval, y_eval: Subsampled evaluation data for feature importance
- selected_features: Features with positive permutation importance (137 features)
- dropped_features: Features with zero or negative importance (181 features)
- perf_df: DataFrame with per-country, per-round accuracy and F1 scores

====================================================================================================
"""


#%%
import time  # For timing execution performance of data processing steps
from pathlib import Path  # For handling file paths in a platform-independent way
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing with arrays and mathematical operations
import pyreadstat  # Read SPSS (.sav) files with metadata support

# Machine Learning imports from scikit-learn
from sklearn.ensemble import HistGradientBoostingClassifier  # Gradient boosting for classification with native missing value handling
from sklearn.preprocessing import LabelEncoder  # Encode target labels with values between 0 and n_classes-1
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # Model evaluation metrics
from sklearn.inspection import permutation_importance  # Compute feature importance via permutation

# Statistical modeling imports for trend analysis
import statsmodels.api as sm  # Statistics and econometrics library
import statsmodels.formula.api as smf  # Formula-based model specification
import plotly.express as px  # Interactive visualization library (high-level)
import plotly.graph_objects as go  # Plotly's low-level API for custom plots

# Custom module for efficient group-wise validity checking using C++ backend
import group_valid


#%%
# ==================================================================================================
# HELPER FUNCTIONS
# ==================================================================================================
def export_separate_feature_importance(model, X_eval, y_eval, selected_features, output_csv="age_features_importances_sorted.csv", output_html="age_total_feature_importance.html"):
    """
    Computes permutation importance on evaluation data, exports sorted CSV, and saves HTML bar plot.
    
    This function evaluates how much each feature contributes to the age prediction model by randomly
    shuffling each feature's values and measuring the resulting decrease in accuracy. Features that
    cause larger performance drops when shuffled are considered more important.
    
    The output includes:
    - A CSV file with features sorted by importance (descending)
    - An interactive HTML bar plot color-coded by feature category:
      * Base Survey Question (blue): Original survey variables
      * Not Applicable Flag (orange): Missing value indicators for NA responses
      * Other Missing Flag (red): Missing value indicators for other missing data
    
    Parameters:
    -----------
    model : HistGradientBoostingClassifier
        The trained model to evaluate feature importance on
    X_eval : pd.DataFrame
        Evaluation features matrix
    y_eval : pd.Series
        True labels for evaluation
    selected_features : list of str
        List of feature names that were retained after selection
    output_csv : str, optional
        Path to save the sorted importance CSV (default: "age_features_importances_sorted.csv")
    output_html : str, optional
        Path to save the interactive HTML plot (default: "age_total_feature_importance.html")
    
    Returns:
    --------
    imp_df_sorted : pd.DataFrame
        DataFrame with Feature, Importance, and Category columns, sorted by importance ascending
    """
    """
    Computes permutation importance on evaluation data, exports sorted CSV, and saves HTML bar plot.
    """
    print("\nEvaluating and exporting feature importance...")
    perm_imp = permutation_importance(model, X_eval, y_eval, n_repeats=5, random_state=42, n_jobs=1)

    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': perm_imp.importances_mean})
    imp_df['Category'] = imp_df['Feature'].apply(
        lambda name: 'Not Applicable Flag' if name.endswith('_is_na') else ('Other Missing Flag' if name.endswith('_is_missing') else 'Base Survey Question')
    )

    imp_df_sorted = imp_df.sort_values('Importance', ascending=True)

    fig_imp = px.bar(
        imp_df_sorted,
        x='Importance',
        y='Feature',
        color='Category',
        orientation='h',
        title="<b>Retained Features Importance - Age Prediction Model</b>",
        template='plotly_white',
        color_discrete_map={
            'Base Survey Question': '#1f77b4',
            'Not Applicable Flag': '#ff7f0e',
            'Other Missing Flag': '#d62728'
        }
    )

    fig_imp.update_yaxes(type='category', tickmode='linear', dtick=1, automargin=True, showgrid=True, gridcolor='#E5E5E5')
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

    imp_df_sorted[['Feature', 'Importance']].to_csv(output_csv, index=False)
    print(f"✅ Feature importances exported to '{output_csv}' and plot saved to '{output_html}'.")
    return imp_df_sorted


def create_unified_all_countries_plot(perf_df, output_html="countries_age_similarity_trends_unified.html", min_rounds=1):
    """
    Generates a single unified Plotly line plot containing countries meeting a minimum round threshold,
    with an overall average trend line and a grand mean horizontal reference line.
    """
    print(f"\nGenerating unified all-countries plot for Age (filtering for countries with ≥ {min_rounds} rounds)")
    
    round_counts = perf_df.groupby('Country')['ESS_round'].nunique()
    eligible_countries = round_counts[round_counts >= min_rounds].index
    filtered_df = perf_df[perf_df['Country'].isin(eligible_countries)].copy()

    fig = go.Figure()

    # 1. Individual country traces (muted opacity to keep focus on average trend)
    for cntry, cntry_df in filtered_df.groupby('Country'):
        cntry_sorted = cntry_df.sort_values('ESS_round')
        fig.add_trace(go.Scatter(
            x=cntry_sorted['ESS_round'],
            y=cntry_sorted['accuracy'],
            mode='lines+markers',
            name=cntry,
            opacity=0.75,
            hovertemplate=f"<b>Country:</b> {cntry}<br><b>Round:</b> %{{x}}<br><b>Accuracy:</b> %{{y:.2%}}<extra></extra>"
        ))

    # 2. Calculate and add average accuracy trend line across eligible countries per round
    avg_accuracy_by_round = (
        filtered_df.groupby('ESS_round')['accuracy']
        .mean()
        .reset_index()
        .sort_values('ESS_round')
    )

    fig.add_trace(go.Scatter(
        x=avg_accuracy_by_round['ESS_round'],
        y=avg_accuracy_by_round['accuracy'],
        mode='lines+markers',
        name='<b>Average Accuracy</b>',
        line=dict(color='black', width=2),
        marker=dict(size=4, color='black'),
        hovertemplate="<b>AVERAGE ACCURACY</b><br><b>Round:</b> %{x}<br><b>Accuracy:</b> %{y:.2%}<extra></extra>"
    ))

    # 3. Calculate overall mean accuracy across all rounds and eligible countries
    overall_avg_accuracy = filtered_df['accuracy'].mean()

    # 4. Reference lines layout updates
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        title=f"<b>Age Predictability Trends Across Countries (≥ {min_rounds} rounds)</b><br><sup>Young (18-34) vs. Older Group | Lower Accuracy = Higher Cross-Generational Similarity</sup>",
        xaxis=dict(title="ESS Survey Round", dtick=1, tickmode='linear', showgrid=False, zeroline=False),
        yaxis=dict(title="Model Accuracy", tickformat=".0%", range=[0.65, 0.90], showgrid=False, zeroline=False),
        height=700,
        width=1200,
        hovermode="x unified",
        margin=dict(l=80, r=120, t=100, b=80)
    )

    # # Random chance baseline (50%)
    # fig.add_hline(
    #     y=0.50, 
    #     line_dash="dash", 
    #     line_color="gray", 
    #     annotation_text="Random Guess (50%)", 
    #     annotation_position="bottom right"
    # )

    # Horizontal grand mean reference line
    fig.add_hline(
        y=overall_avg_accuracy, 
        line_dash="dot", 
        line_color="firebrick", 
        line_width=2,
        annotation_text=f"Overall Mean ({overall_avg_accuracy:.2%})", 
        annotation_position="top right",
        annotation_font_color="firebrick"
    )

    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"✅ Unified plot saved to '{output_html}' ({len(eligible_countries)} countries included).")
    print(f"   Overall Average Accuracy: {overall_avg_accuracy:.4f}")



def create_grouped_trend_plot(perf_df, output_html="age_similarity_trends_indicators_grouped.html", min_rounds=1):
    """
    Generates an interactive Plotly line plot with dropdown buttons filtering countries 
    by their rate of change (OLS slope across rounds), applying an optional round threshold filter.
    """
    print(f"\nGenerating interactive group-filtered trend plot for Age (min {min_rounds} rounds)...")
    
    df_plot = perf_df.copy()
    df_plot['ESS_round'] = df_plot['ESS_round'].astype(int)

    round_counts = df_plot.groupby('Country')['ESS_round'].nunique()
    eligible_countries = round_counts[round_counts >= min_rounds].index
    df_plot = df_plot[df_plot['Country'].isin(eligible_countries)].copy()

    country_slopes = df_plot.groupby('Country').apply(calculate_slope, include_groups=False).reset_index(name='slope')

    groups_order = [
        'Steepest Decrease (--)',
        'Slight Decrease (-)',
        'Neutral / Stable (0)',
        'Slight Increase (+)',
        'Steepest Increase (++)'
    ]

    country_slopes['group'] = pd.qcut(country_slopes['slope'], q=min(5, len(country_slopes)), labels=groups_order[:min(5, len(country_slopes))], duplicates='drop')
    perf_grouped = df_plot.merge(country_slopes[['Country', 'group']], on='Country')

    fig_trend = go.Figure()
    palette = px.colors.qualitative.Bold
    group_trace_indices = {}
    current_trace_idx = 0

    active_groups = [g for g in groups_order if g in perf_grouped['group'].values]

    for grp in active_groups:
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
                visible=(grp == active_groups[0])
            ))
            current_trace_idx += 1
        
        group_trace_indices[grp] = list(range(start_idx, current_trace_idx))

    buttons = []
    for grp in active_groups:
        visible_mask = [False] * current_trace_idx
        for idx in group_trace_indices[grp]:
            visible_mask[idx] = True
            
        buttons.append(dict(
            label=grp,
            method="update",
            args=[
                {"visible": visible_mask},
                {"title": f"<b>Age Predictability Trends: Group {grp}</b><br><sup>Lower Accuracy = Higher Cross-Generational Similarity</sup>"}
            ]
        ))

    default_title_grp = active_groups[0] if active_groups else "N/A"
    fig_trend.update_layout(
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
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
        title=f"<b>Age Predictability Trends: Group {default_title_grp}</b><br><sup>Lower Accuracy = Higher Cross-Generational Similarity</sup>",
                xaxis=dict(
            title="ESS Survey Round",
            dtick=1,
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            ticks='outside',
            ticklen=10,
            tickwidth=1,
            tickcolor='black'
        ),
        yaxis=dict(
            title="Model Accuracy",
            tickformat=".0%",
            range=[0.40, 0.95],
            showgrid=False,
            zeroline=False,
            ticks='outside',
            ticklen=10,
            tickwidth=1,
            tickcolor='black'
        ),        height=650,
        width=1100,
        hovermode="x unified",
        margin=dict(l=80, r=120, t=120, b=80)
    )

    fig_trend.add_hline(y=0.50, line_dash="dash", line_color="gray", annotation_text="Random Guess (50%)", annotation_position="bottom right")
    fig_trend.write_html(output_html, include_plotlyjs='cdn')
    print(f"✅ Group-filtered trend plot saved to '{output_html}'.")


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


#%%
# ==================================================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA AND FILTER BY AGE GROUPS
# ==================================================================================================
print("Step 1: Loading raw SPSS file, setting age thresholds, and building discrete groups...")

FILE_PATH = Path('/data/home/asher.katz/Projects/gender_differences/data/raw/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav')
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)



#%%
raw_labels = meta.column_names_to_labels
code_to_label = {col: label.replace(" ", "_") for col, label in raw_labels.items()}
cntry_val_labels = meta.variable_value_labels.get('cntry', {})

age_raw_col = next((c for c in ['agea', 'age'] if c in df_raw.columns), 'agea')
df_raw['age_numeric'] = pd.to_numeric(df_raw[age_raw_col], errors='coerce')



#%%
# ==================================================================================================
# STEP 1: SET MANUAL AGE BRACKETS FOR FULL PIPELINE TESTING
# ==================================================================================================
START_AGE = 37  # Change to test: 35, 36, 37, 38, 39, 40, 41, etc.
END_AGE   = 51  # Match with table above (e.g., 35-48, 36-50, 37-51, 41-56)

print(f"Step 1: Setting test age brackets [18-34] vs [{START_AGE}-{END_AGE}]...")
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

n_young = (df_sub['AgeGroup'] == 'Young (18-34)').sum()
n_older = (df_sub['AgeGroup'] == f'Older ({START_AGE}-{END_AGE})').sum()

print(f"Sample distribution for [{START_AGE}-{END_AGE}]:")
print(f" - Young (18-34): {n_young:,}")
print(f" - Older ({START_AGE}-{END_AGE}): {n_older:,} ({n_older/(n_young+n_older):.1%} split)")
print(f"Total dataset size: {len(df_sub):,} rows")






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
string_cols = dtypes[~dtypes.apply(pd.api.types.is_numeric_dtype)].index.tolist()
numeric_cols = [c for c in candidate_cols if c not in string_cols]

print(f"{len(numeric_cols)} numeric candidate cols, {len(string_cols)} string candidate cols")


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


#%%
# ---- Validity check: string columns via pandas ----
string_missing_codes = {str(c) for c in ALL_MISSING_CODES} | {'', ' '}

valid_mask_str = df_sub[string_cols].apply(is_strictly_valid_str)
valid_per_group_str = valid_mask_str.groupby(
    [df_sub[group_cols[0]], df_sub[group_cols[1]]]
).any()
valid_per_group_str = valid_per_group_str.reindex(group_labels)
print(f"Validation completed in {time.time() - t0:.2f} seconds")



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
existing_drops = [c for c in all_drops if c in df_subset.columns]

df_subset.drop(columns=existing_drops, inplace=True)
print(f"Dropped {len(existing_drops)} leakage/proxy/metadata columns: {existing_drops}")






#%%
# ==================================================================================================
# STEP 4: HANDLE MISSING VALUES (DUAL INDICATORS & ZERO-FILLING)
# ==================================================================================================
print("Step 4: Constructing _is_na and _is_missing indicator flags...")

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
print(f"Total other missing values: {total_missing}")

df_processed = pd.DataFrame(transformed)

le = LabelEncoder()
df_processed['target_encoded'] = le.fit_transform(df_processed['AgeGroup'].astype(str))

output_csv = "ess_processed_age_experiment.csv"
df_processed.to_csv(output_csv, index=False)
print(f"✅ Processed dataset saved with shape {df_processed.shape} to '{output_csv}'")




# #%%
# # Verify that only two age groups exist: 18-34 and 50-63 (i.e., no respondents aged 40 or 70)
# print("\nVerifying age group composition in df_processed...")

# age_group_values = df_processed['AgeGroup'].unique()
# print(f"Unique AgeGroup values found: {list(age_group_values)}")

# # Extract numeric ranges from the labels
# valid_groups = ['Young (18-34)', 'Older (50-63)']
# if set(age_group_values) != set(valid_groups):
#     raise ValueError(
#         f"❌ Unexpected age group composition! Expected exactly: {valid_groups}, "
#         f"but found: {list(age_group_values)}"
#     )

# # Double-check by inspecting original numeric ages in df_raw (if still available)
# # Note: We'll use the fact that AgeGroup was built from age_numeric
# young_mask_check = (df_sub['age_numeric'] >= 18) & (df_sub['age_numeric'] <= 34)
# older_mask_check = (df_sub['age_numeric'] >= 50) & (df_sub['age_numeric'] <= best_xx)

# # Ensure no respondents aged 40 or 70 exist in the final df_sub
# ages_to_avoid = [40, 70]
# for age in ages_to_avoid:
#     count_age = ((df_sub['age_numeric'] >= age) & (df_sub['age_numeric'] < age + 1)).sum()
#     if count_age > 0:
#         raise ValueError(
#             f"❌ Found {count_age} respondents aged {age}, which violates the constraint. "
#             "Please adjust age group boundaries."
#         )

# # Confirm that all remaining ages fall strictly within [18,34] or [50,best_xx]
# ages_in_data = df_sub['age_numeric'].dropna()
# out_of_bounds = ages_in_data[
#     ~((ages_in_data >= 18) & (ages_in_data <= 34)) &
#     ~((ages_in_data >= 50) & (ages_in_data <= best_xx))
# ]

# if len(out_of_bounds) > 0:
#     raise ValueError(
#         f"❌ Found {len(out_of_bounds)} respondents with ages outside allowed ranges "
#         f"(18-34 or 50-{best_xx}). Sample out-of-range values: {out_of_bounds.head(5).tolist()}"
#     )

# print("✅ Age group composition verified: only 'Young (18-34)' and 'Older (50-63)' present.")
# print(f"   - No respondents aged 40 or 70 (or any other ages outside [18,34] ∪ [50,{best_xx}]).")


#%%
# ==================================================================================================
# STEP 5: BALANCED SAMPLING & TRAIN/TEST SPLIT
# ==================================================================================================
print("Step 5: Performing balanced stratified train/test split...")

# Determine training sample size based on smallest group (minimum country-round combination)
counts = df_processed.groupby(['Country', 'ESS_round']).size()
train_n = int(counts.min() * 0.8)

# Stratified sampling within each Country+ESS_round combination to ensure balanced representation
# This ensures that each demographic group contributes proportionally to both train and test sets,
# preventing bias toward overrepresented groups while maintaining the age distribution structure.
training_data = df_processed.groupby(['Country', 'ESS_round'], group_keys=False).apply(
    lambda x: x.sample(n=min(train_n, len(x)), random_state=42), include_groups=False
)

# Extract remaining samples for test set (data not used in training)
test_data = df_processed.drop(index=training_data.index).copy()

# Define feature columns: exclude identifiers and target variables to prevent data leakage
feature_cols = [c for c in df_processed.columns if c not in ['Country', 'ESS_round', 'AgeGroup', 'target_encoded']]

# Split into training and testing matrices with proper column alignment
X_train = training_data[feature_cols]
y_train = training_data['target_encoded']
X_test = test_data[feature_cols]
y_test = test_data['target_encoded']

# Verify no overlap between train and test indices (data integrity check)
assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test sets must be mutually exclusive"


#%%
# ==================================================================================================
# STEP 6: MODEL TRAINING & SELECTION
# ==================================================================================================
print("Step 6: Fitting HistGradientBoostingClassifier for Age Prediction...")

# Feature selection via initial Permutation Importance
# Using a fast baseline model to identify features that contribute positively to prediction
# This step filters out uninformative features before final model training, improving efficiency
baseline_hgb = HistGradientBoostingClassifier(random_state=42)
baseline_hgb.fit(X_train, y_train)

# Subsample for faster permutation importance calculation (computationally expensive operation)
# Uses minimum of 10,000 samples or entire training set if smaller
X_select_sample = X_train.sample(n=min(10000, len(X_train)), random_state=42)
y_select_sample = y_train.loc[X_select_sample.index]

# Compute permutation importance: measures feature contribution by randomly shuffling each feature
# and measuring the resulting decrease in model accuracy. Higher importance = more dependent on feature.
perm_selection = permutation_importance(
    baseline_hgb, X_select_sample, y_select_sample, n_repeats=5, random_state=42, n_jobs=1
)

# Create DataFrame of features and their average importance scores across permutations
importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': perm_selection.importances_mean})

# Select only features with positive importance (contribute to prediction)
selected_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()

# Track dropped features for reporting and downstream analysis
dropped_features = importance_df[importance_df['Importance'] <= 0]['Feature'].tolist()

print(f"Selected {len(selected_features)} informative features, dropped {len(dropped_features)} uninformative ones.")


#%%
# Retrain on retained features only (reduces overfitting risk and improves generalization)
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train final model with selected features
hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_model.fit(X_train_selected, y_train)

# Generate predictions on test set: class labels and probabilities for AgeGroup=1 (Young 18-34)
test_data['y_pred'] = hgb_model.predict(X_test_selected)
test_data['y_prob'] = hgb_model.predict_proba(X_test_selected)[:, 1]

# Evaluate model performance per Country+ESS_round combination using eval_group function
# This captures how well the model generalizes across different survey waves and countries
perf_df = test_data.groupby(['Country', 'ESS_round']).apply(eval_group, include_groups=False).reset_index()

# Export country-round level performance metrics for downstream analysis and trend visualization
perf_df.to_csv("ess_country_and_round_performance_age_experiment.csv", index=False)




#%%


# Calculate average accuracy across all countries for each round
avg_accuracy_by_round = perf_df.groupby('ESS_round')['accuracy'].mean().reset_index()
avg_accuracy_by_round.columns = ['ESS_round', 'average_accuracy']

# Display the results in a clean table format
print("\n==================================================")
print("   AVERAGE AGE ACCURACY ACROSS ALL COUNTRIES BY ROUND ")
print("==================================================")
# First, compute overall accuracy across all rounds and countries
overall_accuracy = perf_df['accuracy'].mean()
print(f"\nOverall Accuracy Across All Rounds & Countries: {overall_accuracy:.4f}")

# Then display per-round averages
for _, row in avg_accuracy_by_round.iterrows():
    print(f"Round {int(row['ESS_round'])}: {row['average_accuracy']:.4f}")

# Optionally, save to CSV for further analysis
avg_accuracy_by_round.to_csv("ess_average_accuracy_by_round76.csv", index=False)





#%%
# ==================================================================================================
# STEP 7: VISUALIZATIONS & EXPORTS
# ==================================================================================================
print("\nStep 7: Running exports and building plots...")
# Subsample evaluation data for feature importance computation (faster, representative)
X_eval = X_test_selected.sample(n=min(10000, len(X_test_selected)), random_state=42)
y_eval = y_test.loc[X_eval.index]


#%%
# Export step 7 visualizations
export_separate_feature_importance(hgb_model, X_eval, y_eval, selected_features, output_csv="age_features_importances_sorted76.csv", output_html="age_total_feature_importance76.html" )



#%%
# Generate unified plot showing age similarity trends across all countries with >=6 ESS rounds
# This provides a comprehensive overview of model performance consistency over time
create_unified_all_countries_plot(perf_df, "age_similarity_trends_unified76.html", min_rounds=6)



#%%
# Generate grouped plot categorizing countries by demographic indicators (gender balance, etc.)
# This enables comparison across country groups with similar sociodemographic profiles
create_grouped_trend_plot(perf_df, 'age_similarity_trends_indicators_grouped76.html')



#%%
# Save dropped and retained features to text files for documentation and reproducibility
with open("age_dropped_features.txt", "w") as f:
    f.write("Dropped Features (Importance <= 0):\n")
    for feat in dropped_features:
        f.write(f'"{feat}",\n')

with open("age_retained_features.txt", "w") as f:
    f.write("Retained Features (Importance > 0):\n")
    for feat in selected_features:
        f.write(f'"{feat}",\n')
