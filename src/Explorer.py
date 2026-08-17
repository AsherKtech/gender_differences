# %% cell 1
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    log_loss, 
    confusion_matrix, 
    roc_auc_score
)


#%% cell 2
# Auto-detect working directory path
if os.path.exists("data/train_features.csv"):
    base_dir = "."
else:
    base_dir = "Projects/gender_differences"

print(f"✅ Environment initialized. Base directory set to: '{base_dir}'")

#%% cell 3
print("🔄 Loading datasets...")

X_train = pd.read_csv(f"{base_dir}/data/train_features.csv", index_col=0)
# Keep as a Series to preserve index alignment
y_train = pd.read_csv(f"{base_dir}/data/train_target.csv", index_col=0).iloc[:, 0]

X_val = pd.read_csv(f"{base_dir}/data/val_features.csv", index_col=0)
# Keep as a Series to preserve index alignment
y_val = pd.read_csv(f"{base_dir}/data/val_target.csv", index_col=0).iloc[:, 0]

# Load original master subset
df_orig = pd.read_csv(f"{base_dir}/data/processed/subset_data.csv", low_memory=False)

print(f"📊 Train features shape: {X_train.shape} | Val features shape: {X_val.shape}")

#%% cell 4
# Calculate imputation signature for X_train
top_value_pct = X_train.apply(lambda x: x.value_counts(normalize=True).iloc[0] if not x.empty else 0)
imputation_signature = pd.DataFrame({
    'Most_Frequent_Value': X_train.mode().iloc[0],
    'Concentration_Percentage': top_value_pct * 100
})

imputation_signature
# # Drop columns with 100% identical values (non-boolean columns only)
cols_to_drop = imputation_signature[
    ((imputation_signature['Concentration_Percentage'] == 100.0))
].index
cols_to_drop
# # Drop these columns from X_train and X_val
# X_train = X_train.drop(columns=cols_to_drop)
# X_val = X_val.drop(columns=cols_to_drop)

# print(f"✅ Dropped {len(cols_to_drop)} columns with 100% identical values.")

#%% cell 5
print("🧠 Training Random Forest Classifier (using 40 cores)...")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=40)
rf.fit(X_train, y_train)

print("✅ Model training completed successfully.")

#%% cell 6
print("🔗 Aligning country and essround metadata to validation indices...")

X_val['cntry'] = df_orig.loc[X_val.index, 'cntry']
X_val['essround'] = df_orig.loc[X_val.index, 'essround']

print("✅ Metadata alignment complete.")


#%% cell 7
country_map = {
    'AL': 'Albania', 'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 
    'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany', 
    'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FI': 'Finland', 
    'FR': 'France', 'GB': 'United Kingdom', 'GR': 'Greece', 'HR': 'Croatia', 
    'HU': 'Hungary', 'IE': 'Ireland', 'IL': 'Israel', 'IS': 'Iceland', 'IT': 'Italy', 
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'ME': 'Montenegro', 
    'MK': 'North Macedonia', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 
    'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russian Federation', 
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'TR': 'Turkey', 
    'UA': 'Ukraine', 'XK': 'Kosovo'
}

results = []
grouped = X_val.groupby(['cntry', 'essround'])
total_groups = len(grouped)

print(f"📈 Processing evaluation metrics across {total_groups} country/wave subsets...")

for (cntry, essround), group_features in grouped:
    if len(group_features) == 0:
        continue
        
    # Safely slice targets directly by matching index labels
    group_indices = group_features.index
    y_group_raw = y_val.loc[group_indices].values
    
    # Isolate features (dropping the metadata columns)
    X_group_clean = group_features.drop(['cntry', 'essround'], axis=1)
    y_pred_raw = rf.predict(X_group_clean)
    y_prob_raw = rf.predict_proba(X_group_clean)

    # ---------------------------------------------------------
    # DYNAMIC CLASS MAPPING (Fixes the 1/2 vs 0/1 Survey Coding)
    # ---------------------------------------------------------
    # Drop survey sentinel/missing values (9 represents missing/no answer)
    valid_mask = np.isin(y_group_raw, [1, 2]) 
    if not np.any(valid_mask):
        continue
        
    y_group_clean = y_group_raw[valid_mask]
    y_pred_clean = y_pred_raw[valid_mask]
    y_prob = y_prob_raw[valid_mask, 1]

    # Map categories strictly: 1 (Male) -> 0, 2 (Female) -> 1
    class_map = {1: 0, 2: 1}
    
    # Map both ground truth AND predictions so they align perfectly!
    y_group = np.vectorize(class_map.get)(y_group_clean).astype(int)
    y_pred = np.vectorize(class_map.get)(y_pred_clean).astype(int)

    # Skip groups that end up too tiny after sentinel filtering
    if len(y_group) < 5:
        continue

    # Core Metric Calculations
    accuracy = accuracy_score(y_group, y_pred)
    precision = precision_score(y_group, y_pred, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_group, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_group, y_pred, average='binary', pos_label=1, zero_division=0)
    
# Log Loss calculation (Extract and re-normalize only the valid binary columns)
    try:
        # Identify which columns in rf.classes_ correspond to 1 and 2
        class_1_idx = np.where(rf.classes_ == 1)[0][0]
        class_2_idx = np.where(rf.classes_ == 2)[0][0]
        
        # Extract just those two columns
        probs_binary = y_prob_raw[valid_mask][:, [class_1_idx, class_2_idx]]
        
        # Re-normalize so they sum to 1.0 (prob of class 0 + prob of class 1 = 1)
        probs_binary = probs_binary / probs_binary.sum(axis=1, keepdims=True)
        
        logloss = log_loss(y_group, probs_binary, labels=[0, 1])
    except Exception:
        logloss = np.nan

    # Specificity and Sensitivity using Confusion Matrix
    cm = confusion_matrix(y_group, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # AUC Calculation
    try:
        auc = roc_auc_score(y_group, y_prob)
    except ValueError:
        auc = np.nan

    results.append({
        'country_code': cntry,
        'country_name': country_map.get(cntry, 'Unknown Country'),
        'essround': int(essround),
        'sample_size': len(y_group),
        'accuracy': accuracy,
        'precision': precision,
        'recall_sensitivity': recall,
        'f1_score': f1,
        'log_loss': logloss,
        'specificity': specificity,
        'auc': auc
    })

print("✅ Group evaluation complete.")

#%% cell 8
df_metrics = pd.DataFrame(results)

# Sort alphabetically by country name, then chronologically by round
df_metrics = df_metrics.sort_values(by=['country_name', 'essround']).reset_index(drop=True)

# Save output
output_path = f"{base_dir}/data/processed/comprehensive_cultural_metrics.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_metrics.sort_values(by=['country_name', 'essround']).to_csv(output_path, index=False)

print("🎉 Analysis completed successfully!")
print(f"📁 Master Metrics CSV saved to: {output_path}")
print(f"📊 Total country-wave combinations processed: {len(df_metrics)}")
print("\n🏆 Top 5 Cohorts by AUC:")

top_auc = df_metrics.dropna(subset=['auc']).nlargest(5, 'auc')
# Using display() in Jupyter renders a beautifully formatted table
display(top_auc[['country_name', 'essround', 'sample_size', 'accuracy', 'auc']])


#%% cell 9  
# 1. Use X_train's index to extract the corresponding rows from the original dataframe
train_metadata = df_orig.loc[X_train.index, ['cntry', 'essround']]

# 2. Now we can group and count safely
train_counts = train_metadata.groupby(['cntry', 'essround']).size().reset_index(name='train_sample_size')

# 3. Map country codes to full names
train_counts['country_name'] = train_counts['cntry'].map(lambda x: country_map.get(x, 'Unknown Country'))

# Reorder for clean display
train_counts = train_counts[['cntry', 'country_name', 'essround', 'train_sample_size']]

# View largest training distributions
import pandas as pd
display(train_counts.sort_values(by='train_sample_size', ascending=False))

#%% cell 10
train_counts.to_csv("/data/home/asher.katz/Projects/gender_differences/data/processed/counts_of_train_data.csv")

#%% cell 11
import pandas as pd
import numpy as np

# 1. Calculate explicit NaN counts in the training features
nan_counts = X_train.isna().sum()
nan_percentages = (X_train.isna().sum() / len(X_train)) * 100

# 2. Check for common survey missingness codes (e.g., 7, 8, 9 or negative values)
# Note: Adjust this list depending on how your specific features are scaled
suspected_sentinels = [7, 8, 9, 99, -1, -9]
sentinel_counts = X_train.isin(suspected_sentinels).sum()
sentinel_percentages = (X_train.isin(suspected_sentinels).sum() / len(X_train)) * 100

# 3. Combine into a Quality Report DataFrame
data_quality_report = pd.DataFrame({
    'Explicit_NaN_Count': nan_counts,
    'NaN_Percentage': nan_percentages,
    'Sentinel_Code_Count': sentinel_counts,
    'Sentinel_Percentage': sentinel_percentages
})

# Filter to show columns that actually have missingness or sentinels
active_missingness = data_quality_report[
    (data_quality_report['Explicit_NaN_Count'] > 0) | 
    (data_quality_report['Sentinel_Code_Count'] > 0)
]

print(f"📊 Out of {X_train.shape[1]} total features, {len(active_missingness)} contain missing indicators.")
display(active_missingness.sort_values(by='NaN_Percentage', ascending=False).head(20))


#%% cell 12
# Run this if your model is part of a Scikit-Learn Pipeline object
if 'pipeline' in locals() or 'pipeline' in globals():
    print("🛠️ Preprocessing Steps:")
    for step in pipeline.steps:
        print(f" - {step[0]}: {type(step[1]).__name__}")
else:
    print("📋 Model is evaluated standalone. Check your prior feature-engineering notebook for the SimpleImputer or KnnImputer steps.")


#%% cell 13
# Optional: Save the trained model for future inference
imputation_signature.sort_values(by='Concentration_Percentage', ascending=False)[1600:1630]


#%% cell 14
import plotly.express as px

# Create a DataFrame with the required columns
accuracy_df  = df_metrics[['country_name',  'essround',  'accuracy',  'precision',  'recall_sensitivity',  'f1_score',  'log_loss',  'specificity',  'auc']] 

# Define custom color scale
color_scale = ['rgb(0,0,255)', 'rgb(0,255,0)'] # dark green to lime green
#006400
# # Create a scatter plot with country__name on x- axis and essround on y-axis, colored by accuracy
fig  = px.scatter(accuracy_df, x='country_name', y='essround', color='accuracy',
                 hover_data=['precision',  'recall_sensitivity',  'f1_score',  'log_loss',  'specificity',  'auc'],
                 color_continuous_scale="Plasma")

# Update the layout to make it more readable
fig.update_layout(template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  font=dict(color='white'),
                  title='Accuracy of each country_ name/essround combination',
                  xaxis_title='Country Name',
                  yaxis_title='ESS Round')

# Show the plot
fig.show()


#%% cell 15
import plotly.express as px

# Create a DataFrame with the required columns
accuracy_df  = df_metrics[['country_name',  'essround',  'accuracy',  'precision',  'recall_sensitivity',  'f1_score',  'log_loss',  'specificity',  'auc']] 

# Define custom color scale
color_scale = ['rgb(0,0,255)', 'rgb(0,255,0)'] # dark green to lime green
#006400
# # Create a scatter plot with country__name on x- axis and essround on y-axis, colored by accuracy
fig  = px.scatter(accuracy_df, x='country_name', y='essround', color='accuracy',
                 hover_data=['precision',  'recall_sensitivity',  'f1_score',  'log_loss',  'specificity',  'auc'],
                 color_continuous_scale="Plasma")

# Update the layout to make it more readable
fig.update_layout(template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  font=dict(color='white'),
                  title='Accuracy of each country_ name/essround combination',
                  xaxis_title='Country Name',
                  yaxis_title='ESS Round')

# Show the plot
fig.show()

# %% cell 16
# Save the figure as an HTML file in /data/home/asher.katz/Projects/gender_differences/plots

output_dir  =  "/data/home/asher.katz/Projects/gender_differences/plots"
os.makedirs(output_dir, exist_ok=True)

fig.write_html(os.path.join(output_dir,'accuracy_plot.html'))