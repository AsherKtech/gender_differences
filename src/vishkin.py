#%% ==============================================================================
# ESS PIPELINE: SURVEY ADMINISTRATION MODE PREDICTION & GENERALIZATION (LIGHTGBM)
# Based on Vishkin & Bkheet (2025)
# ==============================================================================
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% ==============================================================================
# STEP 1: LOAD RAW SAV DATA & METADATA
# ==============================================================================
print("Step 1: Loading raw SPSS files for ESS Rounds 8, 9, and 10...")
# Target dataset includes ESS Rounds 8-10 across 29 countries
FILE_PATH = Path('/data/home/asher.katz/Projects/ess_mode_effects/data/raw/ESS_R8_R9_R10_subset.sav')
df_raw, meta = pyreadstat.read_sav(FILE_PATH, user_missing=True)

# Define country groups by administration mode in Round 10
SELF_COMPLETION_COUNTRIES = ['AT', 'CY', 'DE', 'ES', 'IL', 'LV', 'PL', 'RS', 'SE'] # 9 countries
INTERVIEW_COUNTRIES = [
    'BE', 'BG', 'HR', 'CZ', 'EE', 'FI', 'FR', 'GB', 'HU', 'IE', 
    'IS', 'IT', 'LT', 'ME', 'NL', 'NO', 'PT', 'SI', 'SK', 'CH'
] # 20 countries

# Exclude countries lacking prior round data (e.g., Greece, North Macedonia)
df_sub = df_raw[df_raw['cntry'].isin(SELF_COMPLETION_COUNTRIES + INTERVIEW_COUNTRIES)].copy()

#%% ==============================================================================
# STEP 2: SELECT CORE VARIABLES ACROSS ALL ROUNDS
# ==============================================================================
print("Step 2: Filtering features present across all rounds and harmonizing...")
# Criteria:
# 1. Present in all three rounds (8, 9, 10)
# 2. Exclude country-specific items (e.g., prtvtcat)
# 3. Retain continuous/ordinal scales; exclude categorical/binary variables
# 4. Exclude variables with structural missingness or missing-data country confounds
# Result: 46 core survey variables retained

CORE_46_FEATURES = [
    # List of 46 core ordinal/continuous survey variables
    'ppltrst', 'pplfair', 'pplhlp', 'polintr', 'trstprl', 'trstlgl', 'trstplc',
    'trstplt', 'trstprt', 'trstep', 'trstun', 'stflife', 'stfeco', 'stfgov',
    'stfdem', 'freehms', 'hmsfllr', 'hmsacp', 'sclact', 'sclmeet', 'inscdrs',
    # ... (remaining core continuous/ordinal features)
]

df_subset = df_sub[['cntry', 'essround'] + CORE_46_FEATURES].copy()

#%% ==============================================================================
# STEP 3: DEFINE TARGET CLASSES & MODES
# ==============================================================================
print("Step 3: Defining class assignments (SC-10, SC-89, IV-10, IV-89)...")

def assign_mode_class(row):
    cntry, rnd = row['cntry'], row['essround']
    if cntry in SELF_COMPLETION_COUNTRIES:
        return 'SC-10' if rnd == 10 else 'SC-89'
    else:
        return 'IV-10' if rnd == 10 else 'IV-89'

df_subset['mode_class'] = df_subset.apply(assign_mode_class, axis=1)

#%% ==============================================================================
# STEP 4: PREPROCESS MISSING VALUES & SCALE RECODING (RQ3 EXPERIMENTS)
# ==============================================================================
print("Step 4: Setting up feature sets for raw values and absolute distance recoding...")

# Note: Missing values are left as NaN for LightGBM's native tree-branching algorithm
X_raw = df_subset[CORE_46_FEATURES].copy()

# Recode feature transformation for RQ3: Absolute Distance from scale mid-point
# Val_recoded = | (Min + Max) / 2 - Val |
X_abs_dist = pd.DataFrame(index=X_raw.index)

for col in CORE_46_FEATURES:
    if col == 'nwspol':
        # Unbounded variable uses distance from mean instead of midpoint
        col_mean = X_raw[col].mean()
        X_abs_dist[f"{col}_abs_dist"] = (X_raw[col] - col_mean).abs()
    else:
        min_val, max_val = X_raw[col].min(), X_raw[col].max()
        midpoint = (min_val + max_val) / 2.0
        X_abs_dist[f"{col}_abs_dist"] = (X_raw[col] - midpoint).abs()

# Combined feature set (Raw + Absolute Distance)
X_combined = pd.concat([X_raw, X_abs_dist], axis=1)

#%% ==============================================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLITTING (RQ1, RQ2 & RQ3)
# ==============================================================================
print("Step 5: Defining dataset split strategies...")

# Global Main Analysis Split (80% Seen, 20% Unseen holdout)
# Seen data is split into 80% Train (64% total) and 20% Validation (16% total)
train_seen, test_unseen = train_test_split(
    df_subset, test_size=0.20, stratify=df_subset[['cntry', 'mode_class']], random_state=42
)
train_data, val_data = train_test_split(
    train_seen, test_size=0.20, stratify=train_seen[['cntry', 'mode_class']], random_state=42
)

#%% ==============================================================================
# STEP 6: LIGHTGBM MODEL TRAINING & HYPERPARAMETER SETUP
# ==============================================================================
print("Step 6: Configuring LightGBM (GOSS sampling & native missing handling)...")

lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss', # Gradient-based One-Side Sampling
    'enable_bundle': True,           # Exclusive Feature Bundling (EFB)
    'use_missing': True,             # Native tree handling for NaNs
    'metric': 'binary_logloss',
    'random_state': 42,
    'verbose': -1
}

def evaluate_lgbm(X_tr, y_tr, X_te, y_te, params):
    """Utility function to fit LightGBM and evaluate Accuracy and F1 Score."""
    train_set = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, train_set)
    
    preds_prob = model.predict(X_te)
    preds_binary = (preds_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_te, preds_binary)
    _, _, f1, _ = precision_recall_fscore_support(y_te, preds_binary, average='binary', zero_division=0)
    
    # Calculate imbalanced chance/random accuracy
    pos_ratio = y_te.mean()
    random_acc = max(pos_ratio, 1 - pos_ratio)
    
    return acc, f1, random_acc, model

#%% ==============================================================================
# STEP 7: EXECUTE EXPERIMENTAL EVALUATIONS (RQ1, RQ2, RQ3)
# ==============================================================================
print("Step 7: Executing classification tasks across scenarios...")

# ------------------------------------------------------------------------------
# RESEARCH QUESTION 1: Predictability of Administration Mode across all countries
# ------------------------------------------------------------------------------
print("\n--- RQ1: Global Model (SC-10 vs SC-89) ---")
sc_train = train_data[train_data['mode_class'].isin(['SC-10', 'SC-89'])]
sc_test = test_unseen[test_unseen['mode_class'].isin(['SC-10', 'SC-89'])]

y_tr_rq1 = (sc_train['mode_class'] == 'SC-10').astype(int)
y_te_rq1 = (sc_test['mode_class'] == 'SC-10').astype(int)

acc_rq1, f1_rq1, rand_rq1, model_rq1 = evaluate_lgbm(
    X_raw.loc[sc_train.index], y_tr_rq1, X_raw.loc[sc_test.index], y_te_rq1, lgb_params
)
print(f"RQ1 Test Acc: {acc_rq1:.3f} | F1: {f1_rq1:.3f} | Random Acc: {rand_rq1:.3f}")

# Time/Round Control Model (IV-10 vs IV-89)
iv_train = train_data[train_data['mode_class'].isin(['IV-10', 'IV-89'])]
iv_test = test_unseen[test_unseen['mode_class'].isin(['IV-10', 'IV-89'])]

y_tr_ctrl = (iv_train['mode_class'] == 'IV-10').astype(int)
y_te_ctrl = (iv_test['mode_class'] == 'IV-10').astype(int)

acc_ctrl, f1_ctrl, rand_ctrl, _ = evaluate_lgbm(
    X_raw.loc[iv_train.index], y_tr_ctrl, X_raw.loc[iv_test.index], y_te_ctrl, lgb_params
)
print(f"Control (Time) Test Acc: {acc_ctrl:.3f} | F1: {f1_ctrl:.3f} | Random Acc: {rand_ctrl:.3f}")

# ------------------------------------------------------------------------------
# RESEARCH QUESTION 2: Generalization to a Novel Country (Leave-One-Country-Out)
# ------------------------------------------------------------------------------
print("\n--- RQ2: Cross-Country Generalization (Leave-One-Country-Out) ---")
loco_results = []

for target_cntry in SELF_COMPLETION_COUNTRIES:
    train_loco = df_subset[(df_subset['cntry'] != target_cntry) & (df_subset['mode_class'].isin(['SC-10', 'SC-89']))]
    test_loco = df_subset[(df_subset['cntry'] == target_cntry) & (df_subset['mode_class'].isin(['SC-10', 'SC-89']))]
    
    y_tr_loco = (train_loco['mode_class'] == 'SC-10').astype(int)
    y_te_loco = (test_loco['mode_class'] == 'SC-10').astype(int)
    
    acc, f1, rand, _ = evaluate_lgbm(
        X_raw.loc[train_loco.index], y_tr_loco, X_raw.loc[test_loco.index], y_te_loco, lgb_params
    )
    loco_results.append({'Country': target_cntry, 'Accuracy': acc, 'F1': f1, 'Random_Acc': rand})

df_loco = pd.DataFrame(loco_results)
print(f"Mean LOCO Generalization Accuracy: {df_loco['Accuracy'].mean():.3f} | F1: {df_loco['F1'].mean():.3f}")

# ------------------------------------------------------------------------------
# RESEARCH QUESTION 3: Extreme Response Style (ERS) Analysis
# ------------------------------------------------------------------------------
print("\n--- RQ3: Extreme Response Style Impact (Absolute Distance Recoding) ---")

# Model using ONLY absolute distance features
acc_ers, f1_ers, _, _ = evaluate_lgbm(
    X_abs_dist.loc[sc_train.index], y_tr_rq1, X_abs_dist.loc[sc_test.index], y_te_rq1, lgb_params
)
print(f"Abs-Distance Model Acc: {acc_ers:.3f} | F1: {f1_ers:.3f}")

# Model using COMBINED features (Raw + Absolute Distance)
acc_comb, f1_comb, _, _ = evaluate_lgbm(
    X_combined.loc[sc_train.index], y_tr_rq1, X_combined.loc[sc_test.index], y_te_rq1, lgb_params
)
print(f"Combined Features Model Acc: {acc_comb:.3f} | F1: {f1_comb:.3f}")

print("\n Execution complete!")