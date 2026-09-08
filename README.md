# Project README: Gender & Demographic Predictability Trends in the European Social Survey (ESS)

## Overview
This repository hosts a data science and machine learning research pipeline designed to evaluate cross-cultural demographic predictability trends across multiple rounds of the European Social Survey (ESS). 

The primary analysis fits high-performance Gradient Boosting models (`HistGradientBoostingClassifier`) to predict respondent demographic attributes (such as gender or age group) from complex, high-dimensional survey responses. Model holdout **accuracy** serves as an empirical proxy for **demographic role similarity**: lower prediction accuracy indicates high response similarity across demographics, whereas higher accuracy indicates distinct response patterns between demographic groups.

The pipeline includes:
- **Gender analysis**: Predicts binary gender categories (Male/Female)
- **Age-group analysis**: Predicts age brackets (e.g., Young [18-34] vs Older [37-51])
- **Stability generalization analysis**: Evaluates temporal cross-round stability and model generalization performance by training on early rounds and testing on later rounds (and vice versa) to assess whether gender response patterns are stable or converging over time.

All pipelines use identical methodology but apply to different target variables.

---

## Directory & File Structure Guide

Below is a detailed breakdown of the repository layout, detailing the role of each script, raw data file, generated artifact, and visualization.

.
├── data
│   ├── external
│   │   └── ... (external data sources, if any)
│   ├── processed
│   │   ├── country_rankings.csv
│   │   ├── age_country_rankings.csv
│   │   ├── gender_average_accuracy_by_round.csv
│   │   ├── age_average_accuracy_by_round.csv
│   │   ├── gender_country_and_round_performance_indicators.csv
│   │   ├── age_country_and_round_performance_indicators.csv
│   │   ├── forward_stability_country_and_round_performance_indicators.csv 
│   │   ├── backward_stability_country_and_round_performance_indicators.csv 
│   │   ├── gender_feature_importance.csv
│   │   ├── age_feature_importance.csv
│   │   ├── forward_stability_feature_importance.csv 
│   │   ├── backward_stability_feature_importance.csv 
│   │   ├── gender_full_row_summary.csv
│   │   ├── age_full_row_summary.csv
│   │   ├── stability_full_row_summary.csv 
│   │   ├── forwards_stability_train_row_summary.csv 
│   │   ├── backwards_stability_train_row_summary.csv 
│   │   ├── forwards_stability_test_row_summary.csv 
│   │   ├── backwards_stability_test_row_summary.csv 
│   │   ├── gender_processed_two_missingness_indicators.csv
│   │   ├── age_processed_two_missingness_indicators.csv
│   │   └── stability_processed_two_missingness_indicators.csv 
│   │   ├── gender_test_row_summary.csv
│   │   ├── age_test_row_summary.csv
│   │   └── gender_train_row_summary.csv
│   │   └── age_train_row_summary.csv
│   ├── raw
│   │   ├── ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav
│   │   ├── ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.csv
│   │   └── ESS3e03_7-ESS4e04_6-ESS5e03_6-ESS6e02_7-ESS7e02_3-ESS8e02_3-ESS9e03_3-subset.sav
│   └── temp
│       └── ... (intermediate processing checkpoints and exploration scripts)
├── logs
│   ├── full_output.log
│   ├── gender_difference_analysis_20260830_1730.log
│   ├── gender_difference_analysis_20260830_1745.log
│   ├── age_difference_analysis_*.log
│   ├── stability_generalization_*.log 
│   └── Gender_specific_output.log
├── plots
│   ├── countries_age_similarity_trends_nw_europe.html
│   ├── countries_age_similarity_trends_other.html
│   ├── age_feature_importance_top20.html
│   ├── forward_stability_countries_gender_similarity_trends_nw_europe.html 
│   ├── forward_stability_countries_gender_similarity_trends_other.html 
│   ├── backward_stability_countries_gender_similarity_trends_nw_europe.html 
│   ├── backward_stability_countries_gender_similarity_trends_other.html 
│   ├── forward_stability_feature_importance_top20.html 
│   ├── backward_stability_feature_importance_top20.html 
│   ├── countries_gender_similarity_trends_nw_europe.html
│   ├── countries_gender_similarity_trends_other.html
│   ├── gender_feature_importance_top20.html
│   └── gender_similarity_trends_unified6.html
├── README.md
├── requirements.txt
├── setup_project.py
└── src
    ├── age_difference_analysis.py
    ├── cpp
    │   ├── group_valid.cpp
    │   └── setup.py
    ├── gender_difference_analysis.py
    ├── gender_difference_visualization.py
    ├── minus_one_gender_difference_analysis.py (planned)
    ├── minus_two_gender_difference_analysis.py (planned)
    ├── minus_three_gender_difference_analysis.py (planned)
    ├── stability_generalization.py
    └── stability_generalization_visualization.py

---
## Detailed Output File Specifications

### 1. Source Scripts (`src/`)
* **`age_difference_analysis.py`**: Parallel pipeline extending the methodology to analyze predictability trends across age groups (e.g., Young [18-34] vs Older [37-51]). Performs SPSS data parsing, C++ accelerated group validity filtering across Country × Round strata, dual missingness indicator flag engineering (`_is_na` and `_is_missing`), balanced stratified sampling, feature selection via permutation importance, and model evaluation.
* **`age_difference_visualization.py`**: Dedicated visualization module. Reads age-group performance outputs from `data/processed/` and renders modular interactive Plotly charts, dividing countries into regionally grouped color-mapped line plots (North & Western Europe vs Southern/Central/Eastern Europe + Israel).
* **`gender_difference_analysis.py`**: The core execution pipeline. Performs SPSS data parsing, C++ accelerated group validity filtering across Country × Round strata, missingness indicator flag engineering (`_is_na` and `_is_missing`), balanced sampling, feature selection via permutation importance, and model evaluation.
* **`gender_difference_visualization.py`**: Dedicated visualization module. Reads Gender performance outputs from `data/processed/` and renders modular interactive Plotly charts, dividing countries into regionally grouped color-mapped line plots.
* **`stability_generalization.py`**: Evaluates temporal cross-round stability and model generalization performance. Trains models on early rounds (1-2) to test predictability of later rounds, and trains on recent rounds (8-9) to backcast historical responses. This asymmetry analysis reveals whether gender response patterns are stable or converging over time. **Note: Iceland is excluded from training calculations** due to its very small sample sizes in ESS surveys (typically < 100 respondents). Including Iceland would result in insufficient training samples (~10-15 rows per gender), which is inadequate for training complex models like `HistGradientBoostingClassifier`. Excluding Iceland allows larger countries (Germany, UK, France, etc.) to contribute their full sample sizes.
* **`stability_generalization_visualization.py`**: Dedicated visualization module for stability generalization experiments. Generates interactive Plotly line plots for forward and backward predictability results, dividing countries into North & Western Europe vs Southern/Central/Eastern Europe + Israel with custom color palettes. Also produces feature importance visualizations for both experimental directions.
* **`minus_one_gender_difference_analysis.py`** *(planned)*: Robustness validation script evaluating model performance and trends after dropping the single highest-importance feature to ensure results are not driven by a single dominant question.
* **`minus_two_gender_difference_analysis.py`** *(planned)*: Robustness validation script excluding the top two highest-importance features.
* **`minus_three_gender_difference_analysis.py`** *(planned)*: Robustness validation script excluding the top three highest-importance features.
* **`cpp/group_valid.cpp`**: C++ source module compiled via Python C-API to accelerate validity checks across numeric survey matrices per country and round.

---

### 2. Processed Data Artifacts (`data/processed/`)

#### Age-Group Analysis Files
* **`age_processed_two_missingness_indicators.csv`**: The fully preprocessed dataset for age-group analysis containing all survey response features with zero-filled values, dual missingness indicator columns (``_is_na`` for skip-logic non-applicable responses and ``_is_missing`` for refusals/don't-know/NaN), the `AgeGroup` target column encoding respondents into Young [18-34] or Older [37-51] brackets, and the encoded target variable `target_encoded`. This is the final dataset used for training and evaluation.
* **`age_full_row_summary.csv`**: Comprehensive data quality summary table with total row counts per Country × Round combination, percentage of cells marked as "Not Applicable" (structural survey skips), and percentage of cells marked as "Missing" (refusals or don't-know responses). This provides an overview of missingness patterns in the complete processed age-group dataset.
* **`age_train_row_summary.csv`**: Data quality summary table restricted exclusively to the balanced training dataset split used for model fitting. Contains row counts and missingness percentages per Country × Round combination, enabling comparison of missingness patterns between full data and training subset.
* **`age_test_row_summary.csv`**: Data quality summary table restricted to the held-out testing dataset split for age-group classification. Contains row counts and missingness percentages per Country × Round combination in the test set, allowing validation that test data quality matches training data quality.
* **`age_country_and_round_performance_indicators.csv`**: Model evaluation metrics output from holdout testing containing sample size (`test_n`), accuracy score, and F1 score for each Country × ESS Round stratum. This file enables identification of specific countries or rounds where age predictability is particularly high (indicating distinct response patterns between age groups) or low (indicating similar response patterns across age groups).
* **`age_average_accuracy_by_round.csv`**: Aggregated mean accuracy scores across all participating countries for each ESS Round. This time-series format summarizes how well age groups can be predicted across different survey waves, revealing temporal trends in demographic similarity of responses and potential societal shifts in age-related behavioral patterns.
* **`age_country_rankings.csv`**: Summary ranking table with each country's overall mean prediction accuracy (averaged across all rounds), total number of ESS Rounds in which the country participated, and relative rank position for age-group analysis. This file serves as input to generate custom color mappings for regional visualization plots, facilitating comparative cross-country analyses.
* **`age_feature_importance.csv`**: Feature importance scores computed via permutation importance evaluation. Lists all retained features sorted by descending mean importance score, with categorical labels identifying each feature as a *Base Survey Question*, *Not Applicable Flag* (``_is_na``), or *Other Missing Flag* (``_is_missing``). This reveals which survey questions and missingness patterns most strongly predict age group membership.

#### Gender Analysis Files
* **`gender_processed_two_missingness_indicators.csv`**: The fully preprocessed dataset for gender analysis containing all survey response features with zero-filled values, dual missingness indicator columns (``_is_na`` for skip-logic non-applicable responses and ``_is_missing`` for refusals/don't-know/NaN), the binary `gender` target column encoding respondents as Male or Female, and the encoded target variable `target_encoded`. This is the final dataset used for training and evaluation.
* **`gender_full_row_summary.csv`**: Comprehensive data quality summary table with total row counts per Country × Round combination, percentage of cells marked as "Not Applicable" (structural survey skips), and percentage of cells marked as "Missing" (refusals or don't-know responses). This provides an overview of missingness patterns in the complete processed gender dataset.
* **`gender_train_row_summary.csv`**: Data quality summary table restricted exclusively to the balanced training dataset split used for model fitting. Contains row counts and missingness percentages per Country × Round combination, enabling comparison of missingness patterns between full data and training subset.
* **`gender_test_row_summary.csv`**: Data quality summary table restricted to the held-out testing dataset split for gender classification. Contains row counts and missingness percentages per Country × Round combination in the test set, allowing validation that test data quality matches training data quality.
* **`gender_country_and_round_performance_indicators.csv`**: Model evaluation metrics output from holdout testing containing sample size (`test_n`), accuracy score, and F1 score for each Country × ESS Round stratum. This file enables identification of specific countries or rounds where gender predictability is particularly high (indicating distinct response patterns between men and women) or low (indicating similar response patterns across genders).
* **`gender_average_accuracy_by_round.csv`**: Aggregated mean accuracy scores across all participating countries for each ESS Round. This time-series format summarizes how well gender can be predicted across different survey waves, revealing temporal trends in demographic similarity of responses and potential convergence or divergence of gendered behavioral patterns over time.
* **`country_rankings.csv`**: Summary ranking table with each country's overall mean prediction accuracy (averaged across all rounds), total number of ESS Rounds in which the country participated, and relative rank position for gender analysis. This file serves as input to generate custom color mappings for regional visualization plots, facilitating comparative cross-country analyses.
* **`gender_feature_importance.csv`**: Feature importance scores computed via permutation importance evaluation. Lists all retained features sorted by descending mean importance score, with categorical labels identifying each feature as a *Base Survey Question*, *Not Applicable Flag* (``_is_na``), or *Other Missing Flag* (``_is_missing``). This reveals which survey questions and missingness patterns most strongly predict gender identity.


#### Gender Stability Analysis Files
* **`forward_stability_country_and_round_performance_indicators.csv`**: Model evaluation metrics from the Forward Predictability experiment (training on Rounds 1-2, testing across all rounds) containing sample size (`test_n`), accuracy score, and F1 score for each Country × ESS Round stratum. This reveals temporal stability of gender predictability across survey waves. **Note: Iceland is excluded from training calculations** due to its very small sample sizes in ESS surveys (typically < 100 respondents). Including Iceland would result in insufficient training samples (~10-15 rows per gender), which is inadequate for training complex models like `HistGradientBoostingClassifier`. Excluding Iceland allows larger countries (Germany, UK, France, etc.) to contribute their full sample sizes.
* **`backward_stability_country_and_round_performance_indicators.csv`**: Model evaluation metrics from the Backward Predictability experiment (training on Rounds 8-9, testing across all rounds) containing sample size (`test_n`), accuracy score, and F1 score for each Country × ESS Round stratum. This enables comparison of historical vs modern prediction performance.**Note: Iceland is excluded from training calculations** due to its very small sample sizes in ESS surveys (typically < 100 respondents). Including Iceland would result in insufficient training samples (~10-15 rows per gender), which is inadequate for training complex models like `HistGradientBoostingClassifier`. Excluding Iceland allows larger countries (Germany, UK, France, etc.) to contribute their full sample sizes.
* **`forward_stability_feature_importance.csv`**: Feature importance scores from the Forward Predictability experiment, computed via permutation importance evaluation. Lists all retained features sorted by descending mean importance score with categorical labels identifying each feature type. This reveals which survey questions most strongly predict gender identity in historical contexts.
* **`backward_stability_feature_importance.csv`**: Feature importance scores from the Backward Predictability experiment, computed via permutation importance evaluation. Lists all retained features sorted by descending mean importance score with categorical labels identifying each feature type. This reveals which survey questions most strongly predict gender identity in modern contexts.
* **`stability_full_row_summary.csv`**: Comprehensive row-level summary of processed data for both forward and backward experiments containing country, round, total rows, and missingness statistics for all survey features used in stability generalization analysis.
* **`forwards_stability_train_row_summary.csv`**: Summary table for the training split (Rounds 1-2) in the Forward Predictability experiment. Contains Country, ESS_round, total_number_rows, pct_cells_Non_applicable, and pct_cells_Missing statistics for each country-round stratum.
* **`forwards_stability_test_row_summary.csv`**: Summary table for the testing split (all rounds 1-9) in the Forward Predictability experiment. Contains Country, ESS_round, total_number_rows, pct_cells_Non_applicable, and pct_cells_Missing statistics for each country-round stratum.
* **`backwards_stability_train_row_summary.csv`**: Summary table for the training split (Rounds 8-9) in the Backward Predictability experiment. Contains Country, ESS_round, total_number_rows, pct_cells_Non_applicable, and pct_cells_Missing statistics for each country-round stratum.
* **`backwards_stability_test_row_summary.csv`**: Summary table for the testing split (all rounds 1-9) in the Backward Predictability experiment. Contains Country, ESS_round, total_number_rows, pct_cells_Non_applicable, and pct_cells_Missing statistics for each country-round stratum.
* **`stability_processed_two_missingness_indicators.csv`**: The fully preprocessed dataset containing all survey response features with zero-filled values, dual missingness indicator flags (``_is_na`` and ``_is_missing``), and gender target encoding for stability generalization analysis.**Note: Iceland is excluded from training calculations** due to its very small sample sizes in ESS surveys (typically < 100 respondents). Including Iceland would result in insufficient training samples (~10-15 rows per gender), which is inadequate for training complex models like `HistGradientBoostingClassifier`. Excluding Iceland allows larger countries (Germany, UK, France, etc.) to contribute their full sample sizes.

---

### 3. Generated Visualizations (`plots/`)
* **`age_feature_importance_top20.html`**: Interactive horizontal bar chart illustrating the top 20 most predictive survey features and indicator flags colored by feature category for age-group analysis. Includes a specific color for each type of feature (Base Survey Question, ``_is_na``, ``_is_missing``) and is arranged as a single stack of bars, sorted from smallest importance on the bottom to largest at the top. This visualization reveals which survey questions and missingness patterns most strongly differentiate age groups.
* **`countries_age_similarity_trends_nw_europe.html`**: High-contrast, custom-colored interactive line plot specifically isolating North & Western European countries for age-group analysis. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `nw_europe` to the header title. This visualization enables temporal comparison of how well age groups can be predicted within culturally similar regions.
* **`countries_age_similarity_trends_other.html`**: High-contrast, custom-colored interactive line plot isolating Southern, Central & Eastern European countries and Israel for age-group analysis. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `other` to the header title. This visualization enables comparison of demographic response patterns in regions outside North & Western Europe.
* **`gender_feature_importance_top20.html`**: Interactive horizontal bar chart illustrating the top 20 most predictive survey features and indicator flags colored by feature category for gender analysis. Includes a specific color for each type of feature (Base Survey Question, ``_is_na``, ``_is_missing``) and is arranged as a single stack of bars, sorted from smallest importance on the bottom to largest at the top. This visualization reveals which survey questions and missingness patterns most strongly differentiate between men and women.
* **`gender_similarity_trends_unified6.html`**: Master interactive line plot of accuracy trends across ESS rounds for all countries with >= 6 rounds of participation, including a grand-mean reference line. This was our original plot with 21 countries on the same plot. The unified view enables direct cross-country comparison and identification of outliers or consistent patterns across multiple survey waves.
* **`countries_gender_similarity_trends_nw_europe.html`**: High-contrast, custom-colored interactive line plot specifically isolating North & Western European countries for gender analysis. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `nw_europe` to the header title. This visualization enables temporal comparison of how well gender can be predicted within culturally similar regions.
* **`countries_gender_similarity_trends_other.html`**: High-contrast, custom-colored interactive line plot isolating Southern, Central & Eastern European countries and Israel for gender analysis. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `other` to the header title. This visualization enables comparison of gender response patterns in regions outside North & Western Europe.
* **`forward_stability_countries_gender_similarity_trends_nw_europe.html`**: High-contrast, custom-colored interactive line plot for Forward Predictability experiment specifically isolating North & Western European countries. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `nw_europe` to the header title. This visualization enables temporal comparison of how well gender can be predicted from historical patterns within culturally similar regions.
* **`forward_stability_countries_gender_similarity_trends_other.html`**: High-contrast, custom-colored interactive line plot for Forward Predictability experiment isolating Southern, Central & Eastern European countries and Israel. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `other` to the header title. This visualization enables comparison of forward predictability patterns in regions outside North & Western Europe.
* **`backward_stability_countries_gender_similarity_trends_nw_europe.html`**: High-contrast, custom-colored interactive line plot for Backward Predictability experiment specifically isolating North & Western European countries. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `nw_europe` to the header title. This visualization enables temporal comparison of how well gender can be predicted from modern patterns within culturally similar regions.
* **`backward_stability_countries_gender_similarity_trends_other.html`**: High-contrast, custom-colored interactive line plot for Backward Predictability experiment isolating Southern, Central & Eastern European countries and Israel. Each country's accuracy trend across ESS rounds is displayed with its own color, and the plot dynamically appends `other` to the header title. This visualization enables comparison of backward predictability patterns in regions outside North & Western Europe.
* **`forward_stability_feature_importance_top20.html`**: Interactive horizontal bar chart illustrating the top 20 most predictive survey features from the Forward Predictability experiment (training on Rounds 1-2). Features are colored by category (Base Survey Question, ``_is_na``, ``_is_missing``) and sorted by importance. This reveals which historical survey questions most strongly differentiate gender.
* **`backward_stability_feature_importance_top20.html`**: Interactive horizontal bar chart illustrating the top 20 most predictive survey features from the Backward Predictability experiment (training on Rounds 8-9). Features are colored by category (Base Survey Question, ``_is_na``, ``_is_missing``) and sorted by importance. This reveals which modern survey questions most strongly differentiate gender.

---

## Technical Pipeline Workflow

1. **High-Performance C++ Coverage Filtering**: Filters features to retain only variables with non-null values across every Country × Round group.
2. **Dual Missingness Flag Encoding**: Distinguishes structural survey skips (`_is_na`) from non-responses (`_is_missing`) prior to zero-imputation.
3. **Stratified Group Sampling**: Samples balanced observations per Country × Round stratum to eliminate sampling quantity bias across strata.
4. **Gradient Boosting & Permutation Selection**: Uses `HistGradientBoostingClassifier` combined with permutation importance to eliminate non-informative features.

---

### Special Considerations for Stability Generalization Analysis

**Iceland Exclusion**: For stability generalization experiments, Iceland is excluded from training calculations due to its very small sample sizes in ESS surveys (typically < 100 respondents). When calculating `train_n` based on the smallest country/round group, including Iceland would result in only ~10-15 training rows per gender, which is insufficient for training complex models. This exclusion allows larger countries (Germany, UK, France, etc.) to contribute their full sample sizes while still including Iceland in test performance evaluations.