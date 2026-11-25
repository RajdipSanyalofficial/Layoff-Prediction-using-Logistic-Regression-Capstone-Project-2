# Employee Layoff Prediction Using Logistic Regression

Predicts whether an employee is likely to be laid off, using employee data and Logistic Regression.

The full work is in one notebook: `Layoff__Prediction.ipynb`.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Workflow](#workflow)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Output Files](#output-files)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

- **Goal:** Predict `laid_off` (1 = laid off, 0 = not laid off) for each employee.
- **Method:** Logistic Regression (built with `statsmodels` for analysis and `scikit-learn` for prediction).
- **Techniques used:** EDA, Weight of Evidence (WoE), Information Value (IV), p-values, Variance Inflation Factor (VIF), ROC/AUC, KS statistic.

---

## Dataset

- **File name used in notebook:** `layoff_survival.csv`
- **Size:** 29,194 rows, 15 columns (14 after dropping `employee_id`)
- **Missing values:** none
- **Target balance:** 5,912 laid off (about 20.25%) and 23,282 not laid off (about 79.75%)
- **Source:** not documented. Add the source or licence here before publishing.
- **Note:** the dataset file is **not included** in this repo. Add it yourself (see [How to Run](#how-to-run)).

---

## Features

| Feature | Description |
|---|---|
| `employee_id` | Unique ID (dropped before modelling) |
| `department` | Employee's department |
| `cost_center_type` | Type of cost centre (cost_center, revenue, support) |
| `team_wind_down` | 1 if the employee's team is being wound down |
| `tenure_years` | Years at the company |
| `is_recent_hire` | 1 if recently hired |
| `performance_rating` | Performance score (range 1 to 5) |
| `on_pip` | 1 if on a Performance Improvement Plan |
| `mgmt_layer` | Management layer (IC, manager, director) |
| `level` | Job level (junior, mid, senior, lead, staff) |
| `salary_k_usd` | Salary in thousands of USD |
| `high_salary` | 1 if in the high-salary group |
| `remote` | 1 if works remotely |
| `role_redundant` | 1 if the role is redundant |
| `laid_off` | **Target.** 1 = laid off, 0 = not laid off |

---

## Workflow

1. Load data and drop `employee_id`.
2. Calculate WoE and IV for every variable.
3. Check salary and tenure with boxplots and quantiles. A function for IQR outlier removal is defined, **but it is not applied to the data**. No rows were removed.
4. Run `describe()` and check missing values.
5. One-hot encode `department`, `cost_center_type`, `mgmt_layer`, `level` (`drop_first=True`).
   - Reference groups: `data` department, `cost_center` cost centre, `IC` management layer, `junior` level.
6. Fit a Logistic Regression with `statsmodels`. Check p-values, coefficients, z-scores and odds ratios.
7. Remove weak variables step by step (3 rounds of refitting) using p-values.
8. Check multicollinearity with VIF on the final model.
9. Fit the final model with `scikit-learn` (`C=1e9`, `max_iter=1000`, so almost no regularisation).
10. Evaluate: confusion matrix, classification report, ROC/AUC, KS table.
11. Save predictions to CSV.

### Final model: 14 variables

`team_wind_down`, `tenure_years`, `is_recent_hire`, `performance_rating`, `on_pip`, `salary_k_usd`, `high_salary`, `role_redundant`, `department_g_and_a`, `department_recruiting`, `department_research`, `cost_center_type_revenue`, `mgmt_layer_director`, `mgmt_layer_manager`

### Variables removed

- `remote`: p = 0.43 (not significant)
- All `level_*` dummies: p-values between 0.36 and 0.98
- `department_engineering`, `department_marketing`, `department_product`, `department_sales`, `department_support`, `cost_center_type_support`: not significant, or unstable in the first model

All 14 final variables have p < 0.001 and VIF below 3 (highest: `salary_k_usd` 2.84 and `high_salary` 2.84).

---

## Results

### Information Value (IV), top variables

| Variable | IV |
|---|---|
| `team_wind_down` | 1.410 |
| `department` | 0.365 |
| `cost_center_type` | 0.362 |
| `performance_rating` | 0.076 |
| `on_pip` | 0.068 |
| `role_redundant` | 0.059 |

`remote` (0.0001) and `level` (0.005) had almost no predictive power.

### Model performance

> **Important:** these numbers are measured on the **same data the model was trained on**. There is no train/test split. Treat them as optimistic.

| Metric | Value |
|---|---|
| Accuracy | 0.87 |
| AUC | 0.838 |
| KS statistic | 53.1% (at decile 3) |
| Pseudo R-squared (statsmodels) | 0.301 |

**Confusion matrix** (default threshold 0.5)

| | Predicted: Not laid off | Predicted: Laid off |
|---|---|---|
| **Actual: Not laid off** | 22,216 | 1,066 |
| **Actual: Laid off** | 2,873 | 3,039 |

**Classification report**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (not laid off) | 0.89 | 0.95 | 0.92 |
| 1 (laid off) | 0.74 | 0.51 | 0.61 |

- The model finds only about half (51%) of the people who were actually laid off.
- The top 20% of employees by predicted risk (deciles 1 and 2) contain about 62% of all layoffs.

---

## Key Findings

Odds ratios below are calculated from the final model's coefficients (`exp(coefficient)`). They are not printed in the notebook for the final model.

**Raise layoff risk**

| Factor | Odds ratio |
|---|---|
| Team being wound down | about 22.7x |
| Role redundant | about 2.8x |
| On PIP | about 2.5x |
| Director level (vs IC) | about 2.2x |
| High salary group | about 1.9x |
| Recent hire | about 1.9x |
| Recruiting department (vs data) | about 1.8x |
| Research department (vs data) | about 1.7x |
| G&A department (vs data) | about 1.5x |
| Manager level (vs IC) | about 1.4x |

**Lower layoff risk**

| Factor | Odds ratio |
|---|---|
| Revenue cost centre (vs cost_center) | about 0.55x |
| Each extra point of performance rating | about 0.74x |
| Each extra year of tenure | about 0.98x |

Odds ratios show association in this dataset. They do not prove cause.

---

## Limitations

- **No train/test split.** All metrics are in-sample. Real performance on new data is unknown.
- **`team_wind_down` is very dominant** (IV 1.41, odds ratio about 22.7). An IV above 0.5 is often a sign of data leakage. Check whether this flag is known before layoffs happen.
- **Weak recall for layoffs (0.51)** at the default 0.5 threshold. No threshold tuning was done.
- **Class imbalance** (about 20% layoffs) was not handled.
- **IV was calculated but not used** to pick features. Selection was done with p-values and VIF.
- **Outlier function is unused.** Salary maximum is 324k USD, and all rows were kept.
- **`salary_k_usd` and `high_salary` overlap** (VIF 2.84 each). Their opposite signs can be confusing to read.
- **Data source is not documented.**

---

## How to Run

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install the requirements:
   ```bash
   pip install pandas numpy statsmodels scikit-learn matplotlib colorama jupyter
   ```
3. Put `layoff_survival.csv` in the project folder.
4. **Edit the file path in the notebook.** It is hardcoded to a local Windows path (`C:\Users\HP\Desktop\...`). Replace both the `os.chdir(...)` and `pd.read_csv(...)` lines with:
   ```python
   data = pd.read_csv("layoff_survival.csv")
   ```
5. Open and run all cells:
   ```bash
   jupyter notebook Layoff__Prediction.ipynb
   ```

---

## Requirements

- Python 3
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- colorama (used to colour the KS output)
- jupyter

The notebook does not record library versions. Add a `requirements.txt` with exact versions for reproducibility.

---

## Output Files

Created when the notebook runs:

- `ROC_CURVE.png`: ROC curve image
- `LAYOFF_Final__Predicted.csv`: employee data with actual layoff status (`y`) and predicted probability (`p`)

---

## Future Improvements

- Add a train/test split or cross-validation and report test metrics.
- Check `team_wind_down` for leakage.
- Tune the probability threshold to improve recall for layoffs.
- Try class weights or resampling.
- Compare with tree-based models (Random Forest, XGBoost).
- Move the code into scripts and add `requirements.txt`.

---

## Author

- Rajdip Sanyal
