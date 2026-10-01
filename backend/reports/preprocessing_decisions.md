# Preprocessing Decisions

Code: `src/data.py` (`prepare`, `split`) and `src/features.py` (`FeatureEngineer`, `build_preprocessor`, `build_pipeline`).

## 1. Cleaning
- **Dropped:** `RowNumber`, `CustomerId` and `Surname`. They are identifiers with no predictive meaning, and a model could
  memorise them and overfit.
- **Rows removed:** none. There are no nulls and no duplicates (`validate()` checks both; `prepare()` re-checks duplicates).

## 2. Outliers: keep them
The IQR rule flags 359 `Age` values above 62 (3.6%) and 15 `CreditScore` values below 383 (0.15%); `Balance` and
`EstimatedSalary` have none (`reports/outlier_report.csv`).

**Decision: keep every row.**
- The values are plausible real customers (ages 63–92, credit scores 350–382), not entry errors, and all of them pass
  the range checks in `validate()`.
- Older customers are the most informative group: churn reaches 56% at ages 50–59. Removing the age tail would throw
  away signal.
- The test set will contain such customers too, so the model has to handle them.
- Instead of deleting rows we rely on:
  - **tree models** (Random Forest, Gradient Boosting, XGBoost, LightGBM), which split on thresholds and are insensitive
    to extreme values;
  - **scaling** for the distance- and linear-based models (LogReg, KNN, SVM). The one truly extreme feature,
    `BalanceSalaryRatio` (median 0.75, max ≈ 10,600 because a few salaries are tiny), gets a `log1p` transform before
    `StandardScaler`.

## 3. Train/test split
- Stratified 80/20 split with `random_state = 42`, done **before** any fitting.
- Train: 8,000 rows, churn rate 20.37%. Test: 2,000 rows, churn rate 20.35%. The overall rate is 20.37%.
- Saved to `data/processed/train.csv` and `data/processed/test.csv` (raw feature columns plus `Exited`).

## 4. Leakage safeguards
- `FeatureEngineer` is stateless: every feature is computed from the row itself, and nothing is learned in `fit`.
- Scalers and encoders live inside the `ColumnTransformer`, inside an `imblearn` `Pipeline`, so they are fitted only on
  the data passed to `fit` (training folds during CV, the full train set for the final model).
- Samplers such as SMOTE are a pipeline step, so they only run during `fit` and never touch validation or test rows.
