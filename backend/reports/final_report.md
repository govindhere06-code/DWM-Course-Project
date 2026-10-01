# Bank Customer Churn Prediction — Final Report

## 1. Problem
A retail bank wants to know **which customers are about to leave** (`Exited = 1`) so it can act before they go. The
goal is a model that ranks customers by churn risk and flags likely churners, plus an interactive dashboard that
explains the drivers to non-technical users.

Churners are only 20% of customers, so a model that never predicts churn is already 79.6% "accurate". The project
therefore optimises **recall on churners and PR-AUC**, not accuracy.

## 2. Data
`Churn_Modelling.csv`: 10,000 customers × 14 columns, a single snapshot, with no missing values and no duplicates
(checked by `src/data.py::validate`).

| Group | Columns |
|---|---|
| Identifiers (dropped) | `RowNumber`, `CustomerId`, `Surname` |
| Categorical | `Geography` (France / Germany / Spain), `Gender` |
| Binary | `HasCrCard`, `IsActiveMember` |
| Discrete | `Tenure` (0–10), `NumOfProducts` (1–4) |
| Continuous | `CreditScore`, `Age`, `Balance`, `EstimatedSalary` |
| Target | `Exited`: 7,963 stayed (79.6%) / 2,037 churned (20.4%) |

## 3. Methodology

### 3.1 Preprocessing (`src/data.py`, `src/features.py`)
- **Validation:** schema, dtypes, nulls, target values and valid ranges (Age 18–100, CreditScore 300–900, NumOfProducts
  1–4, Tenure 0–10).
- **Split first:** a stratified 80/20 train/test split (`random_state = 42`) is made before anything is fitted. Train
  is 8,000 rows at 20.37% churn; test is 2,000 rows at 20.35%.
- **Outliers kept:** the 359 customers aged over 62 and the 15 with credit scores below 383 are plausible, real
  customers, and the older ones are a high-churn segment. Tree models and scaling handle them instead
  (`reports/preprocessing_decisions.md`).
- **Feature engineering:** a stateless `FeatureEngineer` adds `BalanceZero`, `BalanceSalaryRatio`, `TenureByAge`,
  `CreditScoreGivenAge`, `ProductsPerTenure`, `AgeGroup` and `IsSenior`. It learns nothing from the data, so it cannot
  leak.
- **Encoding and scaling:** a `ColumnTransformer` one-hot encodes `Geography`/`AgeGroup` and encodes `Gender` as
  Male = 1. For linear and distance models it applies `StandardScaler` (with `log1p` first on the very skewed
  `BalanceSalaryRatio`); tree models get numeric features unchanged.
- **One pipeline everywhere:** `FeatureEngineer → preprocessor → [sampler] → model` is an `imblearn` Pipeline, so
  encoders, scalers and SMOTE are only ever fitted on training folds.

### 3.2 Exploratory analysis (`src/eda.py`, `notebooks/01_eda.ipynb`, `reports/eda_summary.md`)
Statistical tests (chi-square, Mann-Whitney U) and VIF, plus 14 figures. The main signals were age (non-monotonic,
peaking at 56% churn for ages 50–59), a U-shaped effect of product count, Germany (32% vs 16%), inactivity and gender.
Tenure, salary and credit-card ownership had no significant effect, and no multicollinearity was found (every VIF is
about 1).

### 3.3 Model selection (`src/train.py`, `reports/modeling_summary.md`)
Everything in this section uses 5-fold stratified CV on the **train set only**, with the same folds for every model.
1. **Baselines:** 10 models with default settings: Dummy, Logistic Regression, KNN, Gaussian NB, Decision Tree, SVM,
   Random Forest, Gradient Boosting, XGBoost and LightGBM. The boosted trees led on PR-AUC (0.655–0.692). Dummy scored
   recall 0 at 79.6% accuracy.
2. **Imbalance handling** for the top 3: none / class weights / SMOTE / SMOTEENN. PR-AUC moved by at most about 0.013
   between the first three, while class weights added +0.11 to +0.27 recall at the default threshold. SMOTEENN was the
   worst for every model. Chosen: Gradient Boosting and LightGBM with class weights, XGBoost with SMOTE.
3. **Tuning:** `RandomizedSearchCV`, 40 settings per model, scored on average precision. Every model improved;
   XGBoost + SMOTE went from 0.668 to **0.698**.
4. **Ensembles:** soft voting (0.699) and stacking (0.699) beat the best single model by only 0.0017 PR-AUC, below the
   0.005 bar, so they were not kept.

### 3.4 Threshold and evaluation (`src/evaluate.py`)
- **Threshold chosen on train only:** thresholds come from **out-of-fold predictions on train**, never from the test set.
  The F1-optimal threshold is **0.34**. Also reported: 0.22 (recall ≥ 0.75) and 0.17, which minimises cost if a missed
  churner costs 5× a false alarm.
- **Test set used once:** the final pipeline is refitted on the full train set and scored on the test set a single time.

### 3.5 Explainability (`src/explain.py`, `reports/model_insights.md`)
Built-in gain importance, permutation importance on the test set (shuffling raw columns), and SHAP TreeExplainer
(beeswarm, bar, dependence and waterfall plots). The dashboard adds the SHAP values for derived columns back into the
raw feature they came from, so each prediction is explained as plain-English "top 3 reasons".

## 4. Results — final model on the held-out test set

**Model:** tuned XGBoost + SMOTE (`max_depth 6`, `learning_rate 0.022`, `n_estimators 516`, `gamma 2.56`,
`min_child_weight 9`, `subsample 0.76`, `colsample_bytree 0.61`, SMOTE `k_neighbors 9`).

| | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| 5-fold CV (train) | 0.864 | 0.698 | — | — | — | — |
| **Test, threshold 0.34 (chosen)** | **0.867** | **0.718** | **0.629** | **0.663** | **0.646** | 0.852 |
| Test, threshold 0.50 | 0.867 | 0.718 | 0.763 | 0.523 | 0.621 | 0.870 |
| Test, threshold 0.22 (recall ≥ 0.75) | | | 0.491 | 0.769 | 0.599 | 0.790 |
| Test, threshold 0.17 (cost-optimal) | | | 0.442 | 0.828 | 0.576 | 0.752 |

- **No leakage:** the test scores closely match CV (ROC-AUC 0.867 vs 0.864), as expected when nothing leaks.
- **Calibration:** probabilities are well calibrated (Brier score 0.099), so the Low / Medium / High bands (< 0.3 /
  0.3–0.6 / > 0.6) can be read at face value.
- **Recall gain:** at the chosen threshold the model catches 270 of 407 test churners (66%), versus 52% at 0.5, while
  flagging 429 of 2,000 customers.

## 5. Why this model
- **Best ranking quality:** it has the highest CV PR-AUC of all candidates (0.698), and PR-AUC is the threshold-free
  metric that matters for an imbalanced target.
- **Simple:** ensembles added no meaningful gain (+0.0017) for about 4× the training time and less transparency.
- **The other tuned models are statistically tied:** Gradient Boosting and LightGBM both reached 0.695 PR-AUC, so
  either would be a valid substitute. XGBoost's low recall at 0.5 is fixed by the tuned threshold, because ranking
  quality is what a threshold cannot fix.
- **Explainable:** TreeExplainer gives exact SHAP values per prediction, which the dashboard uses.

**Top churn drivers:** number of products (2 is protective; 3–4 means about 86% churn), age (risk peaks at 50–59),
inactivity, a high balance held with only one product, Germany, and gender. Recommendations are in
`reports/model_insights.md`. The best single target segment is inactive German customers aged 40–59, who churn at
60.9%.

## 6. Limitations
- **Single snapshot, no time dimension:** we cannot see *when* customers churn or how their behaviour changed
  beforehand (transactions, complaints, logins), which real churn models rely on. Survival analysis is impossible.
- **Synthetic-looking data:** `EstimatedSalary` is almost perfectly uniform (0–200k), tenure is uniform, and no German
  customer has a zero balance. The patterns may not transfer to a real bank.
- **Small segments:** the 3–4-product finding rests on 326 customers (60 with four products).
- **Correlation, not causation:** SHAP explains the model, not the customer. "Get them to two products" is a
  hypothesis to test, not a guaranteed effect.
- **Cost assumption:** the 5:1 cost ratio is illustrative; the cost-optimal threshold depends entirely on it.
- **Fairness:** gender and geography are predictors. Using them to target offers needs a fairness and legal review.
- **One test split:** 2,000 rows give test metrics roughly ±0.02 of sampling noise; CV is the more stable estimate.

## 7. Future work
- **More data:** behavioural and time-stamped data (transactions, product changes, service contacts) for a
  time-to-churn or survival model.
- **Monitoring:** watch input drift and calibration in production, retrain on a schedule, and recalibrate if needed.
- **Profit-based thresholds:** replace the 5:1 assumption with real retention-offer costs and customer lifetime value.
- **Uplift modelling:** target the customers a retention action would actually change, not just high-risk ones.
- **Fairness metrics:** add fairness checks by gender and country before any customer-facing use.
- **Deployment:** serve `src/predict.py` as an API (e.g. FastAPI) behind the dashboard, and track experiments with
  MLflow.

## 8. Reproducibility
```bash
pip install -r requirements.txt
cd backend && python -m src.train           # raw CSV -> final model, metrics, figures (~5 min)
cd .. && streamlit run frontend/app.py      # dashboard
```
All randomness uses `RANDOM_STATE = 42` (`backend/src/config.py`). Library versions used for the saved model are in
`backend/models/metadata.json`.
