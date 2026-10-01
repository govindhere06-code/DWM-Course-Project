# EDA Summary — Bank Customer Churn

Source: `notebooks/01_eda.ipynb` (logic in `src/eda.py`). Figures in `reports/figures/`, tables in `reports/`.
Dataset: 10,000 customers, 14 columns, no nulls, no duplicates. Overall churn rate **20.4%** (2,037 / 10,000).

## Key findings

1. **The target is imbalanced ~4:1.** 7,963 retained (79.6%) vs 2,037 churned (20.4%); an all-"stays" model already
   scores 79.6% accuracy, so accuracy is not a useful metric. — `target_distribution.png`

2. **Age is the strongest continuous driver.** Median age is 45 for churners vs 36 for retained customers
   (Mann-Whitney p ≈ 3e-230, rank-biserial r = 0.46, the largest effect of any feature). — `continuous_vs_churn.png`,
   `statistical_tests.csv`

3. **The age effect is non-monotonic.** Churn rises from 7.6% (18–29) and 10.9% (30–39) to 30.8% (40–49) and peaks at
   **56.0% (50–59)**, then drops to 27.9% for 60+. — `churn_rate_by_age_band.png`

4. **Product count has a U-shaped effect.** Churn is 27.7% with 1 product, only **7.6% with 2**, then 82.7% with 3 and
   100% with 4 (n = 266 and n = 60, so small samples). Cramér's V = 0.39, the strongest categorical association, yet
   Pearson r with `Exited` is only −0.05 because the relationship is not linear. — `churn_rate_by_category.png`,
   `correlation_heatmap.png`, `statistical_tests.csv`

5. **German customers churn twice as often.** Germany 32.4% vs France 16.2% and Spain 16.7% (χ² p ≈ 4e-66,
   V = 0.17). — `churn_rate_by_category.png`

6. **Inactive members churn almost twice as often.** 26.9% inactive vs 14.3% active (χ² p ≈ 9e-55, V = 0.16). Among
   single-product customers, inactivity raises churn from 18.9% to **36.7%**. — `churn_rate_by_category.png`,
   `churn_heatmap_numofproducts_x_isactivemember.png`

7. **Women churn more than men in every country.** 25.1% vs 16.5% overall; the highest-risk cell is German women at
   **37.6%**, the lowest is French men at 12.7%. — `churn_heatmap_geography_x_gender.png`

8. **36% of customers have a zero balance, and they churn *less*.** 3,617 customers have `Balance == 0` (bimodal
   distribution); their churn rate is 13.8% vs 24.1% for non-zero balances. This is partly a geography confound:
   **0%** of German customers have a zero balance vs 48% elsewhere. — `distribution_balance.png`,
   `churn_rate_balance_zero.png`

9. **Churners hold more money.** Median balance is 109,349 for churners vs 92,073 for retained customers
   (Mann-Whitney p ≈ 1e-28, r = 0.16). — `continuous_vs_churn.png`

10. **Tenure, salary and credit-card ownership carry almost no signal.** Tenure (χ² p = 0.18), EstimatedSalary
    (p = 0.23) and HasCrCard (p = 0.49) are not significant at α = 0.05. CreditScore is significant (p = 0.02), but its
    effect is negligible (r = −0.03). EstimatedSalary is uniformly distributed between 0 and 200k, which suggests the data
    is synthetic. — `statistical_tests.csv`, `distribution_estimatedsalary.png`

11. **There is no multicollinearity.** Every VIF is ≈ 1.0 (max 1.10 for Balance and NumOfProducts); the only notable
    pairwise correlation is Balance–NumOfProducts (r = −0.30). — `vif.csv`, `correlation_heatmap.png`

12. **The outliers are real customers, not errors.** IQR flags 359 Age values above 62 (3.6%) and 15 CreditScores below
    383 (0.15%); Balance and EstimatedSalary have none. They will be kept: older customers are a high-churn segment.
    — `outlier_report.csv`, `distribution_age.png`

## Feature-engineering hypotheses

| Hypothesis | Feature idea | Based on finding |
|---|---|---|
| Age risk is non-linear and peaks in the 50s | `AgeGroup` bins (18–29 … 60+), `IsSenior` (Age ≥ 60) | 2, 3 |
| Zero balance marks a distinct customer type | `BalanceZero` flag | 8 |
| Balance relative to income is more telling than raw balance | `BalanceSalaryRatio` = Balance / EstimatedSalary | 9, 10 |
| Short relationship relative to age = weaker loyalty | `TenureByAge` = Tenure / Age | 2, 10 |
| Credit score matters more relative to life stage | `CreditScoreGivenAge` = CreditScore / Age | 10 |
| Many products in a short tenure signals mis-selling or churn risk | `ProductsPerTenure` = NumOfProducts / (Tenure + 1) | 4 |
| Product count should be treated as categorical, not linear | one-hot or tree models for `NumOfProducts` | 4 |
| Interactions drive risk (Germany × female, inactive × 1 product) | rely on tree models / gradient boosting, which capture interactions | 6, 7 |
| Imbalance needs explicit handling | `class_weight="balanced"`, `scale_pos_weight ≈ 3.9`, SMOTE inside CV; evaluate on Recall + PR-AUC | 1 |
| No feature has to be dropped for collinearity | keep all 10 raw features | 11 |
