# Model Insights — What Drives Churn

Model: tuned **XGBoost + SMOTE** (`models/churn_pipeline.joblib`). Decision threshold **0.34**, chosen as the F1-optimal
point on out-of-fold train predictions.
Test set (2,000 customers, never used for training or threshold choice): **ROC-AUC 0.867, PR-AUC 0.718**. At the 0.34
threshold the model catches **66% of churners** (270 / 407) with 63% precision, versus 52% recall at the default 0.5.
Full numbers: `reports/final_metrics.json`.

Sources: SHAP values on the test set (`figures/explain_shap_beeswarm.png`, `explain_shap_bar.png`,
`explain_shap_dependence.png`, `reports/shap_mean_abs.csv`), permutation importance on the test set
(`figures/explain_permutation_importance.png`, `reports/permutation_importance.csv`), and churn rates in the full data.

## Top churn drivers (plain English)

The methods agree on the ranking. Permutation importance below is the drop in test PR-AUC when a column is shuffled;
mean |SHAP| is in log-odds.

| Rank | Feature | Permutation importance | Mean \|SHAP\| | What the model learned |
|---|---|---|---|---|
| 1 | `NumOfProducts` | 0.276 | 0.80 | **Two products is the safe spot** (7.6% churn). One product is risky (27.7%). Three or four products is almost certain churn (85.9%, n = 326); the largest single SHAP push in the model is about +3 log-odds. |
| 2 | `Age` | 0.259 | 0.60 (+ `AgeGroup_*`) | Risk is flat and low until about 40, **climbs steeply from 42 to the late 50s**, then falls again after about 65. Customers aged 40–59 make up 35% of the base but **63% of all churners**. |
| 3 | `IsActiveMember` | 0.091 | 0.45 | Inactive customers churn about twice as often (26.9% vs 14.3%). Activity protects older customers most (the SHAP effect is largest for ages 50+). |
| 4 | `Balance` | 0.048 | 0.13 | A high balance with only one product signals a customer with money and little tie to the bank (26.9% churn). A zero balance lowers risk. |
| 5 | `Geography` (Germany) | 0.045 | 0.19 | Being German adds risk on top of every other factor (32.4% vs about 16% churn). |
| 6 | `Gender` | 0.014 | 0.26 | Women churn more than men (25.1% vs 16.5%); German women are the highest-risk group at 37.6%. |
| — | `CreditScore`, `Tenure`, `HasCrCard`, `EstimatedSalary` | ≤ 0.006 | small | **Almost no effect.** Shuffling CreditScore even slightly *improves* PR-AUC (−0.002), so the model gets nothing useful from it. |

**Two example customers** (`figures/explain_shap_waterfall_churner.png`, `explain_shap_waterfall_nonchurner.png`):
- *Churner, 72% predicted:* German man, 46, inactive, one product, balance 141k. Age (+0.67), Germany (+0.42), a
  single product (+0.27) and inactivity (+0.13) outweigh being male (−0.34).
- *Non-churner, 7% predicted:* German man, 38, active, two products, balance 126k. Two products (−1.19), being active
  (−0.57), being male (−0.42) and being in his 30s far outweigh Germany (+0.29) and the high balance (+0.27).

## Actionable recommendations

1. **Move single-product customers to a second product.** *(feature: `NumOfProducts`)*
   Churn falls from 27.7% with one product to 7.6% with two. Cross-sell a second product (savings account, card,
   insurance) to one-product customers, prioritising those with high balances: 3,157 customers hold >100k with a single
   product and churn at 26.9%.

2. **Review 3–4 product bundles urgently.** *(feature: `NumOfProducts`)*
   85.9% of the 326 customers with three or more products churn. This looks like mis-selling or bundles that do not fit.
   Interview these customers and audit how the bundles were sold, rather than pushing more products.

3. **Re-engage inactive customers aged 40–59, starting in Germany.** *(features: `Age`, `IsActiveMember`, `Geography`)*
   Inactive 40–59-year-olds churn at 47.1% (vs 26.8% for active ones of the same age); in Germany it is **60.9%**
   (560 customers, 341 churners). This is the single best target segment for a retention campaign: personal outreach,
   a relationship manager, loyalty benefits.

4. **Run a Germany-specific retention review.** *(feature: `Geography`)*
   German customers churn at twice the rate of French and Spanish ones even after controlling for age, products and
   activity. Investigate local causes (pricing, competition, service quality), with a focus on women (37.6% churn).

5. **Score every customer monthly and act on the High band.** *(all features, via `src/predict.py`)*
   Use the risk bands (Low < 0.3, Medium 0.3–0.6, High > 0.6) to rank customers. At the 0.34 threshold the model flags
   429 of 2,000 test customers and catches 66% of churners. If missing a churner costs 5× a false alarm, lower the
   threshold to **0.17** (cost-optimal), which catches 83% of churners and cuts total cost from 1,036 (at 0.5) to 776
   on the test set. Re-engagement offers are cheap, so the lower threshold is likely the better business choice.

## Caveats
- SHAP values describe the **model**, not causation: moving a customer to two products will not automatically halve
  their risk. Recommendations should be tested (A/B or pilot) before rollout.
- The 3–4-product finding rests on only 326 customers (60 with four products).
- The dataset is a single snapshot with no time dimension, and parts of it (e.g. uniformly distributed salary) look
  synthetic, so the absolute rates may not transfer to a real bank.
