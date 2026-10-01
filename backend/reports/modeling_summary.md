# Modelling Summary (T04)

Code: `src/train.py` (`python -m src.train`, or `--stage baseline|imbalance|tune|ensemble`).
Every number below is a **5-fold stratified CV score on the train set (8,000 rows)**; the test set has not been used yet.
Every candidate is a full pipeline, `FeatureEngineer → preprocessor → [sampler] → model`, so scaling, encoding and
resampling are re-fitted inside each training fold.

## Part A — Baseline models
Default parameters, with class weighting wherever the model supports it (`scale_pos_weight = 3.91` for XGBoost).
Full table: `reports/baseline_results.csv`. Chart: `figures/baseline_comparison.png`.

| Model | PR-AUC | ROC-AUC | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Gradient Boosting | **0.692** | 0.861 | 0.734 | 0.601 | 0.802 |
| LightGBM | 0.688 | 0.855 | 0.682 | 0.607 | 0.821 |
| XGBoost | 0.655 | 0.835 | 0.591 | 0.587 | 0.831 |
| SVM (RBF) | 0.644 | 0.844 | 0.746 | 0.583 | 0.783 |
| Random Forest | 0.643 | 0.845 | 0.580 | 0.593 | 0.838 |
| Logistic Regression | 0.529 | 0.784 | 0.702 | 0.509 | 0.724 |
| KNN | 0.511 | 0.781 | 0.412 | 0.502 | 0.834 |
| Gaussian NB | 0.491 | 0.774 | 0.632 | 0.508 | 0.750 |
| Decision Tree | 0.341 | 0.676 | 0.483 | 0.484 | 0.791 |
| Dummy (most frequent) | 0.204 | 0.500 | **0.000** | 0.000 | **0.796** |

**Accuracy is misleading:** the Dummy model never predicts churn (recall 0), yet its 79.6% accuracy beats Gradient
Boosting, Logistic Regression, SVM and Naive Bayes. Boosted trees lead on PR-AUC: they capture the non-linear effects
seen in EDA (U-shaped product count, age peaking at 50–59) and interactions such as Germany × gender, which linear and
distance-based models miss.

## Part B — Class-imbalance strategies (top 3 models)
Compared: (a) no handling, (b) class weight / `scale_pos_weight`, (c) SMOTE, (d) SMOTEENN. Weighting and resampling
are never combined, to avoid double-correcting. Full table: `reports/imbalance_comparison.csv`.
Chart: `figures/imbalance_comparison.png`.

| Model | No handling | Class weight | SMOTE | SMOTEENN | Chosen |
|---|---|---|---|---|---|
| Gradient Boosting | 0.698 / 0.461 | 0.692 / **0.734** | 0.691 / 0.513 | 0.661 / 0.547 | **Class weight** |
| LightGBM | 0.685 / 0.490 | 0.688 / **0.682** | 0.687 / 0.510 | 0.649 / 0.552 | **Class weight** |
| XGBoost | 0.663 / 0.483 | 0.655 / 0.591 | **0.668** / 0.505 | 0.626 / 0.536 | **SMOTE** |

*(cells are PR-AUC / Recall at the default 0.5 threshold)*

**Selection rule:** highest PR-AUC. Strategies within **0.01 PR-AUC** of the best count as a tie, and the tie is broken
by higher recall. The 0.01 band is about one standard error of a 5-fold mean (fold std ≈ 0.025 / √5 ≈ 0.011);
differences smaller than that are noise.

**Why:**
- PR-AUC measures ranking, so it barely moves between no handling, class weights and SMOTE (spread ≤ 0.013 per model).
  What the strategies mainly change is where the 0.5 cut-off falls.
- **Class weighting** gives Gradient Boosting and LightGBM a large, nearly free recall gain (+0.27 and +0.19) at the same
  PR-AUC. It is also the cheapest option: no synthetic rows, and fit time is unchanged.
- **XGBoost:** class weighting cost 0.013 PR-AUC, just outside the tie band, so SMOTE (best PR-AUC, 0.668) is kept.
- **SMOTEENN** is the worst option for all three models (−0.03 to −0.04 PR-AUC). Its ENN cleaning step removes
  majority-class rows near the boundary, which throws away useful information on a dataset with this much class overlap.

## Part C — Hyperparameter tuning
`RandomizedSearchCV`, `n_iter = 40`, `scoring = "average_precision"`, the same 5 stratified folds, `random_state = 42`.
Optuna is not installed, so it was not used. For the SMOTE pipeline, `k_neighbors` (3–10) is tuned along with the model.
Best parameters: `models/best_params.json`. Table: `reports/tuning_results.csv`.

| Model + strategy | Baseline PR-AUC | Tuned PR-AUC | Gain |
|---|---|---|---|
| XGBoost + SMOTE | 0.668 | **0.698** | +0.030 |
| Gradient Boosting + class weight | 0.692 | 0.695 | +0.004 |
| LightGBM + class weight | 0.688 | 0.695 | +0.007 |

Every tuned model beats its own baseline. The winning settings are consistent across models: a low learning rate
(0.016–0.022) with many trees (270–560), moderate depth (4–6), and row/column subsampling. XGBoost gained the most,
largely from strong regularisation (`gamma` 2.6, `min_child_weight` 9). After tuning, the three models are within
0.003 PR-AUC of each other, which is statistically a tie.

## Part D — Ensembles
Built from the three tuned pipelines (each keeps its own imbalance strategy). Table: `reports/ensemble_comparison.csv`.

| Model | PR-AUC | ROC-AUC | Recall @0.5 | F1 @0.5 |
|---|---|---|---|---|
| Stacking (meta-learner: Logistic Regression) | 0.6993 | 0.864 | 0.518 | 0.601 |
| Soft voting | 0.6992 | 0.864 | 0.652 | 0.628 |
| **XGBoost + SMOTE (tuned)** | **0.6976** | 0.864 | 0.504 | 0.596 |
| Gradient Boosting + class weight (tuned) | 0.6954 | 0.862 | 0.722 | 0.602 |
| LightGBM + class weight (tuned) | 0.6951 | 0.861 | 0.704 | 0.611 |

The best ensemble (Stacking) beats the best single model by only **0.0017 PR-AUC**, under the required 0.005 margin.
It is **not kept**: it costs about 4× the training time and is harder to explain, for a gain smaller than the fold-to-fold
noise. The three base models are highly correlated (all boosted trees on the same features), so averaging them adds
little diversity.

## Final model choice
**Tuned XGBoost + SMOTE**: CV PR-AUC 0.698 ± 0.024, ROC-AUC 0.864 ± 0.010 (`models/final_model.json`).

Its recall at the default 0.5 threshold is only 0.50. This is expected and will be addressed in T05: SMOTE plus strong
regularisation produce conservative probabilities, and the decision threshold will be tuned on the precision/recall
curve (F1-maximising, plus a Recall ≥ 0.75 option). Ranking quality (PR-AUC / ROC-AUC) is what the threshold cannot fix,
and on that this model is the best. Tuned Gradient Boosting + class weight is an equally strong alternative
(0.695 PR-AUC, recall 0.72 at 0.5).
