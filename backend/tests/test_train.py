import numpy as np
import pandas as pd
import pytest
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

from src.train import (
    MODEL_SPECS,
    STRATEGIES,
    BalancedGradientBoostingClassifier,
    make_pipeline,
    pick_strategy,
    positive_weight,
    search_space,
)

PW = 3.9


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_builds_with_supported_strategies(name):
    for strategy in STRATEGIES:
        if strategy == "class_weight" and not MODEL_SPECS[name][1]:
            with pytest.raises(ValueError, match="does not support"):
                make_pipeline(name, strategy, PW)
            continue
        pipe = make_pipeline(name, strategy, PW)
        assert list(pipe.named_steps)[:2] == ["features", "preprocess"]
        assert list(pipe.named_steps)[-1] == "model"


def test_weighting_and_sampling_never_combined():
    # class_weight -> weighted model, no sampler
    weighted = make_pipeline("LightGBM", "class_weight", PW)
    assert "sampler" not in weighted.named_steps
    assert weighted.named_steps["model"].class_weight == "balanced"
    # SMOTE -> sampler step, unweighted model
    smote = make_pipeline("XGBoost", "smote", PW)
    assert isinstance(smote.named_steps["sampler"], SMOTE)
    assert smote.named_steps["model"].scale_pos_weight == 1.0
    smoteenn = make_pipeline("LightGBM", "smoteenn", PW)
    assert isinstance(smoteenn.named_steps["sampler"], SMOTEENN)
    assert smoteenn.named_steps["model"].class_weight is None


def test_xgboost_scale_pos_weight():
    pipe = make_pipeline("XGBoost", "class_weight", PW)
    assert pipe.named_steps["model"].scale_pos_weight == PW


def test_positive_weight():
    y = pd.Series([0] * 39 + [1] * 10)
    assert positive_weight(y) == pytest.approx(3.9)


def test_scaling_matches_model_family():
    for name, (scale, _) in MODEL_SPECS.items():
        num = make_pipeline(name, "none", PW).named_steps["preprocess"].transformers[0][1]
        assert (num != "passthrough") == scale, name


def test_balanced_gradient_boosting_upweights_minority():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (rng.random(400) < 0.2).astype(int)
    plain = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
    balanced = BalancedGradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
    # balancing pushes predicted churn probability up on average
    assert balanced.predict_proba(X)[:, 1].mean() > plain.predict_proba(X)[:, 1].mean()
    assert BalancedGradientBoostingClassifier().get_params()["n_estimators"] == 100


def test_pick_strategy_prefers_recall_within_tie_band():
    group = pd.DataFrame({
        "pr_auc_mean": [0.698, 0.692, 0.650],
        "recall_mean": [0.46, 0.73, 0.90],
    })
    assert pick_strategy(group) == 1  # 0.692 ties with 0.698; 0.650 is out of the band


@pytest.mark.parametrize("name, strategy", [
    ("GradientBoosting", "class_weight"), ("LightGBM", "class_weight"),
    ("XGBoost", "smote"), ("RandomForest", "smoteenn"),
])
def test_search_space_keys_are_valid_params(name, strategy):
    pipe = make_pipeline(name, strategy, PW)
    valid = pipe.get_params().keys()
    for key in search_space(name, strategy):
        assert key in valid, key
