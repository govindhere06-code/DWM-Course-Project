"""Central configuration: paths, constants and column groups."""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "Churn_Modelling.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DATA = PROCESSED_DIR / "train.csv"
TEST_DATA = PROCESSED_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# ------------------------------------------------------------ constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# -------------------------------------------------------------- columns
TARGET = "Exited"

# Identifiers: no predictive value, overfitting risk.
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

CATEGORICAL_COLS = ["Geography", "Gender"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]
DISCRETE_COLS = ["Tenure", "NumOfProducts"]
CONTINUOUS_COLS = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
NUMERIC_COLS = CONTINUOUS_COLS + DISCRETE_COLS

# Short aliases used in TICKETS.md.
CAT_COLS = CATEGORICAL_COLS
NUM_COLS = NUMERIC_COLS

# Raw feature columns a scoring request must provide (IDs optional).
FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# ------------------------------------------------------- risk banding
# Low: p < LOW_RISK_MAX | Medium: LOW_RISK_MAX <= p <= MEDIUM_RISK_MAX | High: p > MEDIUM_RISK_MAX
LOW_RISK_MAX = 0.3
MEDIUM_RISK_MAX = 0.6

# ------------------------------------------------------- model artifacts
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
FINAL_METRICS_PATH = REPORTS_DIR / "final_metrics.json"

# Business cost assumption: missing a churner costs 5x a false alarm.
COST_FN = 5.0
COST_FP = 1.0
