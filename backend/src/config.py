import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_PATH = os.path.join(BASE_DIR, "data", "insurance_claims.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
AUDIT_PATH = os.path.join(MODEL_DIR, "fairness_audit.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_COL = "fraud_reported"
THRESHOLD = 0.40          # chosen from cross-validation, see notebook
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Final settings chosen in the notebook
MODEL_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)

# Columns the model never sees. Each drop was tested in the notebook:
# IDs / near-unique values, dates (no signal), duplicated claim parts,
# auto_model (no gain), and sensitive or stand-in columns (no cost to remove).
DROP_COLS = [
    "policy_number", "policy_bind_date", "insured_zip",
    "incident_date", "incident_location", "auto_model",
    "injury_claim", "property_claim", "vehicle_claim",
    "age", "insured_sex", "months_as_customer", "insured_occupation",
    TARGET_COL,
]

# Columns kept aside (not model inputs) for the fairness audit
AUDIT_COLS = ["age", "insured_sex", "policy_state", "incident_state"]