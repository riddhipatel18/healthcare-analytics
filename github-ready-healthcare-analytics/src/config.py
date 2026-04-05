from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "medical_conditions_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "medical_condition_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

TARGET_COL = "condition"
NUMERIC_FEATURES = ["age", "bmi", "blood_pressure", "glucose_levels"]
CATEGORICAL_FEATURES = ["gender", "smoking_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42
APP_TITLE = "Healthcare & Medical Analytics Dashboard"
APP_SUBTITLE = "Multi-class disease prediction with EDA, model monitoring, and batch scoring"
