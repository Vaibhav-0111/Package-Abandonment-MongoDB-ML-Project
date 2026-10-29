"""Central configuration - Open-source package abandonment predictor (MongoDB + ML)."""
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "package_releases.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- MongoDB -----------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "package_risk_analytics"
RAW_COLL = "packages_raw"              # one document per package, release dates as downloaded
CLEAN_COLL = "packages_clean"          # validated, sorted release timelines
QUAR_COLL = "packages_quarantine"      # packages rejected by the cleaning step
LOG_COLL = "cleaning_log"
FEATURE_COLL = "model_features"        # one document per (package, snapshot date)
RESULTS_COLL = "model_results"
PRED_COLL = "predictions"
RISK_COLL = "risk_scores"              # abandonment risk of packages that are active today
CRUD_COLL = "crud_demo"

# --- Problem definition ------------------------------------------------
HORIZON_DAYS = 730                     # "abandoned" = no release during the next 24 months
ACTIVE_WINDOW_DAYS = 730               # only packages with a release in the 24 months before the snapshot
MIN_AGE_DAYS = 180
MIN_RELEASES = 2
RELEASE_CAP = 50                       # regression target = releases in next 24 months, capped at 50
TRAIN_SNAPSHOTS = ["2021-10-17", "2022-10-17"]
TEST_SNAPSHOT = "2024-10-17"           # its 24-month outcome window ends on the collection date
TEST_FRACTION = 0.30                   # share of packages reserved (as whole packages) for testing

# --- Machine learning --------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
