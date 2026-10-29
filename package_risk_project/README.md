# Predicting Open-Source Package Abandonment - MongoDB + PyMongo + Machine Learning

Release histories of **9,238 PyPI and npm packages** (298,821 releases, collected 17 Oct 2026) are stored in
MongoDB, cleaned with an audit trail, analysed with aggregation pipelines (`$unwind`), and used to predict which
packages will stop publishing releases within the next 24 months - with an explanation for every score.

**Why someone would use it:** a team can scan its dependencies and see which ones are likely to be abandoned,
and why, before they cause problems.

| Task | Target | Models | Metrics |
|------|--------|--------|---------|
| Classification | Package publishes **no** release in the next 24 months ("abandoned") | Logistic Regression, Decision Tree, Random Forest (+ rule baseline) | Accuracy, Precision, Recall, F1, ROC-AUC |
| Regression | Number of releases in the next 24 months (capped at 50) | Linear Regression, Decision Tree, Random Forest (+ persistence baseline) | R2, MAE, RMSE |

No future leakage: features use only releases *before* the snapshot date; train = snapshots 2021-10-17 / 2022-10-17
of 70% of the packages, test = snapshot 2024-10-17 of the other 30% (unseen packages, later time).

## Run
```bash
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017"     # PowerShell: $env:MONGO_URI="mongodb://localhost:27017"
python run_all.py                                # ~5-10 minutes
python collect_data.py                           # optional: re-download the data from PyPI/npm
```
If no MongoDB server is reachable, `db.py` falls back to the in-memory `mongomock` emulator (same PyMongo API).

## MongoDB collections (database `package_risk_analytics`)
`packages_raw`, `packages_clean`, `packages_quarantine`, `cleaning_log`, `model_features`,
`model_results`, `predictions`, `risk_scores`

## Notes
* Abandonment = "no release in 24 months" is a proxy (finished packages may need no updates).
* The npm sample is search-based (biased to discoverable packages); deleted packages are not in the data.
* Risk scores are statistical indicators from release history only, not judgements about any project.
