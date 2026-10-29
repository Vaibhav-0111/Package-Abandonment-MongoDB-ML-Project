"""Step 5 - Train, tune and evaluate models; score today's packages; store everything in MongoDB."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import CV_FOLDS, FIG_DIR, OUTPUT_DIR, PRED_COLL, RANDOM_STATE, RELEASE_CAP, RESULTS_COLL, RISK_COLL
from db import get_db
import preprocess
from preprocess import BINARY, FEATURES

sns.set_theme(style="whitegrid", context="notebook")
RS = RANDOM_STATE
LOG_COLS = [f for f in FEATURES if f not in BINARY]
PRETTY = {
    "age_days": "package age", "n_releases": "total releases", "since_last_days": "time since last release",
    "n_90d": "releases in last 90 days", "n_180d": "releases in last 180 days", "n_365d": "releases in last 12 months",
    "n_730d": "releases in last 24 months", "trend_ratio": "release trend (last 12m vs previous 12m)",
    "active_months_24m": "active months in last 24", "mean_gap": "mean gap between releases",
    "median_gap": "median gap between releases", "std_gap": "variability of gaps", "max_gap": "longest gap ever",
    "last_gap": "most recent gap", "overdue_ratio": "time since last release vs usual gap",
    "overdue_ratio_recent": "time since last release vs recent gap", "prerelease_share_recent": "share of recent pre-releases",
    "major_version": "major version", "n_major_versions": "number of major versions",
    "releases_per_year_life": "lifetime releases per year", "is_stable": "version >= 1.0", "is_npm": "npm package"}


def _clean(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def scaled_prep():
    """log1p (heavy-tailed counts and day-gaps) + standardisation, used by the linear models only."""
    log_part = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    return ColumnTransformer([("log", log_part, LOG_COLS), ("bin", StandardScaler(), BINARY)])


# ----------------------------------------------------------------------
def cls_metrics(y, pred, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    if proba is not None:
        m["roc_auc"] = roc_auc_score(y, proba)
    return m


def reg_metrics(y, pred):
    return {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def run_classification(data):
    tr, te = data["train"], data["test"]
    X_tr, y_tr, X_te, y_te = tr[FEATURES], tr["abandoned"], te[FEATURES], te["abandoned"]
    cv = GroupKFold(CV_FOLDS)
    models = {
        "Logistic Regression": (Pipeline([("prep", scaled_prep()), ("model", LogisticRegression(max_iter=5000, random_state=RS))]),
                                {"model__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (Pipeline([("model", DecisionTreeClassifier(random_state=RS))]),
                          {"model__max_depth": [3, 4, 5, 7, 10], "model__min_samples_leaf": [5, 10, 20, 40]}),
        "Random Forest": (Pipeline([("model", RandomForestClassifier(random_state=RS, n_jobs=-1))]),
                          {"model__n_estimators": [300], "model__max_depth": [5, 8, 12, None],
                           "model__min_samples_leaf": [2, 5, 10]}),
    }
    base_pred = (te["since_last_days"] > 365).astype(int)
    results = [{"task": "classification", "model": "Baseline: no release in last 12 months", "baseline": True,
                **cls_metrics(y_te, base_pred)}]
    fitted, probas = {}, {}
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X_tr, y_tr, groups=tr["package"])
        best = gs.best_estimator_
        pred, proba = best.predict(X_te), best.predict_proba(X_te)[:, 1]
        results.append({"task": "classification", "model": name, "baseline": False, **cls_metrics(y_te, pred, proba),
                        "cv_f1_mean": gs.best_score_, "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
        fitted[name], probas[name] = best, proba
    for r in results:
        print(f"[classification] {r['model']:40s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1']:.4f}")
    best_name = max([r for r in results if not r["baseline"]], key=lambda r: (r["f1"], r["accuracy"]))["model"]
    (OUTPUT_DIR / "classification_report_best.txt").write_text(
        f"Best model: {best_name}\n\n" + classification_report(y_te, fitted[best_name].predict(X_te), target_names=["still active", "abandoned"]))
    return results, fitted, probas, best_name, base_pred


def run_regression(data):
    tr, te = data["train"], data["test"]
    X_tr, y_tr, X_te, y_te = tr[FEATURES], tr["releases_next_24m_capped"], te[FEATURES], te["releases_next_24m_capped"]
    cv = GroupKFold(CV_FOLDS)
    models = {
        "Linear Regression": (Pipeline([("prep", scaled_prep()), ("model", LinearRegression())]), {}),
        "Decision Tree": (Pipeline([("model", DecisionTreeRegressor(random_state=RS))]),
                          {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [5, 10, 20, 40]}),
        "Random Forest": (Pipeline([("model", RandomForestRegressor(random_state=RS, n_jobs=-1))]),
                          {"model__n_estimators": [300], "model__max_depth": [5, 8, 12, None],
                           "model__min_samples_leaf": [2, 5, 10]}),
    }
    base = np.minimum(te["n_730d"].to_numpy(), RELEASE_CAP)
    results = [{"task": "regression", "model": "Baseline: releases in last 24 months", "baseline": True, **reg_metrics(y_te, base)}]
    fitted, preds = {}, {"Baseline: releases in last 24 months": base}
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2", n_jobs=-1).fit(X_tr, y_tr, groups=tr["package"])
        best = gs.best_estimator_
        pred = np.clip(best.predict(X_te), 0, RELEASE_CAP)
        results.append({"task": "regression", "model": name, "baseline": False, **reg_metrics(y_te, pred), "cv_r2_mean": gs.best_score_,
                        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
        fitted[name], preds[name] = best, pred
    for r in results:
        print(f"[regression]     {r['model']:40s} R2={r['r2']:.4f} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f}")
    best_name = max([r for r in results if not r["baseline"]], key=lambda r: r["r2"])["model"]
    return results, fitted, preds, best_name


# ----------------------------------------------------------------------
def lr_contributions(lr_pipe, X):
    """Per-package contribution of every feature to the abandonment log-odds (coefficient x scaled value)."""
    Xt = lr_pipe.named_steps["prep"].transform(X)
    return Xt * lr_pipe.named_steps["model"].coef_[0], LOG_COLS + BINARY


def score_current(db, data, c_fit, c_best):
    cur = data["current"].copy()
    proba = c_fit[c_best].predict_proba(cur[FEATURES])[:, 1]
    contrib, names = lr_contributions(c_fit["Logistic Regression"], cur[FEATURES])
    docs = []
    for i, row in cur.reset_index(drop=True).iterrows():
        top = np.argsort(-np.abs(contrib[i]))[:3]
        reasons = [{"factor": PRETTY[names[j]], "value": round(float(row[names[j]]), 2),
                    "effect": "raises risk" if contrib[i][j] > 0 else "lowers risk", "weight": round(float(contrib[i][j]), 3)} for j in top]
        docs.append({"_id": row["package"], "name": row["name"], "ecosystem": row["ecosystem"], "risk": round(float(proba[i]), 4),
                     "risk_band": "high" if proba[i] >= 0.66 else "medium" if proba[i] >= 0.33 else "low", "reasons": reasons})
    db[RISK_COLL].drop()
    for k in range(0, len(docs), 1000):
        db[RISK_COLL].insert_many(docs[k:k + 1000])
    return docs, contrib, names, cur.reset_index(drop=True), proba


def save_result_figures(data, c_fit, c_proba, r_fit, r_pred, risk_docs, contrib, names, cur, proba):
    te = data["test"]
    y_c = te["abandoned"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, model) in zip(axes, c_fit.items()):
        cm = confusion_matrix(y_c, model.predict(te[FEATURES]))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax, xticklabels=["active", "abandoned"], yticklabels=["active", "abandoned"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_confusion_matrices.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in c_proba.items():
        fpr, tpr, _ = roc_curve(y_c, p); ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_c, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - abandonment classification"); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, fit, title, color in [(ax[0], c_fit, "Abandonment classifier (RF)", "#7B3F98"), (ax[1], r_fit, "Release-count regressor (RF)", "#2A788E")]:
        imp = pd.Series(fit["Random Forest"].named_steps["model"].feature_importances_, index=FEATURES).sort_values().tail(10)
        imp.rename(index=PRETTY).plot.barh(ax=a, color=color); a.set_title(title + " - top 10"); a.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_feature_importance.png", dpi=150); plt.close(fig)

    y_r = te["releases_next_24m_capped"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, pred) in zip(axes, r_pred.items()):
        ax.scatter(y_r, pred, alpha=0.35, s=14, color="#2A788E"); ax.plot([0, RELEASE_CAP], [0, RELEASE_CAP], "k--", lw=1)
        ax.set_title(f"{name}\nR² = {r2_score(y_r, pred):.3f}, MAE = {mean_absolute_error(y_r, pred):.2f}", fontsize=9)
        ax.set_xlabel("Actual releases in next 24 months"); ax.set_ylabel("Predicted")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_actual_vs_predicted.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for eco, color in [("pypi", "#7B3F98"), ("npm", "#2A788E")]:
        ax[0].hist(proba[(cur["ecosystem"] == eco).to_numpy()], bins=20, alpha=0.65, color=color, label=eco)
    ax[0].set_title("Predicted abandonment risk of packages that are active today"); ax[0].set_xlabel("Predicted probability"); ax[0].legend()
    i = int(np.argmax(proba))
    order = np.argsort(np.abs(contrib[i]))[::-1][:8][::-1]
    colors = ["#C0392B" if contrib[i][j] > 0 else "#2A788E" for j in order]
    ax[1].barh([PRETTY[names[j]] for j in order], [contrib[i][j] for j in order], color=colors)
    ax[1].set_title(f"Why one package is flagged (risk {proba[i]:.2f})")
    ax[1].set_xlabel("Contribution to log-odds of abandonment")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig7_risk_scores_and_explanation.png", dpi=150); plt.close(fig)
    return i


def store_in_mongodb(db, c_results, r_results, data, c_fit, c_best, r_pred, r_best):
    te = data["test"]
    db[RESULTS_COLL].delete_many({})
    db[RESULTS_COLL].insert_many(_clean(c_results + r_results))
    db[PRED_COLL].delete_many({})
    c_prob = c_fit[c_best].predict_proba(te[FEATURES])[:, 1]
    docs = [{"package": p, "task": "classification", "model": c_best, "actual": int(a), "risk": round(float(pr), 4)}
            for p, a, pr in zip(te["package"], te["abandoned"], c_prob)]
    docs += [{"package": p, "task": "regression", "model": r_best, "actual": int(a), "predicted": round(float(pr), 3)}
             for p, a, pr in zip(te["package"], te["releases_next_24m_capped"], r_pred[r_best])]
    for k in range(0, len(docs), 1000):
        db[PRED_COLL].insert_many(docs[k:k + 1000])
    print(f"[mongodb] stored {len(c_results) + len(r_results)} result docs and {len(docs)} prediction docs")
    return len(docs)


def main(db=None, data=None):
    db = db if db is not None else get_db()
    data = data if data is not None else preprocess.main(db)
    c_results, c_fit, c_proba, c_best, base_pred = run_classification(data)
    r_results, r_fit, r_pred, r_best = run_regression(data)
    risk_docs, contrib, names, cur, proba = score_current(db, data, c_fit, c_best)
    ex_idx = save_result_figures(data, c_fit, c_proba, r_fit, r_pred, risk_docs, contrib, names, cur, proba)
    n_pred = store_in_mongodb(db, c_results, r_results, data, c_fit, c_best, r_pred, r_best)
    print("[mongodb] read back:", list(db[RESULTS_COLL].find({}, {"_id": 0, "task": 1, "model": 1, "f1": 1, "r2": 1}))[:3], "...")

    te = data["test"]
    by_eco = {}
    best_pred = c_fit[c_best].predict(te[FEATURES])
    for eco, g in te.groupby("ecosystem"):
        idx = te.index.get_indexer(g.index)
        by_eco[eco] = {"rows": int(len(g)), "abandoned_pct": round(float(100 * g["abandoned"].mean()), 1),
                       "best_model_f1": f1_score(g["abandoned"], best_pred[idx]), "best_model_accuracy": accuracy_score(g["abandoned"], best_pred[idx]),
                       "baseline_f1": f1_score(g["abandoned"], base_pred.to_numpy()[idx]), "baseline_accuracy": accuracy_score(g["abandoned"], base_pred.to_numpy()[idx])}
    risk_df = pd.DataFrame([{"package": d["_id"], "ecosystem": d["ecosystem"], "risk": d["risk"], "band": d["risk_band"],
                             "reason_1": f"{d['reasons'][0]['factor']} ({d['reasons'][0]['effect']})",
                             "reason_2": f"{d['reasons'][1]['factor']} ({d['reasons'][1]['effect']})",
                             "reason_3": f"{d['reasons'][2]['factor']} ({d['reasons'][2]['effect']})"} for d in risk_docs]).sort_values("risk", ascending=False)
    risk_df.to_csv(OUTPUT_DIR / "risk_scores_all.csv", index=False)
    ex = risk_docs[ex_idx]
    top_features = {}
    for key, fit in (("classification", c_fit), ("regression", r_fit)):
        imp = pd.Series(fit["Random Forest"].named_steps["model"].feature_importances_, index=FEATURES)
        top_features[key] = imp.sort_values(ascending=False).head(5).round(4).to_dict()
    lr_pipe = c_fit["Logistic Regression"]
    coefs = pd.Series(lr_pipe.named_steps["model"].coef_[0], index=LOG_COLS + BINARY).sort_values(key=np.abs, ascending=False).head(6).round(3).to_dict()
    summary = {"top_features": top_features, "lr_top_coefficients": coefs, "classification": c_results, "regression": r_results,
               "best_classifier": c_best, "best_regressor": r_best, "by_ecosystem_test": by_eco,
               "risk_bands": risk_df["band"].value_counts().to_dict(),
               "risk_bands_by_ecosystem": risk_df.groupby(["ecosystem", "band"]).size().unstack(fill_value=0).to_dict("index"),
               "example_explanation": {"risk": ex["risk"], "ecosystem": ex["ecosystem"], "reasons": ex["reasons"]},
               "prediction_docs_stored": n_pred, "preprocessing": data["report"]}
    (OUTPUT_DIR / "results.json").write_text(json.dumps(_clean(summary), indent=2))
    pd.DataFrame(c_results).drop(columns=["best_params"], errors="ignore").round(4).to_csv(OUTPUT_DIR / "classification_results.csv", index=False)
    pd.DataFrame(r_results).drop(columns=["best_params"], errors="ignore").round(4).to_csv(OUTPUT_DIR / "regression_results.csv", index=False)
    return summary


if __name__ == "__main__":
    main()
