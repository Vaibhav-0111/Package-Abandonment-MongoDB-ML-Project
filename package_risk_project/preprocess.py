"""Step 4 - Turn release timelines into one training row per (package, snapshot date).

For a snapshot date T0 only releases published BEFORE T0 are used as features. The outcome is read from
the following 24 months, so the model never sees the future it has to predict.
"""
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (ACTIVE_WINDOW_DAYS, CLEAN_COLL, FEATURE_COLL, FIG_DIR, HORIZON_DAYS, MIN_AGE_DAYS,
                    MIN_RELEASES, RANDOM_STATE, RELEASE_CAP, TEST_FRACTION, TEST_SNAPSHOT, TRAIN_SNAPSHOTS)
from db import get_db
import clean_data

sns.set_theme(style="whitegrid", context="notebook")
FEATURES = ["age_days", "n_releases", "since_last_days", "n_90d", "n_180d", "n_365d", "n_730d", "trend_ratio",
            "active_months_24m", "mean_gap", "median_gap", "std_gap", "max_gap", "last_gap", "overdue_ratio",
            "overdue_ratio_recent", "prerelease_share_recent", "major_version", "n_major_versions",
            "releases_per_year_life", "is_stable", "is_npm"]
BINARY = ["is_stable", "is_npm"]
PRE = re.compile(r"(?i)(alpha|beta|rc|dev|pre|canary|next|nightly|snapshot|preview)|\d[ab]\d")
MAJOR = re.compile(r"v?(\d+)")


def _major(v):
    m = MAJOR.match(v)
    return int(m.group(1)) if m else 0


def days_since_epoch(ts):
    return np.datetime64(ts, "s").astype("int64") / 86400.0


def snapshot_row(days, versions, t0, with_target=True):
    """Features (and outcome) of one package at snapshot t0 (in days since epoch). None if not eligible."""
    before = days < t0
    n = int(before.sum())
    if n < MIN_RELEASES:
        return None
    d, v = days[before], versions[before]
    age, since_last = t0 - d[0], t0 - d[-1]
    if age < MIN_AGE_DAYS or since_last > ACTIVE_WINDOW_DAYS:
        return None
    gaps = np.diff(d)
    recent_gap = float(np.median(gaps[-5:]))
    last24 = d[d >= t0 - ACTIVE_WINDOW_DAYS]
    n365 = int((d >= t0 - 365).sum())
    prev365 = int(((d >= t0 - 730) & (d < t0 - 365)).sum())
    recent_versions = v[-10:]
    last_major = _major(v[-1])
    row = {
        "age_days": age, "n_releases": n, "since_last_days": since_last,
        "n_90d": int((d >= t0 - 90).sum()), "n_180d": int((d >= t0 - 180).sum()), "n_365d": n365,
        "n_730d": int(len(last24)), "trend_ratio": n365 / (prev365 + 1),
        "active_months_24m": int(np.unique(((t0 - last24) // 30).astype(int)).size),
        "mean_gap": float(gaps.mean()), "median_gap": float(np.median(gaps)), "std_gap": float(gaps.std()),
        "max_gap": float(gaps.max()), "last_gap": float(gaps[-1]),
        "overdue_ratio": since_last / (float(np.median(gaps)) + 1),
        "overdue_ratio_recent": since_last / (recent_gap + 1),
        "prerelease_share_recent": float(np.mean([bool(PRE.search(x)) for x in recent_versions])),
        "major_version": last_major, "n_major_versions": len({_major(x) for x in v}),
        "releases_per_year_life": n / max(age / 365.0, 0.5), "is_stable": int(last_major >= 1),
    }
    if with_target:
        nxt = int(((days >= t0) & (days < t0 + HORIZON_DAYS)).sum())
        row["releases_next_24m"] = nxt
        row["releases_next_24m_capped"] = min(nxt, RELEASE_CAP)
        row["abandoned"] = int(nxt == 0)
    return row


def build_feature_documents(db):
    """Read the clean layer, compute snapshot rows, and write them to the model_features collection."""
    rng = np.random.RandomState(RANDOM_STATE)
    docs, collected = [], None
    pkgs = list(db[CLEAN_COLL].find({}, {"ecosystem": 1, "name": 1, "releases": 1, "metadata": 1}))
    collected = pkgs[0]["metadata"]["collected_at"]
    today = days_since_epoch(pd.Timestamp(collected).tz_convert(None).to_datetime64()) if pd.Timestamp(collected).tzinfo else days_since_epoch(pd.Timestamp(collected).to_datetime64())
    # package-level split: whole packages are reserved for the test snapshot (no package is in both sets)
    order = rng.permutation(len(pkgs))
    is_test = np.zeros(len(pkgs), dtype=bool)
    for eco in {p["ecosystem"] for p in pkgs}:
        idx = [i for i in order if pkgs[i]["ecosystem"] == eco]
        is_test[idx[: int(len(idx) * TEST_FRACTION)]] = True
    snaps = {s: days_since_epoch(np.datetime64(s)) for s in TRAIN_SNAPSHOTS + [TEST_SNAPSHOT]}
    for i, p in enumerate(pkgs):
        days = np.array([r["released_at"] for r in p["releases"]], dtype="datetime64[s]").astype("int64") / 86400.0
        versions = np.array([r["version"] for r in p["releases"]])
        todo = ([(TEST_SNAPSHOT, "test")] if is_test[i] else [(s, "train") for s in TRAIN_SNAPSHOTS])
        for snap, split in todo:
            row = snapshot_row(days, versions, snaps[snap], with_target=True)
            if row:
                row["is_npm"] = int(p["ecosystem"] == "npm")
                docs.append({"_id": f"{p['_id']}@{snap}", "package": p["_id"], "name": p["name"],
                             "ecosystem": p["ecosystem"], "snapshot": snap, "split": split,
                             "features": {k: row[k] for k in FEATURES},
                             "targets": {k: row[k] for k in ("releases_next_24m", "releases_next_24m_capped", "abandoned")}})
        row = snapshot_row(days, versions, today, with_target=False)       # "today": no outcome known yet
        if row:
            row["is_npm"] = int(p["ecosystem"] == "npm")
            docs.append({"_id": f"{p['_id']}@today", "package": p["_id"], "name": p["name"], "ecosystem": p["ecosystem"],
                         "snapshot": "today", "split": "current", "features": {k: row[k] for k in FEATURES}, "targets": {}})
    db[FEATURE_COLL].drop()
    for i in range(0, len(docs), 1000):
        db[FEATURE_COLL].insert_many(docs[i:i + 1000])
    return len(docs)


def load_feature_frame(db):
    docs = list(db[FEATURE_COLL].find({}))
    df = pd.json_normalize(docs, sep="_")
    df.columns = [c.replace("features_", "").replace("targets_", "") for c in df.columns]
    return df.drop(columns=["_id"])


def prepare(db):
    if db[FEATURE_COLL].count_documents({}) == 0:
        build_feature_documents(db)
    df = load_feature_frame(db)
    train, test, current = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "test", "current"))
    rate = lambda d: round(float(100 * d["abandoned"].mean()), 1)
    report = {
        "feature_rows": int(len(df)), "n_features": len(FEATURES), "features": FEATURES,
        "train_rows": int(len(train)), "train_packages": int(train["package"].nunique()),
        "test_rows": int(len(test)), "test_packages": int(test["package"].nunique()),
        "current_rows": int(len(current)),
        "train_snapshots": TRAIN_SNAPSHOTS, "test_snapshot": TEST_SNAPSHOT,
        "abandoned_pct_train": rate(train), "abandoned_pct_test": rate(test),
        "abandoned_test_count": int(test["abandoned"].sum()),
        "abandoned_pct_by_ecosystem_test": {e: rate(g) for e, g in test.groupby("ecosystem")},
        "abandoned_pct_by_ecosystem_train": {e: rate(g) for e, g in train.groupby("ecosystem")},
        "rows_by_ecosystem_train": train["ecosystem"].value_counts().to_dict(),
        "rows_by_ecosystem_test": test["ecosystem"].value_counts().to_dict(),
        "releases_next_mean_test": round(float(test["releases_next_24m_capped"].mean()), 2),
        "releases_next_std_test": round(float(test["releases_next_24m_capped"].std()), 2),
        "capped_share_train_pct": round(float(100 * (train["releases_next_24m"] > RELEASE_CAP).mean()), 2),
    }
    return {"df": df, "train": train, "test": test, "current": current, "features": FEATURES, "report": report}


def save_eda_figures(data, explore_res=None):
    tr = data["train"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    if explore_res:
        rpy = pd.DataFrame(explore_res["releases_per_year"])
        rpy = rpy[(rpy["year"] >= 2010) & (rpy["year"] <= 2026)]
        for eco, g in rpy.groupby("ecosystem"):
            ax[0].plot(g["year"], g["releases"], marker="o", label=eco)
        ax[0].set_title("Releases published per year (packages with <=150 releases)")
        ax[0].set_ylabel("Releases"); ax[0].legend()
    for lab, color, name in [(0, "#2A788E", "still active"), (1, "#C0392B", "abandoned")]:
        ax[1].hist(np.log10(tr.loc[tr["abandoned"] == lab, "since_last_days"] + 1), bins=30, alpha=0.65, color=color, label=name)
    ax[1].set_title("Days since last release at snapshot (log10), by outcome"); ax[1].set_xlabel("log10(days + 1)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_trends_and_gap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    bins = [0, 30, 90, 180, 365, 540, 731]
    tr = tr.assign(gap_bin=pd.cut(tr["since_last_days"], bins, include_lowest=True))
    tab = tr.groupby(["gap_bin", "ecosystem"], observed=True)["abandoned"].mean().unstack() * 100
    tab.plot.bar(ax=ax[0], color=["#7B3F98", "#2A788E"]); ax[0].set_ylabel("% abandoned in next 24 months")
    ax[0].set_title("Abandonment rate by days since last release"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=30)
    tr = tr.assign(n_bin=pd.cut(tr["n_365d"], [-1, 0, 1, 3, 8, 10_000], labels=["0", "1", "2-3", "4-8", "9+"]))
    tab2 = tr.groupby(["n_bin", "ecosystem"], observed=True)["abandoned"].mean().unstack() * 100
    tab2.plot.bar(ax=ax[1], color=["#7B3F98", "#2A788E"]); ax[1].set_title("Abandonment rate by releases in the last 12 months")
    ax[1].set_xlabel("Releases in last 365 days"); ax[1].set_ylabel(""); ax[1].tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_abandonment_rates.png", dpi=150); plt.close(fig)


def main(db=None, explore_res=None):
    db = db if db is not None else get_db()
    if db[CLEAN_COLL].count_documents({}) == 0:
        clean_data.main(db)
    build_feature_documents(db)
    data = prepare(db)
    save_eda_figures(data, explore_res)
    print("[preprocess]", {k: v for k, v in data["report"].items() if k != "features"})
    return data


if __name__ == "__main__":
    main()
