"""Step 2 - Validate release timelines and write the clean layer.

Problems handled (see report, section 5):
  1. Unparseable publish dates and dates outside 1995 .. collection date (clock errors).
  2. Duplicate version entries.
  3. Packages without any valid release -> quarantined.
The raw layer is never modified.
"""
from datetime import datetime, timezone

import pandas as pd

from config import CLEAN_COLL, LOG_COLL, QUAR_COLL, RAW_COLL
from db import get_db
import load_data

MIN_DATE = pd.Timestamp("1995-01-01")


def parse_dates(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def main(db=None):
    db = db if db is not None else get_db()
    if db[RAW_COLL].count_documents({}) == 0:
        load_data.main(db)

    log = {"raw_packages": 0, "raw_releases": 0, "unparseable_dates": 0, "dates_out_of_range": 0,
           "duplicate_versions": 0}
    clean_docs, quarantine = [], []
    collected = None
    for raw in db[RAW_COLL].find({}):
        collected = raw["metadata"]["collected_at"]
        max_date = pd.Timestamp(collected).tz_convert(None) if pd.Timestamp(collected).tzinfo else pd.Timestamp(collected)
        df = pd.DataFrame(raw["releases"])
        log["raw_packages"] += 1
        log["raw_releases"] += len(df)
        df["released_at"] = parse_dates(df["published"])
        log["unparseable_dates"] += int(df["released_at"].isna().sum())
        in_range = df["released_at"].between(MIN_DATE, max_date)
        log["dates_out_of_range"] += int((~in_range & df["released_at"].notna()).sum())
        df = df[in_range]
        before = len(df)
        df = df.sort_values("released_at").drop_duplicates("version", keep="first")
        log["duplicate_versions"] += before - len(df)
        if df.empty:
            quarantine.append({"_id": raw["_id"], "ecosystem": raw["ecosystem"], "name": raw["name"],
                               "reason": "no valid release date"})
            continue
        rel = [{"version": v, "released_at": t.to_pydatetime()} for v, t in zip(df["version"], df["released_at"])]
        clean_docs.append({
            "_id": raw["_id"], "ecosystem": raw["ecosystem"], "name": raw["name"], "releases": rel,
            "n_releases": len(rel), "first_release": rel[0]["released_at"], "last_release": rel[-1]["released_at"],
            "metadata": {"cleaned_at": datetime.now(timezone.utc), "collected_at": collected}})

    db[CLEAN_COLL].drop()
    for i in range(0, len(clean_docs), 1000):
        db[CLEAN_COLL].insert_many(clean_docs[i:i + 1000])
    db[QUAR_COLL].drop()
    if quarantine:
        db[QUAR_COLL].insert_many(quarantine)
    log.update({"clean_packages": len(clean_docs), "quarantined_packages": len(quarantine),
                "collected_at": collected,
                "single_release_packages": int(sum(d["n_releases"] == 1 for d in clean_docs))})
    db[LOG_COLL].drop()
    db[LOG_COLL].insert_one({"_id": "cleaning-run", "created_at": datetime.now(timezone.utc), **log})
    print("[clean]", log)
    return log


if __name__ == "__main__":
    main()
