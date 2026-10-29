"""Step 3 - CRUD, indexes, queries and aggregation pipelines with PyMongo."""
import json
from datetime import datetime

from config import CLEAN_COLL, CRUD_COLL, LOG_COLL, OUTPUT_DIR, QUAR_COLL, RAW_COLL
from db import get_db
import clean_data

UNWIND_MAX = 150


def crud_demo(db):
    coll = db[CRUD_COLL]
    coll.drop()
    out = {}
    coll.insert_one({"_id": "demo:pkg-a", "ecosystem": "pypi", "n_releases": 12})              # Create
    coll.insert_many([{"_id": "demo:pkg-b", "ecosystem": "npm", "n_releases": 40},
                      {"_id": "demo:pkg-c", "ecosystem": "npm", "n_releases": 3}])
    out["read_one"] = coll.find_one({"_id": "demo:pkg-a"})                                     # Read
    out["read_filtered"] = sorted(d["_id"] for d in coll.find({"n_releases": {"$gt": 10}}))
    out["updated"] = coll.update_one({"_id": "demo:pkg-a"}, {"$set": {"n_releases": 13}}).modified_count  # Update
    out["deleted"] = coll.delete_one({"_id": "demo:pkg-c"}).deleted_count                      # Delete
    out["remaining"] = coll.count_documents({})
    coll.drop()
    return out


def main(db=None):
    db = db if db is not None else get_db()
    if db[CLEAN_COLL].count_documents({}) == 0:
        clean_data.main(db)
    clean = db[CLEAN_COLL]
    res = {"crud_demo": crud_demo(db)}

    clean.create_index([("ecosystem", 1), ("name", 1)], unique=True)
    clean.create_index([("last_release", -1)])
    clean.create_index("n_releases")
    res["indexes"] = sorted(clean.index_information().keys())

    res["counts"] = {"raw": db[RAW_COLL].count_documents({}), "clean": clean.count_documents({}),
                     "quarantine": db[QUAR_COLL].count_documents({})}
    res["by_ecosystem"] = list(clean.aggregate([
        {"$group": {"_id": "$ecosystem", "packages": {"$sum": 1}, "avg_releases": {"$avg": "$n_releases"},
                    "max_releases": {"$max": "$n_releases"}}}, {"$sort": {"_id": 1}}]))
    res["single_release_packages"] = clean.count_documents({"n_releases": 1})
    res["stale_since_2024"] = clean.count_documents({"last_release": {"$lt": datetime(2024, 10, 17)}})
    res["packages_over_100_releases"] = clean.count_documents({"n_releases": {"$gt": 100}})

    # $unwind turns every release into its own record -> releases published per year and ecosystem
    # (packages with more than 150 releases are left out of this one query to keep $unwind light)
    res["unwind_max_releases"] = UNWIND_MAX
    res["packages_excluded_from_unwind"] = clean.count_documents({"n_releases": {"$gt": UNWIND_MAX}})
    res["releases_per_year"] = [
        {"year": r["_id"]["y"], "ecosystem": r["_id"]["e"], "releases": r["releases"]} for r in clean.aggregate([
            {"$match": {"n_releases": {"$lte": UNWIND_MAX}}},
            {"$unwind": "$releases"},
            {"$group": {"_id": {"y": {"$year": "$releases.released_at"}, "e": "$ecosystem"}, "releases": {"$sum": 1}}},
            {"$sort": {"_id.y": 1, "_id.e": 1}}])]

    # packages whose last release is more than two years old (potentially abandoned) per ecosystem
    res["inactive_over_2y_by_ecosystem"] = list(clean.aggregate([
        {"$match": {"last_release": {"$lt": datetime(2024, 10, 17)}}},
        {"$group": {"_id": "$ecosystem", "packages": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))

    res["top5_by_releases"] = list(clean.aggregate([
        {"$project": {"_id": 1, "n_releases": 1}}, {"$sort": {"n_releases": -1}}, {"$limit": 5}]))
    res["cleaning_log"] = db[LOG_COLL].find_one({}, {"_id": 0, "created_at": 0})

    print("[explore] CRUD:", res["crud_demo"])
    print("[explore] indexes:", res["indexes"], "| counts:", res["counts"])
    print("[explore] by ecosystem:", res["by_ecosystem"])
    print("[explore] stale since 2024:", res["stale_since_2024"], res["inactive_over_2y_by_ecosystem"])
    print("[explore] releases/year (first rows):", res["releases_per_year"][:4])
    (OUTPUT_DIR / "mongodb_exploration.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
