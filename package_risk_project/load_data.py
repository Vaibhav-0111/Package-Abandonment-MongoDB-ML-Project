"""Step 1 - Load the downloaded release histories into MongoDB (raw layer)."""
import json
from datetime import datetime

from config import DATA_FILE, RAW_COLL
from db import get_db


def build_raw_documents():
    docs = []
    with open(DATA_FILE) as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append({
                "_id": f"{rec['ecosystem']}:{rec['name']}",
                "ecosystem": rec["ecosystem"], "name": rec["name"],
                # dates are kept exactly as the registry returned them (strings) in the raw layer
                "releases": [{"version": v, "published": t} for v, t in rec["releases"]],
                "metadata": {"source": "registry API", "collected_at": rec["collected_at"]},
            })
    return docs


def main(db=None):
    db = db if db is not None else get_db()
    coll = db[RAW_COLL]
    coll.drop()
    docs = build_raw_documents()
    for i in range(0, len(docs), 1000):                 # insert in batches of 1000 documents
        coll.insert_many(docs[i:i + 1000])
    print(f"[load] {RAW_COLL}: inserted {len(docs)} raw package documents")
    return len(docs)


if __name__ == "__main__":
    main()
