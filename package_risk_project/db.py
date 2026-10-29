"""MongoDB connection helper (PyMongo).

Connects to a real MongoDB server (local or Atlas) using MONGO_URI.
If no server can be reached, it falls back to `mongomock`, an in-memory
emulator of the PyMongo API, so the whole pipeline still runs end-to-end.
"""
import os

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import DB_NAME, MONGO_URI


def get_db(force_mock: bool = False):
    """Return a database handle (real MongoDB if available, else mongomock)."""
    if not force_mock and os.environ.get("USE_MOCK_MONGO") != "1":
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")          # fails fast if server is down
            print(f"[db] Connected to MongoDB at {MONGO_URI} (database: {DB_NAME})")
            return client[DB_NAME]
        except PyMongoError as exc:
            print(f"[db] WARNING: could not reach MongoDB at {MONGO_URI} ({type(exc).__name__}).")

    import mongomock  # imported lazily - only needed for the fallback
    print("[db] Using in-memory mongomock (PyMongo-compatible). Start a MongoDB server "
          "and re-run to use a real database.")
    return mongomock.MongoClient()[DB_NAME]
