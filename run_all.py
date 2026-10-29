"""Run the complete pipeline: load -> clean -> explore -> features -> train/evaluate -> risk scores."""
from db import get_db
import clean_data, explore_data, load_data, preprocess, train_evaluate

if __name__ == "__main__":
    db = get_db()                              # one connection shared by every step
    load_data.main(db)                         # 1. JSONL -> raw documents
    clean_data.main(db)                        # 2. raw -> clean documents (+ quarantine, cleaning log)
    ex = explore_data.main(db)                 # 3. CRUD, indexes, queries, aggregations
    data = preprocess.main(db, ex)             # 4. snapshot features stored in MongoDB
    train_evaluate.main(db, data)              # 5. models, metrics, risk scores -> MongoDB
    print("\nDone. See the 'outputs' folder for figures, CSV and JSON results.")
