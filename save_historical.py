"""
save_historical.py — Export all historical docs from Chroma to a permanent JSON backup.
Run once. After this, historical data never needs to be re-fetched from the NBA API.
data/historical_db.json is the source of truth for all past data.
"""
import json
import chromadb

CURRENT_SEASON = "2025-26"
BACKUP_FILE = "data/historical_db.json"

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")

print("Reading all documents from Chroma...")
all_data = col.get(limit=10000, include=["documents", "embeddings"])

historical_ids = []
historical_docs = []
historical_embeddings = []

for id_, doc, emb in zip(all_data["ids"], all_data["documents"], all_data["embeddings"]):
    # Skip current season summaries and live data — these get refreshed
    if id_.startswith("summary_"):
        continue
    if CURRENT_SEASON in doc and (id_.startswith("pglog_") or id_.startswith("boxscore_")):
        continue
    historical_ids.append(id_)
    historical_docs.append(doc)
    historical_embeddings.append(emb if isinstance(emb, list) else emb.tolist())

backup = {
    "ids": historical_ids,
    "documents": historical_docs,
    "embeddings": historical_embeddings,
}

with open(BACKUP_FILE, "w") as f:
    json.dump(backup, f)

print(f"Saved {len(historical_ids)} historical documents to {BACKUP_FILE}")
print("This file is now the permanent source of truth for historical data.")
print("Never delete it. It will never need to be re-fetched from the NBA API.")
