from db_path import CHROMA_PATH
"""
rebuild_db.py — Rebuild Chroma from scratch using locked historical data + fresh current season.
Use this if chroma_db ever gets corrupted or you need to start fresh.

Steps:
1. Loads data/historical_db.json (permanent, never re-fetched)
2. Loads data/player_game_logs_2025.json (current season player logs)
3. Loads data/box_scores_2025.json (current season box scores)
4. Re-embeds and stores everything
5. Runs build_summaries.py to add fresh summary docs
"""
import json
import os
import chromadb
from embed_utils import embedder

chroma = chromadb.PersistentClient(path=CHROMA_PATH)

# Wipe and recreate
try:
    chroma.delete_collection("knicks")
    print("Deleted existing collection.")
except Exception:
    pass
col = chroma.create_collection("knicks")

# ── Load historical (pre-embedded, fast) ──────────────────────────────────
print("Loading historical data...")
with open("data/historical_db.json") as f:
    hist = json.load(f)

batch_size = 500
for i in range(0, len(hist["ids"]), batch_size):
    col.add(
        ids=hist["ids"][i:i+batch_size],
        documents=hist["documents"][i:i+batch_size],
        embeddings=hist["embeddings"][i:i+batch_size],
    )
print(f"Restored {len(hist['ids'])} historical documents.")

# ── Load current season data ───────────────────────────────────────────────
current_docs = []
for cache_file in ["data/player_game_logs_2025.json", "data/box_scores_2025.json"]:
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            current_docs.extend(json.load(f))
        print(f"Loaded {cache_file}")

if current_docs:
    print(f"Embedding {len(current_docs)} current season documents...")
    texts = [d["text"] for d in current_docs]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    for i in range(0, len(current_docs), batch_size):
        col.upsert(
            ids=[d["id"] for d in current_docs[i:i+batch_size]],
            documents=texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
        )
    print(f"Added {len(current_docs)} current season documents.")

print(f"\nTotal documents in DB: {col.count()}")
print("Now run: python build_summaries.py")
