"""
cleanup_old_season.py — Remove old season player game logs and box scores from Chroma.
Run once to clean up 2024-25 docs before re-ingesting 2025-26.
"""
import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")

all_ids = col.get(limit=10000)["ids"]
old_ids = [id_ for id_ in all_ids if id_.startswith("pglog_") or id_.startswith("boxscore_")]

if old_ids:
    col.delete(ids=old_ids)
    print(f"Deleted {len(old_ids)} old season docs.")
else:
    print("Nothing to delete.")

print(f"Remaining: {col.count()} documents.")
