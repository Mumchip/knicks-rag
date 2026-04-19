"""
init_db.py — Run on deploy. Builds the DB if empty, skips if already populated.
"""
import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
try:
    col = chroma.get_collection("knicks")
    count = col.count()
except Exception:
    count = 0

if count > 100:
    print(f"DB already populated ({count} docs). Skipping ingest.")
else:
    print(f"DB empty ({count} docs). Running ingest...")
    import subprocess, sys
    for script in ["ingest_full.py", "ingest_current.py", "build_summaries.py", "add_roster_doc.py"]:
        print(f"Running {script}...")
        subprocess.run([sys.executable, script], check=True)
    print("DB initialized.")
