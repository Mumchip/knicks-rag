"""
update.py — Refresh all current season data. Run daily or after every game.
"""
import subprocess
import sys

scripts = [
    "ingest_current.py",
    "build_summaries.py",
    "add_roster_doc.py",
]

for script in scripts:
    print(f"\n{'='*40}")
    print(f"Running {script}...")
    print('='*40)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERROR: {script} failed. Stopping.")
        sys.exit(1)

print("\nAll done. Current season data is up to date.")
