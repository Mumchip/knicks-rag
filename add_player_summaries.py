"""
add_player_summaries.py — Compute 2025-26 season averages per player and store as summary docs.
Run after ingest_current.py. Re-run whenever you want fresh averages.
"""
import chromadb
import re
from sentence_transformers import SentenceTransformer

SEASON = "2025-26"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")

# Pull all player game log docs for current season
all_data = col.get(limit=10000)
player_games = {}

for id_, doc in zip(all_data["ids"], all_data["documents"]):
    if not id_.startswith("pglog_"):
        continue
    if SEASON not in doc:
        continue

    # Extract player name and stats from doc text
    name_match = re.match(r"On .+?, (.+?) played", doc)
    if not name_match:
        continue
    name = name_match.group(1)

    pts = re.search(r"scored (\d+) points", doc)
    reb = re.search(r"grabbed (\d+) rebounds", doc)
    ast = re.search(r"dished (\d+) assists", doc)
    stl = re.search(r"(\d+) steals", doc)
    blk = re.search(r"(\d+) blocks", doc)
    mins = re.search(r"played (\d+) minutes", doc)

    if not pts:
        continue

    if name not in player_games:
        player_games[name] = {"pts": [], "reb": [], "ast": [], "stl": [], "blk": [], "mins": []}

    player_games[name]["pts"].append(int(pts.group(1)))
    if reb: player_games[name]["reb"].append(int(reb.group(1)))
    if ast: player_games[name]["ast"].append(int(ast.group(1)))
    if stl: player_games[name]["stl"].append(int(stl.group(1)))
    if blk: player_games[name]["blk"].append(int(blk.group(1)))
    if mins: player_games[name]["mins"].append(int(mins.group(1)))

def avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0.0

docs = []
for name, stats in player_games.items():
    gp = len(stats["pts"])
    doc = (
        f"{name} 2025-26 season averages ({gp} games played): "
        f"{avg(stats['pts'])} PPG, {avg(stats['reb'])} RPG, {avg(stats['ast'])} APG, "
        f"{avg(stats['stl'])} SPG, {avg(stats['blk'])} BPG, "
        f"{avg(stats['mins'])} minutes per game."
    )
    docs.append({"id": f"summary_player_{name.replace(' ', '_').lower()}", "text": doc})
    print(f"  {name}: {avg(stats['pts'])} PPG over {gp} games")

if docs:
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    col.upsert(ids=[d["id"] for d in docs], documents=texts, embeddings=embeddings)
    print(f"\nStored {len(docs)} player summary docs.")
else:
    print("No player game logs found for current season.")
