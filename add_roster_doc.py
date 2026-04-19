from db_path import CHROMA_PATH
"""
add_roster_doc.py — Adds a single roster summary document to Chroma.
Re-run whenever the roster changes.
"""
import chromadb
from embed_utils import embedder
from nba_api.stats.endpoints import commonteamroster

KNICKS_ID = 1610612752
SEASON = "2025-26"


roster = commonteamroster.CommonTeamRoster(team_id=KNICKS_ID, season=SEASON).get_data_frames()[0]
players = roster["PLAYER"].tolist()

doc = f"The 2025-26 New York Knicks roster includes the following players: {', '.join(players)}."
print(doc)

embedding = embedder.encode(doc)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
col = chroma.get_collection("knicks")
col.upsert(ids=["roster_2025-26"], documents=[doc], embeddings=[embedding])
print("Done.")
