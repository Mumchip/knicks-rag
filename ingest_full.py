"""
ingest_full.py — Full Knicks history: all seasons, all players, recent game logs.
Run once (takes 15-25 min due to NBA API rate limits): python ingest_full.py

Checkpointing: saves raw data to data/ so you can resume if interrupted.
"""

import os
import time
import json
import pandas as pd
import chromadb
from embed_utils import embedder
from dotenv import load_dotenv
from nba_api.stats.endpoints import (
    franchiseplayers,
    teamgamelog,
    teamyearbyyearstats,
)

load_dotenv()

KNICKS_ID = 1610612752
FIRST_SEASON = 1946
CURRENT_SEASON_YEAR = 2024
GAME_LOG_SEASONS = 10  # seasons of game-by-game detail
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)



# ── Helpers ────────────────────────────────────────────────────────────────

def season_str(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def load_checkpoint(name: str):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(name: str, data):
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def safe_fetch(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            time.sleep(0.7)
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1} after error: {e}")
                time.sleep(3)
            else:
                print(f"    Failed after {retries} attempts: {e}")
                return None


# ── Data fetchers ──────────────────────────────────────────────────────────

def fetch_season_stats() -> list[dict]:
    cached = load_checkpoint("season_stats")
    if cached:
        print("  [cache] season stats")
        return cached

    print("  Fetching all seasons...")
    result = safe_fetch(teamyearbyyearstats.TeamYearByYearStats, team_id=KNICKS_ID)
    if result is None:
        return []

    df = result.get_data_frames()[0]
    docs = []
    for _, row in df.iterrows():
        season = row.get("YEAR", "")
        w = int(row["WINS"]) if pd.notna(row.get("WINS")) else "?"
        l = int(row["LOSSES"]) if pd.notna(row.get("LOSSES")) else "?"
        gp = int(row["GP"]) if pd.notna(row.get("GP")) and int(row["GP"]) > 0 else None
        pts_pg = f"{row['PTS'] / gp:.1f}" if gp and pd.notna(row.get("PTS")) else "?"
        playoffs = row.get("NBA_FINALS_APPEARANCE", "N/A")
        finals_note = " They won the NBA Championship." if playoffs == "LEAGUE CHAMPION" else (
            " They appeared in the NBA Finals." if playoffs == "FINALS APPEARANCE" else ""
        )
        docs.append({
            "id": f"season_{season}",
            "text": (
                f"In the {season} season, the New York Knicks finished with "
                f"{w} wins and {l} losses, averaging {pts_pg} points per game.{finals_note}"
            )
        })

    save_checkpoint("season_stats", docs)
    print(f"  Got {len(docs)} season summaries.")
    return docs


def fetch_game_logs() -> list[dict]:
    cached = load_checkpoint("game_logs")
    if cached:
        print("  [cache] game logs")
        return cached

    docs = []
    for year in range(CURRENT_SEASON_YEAR, CURRENT_SEASON_YEAR - GAME_LOG_SEASONS, -1):
        season = season_str(year)
        print(f"  Game log {season}...")
        result = safe_fetch(teamgamelog.TeamGameLog, team_id=KNICKS_ID, season=season)
        if result is None:
            continue
        df = result.get_data_frames()[0]
        for _, game in df.iterrows():
            outcome = "won" if game["WL"] == "W" else "lost"
            docs.append({
                "id": f"game_{season}_{game['Game_ID']}",
                "text": (
                    f"On {game['GAME_DATE']} ({season}), the Knicks {outcome} "
                    f"against {game['MATCHUP'].split()[-1]} scoring {int(game['PTS'])} points. "
                    f"{int(game['REB'])} rebounds, {int(game['AST'])} assists, "
                    f"{int(game['TOV'])} turnovers."
                )
            })

    save_checkpoint("game_logs", docs)
    print(f"  Got {len(docs)} game log entries.")
    return docs


def fetch_player_stats() -> list[dict]:
    cached = load_checkpoint("player_stats")
    if cached:
        print("  [cache] player stats")
        return cached

    print("  Fetching franchise players list...")
    result = safe_fetch(franchiseplayers.FranchisePlayers, team_id=KNICKS_ID)
    if result is None:
        return []

    players_df = result.get_data_frames()[0]
    print(f"  Found {len(players_df)} all-time Knicks players.")

    docs = []
    for _, player in players_df.iterrows():
        name = player["PLAYER"]
        player_id = player["PERSON_ID"]
        gp = int(player["GP"]) if pd.notna(player.get("GP")) else 0
        pts = int(player["PTS"]) if pd.notna(player.get("PTS")) else 0
        reb = int(player["REB"]) if pd.notna(player.get("REB")) else 0
        ast = int(player["AST"]) if pd.notna(player.get("AST")) else 0
        active = "current" if player.get("ACTIVE_WITH_TEAM") == 1 else "former"
        docs.append({
            "id": f"player_{player_id}",
            "text": (
                f"{name} is a {active} New York Knicks player. "
                f"Career totals with the Knicks: {gp} games played, "
                f"{pts} total points, {reb} total rebounds, {ast} total assists."
            )
        })

    save_checkpoint("player_stats", docs)
    print(f"  Got {len(docs)} player entries.")
    return docs


# ── Embed + store ──────────────────────────────────────────────────────────

def store_all(docs: list[dict]):
    print(f"\nEmbedding {len(docs)} documents...")
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    chroma = chromadb.PersistentClient(path="./chroma_db")
    try:
        chroma.delete_collection("knicks")
    except Exception:
        pass

    collection = chroma.create_collection("knicks")

    # Chroma has a batch limit — insert in chunks
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            embeddings=embeddings[i:i + batch_size],
        )
        print(f"  Stored batch {i // batch_size + 1}/{-(-len(docs) // batch_size)}")

    print(f"Done. {len(docs)} documents in Chroma.")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Full Knicks Knowledge Base ===\n")

    print("[1/3] Season stats...")
    season_docs = fetch_season_stats()

    print("\n[2/3] Game logs (last 10 seasons)...")
    game_docs = fetch_game_logs()

    print("\n[3/3] All-time player stats...")
    player_docs = fetch_player_stats()

    all_docs = season_docs + game_docs + player_docs
    print(f"\nTotal: {len(all_docs)} documents")
    print(f"  - {len(season_docs)} season summaries")
    print(f"  - {len(game_docs)} game logs")
    print(f"  - {len(player_docs)} player entries")

    store_all(all_docs)
    print("\nDone. Run `uvicorn api:app --reload` to start the server.")
