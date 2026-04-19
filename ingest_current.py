"""
ingest_current.py — Current season deep data: player game logs + box scores.
Run once to add to existing chroma_db: python ingest_current.py
"""

import os
import time
import json
import pandas as pd
import chromadb
from embed_utils import embedder
from dotenv import load_dotenv
from nba_api.stats.endpoints import (
    commonteamroster,
    playergamelog,
    boxscoretraditionalv3,
    teamgamelog,
)

load_dotenv()

KNICKS_ID = 1610612752
SEASON = "2025-26"
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)



def load_checkpoint(name):
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(name, data):
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
                print(f"    Retry {attempt + 1}: {e}")
                time.sleep(3)
            else:
                print(f"    Failed: {e}")
                return None


def fetch_player_game_logs() -> list[dict]:
    cached = load_checkpoint("player_game_logs_2025")
    if cached:
        print("  [cache] player game logs")
        return cached

    result = safe_fetch(commonteamroster.CommonTeamRoster, team_id=KNICKS_ID, season=SEASON)
    if result is None:
        return []
    roster = result.get_data_frames()[0]

    docs = []
    for i, (_, player) in enumerate(roster.iterrows()):
        name = player["PLAYER"]
        player_id = player["PLAYER_ID"]
        print(f"  [{i+1}/{len(roster)}] {name}...")

        result = safe_fetch(playergamelog.PlayerGameLog, player_id=player_id, season=SEASON)
        if result is None:
            continue

        df = result.get_data_frames()[0]
        for _, game in df.iterrows():
            outcome = "won" if game["WL"] == "W" else "lost"
            mins = int(float(game["MIN"])) if pd.notna(game.get("MIN")) else 0
            docs.append({
                "id": f"pglog_{player_id}_{game['Game_ID']}",
                "text": (
                    f"On {game['GAME_DATE']} ({SEASON}), {name} played {mins} minutes "
                    f"in a game the Knicks {outcome} vs {game['MATCHUP'].split()[-1]}. "
                    f"He scored {int(game['PTS'])} points, grabbed {int(game['REB'])} rebounds, "
                    f"dished {int(game['AST'])} assists, with {int(game['STL'])} steals "
                    f"and {int(game['BLK'])} blocks. +/- of {int(game['PLUS_MINUS'])}."
                )
            })

    save_checkpoint("player_game_logs_2025", docs)
    print(f"  Got {len(docs)} player game log entries.")
    return docs


def fetch_box_scores() -> list[dict]:
    cached = load_checkpoint("box_scores_2025")
    if cached:
        print("  [cache] box scores")
        return cached

    result = safe_fetch(teamgamelog.TeamGameLog, team_id=KNICKS_ID, season=SEASON)
    if result is None:
        return []

    games = result.get_data_frames()[0]
    docs = []

    for i, (_, game) in enumerate(games.iterrows()):
        game_id = game["Game_ID"]
        print(f"  [{i+1}/{len(games)}] Box score {game['GAME_DATE']} vs {game['MATCHUP'].split()[-1]}...")

        result = safe_fetch(boxscoretraditionalv3.BoxScoreTraditionalV3, game_id=game_id)
        if result is None:
            continue

        players_df = result.get_data_frames()[0]
        knicks_players = players_df[players_df["teamId"] == KNICKS_ID]

        lines = []
        for _, p in knicks_players.iterrows():
            if pd.isna(p.get("points")):
                continue
            mins = str(p.get("minutes", "0")).split(".")[0]
            name = f"{p['firstName']} {p['familyName']}"
            lines.append(
                f"{name}: {int(p['points'])}pts/{int(p['reboundsTotal'])}reb/"
                f"{int(p['assists'])}ast in {mins}min"
            )

        if lines:
            outcome = "won" if game["WL"] == "W" else "lost"
            docs.append({
                "id": f"boxscore_{game_id}",
                "text": (
                    f"Box score — {game['GAME_DATE']}, Knicks {outcome} vs "
                    f"{game['MATCHUP'].split()[-1]}: " + ", ".join(lines) + "."
                )
            })

    save_checkpoint("box_scores_2025", docs)
    print(f"  Got {len(docs)} box score entries.")
    return docs


def upsert_to_chroma(docs: list[dict]):
    print(f"\nEmbedding {len(docs)} documents...")
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("knicks")

    # Upsert in batches (add new, update existing)
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        collection.upsert(
            ids=[d["id"] for d in batch_docs],
            documents=[d["text"] for d in batch_docs],
            embeddings=embeddings[i:i + batch_size],
        )
        print(f"  Upserted batch {i // batch_size + 1}/{-(-len(docs) // batch_size)}")

    print(f"Done. {len(docs)} documents upserted into Chroma.")


if __name__ == "__main__":
    print("=== Current Season Deep Data ===\n")

    print("[1/2] Player game logs...")
    player_logs = fetch_player_game_logs()

    print("\n[2/2] Box scores...")
    box_scores = fetch_box_scores()

    all_docs = player_logs + box_scores
    print(f"\nTotal: {len(all_docs)} new documents")
    print(f"  - {len(player_logs)} player game log entries")
    print(f"  - {len(box_scores)} box score entries")

    upsert_to_chroma(all_docs)
    print("\nDone. Restart the server to use the updated knowledge base.")
