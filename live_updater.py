from db_path import CHROMA_PATH
"""
live_updater.py — Polls for new Knicks games and updates Chroma automatically.
Run alongside the server: python live_updater.py

Checks every 30 minutes. When a new completed game is detected, fetches
box scores + player logs and upserts into Chroma — no server restart needed.
"""

import os
import time
import json
import pandas as pd
import chromadb
from datetime import datetime
from embed_utils import embedder
from dotenv import load_dotenv
from nba_api.stats.endpoints import (
    teamgamelog,
    boxscoretraditionalv3,
    playergamelog,
    commonteamroster,
)

load_dotenv()

KNICKS_ID = 1610612752
SEASON = "2025-26"
STATE_FILE = "data/live_state.json"
CHECK_INTERVAL = 30 * 60  # 30 minutes



def load_state() -> set:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return set(json.load(f))
    return set()


def save_state(seen_ids: set):
    with open(STATE_FILE, "w") as f:
        json.dump(list(seen_ids), f)


def safe_fetch(fn, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            time.sleep(0.7)
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"  Failed: {e}")
                return None


def get_new_game_ids(seen_ids: set) -> list[str]:
    result = safe_fetch(teamgamelog.TeamGameLog, team_id=KNICKS_ID, season=SEASON)
    if result is None:
        return []
    games = result.get_data_frames()[0]
    return [g["Game_ID"] for _, g in games.iterrows() if g["Game_ID"] not in seen_ids]


def build_docs_for_game(game_id: str, game_date: str, matchup: str, wl: str) -> list[dict]:
    docs = []
    opponent = matchup.split()[-1]
    outcome = "won" if wl == "W" else "lost"

    # Box score
    result = safe_fetch(boxscoretraditionalv3.BoxScoreTraditionalV3, game_id=game_id)
    if result:
        players_df = result.get_data_frames()[0]
        knicks = players_df[players_df["teamId"] == KNICKS_ID]
        lines = []
        for _, p in knicks.iterrows():
            if pd.isna(p.get("points")):
                continue
            mins = str(p.get("minutes", "0")).split(".")[0]
            name = f"{p['firstName']} {p['familyName']}"
            lines.append(
                f"{name}: {int(p['points'])}pts/{int(p['reboundsTotal'])}reb/"
                f"{int(p['assists'])}ast in {mins}min"
            )
        if lines:
            docs.append({
                "id": f"boxscore_{game_id}",
                "text": (
                    f"Box score — {game_date}, Knicks {outcome} vs {opponent}: "
                    + ", ".join(lines) + "."
                )
            })

    # Player game logs for this game
    roster_result = safe_fetch(commonteamroster.CommonTeamRoster, team_id=KNICKS_ID, season=SEASON)
    if roster_result:
        roster = roster_result.get_data_frames()[0]
        for _, player in roster.iterrows():
            name = player["PLAYER"]
            player_id = player["PLAYER_ID"]
            gl_result = safe_fetch(playergamelog.PlayerGameLog, player_id=player_id, season=SEASON)
            if gl_result is None:
                continue
            df = gl_result.get_data_frames()[0]
            match = df[df["Game_ID"] == game_id]
            if match.empty:
                continue
            game = match.iloc[0]
            mins = int(float(game["MIN"])) if pd.notna(game.get("MIN")) else 0
            docs.append({
                "id": f"pglog_{player_id}_{game_id}",
                "text": (
                    f"On {game_date} ({SEASON}), {name} played {mins} minutes "
                    f"in a game the Knicks {outcome} vs {opponent}. "
                    f"He scored {int(game['PTS'])} points, grabbed {int(game['REB'])} rebounds, "
                    f"dished {int(game['AST'])} assists, with {int(game['STL'])} steals "
                    f"and {int(game['BLK'])} blocks. +/- of {int(game['PLUS_MINUS'])}."
                )
            })

    return docs


def upsert(docs: list[dict]):
    if not docs:
        return
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, batch_size=64)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection("knicks")
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=texts,
        embeddings=embeddings,
    )


def run():
    print(f"Live updater started. Checking every {CHECK_INTERVAL // 60} minutes.")
    seen_ids = load_state()
    print(f"  Tracking {len(seen_ids)} known games.")

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Checking for new games...")

        try:
            new_ids = get_new_game_ids(seen_ids)
            if not new_ids:
                print("  No new games.")
            else:
                print(f"  Found {len(new_ids)} new game(s).")
                result = safe_fetch(teamgamelog.TeamGameLog, team_id=KNICKS_ID, season=SEASON)
                if result:
                    games_df = result.get_data_frames()[0]
                    for game_id in new_ids:
                        row = games_df[games_df["Game_ID"] == game_id]
                        if row.empty:
                            continue
                        row = row.iloc[0]
                        print(f"  Processing {row['GAME_DATE']} vs {row['MATCHUP'].split()[-1]}...")
                        docs = build_docs_for_game(
                            game_id, row["GAME_DATE"], row["MATCHUP"], row["WL"]
                        )
                        upsert(docs)
                        seen_ids.add(game_id)
                        save_state(seen_ids)  # save after each game so restarts resume
                        print(f"  Upserted {len(docs)} documents.")
        except Exception as e:
            print(f"  Error: {e}")

        print(f"  Next check in {CHECK_INTERVAL // 60} minutes.")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
