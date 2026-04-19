from db_path import CHROMA_PATH
"""
ingest.py — Pull Knicks data from NBA API, embed with local model, store in Chroma.
Run once to build the vector DB: python ingest.py
"""

import os
import time
import pandas as pd
import chromadb
from embed_utils import embedder
from dotenv import load_dotenv
from nba_api.stats.endpoints import (
    commonteamroster,
    playercareerstats,
    teamgamelog,
    leaguedashteamstats,
)

load_dotenv()

KNICKS_ID = 1610612752
SEASON = "2024-25"


def get_knicks_roster() -> pd.DataFrame:
    roster = commonteamroster.CommonTeamRoster(team_id=KNICKS_ID, season=SEASON)
    return roster.get_data_frames()[0]


def get_player_stats(player_id: int) -> pd.DataFrame:
    time.sleep(0.6)
    stats = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = stats.get_data_frames()[0]
    return df[df["SEASON_ID"] == SEASON]


def get_team_game_log() -> pd.DataFrame:
    log = teamgamelog.TeamGameLog(team_id=KNICKS_ID, season=SEASON)
    return log.get_data_frames()[0]


def get_team_stats() -> pd.DataFrame:
    stats = leaguedashteamstats.LeagueDashTeamStats(season=SEASON)
    df = stats.get_data_frames()[0]
    return df[df["TEAM_ID"] == KNICKS_ID]


def build_documents() -> list[dict]:
    docs = []

    print("Fetching roster...")
    roster = get_knicks_roster()

    print("Fetching team stats...")
    team_stats = get_team_stats()
    if not team_stats.empty:
        row = team_stats.iloc[0]
        docs.append({
            "id": "team_stats",
            "text": (
                f"The New York Knicks in the {SEASON} season: "
                f"{int(row['W'])} wins, {int(row['L'])} losses. "
                f"Averaging {row['PTS']:.1f} points, {row['REB']:.1f} rebounds, "
                f"{row['AST']:.1f} assists per game. "
                f"Field goal percentage: {row['FG_PCT']:.1%}. "
                f"Three-point percentage: {row['FG3_PCT']:.1%}."
            )
        })

    print("Fetching game log...")
    game_log = get_team_game_log()
    for _, game in game_log.head(20).iterrows():
        result = "won" if game["WL"] == "W" else "lost"
        docs.append({
            "id": f"game_{game['Game_ID']}",
            "text": (
                f"On {game['GAME_DATE']}, the Knicks {result} against {game['MATCHUP'].split()[-1]} "
                f"scoring {int(game['PTS'])} points. "
                f"They had {int(game['REB'])} rebounds, {int(game['AST'])} assists, "
                f"and {int(game['TOV'])} turnovers."
            )
        })

    print("Fetching player stats...")
    for _, player in roster.iterrows():
        name = player["PLAYER"]
        player_id = player["PLAYER_ID"]
        position = player.get("POSITION", "")
        print(f"  {name}...")
        stats = get_player_stats(player_id)
        if stats.empty:
            docs.append({
                "id": f"player_{player_id}",
                "text": f"{name} is on the {SEASON} New York Knicks roster as a {position}."
            })
        else:
            row = stats.iloc[0]
            gp = int(row["GP"]) if row["GP"] else 0
            docs.append({
                "id": f"player_{player_id}",
                "text": (
                    f"{name} ({position}) played {gp} games for the Knicks in {SEASON}. "
                    f"Averaged {row['PTS']:.1f} points, {row['REB']:.1f} rebounds, "
                    f"{row['AST']:.1f} assists per game. "
                    f"Shot {row['FG_PCT']:.1%} from the field and {row['FG3_PCT']:.1%} from three."
                )
            })

    return docs


def store_in_chroma(docs: list[dict]):
    print(f"\nEmbedding {len(docs)} documents...")
    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma.delete_collection("knicks")
    except Exception:
        pass

    collection = chroma.create_collection("knicks")
    collection.add(
        ids=[d["id"] for d in docs],
        documents=texts,
        embeddings=embeddings,
    )
    print(f"Stored {len(docs)} documents in Chroma.")


if __name__ == "__main__":
    print("Building Knicks knowledge base...")
    docs = build_documents()
    store_in_chroma(docs)
    print("Done. Run `uvicorn api:app --reload` to start the server.")
