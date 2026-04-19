"""
build_summaries.py — Pre-compute all summary documents for reliable retrieval.
Run after any ingest. Re-run daily to keep current.
"""
import re
import chromadb
from embed_utils import embedder
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguedashplayerstats,
    commonteamroster,
    teamgamelog,
)

SEASON = "2025-26"
KNICKS_ID = 1610612752
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_collection("knicks")

docs = {}

# ── 1. Team season record ──────────────────────────────────────────────────
print("Building team season summary...")
try:
    team_df = leaguedashteamstats.LeagueDashTeamStats(season=SEASON).get_data_frames()[0]
    row = team_df[team_df["TEAM_ID"] == KNICKS_ID].iloc[0]
    w, l = int(row["W"]), int(row["L"])
    pct = f"{row['W_PCT']:.1%}"
    pts = f"{row['PTS']:.1f}"
    reb = f"{row['REB']:.1f}"
    ast = f"{row['AST']:.1f}"
    docs["summary_current_season"] = (
        f"New York Knicks 2025-26 season: {w} wins, {l} losses ({pct} win percentage). "
        f"Averaging {pts} points, {reb} rebounds, {ast} assists per game."
    )
    print(f"  Record: {w}-{l}")
except Exception as e:
    print(f"  Failed: {e}")

# ── 2. Player season averages ──────────────────────────────────────────────
print("Building player averages from NBA API...")
try:
    players_df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON, per_mode_simple="PerGame"
    ).get_data_frames()[0]

    # Get Knicks roster to filter
    roster = commonteamroster.CommonTeamRoster(team_id=KNICKS_ID, season=SEASON).get_data_frames()[0]
    knicks_ids = set(roster["PLAYER_ID"].tolist())
    knicks_players = players_df[players_df["PLAYER_ID"].isin(knicks_ids)]

    for _, p in knicks_players.iterrows():
        name = p["PLAYER_NAME"]
        gp = int(p["GP"])
        pts = round(float(p["PTS"]), 1)
        reb = round(float(p["REB"]), 1)
        ast = round(float(p["AST"]), 1)
        stl = round(float(p["STL"]), 1)
        blk = round(float(p["BLK"]), 1)
        fg_pct = f"{float(p['FG_PCT']):.1%}"
        fg3_pct = f"{float(p['FG3_PCT']):.1%}"
        mins = round(float(p["MIN"]), 1)
        key = f"summary_player_{name.replace(' ', '_').lower()}"
        docs[key] = (
            f"{name} 2025-26 season averages ({gp} games): "
            f"{pts} PPG, {reb} RPG, {ast} APG, {stl} SPG, {blk} BPG. "
            f"Shoots {fg_pct} from the field, {fg3_pct} from three. "
            f"{mins} minutes per game."
        )
        print(f"  {name}: {pts} PPG, {reb} RPG, {ast} APG")
except Exception as e:
    print(f"  Failed: {e}")

# ── 3. Statistical leaders ─────────────────────────────────────────────────
print("Building statistical leaders...")
try:
    players_df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON, per_mode_simple="PerGame"
    ).get_data_frames()[0]
    roster = commonteamroster.CommonTeamRoster(team_id=KNICKS_ID, season=SEASON).get_data_frames()[0]
    knicks_ids = set(roster["PLAYER_ID"].tolist())
    kp = players_df[players_df["PLAYER_ID"].isin(knicks_ids)].copy()

    def leaders(stat, label, n=5):
        top = kp.nlargest(n, stat)[["PLAYER_NAME", stat]]
        lines = [f"{i+1}. {row['PLAYER_NAME']} ({round(float(row[stat]), 1)} {label})"
                 for i, (_, row) in enumerate(top.iterrows())]
        return ", ".join(lines)

    docs["summary_scoring_leaders"] = (
        f"2025-26 Knicks scoring leaders (PPG): {leaders('PTS', 'PPG')}"
    )
    docs["summary_assist_leaders"] = (
        f"2025-26 Knicks assist leaders (APG): {leaders('AST', 'APG')}"
    )
    docs["summary_rebound_leaders"] = (
        f"2025-26 Knicks rebound leaders (RPG): {leaders('REB', 'RPG')}"
    )
    docs["summary_steal_leaders"] = (
        f"2025-26 Knicks steal leaders (SPG): {leaders('STL', 'SPG')}"
    )
    docs["summary_block_leaders"] = (
        f"2025-26 Knicks block leaders (BPG): {leaders('BLK', 'BPG')}"
    )
    print(f"  Scoring: {docs['summary_scoring_leaders'][:80]}...")
except Exception as e:
    print(f"  Failed: {e}")

# ── 4. Recent form (last 10 games) ─────────────────────────────────────────
print("Building recent form...")
try:
    log = teamgamelog.TeamGameLog(team_id=KNICKS_ID, season=SEASON).get_data_frames()[0]
    last10 = log.head(10)
    wins = (last10["WL"] == "W").sum()
    losses = (last10["WL"] == "L").sum()
    results = " ".join(["W" if r == "W" else "L" for r in last10["WL"]])
    avg_pts = round(last10["PTS"].mean(), 1)
    docs["summary_recent_form"] = (
        f"Knicks last 10 games (2025-26): {wins} wins, {losses} losses. "
        f"Results (most recent first): {results}. "
        f"Averaging {avg_pts} points per game in that stretch."
    )
    print(f"  Last 10: {wins}W-{losses}L")
except Exception as e:
    print(f"  Failed: {e}")

# ── 5. Upsert all ─────────────────────────────────────────────────────────
print(f"\nUpserting {len(docs)} summary documents...")
ids = list(docs.keys())
texts = list(docs.values())
embeddings = embedder.encode(texts, show_progress_bar=True)
col.upsert(ids=ids, documents=texts, embeddings=embeddings)
print(f"Done. {len(docs)} summary docs stored.")
print("\nSummary docs created:")
for k in docs:
    print(f"  {k}: {docs[k][:80]}...")
