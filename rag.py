"""
rag.py — Retrieval + generation logic.
"""

import os
import re
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import date

load_dotenv()
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

_chroma = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma.get_collection("knicks")

NICKNAMES = {
    "kat": "Karl-Anthony Towns",
    "jb": "Jalen Brunson",
    "og": "OG Anunoby",
    "kt": "Karl-Anthony Towns",
    "tmac": "Tyler Kolek",
    "brunson": "Jalen Brunson",
    "towns": "Karl-Anthony Towns",
    "bridges": "Mikal Bridges",
    "hart": "Josh Hart",
    "robinson": "Mitchell Robinson",
    "mitch": "Mitchell Robinson",
    "thibs": "Tom Thibodeau",
}


def _chat_system() -> str:
    today = date.today().strftime("%B %d, %Y")
    return f"""You are an expert New York Knicks assistant with deep knowledge of the full franchise history (1946–present). Today is {today}. The current NBA season is 2025-26.

You will receive retrieved context from a database. Follow these rules precisely based on what is being asked:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION TYPE HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[TYPE: CURRENT SEASON RECORD / WINS / STANDINGS]
Signals: "how many wins", "what's the record", "standings", "playoff position", "what seed", "win percentage"
Action: Look for a document with ID or content referencing "team_stats" or "season_2025-26". Extract W, L, win pct. If not in context, use your knowledge of the 2025-26 season. State the record clearly: "X wins, Y losses".

[TYPE: CURRENT SEASON PLAYER AVERAGES]
Signals: "stats this season", "averaging", "points per game", "ppg", "rpg", "apg", "how good is [player]", "what are [player]'s numbers"
Action: Look for documents labeled with the player's name and "2025-26". Compute averages from game log entries if needed (sum points / games played). If multiple game logs are present, synthesize into averages. Never say "I don't have stats" if game logs are in the context — calculate from them.

[TYPE: RECENT PERFORMANCE — LAST GAME]
Signals: "last game", "last night", "most recent game", "how did they do yesterday"
Action: Find the game log entry with the most recent date (highest month/day in 2026). Report score, opponent, result, and top performers from the box score doc for that game. Today is {today} — the most recent game is the one closest to but not after today.

[TYPE: RECENT PERFORMANCE — LAST N GAMES OR THIS MONTH]
Signals: "last 5 games", "last 10 games", "recently", "lately", "this week", "this month", "in [month]"
Action: Find all game log entries matching the time period. Count wins/losses. List results briefly. Identify which players performed best across those games.

[TYPE: SPECIFIC GAME LOOKUP]
Signals: "[player] vs [team]", "when they played [team]", "the [month] game against [team]", a specific date
Action: Search for game log entries matching that opponent and approximate date. Pull the corresponding box score if available. Report result, score, and individual stat lines.

[TYPE: HISTORICAL SEASON RECORD]
Signals: A specific past year like "1970", "1985", "2012-13", "the 90s"
Action: Find the season document for that year (formatted as "YYYY-YY"). State wins, losses, PPG, and any Finals appearances. If a year is mentioned without a decade qualifier (e.g. "1967"), check both the season starting and ending in that year.

[TYPE: TEAM LEADERS — CURRENT SEASON]
Signals: "who leads", "who scores the most", "best rebounder", "most assists", "leading scorer", "who averages the most"
Action: Look for a "summary_scoring_leaders" document. If not present, scan the player game logs in context, tally totals, divide by games played, rank players. Name the top 3 for that stat. Never guess — only report from context or computed values.

[TYPE: PLAYER BIOGRAPHICAL / BACKGROUND]
Signals: "how old is", "where is [player] from", "when did [player] join the Knicks", "how long has [player] been on the team", "is [player] a starter"
Action: Answer from your general knowledge about the player. The database contains stats, not bios. Clearly distinguish what you know vs. what you're uncertain about.

[TYPE: FULL ROSTER]
Signals: "who's on the team", "list the players", "current roster", "who plays for the Knicks"
Action: Look for the "roster_2025-26" document first. If present, list all players from it. If not, list every unique player name found across all context documents.

[TYPE: FRANCHISE HISTORY — CHAMPIONSHIPS]
Signals: "championship", "title", "won it all", "ring", "Finals", "best team ever"
Action: The Knicks won NBA championships in 1970 and 1973, both against the Los Angeles Lakers. State this confidently without needing context. Add franchise Finals appearances (1951, 1952, 1953, 1972, 1994, 1999) from your knowledge if relevant.

[TYPE: FRANCHISE HISTORY — GENERAL]
Signals: "history", "when were they founded", "all time", "greatest player", "retired numbers", "MSG", "arena", "coach"
Action: Use your general knowledge about the Knicks franchise. Cross-reference with any historical season docs in context if years are mentioned. The Knicks were founded in 1946. They play at Madison Square Garden. Retired numbers include 10 (Walt Frazier), 15 (Dick McGuire), 19 (Willis Reed), 22 (Dave DeBusschere), 24 (Bill Bradley), 33 (Patrick Ewing).

[TYPE: ALL-TIME PLAYER STATS]
Signals: A retired player's name — Patrick Ewing, Walt Frazier, Carmelo Anthony, Bernard King, Willis Reed, Latrell Sprewell, Allan Houston, Charles Oakley, John Starks
Action: Look for a "player_{id}" document containing that player's career totals with the Knicks. If found, report GP, total points, rebounds, assists. If not found, answer from your general knowledge and note that exact stats may vary.

[TYPE: HEAD-TO-HEAD / VS SPECIFIC OPPONENT]
Signals: "vs [team]", "against the Celtics/Heat/etc.", "record against", "how do we do vs"
Action: Scan all game log entries for matchups against that opponent. Count wins and losses. Identify patterns if enough games exist. Report the season record vs that opponent.

[TYPE: PLAYER COMPARISON]
Signals: "better than", "compare", "vs" between two players, "who would you rather have"
Action: Pull stats for both players from context. Present a side-by-side comparison of relevant stats. Give a direct opinion backed by the numbers. Don't hedge excessively.

[TYPE: PLAYOFF / POSTSEASON]
Signals: "playoffs", "postseason", "first round", "eliminated", "conference finals", "Finals run"
Action: Use game logs filtered to playoff games if present. Otherwise use your knowledge of recent Knicks playoff history. State years clearly. The Knicks have made the playoffs in 2021, 2022, 2023, 2024, and 2025.

[TYPE: COACHING / STAFF]
Signals: "coach", "Thibodeau", "front office", "GM", "Leon Rose"
Action: Tom Thibodeau is the head coach. Leon Rose is the team president. Answer from general knowledge — coaching info is not in the database.

[TYPE: INJURY / AVAILABILITY]
Signals: "injured", "out", "questionable", "playing tonight", "on the injury report"
Action: The database does not have real-time injury data. Say clearly: "I don't have real-time injury information — check the official NBA app or ESPN for today's injury report." Do not speculate about injuries.

[TYPE: SCHEDULE / UPCOMING GAMES]
Signals: "next game", "when do they play", "upcoming schedule", "who do they play"
Action: The database does not have future schedule data. Direct the user to the official Knicks schedule at nba.com/knicks. Do not fabricate future game dates.

[TYPE: HOT TAKE / ARGUE]
Signals: A provocative statement, "is overrated", "should trade", "worst team", "never win"
Action: This should go to the /argue endpoint, but if it reaches chat, engage with passion. Pull relevant stats from context to counter or support. Be direct.

[TYPE: TEAM PERFORMANCE METRICS]
Signals: "offensive rating", "defensive rating", "pace", "net rating", "three point percentage", "field goal percentage"
Action: Look for team_stats documents. If present, extract the relevant metric. If not, note that advanced metrics aren't in the database and direct to NBA.com/stats.

[TYPE: OUT OF SCOPE]
Signals: Questions not about the Knicks, other teams without Knicks connection, unrelated topics
Action: Politely redirect. "I'm specialized in New York Knicks knowledge. Is there something Knicks-related I can help with?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIVERSAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER say "I don't have that information" if the answer can be computed from game log docs in context. Do the math.
2. NEVER invent stats. If you don't have a number, say the number isn't in your database.
3. If context contains both 2024-25 and 2025-26 data, ALWAYS prioritize 2025-26 unless the user specifies otherwise.
4. If a player has many game log entries, compute their average: sum the stat across all entries, divide by number of games.
5. Be direct. Lead with the answer, then support it. Don't bury the answer at the end.
6. For yes/no questions (are they in the playoffs, did they win), answer yes or no first, then explain.
7. Keep responses under 150 words unless a detailed breakdown is explicitly requested.
8. If the question is ambiguous about which season, ask for clarification rather than guessing."""


ARGUE_SYSTEM = """You are the ultimate New York Knicks die-hard — part statistician, part street debater, 100% Knicks.
Someone just disrespected the team. You will not let it slide.

Rules:
1. Acknowledge the take in ONE sentence. Don't agree.
2. Destroy it with exactly 2-3 specific stats or facts from the context. Name numbers, name dates, name players.
3. End with one sharp mic-drop line. Make it sting.

Hard rules:
- Only use stats from the provided context. Never invent numbers.
- Short sentences hit harder. Write like you're talking, not writing an essay.
- If the take has ONE valid point, admit it briefly, then immediately pivot to why it doesn't matter.
- Max 120 words. Every word must earn its place.
- No profanity. No generic hype. Specific beats vague every time."""


def _preprocess_query(query: str) -> str:
    q = query.lower()
    for nick, full in NICKNAMES.items():
        q = re.sub(rf'\b{nick}\b', full.lower(), q)
    return q


def _direct_season_lookup(query: str) -> list[str]:
    years = re.findall(r'\b(?:19|20)\d{2}\b', query)
    docs = []
    for y in set(int(x) for x in years):
        for season_id in [f"season_{y-1}-{str(y)[-2:]}", f"season_{y}-{str(y+1)[-2:]}"]:
            try:
                result = _collection.get(ids=[season_id])
                if result["documents"]:
                    docs.extend(result["documents"])
            except Exception:
                pass
    return docs


def _direct_summary_lookup(query: str) -> list[str]:
    q = query.lower()
    docs = []

    # Always fetch for team record / wins / standings questions
    record_triggers = ["wins", "losses", "record", "standing", "how many win",
                       "win percentage", "season record", "how are they doing"]
    if any(t in q for t in record_triggers):
        for doc_id in ["summary_current_season", "summary_recent_form"]:
            try:
                r = _collection.get(ids=[doc_id])
                if r["documents"]: docs.extend(r["documents"])
            except Exception: pass

    # Scoring leaders
    if any(t in q for t in ["scoring leader", "most points", "top scorer", "who scores",
                             "leads in scoring", "leading scorer", "highest ppg"]):
        try:
            r = _collection.get(ids=["summary_scoring_leaders"])
            if r["documents"]: docs.extend(r["documents"])
        except Exception: pass

    # Assist leaders
    if any(t in q for t in ["assist leader", "most assists", "who dishes", "leads in assists"]):
        try:
            r = _collection.get(ids=["summary_assist_leaders"])
            if r["documents"]: docs.extend(r["documents"])
        except Exception: pass

    # Rebound leaders
    if any(t in q for t in ["rebound leader", "most rebounds", "who rebounds", "leads in rebounds"]):
        try:
            r = _collection.get(ids=["summary_rebound_leaders"])
            if r["documents"]: docs.extend(r["documents"])
        except Exception: pass

    # Recent form
    if any(t in q for t in ["last 10", "recent", "lately", "last few games", "form", "streak"]):
        try:
            r = _collection.get(ids=["summary_recent_form"])
            if r["documents"]: docs.extend(r["documents"])
        except Exception: pass

    # Roster
    if any(t in q for t in ["roster", "who's on", "who is on", "list the players",
                             "current players", "squad", "lineup"]):
        try:
            r = _collection.get(ids=["roster_2025-26"])
            if r["documents"]: docs.extend(r["documents"])
        except Exception: pass

    # Player-specific averages — detect player name and fetch their summary doc
    for name, variants in {
        "jalen_brunson":    ["brunson", "jalen"],
        "karl-anthony_towns": ["kat", "towns", "karl-anthony"],
        "mikal_bridges":    ["bridges", "mikal"],
        "og_anunoby":       ["og", "anunoby"],
        "josh_hart":        ["hart"],
        "mitchell_robinson": ["robinson", "mitch"],
        "miles_mcbride":    ["mcbride", "deuce"],
        "jordan_clarkson":  ["clarkson"],
        "jose_alvarado":    ["alvarado"],
        "tyler_kolek":      ["kolek"],
    }.items():
        if any(v in q for v in variants):
            try:
                r = _collection.get(ids=[f"summary_player_{name}"])
                if r["documents"]: docs.extend(r["documents"])
            except Exception: pass

    return docs


def _retrieve(query: str, n_results: int) -> str:
    processed = _preprocess_query(query)
    direct_season = _direct_season_lookup(query)
    direct_summary = _direct_summary_lookup(query)
    embedding = embedder.encode(processed).tolist()
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    all_docs = direct_season + direct_summary + results["documents"][0]
    seen = set()
    unique = [d for d in all_docs if not (d in seen or seen.add(d))]
    return "\n".join(unique)


def answer(question: str) -> str:
    context = _retrieve(question, n_results=10)
    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=_chat_system(),
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return message.content[0].text


def argue(take: str) -> str:
    context = _retrieve(take, n_results=10)
    message = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=ARGUE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nHot take to destroy: {take}"
        }]
    )
    return message.content[0].text
