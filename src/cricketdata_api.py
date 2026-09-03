"""
CricketData.org (CricAPI) API Client and Live Feature Extractor
"""

import os
import requests
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from src.elo import EloCalculator


def get_api_key():
    """Retrieve CricketData.org API key from environment."""
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    key = os.getenv("CRICKETDATA_API_KEY") or os.getenv("CRICAPI_KEY") or os.getenv("SPORTMONKS_API_TOKEN")
    return key.strip() if key else None


class CricketDataClient:
    """
    Client for interacting with CricketData.org (api.cricapi.com/v1)
    """

    BASE_URL = "https://api.cricapi.com/v1"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_api_key()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def fetch_current_matches(self) -> dict:
        """Fetch list of current live & recent matches from api.cricapi.com."""
        if not self.api_key:
            return {"status": "failure", "reason": "No API key configured"}

        url = f"{self.BASE_URL}/currentMatches"
        params = {"apikey": self.api_key, "offset": 0}

        try:
            res = requests.get(url, params=params, timeout=12)
            if res.status_code == 200:
                return res.json()
            else:
                return {"status": "failure", "reason": f"HTTP {res.status_code}: {res.text[:100]}"}
        except Exception as e:
            return {"status": "failure", "reason": str(e)}

    def fetch_matches_list(self) -> dict:
        """Fetch list of upcoming/all matches from api.cricapi.com."""
        if not self.api_key:
            return {"status": "failure", "reason": "No API key configured"}

        url = f"{self.BASE_URL}/matches"
        params = {"apikey": self.api_key, "offset": 0}

        try:
            res = requests.get(url, params=params, timeout=12)
            if res.status_code == 200:
                return res.json()
            else:
                return {"status": "failure", "reason": f"HTTP {res.status_code}: {res.text[:100]}"}
        except Exception as e:
            return {"status": "failure", "reason": str(e)}


    def fetch_match_info(self, match_id: str) -> dict:
        """Fetch detailed match info by match ID."""
        if not self.api_key:
            return {"status": "failure", "reason": "No API key configured"}

        url = f"{self.BASE_URL}/match_info"
        params = {"apikey": self.api_key, "id": match_id}

        try:
            res = requests.get(url, params=params, timeout=12)
            if res.status_code == 200:
                return res.json()
            else:
                return {"status": "failure", "reason": f"HTTP {res.status_code}"}
        except Exception as e:
            return {"status": "failure", "reason": str(e)}


def filter_upcoming_matches(matches: list) -> list:
    """Filter API match list for upcoming / scheduled matches that have not started yet."""
    upcoming = []
    seen_ids = set()
    for m in matches:
        m_id = m.get("id") or m.get("name")
        if m_id in seen_ids:
            continue
        started = m.get("matchStarted", False)
        ended = m.get("matchEnded", False)
        status = str(m.get("status", "")).lower()

        # If not started, or status indicates upcoming / scheduled
        if not started and not ended:
            upcoming.append(m)
            seen_ids.add(m_id)
        elif not started:
            upcoming.append(m)
            seen_ids.add(m_id)
        elif any(k in status for k in ["not started", "starts", "scheduled", "yet to start", "upcoming"]):
            upcoming.append(m)
            seen_ids.add(m_id)

    # Return all upcoming matches, or fallback to all non-ended matches, or all returned matches
    if upcoming:
        return upcoming
    non_ended = [m for m in matches if not m.get("matchEnded", False)]
    return non_ended if non_ended else matches


def filter_live_matches(matches: list) -> list:
    """Filter API match list for active live matches in progress."""
    live = []
    for m in matches:
        started = m.get("matchStarted", False)
        ended = m.get("matchEnded", False)
        score = m.get("score", [])
        status = str(m.get("status", "")).lower()

        if started and not ended:
            live.append(m)
        elif len(score) > 0 and not ended:
            live.append(m)
        elif "need" in status or "in progress" in status or "opt to" in status or "won the toss" in status:
            if not ended:
                live.append(m)

    return live if live else [m for m in matches if m.get("matchStarted", False)]


TEAM_ALIASES = {
    'nam': 'Namibia', 'sa': 'South Africa', 'rsa': 'South Africa',
    'ind': 'India', 'aus': 'Australia', 'eng': 'England', 'pak': 'Pakistan',
    'nz': 'New Zealand', 'wi': 'West Indies', 'sl': 'Sri Lanka', 'ban': 'Bangladesh',
    'afg': 'Afghanistan', 'zim': 'Zimbabwe', 'ire': 'Ireland', 'ned': 'Netherlands',
    'uae': 'United Arab Emirates', 'usa': 'United States of America', 'scot': 'Scotland',
    'nep': 'Nepal', 'can': 'Canada', 'om': 'Oman', 'oman': 'Oman'
}

def resolve_team_name(name: str) -> str:
    """Normalize raw team name or shortcode to standard historical team name."""
    if not name:
        return "Unknown"
    clean = str(name).strip()
    lower = clean.lower()

    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]

    stripped = re.sub(r"\b(t20|women|men|xi|u19|squad)\b", "", clean, flags=re.IGNORECASE).strip()
    stripped_lower = stripped.lower()

    if stripped_lower in TEAM_ALIASES:
        return TEAM_ALIASES[stripped_lower]

    matches_csv = os.path.join("data", "processed", "matches.csv")
    if os.path.exists(matches_csv):
        df_m = pd.read_csv(matches_csv)
        known = set(df_m['team1'].dropna().unique()) | set(df_m['team2'].dropna().unique())
        if clean in known:
            return clean
        if stripped in known:
            return stripped
        for k in known:
            if k.lower() == stripped_lower or k.lower() in stripped_lower or stripped_lower in k.lower():
                return k

    return stripped if stripped else clean




def parse_overs(overs_val) -> tuple[int, int]:
    """
    Parse overs input (e.g., 14.3 or '14.3' or 14) into (overs_completed, balls_in_current_over).
    Returns total_balls_completed.
    """
    try:
        val_str = str(overs_val).strip()
        if "." in val_str:
            parts = val_str.split(".")
            completed_overs = int(parts[0])
            extra_balls = int(parts[1])
        else:
            completed_overs = int(val_str)
            extra_balls = 0
        total_balls = completed_overs * 6 + min(extra_balls, 5)
        return total_balls
    except Exception:
        return 0


def load_historical_elo_calculator() -> EloCalculator:
    """Compute current Elo ratings for teams using historical matches dataset."""
    matches_csv = os.path.join("data", "processed", "matches.csv")
    elo_calc = EloCalculator(k_factor=32.0, initial_elo=1500.0)

    if os.path.exists(matches_csv):
        df_matches = pd.read_csv(matches_csv)
        df_matches = df_matches.sort_values(by=["date", "match_id"], ascending=[True, True])
        for _, row in df_matches.iterrows():
            team1 = str(row["team1"])
            team2 = str(row["team2"])
            winner = str(row["winner"])
            t1_won = 1.0 if winner == team1 else 0.0
            elo_calc.update_ratings(team1, team2, actual_a_won=t1_won)

    return elo_calc


def extract_live_match_state(match_data: dict, elo_calc: EloCalculator = None) -> dict:
    """
    Extract match state dictionary required by LivePredictor model from CricketData match JSON.
    """
    if elo_calc is None:
        elo_calc = load_historical_elo_calculator()

    teams = match_data.get("teams", [])
    if len(teams) >= 2:
        team1, team2 = resolve_team_name(teams[0]), resolve_team_name(teams[1])
    else:
        team1, team2 = "Team A", "Team B"

    score_list = match_data.get("score", [])
    match_status = match_data.get("status", "")

    # Default state variables
    batting_team = team2
    bowling_team = team1
    score = 100
    wickets = 3
    balls_completed = 60
    target = 160
    innings_number = 2

    if len(score_list) >= 2:
        # Inning 2 active or completed
        inn1 = score_list[0]
        inn2 = score_list[1]
        
        target = int(inn1.get("r", 150)) + 1
        score = int(inn2.get("r", 0))
        wickets = int(inn2.get("w", 0))
        overs_raw = inn2.get("o", 0)
        balls_completed = parse_overs(overs_raw)

        inset2 = str(inn2.get("inset", ""))
        for t in teams:
            if t.lower() in inset2.lower():
                batting_team = t
                bowling_team = [x for x in teams if x != t][0] if len(teams) >= 2 else team1
                break
        innings_number = 2

    elif len(score_list) == 1:
        # Inning 1 active
        inn1 = score_list[0]
        score = int(inn1.get("r", 0))
        wickets = int(inn1.get("w", 0))
        overs_raw = inn1.get("o", 0)
        balls_completed = parse_overs(overs_raw)

        inset1 = str(inn1.get("inset", ""))
        for t in teams:
            if t.lower() in inset1.lower():
                bowling_team = t
                batting_team = [x for x in teams if x != t][0] if len(teams) >= 2 else team2
                break
        
        # Estimate target for 2nd innings model (e.g. 1.5x current score or 150 min)
        projected = max(140, int(score * (120 / max(1, balls_completed)))) if balls_completed > 0 else 160
        target = projected
        innings_number = 1

    else:
        # Try parsing match_status text e.g. "India need 45 runs in 30 balls"
        need_match = re.search(r"(\d+)\s+runs?\s+in\s+(\d+)\s+balls?", match_status, re.IGNORECASE)
        if need_match:
            runs_req = int(need_match.group(1))
            balls_rem = int(need_match.group(2))
            balls_completed = max(0, 120 - balls_rem)
            target = 160
            score = target - runs_req

    # Calculate live features
    balls_completed = min(120, max(0, balls_completed))
    balls_remaining = 120 - balls_completed
    wickets = min(10, max(0, wickets))
    wickets_remaining = 10 - wickets
    runs_required = max(0, target - score)

    overs_dec = balls_completed / 6.0 if balls_completed > 0 else 0.001
    current_run_rate = float(score / overs_dec) if overs_dec > 0 else 0.0

    rem_overs_dec = balls_remaining / 6.0 if balls_remaining > 0 else 0.001
    required_run_rate = float(runs_required / rem_overs_dec) if rem_overs_dec > 0 else (0.0 if runs_required == 0 else 36.0)

    run_rate_difference = current_run_rate - required_run_rate

    # Match team names to Elo calculator
    batting_team_elo = elo_calc.get_rating(batting_team)
    bowling_team_elo = elo_calc.get_rating(bowling_team)
    elo_difference = batting_team_elo - bowling_team_elo

    feature_dict = {
        "batting_team_elo": float(batting_team_elo),
        "bowling_team_elo": float(bowling_team_elo),
        "elo_difference": float(elo_difference),
        "score": float(score),
        "wickets": float(wickets),
        "balls_remaining": float(balls_remaining),
        "target": float(target),
        "runs_required": float(runs_required),
        "wickets_remaining": float(wickets_remaining),
        "current_run_rate": float(round(current_run_rate, 2)),
        "required_run_rate": float(round(required_run_rate, 2)),
        "run_rate_difference": float(round(run_rate_difference, 2)),
        # Meta info
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "match_name": match_data.get("name", f"{team1} vs {team2}"),
        "status": match_status or "In Progress",
        "venue": match_data.get("venue", "Unknown Venue"),
        "match_type": match_data.get("matchType", "t20"),
        "overs_completed": float(balls_completed // 6),
        "balls_completed": float(balls_completed),
        "innings_number": innings_number,
    }

    return feature_dict
