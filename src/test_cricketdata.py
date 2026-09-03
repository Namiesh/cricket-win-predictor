import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv, find_dotenv
from src.cricketdata_api import CricketDataClient, extract_live_match_state, load_historical_elo_calculator
from src.predict_live import LivePredictor

def test_cricketdata():
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    client = CricketDataClient()
    print("========================================")
    print("CRICKETDATA.ORG API TEST")
    print("========================================")
    print(f"API Key Configured: {'YES' if client.is_configured() else 'NO'}")

    if not client.is_configured():
        print("Error: CRICKETDATA_API_KEY missing from .env")
        return

    data = client.fetch_current_matches()
    status = data.get("status")
    print(f"API Status: {status}")

    if status == "success":
        matches = data.get("data", [])
        print(f"Total Matches Found: {len(matches)}")
        
        elo_calc = load_historical_elo_calculator()
        predictor = LivePredictor()

        for idx, m in enumerate(matches[:5], 1):
            name = m.get("name")
            m_status = m.get("status")
            m_type = m.get("matchType")
            print(f"\nMatch {idx}: {name} [{m_type}]")
            print(f"  Status: {m_status}")

            features = extract_live_match_state(m, elo_calc)
            print(f"  Batting Team: {features['batting_team']}")
            print(f"  Score: {features['score']}/{features['wickets']} (Target: {features['target']})")
            print(f"  Runs Req: {features['runs_required']} off {features['balls_remaining']}b")

            pred = predictor.predict(features)
            bat_p = pred["batting_team_win_probability"] * 100.0
            bowl_p = pred["bowling_team_win_probability"] * 100.0
            print(f"  Predictions -> {features['batting_team']}: {bat_p:.1f}%, {features['bowling_team']}: {bowl_p:.1f}%")
    else:
        reason = data.get("reason", "Unknown error")
        print(f"API Failed: {reason}")

if __name__ == "__main__":
    test_cricketdata()
