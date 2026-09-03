import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv, find_dotenv
from src.cricketdata_api import CricketDataClient

def find_target_fixture():
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    client = CricketDataClient()
    if not client.is_configured():
        print("Error: CRICKETDATA_API_KEY missing from .env file.")
        return

    print("========================================")
    print("CRICKETDATA FIXTURE SEARCH")
    print("========================================")

    res = client.fetch_current_matches()
    status = res.get("status")
    print(f"\nAPI status: {status}")

    if status != "success":
        print("Fixtures returned: 0")
        print(f"API error: {res.get('reason', 'Unknown error')}")
        return

    matches = res.get("data", [])
    print(f"Fixtures returned: {len(matches)}")

    matching_fixture = None
    for f in matches:
        name = f.get("name", "").lower()
        teams = [t.lower() for t in f.get("teams", [])]
        
        is_target = any("zimbabwe" in t for t in teams) and any("south africa" in t for t in teams)
        if is_target or "zimbabwe" in name and "south africa" in name:
            matching_fixture = f
            break

    if matching_fixture:
        print("\nFixture found: YES\n")
        print(f"ID: {matching_fixture.get('id')}")
        print(f"Match Name: {matching_fixture.get('name')}")
        print(f"Status: {matching_fixture.get('status')}")
        print(f"Venue: {matching_fixture.get('venue')}")
        print(f"Date: {matching_fixture.get('date')}")
    else:
        print("\nTarget fixture (Zimbabwe vs South Africa) not active in current matches.")
        if len(matches) > 0:
            print("\nCurrent live/recent matches:")
            for i, m in enumerate(matches[:5], 1):
                print(f"  {i}. {m.get('name')} [{m.get('matchType')}] - {m.get('status')}")

if __name__ == "__main__":
    find_target_fixture()
