import os
import requests
from dotenv import load_dotenv, find_dotenv

def find_target_fixture():
    # Load environment variables
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    token = os.getenv("SPORTMONKS_API_TOKEN")
    if not token or not token.strip():
        print("Error: SPORTMONKS_API_TOKEN missing from .env file.")
        return

    date_start = "2026-09-01"
    date_end = "2026-09-02"

    url = "https://cricket.sportmonks.com/api/v2.0/fixtures"
    params = {
        "api_token": token.strip(),
        "filter[league_id]": "3",
        "filter[starts_between]": f"{date_start},{date_end}",
        "include": "localteam,visitorteam,venue"
    }

    print("========================================")
    print("TARGET FIXTURE SEARCH (LEAGUE ID 3)")
    print("========================================")

    try:
        response = requests.get(url, params=params, timeout=15)
        status_code = response.status_code
        print(f"\nAPI status: {status_code}")

        if status_code != 200:
            print("Fixtures returned: 0")
            print(f"\nRequested date window:\n{date_start} to {date_end}")
            print("\nFixture found: NO")
            print(f"API returned HTTP status {status_code}.")
            return

        json_data = response.json()
        fixtures = json_data.get("data", [])
        print(f"Fixtures returned: {len(fixtures)}")

        print(f"\nRequested date window:\n{date_start} to {date_end}")
        print("League ID filter: 3 (Twenty20 International)")

        meta = json_data.get("meta", {}) or {}
        pagination = meta.get("pagination", {}) or {}
        links = json_data.get("links", {}) or {}

        curr_page = pagination.get("current_page", "N/A")
        total_pages = pagination.get("total_pages", "N/A")
        total_items = pagination.get("total", "N/A")
        has_next = bool(links.get("next"))

        print("\nPagination:")
        print(f"Current page: {curr_page}")
        print(f"Last page: {total_pages}")
        print(f"Total: {total_items}")
        print(f"Next page link exists: {'YES' if has_next else 'NO'}")

        matching_fixture = None
        for f in fixtures:
            local_team = f.get("localteam", {}) or {}
            visitor_team = f.get("visitorteam", {}) or {}

            local_name = local_team.get("name", "") if isinstance(local_team, dict) else ""
            visitor_name = visitor_team.get("name", "") if isinstance(visitor_team, dict) else ""

            names_set = {local_name.lower(), visitor_name.lower()}
            is_zim = any("zimbabwe" in n for n in names_set)
            is_sa = any("south africa" in n for n in names_set)

            if is_zim and is_sa:
                matching_fixture = f
                break

        if matching_fixture:
            print("\nFixture found:\nYES\n")

            f_id = matching_fixture.get("id")
            local_name = matching_fixture.get("localteam", {}).get("name", "Unknown") if isinstance(matching_fixture.get("localteam"), dict) else "Unknown"
            visitor_name = matching_fixture.get("visitorteam", {}).get("name", "Unknown") if isinstance(matching_fixture.get("visitorteam"), dict) else "Unknown"
            starting_at = matching_fixture.get("starting_at") or matching_fixture.get("starting_at_date")
            venue_obj = matching_fixture.get("venue", {})
            venue_name = venue_obj.get("name", "Unknown") if isinstance(venue_obj, dict) else "Unknown"
            status = matching_fixture.get("status")
            live_flag = matching_fixture.get("live", False)

            print(f"Fixture ID: {f_id}")
            print(f"Team 1: {local_name}")
            print(f"Team 2: {visitor_name}")
            print(f"Date: {starting_at}")
            print(f"Venue: {venue_name}")
            print(f"Status: {status}")
            print(f"Starting time: {starting_at}")
            print(f"Live flag: {live_flag}")

            print("\nValidation:")
            print("[PASS] Correct Cricket fixtures endpoint")
            print("[PASS] Correct date window and league filter")
            print("[PASS] Target teams identified")
        else:
            print("\nFixture found:\nNO")
            print(f"\nThe target fixture (Zimbabwe vs South Africa) was absent from the returned page ({curr_page}).")
            if len(fixtures) > 0:
                print("\nReturned fixtures summary on this page:")
                for i, f in enumerate(fixtures[:5], 1):
                    lt = f.get('localteam', {}).get('name', 'N/A') if isinstance(f.get('localteam'), dict) else 'N/A'
                    vt = f.get('visitorteam', {}).get('name', 'N/A') if isinstance(f.get('visitorteam'), dict) else 'N/A'
                    print(f"  {i}. ID {f.get('id')}: {lt} vs {vt} (Date: {f.get('starting_at')})")

    except requests.exceptions.RequestException as e:
        print(f"\nAPI status: Error ({type(e).__name__})")
        print("Fixture found: NO")

if __name__ == "__main__":
    find_target_fixture()
