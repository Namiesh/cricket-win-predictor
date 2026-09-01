import os
import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Configure logger
logger = logging.getLogger("cricsheet_loader")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_json_filepaths(raw_dir: str) -> List[str]:
    """
    Recursively find all .json files under raw_dir.
    """
    pattern = os.path.join(raw_dir, "**", "*.json")
    json_files = glob.glob(pattern, recursive=True)
    json_files.sort()
    return json_files


def parse_single_json(filepath: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a single Cricsheet JSON file into:
    1. Match-level dictionary
    2. List of delivery-level dictionaries
    3. Metadata flags (DLS, Super Over, Tie, Abandoned status)
    """
    match_id = Path(filepath).stem

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    info = data.get("info", {})
    innings_list = data.get("innings", [])

    # Extract Teams
    teams = info.get("teams", [])
    team1 = teams[0] if len(teams) > 0 else "Unknown"
    team2 = teams[1] if len(teams) > 1 else "Unknown"

    # Extract Dates
    dates = info.get("dates", [])
    date_str = dates[0] if len(dates) > 0 else None

    # Extract Venue & City
    venue = info.get("venue", "Unknown")
    city = info.get("city", venue if venue != "Unknown" else "Unknown")

    # Extract Toss
    toss = info.get("toss", {})
    toss_winner = toss.get("winner", "Unknown")
    toss_decision = toss.get("decision", "Unknown")

    # Extract Outcome
    outcome = info.get("outcome", {})
    winner = outcome.get("winner", None)
    outcome_result = outcome.get("result", "completed" if winner else "no result")

    # Detection Flags
    method = outcome.get("method", "")
    is_dls = True if method in ["D/L", "DLS"] or "DLS" in str(method) else False
    is_tie = True if outcome_result == "tie" else False
    
    # Check for Super Over in outcome or innings
    is_super_over_flag = False
    if "eliminator" in outcome or outcome_result == "tie" or "bowl out" in str(method).lower():
        is_super_over_flag = True

    # Check innings for super over
    for inn in innings_list:
        if inn.get("super_over", False):
            is_super_over_flag = True
            break

    match_type = info.get("match_type", "Unknown")

    match_record = {
        "match_id": match_id,
        "date": date_str,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "city": city,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": winner,
        "match_type": match_type,
        "outcome_result": outcome_result
    }

    match_flags = {
        "match_id": match_id,
        "is_dls": is_dls,
        "is_tie": is_tie,
        "is_super_over": is_super_over_flag,
        "is_completed_with_winner": True if (winner is not None and outcome_result in ["completed", None]) else False,
        "is_no_result_abandoned": True if (outcome_result in ["no result", "abandoned", "cancelled"] or winner is None) else False,
        "num_teams": len(teams)
    }

    deliveries_records = []

    for inn_idx, inn in enumerate(innings_list, start=1):
        batting_team = inn.get("team", "Unknown")
        # Determine bowling team
        if batting_team == team1:
            bowling_team = team2
        elif batting_team == team2:
            bowling_team = team1
        else:
            bowling_team = "Unknown"

        is_super_over_inn = inn.get("super_over", False)

        for over_obj in inn.get("overs", []):
            over_num = over_obj.get("over", 0)  # Preserve integer over number
            deliveries = over_obj.get("deliveries", [])

            for deliv_idx, deliv in enumerate(deliveries, start=1):
                batter = deliv.get("batter", "Unknown")
                bowler = deliv.get("bowler", "Unknown")

                runs = deliv.get("runs", {})
                runs_off_bat = runs.get("batter", 0)
                extras = runs.get("extras", 0)
                total_runs = runs.get("total", 0)

                extras_dict = deliv.get("extras", {})
                wides = extras_dict.get("wides", 0)
                noballs = extras_dict.get("noballs", 0)
                byes = extras_dict.get("byes", 0)
                legbyes = extras_dict.get("legbyes", 0)
                penalty = extras_dict.get("penalty", 0)

                wickets_list = deliv.get("wickets", [])
                if wickets_list:
                    is_wicket = 1
                    wicket_type = wickets_list[0].get("kind", "unknown")
                    player_dismissed = wickets_list[0].get("player_out", "Unknown")
                else:
                    is_wicket = 0
                    wicket_type = "None"
                    player_dismissed = "None"

                deliv_record = {
                    "match_id": match_id,
                    "innings_number": inn_idx,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": int(over_num),
                    "delivery_number": int(deliv_idx),
                    "batter": batter,
                    "bowler": bowler,
                    "runs_off_bat": int(runs_off_bat),
                    "extras": int(extras),
                    "total_runs": int(total_runs),
                    "wides": int(wides),
                    "noballs": int(noballs),
                    "byes": int(byes),
                    "legbyes": int(legbyes),
                    "penalty": int(penalty),
                    "is_wicket": int(is_wicket),
                    "wicket_type": wicket_type,
                    "player_dismissed": player_dismissed,
                    "is_super_over": is_super_over_inn
                }

                deliveries_records.append(deliv_record)

    return match_record, deliveries_records, match_flags


def load_all_json_data(raw_dir: str):
    """
    Loads all Cricsheet JSON files from raw_dir and returns:
    - matches_list: List of raw match records
    - deliveries_list: List of raw delivery records
    - parse_report: Summary dictionary of ingestion results
    """
    json_paths = get_json_filepaths(raw_dir)
    
    matches_list = []
    deliveries_list = []
    flags_list = []
    failed_files = []

    for filepath in json_paths:
        try:
            m_rec, d_recs, m_flags = parse_single_json(filepath)
            matches_list.append(m_rec)
            deliveries_list.extend(d_recs)
            flags_list.append(m_flags)
        except Exception as e:
            logger.warning(f"Error parsing file {filepath}: {str(e)}")
            failed_files.append((filepath, str(e)))

    parse_report = {
        "json_files_found": len(json_paths),
        "successfully_parsed": len(matches_list),
        "failed_files": failed_files,
        "flags": flags_list
    }

    return matches_list, deliveries_list, parse_report
