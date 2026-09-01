import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger("cricsheet_preprocessor")


def preprocess_and_validate(matches_list: list, deliveries_list: list, parse_report: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Cleans, filters, validates, and prepares match and delivery DataFrames.
    """
    df_matches_raw = pd.DataFrame(matches_list)
    df_deliveries_raw = pd.DataFrame(deliveries_list)
    flags_list = parse_report["flags"]
    df_flags = pd.DataFrame(flags_list)

    # 1. Statistical Summaries from Raw Data
    total_json_found = parse_report["json_files_found"]
    successfully_parsed = parse_report["successfully_parsed"]
    failed_files = parse_report["failed_files"]

    num_no_result_abandoned = int(df_flags["is_no_result_abandoned"].sum())
    num_ties = int(df_flags["is_tie"].sum())
    num_super_overs = int(df_flags["is_super_over"].sum())
    num_dls = int(df_flags["is_dls"].sum())

    # 2. Filter Matches for Completed T20 Matches with a Clear Winner
    # Exclude abandoned / no-result / tie without winner matches
    valid_matches_mask = (
        df_matches_raw["winner"].notna() &
        (df_matches_raw["winner"] != "None") &
        (df_matches_raw["winner"] != "") &
        (~df_matches_raw["outcome_result"].isin(["no result", "abandoned", "cancelled"]))
    )
    df_matches = df_matches_raw[valid_matches_mask].copy()

    # 3. Validation: Winner must be one of the playing teams
    valid_winner_mask = (df_matches["winner"] == df_matches["team1"]) | (df_matches["winner"] == df_matches["team2"])
    df_matches = df_matches[valid_winner_mask].copy()

    # 4. Fill missing values safely in matches
    df_matches["city"] = df_matches["city"].fillna(df_matches["venue"]).replace("", "Unknown")
    df_matches["toss_winner"] = df_matches["toss_winner"].fillna("Unknown")
    df_matches["toss_decision"] = df_matches["toss_decision"].fillna("Unknown")

    # Parse and validate dates
    df_matches["date"] = pd.to_datetime(df_matches["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_matches = df_matches[df_matches["date"].notna()].copy()

    # Deduplicate matches by match_id
    df_matches = df_matches.drop_duplicates(subset=["match_id"]).copy()

    valid_match_ids = set(df_matches["match_id"].unique())

    # 5. Filter and clean Deliveries
    df_deliveries = df_deliveries_raw[df_deliveries_raw["match_id"].isin(valid_match_ids)].copy()

    # Fill delivery missing values safely
    numeric_cols = [
        "over", "delivery_number", "runs_off_bat", "extras", "total_runs",
        "wides", "noballs", "byes", "legbyes", "penalty", "is_wicket"
    ]
    for col in numeric_cols:
        if col in df_deliveries.columns:
            df_deliveries[col] = pd.to_numeric(df_deliveries[col], errors="coerce").fillna(0).astype(int)

    df_deliveries["wicket_type"] = df_deliveries["wicket_type"].fillna("None").replace("", "None")
    df_deliveries["player_dismissed"] = df_deliveries["player_dismissed"].fillna("None").replace("", "None")

    # Drop duplicate deliveries if any
    dedup_cols = ["match_id", "innings_number", "over", "delivery_number"]
    if "is_super_over" in df_deliveries.columns:
        dedup_cols.append("is_super_over")
    df_deliveries = df_deliveries.drop_duplicates(subset=dedup_cols).copy()

    # Select and order final output columns for matches.csv
    match_cols = [
        "match_id", "date", "team1", "team2", "venue", "city",
        "toss_winner", "toss_decision", "winner", "match_type", "outcome_result"
    ]
    df_matches_out = df_matches[match_cols].copy()

    # Select and order final output columns for deliveries.csv
    delivery_cols = [
        "match_id", "innings_number", "batting_team", "bowling_team",
        "over", "delivery_number", "batter", "bowler", "runs_off_bat",
        "extras", "total_runs", "wides", "noballs", "byes", "legbyes",
        "penalty", "is_wicket", "wicket_type", "player_dismissed"
    ]
    df_deliveries_out = df_deliveries[delivery_cols].copy()

    # 6. Comprehensive Validation Checks
    # Validation 1: Every delivery match_id exists in matches.csv
    deliv_match_ids = set(df_deliveries_out["match_id"].unique())
    orphaned_deliveries = deliv_match_ids - valid_match_ids
    assert len(orphaned_deliveries) == 0, f"Validation Failed: {len(orphaned_deliveries)} orphaned delivery match_ids found!"

    # Validation 2: Each match has exactly two playing teams
    two_teams_valid = (df_matches_out["team1"] != "Unknown") & (df_matches_out["team2"] != "Unknown")
    assert two_teams_valid.all(), "Validation Failed: Some matches do not have two valid teams!"

    # Validation 3: Winner is one of the teams
    winner_valid = (df_matches_out["winner"] == df_matches_out["team1"]) | (df_matches_out["winner"] == df_matches_out["team2"])
    assert winner_valid.all(), "Validation Failed: Winner is not one of team1 or team2!"

    # Validation 4: Dates parse correctly
    assert df_matches_out["date"].notna().all(), "Validation Failed: Missing or unparseable dates!"

    # Validation 5: No duplicate match IDs
    assert df_matches_out["match_id"].is_unique, "Validation Failed: Duplicate match IDs found in matches.csv!"

    # Validation 6: No duplicate delivery records
    assert not df_deliveries_out.duplicated(subset=["match_id", "innings_number", "over", "delivery_number"]).any(), \
        "Validation Failed: Duplicate delivery records found in deliveries.csv!"

    # 7. Collect Pipeline Statistics for Reporting
    all_teams = pd.concat([df_matches_out["team1"], df_matches_out["team2"]]).unique()
    num_unique_teams = len(all_teams)

    team_counts = pd.concat([df_matches_out["team1"], df_matches_out["team2"]]).value_counts().head(20)

    date_min = df_matches_out["date"].min()
    date_max = df_matches_out["date"].max()

    summary_metrics = {
        "json_files_found": total_json_found,
        "successfully_parsed": successfully_parsed,
        "failed_files_count": len(failed_files),
        "failed_files_list": failed_files,
        "num_completed_matches": len(df_matches_out),
        "num_deliveries": len(df_deliveries_out),
        "num_unique_teams": num_unique_teams,
        "date_range": f"{date_min} to {date_max}",
        "top_20_teams": team_counts.to_dict(),
        "num_no_result_abandoned": num_no_result_abandoned,
        "num_ties": num_ties,
        "num_super_overs": num_super_overs,
        "num_dls_matches": num_dls,
        "matches_missing_summary": df_matches_out.isnull().sum().to_dict(),
        "deliveries_missing_summary": df_deliveries_out.isnull().sum().to_dict()
    }

    return df_matches_out, df_deliveries_out, summary_metrics
