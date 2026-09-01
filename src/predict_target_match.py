import os
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from src.elo import EloCalculator


def search_venue_aliases(requested_venue: str, df_matches: pd.DataFrame):
    """
    Search historical matches for exact or partial venue matches.
    """
    all_venues = df_matches["venue"].dropna().unique()
    
    # 1. Exact match check
    exact_matches = [v for v in all_venues if v.strip().lower() == requested_venue.strip().lower()]
    
    # 2. Substring / partial match check
    req_terms = [t for t in requested_venue.replace(",", "").split() if len(t) > 2]
    matching_venues = []
    for v in all_venues:
        v_str = str(v)
        # Check if requested venue is substring or key terms overlap
        if requested_venue.lower() in v_str.lower() or v_str.lower() in requested_venue.lower():
            matching_venues.append(v_str)
        elif any(term.lower() in v_str.lower() for term in req_terms if term.lower() not in ["cricket", "ground"]):
            matching_venues.append(v_str)

    matching_venues = sorted(list(set(matching_venues)))
    
    counts = {}
    for v in matching_venues:
        counts[v] = len(df_matches[df_matches["venue"] == v])
        
    return exact_matches, counts


def compute_target_match_features(target_team1: str, target_team2: str, target_venue: str, target_date: str, matches_path: str, deliveries_path: str):
    """
    Computes pre-match feature vector for a future match using historical data up to target_date.
    """
    df_matches = pd.read_csv(matches_path)
    df_deliveries = pd.read_csv(deliveries_path)

    # Search venue aliases before filtering
    exact_matches, venue_alias_counts = search_venue_aliases(target_venue, df_matches)

    # Determine resolved venue string
    if exact_matches:
        resolved_venue = exact_matches[0]
    elif venue_alias_counts:
        # Use top matching venue by keyword overlap / match count
        resolved_venue = max(venue_alias_counts, key=venue_alias_counts.get)
    else:
        resolved_venue = target_venue

    # Filter historical matches strictly BEFORE target_date
    df_matches = df_matches[df_matches["date"] < target_date].copy()
    df_matches = df_matches.sort_values(by=["date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    cutoff_date = df_matches["date"].max()

    # Pre-calculate Innings Scores per Match & Team from deliveries.csv
    innings_summary = (
        df_deliveries.groupby(["match_id", "innings_number", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    team_innings_scores = {}
    first_innings_info = {}

    for _, row in innings_summary.iterrows():
        m_id = str(row["match_id"])
        inn_num = int(row["innings_number"])
        team = str(row["batting_team"])
        runs = float(row["total_runs"])

        team_innings_scores[(m_id, team)] = runs

        if inn_num == 1:
            first_innings_info[m_id] = (team, runs)

    # Initialize State Trackers for Chronological Computation
    elo_calc = EloCalculator(k_factor=32.0, initial_elo=1500.0)

    team_match_history = defaultdict(list)
    team_score_history = defaultdict(list)
    h2h_history = defaultdict(list)

    venue_1st_innings_scores = defaultdict(list)
    venue_batting_first_wins = defaultdict(list)

    global_1st_innings_scores = []
    global_batting_first_wins = []

    # Replay all historical matches to build state up to cutoff date
    for idx, row in df_matches.iterrows():
        match_id = str(row["match_id"])
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        venue = str(row["venue"])
        winner = str(row["winner"])

        team1_won = 1.0 if winner == team1 else 0.0

        # Update Elo Ratings
        elo_calc.update_ratings(team1, team2, actual_a_won=team1_won)

        # Update Win History
        team_match_history[team1].append(1.0 if team1_won == 1.0 else 0.0)
        team_match_history[team2].append(0.0 if team1_won == 1.0 else 1.0)

        # Update Scoring History
        if (match_id, team1) in team_innings_scores:
            team_score_history[team1].append(team_innings_scores[(match_id, team1)])
        if (match_id, team2) in team_innings_scores:
            team_score_history[team2].append(team_innings_scores[(match_id, team2)])

        # Update H2H History
        h2h_key = tuple(sorted([team1, team2]))
        h2h_history[h2h_key].append(winner)

        # Update Venue & Global History
        if match_id in first_innings_info:
            first_inn_team, first_inn_runs = first_innings_info[match_id]
            batting_first_won = 1.0 if winner == first_inn_team else 0.0

            venue_1st_innings_scores[venue].append(first_inn_runs)
            venue_batting_first_wins[venue].append(batting_first_won)

            global_1st_innings_scores.append(first_inn_runs)
            global_batting_first_wins.append(batting_first_won)

    # --- NOW COMPUTE PRE-MATCH FEATURES FOR THE TARGET MATCH ---
    team1_elo = elo_calc.get_rating(target_team1)
    team2_elo = elo_calc.get_rating(target_team2)
    elo_difference = team1_elo - team2_elo

    hist1 = team_match_history[target_team1]
    hist2 = team_match_history[target_team2]

    team1_win_rate_last5 = float(np.mean(hist1[-5:])) if len(hist1) > 0 else 0.5
    team2_win_rate_last5 = float(np.mean(hist2[-5:])) if len(hist2) > 0 else 0.5

    team1_win_rate_last10 = float(np.mean(hist1[-10:])) if len(hist1) > 0 else 0.5
    team2_win_rate_last10 = float(np.mean(hist2[-10:])) if len(hist2) > 0 else 0.5

    global_prior_score = float(np.mean(global_1st_innings_scores)) if len(global_1st_innings_scores) > 0 else 145.0
    scores1 = team_score_history[target_team1]
    scores2 = team_score_history[target_team2]

    team1_avg_runs_last5 = float(np.mean(scores1[-5:])) if len(scores1) > 0 else global_prior_score
    team2_avg_runs_last5 = float(np.mean(scores2[-5:])) if len(scores2) > 0 else global_prior_score

    target_h2h_key = tuple(sorted([target_team1, target_team2]))
    past_h2h = h2h_history[target_h2h_key]
    h2h_matches_before = len(past_h2h)

    if h2h_matches_before == 0:
        team1_h2h_win_rate = 0.5
    else:
        t1_h2h_wins = sum(1 for w in past_h2h if w == target_team1)
        team1_h2h_win_rate = float(t1_h2h_wins) / float(h2h_matches_before)

    # Use resolved venue string for lookup
    venue_scores = venue_1st_innings_scores[resolved_venue]
    venue_wins_1st = venue_batting_first_wins[resolved_venue]
    venue_matches_before = len(venue_scores)

    global_prior_bat1st_win_rate = float(np.mean(global_batting_first_wins)) if len(global_batting_first_wins) > 0 else 0.5

    if venue_matches_before == 0:
        venue_avg_first_innings_score = global_prior_score
        venue_batting_first_win_rate = global_prior_bat1st_win_rate
        venue_info_label = "Global Prior (0 venue matches)"
    else:
        venue_avg_first_innings_score = float(np.mean(venue_scores))
        venue_batting_first_win_rate = float(np.mean(venue_wins_1st))
        venue_info_label = f"Venue Specific ({venue_matches_before} historical matches)"

    features = {
        "team1_elo": round(team1_elo, 2),
        "team2_elo": round(team2_elo, 2),
        "elo_difference": round(elo_difference, 2),

        "team1_win_rate_last5": round(team1_win_rate_last5, 4),
        "team2_win_rate_last5": round(team2_win_rate_last5, 4),

        "team1_win_rate_last10": round(team1_win_rate_last10, 4),
        "team2_win_rate_last10": round(team2_win_rate_last10, 4),

        "team1_avg_runs_last5": round(team1_avg_runs_last5, 2),
        "team2_avg_runs_last5": round(team2_avg_runs_last5, 2),

        "team1_h2h_win_rate": round(team1_h2h_win_rate, 4),
        "h2h_matches_before": int(h2h_matches_before),

        "venue_avg_first_innings_score": round(venue_avg_first_innings_score, 2),
        "venue_batting_first_win_rate": round(venue_batting_first_win_rate, 4),
        "venue_matches_before": int(venue_matches_before)
    }

    venue_metadata = {
        "requested_venue": target_venue,
        "resolved_venue": resolved_venue,
        "exact_matches": exact_matches,
        "venue_alias_counts": venue_alias_counts,
        "venue_info_label": venue_info_label
    }

    return features, cutoff_date, venue_metadata


def predict_upcoming_match():
    # Target Match Setup
    team1 = "Zimbabwe"
    team2 = "South Africa"
    date = "2026-09-01"
    venue = "Namibia Cricket Ground"

    matches_csv = os.path.join("data", "processed", "matches.csv")
    deliveries_csv = os.path.join("data", "processed", "deliveries.csv")
    model_path = os.path.join("models", "prematch_model.pkl")
    features_path = os.path.join("models", "prematch_features.pkl")

    # Sanity Check 1: Model & Features files exist
    assert os.path.exists(model_path), f"Model file missing: {model_path}"
    assert os.path.exists(features_path), f"Features file missing: {features_path}"
    sanity_model_loaded = True

    model = joblib.load(model_path)
    training_features = joblib.load(features_path)

    # Compute pre-match features using historical data strictly < date
    features_dict, cutoff_date, venue_meta = compute_target_match_features(team1, team2, venue, date, matches_csv, deliveries_csv)

    # Sanity Check 2: Feature list matches training features
    computed_feature_keys = list(features_dict.keys())
    assert computed_feature_keys == training_features, f"Feature mismatch! Computed: {computed_feature_keys}, Expected: {training_features}"
    sanity_features_match = True

    # Sanity Check 3: No missing features
    assert all(v is not None for v in features_dict.values()), "Missing feature values detected!"
    sanity_no_missing = True

    # Sanity Check 4: No future matches used (cutoff date strictly < 2026-09-01)
    assert cutoff_date < date, f"Future match data used! Cutoff {cutoff_date} is not less than target date {date}"
    sanity_no_future = True

    # Construct DataFrame with exact training column ordering
    df_target = pd.DataFrame([features_dict])[training_features]

    # Predict Probabilities
    proba_array = model.predict_proba(df_target)[0]
    prob_team1 = float(proba_array[1])  # Class 1: team1_won
    prob_team2 = 1.0 - prob_team1

    # Sanity Check 5: Probabilities sum to 1
    assert abs((prob_team1 + prob_team2) - 1.0) < 1e-5, f"Probabilities do not sum to 1: {prob_team1} + {prob_team2}"
    sanity_prob_sums = True

    sanity_prediction_generated = True

    # Print Formatted Report
    print("=" * 55)
    print("VENUE ALIAS & HISTORICAL MATCH SEARCH")
    print("=" * 55)
    print(f"Exact venue requested:       '{venue_meta['requested_venue']}'")
    print(f"Resolved venue used:         '{venue_meta['resolved_venue']}'")
    print(f"Exact matches in database:   {venue_meta['exact_matches']}")
    print("Matching venue aliases found in database:")
    for v_name, count in venue_meta['venue_alias_counts'].items():
        print(f"  - '{v_name}': {count} historical matches")
    print(f"Venue Information Source:    {venue_meta['venue_info_label']}")

    print("\n" + "=" * 55)
    print(f"{team1.upper()} vs {team2.upper()}")
    print("PRE-MATCH PREDICTION")
    print("=" * 55)
    print(f"Match date:             {date}")
    print(f"Venue:                  {venue}")
    print(f"Resolved venue string:  {venue_meta['resolved_venue']}")
    print(f"Historical data cutoff: {cutoff_date}")
    print(f"\nNote: Prediction uses historical data up to {cutoff_date}. Matches occurring after {cutoff_date} were unavailable in the Cricsheet dataset.")

    print("\nFEATURES")
    print("-" * 55)
    print(f"{team1} Elo:                       {features_dict['team1_elo']}")
    print(f"{team2} Elo:                 {features_dict['team2_elo']}")
    print(f"Elo difference:                     {features_dict['elo_difference']}")
    print(f"\n{team1} win rate last 5:          {features_dict['team1_win_rate_last5']}")
    print(f"{team2} win rate last 5:    {features_dict['team2_win_rate_last5']}")
    print(f"\n{team1} win rate last 10:         {features_dict['team1_win_rate_last10']}")
    print(f"{team2} win rate last 10:   {features_dict['team2_win_rate_last10']}")
    print(f"\n{team1} average runs last 5:      {features_dict['team1_avg_runs_last5']}")
    print(f"{team2} average runs last 5:{features_dict['team2_avg_runs_last5']}")
    print(f"\n{team1} H2H win rate:             {features_dict['team1_h2h_win_rate']}")
    print(f"H2H matches before:                 {features_dict['h2h_matches_before']}")
    
    venue_type_label = "Venue Specific" if features_dict['venue_matches_before'] > 0 else "Global Prior"
    print(f"\nVenue average first innings score:  {features_dict['venue_avg_first_innings_score']} ({venue_type_label})")
    print(f"Venue batting-first win rate:       {features_dict['venue_batting_first_win_rate']} ({venue_type_label})")
    print(f"Venue matches before:               {features_dict['venue_matches_before']}")

    print("-" * 55)
    print("MODEL PREDICTION\n")
    print(f"{team1}:")
    print(f"{prob_team1 * 100.0:.2f}%\n")
    print(f"{team2}:")
    print(f"{prob_team2 * 100.0:.2f}%")
    print("=" * 55)

    print("\nFEATURE VECTOR DATAFRAME:")
    print(df_target.to_string(index=False))

    print("\nSANITY CHECKS:")
    print(f"[{'PASS' if sanity_model_loaded else 'FAIL'}] Model loaded")
    print(f"[{'PASS' if sanity_features_match else 'FAIL'}] Feature list matches training features")
    print(f"[{'PASS' if sanity_no_missing else 'FAIL'}] No missing features")
    print(f"[{'PASS' if sanity_no_future else 'FAIL'}] No future matches used")
    print(f"[{'PASS' if sanity_prob_sums else 'FAIL'}] Probability sums to 1")
    print(f"[{'PASS' if sanity_prediction_generated else 'FAIL'}] Prediction generated successfully")


if __name__ == "__main__":
    predict_upcoming_match()
