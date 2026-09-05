import os
import pandas as pd
import numpy as np
from collections import defaultdict
try:
    from src.elo import EloCalculator
except ImportError:
    from elo import EloCalculator



def generate_prematch_features(matches_path: str, deliveries_path: str, output_path: str):
    """
    Chronological Pre-Match Feature Engineering Pipeline.
    
    Creates features strictly using historical data available BEFORE each match starts.
    Applies explicit default priors for missing/first-time entries.
    """
    print("Loading processed matches and deliveries...")
    df_matches = pd.read_csv(matches_path)
    df_deliveries = pd.read_csv(deliveries_path)

    # 1. Deterministic Chronological Sorting
    # Note: Cricsheet provides match dates rather than reliable match start timestamps in this pipeline,
    # so matches occurring on the same calendar date are ordered deterministically by match_id.
    df_matches = df_matches.sort_values(by=["date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    # 2. Pre-calculate Innings Scores per Match & Team from deliveries.csv
    # Sum total_runs by match_id, innings_number, batting_team
    innings_summary = (
        df_deliveries.groupby(["match_id", "innings_number", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    # Map (match_id, batting_team) -> innings total runs
    team_innings_scores = {}
    # Map match_id -> (1st_innings_batting_team, 1st_innings_score)
    first_innings_info = {}

    for _, row in innings_summary.iterrows():
        m_id = str(row["match_id"])
        inn_num = int(row["innings_number"])
        team = str(row["batting_team"])
        runs = float(row["total_runs"])

        team_innings_scores[(m_id, team)] = runs

    # Map match_id -> team scores & conceded runs
    match_team_runs = {}
    for _, r_m in df_matches.iterrows():
        m_id = str(r_m["match_id"])
        t1, t2 = str(r_m["team1"]), str(r_m["team2"])
        s1 = team_innings_scores.get((m_id, t1), np.nan)
        s2 = team_innings_scores.get((m_id, t2), np.nan)
        match_team_runs[m_id] = {t1: (s1, s2), t2: (s2, s1)}

    # 3. Initialize State Trackers for Chronological Computation
    elo_calc = EloCalculator(k_factor=28.0, initial_elo=1500.0)

    team_match_history = defaultdict(list)    # team -> list of 1.0 (win) or 0.0 (loss)
    team_score_history = defaultdict(list)    # team -> list of scores in past matches
    team_conceded_history = defaultdict(list) # team -> list of conceded runs in past matches
    team_venue_history = defaultdict(lambda: defaultdict(list)) # venue -> team -> list of wins/losses
    h2h_history = defaultdict(list)           # tuple(sorted([teamA, teamB])) -> list of winner team names

    venue_1st_innings_scores = defaultdict(list)  # venue -> list of 1st innings scores
    venue_batting_first_wins = defaultdict(list)  # venue -> list of 1.0 (batting 1st won) or 0.0

    global_1st_innings_scores = []
    global_batting_first_wins = []

    features_list = []

    # Define model training feature names list
    training_feature_names = [
        "team1_elo", "team2_elo", "elo_difference",
        "team1_win_rate_last3", "team2_win_rate_last3",
        "team1_win_rate_last5", "team2_win_rate_last5",
        "team1_win_rate_last10", "team2_win_rate_last10",
        "win_rate_diff_last5", "win_rate_diff_last10",
        "team1_avg_runs_last5", "team2_avg_runs_last5",
        "team1_avg_conc_last5", "team2_avg_conc_last5", "net_runs_diff5",
        "team1_h2h_win_rate", "h2h_matches_before",
        "venue_avg_first_innings_score", "venue_batting_first_win_rate", "venue_matches_before",
        "team1_ven_win_rate", "team2_ven_win_rate", "team1_toss_won"
    ]

    print("Computing chronological pre-match features...")

    for idx, row in df_matches.iterrows():
        match_id = str(row["match_id"])
        date_str = str(row["date"])
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        venue = str(row["venue"])
        winner = str(row["winner"])
        toss_winner = str(row.get("toss_winner", ""))

        # Target: team1_won (deterministic based on original team order)
        team1_won = 1 if winner == team1 else 0

        # --- A. RECORD PRE-MATCH FEATURES (STRICTLY BEFORE UPDATING STATE) ---

        # 1. Elo Features
        team1_elo = elo_calc.get_rating(team1)
        team2_elo = elo_calc.get_rating(team2)
        elo_difference = team1_elo - team2_elo

        # 2. Recent Form Features (Win Rate Last 3, 5, 10)
        hist1 = team_match_history[team1]
        hist2 = team_match_history[team2]

        team1_win_rate_last3 = float(np.mean(hist1[-3:])) if len(hist1) > 0 else 0.5
        team2_win_rate_last3 = float(np.mean(hist2[-3:])) if len(hist2) > 0 else 0.5

        team1_win_rate_last5 = float(np.mean(hist1[-5:])) if len(hist1) > 0 else 0.5
        team2_win_rate_last5 = float(np.mean(hist2[-5:])) if len(hist2) > 0 else 0.5

        team1_win_rate_last10 = float(np.mean(hist1[-10:])) if len(hist1) > 0 else 0.5
        team2_win_rate_last10 = float(np.mean(hist2[-10:])) if len(hist2) > 0 else 0.5

        win_rate_diff_last5 = team1_win_rate_last5 - team2_win_rate_last5
        win_rate_diff_last10 = team1_win_rate_last10 - team2_win_rate_last10

        # 3. Recent Scoring & Conceding Form Features (Avg Runs / Conceded Last 5)
        global_prior_score = float(np.mean(global_1st_innings_scores)) if len(global_1st_innings_scores) > 0 else 145.0

        scores1 = team_score_history[team1]
        scores2 = team_score_history[team2]
        conc1 = team_conceded_history[team1]
        conc2 = team_conceded_history[team2]

        valid_s1 = [s for s in scores1[-5:] if not np.isnan(s)]
        valid_s2 = [s for s in scores2[-5:] if not np.isnan(s)]
        valid_c1 = [c for c in conc1[-5:] if not np.isnan(c)]
        valid_c2 = [c for c in conc2[-5:] if not np.isnan(c)]

        team1_avg_runs_last5 = float(np.mean(valid_s1)) if len(valid_s1) > 0 else global_prior_score
        team2_avg_runs_last5 = float(np.mean(valid_s2)) if len(valid_s2) > 0 else global_prior_score

        team1_avg_conc_last5 = float(np.mean(valid_c1)) if len(valid_c1) > 0 else global_prior_score
        team2_avg_conc_last5 = float(np.mean(valid_c2)) if len(valid_c2) > 0 else global_prior_score

        team1_net_runs5 = team1_avg_runs_last5 - team1_avg_conc_last5
        team2_net_runs5 = team2_avg_runs_last5 - team2_avg_conc_last5
        net_runs_diff5 = team1_net_runs5 - team2_net_runs5

        # 4. Head to Head Features (H2H)
        h2h_key = tuple(sorted([team1, team2]))
        past_h2h = h2h_history[h2h_key]
        h2h_matches_before = len(past_h2h)

        if h2h_matches_before == 0:
            team1_h2h_win_rate = 0.5
        else:
            t1_h2h_wins = sum(1 for w in past_h2h if w == team1)
            team1_h2h_win_rate = float(t1_h2h_wins) / float(h2h_matches_before)

        # 5. Venue History Features
        venue_scores = venue_1st_innings_scores[venue]
        venue_wins_1st = venue_batting_first_wins[venue]
        venue_matches_before = len(venue_scores)

        global_prior_bat1st_win_rate = float(np.mean(global_batting_first_wins)) if len(global_batting_first_wins) > 0 else 0.5

        if venue_matches_before == 0:
            venue_avg_first_innings_score = global_prior_score
            venue_batting_first_win_rate = global_prior_bat1st_win_rate
        else:
            venue_avg_first_innings_score = float(np.mean(venue_scores))
            venue_batting_first_win_rate = float(np.mean(venue_wins_1st))

        # Team specific venue win rate
        t1_ven_hist = team_venue_history[venue][team1]
        t2_ven_hist = team_venue_history[venue][team2]
        team1_ven_win_rate = float(np.mean(t1_ven_hist)) if len(t1_ven_hist) >= 2 else 0.5
        team2_ven_win_rate = float(np.mean(t2_ven_hist)) if len(t2_ven_hist) >= 2 else 0.5

        # 6. Toss Feature
        team1_toss_won = 1.0 if toss_winner == team1 else (0.0 if toss_winner == team2 else 0.5)

        # Assemble row dictionary
        feature_row = {
            "match_id": match_id,
            "date": date_str,
            "team1": team1,
            "team2": team2,
            "venue": venue,

            "team1_elo": round(team1_elo, 2),
            "team2_elo": round(team2_elo, 2),
            "elo_difference": round(elo_difference, 2),

            "team1_win_rate_last3": round(team1_win_rate_last3, 4),
            "team2_win_rate_last3": round(team2_win_rate_last3, 4),

            "team1_win_rate_last5": round(team1_win_rate_last5, 4),
            "team2_win_rate_last5": round(team2_win_rate_last5, 4),

            "team1_win_rate_last10": round(team1_win_rate_last10, 4),
            "team2_win_rate_last10": round(team2_win_rate_last10, 4),

            "win_rate_diff_last5": round(win_rate_diff_last5, 4),
            "win_rate_diff_last10": round(win_rate_diff_last10, 4),

            "team1_avg_runs_last5": round(team1_avg_runs_last5, 2),
            "team2_avg_runs_last5": round(team2_avg_runs_last5, 2),

            "team1_avg_conc_last5": round(team1_avg_conc_last5, 2),
            "team2_avg_conc_last5": round(team2_avg_conc_last5, 2),
            "net_runs_diff5": round(net_runs_diff5, 2),

            "team1_h2h_win_rate": round(team1_h2h_win_rate, 4),
            "h2h_matches_before": int(h2h_matches_before),

            "venue_avg_first_innings_score": round(venue_avg_first_innings_score, 2),
            "venue_batting_first_win_rate": round(venue_batting_first_win_rate, 4),
            "venue_matches_before": int(venue_matches_before),

            "team1_ven_win_rate": round(team1_ven_win_rate, 4),
            "team2_ven_win_rate": round(team2_ven_win_rate, 4),
            "team1_toss_won": round(team1_toss_won, 2),

            "team1_won": int(team1_won)
        }

        features_list.append(feature_row)

        # --- B. UPDATE STATE TRACKERS WITH CURRENT MATCH RESULTS ---

        # Update Elo Ratings
        elo_calc.update_ratings(team1, team2, actual_a_won=float(team1_won))

        # Update Win History
        team_match_history[team1].append(1.0 if team1_won == 1 else 0.0)
        team_match_history[team2].append(0.0 if team1_won == 1 else 1.0)
        team_venue_history[venue][team1].append(1.0 if team1_won == 1 else 0.0)
        team_venue_history[venue][team2].append(0.0 if team1_won == 1 else 1.0)

        # Update Scoring & Conceded History
        tr_info = match_team_runs.get(match_id, {})
        if team1 in tr_info:
            team_score_history[team1].append(tr_info[team1][0])
            team_conceded_history[team1].append(tr_info[team1][1])
        if team2 in tr_info:
            team_score_history[team2].append(tr_info[team2][0])
            team_conceded_history[team2].append(tr_info[team2][1])

        # Update H2H History
        h2h_history[h2h_key].append(winner)

        # Update Venue & Global History
        if match_id in first_innings_info:
            first_inn_team, first_inn_runs = first_innings_info[match_id]
            batting_first_won = 1.0 if winner == first_inn_team else 0.0

            venue_1st_innings_scores[venue].append(first_inn_runs)
            venue_batting_first_wins[venue].append(batting_first_won)

            global_1st_innings_scores.append(first_inn_runs)
            global_batting_first_wins.append(batting_first_won)

    df_output = pd.DataFrame(features_list)

    # 4. Execute Strict Data Leakage Validation Checks
    print("\nExecuting Data Leakage Validation Checks...")
    leakage_checks = {}

    # Check 1: Chronological Order
    is_sorted = df_output["date"].is_monotonic_increasing
    leakage_checks["1. Rows chronologically ordered by date"] = is_sorted

    # Check 2: Features calculated BEFORE current match update (Elo for first match of any team is 1500.0)
    seen_teams = set()
    elo_pre_correct = True
    for _, r in df_output.iterrows():
        t1, t2 = r["team1"], r["team2"]
        if t1 not in seen_teams:
            if r["team1_elo"] != 1500.0:
                elo_pre_correct = False
            seen_teams.add(t1)
        if t2 not in seen_teams:
            if r["team2_elo"] != 1500.0:
                elo_pre_correct = False
            seen_teams.add(t2)
    leakage_checks["2. Elo ratings captured BEFORE current match update"] = elo_pre_correct

    # Check 3: Recent win-rate windows contain only previous matches (First match of any team has win rate 0.5)
    seen_teams_form = set()
    form_pre_correct = True
    for _, r in df_output.iterrows():
        t1, t2 = r["team1"], r["team2"]
        if t1 not in seen_teams_form:
            if r["team1_win_rate_last5"] != 0.5 or r["team1_win_rate_last10"] != 0.5:
                form_pre_correct = False
            seen_teams_form.add(t1)
        if t2 not in seen_teams_form:
            if r["team2_win_rate_last5"] != 0.5 or r["team2_win_rate_last10"] != 0.5:
                form_pre_correct = False
            seen_teams_form.add(t2)
    leakage_checks["3. Recent-form windows contain ONLY previous matches"] = form_pre_correct

    # Check 4: H2H statistics contain only previous meetings (First H2H match has h2h_matches_before == 0 and win rate 0.5)
    seen_h2h = set()
    h2h_pre_correct = True
    for _, r in df_output.iterrows():
        pair = tuple(sorted([r["team1"], r["team2"]]))
        if pair not in seen_h2h:
            if r["h2h_matches_before"] != 0 or r["team1_h2h_win_rate"] != 0.5:
                h2h_pre_correct = False
            seen_h2h.add(pair)
    leakage_checks["4. H2H statistics contain ONLY previous meetings"] = h2h_pre_correct

    # Check 5: Venue statistics contain only previous matches at that venue (First match at venue has venue_matches_before == 0)
    seen_venues = set()
    venue_pre_correct = True
    for _, r in df_output.iterrows():
        v = r["venue"]
        if v not in seen_venues:
            if r["venue_matches_before"] != 0:
                venue_pre_correct = False
            seen_venues.add(v)
    leakage_checks["5. Venue statistics contain ONLY previous matches at that venue"] = venue_pre_correct

    # Check 6: Recent scoring averages contain only previous matches (First match uses initial global prior 145.0)
    score_pre_correct = (df_output.iloc[0]["team1_avg_runs_last5"] == 145.0) and (df_output.iloc[0]["team2_avg_runs_last5"] == 145.0)
    leakage_checks["6. Recent scoring averages contain ONLY previous matches"] = score_pre_correct

    # Check 7: Target column ('team1_won') is NEVER included in model feature list
    target_excluded = "team1_won" not in training_feature_names
    leakage_checks["7. Target column 'team1_won' excluded from feature list"] = target_excluded

    # Check 8: No winner/outcome information used as input feature
    outcome_cols_in_features = [col for col in training_feature_names if "winner" in col or "outcome" in col or "result" in col]
    no_outcome_in_features = len(outcome_cols_in_features) == 0
    leakage_checks["8. No winner/outcome info used as input feature"] = no_outcome_in_features

    # Save dataset to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"\nSaved pre-match features dataset to: {output_path} ({len(df_output)} rows)")

    # Print Final Report
    print("\n" + "=" * 70)
    print("          PHASE 2 PRE-MATCH FEATURE ENGINEERING REPORT          ")
    print("=" * 70)
    print(f"Total Matches (Rows):              {len(df_output)}")
    print(f"Date Range:                        {df_output['date'].min()} to {df_output['date'].max()}")
    print(f"Training Feature Count:            {len(training_feature_names)}")
    print(f"Training Feature Names:            {training_feature_names}")

    print("\nTarget Distribution ('team1_won'):")
    target_counts = df_output["team1_won"].value_counts()
    for val, cnt in target_counts.items():
        pct = (cnt / len(df_output)) * 100.0
        print(f"  Class {val} ({'Team 1 Won' if val == 1 else 'Team 2 Won'}): {cnt:<5d} ({pct:.2f}%)")

    print("\nUnexpected Missing Values Summary:")
    missing_sum = df_output.isnull().sum()
    unexpected_missing = missing_sum[missing_sum > 0]
    if len(unexpected_missing) == 0:
        print("  0 unexpected missing values across all features!")
    else:
        for col, count in unexpected_missing.items():
            print(f"  WARNING: Feature '{col}' has {count} missing values!")

    print("\nData Leakage & Validation Checks:")
    all_passed = True
    for check_desc, passed in leakage_checks.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  [{status}] {check_desc}")

    if all_passed:
        print("\n--> ALL DATA LEAKAGE CHECKS PASSED PERFECTLY!")
    else:
        print("\n--> WARNING: SOME LEAKAGE CHECKS FAILED! Please inspect.")

    print("\nFirst 10 Rows:")
    print(df_output[["date", "team1", "team2", "team1_elo", "team2_elo", "team1_h2h_win_rate", "team1_won"]].head(10).to_string(index=False))

    print("\nLast 10 Rows:")
    print(df_output[["date", "team1", "team2", "team1_elo", "team2_elo", "team1_h2h_win_rate", "team1_won"]].tail(10).to_string(index=False))

    # Inspect recent Zimbabwe and South Africa rows
    zim_rows = df_output[(df_output["team1"] == "Zimbabwe") | (df_output["team2"] == "Zimbabwe")].tail(5)
    sa_rows = df_output[(df_output["team1"] == "South Africa") | (df_output["team2"] == "South Africa")].tail(5)

    print("\nRecent Zimbabwe Matches (Last 5):")
    print(zim_rows[["date", "team1", "team2", "team1_elo", "team2_elo", "team1_win_rate_last5", "team1_avg_runs_last5", "team1_won"]].to_string(index=False))

    print("\nRecent South Africa Matches (Last 5):")
    print(sa_rows[["date", "team1", "team2", "team1_elo", "team2_elo", "team1_win_rate_last5", "team1_avg_runs_last5", "team1_won"]].to_string(index=False))

    print("=" * 70)


if __name__ == "__main__":
    matches_csv = os.path.join("data", "processed", "matches.csv")
    deliveries_csv = os.path.join("data", "processed", "deliveries.csv")
    output_csv = os.path.join("data", "processed", "prematch_features.csv")

    generate_prematch_features(matches_csv, deliveries_csv, output_csv)
