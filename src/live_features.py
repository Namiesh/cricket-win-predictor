"""
Phase 4A — Live Win-Probability Training Dataset Builder

Creates ball-by-ball match-state snapshots for every second innings (chase)
in historical T20I matches. Each snapshot captures the game state after a
delivery and labels whether the chasing team eventually won.

Includes pre-match Elo ratings for batting and bowling teams, captured
BEFORE each match result is incorporated.

Output: data/processed/live_features.csv
"""

import os
import json
import glob
import time
import numpy as np
import pandas as pd
from collections import defaultdict
try:
    from src.elo import EloCalculator
except ImportError:
    from elo import EloCalculator



# ── Wicket types that count as an actual dismissal ───────────────────────────
DISMISSAL_TYPES = {
    "caught", "bowled", "lbw", "run out", "stumped",
    "caught and bowled", "hit wicket", "obstructing the field",
    "hit the ball twice", "retired out",
}
# "retired hurt" and "retired not out" are NOT dismissals.


def _load_target_info_from_json(json_dir: str, valid_match_ids: set):
    """
    Read every raw Cricsheet JSON and extract:
      - target runs / overs from the 2nd innings key
      - outcome method (D/L, Awarded, etc.)
      - whether the match has a super-over (>2 innings)
    Returns a dict keyed by match_id.
    """
    target_info = {}
    for fpath in glob.glob(os.path.join(json_dir, "*.json")):
        mid = int(os.path.splitext(os.path.basename(fpath))[0])
        if mid not in valid_match_ids:
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        info = data.get("info", {})
        outcome = info.get("outcome", {})
        method = outcome.get("method", None)
        innings = data.get("innings", [])

        has_super_over = len(innings) > 2
        has_target_key = len(innings) >= 2 and "target" in innings[1]

        target_runs = None
        target_overs = None
        if has_target_key:
            tgt = innings[1]["target"]
            target_runs = tgt.get("runs", None)
            target_overs = tgt.get("overs", None)

        target_info[mid] = {
            "target_runs": target_runs,
            "target_overs": target_overs,
            "method": method,
            "has_super_over": has_super_over,
            "has_target_key": has_target_key,
        }
    return target_info


def generate_live_features():
    t_start = time.time()

    OLD_FEATURE_COUNT = 16  # Previous version without Elo

    # ── Paths ────────────────────────────────────────────────────────────────
    matches_csv = os.path.join("data", "processed", "matches.csv")
    deliveries_csv = os.path.join("data", "processed", "deliveries.csv")
    json_dir = os.path.join("data", "raw", "t20s_male_json")
    output_csv = os.path.join("data", "processed", "live_features.csv")

    # ── Load processed data ──────────────────────────────────────────────────
    df_matches = pd.read_csv(matches_csv)
    df_deliveries = pd.read_csv(deliveries_csv)

    valid_match_ids = set(df_matches["match_id"].tolist())
    total_completed = len(df_matches)

    # ── Load target info from raw JSON ───────────────────────────────────────
    target_info = _load_target_info_from_json(json_dir, valid_match_ids)

    # ── Categorise exclusions ────────────────────────────────────────────────
    excl_no_result = 0
    excl_abandoned = 0
    excl_no_target = 0
    excl_dls = 0
    excl_super_over = 0
    excl_awarded = 0
    excl_other = 0

    eligible_match_ids = []

    for _, row in df_matches.iterrows():
        mid = row["match_id"]
        outcome_result = str(row.get("outcome_result", ""))
        winner = row.get("winner", None)

        # Already filtered in Phase 1, but double-check
        if outcome_result in ("no result",):
            excl_no_result += 1
            continue
        if outcome_result in ("abandoned",):
            excl_abandoned += 1
            continue
        if pd.isna(winner) or str(winner).strip() == "":
            excl_no_result += 1
            continue

        tinfo = target_info.get(mid)
        if tinfo is None:
            excl_other += 1
            continue

        if tinfo["has_super_over"]:
            excl_super_over += 1
            continue

        if tinfo["method"] == "Awarded":
            excl_awarded += 1
            continue

        if tinfo["method"] == "D/L":
            excl_dls += 1
            continue

        if not tinfo["has_target_key"] or tinfo["target_runs"] is None:
            excl_no_target += 1
            continue

        eligible_match_ids.append(mid)

    eligible_set = set(eligible_match_ids)

    # ── Build match metadata lookup ──────────────────────────────────────────
    match_meta = {}
    for _, row in df_matches.iterrows():
        mid = row["match_id"]
        if mid in eligible_set:
            match_meta[mid] = {
                "date": row["date"],
                "winner": str(row["winner"]),
                "team1": str(row["team1"]),
                "team2": str(row["team2"]),
            }

    # ── Build chronologically ordered match list ─────────────────────────────
    # ALL matches (not just eligible) must be replayed for Elo computation,
    # but only eligible matches produce snapshots.
    df_all_chrono = df_matches.sort_values(
        by=["date", "match_id"], ascending=[True, True]
    ).reset_index(drop=True)

    # ── Pre-index 2nd innings deliveries by match_id ─────────────────────────
    df_inn2 = df_deliveries[
        (df_deliveries["match_id"].isin(eligible_set)) &
        (df_deliveries["innings_number"] == 2)
    ].copy()

    df_inn2 = df_inn2.sort_values(
        by=["match_id", "over", "delivery_number"],
        ascending=True,
    ).reset_index(drop=True)

    # Group deliveries by match_id for fast lookup
    inn2_groups = {mid: grp for mid, grp in df_inn2.groupby("match_id", sort=False)}

    # ── Chronological Elo replay + snapshot generation ───────────────────────
    elo_calc = EloCalculator(k_factor=32.0, initial_elo=1500.0)
    snapshot_rows = []
    matches_processed = 0
    prematch_elo_log = {}  # match_id -> (batting_elo, bowling_elo) for validation

    for _, match_row in df_all_chrono.iterrows():
        mid = match_row["match_id"]
        team1 = str(match_row["team1"])
        team2 = str(match_row["team2"])
        winner = str(match_row["winner"])

        # ── Step 1: Capture PRE-MATCH Elo ratings ────────────────────────
        pre_elo_team1 = elo_calc.get_rating(team1)
        pre_elo_team2 = elo_calc.get_rating(team2)

        # ── Step 2: Generate snapshots if this match is eligible ─────────
        if mid in eligible_set and mid in inn2_groups:
            grp = inn2_groups[mid]
            meta = match_meta[mid]
            tinfo = target_info[mid]
            target = tinfo["target_runs"]
            match_date = meta["date"]

            batting_team = grp["batting_team"].iloc[0]
            bowling_team = grp["bowling_team"].iloc[0]

            batting_team_won = 1 if winner == batting_team else 0

            # Determine pre-match Elo for batting and bowling teams
            if batting_team == team1:
                batting_elo = round(pre_elo_team1, 2)
                bowling_elo = round(pre_elo_team2, 2)
            else:
                batting_elo = round(pre_elo_team2, 2)
                bowling_elo = round(pre_elo_team1, 2)

            elo_diff = round(batting_elo - bowling_elo, 2)

            # Store for validation
            prematch_elo_log[mid] = (batting_elo, bowling_elo)

            cumulative_score = 0
            cumulative_wickets = 0
            legal_balls = 0
            max_legal_balls = 120  # Standard T20

            for _, ball_row in grp.iterrows():
                runs_this_ball = int(ball_row["total_runs"])
                cumulative_score += runs_this_ball

                # ── Determine if this is a legal ball ────────────────────
                is_wide = int(ball_row.get("wides", 0)) > 0
                is_noball = int(ball_row.get("noballs", 0)) > 0
                is_legal = not is_wide and not is_noball

                if is_legal:
                    legal_balls += 1

                # ── Determine if a wicket fell ───────────────────────────
                is_wicket = int(ball_row.get("is_wicket", 0))
                wicket_type = str(ball_row.get("wicket_type", "")).strip().lower()

                if is_wicket == 1 and wicket_type in DISMISSAL_TYPES:
                    cumulative_wickets += 1

                # ── Derived features ─────────────────────────────────────
                overs_completed = legal_balls // 6
                balls_remaining = max(max_legal_balls - legal_balls, 0)
                wickets_remaining = max(10 - cumulative_wickets, 0)
                runs_required = max(target - cumulative_score, 0)

                # Current run rate
                if legal_balls > 0:
                    overs_decimal = legal_balls / 6.0
                    current_run_rate = round(cumulative_score / overs_decimal, 4)
                else:
                    current_run_rate = 0.0

                # Required run rate
                if balls_remaining > 0:
                    overs_remaining = balls_remaining / 6.0
                    required_run_rate = round(runs_required / overs_remaining, 4)
                else:
                    required_run_rate = 36.0 if runs_required > 0 else 0.0

                run_rate_difference = round(current_run_rate - required_run_rate, 4)

                rrr_crr_ratio = round(required_run_rate / (current_run_rate + 0.1), 4)
                runs_per_wicket_needed = round(runs_required / (wickets_remaining + 0.1), 4)
                pressure_index = round(required_run_rate * (10.0 / (wickets_remaining + 0.5)), 4)
                balls_per_wicket_remaining = round(balls_remaining / (wickets_remaining + 0.1), 4)
                phase_powerplay = 1.0 if legal_balls <= 36 else 0.0
                phase_middle = 1.0 if 36 < legal_balls <= 90 else 0.0
                phase_death = 1.0 if balls_remaining <= 30 else 0.0

                snapshot_rows.append({
                    "match_id": mid,
                    "match_date": match_date,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batting_team_elo": batting_elo,
                    "bowling_team_elo": bowling_elo,
                    "elo_difference": elo_diff,
                    "score": cumulative_score,
                    "wickets": cumulative_wickets,
                    "overs_completed": overs_completed,
                    "balls_completed": legal_balls,
                    "balls_remaining": balls_remaining,
                    "target": target,
                    "runs_required": runs_required,
                    "wickets_remaining": wickets_remaining,
                    "current_run_rate": current_run_rate,
                    "required_run_rate": required_run_rate,
                    "run_rate_difference": run_rate_difference,
                    "rrr_crr_ratio": rrr_crr_ratio,
                    "runs_per_wicket_needed": runs_per_wicket_needed,
                    "pressure_index": pressure_index,
                    "balls_per_wicket_remaining": balls_per_wicket_remaining,
                    "phase_powerplay": phase_powerplay,
                    "phase_middle": phase_middle,
                    "phase_death": phase_death,
                    "batting_team_won": batting_team_won,
                })


            matches_processed += 1

        # ── Step 3: Update Elo AFTER the match is fully processed ────────
        team1_won = 1.0 if winner == team1 else 0.0
        elo_calc.update_ratings(team1, team2, actual_a_won=team1_won)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df_live = pd.DataFrame(snapshot_rows)

    # Ensure chronological ordering by match_date, match_id, then delivery order
    # (already built in chronological order, but enforce the sort for safety)
    df_live = df_live.sort_values(
        by=["match_date", "match_id"],
        ascending=[True, True],
        kind="mergesort",  # Stable sort preserves within-match delivery order
    ).reset_index(drop=True)

    NEW_FEATURE_COUNT = len(df_live.columns)

    # ── Data Integrity / Leakage Checks ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA INTEGRITY & LEAKAGE CHECKS")
    print("=" * 60)

    checks_passed = 0
    checks_total = 11

    # Check 1: Score never decreases within a match
    check1_pass = True
    for mid, grp in df_live.groupby("match_id"):
        if not (grp["score"].diff().dropna() >= 0).all():
            check1_pass = False
            break
    print(f"[{'PASS' if check1_pass else 'FAIL'}] Score never decreases within a match")
    checks_passed += int(check1_pass)

    # Check 2: Wickets never decrease within a match
    check2_pass = True
    for mid, grp in df_live.groupby("match_id"):
        if not (grp["wickets"].diff().dropna() >= 0).all():
            check2_pass = False
            break
    print(f"[{'PASS' if check2_pass else 'FAIL'}] Wickets never decrease within a match")
    checks_passed += int(check2_pass)

    # Check 3: balls_remaining never increases within a match
    check3_pass = True
    for mid, grp in df_live.groupby("match_id"):
        if not (grp["balls_remaining"].diff().dropna() <= 0).all():
            check3_pass = False
            break
    print(f"[{'PASS' if check3_pass else 'FAIL'}] balls_remaining never increases within a match")
    checks_passed += int(check3_pass)

    # Check 4: wickets_remaining in [0, 10]
    check4_pass = (df_live["wickets_remaining"] >= 0).all() and (df_live["wickets_remaining"] <= 10).all()
    print(f"[{'PASS' if check4_pass else 'FAIL'}] wickets_remaining in [0, 10]")
    checks_passed += int(check4_pass)

    # Check 5: batting_team_won is only 0 or 1
    check5_pass = set(df_live["batting_team_won"].unique()) <= {0, 1}
    print(f"[{'PASS' if check5_pass else 'FAIL'}] batting_team_won is binary (0/1)")
    checks_passed += int(check5_pass)

    # Check 6: No NaN in any column
    check6_nan = df_live.isna().sum().sum()
    check6_pass = check6_nan == 0
    print(f"[{'PASS' if check6_pass else 'FAIL'}] No missing values (NaN={check6_nan})")
    checks_passed += int(check6_pass)

    # Check 7: No Inf in numeric columns
    check7_inf = np.isinf(df_live.select_dtypes(include=[np.number])).sum().sum()
    check7_pass = check7_inf == 0
    print(f"[{'PASS' if check7_pass else 'FAIL'}] No infinite values (Inf={check7_inf})")
    checks_passed += int(check7_pass)

    # Check 8: All snapshots from one match are grouped (consecutive match_ids)
    check8_pass = True
    seen = set()
    prev_mid = None
    for mid in df_live["match_id"]:
        if mid != prev_mid:
            if mid in seen:
                check8_pass = False
                break
            seen.add(mid)
        prev_mid = mid
    print(f"[{'PASS' if check8_pass else 'FAIL'}] Snapshots grouped by match_id (no interleaving)")
    checks_passed += int(check8_pass)

    # ── ELO-SPECIFIC LEAKAGE CHECKS ─────────────────────────────────────────

    # Check 9: All snapshots from the same match use identical Elo values
    check9_pass = True
    for mid, grp in df_live.groupby("match_id"):
        bat_elos = grp["batting_team_elo"].unique()
        bowl_elos = grp["bowling_team_elo"].unique()
        if len(bat_elos) != 1 or len(bowl_elos) != 1:
            check9_pass = False
            print(f"  FAIL detail: match {mid} has {len(bat_elos)} batting Elo values, {len(bowl_elos)} bowling Elo values")
            break
    print(f"[{'PASS' if check9_pass else 'FAIL'}] All snapshots in a match share the same pre-match Elo")
    checks_passed += int(check9_pass)

    # Check 10: Elo is captured BEFORE current match (verify against elo_log)
    # We verify that the stored Elo matches what the EloCalculator had before the
    # match was processed. This was guaranteed by construction (Step 1 before Step 3),
    # but we validate the stored values match.
    check10_pass = True
    for mid in prematch_elo_log:
        expected_bat, expected_bowl = prematch_elo_log[mid]
        match_snaps = df_live[df_live["match_id"] == mid]
        if len(match_snaps) == 0:
            continue
        actual_bat = match_snaps["batting_team_elo"].iloc[0]
        actual_bowl = match_snaps["bowling_team_elo"].iloc[0]
        if abs(actual_bat - expected_bat) > 1e-4 or abs(actual_bowl - expected_bowl) > 1e-4:
            check10_pass = False
            print(f"  FAIL detail: match {mid} Elo mismatch")
            break
    print(f"[{'PASS' if check10_pass else 'FAIL'}] Elo values match pre-match capture log")
    checks_passed += int(check10_pass)

    # Check 11: elo_difference == batting_team_elo - bowling_team_elo
    check11_diff = (df_live["elo_difference"] - (df_live["batting_team_elo"] - df_live["bowling_team_elo"])).abs()
    check11_pass = (check11_diff < 0.02).all()  # tolerance for rounding
    print(f"[{'PASS' if check11_pass else 'FAIL'}] elo_difference = batting_team_elo - bowling_team_elo")
    checks_passed += int(check11_pass)

    print(f"\nChecks passed: {checks_passed}/{checks_total}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_live.to_csv(output_csv, index=False)

    t_elapsed = time.time() - t_start

    # ── Report ───────────────────────────────────────────────────────────────
    class_dist = df_live["batting_team_won"].value_counts()
    class_1 = int(class_dist.get(1, 0))
    class_0 = int(class_dist.get(0, 0))
    date_min = df_live["match_date"].min()
    date_max = df_live["match_date"].max()
    avg_snapshots = df_live.groupby("match_id").size().mean()
    total_missing = df_live.isna().sum().sum()
    total_inf = np.isinf(df_live.select_dtypes(include=[np.number])).sum().sum()

    print("\n" + "=" * 60)
    print("LIVE FEATURE DATASET")
    print("=" * 60)

    print(f"\nOld feature count:                   {OLD_FEATURE_COUNT}")
    print(f"New feature count:                   {NEW_FEATURE_COUNT}")
    print(f"\nNew features added:")
    print(f"  batting_team_elo")
    print(f"  bowling_team_elo")
    print(f"  elo_difference")

    print(f"\nCompleted matches in database:       {total_completed}")
    print(f"Matches excluded:                    {total_completed - matches_processed}")
    print(f"Second innings processed:            {matches_processed}")
    print()
    print(f"Total snapshots:                     {len(df_live)}")
    print(f"Average snapshots per innings:        {avg_snapshots:.1f}")
    print()
    print(f"Date range:                          {date_min} to {date_max}")
    print()
    print(f"Feature count:                       {NEW_FEATURE_COUNT}")
    print(f"Columns:                             {list(df_live.columns)}")
    print()
    print(f"Class distribution:")
    print(f"  batting_team_won = 1:              {class_1} ({class_1/len(df_live)*100:.1f}%)")
    print(f"  batting_team_won = 0:              {class_0} ({class_0/len(df_live)*100:.1f}%)")
    print()
    print(f"Missing values:                      {total_missing}")
    print(f"Infinite values:                     {total_inf}")
    print()
    print(f"Exclusion counts:")
    print(f"  No result:                         {excl_no_result}")
    print(f"  Abandoned:                         {excl_abandoned}")
    print(f"  Target unavailable:                {excl_no_target}")
    print(f"  DLS/revised target excluded:       {excl_dls}")
    print(f"  Super over:                        {excl_super_over}")
    print(f"  Awarded:                           {excl_awarded}")
    print(f"  Other:                             {excl_other}")
    print()
    print(f"Processing time:                     {t_elapsed:.2f}s")
    print()
    print(f"Saved to:                            {output_csv}")

    # ── Sanity Check — 3 sample matches, 5 snapshots each ───────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECK — SAMPLE MATCH SNAPSHOTS")
    print("=" * 60)

    sample_match_ids = df_live["match_id"].unique()
    # Pick 3 well-spaced matches (early, middle, late)
    n_matches = len(sample_match_ids)
    indices = [0, n_matches // 2, n_matches - 1]
    selected_ids = [sample_match_ids[i] for i in indices]

    cols_to_show = [
        "batting_team", "bowling_team",
        "batting_team_elo", "bowling_team_elo",
        "score", "wickets",
        "balls_remaining", "target", "runs_required",
        "required_run_rate", "batting_team_won",
    ]

    for sample_mid in selected_ids:
        match_snaps = df_live[df_live["match_id"] == sample_mid]
        meta = match_meta.get(sample_mid, {})
        n_snaps = len(match_snaps)

        print(f"\n--- Match {sample_mid} | {meta.get('date', '?')} ---")
        print(f"    {match_snaps['batting_team'].iloc[0]} (Elo {match_snaps['batting_team_elo'].iloc[0]}) chasing {match_snaps['target'].iloc[0]}")
        print(f"    vs {match_snaps['bowling_team'].iloc[0]} (Elo {match_snaps['bowling_team_elo'].iloc[0]})")
        print(f"    Elo difference: {match_snaps['elo_difference'].iloc[0]}")
        print(f"    Total snapshots: {n_snaps}")
        print(f"    Result: {'Chasing team WON' if match_snaps['batting_team_won'].iloc[0] == 1 else 'Chasing team LOST'}")
        print()

        # Show 5 evenly-spaced snapshots
        if n_snaps <= 5:
            sample_indices = list(range(n_snaps))
        else:
            sample_indices = [0, n_snaps // 4, n_snaps // 2, 3 * n_snaps // 4, n_snaps - 1]

        sample_df = match_snaps.iloc[sample_indices][cols_to_show]
        print(sample_df.to_string(index=False))

        # Verify monotonicity within this match
        scores = match_snaps["score"].values
        wickets = match_snaps["wickets"].values
        balls_rem = match_snaps["balls_remaining"].values

        score_ok = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
        wickets_ok = all(wickets[i] <= wickets[i + 1] for i in range(len(wickets) - 1))
        balls_ok = all(balls_rem[i] >= balls_rem[i + 1] for i in range(len(balls_rem) - 1))

        print(f"    Score non-decreasing: {'OK' if score_ok else 'FAIL'}")
        print(f"    Wickets non-decreasing: {'OK' if wickets_ok else 'FAIL'}")
        print(f"    Balls remaining non-increasing: {'OK' if balls_ok else 'FAIL'}")


if __name__ == "__main__":
    generate_live_features()
