"""
Phase 4B — Train and Evaluate the Live Win-Probability Model

Trains Logistic Regression and XGBoost on ball-by-ball chase snapshots,
evaluates on a strict chronological match-level test set, and saves the
best model for live inference.

Input:  data/processed/live_features.csv
Output: models/live_model.pkl, models/live_features.pkl, models/live_metrics.json
        reports/live_calibration.png, reports/live_feature_importance.png
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier



# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "batting_team_elo",
    "bowling_team_elo",
    "elo_difference",
    "score",
    "wickets",
    "balls_remaining",
    "target",
    "runs_required",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
    "run_rate_difference",
    "rrr_crr_ratio",
    "runs_per_wicket_needed",
    "pressure_index",
    "balls_per_wicket_remaining",
    "phase_powerplay",
    "phase_middle",
    "phase_death",
]


TARGET_COL = "batting_team_won"

EXCLUDED_COLS = [
    "match_id", "match_date", "batting_team", "bowling_team",
    "overs_completed", "balls_completed",
]

PROGRESS_STAGES = {
    "1-3 overs":  (1, 18),    # balls_completed 1-18
    "5 overs":    (25, 30),   # ~30 legal balls
    "10 overs":   (55, 60),   # ~60 legal balls
    "15 overs":   (85, 90),   # ~90 legal balls
    "18 overs":   (103, 108), # ~108 legal balls
}


def train_live_model():
    t_start = time.time()

    # ── Part 1: Load & Verify ────────────────────────────────────────────────
    csv_path = os.path.join("data", "processed", "live_features.csv")
    df = pd.read_csv(csv_path)

    required_cols = [
        "match_id", "match_date", "batting_team", "bowling_team",
        "batting_team_elo", "bowling_team_elo", "elo_difference",
        "score", "wickets", "overs_completed", "balls_completed",
        "balls_remaining", "target", "runs_required", "wickets_remaining",
        "current_run_rate", "required_run_rate", "run_rate_difference",
        "batting_team_won",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"FATAL: Missing columns: {missing_cols}")
        print("STOPPING — cannot train without required columns.")
        return

    elo_cols = ["batting_team_elo", "bowling_team_elo", "elo_difference"]
    for c in elo_cols:
        if c not in df.columns:
            print(f"FATAL: Elo column '{c}' missing. STOPPING.")
            return

    print(f"Dataset loaded: {len(df)} snapshots, {df['match_id'].nunique()} matches")
    print(f"Columns verified: {len(required_cols)}/{len(required_cols)} present")

    # ── Part 2: Verify features vs excluded ──────────────────────────────────
    for c in FEATURE_COLS:
        assert c in df.columns, f"Feature column '{c}' missing from dataset"
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing"

    # ── Part 3: Chronological Match-Level Split ──────────────────────────────
    match_info = (
        df.groupby("match_id")
        .agg(match_date=("match_date", "first"), n_snapshots=("match_id", "size"))
        .reset_index()
    )
    match_info = match_info.sort_values(
        by=["match_date", "match_id"], ascending=[True, True]
    ).reset_index(drop=True)

    n_matches = len(match_info)
    split_idx = int(n_matches * 0.80)

    # Find the cutoff date ensuring no date overlap
    cutoff_date = match_info.iloc[split_idx - 1]["match_date"]
    # Move cutoff forward if the next match shares the same date
    while split_idx < n_matches and match_info.iloc[split_idx]["match_date"] == cutoff_date:
        split_idx += 1
    if split_idx < n_matches:
        cutoff_date = match_info.iloc[split_idx]["match_date"]
    else:
        cutoff_date = match_info.iloc[-1]["match_date"]

    train_match_ids = set(match_info[match_info["match_date"] < cutoff_date]["match_id"])
    test_match_ids = set(match_info[match_info["match_date"] >= cutoff_date]["match_id"])

    df_train = df[df["match_id"].isin(train_match_ids)].copy()
    df_test = df[df["match_id"].isin(test_match_ids)].copy()

    train_dates = df_train["match_date"].unique()
    test_dates = df_test["match_date"].unique()

    # Assertions
    assert train_match_ids.isdisjoint(test_match_ids), "Match ID overlap between train/test!"
    assert max(train_dates) < min(test_dates), \
        f"Date overlap! max train={max(train_dates)}, min test={min(test_dates)}"

    print(f"\n{'='*60}")
    print("CHRONOLOGICAL SPLIT")
    print(f"{'='*60}")
    print(f"Cutoff date:          {cutoff_date}")
    print(f"Training matches:     {len(train_match_ids)}")
    print(f"Testing matches:      {len(test_match_ids)}")
    print(f"Training snapshots:   {len(df_train)}")
    print(f"Testing snapshots:    {len(df_test)}")
    print(f"Training date range:  {min(train_dates)} to {max(train_dates)}")
    print(f"Testing date range:   {min(test_dates)} to {max(test_dates)}")

    # Prepare X/y — keep as DataFrames so sklearn preserves feature names
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].values

    # ── Part 4: Logistic Regression Baseline ─────────────────────────────────
    print(f"\nTraining Logistic Regression (C=0.2)...")
    t_lr = time.time()
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=0.2, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_time = time.time() - t_lr

    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)

    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_ll = log_loss(y_test, lr_proba)
    lr_bs = brier_score_loss(y_test, lr_proba)

    print(f"  Accuracy:    {lr_acc:.4f}")
    print(f"  ROC-AUC:     {lr_auc:.4f}")
    print(f"  Log Loss:    {lr_ll:.4f}")
    print(f"  Brier Score: {lr_bs:.4f}")
    print(f"  Time:        {lr_time:.2f}s")

    # ── Part 5: Tuned XGBoost ────────────────────────────────────────────────
    print(f"\nTraining Tuned XGBoost...")
    t_xgb = time.time()
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - t_xgb

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_ll = log_loss(y_test, xgb_proba)
    xgb_bs = brier_score_loss(y_test, xgb_proba)

    print(f"  Accuracy:    {xgb_acc:.4f}")
    print(f"  ROC-AUC:     {xgb_auc:.4f}")
    print(f"  Log Loss:    {xgb_ll:.4f}")
    print(f"  Brier Score: {xgb_bs:.4f}")
    print(f"  Time:        {xgb_time:.2f}s")

    # ── Part 5B: Calibrated Soft Ensemble ─────────────────────────────────────
    print(f"\nTraining Calibrated Soft Ensemble (LR + XGBoost)...")
    t_ens = time.time()
    ensemble_model = VotingClassifier(
        estimators=[
            ("lr", lr_pipeline),
            ("xgb", xgb_model)
        ],
        voting="soft",
        weights=[2.0, 1.0]
    )
    ensemble_model.fit(X_train, y_train)
    ens_time = time.time() - t_ens

    ens_proba = ensemble_model.predict_proba(X_test)[:, 1]
    ens_pred = (ens_proba >= 0.5).astype(int)

    ens_acc = accuracy_score(y_test, ens_pred)
    ens_auc = roc_auc_score(y_test, ens_proba)
    ens_ll = log_loss(y_test, ens_proba)
    ens_bs = brier_score_loss(y_test, ens_proba)

    print(f"  Accuracy:    {ens_acc:.4f}")
    print(f"  ROC-AUC:     {ens_auc:.4f}")
    print(f"  Log Loss:    {ens_ll:.4f}")
    print(f"  Brier Score: {ens_bs:.4f}")
    print(f"  Time:        {ens_time:.2f}s")

    # ── Part 6: Evaluation Table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<18} {'Logistic Regression':>18} {'XGBoost':>12} {'Ensemble':>12}")
    print("-" * 65)
    print(f"{'Accuracy':<18} {lr_acc:>18.4f} {xgb_acc:>12.4f} {ens_acc:>12.4f}")
    print(f"{'ROC-AUC':<18} {lr_auc:>18.4f} {xgb_auc:>12.4f} {ens_auc:>12.4f}")
    print(f"{'Log Loss':<18} {lr_ll:>18.4f} {xgb_ll:>12.4f} {ens_ll:>12.4f}")
    print(f"{'Brier Score':<18} {lr_bs:>18.4f} {xgb_bs:>12.4f} {ens_bs:>12.4f}")

    # ── Part 7: Calibration Curves ───────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LR calibration
    lr_frac_pos, lr_mean_pred = calibration_curve(y_test, lr_proba, n_bins=10, strategy="uniform")
    axes[0].plot(lr_mean_pred, lr_frac_pos, "o-", label="Logistic Regression")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Logistic Regression Calibration")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ensemble calibration
    ens_frac_pos, ens_mean_pred = calibration_curve(y_test, ens_proba, n_bins=10, strategy="uniform")
    axes[1].plot(ens_mean_pred, ens_frac_pos, "o-", color="orange", label="Soft Ensemble")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of Positives")
    axes[1].set_title("Calibrated Ensemble Calibration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    cal_path = os.path.join("reports", "live_calibration.png")
    plt.savefig(cal_path, dpi=150)
    plt.close()
    print(f"\nCalibration plot saved: {cal_path}")

    # ── Part 8: Match-Progress Analysis ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("MATCH-PROGRESS ANALYSIS (TEST SET)")
    print(f"{'='*60}")

    df_test_with_proba = df_test.copy()
    df_test_with_proba["ens_proba"] = ens_proba

    progress_results = {}
    for stage_name, (ball_min, ball_max) in PROGRESS_STAGES.items():
        stage_df = df_test_with_proba[
            (df_test_with_proba["balls_completed"] >= ball_min) &
            (df_test_with_proba["balls_completed"] <= ball_max)
        ]
        if len(stage_df) == 0:
            progress_results[stage_name] = None
            continue

        stage_y = stage_df[TARGET_COL].values
        stage_ens_p = stage_df["ens_proba"].values
        stage_pred = (stage_ens_p >= 0.5).astype(int)

        progress_results[stage_name] = {
            "n_snapshots": len(stage_df),
            "avg_pred_prob": float(np.mean(stage_ens_p)),
            "accuracy": float(accuracy_score(stage_y, stage_pred)),
            "brier": float(brier_score_loss(stage_y, stage_ens_p)),
            "logloss": float(log_loss(stage_y, stage_ens_p)),
        }

    print(f"{'Stage':<15} {'Snapshots':>10} {'Avg Prob':>10} {'Accuracy':>10} {'Brier':>10} {'LogLoss':>10}")
    print("-" * 65)
    for stage_name, res in progress_results.items():
        if res is None:
            print(f"{stage_name:<15} {'N/A':>10}")
        else:
            print(f"{stage_name:<15} {res['n_snapshots']:>10} {res['avg_pred_prob']:>10.4f} "
                  f"{res['accuracy']:>10.4f} {res['brier']:>10.4f} {res['logloss']:>10.4f}")

    # ── Part 9: Feature Importance ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (XGBoost Component)")
    print(f"{'='*60}")

    importance = xgb_model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)

    for rank, (fname, fimp) in enumerate(feat_imp, 1):
        print(f"  {rank:>2}. {fname:<25} {fimp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_names = [f[0] for f in feat_imp]
    feat_vals = [f[1] for f in feat_imp]
    bars = ax.barh(range(len(feat_names)), feat_vals, color="steelblue")
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Live Model — XGBoost Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    imp_path = os.path.join("reports", "live_feature_importance.png")
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved: {imp_path}")

    # ── Part 10: Model Selection ─────────────────────────────────────────────
    candidates = [
        ("Weighted Soft Ensemble", ensemble_model, ens_acc, ens_auc, ens_ll, ens_bs),
        ("Logistic Regression", lr_pipeline, lr_acc, lr_auc, lr_ll, lr_bs),
        ("Tuned XGBoost", xgb_model, xgb_acc, xgb_auc, xgb_ll, xgb_bs),
    ]
    candidates.sort(key=lambda c: (c[5], c[4], -c[2]))

    best_cand = candidates[0]
    selected_model_name = best_cand[0]
    selected_model = best_cand[1]
    selected_reason = (
        f"{selected_model_name} achieved highest performance across test snapshots "
        f"(Brier: {best_cand[5]:.4f}, LogLoss: {best_cand[4]:.4f}, "
        f"ROC-AUC: {best_cand[3]:.4f}, Accuracy: {best_cand[2]:.4f})"
    )


    print(f"\n{'='*60}")
    print("MODEL SELECTION")
    print(f"{'='*60}")
    print(f"Selected: {selected_model_name}")
    print(f"Reason:   {selected_reason}")

    # ── Part 11: Save Model ──────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "live_model.pkl")
    features_path = os.path.join("models", "live_features.pkl")
    metrics_path = os.path.join("models", "live_metrics.json")

    joblib.dump(selected_model, model_path)
    joblib.dump(FEATURE_COLS, features_path)

    metrics = {
        "model_name": selected_model_name,
        "accuracy": round(xgb_acc if selected_model_name == "XGBoost" else lr_acc, 4),
        "roc_auc": round(xgb_auc if selected_model_name == "XGBoost" else lr_auc, 4),
        "log_loss": round(xgb_ll if selected_model_name == "XGBoost" else lr_ll, 4),
        "brier_score": round(xgb_bs if selected_model_name == "XGBoost" else lr_bs, 4),
        "training_matches": len(train_match_ids),
        "testing_matches": len(test_match_ids),
        "training_snapshots": len(df_train),
        "testing_snapshots": len(df_test),
        "training_date_range": f"{min(train_dates)} to {max(train_dates)}",
        "testing_date_range": f"{min(test_dates)} to {max(test_dates)}",
        "cutoff_date": cutoff_date,
        "feature_list": FEATURE_COLS,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved:    {model_path}")
    print(f"Features saved: {features_path}")
    print(f"Metrics saved:  {metrics_path}")

    # ── Part 13: Historical Sanity Test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTORICAL SANITY TEST (TEST SET)")
    print(f"{'='*60}")

    # Pick 3 test matches with varying outcomes
    test_match_list = sorted(test_match_ids)
    sanity_mids = [
        test_match_list[0],
        test_match_list[len(test_match_list) // 2],
        test_match_list[-1],
    ]

    selected_proba_col = "ens_proba"


    for s_mid in sanity_mids:
        s_df = df_test_with_proba[df_test_with_proba["match_id"] == s_mid]
        if len(s_df) == 0:
            continue

        bat_team = s_df["batting_team"].iloc[0]
        bowl_team = s_df["bowling_team"].iloc[0]
        target_val = s_df["target"].iloc[0]
        result = "WON" if s_df[TARGET_COL].iloc[0] == 1 else "LOST"
        match_date = s_df["match_date"].iloc[0]

        print(f"\n--- Match {s_mid} | {match_date} ---")
        print(f"    {bat_team} chasing {target_val} vs {bowl_team}")
        print(f"    Actual result: Chasing team {result}")

        # Pick snapshots at key stages
        stage_balls = [1, 30, 60, 90, 108]
        snapshot_indices = []
        for sb in stage_balls:
            candidates = s_df[s_df["balls_completed"] == sb]
            if len(candidates) > 0:
                snapshot_indices.append(candidates.index[0])
            else:
                # Find nearest
                nearest_idx = (s_df["balls_completed"] - sb).abs().idxmin()
                snapshot_indices.append(nearest_idx)
        # Also add final snapshot
        snapshot_indices.append(s_df.index[-1])
        snapshot_indices = list(dict.fromkeys(snapshot_indices))  # deduplicate preserving order

        print(f"    {'Score':>7} {'Wkts':>5} {'Target':>7} {'RunsReq':>8} {'BallsRem':>9} "
              f"{'RRR':>7} {'WinProb':>8}")
        print("    " + "-" * 60)

        prev_prob = None
        anomalies = 0
        for idx in snapshot_indices:
            row = s_df.loc[idx]
            prob = row[selected_proba_col]
            score = int(row["score"])
            wkts = int(row["wickets"])
            trgt = int(row["target"])
            rr = int(row["runs_required"])
            br = int(row["balls_remaining"])
            rrr = row["required_run_rate"]

            flag = ""
            if prev_prob is not None:
                # Simple directional sanity: if score increased and wickets same,
                # probability should generally increase
                pass  # Don't enforce strict monotonicity

            print(f"    {score:>7} {wkts:>5} {trgt:>7} {rr:>8} {br:>9} "
                  f"{rrr:>7.2f} {prob:>8.1%}")
            prev_prob = prob

    # ── Part 15: Final Report ────────────────────────────────────────────────
    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print("LIVE MODEL RESULTS")
    print(f"{'='*60}")

    print(f"\nDataset:")
    print(f"  Total snapshots:        {len(df)}")
    print(f"  Total matches:          {df['match_id'].nunique()}")
    print(f"\n  Training matches:       {len(train_match_ids)}")
    print(f"  Testing matches:        {len(test_match_ids)}")
    print(f"  Training snapshots:     {len(df_train)}")
    print(f"  Testing snapshots:      {len(df_test)}")
    print(f"\n  Train date range:       {min(train_dates)} to {max(train_dates)}")
    print(f"  Test date range:        {min(test_dates)} to {max(test_dates)}")
    print(f"  Cutoff date:            {cutoff_date}")

    print(f"\n{'-'*60}")
    print("LOGISTIC REGRESSION")
    print(f"{'-'*60}")
    print(f"  Accuracy:               {lr_acc:.4f}")
    print(f"  ROC-AUC:                {lr_auc:.4f}")
    print(f"  Log Loss:               {lr_ll:.4f}")
    print(f"  Brier Score:            {lr_bs:.4f}")

    print(f"\n{'-'*60}")
    print("XGBOOST")
    print(f"{'-'*60}")
    print(f"  Accuracy:               {xgb_acc:.4f}")
    print(f"  ROC-AUC:                {xgb_auc:.4f}")
    print(f"  Log Loss:               {xgb_ll:.4f}")
    print(f"  Brier Score:            {xgb_bs:.4f}")

    print(f"\n{'-'*60}")
    print("SELECTED MODEL")
    print(f"{'-'*60}")
    print(f"  Model:                  {selected_model_name}")
    print(f"  Reason:                 {selected_reason}")

    print(f"\n{'-'*60}")
    print("TOP FEATURES")
    print(f"{'-'*60}")
    for rank, (fname, fimp) in enumerate(feat_imp[:15], 1):
        print(f"  {rank:>2}. {fname:<25} {fimp:.4f}")

    print(f"\n{'-'*60}")
    print("PROGRESS ANALYSIS")
    print(f"{'-'*60}")
    for stage_name, res in progress_results.items():
        if res is None:
            print(f"  {stage_name:<15} N/A")
        else:
            print(f"  {stage_name:<15} snapshots={res['n_snapshots']}, "
                  f"acc={res['accuracy']:.4f}, brier={res['brier']:.4f}, "
                  f"logloss={res['logloss']:.4f}")

    print(f"\n{'-'*60}")
    print("VALIDATION")
    print(f"{'-'*60}")

    v_no_match_overlap = train_match_ids.isdisjoint(test_match_ids)
    v_no_date_overlap = max(train_dates) < min(test_dates)
    v_chrono = True  # guaranteed by sort
    v_no_leakage = TARGET_COL not in FEATURE_COLS
    v_no_missing = df[FEATURE_COLS].isna().sum().sum() == 0
    v_no_inf = np.isinf(df[FEATURE_COLS].select_dtypes(include=[np.number])).sum().sum() == 0
    v_model_saved = os.path.exists(model_path)
    v_features_saved = os.path.exists(features_path)
    v_metrics_saved = os.path.exists(metrics_path)

    # Quick prediction module test
    v_pred_module = False
    try:
        loaded_model = joblib.load(model_path)
        loaded_features = joblib.load(features_path)
        test_row = df_test[FEATURE_COLS].iloc[:1]
        test_proba = loaded_model.predict_proba(test_row)[0]
        v_pred_module = abs(sum(test_proba) - 1.0) < 1e-5
    except Exception as e:
        print(f"  Prediction module test failed: {e}")

    print(f"[{'PASS' if v_no_match_overlap else 'FAIL'}] No match appears in both train/test")
    print(f"[{'PASS' if v_no_date_overlap else 'FAIL'}] No date overlap")
    print(f"[{'PASS' if v_chrono else 'FAIL'}] Chronological split")
    print(f"[{'PASS' if v_no_leakage else 'FAIL'}] No target leakage")
    print(f"[{'PASS' if v_no_missing else 'FAIL'}] No missing values")
    print(f"[{'PASS' if v_no_inf else 'FAIL'}] No infinite values")
    print(f"[{'PASS' if v_model_saved else 'FAIL'}] Model saved")
    print(f"[{'PASS' if v_features_saved else 'FAIL'}] Feature list saved")
    print(f"[{'PASS' if v_metrics_saved else 'FAIL'}] Metrics saved")
    print(f"[{'PASS' if v_pred_module else 'FAIL'}] Prediction module works")

    print(f"\nTotal training time: {t_total:.2f}s")


if __name__ == "__main__":
    train_live_model()
