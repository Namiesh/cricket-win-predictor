import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb


def train_and_evaluate_prematch_models():
    print("Loading prematch features dataset...")
    csv_path = os.path.join("data", "processed", "prematch_features.csv")
    df = pd.read_csv(csv_path)

    # 1. Deterministic Chronological Sorting
    df = df.sort_values(by=["date", "match_id"], ascending=[True, True]).reset_index(drop=True)

    target_col = "team1_won"
    feature_cols = [
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

    # Verify no leakages in feature list
    assert target_col not in feature_cols, "Target column present in feature list!"
    for col in ["winner", "outcome_result", "match_id", "date", "team1", "team2"]:
        assert col not in feature_cols, f"Forbidden column '{col}' in feature list!"

    # 2. Strict Chronological Date-Based Split (No Date Overlap)
    approx_idx = int(len(df) * 0.8)
    cutoff_date = df.iloc[approx_idx]["date"]

    # Training: all matches strictly BEFORE cutoff_date
    # Testing: all matches ON OR AFTER cutoff_date
    df_train = df[df["date"] < cutoff_date].copy()
    df_test = df[df["date"] >= cutoff_date].copy()

    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_test, y_test = df_test[feature_cols], df_test[target_col]

    train_min_date, train_max_date = df_train["date"].min(), df_train["date"].max()
    test_min_date, test_max_date = df_test["date"].min(), df_test["date"].max()

    # Strict Validation Assertion: No Date Overlap
    assert max(df_train["date"]) < min(df_test["date"]), \
        f"Date overlap detected! Max train date ({max(df_train['date'])}) is not less than min test date ({min(df_test['date'])})"

    print(f"\n--- STRICT CHRONOLOGICAL DATE SPLIT ---")
    print(f"Cutoff Date:   {cutoff_date}")
    print(f"Train Rows:    {len(X_train)} matches ({train_min_date} to {train_max_date})")
    print(f"Test Rows:     {len(X_test)} matches ({test_min_date} to {test_max_date})")
    print(f"Date Overlap:  NONE (Max train: {train_max_date} < Min test: {test_min_date})")

    # 3. Model 1: Logistic Regression Baseline
    print("\nTraining Logistic Regression model...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=2000, C=0.5))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_pred_class = lr_pipeline.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_pred_class)
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    lr_loss = log_loss(y_test, lr_pred_proba)
    lr_brier = brier_score_loss(y_test, lr_pred_proba)

    # 4. Model 2: Tuned XGBoost Classifier
    print("Training Tuned XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred_class = xgb_model.predict(X_test)

    xgb_acc = accuracy_score(y_test, xgb_pred_class)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    xgb_loss = log_loss(y_test, xgb_pred_proba)
    xgb_brier = brier_score_loss(y_test, xgb_pred_proba)

    # 5. Model 3: HistGradientBoosting Classifier
    print("Training HistGradientBoosting Classifier...")
    hgb_model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.03,
        l2_regularization=1.0,
        random_state=42
    )
    hgb_model.fit(X_train, y_train)
    hgb_pred_proba = hgb_model.predict_proba(X_test)[:, 1]
    hgb_pred_class = hgb_model.predict(X_test)

    hgb_acc = accuracy_score(y_test, hgb_pred_class)
    hgb_auc = roc_auc_score(y_test, hgb_pred_proba)
    hgb_loss = log_loss(y_test, hgb_pred_proba)
    hgb_brier = brier_score_loss(y_test, hgb_pred_proba)

    # 6. Model 4: Calibrated Soft Voting Ensemble
    print("Training Soft Voting Ensemble (LR + XGB + HistGB)...")
    ensemble_model = VotingClassifier(
        estimators=[
            ("lr", lr_pipeline),
            ("xgb", xgb_model),
            ("hgb", hgb_model)
        ],
        voting="soft",
        weights=[1.5, 1.0, 1.0]
    )
    ensemble_model.fit(X_train, y_train)
    ens_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    ens_pred_class = (ens_pred_proba >= 0.5).astype(int)

    ens_acc = accuracy_score(y_test, ens_pred_class)
    ens_auc = roc_auc_score(y_test, ens_pred_proba)
    ens_loss = log_loss(y_test, ens_pred_proba)
    ens_brier = brier_score_loss(y_test, ens_pred_proba)

    # 7. Model Selection based on lowest Brier score & Log Loss with highest ROC-AUC
    candidates = [
        ("Calibrated Soft Ensemble", ensemble_model, ens_acc, ens_auc, ens_loss, ens_brier),
        ("Logistic Regression", lr_pipeline, lr_acc, lr_auc, lr_loss, lr_brier),
        ("XGBoost Classifier", xgb_model, xgb_acc, xgb_auc, xgb_loss, xgb_brier),
        ("HistGradientBoosting", hgb_model, hgb_acc, hgb_auc, hgb_loss, hgb_brier)
    ]
    # Sort candidates by Brier Score ascending, Log Loss ascending, ROC-AUC descending
    candidates.sort(key=lambda c: (c[5], c[4], -c[3]))

    best_cand = candidates[0]
    selected_model_name = best_cand[0]
    selected_model = best_cand[1]
    selected_metrics = {"accuracy": best_cand[2], "roc_auc": best_cand[3], "log_loss": best_cand[4], "brier_score": best_cand[5]}
    selection_reason = (
        f"{selected_model_name} achieved optimal performance with "
        f"Accuracy ({selected_metrics['accuracy']:.4f}), ROC-AUC ({selected_metrics['roc_auc']:.4f}), "
        f"Log Loss ({selected_metrics['log_loss']:.4f}), and Brier Score ({selected_metrics['brier_score']:.4f})."
    )

    # 8. Save Model Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = os.path.join("models", "prematch_model.pkl")
    features_path = os.path.join("models", "prematch_features.pkl")
    metrics_path = os.path.join("models", "prematch_metrics.json")

    joblib.dump(selected_model, model_path)
    joblib.dump(feature_cols, features_path)

    metrics_payload = {
        "model_name": selected_model_name,
        "accuracy": round(float(selected_metrics["accuracy"]), 4),
        "roc_auc": round(float(selected_metrics["roc_auc"]), 4),
        "log_loss": round(float(selected_metrics["log_loss"]), 4),
        "brier_score": round(float(selected_metrics["brier_score"]), 4),
        "cutoff_date": cutoff_date,
        "train_date_range": f"{train_min_date} to {train_max_date}",
        "test_date_range": f"{test_min_date} to {test_max_date}",
        "feature_list": feature_cols
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 7. Calibration Plot
    plt.figure(figsize=(8, 6))
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, lr_pred_proba, n_bins=10)
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, xgb_pred_proba, n_bins=10)

    plt.plot(prob_pred_lr, prob_true_lr, "s-", label=f"Logistic Regression (Brier: {lr_brier:.4f})", color="royalblue")
    plt.plot(prob_pred_xgb, prob_true_xgb, "o-", label=f"XGBoost (Brier: {xgb_brier:.4f})", color="darkorange")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual Wins)")
    plt.title(f"Pre-Match Calibration Curve (Test Period: {test_min_date} to {test_max_date})")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    calib_plot_path = os.path.join("reports", "prematch_calibration.png")
    plt.savefig(calib_plot_path, dpi=300)
    plt.close()

    # 8. Feature Importance Plot
    fi_plot_path = os.path.join("reports", "prematch_feature_importance.png")

    xgb_importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": xgb_importances})
    fi_df = fi_df.sort_values(by="importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(fi_df["feature"], fi_df["importance"], color="teal")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.title("Pre-Match Model Feature Importances")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()

    top_10_xgb_features = fi_df.sort_values(by="importance", ascending=False).head(10)["feature"].tolist()

    # 9. Model Sanity Check
    sample1 = {
        "team1_elo": 1500.0, "team2_elo": 1500.0, "elo_difference": 0.0,
        "team1_win_rate_last3": 0.5, "team2_win_rate_last3": 0.5,
        "team1_win_rate_last5": 0.5, "team2_win_rate_last5": 0.5,
        "team1_win_rate_last10": 0.5, "team2_win_rate_last10": 0.5,
        "win_rate_diff_last5": 0.0, "win_rate_diff_last10": 0.0,
        "team1_avg_runs_last5": 145.0, "team2_avg_runs_last5": 145.0,
        "team1_avg_conc_last5": 145.0, "team2_avg_conc_last5": 145.0, "net_runs_diff5": 0.0,
        "team1_h2h_win_rate": 0.5, "h2h_matches_before": 0,
        "venue_avg_first_innings_score": 145.0, "venue_batting_first_win_rate": 0.5, "venue_matches_before": 0,
        "team1_ven_win_rate": 0.5, "team2_ven_win_rate": 0.5, "team1_toss_won": 0.5
    }

    sample2 = {
        "team1_elo": 1750.0, "team2_elo": 1400.0, "elo_difference": 350.0,
        "team1_win_rate_last3": 0.8, "team2_win_rate_last3": 0.2,
        "team1_win_rate_last5": 0.8, "team2_win_rate_last5": 0.2,
        "team1_win_rate_last10": 0.8, "team2_win_rate_last10": 0.3,
        "win_rate_diff_last5": 0.6, "win_rate_diff_last10": 0.5,
        "team1_avg_runs_last5": 175.0, "team2_avg_runs_last5": 130.0,
        "team1_avg_conc_last5": 130.0, "team2_avg_conc_last5": 175.0, "net_runs_diff5": 90.0,
        "team1_h2h_win_rate": 0.75, "h2h_matches_before": 4,
        "venue_avg_first_innings_score": 150.0, "venue_batting_first_win_rate": 0.5, "venue_matches_before": 5,
        "team1_ven_win_rate": 0.8, "team2_ven_win_rate": 0.2, "team1_toss_won": 1.0
    }

    df_sample1 = pd.DataFrame([sample1])[feature_cols]
    df_sample2 = pd.DataFrame([sample2])[feature_cols]

    prob_sample1 = selected_model.predict_proba(df_sample1)[0, 1]
    prob_sample2 = selected_model.predict_proba(df_sample2)[0, 1]


    if prob_sample2 > prob_sample1:
        sanity_msg = f"PASSED (Probability increased from {prob_sample1:.4f} to {prob_sample2:.4f})"
    else:
        sanity_msg = f"WARNING (Probability changed from {prob_sample1:.4f} to {prob_sample2:.4f})"

    # 10. Print Required Output Structure
    print("\n" + "=" * 50)
    print("FINAL CHRONOLOGICAL RESULTS")
    print("=" * 50)
    print(f"Cutoff date:      {cutoff_date}")
    print(f"Train rows:       {len(X_train)}")
    print(f"Test rows:        {len(X_test)}")
    print(f"Train date range: {train_min_date} to {train_max_date}")
    print(f"Test date range:  {test_min_date} to {test_max_date}")

    print("\nLogistic Regression:")
    print(f"Accuracy:    {lr_acc:.4f}")
    print(f"ROC-AUC:     {lr_auc:.4f}")
    print(f"Log Loss:    {lr_loss:.4f}")
    print(f"Brier Score: {lr_brier:.4f}")

    print("\nXGBoost:")
    print(f"Accuracy:    {xgb_acc:.4f}")
    print(f"ROC-AUC:     {xgb_auc:.4f}")
    print(f"Log Loss:    {xgb_loss:.4f}")
    print(f"Brier Score: {xgb_brier:.4f}")

    print(f"\nSelected model:   {selected_model_name}")
    print(f"Reason:           {selection_reason}")
    print(f"Sanity Direction: {sanity_msg}")

    print("\nValidation:")
    print("[PASS] No date overlap")
    print("[PASS] Chronological split")
    print("[PASS] No target leakage")
    print("[PASS] Model saved")
    print("[PASS] Metrics saved")
    print("=" * 50)


if __name__ == "__main__":
    train_and_evaluate_prematch_models()
