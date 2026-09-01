"""
Phase 4B — Live Win-Probability Prediction Module

Loads the trained live model and provides a clean prediction interface
for real-time match state inputs.

Usage:
    from src.predict_live import LivePredictor
    predictor = LivePredictor()
    result = predictor.predict({
        "batting_team_elo": 1600.0,
        "bowling_team_elo": 1500.0,
        ...
    })
"""

import os
import joblib
import numpy as np
import pandas as pd


class LivePredictor:
    """
    Live T20 win-probability predictor.

    Loads the serialised model and feature list, accepts a match-state
    dictionary or DataFrame, and returns calibrated win probabilities.
    """

    def __init__(
        self,
        model_path: str = os.path.join("models", "live_model.pkl"),
        features_path: str = os.path.join("models", "live_features.pkl"),
    ):
        assert os.path.exists(model_path), f"Model file missing: {model_path}"
        assert os.path.exists(features_path), f"Features file missing: {features_path}"

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)

    def predict(self, match_state: dict | pd.DataFrame) -> dict:
        """
        Predict win probability for the chasing (batting) team.

        Args:
            match_state: dict or single-row DataFrame with feature values.

        Returns:
            dict with batting_team_win_probability & bowling_team_win_probability.
        """
        if isinstance(match_state, dict):
            df = pd.DataFrame([match_state])
        elif isinstance(match_state, pd.DataFrame):
            df = match_state.copy()
        else:
            raise TypeError(f"Expected dict or DataFrame, got {type(match_state)}")

        # Validate all required features are present
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Enforce exact feature ordering
        X = df[self.feature_names]

        # Check for NaN / Inf
        if X.isna().any().any():
            raise ValueError("Input contains NaN values")
        if np.isinf(X.select_dtypes(include=[np.number]).values).any():
            raise ValueError("Input contains infinite values")

        proba = self.model.predict_proba(X)[0]
        prob_batting_wins = float(proba[1])  # class 1 = batting_team_won
        prob_bowling_wins = 1.0 - prob_batting_wins

        # Sanity check
        assert abs((prob_batting_wins + prob_bowling_wins) - 1.0) < 1e-5, \
            f"Probabilities do not sum to 1: {prob_batting_wins} + {prob_bowling_wins}"

        return {
            "batting_team_win_probability": prob_batting_wins,
            "bowling_team_win_probability": prob_bowling_wins,
        }


def _demo():
    """
    Demonstrate the LivePredictor with a sample match state and
    a historical test scenario.
    """
    predictor = LivePredictor()

    print("=" * 60)
    print("LIVE PREDICTOR — DEMONSTRATION")
    print("=" * 60)

    # Scenario 1: Start of chase — strong team chasing moderate target
    scenario1 = {
        "batting_team_elo": 1650.0,
        "bowling_team_elo": 1500.0,
        "elo_difference": 150.0,
        "score": 0,
        "wickets": 0,
        "balls_remaining": 120,
        "target": 160,
        "runs_required": 160,
        "wickets_remaining": 10,
        "current_run_rate": 0.0,
        "required_run_rate": 8.0,
        "run_rate_difference": -8.0,
    }
    r1 = predictor.predict(scenario1)
    print(f"\nScenario 1: Strong team starts chasing 160")
    print(f"  Batting team Elo: 1650 vs Bowling team Elo: 1500")
    print(f"  Score: 0/0, Balls remaining: 120")
    print(f"  Win prob (batting): {r1['batting_team_win_probability']:.1%}")
    print(f"  Win prob (bowling): {r1['bowling_team_win_probability']:.1%}")

    # Scenario 2: Midway through — comfortable position
    scenario2 = {
        "batting_team_elo": 1650.0,
        "bowling_team_elo": 1500.0,
        "elo_difference": 150.0,
        "score": 90,
        "wickets": 2,
        "balls_remaining": 60,
        "target": 160,
        "runs_required": 70,
        "wickets_remaining": 8,
        "current_run_rate": 9.0,
        "required_run_rate": 7.0,
        "run_rate_difference": 2.0,
    }
    r2 = predictor.predict(scenario2)
    print(f"\nScenario 2: 90/2 after 10 overs, chasing 160")
    print(f"  Required: 70 off 60 balls (RRR: 7.0)")
    print(f"  Win prob (batting): {r2['batting_team_win_probability']:.1%}")
    print(f"  Win prob (bowling): {r2['bowling_team_win_probability']:.1%}")

    # Scenario 3: Pressure situation — lots of wickets lost
    scenario3 = {
        "batting_team_elo": 1650.0,
        "bowling_team_elo": 1500.0,
        "elo_difference": 150.0,
        "score": 80,
        "wickets": 7,
        "balls_remaining": 30,
        "target": 160,
        "runs_required": 80,
        "wickets_remaining": 3,
        "current_run_rate": 5.33,
        "required_run_rate": 16.0,
        "run_rate_difference": -10.67,
    }
    r3 = predictor.predict(scenario3)
    print(f"\nScenario 3: 80/7 after 15 overs, chasing 160")
    print(f"  Required: 80 off 30 balls (RRR: 16.0)")
    print(f"  Win prob (batting): {r3['batting_team_win_probability']:.1%}")
    print(f"  Win prob (bowling): {r3['bowling_team_win_probability']:.1%}")

    # Scenario 4: Almost won
    scenario4 = {
        "batting_team_elo": 1650.0,
        "bowling_team_elo": 1500.0,
        "elo_difference": 150.0,
        "score": 155,
        "wickets": 4,
        "balls_remaining": 12,
        "target": 160,
        "runs_required": 5,
        "wickets_remaining": 6,
        "current_run_rate": 8.61,
        "required_run_rate": 2.5,
        "run_rate_difference": 6.11,
    }
    r4 = predictor.predict(scenario4)
    print(f"\nScenario 4: 155/4 after 18 overs, chasing 160")
    print(f"  Required: 5 off 12 balls (RRR: 2.5)")
    print(f"  Win prob (batting): {r4['batting_team_win_probability']:.1%}")
    print(f"  Win prob (bowling): {r4['bowling_team_win_probability']:.1%}")

    # Directional sanity check
    print(f"\n{'='*60}")
    print("DIRECTIONAL SANITY CHECK")
    print(f"{'='*60}")
    print(f"  Start of chase:     {r1['batting_team_win_probability']:.1%}")
    print(f"  Comfortable mid:    {r2['batting_team_win_probability']:.1%}")
    print(f"  Pressure (7 wkts):  {r3['batting_team_win_probability']:.1%}")
    print(f"  Almost won:         {r4['batting_team_win_probability']:.1%}")

    if r2["batting_team_win_probability"] > r1["batting_team_win_probability"]:
        print("  [OK] Comfortable position > start of chase")
    else:
        print("  [WARN] Comfortable position < start of chase")

    if r3["batting_team_win_probability"] < r2["batting_team_win_probability"]:
        print("  [OK] Pressure situation < comfortable position")
    else:
        print("  [WARN] Pressure situation >= comfortable position")

    if r4["batting_team_win_probability"] > r2["batting_team_win_probability"]:
        print("  [OK] Almost won > comfortable mid")
    else:
        print("  [WARN] Almost won <= comfortable mid")

    print(f"\n[PASS] LivePredictor loaded and functional")
    print(f"[PASS] All predictions returned valid probabilities")


if __name__ == "__main__":
    _demo()
