import os
import joblib
import pandas as pd
from typing import Dict, Union, Any


class PreMatchPredictor:
    """
    Inference class for Pre-Match Win Probability Prediction.
    
    Loads trained model and exact feature list to ensure correct feature ordering and column alignment.
    """

    def __init__(self, model_path: str = None, features_path: str = None):
        if model_path is None:
            model_path = os.path.join("models", "prematch_model.pkl")
        if features_path is None:
            features_path = os.path.join("models", "prematch_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first using src.train_prematch.")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}. Train the model first using src.train_prematch.")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)

    def predict_probability(self, features: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, float]:
        """
        Predict pre-match win probabilities for Team 1 and Team 2.
        
        Args:
          features: Dictionary or DataFrame containing pre-match feature values.
          
        Returns:
          Dict with keys:
            - team1_win_probability: float between 0 and 1
            - team2_win_probability: float between 0 and 1
            (Probabilities sum strictly to 1.0)
        """
        if isinstance(features, dict):
            df_input = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            df_input = features.copy()
        else:
            raise ValueError("Input features must be a dictionary or pandas DataFrame.")

        # Ensure all required features are present
        missing_features = [col for col in self.feature_names if col not in df_input.columns]
        if missing_features:
            raise KeyError(f"Missing required input features: {missing_features}")

        # Enforce exact feature column ordering
        df_input = df_input[self.feature_names]

        # Predict probability for Team 1 winning (class 1)
        proba_team1 = float(self.model.predict_proba(df_input)[0, 1])
        proba_team2 = 1.0 - proba_team1

        return {
            "team1_win_probability": round(proba_team1, 4),
            "team2_win_probability": round(proba_team2, 4)
        }


def test_predictor_standalone():
    """Unit test for PreMatchPredictor inference."""
    predictor = PreMatchPredictor()
    sample_input = {
        "team1_elo": 1650.0, "team2_elo": 1500.0, "elo_difference": 150.0,
        "team1_win_rate_last5": 0.8, "team2_win_rate_last5": 0.4,
        "team1_win_rate_last10": 0.7, "team2_win_rate_last10": 0.5,
        "team1_avg_runs_last5": 168.0, "team2_avg_runs_last5": 142.0,
        "team1_h2h_win_rate": 0.6, "h2h_matches_before": 5,
        "venue_avg_first_innings_score": 155.0, "venue_batting_first_win_rate": 0.52, "venue_matches_before": 12
    }

    result = predictor.predict_probability(sample_input)
    print("Standalone Predictor Test Result:")
    print(f"  Team 1 Win Prob: {result['team1_win_probability']}")
    print(f"  Team 2 Win Prob: {result['team2_win_probability']}")
    assert abs((result['team1_win_probability'] + result['team2_win_probability']) - 1.0) < 1e-4, \
        "Probabilities must sum to 1.0!"
    print("PreMatchPredictor standalone test PASSED!")


if __name__ == "__main__":
    test_predictor_standalone()
