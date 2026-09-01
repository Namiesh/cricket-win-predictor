class EloCalculator:
    """
    Chronological Elo Rating Calculator for T20 Teams.
    
    Formula:
      Expected Score P(A) = 1 / (1 + 10 ** ((Elo_B - Elo_A) / 400))
      New Rating = Old Rating + K * (Actual Result - Expected Score)
    """

    def __init__(self, k_factor: float = 32.0, initial_elo: float = 1500.0):
        self.k_factor = float(k_factor)
        self.initial_elo = float(initial_elo)
        self.ratings = {}

    def get_rating(self, team: str) -> float:
        """Return the current rating for a team, defaulting to initial_elo."""
        return self.ratings.get(team, self.initial_elo)

    def calculate_expected_score(self, elo_a: float, elo_b: float) -> float:
        """Compute expected win probability for Team A against Team B."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update_ratings(self, team_a: str, team_b: str, actual_a_won: float) -> tuple[float, float, float, float]:
        """
        Update ratings for Team A and Team B after a match.
        
        Args:
          team_a: Name of Team A
          team_b: Name of Team B
          actual_a_won: 1.0 if Team A won, 0.0 if Team B won
          
        Returns:
          Tuple of (pre_elo_a, pre_elo_b, post_elo_a, post_elo_b)
        """
        pre_elo_a = self.get_rating(team_a)
        pre_elo_b = self.get_rating(team_b)

        expected_a = self.calculate_expected_score(pre_elo_a, pre_elo_b)
        expected_b = 1.0 - expected_a

        actual_a = float(actual_a_won)
        actual_b = 1.0 - actual_a

        post_elo_a = pre_elo_a + self.k_factor * (actual_a - expected_a)
        post_elo_b = pre_elo_b + self.k_factor * (actual_b - expected_b)

        self.ratings[team_a] = post_elo_a
        self.ratings[team_b] = post_elo_b

        return pre_elo_a, pre_elo_b, post_elo_a, post_elo_b


def test_elo_logic():
    """Unit tests validating Elo mathematical properties."""
    elo = EloCalculator(k_factor=32.0, initial_elo=1500.0)

    # 1. Equal-rated teams have expected probability 0.5
    p_equal = elo.calculate_expected_score(1500.0, 1500.0)
    assert abs(p_equal - 0.5) < 1e-6, f"Equal teams expected 0.5, got {p_equal}"

    # 2. Stronger team has expected probability > 0.5
    p_stronger = elo.calculate_expected_score(1600.0, 1400.0)
    assert p_stronger > 0.5, f"Stronger team expected > 0.5, got {p_stronger}"
    assert p_stronger > 0.75, f"Expected ~0.76, got {p_stronger}"

    # 3. Upset causes larger rating movement than expected win
    # Case A: Expected win (1600 beats 1400)
    p_fav = elo.calculate_expected_score(1600.0, 1400.0)
    delta_expected = 32.0 * (1.0 - p_fav)

    # Case B: Upset (1400 beats 1600)
    p_underdog = elo.calculate_expected_score(1400.0, 1600.0)
    delta_upset = 32.0 * (1.0 - p_underdog)

    assert delta_upset > delta_expected, \
        f"Upset delta ({delta_upset:.2f}) should be larger than expected win delta ({delta_expected:.2f})"

    print("EloCalculator unit tests passed successfully!")


if __name__ == "__main__":
    test_elo_logic()
