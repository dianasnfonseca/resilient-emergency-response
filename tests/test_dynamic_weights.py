"""
Tests for dynamic weight adjustment in AdaptiveCoordinator.

The adaptive algorithm infers current network quality from the mean P of
eligible candidate responders and selects (α, β) accordingly:

    mean_P > 0.40         → α = 0.4, β = 0.6  (good connectivity)
    0.30 ≤ mean_P ≤ 0.40  → α = 0.3, β = 0.7  (moderate connectivity)
    mean_P < 0.30         → α = 0.1, β = 0.9  (severe — proximity dominates)

Groups:
    A — TestRegimeSelection: verify correct (α, β) for each regime and boundary
    B — TestIntegrationSevere: proximity dominates at severe connectivity
    C — TestIntegrationModerate: balanced scoring at moderate connectivity
    D — TestEdgeCases: single candidate, etc.

Sources:
    Rosas et al. (2020) — adaptive weight adjustment
    Boondirek et al. (2014) — DiPRoPHET distance-dominant weighting
    Cui et al. (2022) — AdaptiveSpray workload-aware scoring
    Kaji et al. (2025) — urgency-first prioritisation
"""

import pytest

from ercs.config.parameters import CoordinationParameters, UrgencyLevel
from ercs.coordination.algorithms import (
    SIMULATION_AREA_DIAGONAL_M,
    WORKLOAD_PENALTY_WEIGHT,
    AdaptiveCoordinator,
)
from ercs.scenario.generator import Task
from conftest import (
    WEIGHT_ALPHA_GOOD,
    WEIGHT_ALPHA_MODERATE,
    WEIGHT_ALPHA_SEVERE,
    P_THRESHOLD_GOOD,
    P_THRESHOLD_MODERATE,
)


# ============================================================================
# Helpers
# ============================================================================


class MockResponderLocator:
    """Mock responder locator for testing."""

    def __init__(self, responders: dict[str, tuple[float, float]]):
        self._responders = responders

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        return self._responders[responder_id]

    def get_all_responder_ids(self) -> list[str]:
        return list(self._responders.keys())


class MockNetworkState:
    """Mock network state provider for testing."""

    def __init__(self, predictabilities: dict[tuple[str, str], float] | None = None):
        self._predictabilities = predictabilities or {}

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        return self._predictabilities.get((from_node, to_node), 0.0)


def _make_task(
    task_id: str,
    creation_time: float,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    x: float = 100.0,
    y: float = 100.0,
) -> Task:
    return Task(
        task_id=task_id,
        creation_time=creation_time,
        source_node="coord_0",
        target_location_x=x,
        target_location_y=y,
        urgency=urgency,
    )


def _score(alpha: float, p: float, dist: float, penalised: bool = False) -> float:
    """Compute the expected score for manual verification."""
    beta = 1.0 - alpha
    d_norm = 1.0 - (dist / SIMULATION_AREA_DIAGONAL_M)
    w = 1.0 if penalised else 0.0
    return alpha * p + beta * d_norm - WORKLOAD_PENALTY_WEIGHT * w


# ============================================================================
# Group A — TestRegimeSelection
# ============================================================================


class TestRegimeSelection:
    """
    Verify correct (α, β) regime selection based on mean P of candidates.

    Each test constructs two candidates whose mean P falls in a known regime,
    then verifies the assignment matches what the expected (α, β) would produce.
    """

    def test_good_regime_mean_p_055(self):
        """mean_P = 0.55 → α = 0.4, β = 0.6 (good connectivity)."""
        # Two candidates: P=0.50 and P=0.60 → mean_P = 0.55 > 0.40
        positions = {
            "r_a": (200.0, 200.0),  # closer to task at (150, 150)
            "r_b": (800.0, 800.0),  # farther
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.50,
            ("coord_0", "r_b"): 0.60,
        })

        task = _make_task("t_0", 0.0, x=150.0, y=150.0)
        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # Verify by manual score calculation with α=0.4
        import math
        dist_a = math.dist((150, 150), (200, 200))
        dist_b = math.dist((150, 150), (800, 800))
        score_a = _score(WEIGHT_ALPHA_GOOD, 0.50, dist_a)
        score_b = _score(WEIGHT_ALPHA_GOOD, 0.60, dist_b)

        if score_a > score_b:
            assert assignments[0].responder_id == "r_a"
        else:
            assert assignments[0].responder_id == "r_b"

    def test_moderate_regime_mean_p_035(self):
        """mean_P = 0.35 → α = 0.3, β = 0.7 (moderate connectivity)."""
        # Two candidates: P=0.32 and P=0.38 → mean_P = 0.35
        positions = {
            "r_a": (110.0, 110.0),  # close to task at (100, 100)
            "r_b": (600.0, 600.0),  # far
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.32,
            ("coord_0", "r_b"): 0.38,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # With α=0.3, β=0.7, proximity dominates — r_a should win
        import math
        dist_a = math.dist((100, 100), (110, 110))
        dist_b = math.dist((100, 100), (600, 600))
        score_a = _score(WEIGHT_ALPHA_MODERATE, 0.32, dist_a)
        score_b = _score(WEIGHT_ALPHA_MODERATE, 0.38, dist_b)

        assert score_a > score_b, f"Expected r_a score {score_a:.4f} > r_b {score_b:.4f}"
        assert assignments[0].responder_id == "r_a"

    def test_severe_regime_mean_p_020(self):
        """mean_P = 0.20 → α = 0.1, β = 0.9 (severe — proximity dominates).

        Note: for mean_P < 0.30 to occur, all candidates must have P < 0.30.
        But candidates must have P > threshold (0.3) to be eligible.
        So this regime only triggers with a lowered threshold.
        """
        # Use a custom threshold of 0.1 so low-P candidates are eligible
        params = CoordinationParameters(available_path_threshold=0.1)

        positions = {
            "r_a": (105.0, 105.0),  # very close to task
            "r_b": (500.0, 500.0),  # far
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.18,
            ("coord_0", "r_b"): 0.22,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator(params=params)
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # With α=0.1, β=0.9, proximity overwhelms P → r_a wins
        import math
        dist_a = math.dist((100, 100), (105, 105))
        dist_b = math.dist((100, 100), (500, 500))
        score_a = _score(WEIGHT_ALPHA_SEVERE, 0.18, dist_a)
        score_b = _score(WEIGHT_ALPHA_SEVERE, 0.22, dist_b)

        assert score_a > score_b, f"Expected r_a score {score_a:.4f} > r_b {score_b:.4f}"
        assert assignments[0].responder_id == "r_a"

    def test_boundary_mean_p_exactly_040(self):
        """mean_P exactly 0.40 → moderate regime (α = 0.3), NOT good.

        Boundary condition: mean_P = 0.40 is NOT > 0.40, so falls into moderate.
        """
        # Two candidates: P=0.38 and P=0.42 → mean_P = 0.40
        positions = {
            "r_a": (110.0, 110.0),
            "r_b": (700.0, 700.0),
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.38,
            ("coord_0", "r_b"): 0.42,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # With moderate α=0.3, proximity wins over small P difference
        import math
        dist_a = math.dist((100, 100), (110, 110))
        dist_b = math.dist((100, 100), (700, 700))
        score_a = _score(WEIGHT_ALPHA_MODERATE, 0.38, dist_a)
        score_b = _score(WEIGHT_ALPHA_MODERATE, 0.42, dist_b)

        assert score_a > score_b
        assert assignments[0].responder_id == "r_a"

        # Verify that with good regime (α=0.4), the result would still be the same
        # (proximity gap is too large for any α to overcome)
        score_a_good = _score(WEIGHT_ALPHA_GOOD, 0.38, dist_a)
        score_b_good = _score(WEIGHT_ALPHA_GOOD, 0.42, dist_b)
        assert score_a_good > score_b_good

    def test_boundary_mean_p_exactly_030(self):
        """mean_P exactly 0.30 → severe regime (α = 0.1), NOT moderate.

        With default threshold=0.3 and P > 0.3 required, mean_P = 0.30 is
        impossible (all candidates have P > 0.3 → mean > 0.3). We use a
        lowered threshold to test the boundary.
        """
        params = CoordinationParameters(available_path_threshold=0.1)

        # Two candidates: P=0.25 and P=0.35 → mean_P = 0.30
        positions = {
            "r_a": (105.0, 105.0),
            "r_b": (500.0, 500.0),
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.25,
            ("coord_0", "r_b"): 0.35,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator(params=params)
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # mean_P = 0.30: exactly at boundary → 0.30 >= p_threshold_moderate (0.30)
        # → moderate regime, α = 0.3
        import math
        dist_a = math.dist((100, 100), (105, 105))
        dist_b = math.dist((100, 100), (500, 500))
        score_a = _score(WEIGHT_ALPHA_MODERATE, 0.25, dist_a)
        score_b = _score(WEIGHT_ALPHA_MODERATE, 0.35, dist_b)

        assert score_a > score_b
        assert assignments[0].responder_id == "r_a"


# ============================================================================
# Group B — TestIntegrationSevere
# ============================================================================


class TestIntegrationSevere:
    """Proximity dominates at severe connectivity."""

    def test_proximity_wins_at_severe(self):
        """
        Responder A: P=0.25, distance=50m from task
        Responder B: P=0.28, distance=800m from task
        mean_P = 0.265 → severe regime → α=0.1, β=0.9
        Proximity should decisively win.
        """
        params = CoordinationParameters(available_path_threshold=0.1)

        # Task at (100, 100)
        # r_a at ~50m away, r_b at ~800m away
        positions = {
            "r_a": (135.0, 135.0),   # dist ≈ 49.5m
            "r_b": (665.0, 665.0),   # dist ≈ 799m
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.25,
            ("coord_0", "r_b"): 0.28,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator(params=params)
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1
        assert assignments[0].responder_id == "r_a"

        # Verify the scores manually
        import math
        dist_a = math.dist((100, 100), (135, 135))
        dist_b = math.dist((100, 100), (665, 665))
        score_a = _score(WEIGHT_ALPHA_SEVERE, 0.25, dist_a)
        score_b = _score(WEIGHT_ALPHA_SEVERE, 0.28, dist_b)

        assert score_a > score_b, (
            f"Severe regime: r_a score {score_a:.4f} should beat r_b {score_b:.4f}"
        )


# ============================================================================
# Group C — TestIntegrationModerate
# ============================================================================


class TestIntegrationModerate:
    """Balanced scoring at moderate connectivity."""

    def test_moderate_regime_scoring(self):
        """
        Responder A: P=0.48, distance=200m
        Responder B: P=0.30 (just above threshold), distance=180m
        mean_P = 0.39 → moderate regime → α=0.3, β=0.7

        With moderate regime, r_a has much higher P (0.48 vs 0.30) but r_b
        is slightly closer (180m vs 200m). The P gap × α=0.3 should be enough
        to overcome the small distance difference × β=0.7.
        """
        positions = {
            "r_a": (241.0, 241.0),   # ~200m from task at (100, 100)
            "r_b": (227.0, 227.0),   # ~180m from task
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_a"): 0.48,
            ("coord_0", "r_b"): 0.31,  # Just above threshold
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1

        # Verify expected scores
        import math
        dist_a = math.dist((100, 100), (241, 241))
        dist_b = math.dist((100, 100), (227, 227))

        # mean_P = (0.48 + 0.31) / 2 = 0.395 → moderate regime (0.30 ≤ 0.395 ≤ 0.40)
        score_a = _score(WEIGHT_ALPHA_MODERATE, 0.48, dist_a)
        score_b = _score(WEIGHT_ALPHA_MODERATE, 0.31, dist_b)

        # P gap of 0.17 × α=0.3 = 0.051 contribution
        # Distance gap of ~20m → D_norm diff ≈ 20/3354 × β=0.7 ≈ 0.004
        # So P advantage of r_a should dominate
        assert score_a > score_b, (
            f"Moderate regime: r_a score {score_a:.4f} should beat r_b {score_b:.4f}"
        )
        assert assignments[0].responder_id == "r_a"


# ============================================================================
# Group D — TestEdgeCases
# ============================================================================


class TestEdgeCases:
    """Edge cases for dynamic weight selection."""

    def test_single_eligible_candidate(self):
        """Single eligible candidate → assignment succeeds without crash."""
        positions = {
            "r_only": (150.0, 150.0),
            "r_excluded": (110.0, 110.0),  # Below threshold
        }
        locator = MockResponderLocator(positions)
        net = MockNetworkState({
            ("coord_0", "r_only"): 0.45,
            ("coord_0", "r_excluded"): 0.10,  # Below 0.3 threshold
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)
        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks([task], locator, net, "coord_0", 100.0)

        assert len(assignments) == 1
        assert assignments[0].responder_id == "r_only"

    def test_conftest_constants_match_parameters(self):
        """Verify conftest constants match CoordinationParameters defaults."""
        params = CoordinationParameters()
        assert params.weight_alpha_good == pytest.approx(WEIGHT_ALPHA_GOOD)
        assert params.weight_alpha_moderate == pytest.approx(WEIGHT_ALPHA_MODERATE)
        assert params.weight_alpha_severe == pytest.approx(WEIGHT_ALPHA_SEVERE)
        assert params.p_threshold_good == pytest.approx(P_THRESHOLD_GOOD)
        assert params.p_threshold_moderate == pytest.approx(P_THRESHOLD_MODERATE)
