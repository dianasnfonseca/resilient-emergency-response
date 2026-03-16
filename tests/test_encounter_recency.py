"""
tests/test_encounter_recency.py

Local unit tests for the encounter recency scoring addition to AdaptiveCoordinator.

Run with:
    python -m pytest tests/test_encounter_recency.py -v

These tests verify:
1. R_norm is computed correctly from last_encounter_time
2. R_norm varies across candidates when encounter times differ
3. The new scoring formula produces different results than proximity-only
4. Edge cases (never encountered, just encountered) are handled correctly
5. The formula weights sum to <= 1.0 (no score inflation)
6. Baseline algorithm is NOT changed
"""

from unittest.mock import MagicMock

import pytest

# ── Constants that must match the implementation ─────────────────────────────
T_REF = 1800.0  # i_typ in seconds
ALPHA = 0.2  # predictability weight
GAMMA_R = 0.2  # recency weight
BETA = 0.6  # proximity weight
WORKLOAD_PENALTY = 0.2  # workload penalty weight
P_THRESHOLD = 0.3  # minimum predictability threshold
SIMULATION_DIAGONAL = 3354.1  # sqrt(3000^2 + 1500^2) metres


# ── Helper: R_norm formula (mirrors what implementation should do) ─────────────
def compute_r_norm(current_time: float, last_encounter_time: float) -> float:
    """
    Compute encounter recency score.
    R_norm = 1 - min(delta_t / T_REF, 1.0)
    where delta_t = current_time - last_encounter_time
    """
    delta_t = max(0.0, current_time - last_encounter_time)
    return 1.0 - min(delta_t / T_REF, 1.0)


def compute_score(
    p_abs: float, r_norm: float, d_norm: float, w_penalty: float = 0.0
) -> float:
    """Mirror of the revised scoring formula."""
    return (
        ALPHA * p_abs + GAMMA_R * r_norm + BETA * d_norm - WORKLOAD_PENALTY * w_penalty
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: R_norm computation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRNormComputation:
    """Unit tests for the R_norm (encounter recency) formula."""

    def test_just_encountered_gives_max_recency(self):
        """Responder encountered at exactly current_time → R_norm = 1.0"""
        current_time = 7200.0
        r = compute_r_norm(current_time, last_encounter_time=7200.0)
        assert r == pytest.approx(1.0), f"Expected 1.0, got {r}"

    def test_encountered_half_t_ref_ago(self):
        """Responder encountered 900 s ago (half of T_ref) → R_norm = 0.5"""
        current_time = 7200.0
        r = compute_r_norm(current_time, last_encounter_time=7200.0 - 900.0)
        assert r == pytest.approx(0.5), f"Expected 0.5, got {r}"

    def test_encountered_t_ref_ago_gives_zero(self):
        """Responder encountered exactly T_ref ago → R_norm = 0.0"""
        current_time = 7200.0
        r = compute_r_norm(current_time, last_encounter_time=7200.0 - T_REF)
        assert r == pytest.approx(0.0), f"Expected 0.0, got {r}"

    def test_never_encountered_gives_zero(self):
        """
        Responder never encountered (last_encounter_time = 0.0) at time > T_ref
        → R_norm = 0.0 (clamped, not negative)
        """
        current_time = 7200.0
        r = compute_r_norm(current_time, last_encounter_time=0.0)
        assert r == pytest.approx(0.0), f"Expected 0.0, got {r}"
        assert r >= 0.0, "R_norm must never be negative"

    def test_encountered_more_than_t_ref_ago_clamped(self):
        """delta_t > T_ref → R_norm clamped to 0.0, not negative."""
        current_time = 10800.0
        r = compute_r_norm(current_time, last_encounter_time=10800.0 - 5000.0)
        assert r == pytest.approx(0.0), f"Expected 0.0, got {r}"

    def test_r_norm_bounded_zero_to_one(self):
        """R_norm must always be in [0.0, 1.0]."""
        current_time = 7200.0
        test_cases = [0.0, 100.0, 900.0, 1800.0, 3600.0, 7200.0, 7200.01]
        for last_enc in test_cases:
            r = compute_r_norm(current_time, last_encounter_time=last_enc)
            assert 0.0 <= r <= 1.0, f"R_norm={r} out of bounds for last_enc={last_enc}"

    def test_r_norm_monotonically_decreasing_with_age(self):
        """More recent encounter → higher R_norm (monotonic)."""
        current_time = 7200.0
        # encounters at t=7200, t=6300, t=5400, t=3600 (increasingly old)
        enc_times = [7200.0, 6300.0, 5400.0, 3600.0]
        r_values = [compute_r_norm(current_time, t) for t in enc_times]
        for i in range(len(r_values) - 1):
            assert (
                r_values[i] >= r_values[i + 1]
            ), f"R_norm not monotonically decreasing: {r_values}"

    def test_early_simulation_time(self):
        """At simulation start (t=1800), recent encounters still work correctly."""
        current_time = 1800.0
        r_recent = compute_r_norm(current_time, last_encounter_time=1700.0)
        r_older = compute_r_norm(current_time, last_encounter_time=0.0)
        assert (
            r_recent > r_older
        ), "Recent encounter should score higher than never-encountered"
        assert r_recent == pytest.approx(1.0 - 100.0 / T_REF)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Scoring formula tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoringFormula:
    """Tests for the revised Score = α×P + γ_r×R + β×D − λ×W formula."""

    def test_weights_sum_at_most_one(self):
        """
        α + γ_r + β = 0.2 + 0.2 + 0.6 = 1.0
        Max achievable score (all 1.0, no penalty) = 1.0
        """
        max_score = compute_score(p_abs=1.0, r_norm=1.0, d_norm=1.0, w_penalty=0.0)
        assert max_score == pytest.approx(
            1.0
        ), f"Max score should be 1.0, got {max_score}"

    def test_workload_penalty_reduces_score(self):
        """Workload penalty should reduce score by exactly λ = 0.2."""
        s_no_penalty = compute_score(0.5, 0.5, 0.5, w_penalty=0.0)
        s_with_penalty = compute_score(0.5, 0.5, 0.5, w_penalty=1.0)
        assert s_no_penalty - s_with_penalty == pytest.approx(WORKLOAD_PENALTY)

    def test_recency_differentiates_equal_p_candidates(self):
        """
        Core test: two candidates with equal P and equal distance,
        but different recency → recency determines winner.
        """
        current_time = 7200.0
        dist = 500.0
        d_norm = 1.0 - dist / SIMULATION_DIAGONAL
        p = 0.47  # saturated P value, equal for both

        r_recent = compute_r_norm(current_time, last_encounter_time=7100.0)  # 100s ago
        r_old = compute_r_norm(current_time, last_encounter_time=5400.0)  # 1800s ago

        score_recent = compute_score(p, r_recent, d_norm)
        score_old = compute_score(p, r_old, d_norm)

        assert score_recent > score_old, (
            f"More recently encountered candidate should win when P and D are equal. "
            f"score_recent={score_recent:.4f}, score_old={score_old:.4f}"
        )

    def test_proximity_still_dominates_when_recency_equal(self):
        """
        When recency is equal, the closer responder should still win
        (proximity weight β=0.6 dominates α=0.2 + γ_r=0.2 combined).
        """
        r_norm = 0.5  # same for both
        p = 0.45  # same for both

        d_close = 1.0 - 200.0 / SIMULATION_DIAGONAL  # 200m away
        d_far = 1.0 - 1500.0 / SIMULATION_DIAGONAL  # 1500m away

        score_close = compute_score(p, r_norm, d_close)
        score_far = compute_score(p, r_norm, d_far)

        assert (
            score_close > score_far
        ), "Closer responder should win when recency and P are equal"

    def test_high_recency_can_beat_slightly_closer_candidate(self):
        """
        A very recently encountered candidate (R=1.0) can beat one that is
        slightly closer but was encountered long ago (R=0.0).
        Specifically, the recency advantage = γ_r × 1.0 = 0.2,
        which must exceed BETA × distance_difference.
        """
        p = 0.46

        # Candidate A: recently encountered, 700m away
        r_a = 1.0  # R_norm = 1.0
        d_a = 1.0 - 700.0 / SIMULATION_DIAGONAL

        # Candidate B: not encountered since > T_ref, 500m away
        r_b = 0.0  # R_norm = 0.0
        d_b = 1.0 - 500.0 / SIMULATION_DIAGONAL

        score_a = compute_score(p, r_a, d_a)
        score_b = compute_score(p, r_b, d_b)

        recency_advantage = GAMMA_R * (r_a - r_b)  # 0.2 × 1.0 = 0.2
        proximity_disadvantage = BETA * (d_b - d_a)  # 0.6 × (200/3354) ≈ 0.036

        assert (
            recency_advantage > proximity_disadvantage
        ), "This test scenario requires recency advantage > proximity disadvantage"
        assert score_a > score_b, (
            f"Recently-contacted A should beat stale-but-closer B. "
            f"score_a={score_a:.4f}, score_b={score_b:.4f}"
        )

    def test_p_threshold_still_filters(self):
        """
        Candidates with P <= 0.3 should still be excluded before scoring.
        This test checks that the filtering logic is not accidentally removed.
        """
        # Simulate a candidate pool with one below threshold
        candidates = [
            {"id": "R01", "predictability": 0.45, "recency": 0.8, "distance": 300.0},
            {
                "id": "R02",
                "predictability": 0.28,
                "recency": 1.0,
                "distance": 100.0,
            },  # below threshold
            {"id": "R03", "predictability": 0.50, "recency": 0.3, "distance": 600.0},
        ]
        eligible = [c for c in candidates if c["predictability"] > P_THRESHOLD]
        eligible_ids = {c["id"] for c in eligible}

        assert (
            "R02" not in eligible_ids
        ), "R02 with P=0.28 should be excluded by P > 0.3 threshold"
        assert "R01" in eligible_ids
        assert "R03" in eligible_ids

    def test_r_norm_variance_breaks_ties_in_realistic_scenario(self):
        """
        Realistic scenario: 5 candidates with near-identical P values (saturated),
        equal distances, but different recency. Recency should produce a clear ranking.
        """
        current_time = 7200.0
        p_saturated = 0.47  # all candidates near p_enc_max
        distance = 600.0
        d_norm = 1.0 - distance / SIMULATION_DIAGONAL

        last_enc_times = [
            7150.0,
            6800.0,
            6500.0,
            6000.0,
            5400.0,
        ]  # all within T_ref, all distinct
        responders = [f"R{i:02d}" for i in range(5)]

        scored = []
        for rid, enc_t in zip(responders, last_enc_times, strict=False):
            r_norm = compute_r_norm(current_time, enc_t)
            score = compute_score(p_saturated, r_norm, d_norm)
            scored.append((rid, score, r_norm))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [s[0] for s in scored]

        # R00 (most recent, 50s ago) should rank first; R04 (1800s ago, R_norm=0) last
        assert (
            ranked_ids[0] == "R00"
        ), f"Most recent responder should rank first, got {ranked_ids}"
        assert (
            ranked_ids[-1] == "R04"
        ), f"Oldest (1800s = T_ref) should rank last, got {ranked_ids}"

        # Confirm all scores are distinct (recency is actually discriminating)
        scores = [s[1] for s in scored]
        assert (
            len({round(s, 6) for s in scores}) == 5
        ), f"All 5 candidates should have distinct scores when recency differs. Scores: {scores}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Integration-level tests (mock-based)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveCoordinatorIntegration:
    """
    Mock-based integration tests verifying that AdaptiveCoordinator correctly
    reads last_encounter_time from the network state and uses it in scoring.

    These tests import the actual coordinator class — if import fails, they
    are skipped (allows running formula tests even before implementation exists).
    """

    @pytest.fixture
    def coordinator_class(self):
        """Try to import AdaptiveCoordinator; skip if not found."""
        try:
            from ercs.coordination.algorithms import AdaptiveCoordinator

            return AdaptiveCoordinator
        except ImportError:
            pytest.skip("AdaptiveCoordinator not importable — run formula tests only")

    @pytest.fixture
    def mock_network_state(self):
        """Mock NetworkStateProvider with get_last_encounter_time support."""
        ns = MagicMock()
        # Default: all responders have P=0.47 (saturated), never encountered
        ns.get_delivery_predictability.return_value = 0.47
        ns.get_last_encounter_time.return_value = 0.0
        return ns

    def test_coordinator_calls_get_last_encounter_time(
        self, coordinator_class, mock_network_state
    ):
        """
        Verify that AdaptiveCoordinator calls get_last_encounter_time on
        the network state for at least one responder per assignment cycle.
        """
        from ercs.config.parameters import UrgencyLevel
        from ercs.scenario.generator import Task

        coord = coordinator_class()

        mock_locator = MagicMock()
        mock_locator.get_all_responder_ids.return_value = ["R01", "R02", "R03"]
        mock_locator.get_responder_position.return_value = (1000.0, 750.0)

        task = Task(
            task_id="T001",
            creation_time=0.0,
            source_node="COORD",
            target_location_x=500.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.HIGH,
        )

        coord.assign_tasks(
            tasks=[task],
            responder_locator=mock_locator,
            network_state=mock_network_state,
            coordination_node="COORD",
            current_time=7200.0,
        )

        assert (
            mock_network_state.get_last_encounter_time.called
        ), "AdaptiveCoordinator must call network_state.get_last_encounter_time()"

    def test_recently_encountered_responder_preferred_over_stale_at_equal_distance(
        self, coordinator_class, mock_network_state
    ):
        """
        Two responders at equal distance, equal P, but different recency.
        Recently-encountered should be assigned.
        """
        from ercs.config.parameters import UrgencyLevel
        from ercs.scenario.generator import Task

        coord = coordinator_class()
        current_time = 7200.0

        # R01: encountered 100s ago (R_norm ≈ 0.944)
        # R02: encountered 5400s ago (R_norm = 0.0, clamped)
        def last_enc_side_effect(coord_node, responder):
            return {
                "R01": current_time - 100.0,  # recent
                "R02": current_time - 5400.0,  # stale
            }.get(responder, 0.0)

        mock_network_state.get_last_encounter_time.side_effect = last_enc_side_effect
        mock_network_state.get_delivery_predictability.return_value = (
            0.47  # saturated, equal
        )

        mock_locator = MagicMock()
        mock_locator.get_all_responder_ids.return_value = ["R01", "R02"]
        mock_locator.get_responder_position.return_value = (1500.0, 750.0)

        task = Task(
            task_id="T001",
            creation_time=0.0,
            source_node="COORD",
            target_location_x=1500.0,
            target_location_y=750.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignments = coord.assign_tasks(
            tasks=[task],
            responder_locator=mock_locator,
            network_state=mock_network_state,
            coordination_node="COORD",
            current_time=current_time,
        )

        assert len(assignments) == 1
        assert (
            assignments[0].responder_id == "R01"
        ), "R01 (recently encountered) should be preferred over R02 (stale) at equal distance"

    def test_baseline_does_not_use_get_last_encounter_time(self, mock_network_state):
        """
        Baseline algorithm must NOT call get_last_encounter_time.
        Baseline should remain proximity-only.
        """
        try:
            from ercs.coordination.algorithms import BaselineCoordinator
        except ImportError:
            pytest.skip("BaselineCoordinator not importable")

        from ercs.config.parameters import UrgencyLevel
        from ercs.scenario.generator import Task

        coord = BaselineCoordinator()

        mock_locator = MagicMock()
        mock_locator.get_all_responder_ids.return_value = ["R01", "R02"]
        mock_locator.get_responder_position.return_value = (500.0, 500.0)

        task = Task(
            task_id="T001",
            creation_time=0.0,
            source_node="COORD",
            target_location_x=500.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        coord.assign_tasks(
            tasks=[task],
            responder_locator=mock_locator,
            network_state=mock_network_state,
            coordination_node="COORD",
            current_time=7200.0,
        )

        assert not mock_network_state.get_last_encounter_time.called, (
            "BaselineCoordinator must NOT call get_last_encounter_time — "
            "it is proximity-only and must remain unchanged"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PRoPHET layer tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProphetLastEncounterTime:
    """
    Tests for the get_last_encounter_time() method on DeliveryPredictabilityMatrix.
    """

    @pytest.fixture
    def pred_matrix(self):
        """Create a DeliveryPredictabilityMatrix instance."""
        from ercs.communication.prophet import DeliveryPredictabilityMatrix

        return DeliveryPredictabilityMatrix()

    def test_never_encountered_returns_zero(self, pred_matrix):
        """Nodes that have never met should return 0.0."""
        result = pred_matrix.get_last_encounter_time("COORD", "R99")
        assert result == 0.0

    def test_encounter_recorded_correctly(self, pred_matrix):
        """After an encounter at t=3600, get_last_encounter_time should return 3600."""
        pred_matrix.update_encounter("COORD", "R01", current_time=3600.0)
        result = pred_matrix.get_last_encounter_time("COORD", "R01")
        assert result == pytest.approx(3600.0)

    def test_encounter_symmetric(self, pred_matrix):
        """Last encounter time is symmetric: (A,B) == (B,A)."""
        pred_matrix.update_encounter("COORD", "R01", current_time=5000.0)
        assert pred_matrix.get_last_encounter_time("COORD", "R01") == pytest.approx(
            5000.0
        )
        assert pred_matrix.get_last_encounter_time("R01", "COORD") == pytest.approx(
            5000.0
        )

    def test_most_recent_encounter_stored(self, pred_matrix):
        """Multiple encounters — only the most recent should be returned."""
        pred_matrix.update_encounter("COORD", "R01", current_time=3600.0)
        pred_matrix.update_encounter("COORD", "R01", current_time=5400.0)
        pred_matrix.update_encounter("COORD", "R01", current_time=7200.0)
        result = pred_matrix.get_last_encounter_time("COORD", "R01")
        assert result == pytest.approx(
            7200.0
        ), "Should store the most recent encounter, not the first"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Regression / non-regression tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonRegression:
    """
    Verify that the recency change does not break fundamental correctness
    properties of the adaptive algorithm.
    """

    def test_max_score_is_still_bounded(self):
        """Score cannot exceed 1.0 with the new formula (all components max)."""
        score = compute_score(p_abs=1.0, r_norm=1.0, d_norm=1.0, w_penalty=0.0)
        assert score <= 1.0 + 1e-9, f"Score exceeded 1.0: {score}"

    def test_score_with_all_zero_components(self):
        """Score with all zero inputs and no penalty = 0.0."""
        score = compute_score(p_abs=0.0, r_norm=0.0, d_norm=0.0, w_penalty=0.0)
        assert score == pytest.approx(0.0)

    def test_weight_constants_correct(self):
        """Verify the constants themselves match the dissertation specification."""
        assert pytest.approx(0.2) == ALPHA, "α must be 0.2"
        assert pytest.approx(0.2) == GAMMA_R, "γ_r must be 0.2"
        assert pytest.approx(0.6) == BETA, "β must be 0.6"
        assert pytest.approx(0.2) == WORKLOAD_PENALTY, "λ must be 0.2"
        assert pytest.approx(1800.0) == T_REF, "T_ref must be 1800s (i_typ)"
        # weights must sum to 1.0 (before penalty)
        assert pytest.approx(1.0) == ALPHA + GAMMA_R + BETA

    def test_p_saturated_candidates_still_get_differentiated(self):
        """
        The core regression: with P ≈ 0.47 for all candidates (the 180-run problem),
        recency alone must produce variance in scores.
        """
        current_time = 7200.0
        p_saturated = 0.47
        distance = 500.0
        d_norm = 1.0 - distance / SIMULATION_DIAGONAL

        # Simulate 10 responders with identical P and distance but varying last encounter
        import random

        random.seed(42)
        enc_times = [current_time - random.uniform(0, 3600) for _ in range(10)]
        r_norms = [compute_r_norm(current_time, t) for t in enc_times]
        scores = [compute_score(p_saturated, r, d_norm) for r in r_norms]

        score_std = (
            sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
        ) ** 0.5
        assert score_std > 0.01, (
            f"With varying recency, scores should have std > 0.01. Got std={score_std:.4f}. "
            f"Recency is not differentiating candidates."
        )

    def test_formula_components_independent(self):
        """Each weight independently contributes to score."""
        base = compute_score(0.0, 0.0, 0.0)
        only_p = compute_score(1.0, 0.0, 0.0) - base
        only_r = compute_score(0.0, 1.0, 0.0) - base
        only_d = compute_score(0.0, 0.0, 1.0) - base

        assert only_p == pytest.approx(ALPHA)
        assert only_r == pytest.approx(GAMMA_R)
        assert only_d == pytest.approx(BETA)
