"""
Tests for Performance Evaluation (Phase 6).

These tests verify that the evaluation module correctly implements
statistical analysis for comparing algorithm performance.

Tests cover:
- Descriptive statistics calculation
- Independent samples t-test
- One-way ANOVA
- Effect size calculations
- Metric extraction
- Evaluation report generation
"""

import pytest
import numpy as np

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.evaluation import (
    ANOVAResult,
    ComparisonResult,
    DescriptiveStats,
    EvaluationReport,
    MetricExtractor,
    MetricType,
    PerformanceEvaluator,
    StatisticalAnalyzer,
    TTestResult,
    evaluate_results,
)
from ercs.simulation.engine import SimulationResults

from conftest import (
    CONNECTIVITY_MILD,
    CONNECTIVITY_MODERATE,
    CONNECTIVITY_SEVERE,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_results() -> list[SimulationResults]:
    """Create sample simulation results for testing."""
    config = SimulationConfig()
    results = []

    # Create results with known patterns
    # Adaptive algorithm: better delivery rate at high connectivity
    # Baseline algorithm: worse delivery rate

    np.random.seed(42)

    for algorithm in [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]:
        for connectivity in [CONNECTIVITY_MILD, CONNECTIVITY_MODERATE, CONNECTIVITY_SEVERE]:
            for run in range(10):
                result = SimulationResults(
                    config=config,
                    algorithm=algorithm,
                    connectivity_level=connectivity,
                    run_number=run,
                    random_seed=run,
                )

                # Set metrics with known patterns
                result.total_tasks = 100

                if algorithm == AlgorithmType.ADAPTIVE:
                    # Adaptive: better performance
                    base_delivery = 0.7 + connectivity * 0.2
                    result.tasks_assigned = int(80 + np.random.normal(0, 5))
                    result.messages_created = result.tasks_assigned
                    result.messages_delivered = int(
                        result.messages_created
                        * (base_delivery + np.random.normal(0, 0.05))
                    )
                    result.response_times = [
                        (f"task_{i}", 30 + np.random.normal(0, 5))
                        for i in range(result.tasks_assigned)
                    ]
                else:
                    # Baseline: worse performance
                    base_delivery = 0.5 + connectivity * 0.2
                    result.tasks_assigned = int(70 + np.random.normal(0, 5))
                    result.messages_created = result.tasks_assigned
                    result.messages_delivered = int(
                        result.messages_created
                        * (base_delivery + np.random.normal(0, 0.05))
                    )
                    result.response_times = [
                        (f"task_{i}", 50 + np.random.normal(0, 10))
                        for i in range(result.tasks_assigned)
                    ]

                results.append(result)

    return results


@pytest.fixture
def analyzer() -> StatisticalAnalyzer:
    """Create statistical analyzer."""
    return StatisticalAnalyzer(alpha=0.05)


# =============================================================================
# Test DescriptiveStats
# =============================================================================


class TestDescriptiveStats:
    """Tests for descriptive statistics."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = DescriptiveStats(
            n=30,
            mean=0.75,
            std=0.05,
            median=0.74,
            min=0.65,
            max=0.85,
            ci_lower=0.73,
            ci_upper=0.77,
        )

        d = stats.to_dict()

        assert d["n"] == 30
        assert d["mean"] == 0.75
        assert d["std"] == 0.05
        assert d["ci_95_lower"] == 0.73
        assert d["ci_95_upper"] == 0.77


# =============================================================================
# Test StatisticalAnalyzer
# =============================================================================


class TestStatisticalAnalyzer:
    """Tests for statistical analyzer."""

    def test_descriptive_stats_basic(self, analyzer: StatisticalAnalyzer):
        """Test basic descriptive statistics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = analyzer.descriptive_stats(values)

        assert stats.n == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)

    def test_descriptive_stats_empty(self, analyzer: StatisticalAnalyzer):
        """Test descriptive stats with empty list."""
        stats = analyzer.descriptive_stats([])

        assert stats.n == 0
        assert stats.mean == 0.0

    def test_descriptive_stats_single_value(self, analyzer: StatisticalAnalyzer):
        """Test descriptive stats with single value."""
        stats = analyzer.descriptive_stats([5.0])

        assert stats.n == 1
        assert stats.mean == 5.0
        assert stats.std == 0.0

    def test_confidence_interval(self, analyzer: StatisticalAnalyzer):
        """Test 95% confidence interval calculation."""
        # Generate values with known distribution
        np.random.seed(42)
        values = list(np.random.normal(100, 10, 100))

        stats = analyzer.descriptive_stats(values)

        # CI should contain the mean
        assert stats.ci_lower < stats.mean < stats.ci_upper
        # CI should be reasonable width
        assert stats.ci_upper - stats.ci_lower < 10  # Not too wide

    def test_ttest_significant_difference(self, analyzer: StatisticalAnalyzer):
        """Test t-test detects significant difference."""
        np.random.seed(42)
        group1 = list(np.random.normal(100, 10, 30))
        group2 = list(np.random.normal(80, 10, 30))

        result = analyzer.independent_ttest(group1, group2, "High", "Low")

        assert result.significant is True
        assert result.p_value < 0.05
        assert result.t_statistic > 0  # group1 mean > group2 mean

    def test_ttest_no_significant_difference(self, analyzer: StatisticalAnalyzer):
        """Test t-test when no significant difference."""
        np.random.seed(42)
        group1 = list(np.random.normal(100, 10, 30))
        group2 = list(np.random.normal(100, 10, 30))

        result = analyzer.independent_ttest(group1, group2)

        # Should not be significant (same distribution)
        assert result.p_value > 0.05 or abs(result.cohens_d) < 0.2

    def test_ttest_effect_size(self, analyzer: StatisticalAnalyzer):
        """Test Cohen's d effect size calculation."""
        # Large effect size (d > 0.8)
        group1 = [10, 11, 12, 13, 14]
        group2 = [1, 2, 3, 4, 5]

        result = analyzer.independent_ttest(group1, group2)

        assert abs(result.cohens_d) > 0.8  # Large effect

    def test_ttest_summary(self, analyzer: StatisticalAnalyzer):
        """Test t-test summary generation."""
        group1 = [10, 11, 12, 13, 14]
        group2 = [8, 9, 10, 11, 12]

        result = analyzer.independent_ttest(group1, group2, "A", "B")
        summary = result.summary()

        assert "A vs B" in summary
        assert "t(" in summary
        assert "p =" in summary

    def test_anova_significant(self, analyzer: StatisticalAnalyzer):
        """Test ANOVA detects significant differences."""
        np.random.seed(42)
        groups = {
            "75%": list(np.random.normal(100, 5, 30)),
            "40%": list(np.random.normal(80, 5, 30)),
            "20%": list(np.random.normal(60, 5, 30)),
        }

        result = analyzer.one_way_anova(groups)

        assert result.significant is True
        assert result.p_value < 0.05
        assert result.f_statistic > 0

    def test_anova_no_significant(self, analyzer: StatisticalAnalyzer):
        """Test ANOVA when no significant differences."""
        np.random.seed(42)
        groups = {
            "A": list(np.random.normal(100, 10, 30)),
            "B": list(np.random.normal(100, 10, 30)),
            "C": list(np.random.normal(100, 10, 30)),
        }

        result = analyzer.one_way_anova(groups)

        # Should not be significant
        assert result.p_value > 0.05 or result.eta_squared < 0.01

    def test_anova_eta_squared(self, analyzer: StatisticalAnalyzer):
        """Test eta-squared effect size calculation."""
        groups = {
            "Low": [10, 11, 12, 13, 14],
            "Medium": [50, 51, 52, 53, 54],
            "High": [90, 91, 92, 93, 94],
        }

        result = analyzer.one_way_anova(groups)

        # Should have large effect size
        assert result.eta_squared > 0.14

    def test_anova_degrees_of_freedom(self, analyzer: StatisticalAnalyzer):
        """Test ANOVA degrees of freedom calculation."""
        groups = {
            "A": [1, 2, 3, 4, 5],  # n=5
            "B": [6, 7, 8, 9, 10],  # n=5
            "C": [11, 12, 13, 14, 15],  # n=5
        }

        result = analyzer.one_way_anova(groups)

        assert result.df_between == 2  # k - 1 = 3 - 1
        assert result.df_within == 12  # N - k = 15 - 3


# =============================================================================
# Test MetricExtractor
# =============================================================================


class TestMetricExtractor:
    """Tests for metric extraction."""

    @pytest.fixture
    def sample_result(self) -> SimulationResults:
        """Create a sample result."""
        result = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        result.total_tasks = 100
        result.tasks_assigned = 80
        result.messages_created = 80
        result.messages_delivered = 60
        result.response_times = [("t1", 30.0), ("t2", 40.0)]
        return result

    def test_extract_delivery_rate(self, sample_result: SimulationResults):
        """Test extracting delivery rate."""
        values = MetricExtractor.extract([sample_result], MetricType.DELIVERY_RATE)

        assert len(values) == 1
        assert values[0] == pytest.approx(0.75)  # 60/80

    def test_extract_assignment_rate(self, sample_result: SimulationResults):
        """Test extracting assignment rate."""
        values = MetricExtractor.extract([sample_result], MetricType.ASSIGNMENT_RATE)

        assert len(values) == 1
        assert values[0] == pytest.approx(0.80)  # 80/100

    def test_extract_response_time(self, sample_result: SimulationResults):
        """Test extracting response time."""
        values = MetricExtractor.extract([sample_result], MetricType.DECISION_TIME)

        assert len(values) == 1
        assert values[0] == pytest.approx(35.0)  # (30+40)/2

    def test_extract_multiple_results(self, sample_result: SimulationResults):
        """Test extracting from multiple results."""
        results = [sample_result, sample_result]
        values = MetricExtractor.extract(results, MetricType.DELIVERY_RATE)

        assert len(values) == 2

    def test_extract_handles_none(self):
        """Test that None values are filtered out."""
        result = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        # No response times set -> average_decision_time is None

        values = MetricExtractor.extract([result], MetricType.DECISION_TIME)

        assert len(values) == 0  # None filtered out


# =============================================================================
# Test PerformanceEvaluator
# =============================================================================


class TestPerformanceEvaluator:
    """Tests for performance evaluator."""

    def test_evaluator_creation(self, sample_results: list[SimulationResults]):
        """Test evaluator initialization."""
        evaluator = PerformanceEvaluator(sample_results)

        assert len(evaluator.results) == 60  # 2 algorithms × 3 connectivity × 10 runs

    def test_compare_algorithms(self, sample_results: list[SimulationResults]):
        """Test algorithm comparison."""
        evaluator = PerformanceEvaluator(sample_results)

        comparison = evaluator.compare_algorithms(MetricType.DELIVERY_RATE)

        assert comparison.metric == MetricType.DELIVERY_RATE
        assert comparison.connectivity_level is None  # Overall
        assert comparison.adaptive_stats.n == 30
        assert comparison.baseline_stats.n == 30

    def test_compare_at_connectivity(self, sample_results: list[SimulationResults]):
        """Test comparison at specific connectivity level."""
        evaluator = PerformanceEvaluator(sample_results)

        comparison = evaluator.compare_algorithms(
            MetricType.DELIVERY_RATE,
            connectivity_level=CONNECTIVITY_MILD,
        )

        assert comparison.connectivity_level == CONNECTIVITY_MILD
        assert comparison.adaptive_stats.n == 10
        assert comparison.baseline_stats.n == 10

    def test_improvement_calculation(self, sample_results: list[SimulationResults]):
        """Test improvement percentage calculation."""
        evaluator = PerformanceEvaluator(sample_results)

        comparison = evaluator.compare_algorithms(MetricType.DELIVERY_RATE)

        # Adaptive should be better (positive improvement)
        # Based on our fixture setup
        assert comparison.improvement != 0

    def test_analyze_connectivity_effect(self, sample_results: list[SimulationResults]):
        """Test ANOVA for connectivity effect."""
        evaluator = PerformanceEvaluator(sample_results)

        anova = evaluator.analyze_connectivity_effect(
            MetricType.DELIVERY_RATE,
            algorithm=AlgorithmType.ADAPTIVE,
        )

        assert len(anova.group_names) == 3  # 3 connectivity levels
        assert "75%" in anova.group_names
        assert "40%" in anova.group_names
        assert "20%" in anova.group_names

    def test_generate_report(self, sample_results: list[SimulationResults]):
        """Test report generation."""
        evaluator = PerformanceEvaluator(sample_results)

        report = evaluator.generate_report()

        assert isinstance(report, EvaluationReport)
        assert len(report.comparisons) > 0
        assert len(report.anova_results) > 0
        assert "total_runs" in report.summary_stats

    def test_report_to_dict(self, sample_results: list[SimulationResults]):
        """Test report conversion to dictionary."""
        evaluator = PerformanceEvaluator(sample_results)
        report = evaluator.generate_report()

        d = report.to_dict()

        assert "comparisons" in d
        assert "anova" in d
        assert "summary" in d


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestEvaluateResults:
    """Tests for evaluate_results convenience function."""

    def test_evaluate_results(self, sample_results: list[SimulationResults]):
        """Test evaluate_results function."""
        report = evaluate_results(sample_results, print_summary=False)

        assert isinstance(report, EvaluationReport)
        assert len(report.comparisons) > 0


# =============================================================================
# Test System Availability (Mudança 3)
# =============================================================================


class TestSystemAvailability:
    """Tests for System Availability metric."""

    def test_system_availability_full(self):
        """All cycles active → 100%."""
        result = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        result.total_coordination_cycles = 10
        result.active_coordination_cycles = 10

        assert result.system_availability == pytest.approx(100.0)

    def test_system_availability_partial(self):
        """7 of 10 cycles active → 70%."""
        result = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        result.total_coordination_cycles = 10
        result.active_coordination_cycles = 7

        assert result.system_availability == pytest.approx(70.0)

    def test_system_availability_zero_cycles(self):
        """No cycles → None."""
        result = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        result.total_coordination_cycles = 0

        assert result.system_availability is None

    def test_compute_system_availability_aggregate(self):
        """PerformanceEvaluator aggregates across runs."""
        config = SimulationConfig()
        results = []
        for i in range(5):
            r = SimulationResults(
                config=config,
                algorithm=AlgorithmType.ADAPTIVE,
                connectivity_level=0.75,
                run_number=i,
                random_seed=i,
            )
            r.total_coordination_cycles = 10
            r.active_coordination_cycles = 7 + i  # 70%-110% range
            results.append(r)

        evaluator = PerformanceEvaluator(results)
        stats = evaluator.compute_system_availability(results)

        assert stats.n == 5
        assert stats.mean > 0


# =============================================================================
# Test Urgency-Stratified Delivery (Mudança 5)
# =============================================================================


class TestUrgencyStratifiedDelivery:
    """Tests for urgency-stratified delivery rate metric."""

    def test_urgency_stratified_correct_rates(self):
        """Verify correct per-urgency delivery rates."""
        from ercs.simulation.engine import SimulationEvent, SimulationEventType

        config = SimulationConfig()
        result = SimulationResults(
            config=config,
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )

        # 5 H tasks assigned, 3 delivered
        for i in range(5):
            result.events.append(SimulationEvent(
                event_type=SimulationEventType.TASK_ASSIGNED,
                timestamp=100.0,
                data={"task_id": f"h_{i}", "urgency": "H", "responder_id": f"r_{i}"},
            ))
        result.delivery_times = [
            ("h_0", 50.0), ("h_1", 60.0), ("h_2", 70.0),
        ]

        # 10 M tasks assigned, 8 delivered
        for i in range(10):
            result.events.append(SimulationEvent(
                event_type=SimulationEventType.TASK_ASSIGNED,
                timestamp=100.0,
                data={"task_id": f"m_{i}", "urgency": "M", "responder_id": f"r_{i}"},
            ))
        result.delivery_times.extend([
            (f"m_{i}", 50.0 + i) for i in range(8)
        ])

        evaluator = PerformanceEvaluator([result])
        rates = evaluator.compute_urgency_stratified_delivery([result])

        assert "H" in rates
        assert "M" in rates
        assert rates["H"].mean == pytest.approx(3 / 5)
        assert rates["M"].mean == pytest.approx(8 / 10)

    def test_urgency_stratified_no_urgency_field(self):
        """Events without urgency field → empty result (no crash)."""
        from ercs.simulation.engine import SimulationEvent, SimulationEventType

        config = SimulationConfig()
        result = SimulationResults(
            config=config,
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )

        # TASK_ASSIGNED events without urgency field
        result.events.append(SimulationEvent(
            event_type=SimulationEventType.TASK_ASSIGNED,
            timestamp=100.0,
            data={"task_id": "t_0", "responder_id": "r_0"},
        ))

        evaluator = PerformanceEvaluator([result])
        rates = evaluator.compute_urgency_stratified_delivery([result])

        # No urgency data → empty dict
        assert len(rates) == 0


# =============================================================================
# Test TTestResult
# =============================================================================


class TestTTestResult:
    """Tests for TTestResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats1 = DescriptiveStats(10, 0.8, 0.1, 0.8, 0.6, 1.0, 0.7, 0.9)
        stats2 = DescriptiveStats(10, 0.6, 0.1, 0.6, 0.4, 0.8, 0.5, 0.7)

        result = TTestResult(
            group1_name="Adaptive",
            group2_name="Baseline",
            group1_stats=stats1,
            group2_stats=stats2,
            t_statistic=3.5,
            p_value=0.001,
            degrees_of_freedom=18,
            cohens_d=1.2,
            significant=True,
        )

        d = result.to_dict()

        assert d["group1"] == "Adaptive"
        assert d["group2"] == "Baseline"
        assert d["t_statistic"] == 3.5
        assert d["significant"] is True

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        stats = DescriptiveStats(10, 0.5, 0.1, 0.5, 0.3, 0.7, 0.4, 0.6)

        # Negligible effect
        result = TTestResult("A", "B", stats, stats, 0.5, 0.6, 18, 0.1, False)
        assert "negligible" in result.summary()

        # Small effect
        result = TTestResult("A", "B", stats, stats, 1.0, 0.3, 18, 0.3, False)
        assert "small" in result.summary()

        # Medium effect
        result = TTestResult("A", "B", stats, stats, 2.0, 0.05, 18, 0.6, True)
        assert "medium" in result.summary()

        # Large effect
        result = TTestResult("A", "B", stats, stats, 4.0, 0.001, 18, 1.0, True)
        assert "large" in result.summary()


# =============================================================================
# Test ANOVAResult
# =============================================================================


class TestANOVAResult:
    """Tests for ANOVAResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = DescriptiveStats(10, 0.5, 0.1, 0.5, 0.3, 0.7, 0.4, 0.6)

        result = ANOVAResult(
            group_names=["A", "B", "C"],
            group_stats={"A": stats, "B": stats, "C": stats},
            f_statistic=5.5,
            p_value=0.01,
            df_between=2,
            df_within=27,
            eta_squared=0.15,
            significant=True,
        )

        d = result.to_dict()

        assert d["groups"] == ["A", "B", "C"]
        assert d["f_statistic"] == 5.5
        assert d["significant"] is True

    def test_eta_squared_interpretation(self):
        """Test eta-squared interpretation."""
        stats = DescriptiveStats(10, 0.5, 0.1, 0.5, 0.3, 0.7, 0.4, 0.6)
        group_stats = {"A": stats, "B": stats}

        # Negligible
        result = ANOVAResult(["A", "B"], group_stats, 0.5, 0.5, 1, 18, 0.005, False)
        assert "negligible" in result.summary()

        # Small
        result = ANOVAResult(["A", "B"], group_stats, 1.0, 0.3, 1, 18, 0.03, False)
        assert "small" in result.summary()

        # Medium
        result = ANOVAResult(["A", "B"], group_stats, 3.0, 0.05, 1, 18, 0.10, True)
        assert "medium" in result.summary()

        # Large
        result = ANOVAResult(["A", "B"], group_stats, 10.0, 0.001, 1, 18, 0.20, True)
        assert "large" in result.summary()


# =============================================================================
# Test ComparisonResult
# =============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats1 = DescriptiveStats(10, 0.8, 0.1, 0.8, 0.6, 1.0, 0.7, 0.9)
        stats2 = DescriptiveStats(10, 0.6, 0.1, 0.6, 0.4, 0.8, 0.5, 0.7)
        ttest = TTestResult("A", "B", stats1, stats2, 3.0, 0.01, 18, 1.0, True)

        result = ComparisonResult(
            metric=MetricType.DELIVERY_RATE,
            connectivity_level=CONNECTIVITY_MILD,
            adaptive_stats=stats1,
            baseline_stats=stats2,
            ttest=ttest,
            improvement=33.33,
        )

        d = result.to_dict()

        assert d["metric"] == "delivery_rate"
        assert d["connectivity_level"] == CONNECTIVITY_MILD
        assert d["improvement_pct"] == 33.33
        assert d["significant"] is True
