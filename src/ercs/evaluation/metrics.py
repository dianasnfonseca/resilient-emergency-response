"""
Performance Evaluation (Phase 6).

This module provides statistical analysis tools for comparing
coordination algorithm performance across experimental conditions.

Implements:
- Metric aggregation from simulation results
- Independent samples t-tests for pairwise comparison
- One-way ANOVA for multi-group comparison
- Effect size calculations (Cohen's d)
- Results export and visualization support

"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from ercs.config.parameters import AlgorithmType
from ercs.simulation.engine import SimulationEventType, SimulationResults


class MetricType(str, Enum):
    """Types of performance metrics."""

    DELIVERY_RATE = "delivery_rate"
    ASSIGNMENT_RATE = "assignment_rate"
    DECISION_TIME = "decision_time"
    DELIVERY_TIME = "delivery_time"
    MESSAGES_CREATED = "messages_created"
    MESSAGES_DELIVERED = "messages_delivered"
    TASKS_ASSIGNED = "tasks_assigned"


@dataclass
class DescriptiveStats:
    """
    Descriptive statistics for a metric.

    Attributes:
        n: Sample size
        mean: Arithmetic mean
        std: Standard deviation
        median: Median value
        min: Minimum value
        max: Maximum value
        ci_lower: 95% confidence interval lower bound
        ci_upper: 95% confidence interval upper bound
    """

    n: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
        }


@dataclass
class TTestResult:
    """
    Result of independent samples t-test.

    Attributes:
        group1_name: Name of first group
        group2_name: Name of second group
        group1_stats: Descriptive stats for group 1
        group2_stats: Descriptive stats for group 2
        t_statistic: T-test statistic
        p_value: Two-tailed p-value
        degrees_of_freedom: Degrees of freedom
        cohens_d: Effect size (Cohen's d)
        significant: Whether p < 0.05
    """

    group1_name: str
    group2_name: str
    group1_stats: DescriptiveStats
    group2_stats: DescriptiveStats
    t_statistic: float
    p_value: float
    degrees_of_freedom: float
    cohens_d: float
    significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "group1": self.group1_name,
            "group2": self.group2_name,
            "group1_mean": self.group1_stats.mean,
            "group2_mean": self.group2_stats.mean,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "cohens_d": self.cohens_d,
            "significant": self.significant,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_text = "significant" if self.significant else "not significant"
        effect = self._interpret_effect_size()
        return (
            f"{self.group1_name} vs {self.group2_name}: "
            f"t({self.degrees_of_freedom:.0f}) = {self.t_statistic:.3f}, "
            f"p = {self.p_value:.4f} ({sig_text}), "
            f"d = {self.cohens_d:.3f} ({effect} effect)"
        )

    def _interpret_effect_size(self) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class ANOVAResult:
    """
    Result of one-way ANOVA.

    Attributes:
        group_names: Names of groups compared
        group_stats: Descriptive stats for each group
        f_statistic: F-test statistic
        p_value: P-value
        df_between: Degrees of freedom between groups
        df_within: Degrees of freedom within groups
        eta_squared: Effect size (η²)
        significant: Whether p < 0.05
    """

    group_names: list[str]
    group_stats: dict[str, DescriptiveStats]
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    eta_squared: float
    significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "groups": self.group_names,
            "group_means": {k: v.mean for k, v in self.group_stats.items()},
            "f_statistic": self.f_statistic,
            "p_value": self.p_value,
            "df_between": self.df_between,
            "df_within": self.df_within,
            "eta_squared": self.eta_squared,
            "significant": self.significant,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_text = "significant" if self.significant else "not significant"
        effect = self._interpret_effect_size()
        return (
            f"ANOVA across {len(self.group_names)} groups: "
            f"F({self.df_between}, {self.df_within}) = {self.f_statistic:.3f}, "
            f"p = {self.p_value:.4f} ({sig_text}), "
            f"η² = {self.eta_squared:.3f} ({effect} effect)"
        )

    def _interpret_effect_size(self) -> str:
        """Interpret eta-squared effect size."""
        if self.eta_squared < 0.01:
            return "negligible"
        elif self.eta_squared < 0.06:
            return "small"
        elif self.eta_squared < 0.14:
            return "medium"
        else:
            return "large"


@dataclass
class ComparisonResult:
    """
    Complete comparison result for a metric.

    Attributes:
        metric: Metric being compared
        connectivity_level: Connectivity level (or None for all)
        adaptive_stats: Stats for adaptive algorithm
        baseline_stats: Stats for baseline algorithm
        ttest: T-test result
        improvement: Percentage improvement of adaptive over baseline
    """

    metric: MetricType
    connectivity_level: float | None
    adaptive_stats: DescriptiveStats
    baseline_stats: DescriptiveStats
    ttest: TTestResult
    improvement: float  # Percentage

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "connectivity_level": self.connectivity_level,
            "adaptive_mean": self.adaptive_stats.mean,
            "baseline_mean": self.baseline_stats.mean,
            "improvement_pct": self.improvement,
            "p_value": self.ttest.p_value,
            "significant": self.ttest.significant,
            "cohens_d": self.ttest.cohens_d,
        }


@dataclass
class EvaluationReport:
    """
    Complete evaluation report.

    Attributes:
        comparisons: List of metric comparisons
        anova_results: ANOVA results by metric
        summary_stats: Overall summary statistics
    """

    comparisons: list[ComparisonResult] = field(default_factory=list)
    anova_results: dict[str, ANOVAResult] = field(default_factory=dict)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "anova": {k: v.to_dict() for k, v in self.anova_results.items()},
            "summary": self.summary_stats,
        }


class MetricExtractor:
    """
    Extracts metric values from simulation results.

    Provides a consistent interface for accessing different metrics
    from SimulationResults objects.
    """

    _extractors: dict[MetricType, Callable[[SimulationResults], float | None]] = {
        MetricType.DELIVERY_RATE: lambda r: r.delivery_rate,
        MetricType.ASSIGNMENT_RATE: lambda r: r.assignment_rate,
        MetricType.DECISION_TIME: lambda r: r.average_decision_time,
        MetricType.DELIVERY_TIME: lambda r: r.average_delivery_time,
        MetricType.MESSAGES_CREATED: lambda r: float(r.messages_created),
        MetricType.MESSAGES_DELIVERED: lambda r: float(r.messages_delivered),
        MetricType.TASKS_ASSIGNED: lambda r: float(r.tasks_assigned),
    }

    @classmethod
    def extract(
        cls,
        results: list[SimulationResults],
        metric: MetricType,
    ) -> list[float]:
        """
        Extract metric values from results.

        Args:
            results: List of simulation results
            metric: Metric to extract

        Returns:
            List of metric values (excludes None values)
        """
        extractor = cls._extractors[metric]
        values = []
        for r in results:
            value = extractor(r)
            if value is not None:
                values.append(value)
        return values


class StatisticalAnalyzer:
    """
    Performs statistical analysis on simulation results.

    Implements:
    - Descriptive statistics with 95% confidence intervals
    - Independent samples t-tests (Welch's t-test)
    - One-way ANOVA
    - Effect size calculations

    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize analyzer.

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha

    def descriptive_stats(self, values: list[float]) -> DescriptiveStats:
        """
        Calculate descriptive statistics.

        Args:
            values: List of numeric values

        Returns:
            DescriptiveStats object
        """
        if not values:
            return DescriptiveStats(
                n=0,
                mean=0.0,
                std=0.0,
                median=0.0,
                min=0.0,
                max=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
            )

        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # 95% confidence interval
        if n > 1:
            se = std / np.sqrt(n)
            t_crit = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = mean
            ci_upper = mean

        return DescriptiveStats(
            n=n,
            mean=mean,
            std=std,
            median=float(np.median(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def independent_ttest(
        self,
        group1: list[float],
        group2: list[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
    ) -> TTestResult:
        """
        Perform independent samples t-test (Welch's t-test).

        Uses Welch's t-test which does not assume equal variances.

        Args:
            group1: Values for first group
            group2: Values for second group
            group1_name: Name of first group
            group2_name: Name of second group

        Returns:
            TTestResult object
        """
        stats1 = self.descriptive_stats(group1)
        stats2 = self.descriptive_stats(group2)

        if stats1.n < 2 or stats2.n < 2:
            # Not enough data for t-test
            return TTestResult(
                group1_name=group1_name,
                group2_name=group2_name,
                group1_stats=stats1,
                group2_stats=stats2,
                t_statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=0.0,
                cohens_d=0.0,
                significant=False,
            )

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

        # Welch-Satterthwaite degrees of freedom
        n1, n2 = stats1.n, stats2.n
        s1, s2 = stats1.std, stats2.std

        if s1 == 0 and s2 == 0:
            df = n1 + n2 - 2
        else:
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            denom = (s1**4 / (n1**2 * (n1 - 1))) + (s2**4 / (n2**2 * (n2 - 1)))
            df = num / denom if denom > 0 else n1 + n2 - 2

        # Cohen's d effect size
        cohens_d = self._cohens_d(group1, group2)

        return TTestResult(
            group1_name=group1_name,
            group2_name=group2_name,
            group1_stats=stats1,
            group2_stats=stats2,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            degrees_of_freedom=float(df),
            cohens_d=cohens_d,
            significant=bool(p_value < self.alpha),
        )

    def one_way_anova(
        self,
        groups: dict[str, list[float]],
    ) -> ANOVAResult:
        """
        Perform one-way ANOVA.

        Args:
            groups: Dictionary mapping group names to values

        Returns:
            ANOVAResult object
        """
        group_names = list(groups.keys())
        group_values = list(groups.values())
        group_stats = {
            name: self.descriptive_stats(vals) for name, vals in groups.items()
        }

        # Check minimum requirements
        valid_groups = [g for g in group_values if len(g) >= 2]
        if len(valid_groups) < 2:
            return ANOVAResult(
                group_names=group_names,
                group_stats=group_stats,
                f_statistic=0.0,
                p_value=1.0,
                df_between=0,
                df_within=0,
                eta_squared=0.0,
                significant=False,
            )

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_values)

        # Degrees of freedom
        k = len(groups)  # Number of groups
        n_total = sum(len(g) for g in group_values)
        df_between = k - 1
        df_within = n_total - k

        # Eta-squared effect size
        eta_squared = self._eta_squared(group_values)

        return ANOVAResult(
            group_names=group_names,
            group_stats=group_stats,
            f_statistic=float(f_stat),
            p_value=float(p_value),
            df_between=df_between,
            df_within=df_within,
            eta_squared=eta_squared,
            significant=bool(p_value < self.alpha),
        )

    def _cohens_d(self, group1: list[float], group2: list[float]) -> float:
        """
        Calculate Cohen's d effect size.

        Uses pooled standard deviation.
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return float((mean1 - mean2) / pooled_std)

    def _eta_squared(self, groups: list[list[float]]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        all_values = [v for g in groups for v in g]
        grand_mean = np.mean(all_values)

        # Sum of squares between groups
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)

        # Sum of squares total
        ss_total = sum((v - grand_mean) ** 2 for v in all_values)

        if ss_total == 0:
            return 0.0

        return float(ss_between / ss_total)


class PerformanceEvaluator:
    """
    Evaluates and compares algorithm performance.

    Provides high-level interface for:
    - Comparing adaptive vs baseline algorithms
    - Analyzing performance across connectivity levels
    - Generating evaluation reports

    Attributes:
        results: List of simulation results to analyze
        analyzer: Statistical analyzer instance
    """

    def __init__(
        self,
        results: list[SimulationResults],
        alpha: float = 0.05,
    ):
        """
        Initialize evaluator.

        Args:
            results: List of simulation results
            alpha: Significance level
        """
        self.results = results
        self.analyzer = StatisticalAnalyzer(alpha=alpha)

    def compare_algorithms(
        self,
        metric: MetricType,
        connectivity_level: float | None = None,
    ) -> ComparisonResult:
        """
        Compare adaptive vs baseline algorithm for a metric.

        Args:
            metric: Metric to compare
            connectivity_level: Filter by connectivity (None for all)

        Returns:
            ComparisonResult with statistical analysis
        """
        # Filter results
        if connectivity_level is not None:
            filtered = [
                r for r in self.results if r.connectivity_level == connectivity_level
            ]
        else:
            filtered = self.results

        # Separate by algorithm
        adaptive = [r for r in filtered if r.algorithm == AlgorithmType.ADAPTIVE]
        baseline = [r for r in filtered if r.algorithm == AlgorithmType.BASELINE]

        # Extract metric values
        adaptive_values = MetricExtractor.extract(adaptive, metric)
        baseline_values = MetricExtractor.extract(baseline, metric)

        # Calculate statistics
        adaptive_stats = self.analyzer.descriptive_stats(adaptive_values)
        baseline_stats = self.analyzer.descriptive_stats(baseline_values)

        # T-test
        ttest = self.analyzer.independent_ttest(
            adaptive_values,
            baseline_values,
            group1_name="Adaptive",
            group2_name="Baseline",
        )

        # Calculate improvement percentage
        if baseline_stats.mean != 0:
            improvement = (
                (adaptive_stats.mean - baseline_stats.mean) / baseline_stats.mean * 100
            )
        else:
            improvement = 0.0

        return ComparisonResult(
            metric=metric,
            connectivity_level=connectivity_level,
            adaptive_stats=adaptive_stats,
            baseline_stats=baseline_stats,
            ttest=ttest,
            improvement=improvement,
        )

    def analyze_connectivity_effect(
        self,
        metric: MetricType,
        algorithm: AlgorithmType | None = None,
    ) -> ANOVAResult:
        """
        Analyze effect of connectivity level on a metric using ANOVA.

        Args:
            metric: Metric to analyze
            algorithm: Filter by algorithm (None for all)

        Returns:
            ANOVAResult
        """
        # Filter by algorithm if specified
        if algorithm is not None:
            filtered = [r for r in self.results if r.algorithm == algorithm]
        else:
            filtered = self.results

        # Group by connectivity level
        connectivity_levels = sorted(set(r.connectivity_level for r in filtered))
        groups = {}

        for conn in connectivity_levels:
            conn_results = [r for r in filtered if r.connectivity_level == conn]
            values = MetricExtractor.extract(conn_results, metric)
            groups[f"{int(conn * 100)}%"] = values

        return self.analyzer.one_way_anova(groups)

    def generate_report(
        self,
        metrics: list[MetricType] | None = None,
    ) -> EvaluationReport:
        """
        Generate complete evaluation report.

        Args:
            metrics: Metrics to include (default: delivery_rate, response_time)

        Returns:
            EvaluationReport with all analyses
        """
        if metrics is None:
            metrics = [MetricType.DELIVERY_RATE, MetricType.DECISION_TIME]

        report = EvaluationReport()

        # Get connectivity levels
        connectivity_levels = sorted(set(r.connectivity_level for r in self.results))

        # Compare algorithms at each connectivity level
        for metric in metrics:
            # Overall comparison
            overall = self.compare_algorithms(metric)
            report.comparisons.append(overall)

            # Per connectivity level
            for conn in connectivity_levels:
                comparison = self.compare_algorithms(metric, conn)
                report.comparisons.append(comparison)

            # ANOVA for connectivity effect
            anova_adaptive = self.analyze_connectivity_effect(
                metric, AlgorithmType.ADAPTIVE
            )
            anova_baseline = self.analyze_connectivity_effect(
                metric, AlgorithmType.BASELINE
            )

            report.anova_results[f"{metric.value}_adaptive"] = anova_adaptive
            report.anova_results[f"{metric.value}_baseline"] = anova_baseline

        # Summary statistics
        report.summary_stats = self._generate_summary()

        return report

    def compute_system_availability(
        self, results: list[SimulationResults],
    ) -> DescriptiveStats:
        """Aggregate System Availability across runs.

        Formula: Active Coordination Cycles / Total Coordination Cycles × 100%
        SpecDesign Section 1.4.10.
        """
        availabilities = [
            r.system_availability
            for r in results
            if r.system_availability is not None
        ]
        return self.analyzer.descriptive_stats(availabilities)

    def compute_urgency_stratified_delivery(
        self, results: list[SimulationResults],
    ) -> dict[str, DescriptiveStats]:
        """Delivery rate stratified by urgency level (H/M/L).

        SpecDesign Section 1.4.10.
        """
        urgency_levels = ["H", "M", "L"]
        per_urgency_rates: dict[str, list[float]] = {u: [] for u in urgency_levels}

        for result in results:
            # Assignments per urgency (requires urgency field in TASK_ASSIGNED events)
            assigned_by_urgency: dict[str, set[str]] = {u: set() for u in urgency_levels}
            for event in result.events:
                if event.event_type == SimulationEventType.TASK_ASSIGNED:
                    urgency = event.data.get("urgency")
                    task_id = event.data.get("task_id")
                    if urgency and task_id and urgency in urgency_levels:
                        assigned_by_urgency[urgency].add(task_id)

            # Tasks delivered (from delivery_times already in results)
            delivered_task_ids = {task_id for task_id, _ in result.delivery_times}

            for urgency in urgency_levels:
                assigned = assigned_by_urgency[urgency]
                if len(assigned) > 0:
                    delivered = len(assigned & delivered_task_ids)
                    per_urgency_rates[urgency].append(delivered / len(assigned))

        return {
            u: self.analyzer.descriptive_stats(rates)
            for u, rates in per_urgency_rates.items()
            if rates  # only include urgency levels with data
        }

    def _generate_summary(self) -> dict:
        """Generate summary statistics."""
        adaptive = [r for r in self.results if r.algorithm == AlgorithmType.ADAPTIVE]
        baseline = [r for r in self.results if r.algorithm == AlgorithmType.BASELINE]

        return {
            "total_runs": len(self.results),
            "adaptive_runs": len(adaptive),
            "baseline_runs": len(baseline),
            "connectivity_levels": sorted(
                set(r.connectivity_level for r in self.results)
            ),
            "algorithms": ["adaptive", "baseline"],
        }

    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        print("=" * 60)
        print("PERFORMANCE EVALUATION SUMMARY")
        print("=" * 60)

        # Overall comparison
        for metric in [MetricType.DELIVERY_RATE, MetricType.DECISION_TIME]:
            print(f"\n{metric.value.upper()}")
            print("-" * 40)

            comparison = self.compare_algorithms(metric)
            print(
                f"  Adaptive: {comparison.adaptive_stats.mean:.4f} "
                f"(±{comparison.adaptive_stats.std:.4f})"
            )
            print(
                f"  Baseline: {comparison.baseline_stats.mean:.4f} "
                f"(±{comparison.baseline_stats.std:.4f})"
            )
            print(f"  Improvement: {comparison.improvement:+.2f}%")
            print(f"  {comparison.ttest.summary()}")

        print("\n" + "=" * 60)


def evaluate_results(
    results: list[SimulationResults],
    print_summary: bool = True,
) -> EvaluationReport:
    """
    Convenience function to evaluate simulation results.

    Args:
        results: List of simulation results
        print_summary: Whether to print summary to console

    Returns:
        EvaluationReport

    Example:
        >>> from ercs.simulation import run_experiment
        >>> from ercs.evaluation import evaluate_results
        >>> results = run_experiment(runs_per_config=5)
        >>> report = evaluate_results(results)
    """
    evaluator = PerformanceEvaluator(results)

    if print_summary:
        evaluator.print_summary()

    return evaluator.generate_report()
