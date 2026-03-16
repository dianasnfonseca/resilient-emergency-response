"""
Performance Evaluation (Phase 6).

This module provides statistical analysis tools for comparing
coordination algorithm performance across experimental conditions.

Classes:
    PerformanceEvaluator: High-level evaluation interface
    StatisticalAnalyzer: Statistical tests (t-test, ANOVA)
    MetricExtractor: Extracts metrics from results
    DescriptiveStats: Descriptive statistics container
    TTestResult: T-test result container
    ANOVAResult: ANOVA result container
    ComparisonResult: Algorithm comparison result
    EvaluationReport: Complete evaluation report

Enums:
    MetricType: Types of performance metrics

Factory Functions:
    evaluate_results: Convenience function for evaluation

"""

from ercs.evaluation.metrics import (
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

__all__ = [
    "ANOVAResult",
    "ComparisonResult",
    "DescriptiveStats",
    "EvaluationReport",
    "MetricExtractor",
    "MetricType",
    "PerformanceEvaluator",
    "StatisticalAnalyzer",
    "TTestResult",
    "evaluate_results",
]
