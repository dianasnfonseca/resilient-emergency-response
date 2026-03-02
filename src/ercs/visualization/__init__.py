"""Visualization module for ERCS experiment results."""

from ercs.visualization.plots import (
    ALGORITHM_LABELS,
    COLORS,
    CONNECTIVITY_COLORS,
    CONNECTIVITY_LABELS,
    apply_thesis_style,
    build_anova_table,
    build_parameter_tables,
    build_results_dataframe,
    build_ttest_table,
    compute_summary_stats,
    plot_box_distributions,
    plot_degradation_lines,
    plot_grouped_bars,
    plot_heatmap,
    save_figure,
)
from ercs.visualization.animation import (
    AnimationEngine,
    FrameData,
    create_animation,
    run_paired_simulation,
)

__all__ = [
    "ALGORITHM_LABELS",
    "COLORS",
    "CONNECTIVITY_COLORS",
    "CONNECTIVITY_LABELS",
    "apply_thesis_style",
    "build_anova_table",
    "build_parameter_tables",
    "build_results_dataframe",
    "build_ttest_table",
    "compute_summary_stats",
    "plot_box_distributions",
    "plot_degradation_lines",
    "plot_grouped_bars",
    "plot_heatmap",
    "save_figure",
    # Animation
    "AnimationEngine",
    "FrameData",
    "create_animation",
    "run_paired_simulation",
]
