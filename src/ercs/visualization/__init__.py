"""Visualization module for ERCS experiment results."""

from ercs.visualization.animation import (
    AnimationEngine,
    ForwardingEntry,
    FrameData,
    create_animation,
    run_paired_simulation,
)
from ercs.visualization.diagnostics import (
    find_message_journeys,
    plot_all_message_paths,
    plot_message_journey,
    plot_predictability_evolution,
    plot_predictability_graph,
    plot_predictability_heatmap,
)
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

__all__ = [
    "ALGORITHM_LABELS",
    "COLORS",
    "CONNECTIVITY_COLORS",
    "CONNECTIVITY_LABELS",
    # Animation
    "AnimationEngine",
    "ForwardingEntry",
    "FrameData",
    "apply_thesis_style",
    "build_anova_table",
    "build_parameter_tables",
    "build_results_dataframe",
    "build_ttest_table",
    "compute_summary_stats",
    "create_animation",
    # Diagnostics
    "find_message_journeys",
    "plot_all_message_paths",
    "plot_box_distributions",
    "plot_degradation_lines",
    "plot_grouped_bars",
    "plot_heatmap",
    "plot_message_journey",
    "plot_predictability_evolution",
    "plot_predictability_graph",
    "plot_predictability_heatmap",
    "run_paired_simulation",
    "save_figure",
]
