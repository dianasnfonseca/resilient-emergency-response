"""
Shared visualization functions for ERCS experiment results.

Provides thesis-quality matplotlib figures used by both the Streamlit
dashboard and the Jupyter notebook. All plot functions return
matplotlib Figure objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ercs.evaluation.metrics import EvaluationReport
    from ercs.simulation.engine import SimulationResults

from ercs.config.parameters import SimulationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = {
    "adaptive": "#2171B5",
    "baseline": "#CB181D",
}

CONNECTIVITY_COLORS = {
    0.75: "#2CA02C",
    0.40: "#FF7F0E",
    0.20: "#D62728",
}

CONNECTIVITY_LABELS = {0.75: "75%", 0.40: "40%", 0.20: "20%"}
ALGORITHM_LABELS = {"adaptive": "Adaptive", "baseline": "Baseline"}

METRICS_CONFIG = {
    "delivery_rate": {
        "ylabel": "Message Delivery Rate",
        "title": "Delivery Rate",
        "fmt": ".3f",
        "pct": True,
    },
    "assignment_rate": {
        "ylabel": "Task Assignment Rate",
        "title": "Assignment Rate",
        "fmt": ".3f",
        "pct": True,
    },
    "avg_decision_time": {
        "ylabel": "Mean Decision Time (s)",
        "title": "Decision Time",
        "fmt": ".1f",
        "pct": False,
    },
    "avg_delivery_time": {
        "ylabel": "Mean Delivery Time (s)",
        "title": "Delivery Time",
        "fmt": ".1f",
        "pct": False,
    },
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def apply_thesis_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.framealpha": 0.9,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#E0E0E0",
            "mathtext.fontset": "dejavuserif",
        }
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def build_results_dataframe(results: list[SimulationResults]) -> pd.DataFrame:
    """Convert simulation results into a pandas DataFrame."""
    records = []
    for r in results:
        records.append(
            {
                "algorithm": r.algorithm.value,
                "connectivity": r.connectivity_level,
                "run": r.run_number,
                "delivery_rate": r.delivery_rate,
                "assignment_rate": r.assignment_rate,
                "avg_decision_time": r.average_decision_time,
                "avg_delivery_time": r.average_delivery_time,
                "total_tasks": r.total_tasks,
                "tasks_assigned": r.tasks_assigned,
                "messages_created": r.messages_created,
                "messages_delivered": r.messages_delivered,
                "messages_expired": r.messages_expired,
            }
        )
    return pd.DataFrame(records)


def compute_summary_stats(
    df: pd.DataFrame,
    metric_col: str,
) -> dict[str, dict[float, dict[str, float]]]:
    """Compute mean and 95% CI for each algorithm x connectivity."""
    summary: dict[str, dict[float, dict[str, float]]] = {}
    for alg in ["adaptive", "baseline"]:
        summary[alg] = {}
        for conn in sorted(df["connectivity"].unique()):
            values = df[(df["algorithm"] == alg) & (df["connectivity"] == conn)][
                metric_col
            ].dropna()
            n = len(values)
            if n == 0:
                summary[alg][conn] = {"mean": 0.0, "ci": 0.0, "std": 0.0, "n": 0}
                continue
            mean = values.mean()
            std = values.std(ddof=1) if n > 1 else 0.0
            se = std / np.sqrt(n)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
            ci = t_crit * se
            summary[alg][conn] = {"mean": mean, "ci": ci, "std": std, "n": n}
    return summary


def build_parameter_tables(
    config: SimulationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build parameter display tables from config."""
    config = config or SimulationConfig()

    network_data = [
        (
            "Node count",
            f"{config.total_nodes} ({config.network.coordination_node_count} coordination + {config.network.mobile_responder_count} mobile)",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Simulation area",
            f"{config.network.simulation_area.width_m:.0f} x {config.network.simulation_area.height_m:.0f} m\u00b2",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Incident zone",
            f"{config.network.incident_zone.width_m:.0f} x {config.network.incident_zone.height_m:.0f} m\u00b2",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Radio range",
            f"{config.network.radio_range_m:.0f} m",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Buffer size",
            f"{config.network.buffer_size_bytes / 1_048_576:.0f} MB",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Message size",
            f"{config.network.message_size_bytes / 1_000:.0f} kB",
            "Kumar et al. (2023)",
        ),
        (
            "Connectivity scenarios",
            ", ".join(f"{c*100:.0f}%" for c in config.network.connectivity_scenarios),
            "Karaman et al. (2026)",
        ),
        (
            "Mobility model",
            config.network.mobility_model.value.replace("_", " ").title(),
            "Ullah & Qayyum (2022)",
        ),
        (
            "Speed range",
            f"{config.network.speed_min_mps:.0f}\u2013{config.network.speed_max_mps:.0f} m/s",
            "Ullah & Qayyum (2022)",
        ),
    ]

    prophet_data = [
        (
            "P_enc_max",
            f"{config.communication.prophet.p_enc_max}",
            "Grasic et al. (2011)",
        ),
        (
            "I_typ",
            f"{config.communication.prophet.i_typ:.0f} s",
            "Grasic et al. (2011)",
        ),
        (
            "\u03b2 (transitivity)",
            f"{config.communication.prophet.beta}",
            "Grasic et al. (2011)",
        ),
        (
            "\u03b3 (aging)",
            f"{config.communication.prophet.gamma}",
            "Grasic et al. (2011)",
        ),
        (
            "Message TTL",
            f"{config.communication.message_ttl_seconds // 60} min",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Transmit speed",
            f"{config.communication.transmit_speed_bps / 1_000_000:.0f} Mbps",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Buffer drop policy",
            config.communication.buffer_drop_policy.value.replace("_", " ").title(),
            "Ullah & Qayyum (2022)",
        ),
    ]

    scenario_data = [
        (
            "Task arrival",
            config.scenario.message_generation_model.title(),
            "Pu et al. (2025)",
        ),
        (
            "Message rate",
            f"{config.scenario.message_rate_per_minute} msgs/min",
            "Kumar et al. (2023)",
        ),
        (
            "Urgency distribution",
            f"{config.scenario.urgency_distribution.high*100:.0f}% H / {config.scenario.urgency_distribution.medium*100:.0f}% M / {config.scenario.urgency_distribution.low*100:.0f}% L",
            "Li et al. (2025)",
        ),
        (
            "Simulation duration",
            f"{config.scenario.simulation_duration_seconds} s ({config.scenario.simulation_duration_seconds // 60} min)",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Runs per configuration",
            f"{config.scenario.runs_per_configuration}",
            "Law (2015)",
        ),
    ]

    coordination_data = [
        (
            "Update interval",
            f"{config.coordination.update_interval_seconds // 60} min",
            "Kaji et al. (2025)",
        ),
        (
            "Priority levels",
            f"{config.coordination.priority_levels}",
            "Rosas et al. (2023)",
        ),
        (
            "Path threshold",
            f"P > {config.coordination.available_path_threshold}",
            "Ullah & Qayyum (2022)",
        ),
        (
            "Adaptive ordering",
            config.coordination.adaptive_task_order.replace("_", " ").title(),
            "Kaji et al. (2025)",
        ),
        (
            "Baseline ordering",
            config.coordination.baseline_task_order.upper(),
            "Design decision",
        ),
    ]

    tables = {}
    for name, data in [
        ("Network Topology", network_data),
        ("PRoPHET Protocol", prophet_data),
        ("Scenario Generation", scenario_data),
        ("Coordination", coordination_data),
    ]:
        tables[name] = pd.DataFrame(data, columns=["Parameter", "Value", "Source"])

    return tables


def build_ttest_table(report: EvaluationReport) -> pd.DataFrame:
    """Build a formatted t-test results DataFrame from an EvaluationReport."""
    records = []
    for comp in report.comparisons:
        records.append(
            {
                "Metric": comp.metric.value.replace("_", " ").title(),
                "Connectivity": (
                    f"{comp.connectivity_level * 100:.0f}%"
                    if comp.connectivity_level is not None
                    else "Overall"
                ),
                "Adaptive (mean \u00b1 std)": f"{comp.adaptive_stats.mean:.4f} \u00b1 {comp.adaptive_stats.std:.4f}",
                "Baseline (mean \u00b1 std)": f"{comp.baseline_stats.mean:.4f} \u00b1 {comp.baseline_stats.std:.4f}",
                "Improvement": f"{comp.improvement:+.2f}%",
                "t": f"{comp.ttest.t_statistic:.3f}",
                "p-value": f"{comp.ttest.p_value:.4f}",
                "Cohen's d": f"{comp.ttest.cohens_d:.3f}",
                "Sig.": "Yes" if comp.ttest.significant else "No",
            }
        )
    return pd.DataFrame(records)


def build_anova_table(report: EvaluationReport) -> pd.DataFrame:
    """Build a formatted ANOVA results DataFrame from an EvaluationReport."""
    records = []
    for key, anova in report.anova_results.items():
        parts = key.rsplit("_", 1)
        metric_name = parts[0].replace("_", " ").title()
        algorithm = parts[1].capitalize() if len(parts) > 1 else "All"
        records.append(
            {
                "Metric": metric_name,
                "Algorithm": algorithm,
                "F": f"{anova.f_statistic:.3f}",
                "p-value": f"{anova.p_value:.4f}",
                "df": f"({anova.df_between}, {anova.df_within})",
                "\u03b7\u00b2": f"{anova.eta_squared:.3f}",
                "Effect": anova._interpret_effect_size(),
                "Sig.": "Yes" if anova.significant else "No",
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_grouped_bars(
    summary: dict[str, dict[float, dict[str, float]]],
    metric_key: str,
) -> Figure:
    """
    Grouped bar chart comparing Adaptive vs Baseline across connectivity levels.

    Args:
        summary: Output of compute_summary_stats()
        metric_key: Key in METRICS_CONFIG
    """
    cfg = METRICS_CONFIG[metric_key]
    conn_levels = sorted(summary["adaptive"].keys(), reverse=True)
    x = np.arange(len(conn_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    adaptive_means = [summary["adaptive"][c]["mean"] for c in conn_levels]
    baseline_means = [summary["baseline"][c]["mean"] for c in conn_levels]
    adaptive_cis = [summary["adaptive"][c]["ci"] for c in conn_levels]
    baseline_cis = [summary["baseline"][c]["ci"] for c in conn_levels]

    bars1 = ax.bar(
        x - width / 2,
        adaptive_means,
        width,
        yerr=adaptive_cis,
        capsize=3,
        label="Adaptive",
        color=COLORS["adaptive"],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"linewidth": 1},
    )
    bars2 = ax.bar(
        x + width / 2,
        baseline_means,
        width,
        yerr=baseline_cis,
        capsize=3,
        label="Baseline",
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=0.5,
        error_kw={"linewidth": 1},
    )

    ax.set_xlabel("Connectivity Level")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{cfg['title']}: Adaptive vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONNECTIVITY_LABELS.get(c, f"{c*100:.0f}%") for c in conn_levels]
    )
    ax.legend()

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            fmt = cfg["fmt"]
            ax.annotate(
                f"{height:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    return fig


def plot_box_distributions(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> Figure:
    """
    Box plots showing distribution of each metric by algorithm x connectivity.

    Args:
        df: Results DataFrame from build_results_dataframe()
        metrics: List of metric column names (default: all 3 core metrics)
    """
    metrics = metrics or list(METRICS_CONFIG.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(5 * n_metrics, 5), constrained_layout=True
    )
    if n_metrics == 1:
        axes = [axes]

    conn_levels = sorted(df["connectivity"].unique(), reverse=True)

    for ax, metric_key in zip(axes, metrics, strict=False):
        cfg = METRICS_CONFIG[metric_key]
        data_groups = []
        labels = []
        colors_list = []

        for conn in conn_levels:
            for alg in ["adaptive", "baseline"]:
                values = (
                    df[(df["algorithm"] == alg) & (df["connectivity"] == conn)][
                        metric_key
                    ]
                    .dropna()
                    .values
                )
                data_groups.append(values)
                labels.append(
                    f"{ALGORITHM_LABELS[alg][:4]}.\n{CONNECTIVITY_LABELS.get(conn, f'{conn*100:.0f}%')}"
                )
                colors_list.append(COLORS[alg])

        bp = ax.boxplot(
            data_groups,
            patch_artist=True,
            labels=labels,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1},
            capprops={"linewidth": 1},
        )

        for patch, color in zip(bp["boxes"], colors_list, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(cfg["title"])
        ax.set_ylabel(cfg["ylabel"])
        ax.tick_params(axis="x", labelsize=9)

    return fig


def plot_heatmap(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> Figure:
    """
    Heatmap showing mean metric values (algorithm x connectivity).

    Args:
        df: Results DataFrame from build_results_dataframe()
        metrics: List of metric column names
    """
    metrics = metrics or list(METRICS_CONFIG.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(5 * n_metrics, 3), constrained_layout=True
    )
    if n_metrics == 1:
        axes = [axes]

    algorithms = ["adaptive", "baseline"]
    conn_levels = sorted(df["connectivity"].unique(), reverse=True)

    for ax, metric_key in zip(axes, metrics, strict=False):
        cfg = METRICS_CONFIG[metric_key]
        matrix = np.zeros((len(algorithms), len(conn_levels)))

        for i, alg in enumerate(algorithms):
            for j, conn in enumerate(conn_levels):
                values = df[(df["algorithm"] == alg) & (df["connectivity"] == conn)][
                    metric_key
                ].dropna()
                matrix[i, j] = values.mean() if len(values) > 0 else 0.0

        # For time metrics, lower is better -> non-reversed colormap
        if metric_key in ("avg_decision_time", "avg_delivery_time"):
            cmap = "YlOrRd"
        else:
            cmap = "YlOrRd_r"

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        # Annotate cells
        for i in range(len(algorithms)):
            for j in range(len(conn_levels)):
                fmt = cfg["fmt"]
                text_color = (
                    "white"
                    if matrix[i, j] > (matrix.max() + matrix.min()) / 2
                    else "black"
                )
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:{fmt}}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

        ax.set_xticks(range(len(conn_levels)))
        ax.set_xticklabels(
            [CONNECTIVITY_LABELS.get(c, f"{c*100:.0f}%") for c in conn_levels]
        )
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels([ALGORITHM_LABELS[a] for a in algorithms])
        ax.set_title(cfg["title"])
        fig.colorbar(im, ax=ax, shrink=0.8)

    return fig


def plot_degradation_lines(
    summary: dict[str, dict[str, dict[float, dict[str, float]]]],
    metrics: list[str] | None = None,
) -> Figure:
    """
    Line plots showing metric degradation as connectivity decreases.

    Args:
        summary: Dict of {metric_key: compute_summary_stats() output}
        metrics: List of metric keys to plot
    """
    metrics = metrics or list(METRICS_CONFIG.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(5 * n_metrics, 5), constrained_layout=True
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_key in zip(axes, metrics, strict=False):
        cfg = METRICS_CONFIG[metric_key]
        data = summary[metric_key]
        conn_levels = sorted(data["adaptive"].keys(), reverse=True)
        x = [c * 100 for c in conn_levels]

        for alg in ["adaptive", "baseline"]:
            means = [data[alg][c]["mean"] for c in conn_levels]
            cis = [data[alg][c]["ci"] for c in conn_levels]
            lower = [m - ci for m, ci in zip(means, cis, strict=False)]
            upper = [m + ci for m, ci in zip(means, cis, strict=False)]

            ax.plot(
                x,
                means,
                "o-",
                color=COLORS[alg],
                label=ALGORITHM_LABELS[alg],
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(x, lower, upper, color=COLORS[alg], alpha=0.15)

        ax.set_xlabel("Connectivity Level (%)")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"])
        ax.set_xticks(x)
        ax.legend()
        ax.invert_xaxis()

    return fig


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------


def save_figure(
    fig: Figure, filename: str, output_dir: str | Path = "outputs/figures"
) -> Path:
    """Save a figure to PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path
