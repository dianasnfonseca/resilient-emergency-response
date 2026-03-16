#!/usr/bin/env python3
"""
Diagnostic: verify PRoPHET warm-up period effectiveness.

Produces three analyses:
1. Predictability distribution after warm-up (histogram)
2. Quick Adaptive vs Baseline comparison across connectivity levels
3. Text summary with key statistics

Usage:
    python scripts/diagnose_warmup.py
    python scripts/diagnose_warmup.py --runs 5 --save
    python scripts/diagnose_warmup.py --warmup 900 --connectivity 0.20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import (
    AlgorithmType,
    CoordinationParameters,
    ScenarioParameters,
    SimulationConfig,
)
from ercs.simulation.engine import SimulationEngine, SimulationEventType

# ---------------------------------------------------------------------------
# 1. Warm-up inspection: P-value distribution
# ---------------------------------------------------------------------------


class WarmupInspector(SimulationEngine):
    """SimulationEngine that captures predictability state at warm-up end."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_p_values: list[float] = []
        self.warmup_coverage = 0.0
        self.warmup_nonzero = 0
        self.warmup_total = 0

    def _handle_warmup_end(self, event, results):
        """Capture P values at warm-up end before logging."""
        coord_nodes = self._topology.get_coordination_node_ids()
        mobile_nodes = self._topology.get_mobile_responder_ids()

        for coord_id in coord_nodes:
            for mobile_id in mobile_nodes:
                p = self._communication.get_delivery_predictability(coord_id, mobile_id)
                self.warmup_p_values.append(p)

        self.warmup_total = len(self.warmup_p_values)
        self.warmup_nonzero = sum(1 for p in self.warmup_p_values if p > 0)
        self.warmup_coverage = (
            self.warmup_nonzero / self.warmup_total * 100
            if self.warmup_total > 0
            else 0
        )

        super()._handle_warmup_end(event, results)


def inspect_warmup(config, connectivity, seed):
    """Run one simulation and return P-value statistics at warm-up end."""
    engine = WarmupInspector(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=connectivity,
        random_seed=seed,
    )
    results = engine.run()
    return engine, results


# ---------------------------------------------------------------------------
# 2. Quick comparison: Adaptive vs Baseline
# ---------------------------------------------------------------------------


def quick_comparison(config, connectivity_levels, runs_per_config, base_seed):
    """Run a quick experiment and return per-cell summary stats."""
    rows = []
    total = len(connectivity_levels) * 2 * runs_per_config
    done = 0

    for conn in connectivity_levels:
        for algo in [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]:
            delivery_rates = []
            assignment_rates = []
            response_times = []

            for run in range(runs_per_config):
                seed = base_seed + run
                engine = SimulationEngine(
                    config=config,
                    algorithm_type=algo,
                    connectivity_level=conn,
                    random_seed=seed,
                )
                r = engine.run(run_number=run)
                delivery_rates.append(r.delivery_rate)
                assignment_rates.append(r.assignment_rate)
                if r.average_decision_time is not None:
                    response_times.append(r.average_decision_time)

                done += 1
                pct = done / total * 100
                print(
                    f"\r  Progress: {done}/{total} ({pct:.0f}%) "
                    f"— {algo.value} @ {conn:.0%}",
                    end="",
                    flush=True,
                )

            rows.append(
                {
                    "algorithm": algo.value,
                    "connectivity": conn,
                    "delivery_rate_mean": np.mean(delivery_rates),
                    "delivery_rate_std": (
                        np.std(delivery_rates, ddof=1) if len(delivery_rates) > 1 else 0
                    ),
                    "assignment_rate_mean": np.mean(assignment_rates),
                    "assignment_rate_std": (
                        np.std(assignment_rates, ddof=1)
                        if len(assignment_rates) > 1
                        else 0
                    ),
                    "response_time_mean": (
                        np.mean(response_times) if response_times else None
                    ),
                    "n": runs_per_config,
                }
            )

    print()  # newline after progress
    return rows


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------


def plot_predictability_histogram(p_values, connectivity, warmup_s, save_path=None):
    """Plot histogram of P(coord, mobile) values after warm-up."""
    import matplotlib.pyplot as plt

    from ercs.visualization.plots import apply_thesis_style

    apply_thesis_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full distribution including zeros
    ax1.hist(p_values, bins=30, color="#2171B5", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Delivery Predictability P(coord, mobile)")
    ax1.set_ylabel("Number of Node Pairs")
    ax1.set_title(f"All Coord\u2192Mobile Pairs (n={len(p_values)})")

    nonzero = [p for p in p_values if p > 0]
    zero_count = len(p_values) - len(nonzero)
    ax1.axvline(0, color="#CB181D", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"{zero_count} pairs at P=0\n({zero_count/len(p_values)*100:.0f}%)",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    # Right: nonzero only (zoomed in)
    if nonzero:
        ax2.hist(nonzero, bins=25, color="#2CA02C", edgecolor="white", alpha=0.85)
        ax2.set_xlabel("Delivery Predictability P(coord, mobile)")
        ax2.set_ylabel("Number of Node Pairs")
        ax2.set_title(f"Nonzero Only (n={len(nonzero)})")

        # Stats annotation
        stats_text = (
            f"mean = {np.mean(nonzero):.3f}\n"
            f"std  = {np.std(nonzero):.3f}\n"
            f"min  = {min(nonzero):.3f}\n"
            f"max  = {max(nonzero):.3f}"
        )
        ax2.annotate(
            stats_text,
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "lightyellow",
                "alpha": 0.8,
            },
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No nonzero P values",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="red",
        )

    fig.suptitle(
        f"PRoPHET Predictability After {warmup_s/60:.0f}-min Warm-Up "
        f"(connectivity={connectivity:.0%})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_algorithm_comparison(rows, warmup_s, runs, save_path=None):
    """Plot Adaptive vs Baseline delivery rate at each connectivity level."""
    import matplotlib.pyplot as plt

    from ercs.visualization.plots import COLORS, apply_thesis_style

    apply_thesis_style()

    conn_levels = sorted({r["connectivity"] for r in rows}, reverse=True)
    x = np.arange(len(conn_levels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in [
        (ax1, "delivery_rate", "Message Delivery Rate"),
        (ax2, "assignment_rate", "Task Assignment Rate"),
    ]:
        adaptive_means = []
        adaptive_errs = []
        baseline_means = []
        baseline_errs = []

        for conn in conn_levels:
            for r in rows:
                if r["connectivity"] == conn and r["algorithm"] == "adaptive":
                    adaptive_means.append(r[f"{metric}_mean"])
                    adaptive_errs.append(r[f"{metric}_std"])
                elif r["connectivity"] == conn and r["algorithm"] == "baseline":
                    baseline_means.append(r[f"{metric}_mean"])
                    baseline_errs.append(r[f"{metric}_std"])

        bars_a = ax.bar(
            x - width / 2,
            adaptive_means,
            width,
            yerr=adaptive_errs,
            capsize=4,
            label="Adaptive",
            color=COLORS["adaptive"],
            alpha=0.85,
        )
        bars_b = ax.bar(
            x + width / 2,
            baseline_means,
            width,
            yerr=baseline_errs,
            capsize=4,
            label="Baseline",
            color=COLORS["baseline"],
            alpha=0.85,
        )

        ax.set_xlabel("Connectivity Level")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.0%}" for c in conn_levels])
        ax.set_ylim(0, 1.05)
        ax.legend()

        # Add value labels
        for bars in [bars_a, bars_b]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.annotate(
                        f"{h:.1%}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    fig.suptitle(
        f"Adaptive vs Baseline ({warmup_s/60:.0f}-min Warm-Up, n={runs}/config)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose PRoPHET warm-up period effectiveness"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1800,
        help="Warm-up duration in seconds (default: 1800)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=6000,
        help="Active simulation duration in seconds (default: 6000)",
    )
    parser.add_argument(
        "--connectivity",
        type=float,
        nargs="*",
        default=[0.75, 0.40, 0.20],
        help="Connectivity levels to test (default: 0.75 0.40 0.20)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per configuration for comparison (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to outputs/figures/",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plots (text output only)",
    )
    args = parser.parse_args()

    config = SimulationConfig(
        scenario=ScenarioParameters(
            warmup_period_seconds=args.warmup,
            simulation_duration_seconds=args.duration,
        ),
        coordination=CoordinationParameters(
            update_interval_seconds=1800,
        ),
    )

    # ===== Part 1: Warm-up P-value inspection =====
    print("=" * 65)
    print("  PART 1: Predictability Distribution After Warm-Up")
    print("=" * 65)

    inspect_conn = args.connectivity[0]
    print(
        f"\n  Running single simulation (warmup={args.warmup}s, "
        f"connectivity={inspect_conn:.0%}, seed={args.seed})..."
    )

    engine, results = inspect_warmup(config, inspect_conn, args.seed)
    p_values = engine.warmup_p_values

    print(f"\n  Coord-Mobile pairs:    {engine.warmup_total}")
    print(
        f"  Nonzero P values:      {engine.warmup_nonzero} "
        f"({engine.warmup_coverage:.1f}%)"
    )

    if p_values:
        nonzero = [p for p in p_values if p > 0]
        print("\n  All P values:")
        print(f"    mean = {np.mean(p_values):.4f}")
        print(f"    std  = {np.std(p_values):.4f}")
        if nonzero:
            print(f"\n  Nonzero P values (n={len(nonzero)}):")
            print(f"    mean = {np.mean(nonzero):.4f}")
            print(f"    std  = {np.std(nonzero):.4f}")
            print(f"    min  = {min(nonzero):.4f}")
            print(f"    max  = {max(nonzero):.4f}")

        if np.std(p_values) > 0.01:
            print("\n  PASS: Predictability values show meaningful variation.")
        else:
            print("\n  WARNING: Low variation in P values — warm-up may be too short.")

    print("\n  Simulation results:")
    print(f"    Tasks created:      {results.total_tasks}")
    print(f"    Tasks assigned:     {results.tasks_assigned}")
    print(f"    Messages delivered: {results.messages_delivered}")
    print(f"    Delivery rate:      {results.delivery_rate:.1%}")
    print(f"    Assignment rate:    {results.assignment_rate:.1%}")

    # Check WARMUP_END event
    warmup_events = [
        e for e in results.events if e.event_type == SimulationEventType.WARMUP_END
    ]
    if warmup_events:
        data = warmup_events[0].data
        print(f"\n  WARMUP_END event at t={warmup_events[0].timestamp:.0f}s:")
        print(
            f"    Coverage: {data.get('coverage_pct', 0):.1f}% "
            f"({data.get('nonzero_predictabilities', 0)}/{data.get('total_pairs', 0)})"
        )

    # ===== Part 2: Algorithm comparison =====
    print(f"\n{'=' * 65}")
    print("  PART 2: Adaptive vs Baseline Comparison")
    print("=" * 65)
    print(
        f"\n  Config: {args.runs} runs/config, "
        f"connectivity={args.connectivity}, warmup={args.warmup}s\n"
    )

    rows = quick_comparison(config, args.connectivity, args.runs, args.seed)

    # Print comparison table
    print(
        f"\n  {'Algorithm':>10}  {'Conn':>5}  {'Delivery':>10}  "
        f"{'Assignment':>10}  {'Resp Time':>10}"
    )
    print(f"  {'—' * 10}  {'—' * 5}  {'—' * 10}  {'—' * 10}  {'—' * 10}")

    for r in rows:
        rt = f"{r['response_time_mean']:.0f}s" if r["response_time_mean"] else "N/A"
        print(
            f"  {r['algorithm']:>10}  {r['connectivity']:>5.0%}  "
            f"{r['delivery_rate_mean']:>9.1%}  "
            f"{r['assignment_rate_mean']:>9.1%}  "
            f"{rt:>10}"
        )

    # Differences
    print(f"\n  {'Conn':>5}  {'Delivery Diff':>14}  {'Assignment Diff':>16}")
    print(f"  {'—' * 5}  {'—' * 14}  {'—' * 16}")

    for conn in sorted({r["connectivity"] for r in rows}, reverse=True):
        a = next(
            r
            for r in rows
            if r["connectivity"] == conn and r["algorithm"] == "adaptive"
        )
        b = next(
            r
            for r in rows
            if r["connectivity"] == conn and r["algorithm"] == "baseline"
        )
        d_diff = a["delivery_rate_mean"] - b["delivery_rate_mean"]
        a_diff = a["assignment_rate_mean"] - b["assignment_rate_mean"]
        d_sign = "+" if d_diff >= 0 else ""
        a_sign = "+" if a_diff >= 0 else ""
        print(f"  {conn:>5.0%}  {d_sign}{d_diff:>12.1%}  {a_sign}{a_diff:>14.1%}")

    # ===== Part 3: Plots =====
    if not args.no_plot:
        print(f"\n{'=' * 65}")
        print("  PART 3: Generating Plots")
        print("=" * 65)

        save_dir = Path("outputs/figures") if args.save else None

        plot_predictability_histogram(
            p_values,
            inspect_conn,
            args.warmup,
            save_path=save_dir / "fig_warmup_predictability.png" if save_dir else None,
        )

        plot_algorithm_comparison(
            rows,
            args.warmup,
            args.runs,
            save_path=save_dir / "fig_warmup_comparison.png" if save_dir else None,
        )

        if not args.save:
            import matplotlib.pyplot as plt

            print("\n  Displaying plots (close windows to exit)...")
            plt.show()
        else:
            print("\n  Plots saved to outputs/figures/")


if __name__ == "__main__":
    main()
