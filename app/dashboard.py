"""
ERCS Experiment Dashboard — Streamlit Application.

Interactive dashboard for the Emergency Response Coordination Simulator.
Designed for a 10-minute video demonstration of the MSc dissertation project.

Run with: streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Ensure src is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.evaluation.metrics import MetricType, PerformanceEvaluator
from ercs.simulation.engine import ExperimentRunner
from ercs.visualization.animation import run_paired_simulation
from ercs.visualization.diagnostics import (
    find_message_journeys,
    plot_all_message_paths,
    plot_message_journey,
    plot_predictability_evolution,
    plot_predictability_graph,
    plot_predictability_heatmap,
)
from ercs.visualization.plots import (
    METRICS_CONFIG,
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ERCS Experiment Dashboard",
    page_icon="\U0001f6d1",
    layout="wide",
)

apply_thesis_style()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None
if "df" not in st.session_state:
    st.session_state.df = None
if "report" not in st.session_state:
    st.session_state.report = None
if "diag_frames" not in st.session_state:
    st.session_state.diag_frames = None
if "diag_fwd" not in st.session_state:
    st.session_state.diag_fwd = None

config = SimulationConfig()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ERCS Dashboard")
    st.caption(
        "Emergency Response Coordination Simulator\n\n"
        "MSc Computer Science -- University of Liverpool, 2026"
    )

    st.divider()

    quick_test = st.toggle("Quick test mode (5 runs)", value=True)
    runs = 5 if quick_test else config.scenario.runs_per_configuration

    total_runs = 2 * len(config.network.connectivity_scenarios) * runs
    st.metric("Total simulation runs", total_runs)

    st.divider()

    with st.expander("Network Topology"):
        st.markdown(f"""
        - **Nodes:** {config.total_nodes} ({config.network.coordination_node_count} coord + {config.network.mobile_responder_count} mobile)
        - **Area:** {config.network.simulation_area.width_m:.0f} x {config.network.simulation_area.height_m:.0f} m
        - **Radio range:** {config.network.radio_range_m:.0f} m
        - **Buffer:** {config.network.buffer_size_bytes / 1_048_576:.0f} MB
        - **Mobility:** Role-based Random Waypoint
        """)

    with st.expander("PRoPHETv2 Protocol"):
        st.markdown(f"""
        - **P_enc_max:** {config.communication.prophet.p_enc_max}
        - **I_typ:** {config.communication.prophet.i_typ:.0f} s
        - **beta:** {config.communication.prophet.beta}
        - **gamma:** {config.communication.prophet.gamma}
        - **TTL:** {config.communication.message_ttl_seconds // 60} min
        """)

    with st.expander("Scenario"):
        st.markdown(f"""
        - **Arrival:** {config.scenario.message_generation_model.title()} process
        - **Rate:** {config.scenario.message_rate_per_minute} msgs/min
        - **Urgency:** {config.scenario.urgency_distribution.high*100:.0f}% H / {config.scenario.urgency_distribution.medium*100:.0f}% M / {config.scenario.urgency_distribution.low*100:.0f}% L
        - **Duration:** {config.scenario.simulation_duration_seconds}s
        """)

    st.divider()

    st.markdown(
        "[GitHub Repository](https://github.com/dianasnfonseca/resilient-emergency-response)"
    )
    st.caption("Diana Fonseca")


# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tab_overview, tab_setup, tab_run, tab_viz, tab_diag, tab_stats, tab_findings = st.tabs(
    [
        "Overview",
        "Parameters",
        "Run Experiment",
        "Visualizations",
        "Network Diagnostics",
        "Statistical Analysis",
        "Key Findings",
    ]
)


# ---- Tab 0: Overview ----
with tab_overview:
    st.header("Emergency Response Coordination Simulator")
    st.subheader(
        "Resilient and Adaptive Scheduling Systems for Emergency Response "
        "in Low Connectivity Environments"
    )

    st.markdown("""
    This project investigates whether **adaptive, network-aware coordination**
    can outperform **simple proximity-based coordination** in emergency response
    scenarios where communication infrastructure is degraded or destroyed.

    The simulator models 50 emergency responder nodes operating in a disaster zone,
    generating tasks via Poisson arrivals, routing assignment messages through a
    **PRoPHETv2 delay-tolerant network**, and comparing two coordination algorithms
    across three connectivity degradation scenarios.
    """)

    st.divider()

    # Research questions
    st.subheader("Research Questions")

    rq_col1, rq_col2 = st.columns([1, 1])
    with rq_col1:
        st.markdown("""
        **Main Research Question (MRQ)**

        Can adaptive scheduling algorithms integrated with delay-tolerant
        communication architectures improve emergency resource coordination
        effectiveness when operating under intermittent connectivity conditions?
        """)

        st.markdown("""
        **Sub-question 1 (SQ1)**

        How do distributed communication strategies combined with adaptive
        scheduling algorithms impact resource allocation effectiveness during
        varying levels of network disruption?
        """)

    with rq_col2:
        st.markdown("""
        **Sub-question 2 (SQ2)**

        What are the optimal trade-offs between system complexity, resilience,
        and performance when adapting centralised emergency coordination
        approaches for decentralised, low-connectivity environments?
        """)

    st.divider()

    # Architecture
    st.subheader("System Architecture")

    st.code(
        """
Phase 6  VISUALIZATION       plots, dashboard, notebook, animation, diagnostics
Phase 5  SIMULATION ENGINE   discrete-event queue, orchestration
Phase 4  COORDINATION        Adaptive (urgency-first, weighted scoring) vs Baseline (FCFS)
Phase 3  SCENARIOS           Poisson task arrivals, urgency distribution
Phase 2  COMMUNICATION       PRoPHETv2 store-and-forward, message buffers, predictability
Phase 1  NETWORK             two-zone topology, role-based Random Waypoint mobility
    """,
        language=None,
    )

    st.divider()

    # Key stats
    st.subheader("Experiment at a Glance")

    stat_cols = st.columns(6)
    stat_cols[0].metric("Nodes", "50")
    stat_cols[1].metric("Algorithms", "2")
    stat_cols[2].metric("Connectivity Levels", "3")
    stat_cols[3].metric("Runs per Cell", "30")
    stat_cols[4].metric("Total Runs", "180")
    stat_cols[5].metric("Tests", "411")

    st.divider()

    # Algorithm comparison
    st.subheader("The Two Algorithms")

    alg_col1, alg_col2 = st.columns(2)
    with alg_col1:
        st.markdown("""
        **Adaptive Coordinator**

        - Urgency-first task ordering (H > M > L)
        - PRoPHETv2 reachability filter (P > 0.3)
        - Weighted scoring: `0.2*P + 0.2*R + 0.6*D - 0.2*W`
        - Network-aware responder selection
        - Capacity bound (k_max)
        """)

    with alg_col2:
        st.markdown("""
        **Baseline Coordinator**

        - First-Come-First-Served task ordering
        - Nearest responder by Euclidean distance
        - No network state awareness
        - No eligibility filtering
        - No workload management
        """)


# ---- Tab 1: Parameters ----
with tab_setup:
    st.header("Experiment Parameters")
    st.markdown(
        "All parameters are sourced from published literature. "
        "See the sidebar for a compact summary."
    )

    tables = build_parameter_tables(config)
    cols = st.columns(2)
    for i, (name, table_df) in enumerate(tables.items()):
        with cols[i % 2]:
            st.subheader(name)
            st.dataframe(table_df, hide_index=True, use_container_width=True)

    st.divider()

    # Experimental design metrics
    st.subheader("Experimental Design")
    design_cols = st.columns(4)
    design_cols[0].metric("Algorithms", "2")
    design_cols[1].metric(
        "Connectivity Levels", str(len(config.network.connectivity_scenarios))
    )
    design_cols[2].metric("Runs per Config", str(runs))
    design_cols[3].metric("Total Runs", str(total_runs))

    st.divider()

    # Network topology visual
    st.subheader("Network Topology Layout")
    st.markdown("""
    The simulation area has two separated zones requiring multi-hop relay:

    - **Incident Zone** (orange): 700 x 600 m at origin (0, 450) -- where tasks occur
    - **Coordination Zone** (blue): 50 x 50 m at origin (800, 300) -- command post location
    - **Inter-zone gap**: ~100-141 m -- exceeds the 100 m radio range, forcing multi-hop routing
    """)

    # Draw topology diagram
    fig_topo, ax_topo = plt.subplots(figsize=(10, 5))
    iz = config.network.incident_zone
    cz = config.network.coordination_zone
    area = config.network.simulation_area

    ax_topo.add_patch(
        plt.Rectangle(
            (iz.origin_x, iz.origin_y),
            iz.width_m,
            iz.height_m,
            facecolor="#FFF3E0",
            edgecolor="#FF9800",
            linestyle="--",
            linewidth=2,
            label="Incident Zone (700 x 600 m)",
        )
    )
    ax_topo.add_patch(
        plt.Rectangle(
            (cz.origin_x, cz.origin_y),
            cz.width_m,
            cz.height_m,
            facecolor="#E8EAF6",
            edgecolor="#3F51B5",
            linestyle="--",
            linewidth=2,
            label="Coordination Zone (50 x 50 m)",
        )
    )

    # Label zones
    ax_topo.text(
        iz.origin_x + iz.width_m / 2,
        iz.origin_y + iz.height_m / 2,
        "INCIDENT ZONE\n~29 Rescue + ~7 Liaison",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#E65100",
    )
    ax_topo.text(
        cz.origin_x + cz.width_m / 2,
        cz.origin_y - 40,
        "COORD\nZONE\n2 fixed nodes",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#283593",
    )

    # Draw inter-zone gap annotation
    ax_topo.annotate(
        "",
        xy=(cz.origin_x, iz.origin_y),
        xytext=(iz.origin_x + iz.width_m, iz.origin_y),
        arrowprops={"arrowstyle": "<->", "color": "#D32F2F", "lw": 1.5},
    )
    ax_topo.text(
        (iz.origin_x + iz.width_m + cz.origin_x) / 2,
        iz.origin_y + 20,
        "100 m gap\n(> radio range)",
        ha="center",
        fontsize=8,
        color="#D32F2F",
    )

    # Transport shuttle arrow
    ax_topo.annotate(
        "~12 Transport\n(data mules)",
        xy=(cz.origin_x + cz.width_m / 2, cz.origin_y + cz.height_m),
        xytext=(iz.origin_x + iz.width_m - 50, iz.origin_y + iz.height_m - 50),
        fontsize=8,
        ha="center",
        color="#1B5E20",
        arrowprops={
            "arrowstyle": "->",
            "color": "#1B5E20",
            "lw": 1.5,
            "connectionstyle": "arc3,rad=0.3",
        },
    )

    ax_topo.set_xlim(-100, area.width_m + 100)
    ax_topo.set_ylim(0, area.height_m + 100)
    ax_topo.set_aspect("equal")
    ax_topo.set_xlabel("X (metres)")
    ax_topo.set_ylabel("Y (metres)")
    ax_topo.legend(loc="upper right", fontsize=9)
    ax_topo.set_title("Two-Zone Simulation Area", fontsize=14, fontweight="bold")
    fig_topo.tight_layout()
    st.pyplot(fig_topo)
    plt.close(fig_topo)

    st.divider()

    # Key parameter rationale
    with st.expander("Why P > 0.3 threshold?"):
        st.markdown("""
        P > 0.3 is the network-awareness mechanism. A responder whose delivery
        predictability is below this threshold is excluded from assignment
        eligibility by the Adaptive algorithm.

        Ullah & Qayyum (2022) established P > 0.25 as the minimum viable
        threshold for PRoPHET-based forwarding -- below this value, delivery
        latency increases sharply even when messages eventually arrive. The
        threshold is set at 0.30 given the 20% High-urgency task profile
        (Li et al., 2025), where the cost of failed delivery to a high-urgency
        task is disproportionate.
        """)

    with st.expander("Why cold-start (no warm-up)?"):
        st.markdown("""
        The simulation uses P(a,b)_0 = 0 for all node pairs, meaning PRoPHET
        predictability starts from zero and builds through actual encounters.
        This is the most conservative evaluation: the Adaptive algorithm must
        earn its network awareness through simulation dynamics rather than
        starting with pre-seeded knowledge.
        """)

    with st.expander("Why 30 runs per configuration?"):
        st.markdown("""
        Law (2015) recommends a minimum of 20-30 replications for stochastic
        simulation experiments to achieve stable confidence intervals. With 30
        runs per cell (2 algorithms x 3 connectivity levels = 6 cells), we run
        180 total simulations, providing robust statistical power for Welch's
        t-tests and ANOVA.
        """)


# ---- Tab 2: Run Experiment ----
with tab_run:
    st.header("Run Experiment")

    mode_label = (
        "Quick test (5 runs/config)"
        if quick_test
        else "Full experiment (30 runs/config)"
    )
    st.info(f"Mode: **{mode_label}** -- {total_runs} total simulations")

    if st.button("Start Experiment", type="primary", use_container_width=True):
        runner = ExperimentRunner(config=config, base_seed=42)
        algorithms = [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]
        connectivity_levels = config.network.connectivity_scenarios

        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def progress_callback(current: int, total: int) -> None:
            pct = current / total
            progress_bar.progress(pct)

            elapsed = time.time() - start_time
            eta = (elapsed / current) * (total - current) if current > 0 else 0

            alg_idx = (current - 1) // (len(connectivity_levels) * runs)
            remainder = (current - 1) % (len(connectivity_levels) * runs)
            conn_idx = remainder // runs
            run_idx = remainder % runs

            alg_name = algorithms[alg_idx].value.capitalize()
            conn_pct = f"{connectivity_levels[conn_idx] * 100:.0f}%"

            status_text.markdown(
                f"**{current}/{total}** -- {alg_name} @ {conn_pct} connectivity, "
                f"run {run_idx + 1}/{runs} -- "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
            )

        results = runner.run_all(
            algorithms=algorithms,
            connectivity_levels=connectivity_levels,
            runs_per_config=runs,
            progress_callback=progress_callback,
        )

        elapsed_total = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.markdown(
            f"**Complete!** {len(results)} runs in {elapsed_total:.1f}s "
            f"({elapsed_total / len(results):.2f}s/run)"
        )

        # Store in session state
        st.session_state.results = results
        st.session_state.df = build_results_dataframe(results)

        evaluator = PerformanceEvaluator(results)
        st.session_state.report = evaluator.generate_report(
            metrics=[
                MetricType.DELIVERY_RATE,
                MetricType.ASSIGNMENT_RATE,
                MetricType.DECISION_TIME,
                MetricType.DELIVERY_TIME,
            ]
        )

        st.success(f"Experiment completed: {len(results)} simulation runs")

    if st.session_state.df is not None:
        st.divider()

        # Headline metrics
        st.subheader("Headline Metrics")
        df = st.session_state.df

        headline_cols = st.columns(4)
        for i, (metric, label) in enumerate(
            [
                ("avg_delivery_time", "Avg Delivery Time (s)"),
                ("delivery_rate", "Delivery Rate"),
                ("assignment_rate", "Assignment Rate"),
                ("avg_decision_time", "Avg Decision Time (s)"),
            ]
        ):
            adaptive_mean = df[df["algorithm"] == "adaptive"][metric].mean()
            baseline_mean = df[df["algorithm"] == "baseline"][metric].mean()
            diff_pct = (
                (adaptive_mean - baseline_mean) / baseline_mean * 100
                if baseline_mean != 0
                else 0
            )
            headline_cols[i].metric(
                label,
                f"{adaptive_mean:.3f}",
                delta=f"{diff_pct:+.1f}% vs Baseline",
                delta_color="inverse" if metric == "avg_delivery_time" else "normal",
            )

        st.divider()

        st.subheader("Summary Statistics")
        summary_df = (
            df.groupby(["algorithm", "connectivity"])[
                [
                    "delivery_rate",
                    "assignment_rate",
                    "avg_decision_time",
                    "avg_delivery_time",
                ]
            ]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )
        st.dataframe(summary_df, use_container_width=True)


# ---- Tab 3: Visualizations ----
with tab_viz:
    st.header("Results Visualizations")

    if st.session_state.df is None:
        st.warning("Run the experiment first to generate visualizations.")
    else:
        df = st.session_state.df

        # Precompute summaries
        summaries = {}
        for metric_key in METRICS_CONFIG:
            summaries[metric_key] = compute_summary_stats(df, metric_key)

        # Metric explanation
        with st.expander("What do these metrics mean?", expanded=False):
            st.markdown("""
            - **avg_delivery_time** (MRQ, SQ1): Mean seconds from task creation to
              message delivery at the assigned responder. Lower = faster coordination.
              The gap between algorithms answers the MRQ. The pattern across
              connectivity levels answers SQ1.

            - **delivery_rate** (SQ2): Proportion of coordination messages that reach
              the assigned responder. The Adaptive algorithm's P > 0.3 filter withholds
              assignments below threshold. A lower delivery_rate for Adaptive reflects
              the trade-off SQ2 asks about: reduced coverage in exchange for higher
              per-assignment reliability.

            - **assignment_rate** (diagnostic): Set by task arrival and simulation
              parameters, not by algorithm decisions. Expected identical across
              algorithms.

            - **avg_decision_time** (diagnostic): Internal processing latency
              determined by the coordination cycle interval. Expected identical across
              algorithms.
            """)

        # Primary metric: avg_delivery_time
        st.subheader("Primary Metric: Coordination Response Time (avg_delivery_time)")
        st.caption("Answers MRQ and SQ1 -- lower is better")
        if "avg_delivery_time" in summaries:
            fig = plot_grouped_bars(summaries["avg_delivery_time"], "avg_delivery_time")
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # Remaining grouped bar charts
        st.subheader("Algorithm Comparison (Grouped Bar Charts)")
        remaining_metrics = [k for k in METRICS_CONFIG if k != "avg_delivery_time"]
        bar_cols = st.columns(min(len(remaining_metrics), 3))
        for i, metric_key in enumerate(remaining_metrics):
            with bar_cols[i % len(bar_cols)]:
                fig = plot_grouped_bars(summaries[metric_key], metric_key)
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

        # Box plots
        st.subheader("Distributions (Box Plots)")
        st.caption(
            "Shows spread and outliers for each algorithm x connectivity combination"
        )
        fig = plot_box_distributions(df)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # Heatmap
        st.subheader("Performance Heatmap")
        st.caption(
            "Color-coded mean values -- quick visual comparison across all conditions"
        )
        fig = plot_heatmap(df)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # Degradation lines
        st.subheader("Connectivity Degradation")
        st.caption(
            "How each metric changes as connectivity decreases -- "
            "shaded regions show 95% confidence intervals"
        )
        fig = plot_degradation_lines(summaries)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # Save all figures
        if st.button("Save All Figures to outputs/figures/"):
            output_dir = project_root / "outputs" / "figures"
            figures = {
                "fig_delivery_rate_bars": plot_grouped_bars(
                    summaries["delivery_rate"], "delivery_rate"
                ),
                "fig_assignment_rate_bars": plot_grouped_bars(
                    summaries["assignment_rate"], "assignment_rate"
                ),
                "fig_response_time_bars": plot_grouped_bars(
                    summaries["avg_decision_time"], "avg_decision_time"
                ),
                "fig_delivery_time_bars": plot_grouped_bars(
                    summaries["avg_delivery_time"], "avg_delivery_time"
                ),
                "fig_box_distributions": plot_box_distributions(df),
                "fig_heatmap": plot_heatmap(df),
                "fig_degradation_lines": plot_degradation_lines(summaries),
            }
            for name, fig in figures.items():
                save_figure(fig, name, output_dir)
                plt.close(fig)
            st.success(f"Saved {len(figures)} figures to `{output_dir}`")


# ---- Tab 4: Network Diagnostics ----
with tab_diag:
    st.header("Network Diagnostics")
    st.markdown(
        "Visualizations of the PRoPHET routing protocol, message forwarding, "
        "and node mobility from a single representative simulation run. "
        "These diagnostics reveal **how** the network behaves internally."
    )

    diag_col1, diag_col2 = st.columns([1, 3])
    with diag_col1:
        diag_connectivity = st.selectbox(
            "Connectivity level",
            options=config.network.connectivity_scenarios,
            format_func=lambda x: f"{x * 100:.0f}%",
            index=len(config.network.connectivity_scenarios) - 1,
        )

    if st.button("Generate Diagnostics", type="primary", use_container_width=True):
        with st.spinner(
            f"Running paired simulation at {diag_connectivity * 100:.0f}% connectivity..."
        ):
            start_diag = time.time()
            adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd = (
                run_paired_simulation(
                    config=config,
                    connectivity_level=diag_connectivity,
                    seed=42,
                    sample_interval=30.0,
                )
            )
            elapsed_diag = time.time() - start_diag

        st.session_state.diag_frames = (adaptive_frames, baseline_frames)
        st.session_state.diag_fwd = (adaptive_fwd, baseline_fwd)
        st.success(
            f"Diagnostics generated in {elapsed_diag:.1f}s "
            f"({len(adaptive_frames)} frames per algorithm)"
        )

    if st.session_state.diag_frames is not None:
        adaptive_frames, baseline_frames = st.session_state.diag_frames
        adaptive_fwd, baseline_fwd = st.session_state.diag_fwd

        # --- PRoPHET Predictability Graph ---
        st.divider()
        st.subheader("PRoPHET Predictability Graph")
        st.caption(
            "Network graph showing delivery predictability between nodes. "
            "Edge color and thickness indicate P-value strength. "
            "Brighter/thicker edges = higher probability of successful message delivery."
        )

        max_frame_idx = len(adaptive_frames) - 1
        snapshot_idx = st.slider(
            "Snapshot (frame index)",
            min_value=0,
            max_value=max_frame_idx,
            value=max_frame_idx // 2,
            key="pred_graph_slider",
        )

        pred_cols = st.columns(2)
        with pred_cols[0]:
            st.markdown("**Adaptive**")
            fig = plot_predictability_graph(
                adaptive_frames[snapshot_idx], config, algorithm_label="Adaptive"
            )
            st.pyplot(fig)
            plt.close(fig)
        with pred_cols[1]:
            st.markdown("**Baseline**")
            fig = plot_predictability_graph(
                baseline_frames[snapshot_idx], config, algorithm_label="Baseline"
            )
            st.pyplot(fig)
            plt.close(fig)

        # --- Predictability Heatmap ---
        st.divider()
        st.subheader("Predictability Heatmap")
        st.caption(
            "Coordination-to-mobile predictability matrix. Each cell shows how "
            "confident PRoPHETv2 is that a message from a coordination node will "
            "reach a specific mobile responder. The Adaptive algorithm uses these "
            "values for its P > 0.3 eligibility filter."
        )

        hm_cols = st.columns(2)
        with hm_cols[0]:
            fig = plot_predictability_heatmap(
                adaptive_frames[snapshot_idx], algorithm_label="Adaptive"
            )
            st.pyplot(fig)
            plt.close(fig)
        with hm_cols[1]:
            fig = plot_predictability_heatmap(
                baseline_frames[snapshot_idx], algorithm_label="Baseline"
            )
            st.pyplot(fig)
            plt.close(fig)

        # --- Predictability Evolution ---
        st.divider()
        st.subheader("Predictability Evolution")
        st.caption(
            "How predictability values build through encounters and decay over time. "
            "Shows the top 10 node pairs by peak P-value. The sawtooth pattern "
            "reflects encounters (sharp rises) followed by aging decay."
        )

        evo_cols = st.columns(2)
        with evo_cols[0]:
            fig = plot_predictability_evolution(
                adaptive_frames, algorithm_label="Adaptive"
            )
            st.pyplot(fig)
            plt.close(fig)
        with evo_cols[1]:
            fig = plot_predictability_evolution(
                baseline_frames, algorithm_label="Baseline"
            )
            st.pyplot(fig)
            plt.close(fig)

        # --- Message Journey ---
        st.divider()
        st.subheader("Message Journey")
        st.caption(
            "Hop-by-hop path of a message through the network. Left panel shows the "
            "spatial path; right panel shows the timeline of forwarding events. "
            "Green = delivered, blue = forwarded, red = failed."
        )

        for label, fwd_log, frames in [
            ("Adaptive", adaptive_fwd, adaptive_frames),
            ("Baseline", baseline_fwd, baseline_frames),
        ]:
            journeys = find_message_journeys(fwd_log)
            if journeys:
                msg_id = min(journeys.keys(), key=lambda m: journeys[m][0].timestamp)
                fig = plot_message_journey(
                    msg_id, journeys[msg_id], frames, config, algorithm_label=label
                )
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(f"No forwarding events recorded for {label}.")

        # --- All Message Paths ---
        st.divider()
        st.subheader("All Message Paths")
        st.caption(
            "Overview of all message routes in the simulation, colored by delivery "
            "status. Green = delivered, red = undelivered. Dense green regions indicate "
            "reliable routing corridors."
        )

        path_cols = st.columns(2)
        with path_cols[0]:
            fig = plot_all_message_paths(
                adaptive_frames, adaptive_fwd, config, algorithm_label="Adaptive"
            )
            st.pyplot(fig)
            plt.close(fig)
        with path_cols[1]:
            fig = plot_all_message_paths(
                baseline_frames, baseline_fwd, config, algorithm_label="Baseline"
            )
            st.pyplot(fig)
            plt.close(fig)

        # --- Save diagnostic figures ---
        st.divider()
        if st.button("Save Diagnostic Figures to outputs/figures/"):
            output_dir = project_root / "outputs" / "figures"
            mid = len(adaptive_frames) // 2
            diag_figures = {
                "fig_predictability_adaptive": plot_predictability_graph(
                    adaptive_frames[mid], config, algorithm_label="Adaptive"
                ),
                "fig_predictability_baseline": plot_predictability_graph(
                    baseline_frames[mid], config, algorithm_label="Baseline"
                ),
                "fig_pred_heatmap_adaptive": plot_predictability_heatmap(
                    adaptive_frames[mid], algorithm_label="Adaptive"
                ),
                "fig_pred_heatmap_baseline": plot_predictability_heatmap(
                    baseline_frames[mid], algorithm_label="Baseline"
                ),
                "fig_pred_evolution_adaptive": plot_predictability_evolution(
                    adaptive_frames, algorithm_label="Adaptive"
                ),
                "fig_pred_evolution_baseline": plot_predictability_evolution(
                    baseline_frames, algorithm_label="Baseline"
                ),
                "fig_paths_adaptive": plot_all_message_paths(
                    adaptive_frames,
                    adaptive_fwd,
                    config,
                    algorithm_label="Adaptive",
                ),
                "fig_paths_baseline": plot_all_message_paths(
                    baseline_frames,
                    baseline_fwd,
                    config,
                    algorithm_label="Baseline",
                ),
            }
            for alg_label, fwd_log, frames in [
                ("adaptive", adaptive_fwd, adaptive_frames),
                ("baseline", baseline_fwd, baseline_frames),
            ]:
                journeys = find_message_journeys(fwd_log)
                if journeys:
                    msg_id = min(
                        journeys.keys(), key=lambda m: journeys[m][0].timestamp
                    )
                    diag_figures[f"fig_journey_{alg_label}"] = plot_message_journey(
                        msg_id,
                        journeys[msg_id],
                        frames,
                        config,
                        algorithm_label=alg_label.capitalize(),
                    )

            for name, fig in diag_figures.items():
                save_figure(fig, name, output_dir)
                plt.close(fig)
            st.success(
                f"Saved {len(diag_figures)} diagnostic figures to `{output_dir}`"
            )


# ---- Tab 5: Statistical Analysis ----
with tab_stats:
    st.header("Statistical Analysis")

    if st.session_state.report is None:
        st.warning("Run the experiment first to generate statistical analysis.")
    else:
        report = st.session_state.report

        with st.expander("How to read these tables", expanded=False):
            st.markdown("""
            **p-value:** Below 0.05 indicates the observed difference is unlikely
            to be due to chance.

            **Cohen's d:** Standardised effect size.
            - |d| >= 0.8: **large** -- distributions are substantially separated
            - 0.5 <= |d| < 0.8: **medium** -- meaningful separation
            - 0.2 <= |d| < 0.5: **small** -- detectable but modest
            - |d| < 0.2: **negligible** -- algorithms produce similar outcomes

            At low connectivity, increasing variance in both algorithms tends to
            reduce Cohen's d even when the mean difference is meaningful.

            **Eta-squared (ANOVA):** Proportion of variance explained by connectivity
            level. This answers SQ1: if the Baseline has higher eta-squared than the
            Adaptive for avg_delivery_time, connectivity degradation affects the
            Baseline more sharply -- the Adaptive algorithm is more robust.

            | Measure | Small | Medium | Large |
            |---------|-------|--------|-------|
            | Cohen's d | 0.2 | 0.5 | 0.8 |
            | Eta-squared | 0.01 | 0.06 | 0.14 |
            """)

        st.subheader("Welch's t-test: Adaptive vs Baseline")
        st.caption(
            "Independent samples t-test (unequal variances). "
            "Significance level: alpha = 0.05"
        )
        ttest_df = build_ttest_table(report)

        def _highlight_sig(val: str) -> str:
            return "background-color: #d4edda" if val == "Yes" else ""

        st.dataframe(
            ttest_df.style.map(_highlight_sig, subset=["Sig."]),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        st.subheader("One-Way ANOVA: Effect of Connectivity")
        st.caption(
            "Tests whether connectivity level significantly affects each metric."
        )
        anova_df = build_anova_table(report)
        st.dataframe(
            anova_df.style.map(_highlight_sig, subset=["Sig."]),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        st.subheader("Effect Size Interpretation")
        st.caption("Cohen's d for each metric x connectivity combination")

        for comp in report.comparisons:
            d = abs(comp.ttest.cohens_d)
            if d < 0.2:
                size_label = "negligible"
            elif d < 0.5:
                size_label = "small"
            elif d < 0.8:
                size_label = "medium"
            else:
                size_label = "large"

            conn_label = (
                f"@ {comp.connectivity_level * 100:.0f}%"
                if comp.connectivity_level is not None
                else "(overall)"
            )
            sig_icon = "Yes" if comp.ttest.significant else "No"
            st.text(
                f"  {comp.metric.value:20s} {conn_label:12s}  "
                f"d = {comp.ttest.cohens_d:+.3f} ({size_label:10s})  "
                f"Sig: {sig_icon}"
            )


# ---- Tab 6: Key Findings ----
with tab_findings:
    st.header("Key Findings")

    if st.session_state.report is None:
        st.warning("Run the experiment first to generate findings.")
    else:
        report = st.session_state.report
        df = st.session_state.df

        # Headline delivery time metrics
        st.subheader("MRQ: Does Adaptive Improve Coordination Effectiveness?")
        st.markdown(
            "**Metric:** avg_delivery_time (lower = faster coordination response)"
        )

        dt_comps = [
            c
            for c in report.comparisons
            if c.metric == MetricType.DELIVERY_TIME and c.connectivity_level is not None
        ]
        if dt_comps:
            dt_cols = st.columns(len(dt_comps))
            for i, comp in enumerate(
                sorted(dt_comps, key=lambda c: -c.connectivity_level)
            ):
                with dt_cols[i]:
                    conn_label = f"{int(comp.connectivity_level * 100)}%"
                    st.metric(
                        f"avg_delivery_time @ {conn_label}",
                        f"{comp.improvement:+.1f}%",
                        delta_color="inverse",
                    )
                    if comp.ttest.significant:
                        st.caption(
                            f"**Significant** (p={comp.ttest.p_value:.4f}, "
                            f"d={comp.ttest.cohens_d:+.3f})"
                        )
                    else:
                        st.caption(
                            f"Not significant (p={comp.ttest.p_value:.4f}, "
                            f"d={comp.ttest.cohens_d:+.3f})"
                        )

            # One-sentence answer
            all_sig = all(c.ttest.significant for c in dt_comps)
            all_lower = all(c.improvement < 0 for c in dt_comps)
            if all_sig and all_lower:
                st.success(
                    "**Answer:** Yes -- the Adaptive algorithm achieves significantly "
                    "lower delivery times across all connectivity levels."
                )
            elif any(c.ttest.significant and c.improvement < 0 for c in dt_comps):
                sig_levels = [
                    f"{int(c.connectivity_level * 100)}%"
                    for c in dt_comps
                    if c.ttest.significant and c.improvement < 0
                ]
                st.info(
                    f"**Partial support:** Significant improvement at "
                    f"{', '.join(sig_levels)} connectivity."
                )
            else:
                st.warning(
                    "**No significant difference** in delivery times "
                    "between algorithms."
                )

        st.divider()

        # SQ1: Connectivity effect
        st.subheader("SQ1: How Does Connectivity Degradation Affect Each Algorithm?")
        st.markdown("**Metric:** ANOVA eta-squared on avg_delivery_time")

        dt_anova_keys = [
            k for k in report.anova_results if k.startswith("delivery_time")
        ]
        if dt_anova_keys:
            anova_cols = st.columns(len(dt_anova_keys))
            for i, key in enumerate(sorted(dt_anova_keys)):
                anova = report.anova_results[key]
                algorithm = key.rsplit("_", 1)[1].capitalize() if "_" in key else "All"
                with anova_cols[i]:
                    st.metric(
                        f"eta-squared ({algorithm})",
                        f"{anova.eta_squared:.3f}",
                    )
                    st.caption(f"Effect: {anova._interpret_effect_size()}")

            # Interpretation
            adaptive_key = [k for k in dt_anova_keys if "adaptive" in k]
            baseline_key = [k for k in dt_anova_keys if "baseline" in k]
            if adaptive_key and baseline_key:
                ada_eta = report.anova_results[adaptive_key[0]].eta_squared
                base_eta = report.anova_results[baseline_key[0]].eta_squared
                if base_eta > ada_eta:
                    st.success(
                        f"**Answer:** Connectivity degradation has a larger effect on "
                        f"the Baseline (eta-squared={base_eta:.3f}) than the Adaptive "
                        f"({ada_eta:.3f}), indicating the network-aware mechanism "
                        f"provides robustness under degradation."
                    )
                else:
                    st.info(
                        f"**Answer:** Both algorithms are similarly affected by "
                        f"connectivity changes (Adaptive eta-squared={ada_eta:.3f}, "
                        f"Baseline={base_eta:.3f})."
                    )

        st.divider()

        # SQ2: Trade-off analysis
        st.subheader("SQ2: What Is the Complexity-Resilience Trade-off?")
        st.markdown("**Metric:** delivery_rate (higher = more assignments delivered)")

        dr_comps = [
            c
            for c in report.comparisons
            if c.metric == MetricType.DELIVERY_RATE and c.connectivity_level is not None
        ]
        if dr_comps:
            dr_cols = st.columns(len(dr_comps))
            for i, comp in enumerate(
                sorted(dr_comps, key=lambda c: -c.connectivity_level)
            ):
                with dr_cols[i]:
                    conn_label = f"{int(comp.connectivity_level * 100)}%"
                    st.metric(
                        f"delivery_rate @ {conn_label}",
                        f"{comp.improvement:+.2f}%",
                    )

            # Interpretation
            ada_lower = any(c.improvement < 0 for c in dr_comps)
            if ada_lower:
                st.info(
                    "**Answer:** The Adaptive algorithm trades some delivery coverage "
                    "for higher per-assignment reliability. The P > 0.3 filter excludes "
                    "unreachable responders, reducing the delivery rate but ensuring "
                    "that assignments that *are* made have a higher probability of "
                    "reaching the responder. This is the trade-off SQ2 investigates."
                )
            else:
                st.success(
                    "**Answer:** The Adaptive algorithm maintains comparable or higher "
                    "delivery rates, suggesting the network-aware filtering does not "
                    "impose a significant coverage cost."
                )

        st.divider()

        # Summary
        st.subheader("Summary")
        st.markdown("""
        **What would support all three research questions:**
        Lower avg_delivery_time for Adaptive at all levels; higher eta-squared
        for Baseline on avg_delivery_time; a delivery_rate gap consistent with
        the P > 0.3 filter being more active at lower connectivity.

        **What would challenge the research questions:**
        No significant delivery_time difference at any connectivity level;
        similar eta-squared for both algorithms; delivery_rate patterns
        inconsistent with the reachability filter mechanism.
        """)
