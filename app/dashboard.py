"""
ERCS Experiment Dashboard — Streamlit Application.

Run with: streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
    COLORS,
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
# Research context (displayed before any controls)
# ---------------------------------------------------------------------------

st.markdown("""
**What is this?**
An experimental simulation comparing two coordination algorithms for emergency
responders operating under degraded network infrastructure.

**Research questions this experiment addresses:**

- **MRQ:** Can adaptive scheduling algorithms integrated with delay-tolerant
  communication architectures improve emergency resource coordination effectiveness
  when operating under intermittent connectivity conditions?
- **SQ1:** How do distributed communication strategies combined with adaptive
  scheduling algorithms impact resource allocation effectiveness during varying
  levels of network disruption?
- **SQ2:** What are the optimal trade-offs between system complexity, resilience,
  and performance when adapting centralised emergency coordination approaches for
  decentralised, low-connectivity environments?

**The setup:** 50 nodes, 3000x1500m area, three connectivity scenarios (75%, 40%,
20%). Each scenario run 30 times (Law, 2015). PRoPHETv2 routing (Grasic et al., 2011).

**The two algorithms:**

- **Adaptive:** Urgency-first task ordering, network-aware responder selection
  (P > 0.3 reachability filter, k_max capacity bound, PRoPHETv2 delivery
  predictability scoring).
- **Baseline:** FCFS task ordering, nearest responder by Euclidean distance,
  no network state.
""")

st.divider()

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
    st.session_state.diag_frames = None  # (adaptive_frames, baseline_frames)
if "diag_fwd" not in st.session_state:
    st.session_state.diag_fwd = None  # (adaptive_fwd, baseline_fwd)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

config = SimulationConfig()

with st.sidebar:
    st.title("ERCS Dashboard")
    st.caption("Emergency Response Coordination Simulator")

    st.divider()

    quick_test = st.toggle("Quick test mode (5 runs)", value=True)
    runs = 5 if quick_test else config.scenario.runs_per_configuration

    total_runs = 2 * len(config.network.connectivity_scenarios) * runs
    st.metric("Total simulation runs", total_runs)

    st.divider()

    with st.expander("Network Topology"):
        st.markdown(f"""
        - **Nodes:** {config.total_nodes} ({config.network.coordination_node_count} coord + {config.network.mobile_responder_count} mobile)
        - **Area:** {config.network.simulation_area.width_m:.0f} x {config.network.simulation_area.height_m:.0f} m\u00b2
        - **Radio range:** {config.network.radio_range_m:.0f} m
        - **Buffer:** {config.network.buffer_size_bytes / 1_048_576:.0f} MB
        """)

    with st.expander("PRoPHETv2 Protocol"):
        st.markdown(f"""
        - **P_enc_max:** {config.communication.prophet.p_enc_max}
        - **I_typ:** {config.communication.prophet.i_typ:.0f} s
        - **\u03b2:** {config.communication.prophet.beta}
        - **\u03b3:** {config.communication.prophet.gamma}
        - **TTL:** {config.communication.message_ttl_seconds // 60} min
        """)

    with st.expander("Scenario"):
        st.markdown(f"""
        - **Arrival:** {config.scenario.message_generation_model.title()} process
        - **Rate:** {config.scenario.message_rate_per_minute} msgs/min
        - **Urgency:** {config.scenario.urgency_distribution.high*100:.0f}% H / {config.scenario.urgency_distribution.medium*100:.0f}% M / {config.scenario.urgency_distribution.low*100:.0f}% L
        - **Duration:** {config.scenario.simulation_duration_seconds}s
        """)

    with st.expander("Coordination"):
        st.markdown(f"""
        - **Interval:** {config.coordination.update_interval_seconds // 60} min
        - **Adaptive:** {config.coordination.adaptive_task_order.replace("_", " ").title()}
        - **Baseline:** {config.coordination.baseline_task_order.upper()}
        - **P threshold:** {config.coordination.available_path_threshold}
        """)

    with st.expander("Why P > 0.3?"):
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

        Setting the threshold too low makes the filter ineffective (the Adaptive
        algorithm behaves like the Baseline). Setting it too high leaves too few
        eligible responders, reducing assignment coverage.
        """)


# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tab_setup, tab_run, tab_viz, tab_diag, tab_stats, tab_findings = st.tabs(
    ["Parameters", "Run Experiment", "Visualizations", "Network Diagnostics",
     "Statistical Analysis", "How to Read Results"]
)

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
    st.subheader("Experimental Design")
    design_cols = st.columns(4)
    design_cols[0].metric("Algorithms", "2")
    design_cols[1].metric("Connectivity Levels", str(len(config.network.connectivity_scenarios)))
    design_cols[2].metric("Runs per Config", str(runs))
    design_cols[3].metric("Total Runs", str(total_runs))


# ---- Tab 2: Run Experiment ----
with tab_run:
    st.header("Run Experiment")

    mode_label = "Quick test (5 runs/config)" if quick_test else "Full experiment (30 runs/config)"
    st.info(f"Mode: **{mode_label}** \u2014 {total_runs} total simulations")

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
                f"**{current}/{total}** \u2014 {alg_name} @ {conn_pct} connectivity, "
                f"run {run_idx + 1}/{runs} \u2014 "
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
            metrics=[MetricType.DELIVERY_RATE, MetricType.ASSIGNMENT_RATE, MetricType.DECISION_TIME, MetricType.DELIVERY_TIME]
        )

        st.success(f"Experiment completed: {len(results)} simulation runs")

    if st.session_state.df is not None:
        st.subheader("Summary Statistics")
        summary_df = (
            st.session_state.df
            .groupby(["algorithm", "connectivity"])[["delivery_rate", "assignment_rate", "avg_decision_time", "avg_delivery_time"]]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )
        st.dataframe(summary_df, use_container_width=True)


# ---- Tab 3: Visualizations ----
with tab_viz:
    st.header("Results Visualizations")

    if st.session_state.df is None:
        st.warning("Run the experiment first (tab 2) to generate visualizations.")
    else:
        df = st.session_state.df

        # Precompute summaries
        summaries = {}
        for metric_key in METRICS_CONFIG:
            summaries[metric_key] = compute_summary_stats(df, metric_key)

        # What do these metrics mean?
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
              per-assignment reliability. This is not a failure mode.

            - **assignment_rate** (diagnostic): Set by task arrival and simulation
              parameters, not by algorithm decisions. Expected identical across
              algorithms.

            - **avg_decision_time** (diagnostic): Internal processing latency
              determined by the coordination cycle interval. Expected identical across
              algorithms.
            """)

        # Primary metric: avg_delivery_time
        st.subheader("Primary Metric: Coordination Response Time (avg_delivery_time) -- MRQ, SQ1")
        if "avg_delivery_time" in summaries:
            fig = plot_grouped_bars(summaries["avg_delivery_time"], "avg_delivery_time")
            st.pyplot(fig)

        st.divider()

        # Remaining grouped bar charts
        st.subheader("Algorithm Comparison (Grouped Bar Charts)")
        remaining_metrics = [k for k in METRICS_CONFIG if k != "avg_delivery_time"]
        bar_cols = st.columns(min(len(remaining_metrics), 3))
        for i, metric_key in enumerate(remaining_metrics):
            with bar_cols[i % len(bar_cols)]:
                fig = plot_grouped_bars(summaries[metric_key], metric_key)
                st.pyplot(fig)

        st.divider()

        # Box plots
        st.subheader("Distributions (Box Plots)")
        fig = plot_box_distributions(df)
        st.pyplot(fig)

        st.divider()

        # Heatmap
        st.subheader("Performance Heatmap")
        fig = plot_heatmap(df)
        st.pyplot(fig)

        st.divider()

        # Degradation lines
        st.subheader("Connectivity Degradation")
        fig = plot_degradation_lines(summaries)
        st.pyplot(fig)

        st.divider()

        # Save all figures
        if st.button("Save All Figures to outputs/figures/"):
            output_dir = project_root / "outputs" / "figures"
            figures = {
                "fig_delivery_rate_bars": plot_grouped_bars(summaries["delivery_rate"], "delivery_rate"),
                "fig_assignment_rate_bars": plot_grouped_bars(summaries["assignment_rate"], "assignment_rate"),
                "fig_response_time_bars": plot_grouped_bars(summaries["avg_decision_time"], "avg_decision_time"),
                "fig_delivery_time_bars": plot_grouped_bars(summaries["avg_delivery_time"], "avg_delivery_time"),
                "fig_box_distributions": plot_box_distributions(df),
                "fig_heatmap": plot_heatmap(df),
                "fig_degradation_lines": plot_degradation_lines(summaries),
            }
            for name, fig in figures.items():
                save_figure(fig, name, output_dir)
            st.success(f"Saved {len(figures)} figures to `{output_dir}`")


# ---- Tab 4: Network Diagnostics ----
with tab_diag:
    st.header("Network Diagnostics")
    st.markdown(
        "Visualizations of the PRoPHET routing protocol, message forwarding, "
        "and node mobility from a single representative simulation run."
    )

    diag_col1, diag_col2 = st.columns([1, 3])
    with diag_col1:
        diag_connectivity = st.selectbox(
            "Connectivity level",
            options=config.network.connectivity_scenarios,
            format_func=lambda x: f"{x * 100:.0f}%",
            index=len(config.network.connectivity_scenarios) - 1,  # lowest (most interesting)
        )

    if st.button("Generate Diagnostics", type="primary", use_container_width=True):
        with st.spinner(f"Running paired simulation at {diag_connectivity * 100:.0f}% connectivity..."):
            start_diag = time.time()
            adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd = run_paired_simulation(
                config=config,
                connectivity_level=diag_connectivity,
                seed=42,
                sample_interval=30.0,
            )
            elapsed_diag = time.time() - start_diag

        st.session_state.diag_frames = (adaptive_frames, baseline_frames)
        st.session_state.diag_fwd = (adaptive_fwd, baseline_fwd)
        st.success(f"Diagnostics generated in {elapsed_diag:.1f}s "
                   f"({len(adaptive_frames)} frames per algorithm)")

    if st.session_state.diag_frames is not None:
        adaptive_frames, baseline_frames = st.session_state.diag_frames
        adaptive_fwd, baseline_fwd = st.session_state.diag_fwd

        # --- PRoPHET Predictability Graph ---
        st.divider()
        st.subheader("PRoPHET Predictability Graph")
        st.caption("Network graph showing predictability values between nodes at a given time snapshot.")

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
            fig = plot_predictability_graph(adaptive_frames[snapshot_idx], config, algorithm_label="Adaptive")
            st.pyplot(fig)
        with pred_cols[1]:
            st.markdown("**Baseline**")
            fig = plot_predictability_graph(baseline_frames[snapshot_idx], config, algorithm_label="Baseline")
            st.pyplot(fig)

        # --- Predictability Heatmap ---
        st.divider()
        st.subheader("Predictability Heatmap")
        st.caption("Coord-to-mobile predictability matrix at the selected snapshot.")

        hm_cols = st.columns(2)
        with hm_cols[0]:
            fig = plot_predictability_heatmap(adaptive_frames[snapshot_idx], algorithm_label="Adaptive")
            st.pyplot(fig)
        with hm_cols[1]:
            fig = plot_predictability_heatmap(baseline_frames[snapshot_idx], algorithm_label="Baseline")
            st.pyplot(fig)

        # --- Predictability Evolution ---
        st.divider()
        st.subheader("Predictability Evolution")
        st.caption("Time series of how predictability values build and decay over the simulation.")

        evo_cols = st.columns(2)
        with evo_cols[0]:
            fig = plot_predictability_evolution(adaptive_frames, algorithm_label="Adaptive")
            st.pyplot(fig)
        with evo_cols[1]:
            fig = plot_predictability_evolution(baseline_frames, algorithm_label="Baseline")
            st.pyplot(fig)

        # --- Message Journey ---
        st.divider()
        st.subheader("Message Journey")
        st.caption("Hop-by-hop path of a message through the network.")

        for label, fwd_log, frames in [
            ("Adaptive", adaptive_fwd, adaptive_frames),
            ("Baseline", baseline_fwd, baseline_frames),
        ]:
            journeys = find_message_journeys(fwd_log)
            if journeys:
                msg_id = min(journeys.keys(), key=lambda m: journeys[m][0].timestamp)
                fig = plot_message_journey(msg_id, journeys[msg_id], frames, config, algorithm_label=label)
                st.pyplot(fig)
            else:
                st.info(f"No forwarding events recorded for {label}.")

        # --- All Message Paths ---
        st.divider()
        st.subheader("All Message Paths")
        st.caption("Overview of all message routes, colored by delivery status.")

        path_cols = st.columns(2)
        with path_cols[0]:
            fig = plot_all_message_paths(adaptive_frames, adaptive_fwd, config, algorithm_label="Adaptive")
            st.pyplot(fig)
        with path_cols[1]:
            fig = plot_all_message_paths(baseline_frames, baseline_fwd, config, algorithm_label="Baseline")
            st.pyplot(fig)

        # --- Save diagnostic figures ---
        st.divider()
        if st.button("Save Diagnostic Figures to outputs/figures/"):
            output_dir = project_root / "outputs" / "figures"
            mid = len(adaptive_frames) // 2
            diag_figures = {
                "fig_predictability_adaptive": plot_predictability_graph(
                    adaptive_frames[mid], config, algorithm_label="Adaptive"),
                "fig_predictability_baseline": plot_predictability_graph(
                    baseline_frames[mid], config, algorithm_label="Baseline"),
                "fig_pred_heatmap_adaptive": plot_predictability_heatmap(
                    adaptive_frames[mid], algorithm_label="Adaptive"),
                "fig_pred_heatmap_baseline": plot_predictability_heatmap(
                    baseline_frames[mid], algorithm_label="Baseline"),
                "fig_pred_evolution_adaptive": plot_predictability_evolution(
                    adaptive_frames, algorithm_label="Adaptive"),
                "fig_pred_evolution_baseline": plot_predictability_evolution(
                    baseline_frames, algorithm_label="Baseline"),
                "fig_paths_adaptive": plot_all_message_paths(
                    adaptive_frames, adaptive_fwd, config, algorithm_label="Adaptive"),
                "fig_paths_baseline": plot_all_message_paths(
                    baseline_frames, baseline_fwd, config, algorithm_label="Baseline"),
            }
            # Add message journey figures if available
            for alg_label, fwd_log, frames in [
                ("adaptive", adaptive_fwd, adaptive_frames),
                ("baseline", baseline_fwd, baseline_frames),
            ]:
                journeys = find_message_journeys(fwd_log)
                if journeys:
                    msg_id = min(journeys.keys(), key=lambda m: journeys[m][0].timestamp)
                    diag_figures[f"fig_journey_{alg_label}"] = plot_message_journey(
                        msg_id, journeys[msg_id], frames, config, algorithm_label=alg_label.capitalize())

            for name, fig in diag_figures.items():
                save_figure(fig, name, output_dir)
            st.success(f"Saved {len(diag_figures)} diagnostic figures to `{output_dir}`")


# ---- Tab 5: Statistical Analysis ----
with tab_stats:
    st.header("Statistical Analysis")

    if st.session_state.report is None:
        st.warning("Run the experiment first (tab 2) to generate statistical analysis.")
    else:
        report = st.session_state.report

        with st.expander("How to read these tables", expanded=False):
            st.markdown("""
            **p-value:** Below 0.05 indicates the observed difference is unlikely
            to be due to chance.

            **Cohen's d:** Standardised effect size. Above 0.8 -- distributions are
            substantially separated, algorithms are producing reliably different
            outcomes. Below 0.2 -- negligible separation. At low connectivity,
            increasing variance in both algorithms tends to reduce Cohen's d even
            when the mean difference is meaningful.

            **eta-squared (ANOVA):** Proportion of variance in the metric explained
            by connectivity level. This answers SQ1 directly: if the Baseline has
            higher eta-squared than the Adaptive for avg_delivery_time, connectivity
            degradation affects the Baseline more sharply -- the Adaptive algorithm
            is more robust. If eta-squared values are similar, the network-aware
            mechanism is not providing the robustness benefit hypothesised.
            """)

        st.subheader("Welch's t-test: Adaptive vs Baseline")
        st.caption("Independent samples t-test (unequal variances). Significance level: \u03b1 = 0.05")
        ttest_df = build_ttest_table(report)
        st.dataframe(
            ttest_df.style.map(
                lambda v: "background-color: #d4edda" if v == "Yes" else "",
                subset=["Sig."],
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        st.subheader("One-Way ANOVA: Effect of Connectivity")
        st.caption("Tests whether connectivity level significantly affects each metric.")
        anova_df = build_anova_table(report)
        st.dataframe(
            anova_df.style.map(
                lambda v: "background-color: #d4edda" if v == "Yes" else "",
                subset=["Sig."],
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        st.subheader("Effect Size Interpretation")
        for comp in report.comparisons:
            d = abs(comp.ttest.cohens_d)
            if d < 0.2:
                size = "negligible"
            elif d < 0.5:
                size = "small"
            elif d < 0.8:
                size = "medium"
            else:
                size = "large"
            conn_label = f"@ {comp.connectivity_level * 100:.0f}%" if comp.connectivity_level is not None else "(overall)"
            sig_icon = "\u2705" if comp.ttest.significant else "\u274c"
            st.text(
                f"  {comp.metric.value:20s} {conn_label:12s}  "
                f"d = {comp.ttest.cohens_d:+.3f} ({size:10s})  {sig_icon}"
            )


# ---- Tab 6: How to Read the Results ----
with tab_findings:
    st.header("How to Read the Results")

    if st.session_state.report is None:
        st.warning("Run the experiment first (tab 2) to generate results.")
    else:
        report = st.session_state.report

        st.subheader("For the MRQ")
        st.markdown(
            "Is avg_delivery_time consistently lower for the Adaptive algorithm "
            "across all connectivity levels? If yes, adaptive coordination improves "
            "coordination effectiveness under intermittent connectivity."
        )

        # Show delivery time comparison
        dt_comps = [c for c in report.comparisons
                    if c.metric == MetricType.DELIVERY_TIME
                    and c.connectivity_level is not None]
        if dt_comps:
            dt_cols = st.columns(len(dt_comps))
            for i, comp in enumerate(sorted(dt_comps, key=lambda c: -c.connectivity_level)):
                with dt_cols[i]:
                    conn_label = f"{int(comp.connectivity_level * 100)}%"
                    st.metric(
                        f"avg_delivery_time @ {conn_label}",
                        f"{comp.improvement:+.1f}%",
                        delta_color="inverse",
                    )
                    sig_text = f"p={comp.ttest.p_value:.4f}, d={comp.ttest.cohens_d:+.3f}"
                    if comp.ttest.significant:
                        st.caption(f"Significant: {sig_text}")
                    else:
                        st.caption(f"Not significant: {sig_text}")

        st.divider()

        st.subheader("For SQ1")
        st.markdown(
            "Does the gap in avg_delivery_time between algorithms grow as connectivity "
            "decreases? Does the Baseline's ANOVA eta-squared exceed the Adaptive's? "
            "If yes, distributed network-aware communication provides increasing "
            "robustness under degradation relative to proximity-only coordination."
        )

        # Show ANOVA eta-squared for delivery_time
        dt_anova_keys = [k for k in report.anova_results if k.startswith("delivery_time")]
        if dt_anova_keys:
            anova_cols = st.columns(len(dt_anova_keys))
            for i, key in enumerate(sorted(dt_anova_keys)):
                anova = report.anova_results[key]
                algorithm = key.rsplit("_", 1)[1].capitalize() if "_" in key else "All"
                with anova_cols[i]:
                    st.metric(f"eta-squared ({algorithm})", f"{anova.eta_squared:.3f}")

        st.divider()

        st.subheader("For SQ2")
        st.markdown(
            "Is delivery_rate lower for the Adaptive algorithm? If yes, this is "
            "evidence of the trade-off SQ2 asks about -- reduced coverage in exchange "
            "for higher reliability per assignment. The size of the gap and whether it "
            "grows with connectivity degradation indicates how much the trade-off costs "
            "operationally."
        )

        # Show delivery rate comparison
        dr_comps = [c for c in report.comparisons
                    if c.metric == MetricType.DELIVERY_RATE
                    and c.connectivity_level is not None]
        if dr_comps:
            dr_cols = st.columns(len(dr_comps))
            for i, comp in enumerate(sorted(dr_comps, key=lambda c: -c.connectivity_level)):
                with dr_cols[i]:
                    conn_label = f"{int(comp.connectivity_level * 100)}%"
                    st.metric(
                        f"delivery_rate @ {conn_label}",
                        f"{comp.improvement:+.2f}%",
                    )

        st.divider()

        st.subheader("What would support all three research questions")
        st.markdown(
            "Lower avg_delivery_time for Adaptive at all levels; higher eta-squared "
            "for Baseline on avg_delivery_time; a delivery_rate gap consistent with "
            "the P > 0.3 filter being more active at lower connectivity."
        )

        st.subheader("What would challenge the research questions")
        st.markdown(
            "No significant delivery_time difference at any connectivity level; "
            "similar eta-squared for both algorithms; delivery_rate patterns "
            "inconsistent with the reachability filter mechanism."
        )
