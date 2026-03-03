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

    with st.expander("PRoPHET Protocol"):
        st.markdown(f"""
        - **P_init:** {config.communication.prophet.p_init}
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
        """)


# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------

tab_setup, tab_run, tab_viz, tab_diag, tab_stats, tab_findings = st.tabs(
    ["Parameters", "Run Experiment", "Visualizations", "Network Diagnostics",
     "Statistical Analysis", "Key Findings"]
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
            metrics=[MetricType.DELIVERY_RATE, MetricType.ASSIGNMENT_RATE, MetricType.RESPONSE_TIME]
        )

        st.success(f"Experiment completed: {len(results)} simulation runs")

    if st.session_state.df is not None:
        st.subheader("Summary Statistics")
        summary_df = (
            st.session_state.df
            .groupby(["algorithm", "connectivity"])[["delivery_rate", "assignment_rate", "avg_response_time"]]
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

        # Grouped bar charts
        st.subheader("Algorithm Comparison (Grouped Bar Charts)")
        bar_cols = st.columns(3)
        for i, metric_key in enumerate(METRICS_CONFIG):
            with bar_cols[i]:
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
                "fig_response_time_bars": plot_grouped_bars(summaries["avg_response_time"], "avg_response_time"),
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


# ---- Tab 6: Key Findings ----
with tab_findings:
    st.header("Key Findings")

    if st.session_state.report is None:
        st.warning("Run the experiment first (tab 2) to generate findings.")
    else:
        report = st.session_state.report

        # Delivery rate overall
        overall_dr = next(
            (c for c in report.comparisons
             if c.metric == MetricType.DELIVERY_RATE and c.connectivity_level is None),
            None,
        )
        if overall_dr:
            st.subheader("1. Overall Delivery Rate")
            col1, col2, col3 = st.columns(3)
            col1.metric("Adaptive", f"{overall_dr.adaptive_stats.mean:.4f}")
            col2.metric("Baseline", f"{overall_dr.baseline_stats.mean:.4f}")
            col3.metric("Improvement", f"{overall_dr.improvement:+.2f}%")
            if overall_dr.ttest.significant:
                st.success(f"Statistically significant (p = {overall_dr.ttest.p_value:.4f}, d = {overall_dr.ttest.cohens_d:.3f})")
            else:
                st.info(f"Not statistically significant (p = {overall_dr.ttest.p_value:.4f})")

        # Largest advantage
        per_conn_dr = [
            c for c in report.comparisons
            if c.metric == MetricType.DELIVERY_RATE and c.connectivity_level is not None
        ]
        if per_conn_dr:
            best = max(per_conn_dr, key=lambda c: c.improvement)
            st.subheader("2. Largest Adaptive Advantage")
            st.metric(
                f"At {best.connectivity_level * 100:.0f}% connectivity",
                f"{best.improvement:+.2f}% improvement",
            )

        # Connectivity effect
        st.subheader("3. Connectivity Effect (ANOVA)")
        for key, anova in report.anova_results.items():
            parts = key.rsplit("_", 1)
            metric_name = parts[0].replace("_", " ").title()
            algorithm = parts[1].capitalize() if len(parts) > 1 else "All"
            effect = anova._interpret_effect_size()
            sig_text = "significant" if anova.significant else "not significant"
            st.text(
                f"  {metric_name} ({algorithm}): "
                f"F({anova.df_between},{anova.df_within}) = {anova.f_statistic:.3f}, "
                f"p = {anova.p_value:.4f} ({sig_text}), "
                f"\u03b7\u00b2 = {anova.eta_squared:.3f} ({effect})"
            )
