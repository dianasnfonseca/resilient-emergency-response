# Scripts

Utility and diagnostic scripts for the ERCS (Emergency Response Coordination
Simulator) project.

## Core

Scripts used in the main experiment workflow.

| Script | Purpose |
|---|---|
| `run_experiment.py` | Main experiment runner. Loads YAML config, validates parameters, and executes the full experiment plan. |
| `run_animation.py` | Visualization suite: side-by-side animation, PRoPHET predictability graphs, heatmaps, evolution plots, and message journey tracking. |
| `pilot_experiment.py` | Pilot sanity check (5 runs per config by default). Verifies static weights, algorithm discrimination, degradation monotonicity, and k_max enforcement before launching the full 180-run experiment. |
| `validate_seeds.py` | Screens candidate seeds (1..200) to ensure transport/liaison nodes can encounter coordination nodes at every connectivity level. Outputs `configs/valid_seeds.json`. |

## Diagnostics (development history)

One-off diagnostic scripts created during development to investigate specific
anomalies or verify fixes.  They are preserved for reproducibility and as a
record of the debugging process.

| Script | Purpose |
|---|---|
| `diagnose_anomalies.py` | Investigates three anomalies: identical response times at 40%/75%, P values exceeding P_enc_max, and high variance in Adaptive assignment_rate at 20%. |
| `diagnose_anomaly3_30seeds.py` | Deep dive into Anomaly 3: runs all 30 seeds at 20% to identify bimodal assignment-rate pattern. |
| `diagnose_coordination.py` | Compares one coordination cycle between Adaptive and Baseline, showing per-task assignment details (predictability, distance, responder). |
| `diagnose_delivery.py` | Investigates 100% delivery rate after the connection-up-only encounter fix. Traces encounter calls, forwarding events, buffer copies, and delivery paths. |
| `diagnose_delivery_time.py` | Analyses avg_delivery_time across algorithms and connectivity levels with Welch's t-test and Cohen's d. |
| `diagnose_encounters.py` | Measures encounter frequency and message delivery bottlenecks. Answers whether messages can leave coordination nodes. |
| `diagnose_kmax_fix.py` | Verifies algorithm discrimination after k_max removal. Runs 5 seeds across all configurations and checks for delivery-time inversions. |
| `diagnose_mobility_prophet.py` | Combined mobility + PRoPHET diagnostic: role assignments, waypoint zones, aging verification, encounter frequency, predictability evolution, and coordination comparison. |
| `diagnose_recency.py` | Encounter-recency diagnostic pilot. Checks R_norm variance, assignment divergence from Baseline, delivery rate, and last-encounter coverage. |
| `diagnose_seed49.py` | Traces the full message lifecycle for seed 49 at 20% connectivity to determine why delivery rate is 0%. |
| `diagnose_warmup.py` | Verifies PRoPHET warm-up effectiveness: P-value distribution after warm-up, algorithm comparison, and histogram/bar-chart plots. |
| `delivery_time_analysis.py` | Statistical analysis of delivery times from pilot results (Welch's t-test, Cohen's d, 95% CI, power analysis). |
| `robustness_check_seed6.py` | Supplementary analysis comparing Adaptive vs Baseline at 20% with and without catastrophic seeds (coord_0 structurally isolated). |
