"""Diagnose Anomaly 3: Run all 30 seeds at 20% for Adaptive to find bimodal pattern."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine, ExperimentRunner


config = SimulationConfig()

# Use the same seed loading as ExperimentRunner
runner = ExperimentRunner(config=config, base_seed=42)
seeds = runner._get_seeds(30)
print(f"Seeds: {seeds[:10]}... (first 10 of {len(seeds)})")

adaptive_rates = []
baseline_rates = []
adaptive_rt = []
baseline_rt = []

print("\n--- Adaptive @ 20% (30 seeds) ---")
print(f"{'seed':>5}  {'assigned':>8}  {'total':>5}  {'rate':>7}  {'failed':>6}  {'P>0.3':>5}  {'rt_mean':>8}")

for i, seed in enumerate(seeds):
    engine = SimulationEngine(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=0.20,
        random_seed=seed,
    )
    result = engine.run(run_number=i)

    # Count P > 0.3 pairs
    matrix = engine._communication.predictability
    coord_ids = engine._topology.get_coordination_node_ids()
    mobile_ids = engine._topology.get_mobile_responder_ids()
    p_above = sum(
        1 for cid in coord_ids for mid in mobile_ids
        if matrix.get_predictability(cid, mid) > 0.3
    )

    stats = engine._manager.coordinator.statistics
    rt_vals = [t for _, t in result.response_times]
    rt_mean = np.mean(rt_vals) if rt_vals else 0

    adaptive_rates.append(result.assignment_rate)
    adaptive_rt.append(rt_mean)

    marker = " <<<" if result.assignment_rate < 0.85 else ""
    print(
        f"{seed:5d}  {result.tasks_assigned:8d}  {result.total_tasks:5d}  "
        f"{result.assignment_rate:7.4f}  {stats['failed_assignments']:6d}  "
        f"{p_above:5d}  {rt_mean:8.1f}{marker}"
    )

print("\n--- Baseline @ 20% (30 seeds) ---")
for i, seed in enumerate(seeds):
    engine = SimulationEngine(
        config=config,
        algorithm_type=AlgorithmType.BASELINE,
        connectivity_level=0.20,
        random_seed=seed,
    )
    result = engine.run(run_number=i)
    rt_vals = [t for _, t in result.response_times]
    rt_mean = np.mean(rt_vals) if rt_vals else 0
    baseline_rates.append(result.assignment_rate)
    baseline_rt.append(rt_mean)

adaptive_rates = np.array(adaptive_rates)
baseline_rates = np.array(baseline_rates)
adaptive_rt = np.array(adaptive_rt)
baseline_rt = np.array(baseline_rt)

print(f"\n{'':>5}  mean={np.mean(baseline_rates):.4f} ± {np.std(baseline_rates, ddof=1):.4f}")

print("\n\n--- COMPARISON ---")
print(f"Assignment Rate Adaptive:  {np.mean(adaptive_rates):.4f} ± {np.std(adaptive_rates, ddof=1):.4f}")
print(f"Assignment Rate Baseline:  {np.mean(baseline_rates):.4f} ± {np.std(baseline_rates, ddof=1):.4f}")
print(f"Response Time Adaptive:    {np.mean(adaptive_rt):.4f} ± {np.std(adaptive_rt, ddof=1):.4f}")
print(f"Response Time Baseline:    {np.mean(baseline_rt):.4f} ± {np.std(baseline_rt, ddof=1):.4f}")

# Show the outlier seeds
print(f"\n--- Seeds where Adaptive assignment_rate < 0.85 ---")
for i, (seed, rate) in enumerate(zip(seeds, adaptive_rates)):
    if rate < 0.85:
        print(f"  seed={seed}, rate={rate:.4f}, baseline_rate={baseline_rates[i]:.4f}")
