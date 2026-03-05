"""Diagnose avg_delivery_time across algorithms and connectivity levels.

Runs 6 simulations (1 seed x 3 connectivity x 2 algorithms) and prints:
- mean +/- std of delivery_time per (algorithm, connectivity)
- Welch's t-test Adaptive vs Baseline for delivery_time at each level
- Cohen's d
- Percentage of assigned tasks with delivery_time recorded
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
from scipy import stats

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import ExperimentRunner

config = SimulationConfig()
runner = ExperimentRunner(config=config, base_seed=42)

RUNS = 5  # Quick diagnostic — enough to see direction

results = runner.run_all(
    algorithms=[AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE],
    connectivity_levels=[0.75, 0.40, 0.20],
    runs_per_config=RUNS,
)

print(f"Total results: {len(results)}")
print()

# Group by (algorithm, connectivity)
from collections import defaultdict

groups = defaultdict(list)
for r in results:
    key = (r.algorithm.value, r.connectivity_level)
    groups[key].append(r)


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else 0.0


print("=" * 80)
print("DELIVERY TIME ANALYSIS")
print("=" * 80)

for conn in [0.75, 0.40, 0.20]:
    print(f"\n--- Connectivity {conn:.0%} ---")

    for alg in ["adaptive", "baseline"]:
        rs = groups[(alg, conn)]
        dt_vals = [r.average_delivery_time for r in rs if r.average_delivery_time is not None]
        assigned_counts = [r.tasks_assigned for r in rs]
        delivery_counts = [len(r.delivery_times) for r in rs]

        # Fraction of assigned tasks that have delivery_time
        total_assigned = sum(assigned_counts)
        total_delivered = sum(delivery_counts)
        delivery_pct = total_delivered / total_assigned * 100 if total_assigned > 0 else 0

        if dt_vals:
            print(
                f"  {alg:8s}: mean={np.mean(dt_vals):8.1f} +/- {np.std(dt_vals, ddof=1):6.1f}s  "
                f"(n={len(dt_vals)})  "
                f"delivered={total_delivered}/{total_assigned} ({delivery_pct:.1f}%)"
            )
        else:
            print(f"  {alg:8s}: NO delivery_time data")

    # t-test
    ada_dt = [
        r.average_delivery_time
        for r in groups[("adaptive", conn)]
        if r.average_delivery_time is not None
    ]
    bas_dt = [
        r.average_delivery_time
        for r in groups[("baseline", conn)]
        if r.average_delivery_time is not None
    ]

    if len(ada_dt) >= 2 and len(bas_dt) >= 2:
        t_stat, p_val = stats.ttest_ind(ada_dt, bas_dt, equal_var=False)
        d = cohens_d(ada_dt, bas_dt)
        diff = np.mean(ada_dt) - np.mean(bas_dt)
        pct_diff = diff / np.mean(bas_dt) * 100 if np.mean(bas_dt) > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  Welch's t-test: t={t_stat:.3f}, p={p_val:.4f} ({sig})")
        print(f"  Cohen's d: {d:.3f}")
        print(f"  Adaptive - Baseline: {diff:+.1f}s ({pct_diff:+.1f}%)")
    else:
        print(f"\n  t-test: insufficient data")

print("\n" + "=" * 80)
print("DELIVERY RATE vs DELIVERY TIME COMPARISON")
print("=" * 80)
print(
    "\n  delivery_rate measures: messages_delivered / messages_created"
    "\n  delivery_time measures: mean(delivery_timestamp - creation_timestamp)"
    "\n  for delivered messages"
    "\n"
    "\n  A task can be assigned (response_time) but the message may:"
    "\n  - Be delivered quickly (low delivery_time)"
    "\n  - Take many hops (high delivery_time)"
    "\n  - Never arrive (no delivery_time, message expired = TTL 300min)"
)

# Show delivery_rate too for context
print(f"\n{'':4s}{'':8s}  {'delivery_rate':>14s}  {'delivery_time':>14s}")
for conn in [0.75, 0.40, 0.20]:
    for alg in ["adaptive", "baseline"]:
        rs = groups[(alg, conn)]
        dr = np.mean([r.delivery_rate for r in rs])
        dt_vals = [r.average_delivery_time for r in rs if r.average_delivery_time is not None]
        dt = np.mean(dt_vals) if dt_vals else float("nan")
        print(f"  {conn:.0%} {alg:8s}  {dr:14.4f}  {dt:14.1f}s")
