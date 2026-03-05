"""Robustness check: analyse Adaptive at 20% with and without catastrophic seeds.

Runs 30 seeds of Adaptive and Baseline at 20% connectivity and compares
statistics with and without seeds where coord_0 is structurally isolated
(assignment_rate == 0.0).

This is supplementary analysis for the dissertation — it does NOT modify
the main experiment results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
from scipy import stats

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import ExperimentRunner, SimulationEngine


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((m1 - m2) / pooled) if pooled > 0 else 0.0


def print_stats(label, values):
    arr = np.array(values)
    print(f"  {label}: mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  n={len(arr)}")


def print_ttest(ada_vals, bas_vals):
    if len(ada_vals) < 2 or len(bas_vals) < 2:
        print("  t-test: insufficient data")
        return
    t_stat, p_val = stats.ttest_ind(ada_vals, bas_vals, equal_var=False)
    d = cohens_d(ada_vals, bas_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f} ({sig}), Cohen's d={d:.3f}")


config = SimulationConfig()
runner = ExperimentRunner(config=config, base_seed=42)
seeds = runner._get_seeds(30)

# -------------------------------------------------------------------------
# Run all 30 seeds for Adaptive and Baseline at 20%
# -------------------------------------------------------------------------
print("Running 30 seeds x 2 algorithms @ 20% connectivity...")

adaptive_results = []
baseline_results = []

for i, seed in enumerate(seeds):
    for alg in [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]:
        engine = SimulationEngine(
            config=config,
            algorithm_type=alg,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run(run_number=i)
        if alg == AlgorithmType.ADAPTIVE:
            adaptive_results.append((seed, result, engine))
        else:
            baseline_results.append((seed, result, engine))

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/30 seeds done")

print(f"Done. {len(adaptive_results)} Adaptive + {len(baseline_results)} Baseline runs.\n")

# -------------------------------------------------------------------------
# Identify catastrophic seeds (assignment_rate == 0.0)
# -------------------------------------------------------------------------
catastrophic_seeds = []
for seed, result, engine in adaptive_results:
    if result.assignment_rate == 0.0:
        catastrophic_seeds.append((seed, result, engine))

print("=" * 70)
print("CATASTROPHIC SEED IDENTIFICATION")
print("=" * 70)

if not catastrophic_seeds:
    print("\n  No catastrophic seeds found (all assignment_rate > 0).")
else:
    print(f"\n  Found {len(catastrophic_seeds)} seed(s) with assignment_rate == 0.0:")
    for seed, result, engine in catastrophic_seeds:
        print(f"\n  --- Seed {seed} ---")
        print(f"    assignment_rate = {result.assignment_rate:.4f}")
        print(f"    tasks_assigned  = {result.tasks_assigned}/{result.total_tasks}")
        print(f"    failed_assignments = {engine._manager.coordinator.statistics['failed_assignments']}")

        # Diagnose coord_0 vs coord_1 isolation
        matrix = engine._communication.predictability
        coord_ids = engine._topology.get_coordination_node_ids()
        mobile_ids = engine._topology.get_mobile_responder_ids()

        for cid in coord_ids:
            p_above = sum(
                1 for mid in mobile_ids
                if matrix.get_predictability(cid, mid) > 0.3
            )
            p_values = [matrix.get_predictability(cid, mid) for mid in mobile_ids]
            p_nonzero = [p for p in p_values if p > 0]
            print(
                f"    {cid}: {p_above}/{len(mobile_ids)} responders with P > 0.3  "
                f"(max P = {max(p_values):.4f}, "
                f"nonzero = {len(p_nonzero)})"
            )

# -------------------------------------------------------------------------
# Compare: ALL seeds vs EXCLUDING catastrophic
# -------------------------------------------------------------------------
ada_rates_all = [r.assignment_rate for _, r, _ in adaptive_results]
bas_rates_all = [r.assignment_rate for _, r, _ in baseline_results]

catastrophic_seed_set = {s for s, _, _ in catastrophic_seeds}

ada_rates_excl = [r.assignment_rate for s, r, _ in adaptive_results if s not in catastrophic_seed_set]
bas_rates_excl = [r.assignment_rate for s, r, _ in baseline_results if s not in catastrophic_seed_set]

print("\n\n" + "=" * 70)
print("ASSIGNMENT RATE COMPARISON")
print("=" * 70)

print(f"\nALL {len(ada_rates_all)} SEEDS:")
print_stats("Adaptive@20% ", ada_rates_all)
print_stats("Baseline@20% ", bas_rates_all)
print_ttest(ada_rates_all, bas_rates_all)

n_excl = len(catastrophic_seed_set)
print(f"\nEXCLUDING {n_excl} CATASTROPHIC FAILURE(S) (n={len(ada_rates_excl)}, "
      f"removed: seed(s) {sorted(catastrophic_seed_set)}):")
print_stats("Adaptive@20% ", ada_rates_excl)
print_stats("Baseline@20% ", bas_rates_excl)
print_ttest(ada_rates_excl, bas_rates_excl)

# -------------------------------------------------------------------------
# Also compare delivery_rate and delivery_time
# -------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("DELIVERY RATE COMPARISON")
print("=" * 70)

ada_dr_all = [r.delivery_rate for _, r, _ in adaptive_results]
bas_dr_all = [r.delivery_rate for _, r, _ in baseline_results]
ada_dr_excl = [r.delivery_rate for s, r, _ in adaptive_results if s not in catastrophic_seed_set]
bas_dr_excl = [r.delivery_rate for s, r, _ in baseline_results if s not in catastrophic_seed_set]

print(f"\nALL {len(ada_dr_all)} SEEDS:")
print_stats("Adaptive@20% ", ada_dr_all)
print_stats("Baseline@20% ", bas_dr_all)
print_ttest(ada_dr_all, bas_dr_all)

print(f"\nEXCLUDING CATASTROPHIC (n={len(ada_dr_excl)}):")
print_stats("Adaptive@20% ", ada_dr_excl)
print_stats("Baseline@20% ", bas_dr_excl)
print_ttest(ada_dr_excl, bas_dr_excl)

# -------------------------------------------------------------------------
# Delivery time
# -------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("DELIVERY TIME COMPARISON")
print("=" * 70)

ada_dt_all = [r.average_delivery_time for _, r, _ in adaptive_results if r.average_delivery_time is not None]
bas_dt_all = [r.average_delivery_time for _, r, _ in baseline_results if r.average_delivery_time is not None]
ada_dt_excl = [r.average_delivery_time for s, r, _ in adaptive_results
               if s not in catastrophic_seed_set and r.average_delivery_time is not None]
bas_dt_excl = [r.average_delivery_time for s, r, _ in baseline_results
               if s not in catastrophic_seed_set and r.average_delivery_time is not None]

print(f"\nALL SEEDS (n_ada={len(ada_dt_all)}, n_bas={len(bas_dt_all)}):")
print_stats("Adaptive@20% ", ada_dt_all)
print_stats("Baseline@20% ", bas_dt_all)
print_ttest(ada_dt_all, bas_dt_all)

print(f"\nEXCLUDING CATASTROPHIC (n_ada={len(ada_dt_excl)}, n_bas={len(bas_dt_excl)}):")
print_stats("Adaptive@20% ", ada_dt_excl)
print_stats("Baseline@20% ", bas_dt_excl)
print_ttest(ada_dt_excl, bas_dt_excl)

# -------------------------------------------------------------------------
# Conclusion
# -------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if catastrophic_seeds:
    seed_list = sorted(catastrophic_seed_set)
    ada_mean_excl = np.mean(ada_rates_excl)
    ada_std_excl = np.std(ada_rates_excl, ddof=1)
    bas_mean_excl = np.mean(bas_rates_excl)
    bas_std_excl = np.std(bas_rates_excl, ddof=1)

    t_stat, p_val = stats.ttest_ind(ada_rates_excl, bas_rates_excl, equal_var=False)
    d = cohens_d(ada_rates_excl, bas_rates_excl)
    sig = "significant" if p_val < 0.05 else "not significant"

    print(f"\n  With seed(s) {seed_list} excluded (coord_0 structural isolation):")
    print(f"  Adaptive@20% assignment_rate = {ada_mean_excl:.4f} +/- {ada_std_excl:.4f}")
    print(f"  Baseline@20% assignment_rate = {bas_mean_excl:.4f} +/- {bas_std_excl:.4f}")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}, d={d:.3f} -- {sig}")

    if ada_mean_excl == bas_mean_excl and ada_std_excl == bas_std_excl:
        print(f"\n  => Without the catastrophic seed, assignment_rate is IDENTICAL")
        print(f"     between Adaptive and Baseline, confirming that the P > 0.3")
        print(f"     filter is inactive for all other seeds at 20%.")
else:
    print("\n  No catastrophic seeds found. Assignment rates are identical")
    print("  between Adaptive and Baseline at all seeds.")
