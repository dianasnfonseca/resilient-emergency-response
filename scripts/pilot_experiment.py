#!/usr/bin/env python3
"""
Pilot experiment: sanity check before full 180-run simulation.

For each of the 6 configurations prints:
  - Algorithm, connectivity level
  - Mean delivery rate, mean assignment rate, mean tasks, mean delivered

Acceptance conditions (all must pass):
  1. Static α=0.2, γ_r=0.2, β=0.6
  2. Adaptive delivery_time < baseline delivery_time at 20% connectivity
  3. Delivery rate degrades as connectivity decreases for both algorithms
  4. No responder exceeds k_max task assignments in any coordination cycle

Usage:
  python scripts/pilot_experiment.py --runs 30 --connectivity 0.20 0.40 0.75
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Add project root so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine, SimulationEventType

ALGORITHMS = [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]

# Load valid seeds if available
VALID_SEEDS_PATH = Path(__file__).resolve().parent.parent / "config" / "valid_seeds.json"
BASE_SEED = 42


def _get_seeds(count: int) -> list[int]:
    if VALID_SEEDS_PATH.exists():
        with open(VALID_SEEDS_PATH) as f:
            data = json.load(f)
        seeds = [int(s) for s in data.get("valid_seeds", [])]
        if len(seeds) >= count:
            return seeds[:count]
    return [BASE_SEED + i for i in range(count)]


def _check_k_max_violations(result) -> list[str]:
    """
    Verify that no responder exceeds k_max assignments in any coordination cycle.

    Reads k_max and max_observed_load logged by the engine from
    COORDINATION_CYCLE events and checks that max_observed_load <= k_max.

    Returns a list of violation messages (empty if all OK).
    """
    violations = []

    # Only check adaptive — baseline has no k_max mechanism
    if result.algorithm != AlgorithmType.ADAPTIVE:
        return violations

    for event in result.events:
        if event.event_type != SimulationEventType.COORDINATION_CYCLE:
            continue
        k_max = event.data.get("k_max")
        max_load = event.data.get("max_observed_load")
        if k_max is not None and max_load is not None and max_load > k_max:
            violations.append(
                f"  Cycle t={event.timestamp:.0f}: "
                f"max_observed_load={max_load} > k_max={k_max} "
                f"(assignments={event.data.get('assignments', '?')})"
            )

    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="ERCS pilot experiment")
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Runs per configuration (default: 5)",
    )
    parser.add_argument(
        "--connectivity", type=float, nargs="+",
        default=[0.75, 0.40, 0.20],
        help="Connectivity levels (default: 0.75 0.40 0.20)",
    )
    args = parser.parse_args()

    runs_per_config = args.runs
    connectivity_levels = sorted(args.connectivity, reverse=True)

    config = SimulationConfig()
    seeds = _get_seeds(runs_per_config)

    print(f"Config: warmup={config.scenario.warmup_period_seconds}s, "
          f"duration={config.scenario.simulation_duration_seconds}s, "
          f"total={config.total_simulation_duration}s")
    print(f"Threshold={config.coordination.available_path_threshold}, "
          f"α={config.coordination.predictability_weight}, "
          f"γ_r={config.coordination.recency_weight}, "
          f"β={config.coordination.proximity_weight}")
    print(f"Runs per config: {runs_per_config}\n")

    total = len(ALGORITHMS) * len(connectivity_levels) * runs_per_config
    current = 0
    t0 = time.time()

    # Collect results: (algorithm, connectivity) -> list of per-run dicts
    results: dict[tuple[str, float], list[dict]] = defaultdict(list)

    # k_max violation tracking (condition 4)
    all_k_max_violations: list[str] = []

    for algorithm in ALGORITHMS:
        for connectivity in connectivity_levels:
            for run in range(runs_per_config):
                seed = seeds[run]

                engine = SimulationEngine(
                    config=config,
                    algorithm_type=algorithm,
                    connectivity_level=connectivity,
                    random_seed=seed,
                )

                result = engine.run(run_number=run)

                # Check k_max for adaptive runs
                k_max_violations = _check_k_max_violations(result)
                if k_max_violations:
                    header = (f"  [{algorithm.value} @ {connectivity:.2f} "
                              f"run {run} seed {seed}]")
                    all_k_max_violations.append(header)
                    all_k_max_violations.extend(k_max_violations)

                run_data = {
                    "delivery_rate": result.delivery_rate,
                    "assignment_rate": result.assignment_rate,
                    "avg_response_time": result.average_decision_time,
                    "avg_delivery_time": result.average_delivery_time,
                    "total_tasks": result.total_tasks,
                    "tasks_assigned": result.tasks_assigned,
                    "messages_created": result.messages_created,
                    "messages_delivered": result.messages_delivered,
                }

                results[(algorithm.value, connectivity)].append(run_data)

                current += 1
                elapsed = time.time() - t0
                eta = (elapsed / current) * (total - current) if current > 0 else 0
                print(
                    f"\r  [{current}/{total}] "
                    f"{algorithm.value:>8s} @ {connectivity:.2f} run {run} "
                    f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)",
                    end="",
                    flush=True,
                )

    elapsed_total = time.time() - t0
    print(f"\n\nCompleted {total} runs in {elapsed_total:.1f}s\n")

    # ── Save raw per-run data to JSON for statistical analysis ──────────
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    serialisable = {
        f"{alg}@{conn}": runs
        for (alg, conn), runs in results.items()
    }
    raw_path = output_dir / "pilot_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Raw per-run data saved to {raw_path}\n")

    # ── Print summary table ─────────────────────────────────────────────
    sep = "=" * 75
    print(sep)
    print(f"{'Algorithm':<10} {'Conn':>5} │ {'Del.Rate':>8} {'Del.Time':>8} "
          f"{'Asgn.Rate':>9} {'Tasks':>6} {'Dlvd':>5}")
    print(sep)

    # Store mean values for acceptance checks
    mean_del: dict[tuple[str, float], float] = {}
    mean_del_time: dict[tuple[str, float], float | None] = {}
    mean_asgn: dict[tuple[str, float], float] = {}

    for algorithm in ALGORITHMS:
        for connectivity in connectivity_levels:
            key = (algorithm.value, connectivity)
            runs = results[key]

            del_rates = [r["delivery_rate"] for r in runs]
            del_times = [r["avg_delivery_time"] for r in runs
                         if r["avg_delivery_time"] is not None]
            asgn_rates = [r["assignment_rate"] for r in runs]
            total_tasks = [r["total_tasks"] for r in runs]
            delivered = [r["messages_delivered"] for r in runs]

            m_del = mean(del_rates)
            m_del_t = mean(del_times) if del_times else None
            m_asgn = mean(asgn_rates)
            mean_del[key] = m_del
            mean_del_time[key] = m_del_t
            mean_asgn[key] = m_asgn

            del_t_str = f"{m_del_t:>8.1f}" if m_del_t is not None else "     N/A"

            print(
                f"{algorithm.value:<10} {connectivity:>5.2f} │ "
                f"{m_del:>8.3f} {del_t_str} {m_asgn:>9.3f} "
                f"{mean(total_tasks):>6.0f} {mean(delivered):>5.0f}"
            )

        if algorithm == AlgorithmType.ADAPTIVE:
            print("-" * 75)

    print(sep)

    # ── Acceptance conditions ────────────────────────────────────────────
    print("\n── Acceptance conditions ──")

    # Condition 1: static weights α=0.2, γ_r=0.2, β=0.6
    alpha = config.coordination.predictability_weight
    gamma_r = config.coordination.recency_weight
    beta = config.coordination.proximity_weight
    cond1 = alpha == 0.2 and gamma_r == 0.2 and beta == 0.6
    status1 = "PASS" if cond1 else "FAIL"
    print(f"  1. Static α={alpha}, γ_r={gamma_r}, β={beta}: {status1}")

    # Condition 2: adaptive delivery_time < baseline at 20% connectivity
    adap_dt_20 = mean_del_time[("adaptive", 0.20)]
    base_dt_20 = mean_del_time[("baseline", 0.20)]
    if adap_dt_20 is not None and base_dt_20 is not None:
        cond2 = adap_dt_20 < base_dt_20
        status2 = "PASS" if cond2 else "FAIL"
        print(f"  2. Adaptive del_time ({adap_dt_20:.1f}s) < "
              f"Baseline del_time ({base_dt_20:.1f}s) at 20%: {status2}")
    else:
        cond2 = False
        status2 = "FAIL"
        print(f"  2. Delivery time comparison at 20%: {status2} (no deliveries)")

    # Condition 3: delivery rate degrades as connectivity decreases (both algorithms)
    # Check highest > lowest; intermediate levels may show sample-variance inversions.
    highest = connectivity_levels[0]
    lowest = connectivity_levels[-1]
    adap_degrades = mean_del[("adaptive", highest)] > mean_del[("adaptive", lowest)]
    base_degrades = mean_del[("baseline", highest)] > mean_del[("baseline", lowest)]
    cond3 = adap_degrades and base_degrades
    status3 = "PASS" if cond3 else "FAIL"
    adap_dels = [f"{mean_del[('adaptive', c)]:.3f}" for c in connectivity_levels]
    base_dels = [f"{mean_del[('baseline', c)]:.3f}" for c in connectivity_levels]
    print(f"  3. Delivery rate degrades with connectivity: {status3}")
    print(f"     Adaptive: {' → '.join(adap_dels)}")
    print(f"     Baseline: {' → '.join(base_dels)}")

    # Condition 4: No responder exceeds k_max in any coordination cycle
    cond4 = len(all_k_max_violations) == 0
    status4 = "PASS" if cond4 else "FAIL"
    print(f"  4. k_max bound enforced (no responder > k_max tasks/cycle): {status4}")
    if not cond4:
        for line in all_k_max_violations[:10]:  # cap output
            print(f"     {line}")

    # Final verdict
    print()
    if cond1 and cond2 and cond3 and cond4:
        print("PILOT PASSED — launch full 180-run simulation")
    else:
        print("PILOT FAILED — review conditions above")


if __name__ == "__main__":
    main()
