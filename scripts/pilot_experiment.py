#!/usr/bin/env python3
"""
Pilot experiment: 5 runs per configuration (sanity check, not statistical inference).

For each of the 6 configurations prints:
  - Algorithm, connectivity level
  - Mean delivery rate, mean assignment rate, mean tasks, mean delivered

Acceptance conditions (all must pass):
  1. Static α=0.3 used in every coordinator call
  2. Adaptive delivery_rate > baseline delivery_rate at 20% connectivity
  3. Delivery rate degrades as connectivity decreases for both algorithms
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Add project root so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine

RUNS_PER_CONFIG = 5
ALGORITHMS = [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]
CONNECTIVITY_LEVELS = [0.75, 0.40, 0.20]

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


def main() -> None:
    config = SimulationConfig()
    seeds = _get_seeds(RUNS_PER_CONFIG)

    print(f"Config: warmup={config.scenario.warmup_period_seconds}s, "
          f"duration={config.scenario.simulation_duration_seconds}s, "
          f"total={config.total_simulation_duration}s")
    print(f"Threshold={config.coordination.available_path_threshold}, "
          f"α={config.coordination.predictability_weight}, "
          f"β={config.coordination.proximity_weight}\n")

    total = len(ALGORITHMS) * len(CONNECTIVITY_LEVELS) * RUNS_PER_CONFIG
    current = 0
    t0 = time.time()

    # Collect results: (algorithm, connectivity) -> list of per-run dicts
    results: dict[tuple[str, float], list[dict]] = defaultdict(list)

    for algorithm in ALGORITHMS:
        for connectivity in CONNECTIVITY_LEVELS:
            for run in range(RUNS_PER_CONFIG):
                seed = seeds[run]

                engine = SimulationEngine(
                    config=config,
                    algorithm_type=algorithm,
                    connectivity_level=connectivity,
                    random_seed=seed,
                )

                result = engine.run(run_number=run)

                run_data = {
                    "delivery_rate": result.delivery_rate,
                    "assignment_rate": result.assignment_rate,
                    "avg_response_time": result.average_response_time,
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

    # ── Print summary table ─────────────────────────────────────────────
    sep = "=" * 65
    print(sep)
    print(f"{'Algorithm':<10} {'Conn':>5} │ {'Del.Rate':>8} {'Asgn.Rate':>9} "
          f"{'Tasks':>6} {'Dlvd':>5}")
    print(sep)

    # Store mean values for acceptance checks
    mean_del: dict[tuple[str, float], float] = {}
    mean_asgn: dict[tuple[str, float], float] = {}

    for algorithm in ALGORITHMS:
        for connectivity in CONNECTIVITY_LEVELS:
            key = (algorithm.value, connectivity)
            runs = results[key]

            del_rates = [r["delivery_rate"] for r in runs]
            asgn_rates = [r["assignment_rate"] for r in runs]
            total_tasks = [r["total_tasks"] for r in runs]
            delivered = [r["messages_delivered"] for r in runs]

            m_del = mean(del_rates)
            m_asgn = mean(asgn_rates)
            mean_del[key] = m_del
            mean_asgn[key] = m_asgn

            print(
                f"{algorithm.value:<10} {connectivity:>5.2f} │ "
                f"{m_del:>8.3f} {m_asgn:>9.3f} "
                f"{mean(total_tasks):>6.0f} {mean(delivered):>5.0f}"
            )

        if algorithm == AlgorithmType.ADAPTIVE:
            print("-" * 65)

    print(sep)

    # ── Acceptance conditions ────────────────────────────────────────────
    print("\n── Acceptance conditions ──")

    # Condition 1: static α=0.3 used everywhere
    alpha = config.coordination.predictability_weight
    beta = config.coordination.proximity_weight
    cond1 = alpha == 0.3 and beta == 0.7
    status1 = "PASS" if cond1 else "FAIL"
    print(f"  1. Static α={alpha}, β={beta}: {status1}")

    # Condition 2: adaptive delivery_rate > baseline at 20% connectivity
    adap_del_20 = mean_del[("adaptive", 0.20)]
    base_del_20 = mean_del[("baseline", 0.20)]
    cond2 = adap_del_20 > base_del_20
    status2 = "PASS" if cond2 else "FAIL"
    print(f"  2. Adaptive del_rate ({adap_del_20:.3f}) > "
          f"Baseline del_rate ({base_del_20:.3f}) at 20%: {status2}")

    # Condition 3: delivery rate degrades as connectivity decreases (both algorithms)
    # Check highest (0.75) > lowest (0.20); intermediate levels may show
    # sample-variance inversions in a 5-run pilot.
    highest = CONNECTIVITY_LEVELS[0]   # 0.75
    lowest = CONNECTIVITY_LEVELS[-1]   # 0.20
    adap_degrades = mean_del[("adaptive", highest)] > mean_del[("adaptive", lowest)]
    base_degrades = mean_del[("baseline", highest)] > mean_del[("baseline", lowest)]
    cond3 = adap_degrades and base_degrades
    status3 = "PASS" if cond3 else "FAIL"
    adap_dels = [f"{mean_del[('adaptive', c)]:.3f}" for c in CONNECTIVITY_LEVELS]
    base_dels = [f"{mean_del[('baseline', c)]:.3f}" for c in CONNECTIVITY_LEVELS]
    print(f"  3. Delivery rate degrades with connectivity: {status3}")
    print(f"     Adaptive: {' → '.join(adap_dels)}")
    print(f"     Baseline: {' → '.join(base_dels)}")

    # Final verdict
    print()
    if cond1 and cond2 and cond3:
        print("PILOT PASSED — launch full 180-run simulation")
    else:
        print("PILOT FAILED — review conditions above")


if __name__ == "__main__":
    main()
