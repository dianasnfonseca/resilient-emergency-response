#!/usr/bin/env python3
"""
Pilot experiment: 5 runs per configuration (sanity check, not statistical inference).

For each of the 6 configurations prints:
  - Algorithm, connectivity level
  - Mean delivery rate, mean assignment rate
  - For Adaptive only: weight regime breakdown (good / moderate / severe)
  - For Adaptive only: mean_P distribution across coordination calls
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

# Add project root so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.coordination.algorithms import AdaptiveCoordinator
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
          f"regimes: good>{config.coordination.p_threshold_good}, "
          f"moderate>={config.coordination.p_threshold_moderate}\n")

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

                # Capture regime counts and mean_P history for adaptive
                if algorithm == AlgorithmType.ADAPTIVE:
                    coordinator = engine._coordinator
                    if isinstance(coordinator, AdaptiveCoordinator):
                        run_data["regime_counts"] = dict(
                            coordinator._regime_counts
                        )
                        run_data["mean_p_history"] = list(
                            coordinator._mean_p_history
                        )

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
    sep = "=" * 85
    print(sep)
    print(f"{'Algorithm':<10} {'Conn':>5} │ {'Del.Rate':>8} {'Asgn.Rate':>9} "
          f"{'Tasks':>6} {'Dlvd':>5} │ {'Good':>5} {'Mod':>5} {'Sev':>5} │ "
          f"{'mean_P':>6}")
    print(sep)

    for algorithm in ALGORITHMS:
        for connectivity in CONNECTIVITY_LEVELS:
            key = (algorithm.value, connectivity)
            runs = results[key]

            del_rates = [r["delivery_rate"] for r in runs]
            asgn_rates = [r["assignment_rate"] for r in runs]
            total_tasks = [r["total_tasks"] for r in runs]
            delivered = [r["messages_delivered"] for r in runs]

            if algorithm == AlgorithmType.ADAPTIVE:
                good = sum(
                    r.get("regime_counts", {}).get("good", 0) for r in runs
                )
                moderate = sum(
                    r.get("regime_counts", {}).get("moderate", 0) for r in runs
                )
                severe = sum(
                    r.get("regime_counts", {}).get("severe", 0) for r in runs
                )
                regime_str = f"{good:5d} {moderate:5d} {severe:5d}"

                # Aggregate mean_P across all runs for this config
                all_mean_ps = []
                for r in runs:
                    all_mean_ps.extend(r.get("mean_p_history", []))
                avg_mean_p = mean(all_mean_ps) if all_mean_ps else 0.0
                mp_str = f"{avg_mean_p:>6.3f}"
            else:
                regime_str = f"{'—':>5} {'—':>5} {'—':>5}"
                mp_str = f"{'—':>6}"

            print(
                f"{algorithm.value:<10} {connectivity:>5.2f} │ "
                f"{mean(del_rates):>8.3f} {mean(asgn_rates):>9.3f} "
                f"{mean(total_tasks):>6.0f} {mean(delivered):>5.0f} │ "
                f"{regime_str} │ {mp_str}"
            )

        if algorithm == AlgorithmType.ADAPTIVE:
            print("-" * 85)

    print(sep)

    # ── Per-run detail for adaptive ─────────────────────────────────────
    print("\nAdaptive per-run detail:")
    print(f"  {'Conn':>5} {'Run':>3} │ {'Del':>6} {'Asgn':>6} "
          f"{'Tasks':>5} {'Dlvd':>4} │ {'Good':>4} {'Mod':>4} {'Sev':>4} │ "
          f"{'mean_P':>6} {'min_P':>6} {'max_P':>6} {'#elig':>5}")

    for connectivity in CONNECTIVITY_LEVELS:
        key = ("adaptive", connectivity)
        for i, r in enumerate(results[key]):
            rc = r.get("regime_counts", {})
            mps = r.get("mean_p_history", [])
            if mps:
                avg_mp = mean(mps)
                min_mp = min(mps)
                max_mp = max(mps)
                n_calls = len(mps)
            else:
                avg_mp = min_mp = max_mp = 0.0
                n_calls = 0

            print(
                f"  {connectivity:>5.2f} {i:>3} │ "
                f"{r['delivery_rate']:>6.3f} {r['assignment_rate']:>6.3f} "
                f"{r['total_tasks']:>5} {r['messages_delivered']:>4} │ "
                f"{rc.get('good', 0):>4} {rc.get('moderate', 0):>4} "
                f"{rc.get('severe', 0):>4} │ "
                f"{avg_mp:>6.3f} {min_mp:>6.3f} {max_mp:>6.3f} {n_calls:>5}"
            )

    # ── mean_P histogram per connectivity ───────────────────────────────
    print("\nmean_P distribution per connectivity level:")
    bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    bin_labels = ["<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40",
                  "0.40-0.50", ">=0.50"]

    for connectivity in CONNECTIVITY_LEVELS:
        key = ("adaptive", connectivity)
        all_mps = []
        for r in results[key]:
            all_mps.extend(r.get("mean_p_history", []))

        if not all_mps:
            print(f"  {connectivity:.2f}: no data")
            continue

        counts, _ = np.histogram(all_mps, bins=bins)
        total_calls = len(all_mps)
        print(f"  {connectivity:.2f} (n={total_calls}):")
        for label, count in zip(bin_labels, counts):
            pct = 100 * count / total_calls
            bar = "█" * int(pct / 2)
            print(f"    {label:>10}: {count:>4} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
