#!/usr/bin/env python3
"""
Quick diagnostic: verify algorithm discrimination after k_max removal.

Runs 5 seeds × 3 connectivity levels × 2 algorithms = 30 runs.
Prints summary table and per-seed detail for Adaptive at 20% connectivity.

Usage:
    python scripts/diagnose_kmax_fix.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine

ALGORITHMS = [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]
CONNECTIVITY_LEVELS = [0.75, 0.40, 0.20]
NUM_SEEDS = 5

VALID_SEEDS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "valid_seeds.json"
)


def _get_seeds(count: int) -> list[int]:
    if VALID_SEEDS_PATH.exists():
        with VALID_SEEDS_PATH.open() as f:
            data = json.load(f)
        seeds = [int(s) for s in data.get("valid_seeds", [])]
        if len(seeds) >= count:
            return seeds[:count]
    return [42 + i for i in range(count)]


def main():
    config = SimulationConfig()
    seeds = _get_seeds(NUM_SEEDS)
    total_runs = len(ALGORITHMS) * len(CONNECTIVITY_LEVELS) * NUM_SEEDS
    print(f"Running {total_runs} simulations (seeds: {seeds})\n")

    # results[algo][conn] = list of dicts
    results: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))

    run_count = 0
    t0 = time.time()

    for algo in ALGORITHMS:
        for conn in CONNECTIVITY_LEVELS:
            for i, seed in enumerate(seeds):
                run_count += 1
                elapsed = time.time() - t0
                print(
                    f"  [{run_count:2d}/{total_runs}] {algo.value:>8s}  "
                    f"conn={conn:.2f}  seed={seed}  ({elapsed:.0f}s)",
                    flush=True,
                )
                engine = SimulationEngine(
                    config=config,
                    algorithm_type=algo,
                    connectivity_level=conn,
                    random_seed=seed,
                )
                r = engine.run(run_number=i)
                results[algo.value][conn].append(
                    {
                        "seed": seed,
                        "delivery_rate": r.delivery_rate,
                        "delivery_time": r.average_delivery_time,
                        "assignment_rate": r.assignment_rate,
                    }
                )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s\n")

    # ── Summary table ──────────────────────────────────────────────────
    header = (
        f"{'Conn':>5s}  {'Algorithm':>10s}  "
        f"{'Del.Rate':>14s}  {'Del.Time (s)':>16s}  {'Assign.Rate':>14s}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for conn in CONNECTIVITY_LEVELS:
        for algo_name in ["adaptive", "baseline"]:
            runs = results[algo_name][conn]
            dr = [r["delivery_rate"] for r in runs]
            dt = [r["delivery_time"] for r in runs if r["delivery_time"] is not None]
            ar = [r["assignment_rate"] for r in runs]

            dr_str = (
                f"{mean(dr):.4f} ± {stdev(dr):.4f}" if len(dr) > 1 else f"{dr[0]:.4f}"
            )
            dt_str = (
                f"{mean(dt):.1f} ± {stdev(dt):.1f}"
                if len(dt) > 1
                else (f"{dt[0]:.1f}" if dt else "N/A")
            )
            ar_str = (
                f"{mean(ar):.4f} ± {stdev(ar):.4f}" if len(ar) > 1 else f"{ar[0]:.4f}"
            )

            print(
                f"{conn:5.2f}  {algo_name:>10s}  {dr_str:>14s}  {dt_str:>16s}  {ar_str:>14s}"
            )

        # Check discrimination at this connectivity level
        a_dr = [r["delivery_rate"] for r in results["adaptive"][conn]]
        b_dr = [r["delivery_rate"] for r in results["baseline"][conn]]
        a_dt = [
            r["delivery_time"]
            for r in results["adaptive"][conn]
            if r["delivery_time"] is not None
        ]
        b_dt = [
            r["delivery_time"]
            for r in results["baseline"][conn]
            if r["delivery_time"] is not None
        ]

        warn = False
        if a_dt and b_dt and mean(a_dt) >= mean(b_dt):
            print(
                f"  WARNING: algorithms not discriminating at conn={conn:.2f} "
                f"(Adaptive dt={mean(a_dt):.1f} >= Baseline dt={mean(b_dt):.1f})"
            )
            warn = True
        if mean(a_dr) < mean(b_dr) - 0.01:
            print(
                f"  WARNING: Adaptive delivery_rate < Baseline at conn={conn:.2f} "
                f"({mean(a_dr):.4f} < {mean(b_dr):.4f})"
            )
            warn = True
        if not warn:
            print("  OK")
        print()

    print("=" * len(header))

    # ── Per-seed detail: Adaptive at 20% ───────────────────────────────
    print("\nPer-seed detail: Adaptive at 20% connectivity")
    print(f"{'Seed':>6s}  {'Del.Rate':>10s}  {'Del.Time':>10s}  {'Assign.Rate':>12s}")
    print("-" * 44)
    for r in results["adaptive"][0.20]:
        dt_str = (
            f"{r['delivery_time']:.1f}" if r["delivery_time"] is not None else "N/A"
        )
        print(
            f"{r['seed']:6d}  {r['delivery_rate']:10.4f}  {dt_str:>10s}  {r['assignment_rate']:12.4f}"
        )

    a_dt_20 = [
        r["delivery_time"]
        for r in results["adaptive"][0.20]
        if r["delivery_time"] is not None
    ]
    if a_dt_20:
        mu = mean(a_dt_20)
        sd = stdev(a_dt_20) if len(a_dt_20) > 1 else 0.0
        outliers = [
            r["seed"]
            for r in results["adaptive"][0.20]
            if r["delivery_time"] is not None and abs(r["delivery_time"] - mu) > 2 * sd
        ]
        if outliers:
            print(f"\n  Outlier seeds (>2σ from mean): {outliers}")
        else:
            print(f"\n  No outlier seeds (all within 2σ of mean={mu:.1f}s)")


if __name__ == "__main__":
    main()
