"""Diagnostic script for the three anomalies in ERCS results.

Investigates three anomalies observed in the experiment output:
  1. Identical response_time and assignment_rate at 40% and 75% connectivity
  2. PRoPHET P values exceeding P_enc_max = 0.5
  3. High variance in Adaptive assignment_rate at 20% connectivity

Usage:
    python scripts/diagnose_anomalies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import ExperimentRunner, SimulationEngine


def diagnose_anomaly1():
    """Anomaly 1: Identical response_time and assignment_rate at 40%/75%."""
    print("=" * 70)
    print("ANOMALY 1: Identical response_time and assignment_rate")
    print("=" * 70)

    config = SimulationConfig()
    runner = ExperimentRunner(config=config, base_seed=42)

    results = runner.run_all(
        algorithms=[AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE],
        connectivity_levels=[0.75, 0.40, 0.20],
        runs_per_config=3,
    )

    print(f"\nTotal results: {len(results)}")
    print()

    # Group by connectivity and algorithm
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r.algorithm.value, r.connectivity_level)
        groups[key].append(r)

    for (alg, conn), rs in sorted(groups.items()):
        print(f"\n--- {alg} @ {conn:.0%} ---")
        for r in rs:
            rt_vals = [t for _, t in r.response_times] if r.response_times else []
            rt_mean = sum(rt_vals) / len(rt_vals) if rt_vals else 0
            print(
                f"  run={r.run_number} seed={r.random_seed} "
                f"total_tasks={r.total_tasks} assigned={r.tasks_assigned} "
                f"rate={r.assignment_rate:.4f} "
                f"rt_mean={rt_mean:.1f} rt_n={len(rt_vals)} "
                f"id(rt_list)={id(r.response_times)}"
            )

    # Verify: are response_times lists the same objects?
    print("\n\n--- OBJECT IDENTITY CHECK ---")
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i < j and r1.response_times is r2.response_times:
                print(
                    f"  SHARED response_times: result[{i}] ({r1.algorithm.value}@{r1.connectivity_level}) "
                    f"and result[{j}] ({r2.algorithm.value}@{r2.connectivity_level})"
                )
    print("  (no output above = no shared objects)")

    # Key insight: check if response_time is just cycle_time - creation_time
    print("\n\n--- RESPONSE TIME ANALYSIS ---")
    for r in results[:6]:  # First 6 results (3 adaptive + 3 baseline at 0.75)
        if r.connectivity_level != 0.75 or r.run_number != 0:
            continue
        rt_by_cycle = defaultdict(list)
        for _task_id, rt in r.response_times:
            # Bucket response times by rounding to nearest 100
            rt_by_cycle[round(rt / 100) * 100].append(rt)

        print(f"\n  {r.algorithm.value}@{r.connectivity_level:.0%} run={r.run_number}:")
        print(f"    total_tasks={r.total_tasks}, assigned={r.tasks_assigned}")
        print(
            f"    unassigned = tasks after last cycle = {r.total_tasks - r.tasks_assigned}"
        )

        # Count tasks per coordination cycle
        # Cycles at t=0, 1800, 3600, 5400
        # Tasks arrive uniformly 0-6000s
        # Tasks arriving 5400-6000 won't get a cycle
        expected_unassigned = r.total_tasks * (600 / 6000)
        print(
            f"    expected unassigned (~tasks in last 600s): ~{expected_unassigned:.0f}"
        )


def diagnose_anomaly2():
    """Anomaly 2: P values exceeding P_enc_max=0.5."""
    print("\n\n" + "=" * 70)
    print("ANOMALY 2: P values exceeding P_enc_max=0.5")
    print("=" * 70)

    config = SimulationConfig()
    engine = SimulationEngine(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=0.75,
        random_seed=42,
    )
    engine.run(run_number=0)

    # Access the predictability matrix
    matrix = engine._communication.predictability
    coord_ids = engine._topology.get_coordination_node_ids()
    mobile_ids = engine._topology.get_mobile_responder_ids()

    # Collect all P values
    all_p = []
    coord_to_mobile_p = []

    for src in matrix._matrix:
        for dst, p in matrix._matrix[src].items():
            all_p.append(p)
            if src in coord_ids and dst in mobile_ids:
                coord_to_mobile_p.append(p)

    all_p = np.array(all_p) if all_p else np.array([0.0])
    coord_to_mobile_p = (
        np.array(coord_to_mobile_p) if coord_to_mobile_p else np.array([0.0])
    )

    print(f"\nAll P values (n={len(all_p)}):")
    print(f"  min={all_p.min():.4f}, max={all_p.max():.4f}")
    print(f"  mean={all_p.mean():.4f}, median={np.median(all_p):.4f}")
    print(f"  >0.5: {(all_p > 0.5).sum()} ({(all_p > 0.5).mean():.1%})")
    print(f"  >0.3: {(all_p > 0.3).sum()} ({(all_p > 0.3).mean():.1%})")
    print(f"  >0.8: {(all_p > 0.8).sum()} ({(all_p > 0.8).mean():.1%})")

    print(f"\nCoord -> Mobile P values (n={len(coord_to_mobile_p)}):")
    print(f"  min={coord_to_mobile_p.min():.4f}, max={coord_to_mobile_p.max():.4f}")
    print(
        f"  mean={coord_to_mobile_p.mean():.4f}, median={np.median(coord_to_mobile_p):.4f}"
    )
    print(
        f"  >0.3: {(coord_to_mobile_p > 0.3).sum()}/{len(coord_to_mobile_p)} ({(coord_to_mobile_p > 0.3).mean():.1%})"
    )
    print(
        f"  >0.5: {(coord_to_mobile_p > 0.5).sum()}/{len(coord_to_mobile_p)} ({(coord_to_mobile_p > 0.5).mean():.1%})"
    )

    # Mathematical demonstration: P_enc_max doesn't cap P
    print("\n--- Mathematical proof: repeated encounters exceed P_enc_max ---")
    p = 0.0
    p_enc_max = 0.5
    for i in range(1, 11):
        p = p + (1 - p) * p_enc_max
        print(f"  After encounter {i}: P = {p:.6f}")

    # Also check at 40% and 20%
    for conn in [0.40, 0.20]:
        engine2 = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=conn,
            random_seed=42,
        )
        engine2.run(run_number=0)
        matrix2 = engine2._communication.predictability

        c2m = []
        for cid in coord_ids:
            for mid in mobile_ids:
                p_val = matrix2.get_predictability(cid, mid)
                if p_val > 0:
                    c2m.append(p_val)

        c2m = np.array(c2m) if c2m else np.array([0.0])
        n_above_03 = (c2m > 0.3).sum()
        print(
            f"\nCoord->Mobile at {conn:.0%}: n_nonzero={len(c2m)}, >0.3: {n_above_03}/{len(coord_to_mobile_p)}"
        )
        if len(c2m) > 0:
            print(f"  min={c2m.min():.4f}, max={c2m.max():.4f}, mean={c2m.mean():.4f}")


def diagnose_anomaly3():
    """Anomaly 3: High variance in Adaptive assignment_rate at 20%."""
    print("\n\n" + "=" * 70)
    print("ANOMALY 3: Adaptive assignment_rate variance at 20%")
    print("=" * 70)

    config = SimulationConfig()
    seeds = [1, 42, 100]

    for seed in seeds:
        print(f"\n--- Seed {seed}, Adaptive @ 20% ---")
        engine = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run(run_number=0)

        # Count failed assignments from coordinator events
        coordinator = engine._manager.coordinator
        (
            coordinator.get_events_by_type(
                coordinator._events[0].event_type.__class__("assignment_failed")
                if False
                else None
            )
            if False
            else []
        )

        # Use the coordinator stats directly
        stats = coordinator.statistics

        print(f"  total_tasks={result.total_tasks}")
        print(f"  tasks_assigned={result.tasks_assigned}")
        print(f"  assignment_rate={result.assignment_rate:.4f}")
        print(f"  failed_assignments={stats['failed_assignments']}")
        print(f"  unique_responders={stats['unique_responders_used']}")

        # Check how many responders have P > 0.3
        matrix = engine._communication.predictability
        coord_ids = engine._topology.get_coordination_node_ids()
        mobile_ids = engine._topology.get_mobile_responder_ids()

        p_above_threshold = 0
        for cid in coord_ids:
            for mid in mobile_ids:
                if matrix.get_predictability(cid, mid) > 0.3:
                    p_above_threshold += 1
        print(
            f"  coord->mobile pairs with P>0.3: {p_above_threshold} / {len(coord_ids) * len(mobile_ids)}"
        )

    # Also check Baseline at 20% for comparison
    print("\n--- Seed 42, Baseline @ 20% ---")
    engine = SimulationEngine(
        config=config,
        algorithm_type=AlgorithmType.BASELINE,
        connectivity_level=0.20,
        random_seed=42,
    )
    result = engine.run(run_number=0)
    stats = engine._manager.coordinator.statistics
    print(f"  total_tasks={result.total_tasks}")
    print(f"  tasks_assigned={result.tasks_assigned}")
    print(f"  assignment_rate={result.assignment_rate:.4f}")
    print(f"  failed_assignments={stats['failed_assignments']}")


if __name__ == "__main__":
    diagnose_anomaly1()
    diagnose_anomaly2()
    diagnose_anomaly3()
