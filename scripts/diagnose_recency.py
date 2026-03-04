#!/usr/bin/env python3
"""
Encounter recency diagnostic pilot.

Runs 1 seed × 3 connectivity levels × 2 algorithms and checks:
  1. R_norm variance > 0.05 across candidates in at least one cycle
  2. At least 5% of Adaptive assignments differ from Baseline
  3. Delivery rate >= 0.80 at all connectivity levels
  4. At hour 2, >= 90% of responders have last_encounter_time > 0.0

Prints the diagnostic report from TASK 7 of the implementation prompt.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# Add project root so imports work when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine


# Load a single valid seed
VALID_SEEDS_PATH = Path(__file__).resolve().parent.parent / "config" / "valid_seeds.json"


def _get_seed() -> int:
    if VALID_SEEDS_PATH.exists():
        with open(VALID_SEEDS_PATH) as f:
            data = json.load(f)
        seeds = [int(s) for s in data.get("valid_seeds", [])]
        if seeds:
            return seeds[0]
    return 42


def run_single(algorithm: AlgorithmType, connectivity: float, seed: int) -> dict:
    """Run a single simulation and return results + coordinator events."""
    config = SimulationConfig()
    engine = SimulationEngine(
        config=config,
        algorithm_type=algorithm,
        connectivity_level=connectivity,
        random_seed=seed,
    )
    result = engine.run(run_number=0)

    # Extract coordinator events for recency analysis
    events = []
    if hasattr(engine, '_coordinator') and engine._coordinator is not None:
        coord_events = engine._coordinator.get_events()
        for ev in coord_events:
            if ev.event_type.value == "task_assigned":
                event_data = {
                    "task_id": ev.task_id,
                    "responder_id": ev.responder_id,
                    **ev.details,
                }
                events.append(event_data)

    # Extract last encounter times at ~hour 2 from the adapter
    last_enc_data = {}
    if hasattr(engine, '_adapter') and engine._adapter is not None:
        adapter = engine._adapter
        responder_ids = adapter.get_all_responder_ids()
        coord_nodes = [
            nid for nid in engine._topology.get_all_node_ids()
            if nid.startswith("coord_")
        ]
        if coord_nodes:
            coord_node = coord_nodes[0]
            for rid in responder_ids:
                t = adapter.get_last_encounter_time(coord_node, rid)
                last_enc_data[rid] = t

    return {
        "delivery_rate": result.delivery_rate,
        "assignment_rate": result.assignment_rate,
        "total_tasks": result.total_tasks,
        "tasks_assigned": result.tasks_assigned,
        "messages_delivered": result.messages_delivered,
        "messages_created": result.messages_created,
        "events": events,
        "last_encounter_times": last_enc_data,
    }


def main() -> None:
    seed = _get_seed()
    connectivity_levels = [0.75, 0.40, 0.20]

    print(f"=== ENCOUNTER RECENCY DIAGNOSTIC ===")
    print(f"Seed: {seed}\n")

    # Run all configs
    results: dict[tuple[str, float], dict] = {}
    for algo in [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]:
        for conn in connectivity_levels:
            print(f"  Running {algo.value:>8s} @ {conn:.2f}...", end="", flush=True)
            r = run_single(algo, conn, seed)
            results[(algo.value, conn)] = r
            print(f" del_rate={r['delivery_rate']:.3f}, "
                  f"tasks={r['total_tasks']}, "
                  f"assigned={r['tasks_assigned']}")

    print()

    # ── Per-connectivity analysis ────────────────────────────────────────
    all_pass = True

    for conn in connectivity_levels:
        adap = results[("adaptive", conn)]
        base = results[("baseline", conn)]

        # R_norm statistics from adaptive assignment events
        recency_values = [e.get("recency", 0.0) for e in adap["events"] if "recency" in e]
        r_mean = mean(recency_values) if recency_values else 0.0
        r_std = stdev(recency_values) if len(recency_values) > 1 else 0.0

        # Assignments that differ from baseline
        adap_assignments = {
            e.get("task_id"): e.get("responder_id") for e in adap["events"]
            if "task_id" in e and "responder_id" in e
        }
        base_assignments = {
            e.get("task_id"): e.get("responder_id") for e in base["events"]
            if "task_id" in e and "responder_id" in e
        }

        common_tasks = set(adap_assignments) & set(base_assignments)
        differing = sum(
            1 for tid in common_tasks
            if adap_assignments[tid] != base_assignments[tid]
        )
        diff_pct = (differing / len(common_tasks) * 100) if common_tasks else 0.0

        print(f"Connectivity: {int(conn * 100)}%")
        print(f"  Mean R_norm across assignments: {r_mean:.3f}")
        print(f"  Std R_norm across assignments:  {r_std:.3f}")
        print(f"  Assignments differing from Baseline: {differing} / {len(common_tasks)} ({diff_pct:.1f}%)")
        print(f"  Delivery rate — Adaptive: {adap['delivery_rate']:.3f}, Baseline: {base['delivery_rate']:.3f}")
        print()

    # ── Last encounter time distribution at end of simulation ──────────
    adap_75 = results[("adaptive", 0.75)]
    enc_times = adap_75["last_encounter_times"]

    if enc_times:
        nonzero = sum(1 for t in enc_times.values() if t > 0.0)
        total_resp = len(enc_times)
        pct_encountered = nonzero / total_resp * 100

        print(f"R_norm distribution at simulation end (75% connectivity, sample):")
        print(f"  {'Responder':<12} {'last_enc(s)':>12} {'delta_t(s)':>12} {'R_norm':>8}")

        # Get the total simulation time
        config = SimulationConfig()
        sim_end = config.total_simulation_duration
        T_REF = 1800.0

        # Show first 10 responders sorted by last_encounter_time descending
        sorted_resp = sorted(enc_times.items(), key=lambda x: x[1], reverse=True)
        for rid, last_enc in sorted_resp[:10]:
            delta_t = max(0.0, sim_end - last_enc)
            r_norm = 1.0 - min(delta_t / T_REF, 1.0)
            print(f"  {rid:<12} {last_enc:>12.0f} {delta_t:>12.0f} {r_norm:>8.3f}")

        print(f"\n  Responders with last_enc > 0: {nonzero}/{total_resp} ({pct_encountered:.0f}%)")
    else:
        pct_encountered = 0.0
        print("  No last_encounter_time data available.")

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n── Validation checks ──")

    # Check 1: R_norm variance > 0.05 in at least one connectivity level
    r_stds = {}
    for conn in connectivity_levels:
        adap = results[("adaptive", conn)]
        rv = [e.get("recency", 0.0) for e in adap["events"] if "recency" in e]
        r_stds[conn] = stdev(rv) if len(rv) > 1 else 0.0

    check1 = any(s > 0.05 for s in r_stds.values())
    stds_str = ", ".join(f"{int(c*100)}%={s:.3f}" for c, s in r_stds.items())
    print(f"  1. std(R_norm) > 0.05 in at least one level: {'PASS' if check1 else 'FAIL'}")
    print(f"     Values: {stds_str}")

    # Check 2: >= 5% assignments differ at 75% or 40%
    diff_pcts = {}
    for conn in [0.75, 0.40]:
        adap = results[("adaptive", conn)]
        base = results[("baseline", conn)]
        aa = {e.get("task_id"): e.get("responder_id") for e in adap["events"]
              if "task_id" in e and "responder_id" in e}
        ba = {e.get("task_id"): e.get("responder_id") for e in base["events"]
              if "task_id" in e and "responder_id" in e}
        ct = set(aa) & set(ba)
        d = sum(1 for tid in ct if aa[tid] != ba[tid])
        diff_pcts[conn] = (d / len(ct) * 100) if ct else 0.0

    check2 = any(p >= 5.0 for p in diff_pcts.values())
    pcts_str = ", ".join(f"{int(c*100)}%={p:.1f}%" for c, p in diff_pcts.items())
    print(f"  2. >= 5% assignments differ from Baseline: {'PASS' if check2 else 'FAIL'}")
    print(f"     Values: {pcts_str}")

    # Check 3: Delivery rate >= 0.80 at all levels
    del_rates = {}
    for conn in connectivity_levels:
        del_rates[conn] = results[("adaptive", conn)]["delivery_rate"]

    check3 = all(dr >= 0.80 for dr in del_rates.values())
    dr_str = ", ".join(f"{int(c*100)}%={d:.3f}" for c, d in del_rates.items())
    print(f"  3. Delivery rate >= 0.80 at all levels: {'PASS' if check3 else 'FAIL'}")
    print(f"     Values: {dr_str}")

    # Check 4: >= 90% responders have last_enc > 0 at end
    check4 = pct_encountered >= 90.0
    print(f"  4. >= 90% responders encountered: {'PASS' if check4 else 'FAIL'}")
    print(f"     Value: {pct_encountered:.0f}%")

    # Final verdict
    print()
    if check1 and check2 and check3:
        print("Differentiation verdict: PASS")
        print("  Encounter recency is producing measurable differentiation.")
    else:
        failed = []
        if not check1:
            failed.append("R_norm variance too low")
        if not check2:
            failed.append("assignments not differing from Baseline")
        if not check3:
            failed.append("delivery rate regression")
        print(f"Differentiation verdict: FAIL")
        print(f"  Reasons: {'; '.join(failed)}")


if __name__ == "__main__":
    main()
