#!/usr/bin/env python3
"""
Diagnostic: compare ONE coordination cycle between Adaptive and Baseline.

Shows exactly which tasks were assigned, to which responders,
with what predictability/distance — and whether they differ.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import SimulationConfig
from ercs.coordination.algorithms import AlgorithmType
from ercs.simulation.engine import SimulationEngine, SimulationEventType


class CoordinationSpy(SimulationEngine):
    """SimulationEngine that captures coordination cycle details."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_cycles = []

    def _handle_coordination_cycle(self, event, results):
        """Intercept coordination cycle to capture full details."""
        coord_nodes = self._topology.get_coordination_node_ids()
        coord_node = coord_nodes[0] if coord_nodes else "coord_0"

        # Snapshot: all responder positions and predictabilities BEFORE assignment
        responder_ids = self._adapter.get_all_responder_ids()
        responder_info = {}
        for rid in responder_ids:
            rx, ry = self._adapter.get_responder_position(rid)
            pred = self._adapter.get_delivery_predictability(coord_node, rid)
            responder_info[rid] = {"x": rx, "y": ry, "predictability": pred}

        # Snapshot: pending tasks
        pending = list(self._manager._pending_tasks)

        # Run the actual coordination cycle
        super()._handle_coordination_cycle(event, results)

        # Capture what happened
        cycle_data = {
            "timestamp": event.timestamp,
            "coord_node": coord_node,
            "pending_tasks": len(pending),
            "responder_snapshot": responder_info,
            "tasks": [],
        }

        # Get the assignments that were just made
        all_assignments = self._manager._all_assignments
        # The last N assignments are from this cycle
        new_assignments = all_assignments[-(len(all_assignments) - sum(
            len(c["tasks"]) for c in self.captured_cycles
        )):]

        for a in new_assignments:
            rid = a.responder_id
            info = responder_info.get(rid, {})
            # Distance from task to responder
            task_x, task_y = a.task.target_location_x, a.task.target_location_y
            rx, ry = info.get("x", 0), info.get("y", 0)
            dist = ((task_x - rx)**2 + (task_y - ry)**2) ** 0.5

            cycle_data["tasks"].append({
                "task_id": a.task_id[:12],
                "urgency": a.task.urgency.value,
                "task_pos": (task_x, task_y),
                "responder_id": rid,
                "responder_pos": (rx, ry),
                "distance": dist,
                "predictability": a.predictability,
                "created_at": a.task.creation_time,
            })

        self.captured_cycles.append(cycle_data)


def run_one(algorithm, config, seed=42, connectivity=0.20):
    """Run simulation and return first non-empty coordination cycle."""
    algo = AlgorithmType.ADAPTIVE if algorithm == "adaptive" else AlgorithmType.BASELINE
    engine = CoordinationSpy(
        config=config,
        algorithm_type=algo,
        connectivity_level=connectivity,
        random_seed=seed,
    )
    engine.run()
    return engine.captured_cycles


def print_cycle(cycle, algorithm):
    """Pretty-print one coordination cycle."""
    print(f"\n{'=' * 70}")
    print(f"  {algorithm.upper()} — Coordination Cycle at t={cycle['timestamp']:.0f}s")
    print(f"  Coord node: {cycle['coord_node']}")
    print(f"  Pending tasks: {cycle['pending_tasks']}")
    print(f"  Assigned: {len(cycle['tasks'])}")
    print(f"{'=' * 70}")

    if not cycle["tasks"]:
        print("  (no assignments)")
        return

    # Header
    print(f"  {'#':>2}  {'Urg':>3}  {'Task':12}  {'Responder':12}  "
          f"{'Dist':>7}  {'Pred':>8}  {'Task Pos':>16}  {'Resp Pos':>16}")
    print(f"  {'—' * 2}  {'—' * 3}  {'—' * 12}  {'—' * 12}  "
          f"{'—' * 7}  {'—' * 8}  {'—' * 16}  {'—' * 16}")

    for i, t in enumerate(cycle["tasks"]):
        pred_str = f"{t['predictability']:.6f}" if t['predictability'] is not None else "   N/A"
        tx, ty = t["task_pos"]
        rx, ry = t["responder_pos"]
        print(f"  {i + 1:2d}  {t['urgency']:>3}  {t['task_id']:12}  {t['responder_id']:12}  "
              f"{t['distance']:7.0f}m  {pred_str}  ({tx:6.0f},{ty:6.0f})  ({rx:6.0f},{ry:6.0f})")

    # Show responders with P > 0 at this moment
    nonzero_p = {rid: info for rid, info in cycle["responder_snapshot"].items()
                 if info["predictability"] > 0.001}
    print(f"\n  Responders with P > 0.001: {len(nonzero_p)} / {len(cycle['responder_snapshot'])}")
    if nonzero_p:
        sorted_p = sorted(nonzero_p.items(), key=lambda x: -x[1]["predictability"])
        for rid, info in sorted_p[:10]:
            print(f"    {rid:12s}  P = {info['predictability']:.6f}  "
                  f"at ({info['x']:.0f}, {info['y']:.0f})")


def main():
    config = SimulationConfig()
    connectivity = 0.20
    seed = 42

    print(f"Running Adaptive (connectivity={connectivity}, seed={seed})...")
    adaptive_cycles = run_one("adaptive", config, seed, connectivity)

    print(f"Running Baseline (connectivity={connectivity}, seed={seed})...")
    baseline_cycles = run_one("baseline", config, seed, connectivity)

    # Show first cycle with assignments for each
    # The first cycle (t=0) usually has 0 tasks; t=1800 is the first real one
    for cycles, label in [(adaptive_cycles, "Adaptive"), (baseline_cycles, "Baseline")]:
        for cycle in cycles:
            if cycle["tasks"]:
                print_cycle(cycle, label)
                break
        else:
            print(f"\n{label}: No cycles with assignments found!")

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("  COMPARISON: Are the responders different?")
    print(f"{'=' * 70}")

    # Find matching cycles (same timestamp)
    adaptive_by_t = {c["timestamp"]: c for c in adaptive_cycles}
    baseline_by_t = {c["timestamp"]: c for c in baseline_cycles}

    common_times = sorted(set(adaptive_by_t) & set(baseline_by_t))
    for t in common_times:
        ac = adaptive_by_t[t]
        bc = baseline_by_t[t]
        if not ac["tasks"] and not bc["tasks"]:
            continue

        a_assignments = {t["task_id"]: t["responder_id"] for t in ac["tasks"]}
        b_assignments = {t["task_id"]: t["responder_id"] for t in bc["tasks"]}

        # Tasks in common
        common_tasks = set(a_assignments) & set(b_assignments)
        same = sum(1 for tid in common_tasks if a_assignments[tid] == b_assignments[tid])
        diff = sum(1 for tid in common_tasks if a_assignments[tid] != b_assignments[tid])

        print(f"\n  t={t:.0f}s:")
        print(f"    Adaptive assigned: {len(a_assignments)} tasks")
        print(f"    Baseline assigned: {len(b_assignments)} tasks")
        print(f"    Common tasks: {len(common_tasks)}")
        print(f"    Same responder: {same}  |  Different responder: {diff}")

        if diff > 0:
            print(f"\n    Tasks with DIFFERENT responders:")
            print(f"    {'Task':12}  {'Urg':>3}  {'Adaptive':12}  {'A.Dist':>7}  {'A.Pred':>8}  "
                  f"{'Baseline':12}  {'B.Dist':>7}")
            print(f"    {'—' * 12}  {'—' * 3}  {'—' * 12}  {'—' * 7}  {'—' * 8}  "
                  f"{'—' * 12}  {'—' * 7}")
            for tid in sorted(common_tasks):
                if a_assignments[tid] != b_assignments[tid]:
                    at = next(x for x in ac["tasks"] if x["task_id"] == tid)
                    bt = next(x for x in bc["tasks"] if x["task_id"] == tid)
                    pred_str = f"{at['predictability']:.6f}" if at['predictability'] else "   N/A"
                    print(f"    {tid:12}  {at['urgency']:>3}  "
                          f"{at['responder_id']:12}  {at['distance']:7.0f}m  {pred_str}  "
                          f"{bt['responder_id']:12}  {bt['distance']:7.0f}m")

        # Only show first meaningful cycle
        break


if __name__ == "__main__":
    main()
