#!/usr/bin/env python3
"""
Run ERCS visualizations: animation, predictability graphs, message journeys.

Modes:
    animation       Side-by-side Adaptive vs Baseline animation (default)
    predictability  PRoPHET predictability network graph at a given time
    heatmap         Predictability matrix heatmap at a given time
    evolution       Predictability time series for top node pairs
    journey         Spatial path + timeline for a specific message
    paths           Overview map of all message paths (delivered vs not)

Usage:
    # Side-by-side animation (default)
    python scripts/run_animation.py --duration 3600 --sample-interval 30

    # PRoPHET graph at t=1800
    python scripts/run_animation.py --mode predictability --duration 3600 --time 1800

    # Heatmap of coord→mobile predictabilities
    python scripts/run_animation.py --mode heatmap --duration 3600 --time 1800

    # Predictability evolution over time
    python scripts/run_animation.py --mode evolution --duration 3600

    # First message journey
    python scripts/run_animation.py --mode journey --duration 3600 --message-id first

    # All message paths overview
    python scripts/run_animation.py --mode paths --duration 3600

    # Save any mode to file
    python scripts/run_animation.py --mode predictability --duration 3600 --time 1800 --save outputs/pred.png
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import SimulationConfig
from ercs.visualization.animation import (
    create_animation,
    run_paired_simulation,
)


def _run_simulation(args, config):
    """Run paired simulation and return frames + logs."""
    def progress(msg):
        print(f"  {msg}")

    t0 = time.time()
    adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd = run_paired_simulation(
        config=config,
        connectivity_level=args.connectivity,
        seed=args.seed,
        sample_interval=args.sample_interval,
        progress_callback=progress,
    )
    elapsed = time.time() - t0

    print(f"\nSimulations completed in {elapsed:.1f}s")
    print(f"  Adaptive: {len(adaptive_frames)} frames, {len(adaptive_fwd)} forwarding events")
    print(f"  Baseline: {len(baseline_frames)} frames, {len(baseline_fwd)} forwarding events")

    if adaptive_frames:
        af = adaptive_frames[-1].metrics
        bf = baseline_frames[-1].metrics
        print(f"\nFinal metrics:")
        print(f"  Adaptive — assigned: {af.tasks_assigned}/{af.tasks_created}, "
              f"delivered: {af.messages_delivered}/{af.messages_created}")
        print(f"  Baseline — assigned: {bf.tasks_assigned}/{bf.tasks_created}, "
              f"delivered: {bf.messages_delivered}/{bf.messages_created}")

    return adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd


def _find_frame_at_time(frames, target_time):
    """Find the frame closest to target_time."""
    best = None
    best_dt = float("inf")
    for f in frames:
        dt = abs(f.timestamp - target_time)
        if dt < best_dt:
            best_dt = dt
            best = f
    return best


def _mode_animation(args, config):
    """Standard side-by-side animation."""
    adaptive_frames, baseline_frames, _, _ = _run_simulation(args, config)

    if not adaptive_frames or not baseline_frames:
        print("ERROR: No frames captured.")
        return 1

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    print("\nCreating animation...")
    anim = create_animation(
        adaptive_frames, baseline_frames,
        config=config, fps=args.fps, save_path=args.save,
    )

    if not args.no_display and not args.save:
        import matplotlib.pyplot as plt
        print("Displaying animation (close window to exit)...")
        plt.show()

    return 0


def _mode_predictability(args, config):
    """PRoPHET predictability graph at a specific time."""
    from ercs.visualization.diagnostics import plot_predictability_graph

    adaptive_frames, baseline_frames, _, _ = _run_simulation(args, config)

    target = args.time if args.time is not None else config.scenario.simulation_duration_seconds / 2
    print(f"\nGenerating predictability graph at t={target:.0f}s...")

    for label, frames in [("Adaptive", adaptive_frames), ("Baseline", baseline_frames)]:
        frame = _find_frame_at_time(frames, target)
        if frame:
            fig = plot_predictability_graph(frame, config, algorithm_label=label)
            if args.save:
                stem = Path(args.save).stem
                suffix = Path(args.save).suffix
                path = str(Path(args.save).parent / f"{stem}_{label.lower()}{suffix}")
                fig.savefig(path, dpi=150)
                print(f"  Saved: {path}")

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def _mode_heatmap(args, config):
    """Predictability heatmap at a specific time."""
    from ercs.visualization.diagnostics import plot_predictability_heatmap

    adaptive_frames, baseline_frames, _, _ = _run_simulation(args, config)

    target = args.time if args.time is not None else config.scenario.simulation_duration_seconds / 2
    print(f"\nGenerating predictability heatmap at t={target:.0f}s...")

    for label, frames in [("Adaptive", adaptive_frames), ("Baseline", baseline_frames)]:
        frame = _find_frame_at_time(frames, target)
        if frame:
            fig = plot_predictability_heatmap(frame, algorithm_label=label)
            if args.save:
                stem = Path(args.save).stem
                suffix = Path(args.save).suffix
                path = str(Path(args.save).parent / f"{stem}_{label.lower()}{suffix}")
                fig.savefig(path, dpi=150)
                print(f"  Saved: {path}")

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def _mode_evolution(args, config):
    """Predictability time series."""
    from ercs.visualization.diagnostics import plot_predictability_evolution

    adaptive_frames, baseline_frames, _, _ = _run_simulation(args, config)

    print("\nGenerating predictability evolution plots...")

    for label, frames in [("Adaptive", adaptive_frames), ("Baseline", baseline_frames)]:
        fig = plot_predictability_evolution(frames, algorithm_label=label)
        if args.save:
            stem = Path(args.save).stem
            suffix = Path(args.save).suffix
            path = str(Path(args.save).parent / f"{stem}_{label.lower()}{suffix}")
            fig.savefig(path, dpi=150)
            print(f"  Saved: {path}")

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def _mode_journey(args, config):
    """Message journey tracker."""
    from ercs.visualization.diagnostics import (
        find_message_journeys,
        plot_message_journey,
    )

    adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd = _run_simulation(args, config)

    for label, frames, fwd_log in [
        ("Adaptive", adaptive_frames, adaptive_fwd),
        ("Baseline", baseline_frames, baseline_fwd),
    ]:
        journeys = find_message_journeys(fwd_log)
        if not journeys:
            print(f"  {label}: no messages forwarded")
            continue

        # Select message
        msg_id = args.message_id
        if msg_id == "first":
            msg_id = min(journeys.keys(), key=lambda m: journeys[m][0].timestamp)
        elif msg_id == "random":
            import random
            msg_id = random.choice(list(journeys.keys()))

        if msg_id not in journeys:
            print(f"  {label}: message '{msg_id}' not found in forwarding log")
            print(f"  Available: {list(journeys.keys())[:5]}...")
            continue

        print(f"\n{label}: tracking message {msg_id[:16]}... ({len(journeys[msg_id])} hops)")

        fig = plot_message_journey(
            msg_id, journeys[msg_id], frames, config, algorithm_label=label,
        )
        if args.save:
            stem = Path(args.save).stem
            suffix = Path(args.save).suffix
            path = str(Path(args.save).parent / f"{stem}_{label.lower()}{suffix}")
            fig.savefig(path, dpi=150)
            print(f"  Saved: {path}")

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def _mode_paths(args, config):
    """All message paths overview."""
    from ercs.visualization.diagnostics import plot_all_message_paths

    adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd = _run_simulation(args, config)

    print("\nGenerating message paths overview...")

    for label, frames, fwd_log in [
        ("Adaptive", adaptive_frames, adaptive_fwd),
        ("Baseline", baseline_frames, baseline_fwd),
    ]:
        fig = plot_all_message_paths(frames, fwd_log, config, algorithm_label=label)
        if args.save:
            stem = Path(args.save).stem
            suffix = Path(args.save).suffix
            path = str(Path(args.save).parent / f"{stem}_{label.lower()}{suffix}")
            fig.savefig(path, dpi=150)
            print(f"  Saved: {path}")

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ERCS Visualization Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["animation", "predictability", "heatmap", "evolution", "journey", "paths"],
        default="animation",
        help="Visualization mode (default: animation)",
    )
    parser.add_argument(
        "--connectivity",
        type=float,
        default=0.75,
        help="Connectivity level (default: 0.75). Options: 0.75, 0.40, 0.20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override simulation duration in seconds (default: from config, 6000)",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=30.0,
        help="Seconds between captured frames (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation frames per second (default: 30)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save output to file (.gif, .mp4, .png, .pdf)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the interactive matplotlib window",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Snapshot time in seconds (for predictability/heatmap modes)",
    )
    parser.add_argument(
        "--message-id",
        type=str,
        default="first",
        help="Message ID to track, or 'first'/'random' (for journey mode)",
    )
    args = parser.parse_args()

    # Build config
    config = SimulationConfig()
    if args.duration is not None:
        config = config.model_copy(
            update={
                "scenario": config.scenario.model_copy(
                    update={"simulation_duration_seconds": args.duration}
                )
            }
        )

    duration = config.scenario.simulation_duration_seconds
    est_frames = int(duration / args.sample_interval)

    print("=" * 60)
    print(f"ERCS Visualization — mode: {args.mode}")
    print("=" * 60)
    print(f"  Connectivity: {args.connectivity:.0%}")
    print(f"  Seed:         {args.seed}")
    print(f"  Duration:     {duration}s ({duration/3600:.1f}h)")
    print(f"  Sample every: {args.sample_interval}s (~{est_frames} frames)")
    if args.mode == "animation":
        print(f"  FPS:          {args.fps}")
    if args.time is not None:
        print(f"  Snapshot at:  {args.time}s")
    if args.save:
        print(f"  Save to:      {args.save}")
    print("=" * 60)

    modes = {
        "animation": _mode_animation,
        "predictability": _mode_predictability,
        "heatmap": _mode_heatmap,
        "evolution": _mode_evolution,
        "journey": _mode_journey,
        "paths": _mode_paths,
    }

    return modes[args.mode](args, config)


if __name__ == "__main__":
    sys.exit(main())
