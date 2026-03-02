#!/usr/bin/env python3
"""
Run ERCS side-by-side animation: Adaptive vs Baseline.

Shows both algorithms running on the same scenario (same seed, topology,
tasks, and mobility) so you can visually compare their behaviour.

Usage:
    # Quick test (10-minute sim, fast)
    python scripts/run_animation.py --duration 600 --sample-interval 10

    # Specific connectivity level
    python scripts/run_animation.py --connectivity 0.40 --seed 42

    # Save as GIF
    python scripts/run_animation.py --duration 600 --sample-interval 10 --save outputs/animation.gif

    # Save as MP4 (requires ffmpeg)
    python scripts/run_animation.py --duration 3600 --save outputs/animation.mp4 --fps 30
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ERCS Animation: Adaptive vs Baseline side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Override simulation duration in seconds (default: from config, 43200)",
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
        help="Save animation to file (.gif or .mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show the interactive matplotlib window",
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
    print("ERCS Animation: Adaptive vs Baseline")
    print("=" * 60)
    print(f"  Connectivity: {args.connectivity:.0%}")
    print(f"  Seed:         {args.seed}")
    print(f"  Duration:     {duration}s ({duration/3600:.1f}h)")
    print(f"  Sample every: {args.sample_interval}s (~{est_frames} frames)")
    print(f"  FPS:          {args.fps}")
    if args.save:
        print(f"  Save to:      {args.save}")
    print("=" * 60)

    def progress(msg):
        print(f"  {msg}")

    t0 = time.time()

    adaptive_frames, baseline_frames = run_paired_simulation(
        config=config,
        connectivity_level=args.connectivity,
        seed=args.seed,
        sample_interval=args.sample_interval,
        progress_callback=progress,
    )

    elapsed = time.time() - t0
    print(f"\nSimulations completed in {elapsed:.1f}s")
    print(f"  Adaptive: {len(adaptive_frames)} frames")
    print(f"  Baseline: {len(baseline_frames)} frames")

    if not adaptive_frames or not baseline_frames:
        print("ERROR: No frames captured. Check simulation duration and sample interval.")
        return 1

    # Show final metrics
    af = adaptive_frames[-1].metrics
    bf = baseline_frames[-1].metrics
    print(f"\nFinal metrics:")
    print(f"  Adaptive — assigned: {af.tasks_assigned}/{af.tasks_created}, "
          f"delivered: {af.messages_delivered}/{af.messages_created}")
    print(f"  Baseline — assigned: {bf.tasks_assigned}/{bf.tasks_created}, "
          f"delivered: {bf.messages_delivered}/{bf.messages_created}")

    # Ensure output directory exists if saving
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nCreating animation...")
    anim = create_animation(
        adaptive_frames,
        baseline_frames,
        config=config,
        fps=args.fps,
        save_path=args.save,
    )

    if not args.no_display and not args.save:
        import matplotlib.pyplot as plt

        print("Displaying animation (close window to exit)...")
        plt.show()
    elif not args.no_display and args.save:
        print("Animation saved. Use --no-display to skip interactive window.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
