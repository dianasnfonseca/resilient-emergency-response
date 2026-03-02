"""
Real-time animation of ERCS simulation: Adaptive vs Baseline side-by-side.

Shows two synchronized panels with:
- Node movement (Random Waypoint mobility)
- Dynamic edges (100m radio range)
- Task locations by urgency (stars: red=HIGH, orange=MEDIUM, green=LOW)
- Message delivery flash effects
- Live metrics counters
- Buffer utilisation indicators (node size)
- PRoPHET predictability snapshots

Usage:
    python scripts/run_animation.py --duration 600 --sample-interval 10
    python scripts/run_animation.py --connectivity 0.40 --save outputs/animation.gif
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.animation as mpl_animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ercs.communication.prophet import MessageStatus
from ercs.config.parameters import AlgorithmType, SimulationConfig, UrgencyLevel
from ercs.network.topology import NodeType
from ercs.simulation.engine import SimulationEngine, SimulationEventType
from ercs.visualization.plots import COLORS, apply_thesis_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URGENCY_COLORS: dict[UrgencyLevel, str] = {
    UrgencyLevel.HIGH: "#D62728",
    UrgencyLevel.MEDIUM: "#FF7F0E",
    UrgencyLevel.LOW: "#2CA02C",
}

COORDINATION_COLOR = "#9467BD"
EDGE_COLOR = "#999999"
EDGE_ALPHA = 0.5

ZONE_STYLES = {
    "incident": {"facecolor": "#FFF3E0", "edgecolor": "#FF9800", "linestyle": "--"},
    "coordination": {"facecolor": "#E8EAF6", "edgecolor": "#3F51B5", "linestyle": "--"},
}

# Node size range for buffer utilisation indicator
NODE_SIZE_MIN = 20
NODE_SIZE_MAX = 60
# Number of past positions to keep for movement trails
TRAIL_LENGTH = 5


# ---------------------------------------------------------------------------
# Frame data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskSnapshot:
    """Snapshot of a single task's state at a point in time."""

    task_id: str
    x: float
    y: float
    urgency: UrgencyLevel
    status: str
    assigned_to: str | None


@dataclass
class BufferSnapshot:
    """Per-node buffer state at a point in time."""

    message_count: int
    utilisation: float  # 0.0 – 1.0
    message_ids: list[str]


@dataclass
class MetricsSnapshot:
    """Cumulative simulation metrics at a point in time."""

    tasks_created: int
    tasks_assigned: int
    messages_created: int
    messages_delivered: int
    messages_expired: int


@dataclass
class ForwardingEntry:
    """Single message hop record."""

    timestamp: float
    message_id: str
    from_node: str
    to_node: str
    reason: str  # "delivered", "forwarded", "buffer_full", etc.


@dataclass
class FrameData:
    """Complete world-state snapshot for one animation frame."""

    timestamp: float
    node_positions: dict[str, tuple[float, float]]
    node_types: dict[str, str]  # node_id -> "coordination" or "mobile_responder"
    active_edges: list[tuple[str, str]]
    tasks: list[TaskSnapshot]
    metrics: MetricsSnapshot
    recent_deliveries: list[str] = field(default_factory=list)
    recent_assignments: list[str] = field(default_factory=list)
    # Extended fields for diagnostics
    predictabilities: dict[tuple[str, str], float] = field(default_factory=dict)
    buffer_summary: dict[str, BufferSnapshot] = field(default_factory=dict)
    edge_count: int = 0
    node_trails: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AnimationEngine — captures per-frame state during simulation
# ---------------------------------------------------------------------------


class AnimationEngine(SimulationEngine):
    """
    Subclass of SimulationEngine that captures world-state snapshots
    at regular intervals for later animation.

    Usage:
        engine = AnimationEngine(
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
            sample_interval=30.0,
        )
        engine.run()
        frames = engine.get_frames()
    """

    def __init__(
        self,
        *args,
        sample_interval: float = 30.0,
        flash_window: float = 60.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_interval = sample_interval
        self.flash_window = flash_window
        self._frames: list[FrameData] = []
        self._last_sample_time: float = -sample_interval
        self._delivery_log: list[tuple[str, float]] = []
        self._assignment_log: list[tuple[str, float]] = []
        self._forwarding_log: list[ForwardingEntry] = []
        self._trail_history: dict[str, list[tuple[float, float]]] = {}

    def _handle_mobility_update(self, event, results):
        """Override to capture snapshots at sample intervals."""
        super()._handle_mobility_update(event, results)

        if event.timestamp - self._last_sample_time >= self.sample_interval:
            self._capture_frame(event.timestamp, results)
            self._last_sample_time = event.timestamp

    def _handle_coordination_cycle(self, event, results):
        """Override to track assignment times for flash effects."""
        old_assigned = results.tasks_assigned
        super()._handle_coordination_cycle(event, results)

        if results.tasks_assigned > old_assigned:
            for ev in reversed(self._events):
                if ev.event_type == SimulationEventType.TASK_ASSIGNED:
                    tid = ev.data.get("task_id", "")
                    if not any(t == tid for t, _ in self._assignment_log[-50:]):
                        self._assignment_log.append((tid, event.timestamp))
                elif ev.timestamp < event.timestamp:
                    break

    def _process_delivered_messages(self, delivered, timestamp, results):
        """Override to track all transmission results for diagnostics."""
        for result in delivered:
            self._forwarding_log.append(
                ForwardingEntry(
                    timestamp=timestamp,
                    message_id=result.message.message_id,
                    from_node=result.source_node,
                    to_node=result.target_node,
                    reason=result.reason if not result.success else (
                        "delivered" if result.target_node == result.message.destination_id
                        else "forwarded"
                    ),
                )
            )
            if result.success:
                self._delivery_log.append(
                    (result.message.message_id, timestamp)
                )
        super()._process_delivered_messages(delivered, timestamp, results)

    def _capture_frame(self, timestamp: float, results) -> None:
        """Capture current world state as a FrameData."""
        positions = {}
        types = {}
        for node_id, node in self._topology.nodes.items():
            positions[node_id] = (node.x, node.y)
            types[node_id] = node.node_type.value

        edges = list(self._topology.graph.edges())

        # Update movement trails (keep last TRAIL_LENGTH positions)
        for node_id, pos in positions.items():
            if types.get(node_id) == NodeType.MOBILE_RESPONDER.value:
                trail = self._trail_history.setdefault(node_id, [])
                trail.append(pos)
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)
        node_trails = {
            nid: list(trail) for nid, trail in self._trail_history.items()
        }

        task_snaps = []
        for task in self._scenario.tasks:
            if task.creation_time <= timestamp:
                task_snaps.append(
                    TaskSnapshot(
                        task_id=task.task_id,
                        x=task.target_location_x,
                        y=task.target_location_y,
                        urgency=task.urgency,
                        status=task.status.value,
                        assigned_to=task.assigned_to,
                    )
                )

        metrics = MetricsSnapshot(
            tasks_created=results.total_tasks,
            tasks_assigned=results.tasks_assigned,
            messages_created=results.messages_created,
            messages_delivered=results.messages_delivered,
            messages_expired=results.messages_expired,
        )

        # Capture predictability snapshot (only P > 0.01 to keep it compact)
        pred_snapshot: dict[tuple[str, str], float] = {}
        for node_id in self._communication.buffers:
            preds = self._communication.predictability.get_all_predictabilities(node_id)
            for dest_id, p_val in preds.items():
                if p_val > 0.01:
                    pred_snapshot[(node_id, dest_id)] = p_val

        # Capture buffer summary
        buf_summary: dict[str, BufferSnapshot] = {}
        for node_id, buf in self._communication.buffers.items():
            msg_ids = [m.message_id for m in buf]
            buf_summary[node_id] = BufferSnapshot(
                message_count=buf.message_count,
                utilisation=buf.utilisation,
                message_ids=msg_ids,
            )

        cutoff = timestamp - self.flash_window
        recent_del = [mid for mid, t in self._delivery_log if t >= cutoff]
        recent_asgn = [tid for tid, t in self._assignment_log if t >= cutoff]

        self._frames.append(
            FrameData(
                timestamp=timestamp,
                node_positions=positions,
                node_types=types,
                active_edges=edges,
                tasks=task_snaps,
                metrics=metrics,
                recent_deliveries=recent_del,
                recent_assignments=recent_asgn,
                predictabilities=pred_snapshot,
                buffer_summary=buf_summary,
                edge_count=len(edges),
                node_trails=node_trails,
            )
        )

    def get_frames(self) -> list[FrameData]:
        """Return captured frames."""
        return list(self._frames)

    def get_forwarding_log(self) -> list[ForwardingEntry]:
        """Return the complete forwarding log."""
        return list(self._forwarding_log)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_paired_simulation(
    config: SimulationConfig | None = None,
    connectivity_level: float = 0.75,
    seed: int = 42,
    sample_interval: float = 30.0,
    progress_callback=None,
) -> tuple[list[FrameData], list[FrameData], list[ForwardingEntry], list[ForwardingEntry]]:
    """
    Run both Adaptive and Baseline with the same seed and capture frames.

    Returns:
        (adaptive_frames, baseline_frames, adaptive_forwarding_log, baseline_forwarding_log)
    """
    config = config or SimulationConfig()

    if progress_callback:
        progress_callback("Running Adaptive simulation...")

    adaptive_engine = AnimationEngine(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=connectivity_level,
        random_seed=seed,
        sample_interval=sample_interval,
    )
    adaptive_engine.run()
    adaptive_frames = adaptive_engine.get_frames()
    adaptive_fwd = adaptive_engine.get_forwarding_log()

    if progress_callback:
        progress_callback("Running Baseline simulation...")

    baseline_engine = AnimationEngine(
        config=config,
        algorithm_type=AlgorithmType.BASELINE,
        connectivity_level=connectivity_level,
        random_seed=seed,
        sample_interval=sample_interval,
    )
    baseline_engine.run()
    baseline_frames = baseline_engine.get_frames()
    baseline_fwd = baseline_engine.get_forwarding_log()

    return adaptive_frames, baseline_frames, adaptive_fwd, baseline_fwd


# ---------------------------------------------------------------------------
# Animation rendering
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_zones(ax, config: SimulationConfig) -> None:
    """Draw incident and coordination zone rectangles."""
    iz = config.network.incident_zone
    cz = config.network.coordination_zone

    ax.add_patch(
        plt.Rectangle(
            (iz.origin_x, iz.origin_y),
            iz.width_m,
            iz.height_m,
            fill=True,
            linewidth=1.5,
            **ZONE_STYLES["incident"],
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (cz.origin_x, cz.origin_y),
            cz.width_m,
            cz.height_m,
            fill=True,
            linewidth=1.5,
            **ZONE_STYLES["coordination"],
        )
    )


def create_animation(
    adaptive_frames: list[FrameData],
    baseline_frames: list[FrameData],
    config: SimulationConfig | None = None,
    fps: int = 30,
    save_path: str | None = None,
) -> mpl_animation.FuncAnimation:
    """
    Create side-by-side FuncAnimation from captured frames.

    Args:
        adaptive_frames: Frames from Adaptive simulation
        baseline_frames: Frames from Baseline simulation
        config: Simulation config (for zone dimensions)
        fps: Frames per second
        save_path: If provided, save animation to this path (.gif or .mp4)

    Returns:
        matplotlib FuncAnimation object
    """
    config = config or SimulationConfig()
    apply_thesis_style()

    n_frames = min(len(adaptive_frames), len(baseline_frames))
    if n_frames == 0:
        raise ValueError("No frames to animate")

    area = config.network.simulation_area
    x_min = area.origin_x - 100
    x_max = area.origin_x + area.width_m + 100
    y_min = area.origin_y - 100
    y_max = area.origin_y + area.height_m + 100

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(20, 8), constrained_layout=True
    )

    # Identify node types from first frame
    first_frame = adaptive_frames[0]
    coord_ids = [
        nid
        for nid, ntype in first_frame.node_types.items()
        if ntype == NodeType.COORDINATION.value
    ]
    mobile_ids = [
        nid
        for nid, ntype in first_frame.node_types.items()
        if ntype == NodeType.MOBILE_RESPONDER.value
    ]

    # Set up axes
    panels = [
        (ax_left, "Adaptive", COLORS["adaptive"], adaptive_frames),
        (ax_right, "Baseline", COLORS["baseline"], baseline_frames),
    ]

    # Per-panel artists
    edge_collections = []
    mobile_scatters = []
    trail_scatters = []
    coord_scatters = []
    task_pending_scatters = {u: [] for u in UrgencyLevel}
    task_assigned_scatters = {u: [] for u in UrgencyLevel}
    metrics_texts = []
    time_texts = []
    delivery_scatters = []

    for ax, title, color, frames in panels:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, color=color, fontsize=16, fontweight="bold")
        ax.set_xlabel("X (metres)", fontsize=10)
        ax.set_ylabel("Y (metres)", fontsize=10)

        _draw_zones(ax, config)

        # Edges (improved: thicker, darker)
        lc = LineCollection([], colors=EDGE_COLOR, alpha=EDGE_ALPHA, linewidths=0.8)
        ax.add_collection(lc)
        edge_collections.append(lc)

        # Movement trails (fading dots behind mobile nodes)
        sc_trail = ax.scatter(
            [], [], s=8, c=color, alpha=0.15, zorder=2, edgecolors="none",
        )
        trail_scatters.append(sc_trail)

        # Mobile nodes (size varies with buffer utilisation)
        sc_mobile = ax.scatter(
            [], [], s=NODE_SIZE_MIN, c=color, alpha=0.8, zorder=3,
            edgecolors="white", linewidths=0.3,
        )
        mobile_scatters.append(sc_mobile)

        # Coordination nodes
        sc_coord = ax.scatter(
            [], [], s=120, c=COORDINATION_COLOR, marker="s", zorder=4, edgecolors="k"
        )
        coord_scatters.append(sc_coord)

        # Task markers — pending (bright) and assigned (dimmed)
        for urgency in UrgencyLevel:
            sc_pending = ax.scatter(
                [], [], s=60, c=URGENCY_COLORS[urgency], marker="*",
                alpha=0.8, zorder=2,
            )
            task_pending_scatters[urgency].append(sc_pending)

            sc_assigned = ax.scatter(
                [], [], s=40, c=URGENCY_COLORS[urgency], marker="*",
                alpha=0.15, zorder=1,
            )
            task_assigned_scatters[urgency].append(sc_assigned)

        # Delivery flash markers
        sc_del = ax.scatter(
            [], [], s=200, facecolors="none", edgecolors="#FFD700", linewidths=2,
            marker="o", zorder=5, alpha=0.8,
        )
        delivery_scatters.append(sc_del)

        # Metrics text
        mt = ax.text(
            0.98,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
            zorder=10,
        )
        metrics_texts.append(mt)

        # Timestamp
        tt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            zorder=10,
        )
        time_texts.append(tt)

    # Shared title
    fig.suptitle(
        "ERCS: Adaptive vs Baseline Coordination",
        fontsize=16,
        fontweight="bold",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["adaptive"], label="Adaptive nodes"),
        mpatches.Patch(color=COLORS["baseline"], label="Baseline nodes"),
        mpatches.Patch(color=COORDINATION_COLOR, label="Coordination nodes"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=URGENCY_COLORS[UrgencyLevel.HIGH], markersize=10, label="HIGH task"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=URGENCY_COLORS[UrgencyLevel.MEDIUM], markersize=10, label="MEDIUM task"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=URGENCY_COLORS[UrgencyLevel.LOW], markersize=10, label="LOW task"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#FFD700", markeredgewidth=2, markersize=10, label="Message delivered"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    def update(frame_idx):
        artists = []

        for panel_idx, (ax, title, color, frames) in enumerate(panels):
            frame = frames[frame_idx]

            # -- Edges --
            segments = []
            for na, nb in frame.active_edges:
                if na in frame.node_positions and nb in frame.node_positions:
                    pa = frame.node_positions[na]
                    pb = frame.node_positions[nb]
                    segments.append([pa, pb])
            edge_collections[panel_idx].set_segments(segments)
            artists.append(edge_collections[panel_idx])

            # -- Movement trails --
            trail_pts = []
            for nid in mobile_ids:
                trail = frame.node_trails.get(nid, [])
                # Skip the last point (that's the current position)
                trail_pts.extend(trail[:-1] if len(trail) > 1 else [])
            if trail_pts:
                trail_scatters[panel_idx].set_offsets(np.array(trail_pts))
            else:
                trail_scatters[panel_idx].set_offsets(np.empty((0, 2)))
            artists.append(trail_scatters[panel_idx])

            # -- Mobile nodes (size scaled by buffer utilisation) --
            m_positions = []
            m_sizes = []
            for nid in mobile_ids:
                if nid in frame.node_positions:
                    m_positions.append(frame.node_positions[nid])
                    buf = frame.buffer_summary.get(nid)
                    util = buf.utilisation if buf else 0.0
                    size = NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * util
                    m_sizes.append(size)
            if m_positions:
                mobile_scatters[panel_idx].set_offsets(np.array(m_positions))
                mobile_scatters[panel_idx].set_sizes(m_sizes)
            else:
                mobile_scatters[panel_idx].set_offsets(np.empty((0, 2)))
            artists.append(mobile_scatters[panel_idx])

            # -- Coordination nodes --
            cx = [frame.node_positions[nid][0] for nid in coord_ids if nid in frame.node_positions]
            cy = [frame.node_positions[nid][1] for nid in coord_ids if nid in frame.node_positions]
            coord_scatters[panel_idx].set_offsets(np.column_stack([cx, cy]) if cx else np.empty((0, 2)))
            artists.append(coord_scatters[panel_idx])

            # -- Tasks by urgency (pending = bright, assigned = dimmed) --
            for urgency in UrgencyLevel:
                pending = [t for t in frame.tasks if t.urgency == urgency and t.status == "pending"]
                assigned = [t for t in frame.tasks if t.urgency == urgency and t.status == "assigned"]

                if pending:
                    task_pending_scatters[urgency][panel_idx].set_offsets(
                        np.array([(t.x, t.y) for t in pending])
                    )
                else:
                    task_pending_scatters[urgency][panel_idx].set_offsets(np.empty((0, 2)))
                artists.append(task_pending_scatters[urgency][panel_idx])

                if assigned:
                    task_assigned_scatters[urgency][panel_idx].set_offsets(
                        np.array([(t.x, t.y) for t in assigned])
                    )
                else:
                    task_assigned_scatters[urgency][panel_idx].set_offsets(np.empty((0, 2)))
                artists.append(task_assigned_scatters[urgency][panel_idx])

            # -- Delivery flash markers --
            if frame.recent_deliveries:
                del_positions = []
                for mid in frame.recent_deliveries:
                    for t in frame.tasks:
                        if t.assigned_to and t.status == "assigned":
                            pos = frame.node_positions.get(t.assigned_to)
                            if pos:
                                del_positions.append(pos)
                                break
                if del_positions:
                    delivery_scatters[panel_idx].set_offsets(np.array(del_positions))
                else:
                    delivery_scatters[panel_idx].set_offsets(np.empty((0, 2)))
            else:
                delivery_scatters[panel_idx].set_offsets(np.empty((0, 2)))
            artists.append(delivery_scatters[panel_idx])

            # -- Metrics text (enriched) --
            m = frame.metrics
            dr = m.messages_delivered / m.messages_created if m.messages_created > 0 else 0
            ar = m.tasks_assigned / m.tasks_created if m.tasks_created > 0 else 0
            total_buf_msgs = sum(
                b.message_count for b in frame.buffer_summary.values()
            ) if frame.buffer_summary else 0
            metrics_texts[panel_idx].set_text(
                f"Tasks: {m.tasks_assigned}/{m.tasks_created} ({ar:.0%})\n"
                f"Msgs:  {m.messages_delivered}/{m.messages_created} ({dr:.0%})\n"
                f"Edges: {frame.edge_count}  Buffers: {total_buf_msgs}\n"
                f"Expired: {m.messages_expired}"
            )
            artists.append(metrics_texts[panel_idx])

            # -- Timestamp --
            time_texts[panel_idx].set_text(
                f"t = {_format_time(frame.timestamp)}"
            )
            artists.append(time_texts[panel_idx])

        return artists

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    if save_path:
        print(f"Saving animation to {save_path} ({n_frames} frames at {fps} fps)...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps, dpi=100)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps, dpi=100)
        print(f"Saved: {save_path}")

    return anim
