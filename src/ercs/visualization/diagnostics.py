"""
Diagnostic visualizations for ERCS simulation analysis.

Provides static plot functions for inspecting simulation internals:
- PRoPHET predictability graph (spatial network view)
- Predictability heatmap (full matrix)
- Predictability evolution (time series)
- Message journey tracker (spatial path + timeline)

All functions take FrameData / ForwardingEntry objects produced by
AnimationEngine and return matplotlib Figure objects.

Usage:
    from ercs.visualization.diagnostics import plot_predictability_graph
    fig = plot_predictability_graph(frame, config)
    fig.savefig("predictability.png")
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from ercs.config.parameters import SimulationConfig
from ercs.network.topology import NodeType
from ercs.visualization.animation import (
    COORDINATION_COLOR,
    URGENCY_COLORS,
    ZONE_STYLES,
    ForwardingEntry,
    FrameData,
    _draw_zones,
    _format_time,
)
from ercs.visualization.plots import COLORS, apply_thesis_style


# ---------------------------------------------------------------------------
# PRoPHET Predictability Graph
# ---------------------------------------------------------------------------


def plot_predictability_graph(
    frame: FrameData,
    config: SimulationConfig | None = None,
    threshold: float = 0.1,
    algorithm_label: str = "",
) -> plt.Figure:
    """
    Network graph snapshot coloured by PRoPHET delivery predictability.

    Args:
        frame: FrameData snapshot at a specific simulation time.
        config: SimulationConfig for zone dimensions.
        threshold: Minimum P value to draw an edge (default 0.1).
        algorithm_label: Optional label for the title.

    Returns:
        matplotlib Figure.
    """
    config = config or SimulationConfig()
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(14, 7))

    area = config.network.simulation_area
    ax.set_xlim(area.origin_x - 50, area.origin_x + area.width_m + 50)
    ax.set_ylim(area.origin_y - 50, area.origin_y + area.height_m + 50)
    ax.set_aspect("equal")

    _draw_zones(ax, config)

    # Draw predictability edges
    norm = Normalize(vmin=threshold, vmax=1.0)
    cmap = plt.cm.viridis

    segments = []
    colors = []
    widths = []

    for (src, dst), p_val in frame.predictabilities.items():
        if p_val < threshold:
            continue
        pos_src = frame.node_positions.get(src)
        pos_dst = frame.node_positions.get(dst)
        if pos_src and pos_dst:
            segments.append([pos_src, pos_dst])
            colors.append(cmap(norm(p_val)))
            widths.append(0.5 + 2.5 * norm(p_val))

    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.6, zorder=1)
        ax.add_collection(lc)

    # Draw nodes
    coord_ids = [nid for nid, nt in frame.node_types.items() if nt == NodeType.COORDINATION.value]
    mobile_ids = [nid for nid, nt in frame.node_types.items() if nt == NodeType.MOBILE_RESPONDER.value]

    mx = [frame.node_positions[n][0] for n in mobile_ids if n in frame.node_positions]
    my = [frame.node_positions[n][1] for n in mobile_ids if n in frame.node_positions]
    ax.scatter(mx, my, s=20, c="#555555", alpha=0.7, zorder=3, label="Mobile responders")

    cx = [frame.node_positions[n][0] for n in coord_ids if n in frame.node_positions]
    cy = [frame.node_positions[n][1] for n in coord_ids if n in frame.node_positions]
    ax.scatter(cx, cy, s=100, c=COORDINATION_COLOR, marker="s", zorder=4,
               edgecolors="k", label="Coordination nodes")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Delivery Predictability P", fontsize=10)

    title = f"PRoPHET Predictability at t = {_format_time(frame.timestamp)}"
    if algorithm_label:
        title = f"{algorithm_label}: {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")

    n_edges = sum(1 for p in frame.predictabilities.values() if p >= threshold)
    ax.text(
        0.02, 0.02,
        f"Edges with P >= {threshold}: {n_edges}",
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round,pad=0.3"},
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Predictability Heatmap
# ---------------------------------------------------------------------------


def plot_predictability_heatmap(
    frame: FrameData,
    node_subset: str = "coord_vs_mobile",
    algorithm_label: str = "",
) -> plt.Figure:
    """
    Heatmap of the delivery predictability matrix.

    Args:
        frame: FrameData snapshot.
        node_subset: "coord_vs_mobile" (2 x 48 matrix) or "all" (50 x 50).
        algorithm_label: Optional label for the title.

    Returns:
        matplotlib Figure.
    """
    apply_thesis_style()

    coord_ids = sorted(
        nid for nid, nt in frame.node_types.items()
        if nt == NodeType.COORDINATION.value
    )
    mobile_ids = sorted(
        nid for nid, nt in frame.node_types.items()
        if nt == NodeType.MOBILE_RESPONDER.value
    )

    if node_subset == "coord_vs_mobile":
        row_ids = coord_ids
        col_ids = mobile_ids
    else:
        row_ids = coord_ids + mobile_ids
        col_ids = coord_ids + mobile_ids

    matrix = np.zeros((len(row_ids), len(col_ids)))
    for i, src in enumerate(row_ids):
        for j, dst in enumerate(col_ids):
            matrix[i, j] = frame.predictabilities.get((src, dst), 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(col_ids) * 0.3), max(4, len(row_ids) * 0.6)))

    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    # Labels
    ax.set_xticks(range(len(col_ids)))
    ax.set_yticks(range(len(row_ids)))
    short_col = [c.replace("mobile_", "m").replace("coord_", "C") for c in col_ids]
    short_row = [r.replace("mobile_", "m").replace("coord_", "C") for r in row_ids]
    ax.set_xticklabels(short_col, fontsize=6, rotation=90)
    ax.set_yticklabels(short_row, fontsize=6)

    # Annotate cells for small matrices
    if len(row_ids) <= 5 and len(col_ids) <= 20:
        for i in range(len(row_ids)):
            for j in range(len(col_ids)):
                val = matrix[i, j]
                if val > 0.01:
                    text_color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Delivery Predictability P", fontsize=10)

    title = f"Predictability Matrix at t = {_format_time(frame.timestamp)}"
    if algorithm_label:
        title = f"{algorithm_label}: {title}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Destination")
    ax.set_ylabel("Source")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Predictability Evolution
# ---------------------------------------------------------------------------


def plot_predictability_evolution(
    frames: list[FrameData],
    node_pairs: list[tuple[str, str]] | None = None,
    top_n: int = 10,
    algorithm_label: str = "",
) -> plt.Figure:
    """
    Time series of PRoPHET predictability values.

    Args:
        frames: List of FrameData snapshots (time-ordered).
        node_pairs: Specific (src, dst) pairs to plot. If None, auto-selects
            the top_n pairs with highest max P across the run.
        top_n: Number of pairs to auto-select when node_pairs is None.
        algorithm_label: Optional label for the title.

    Returns:
        matplotlib Figure.
    """
    apply_thesis_style()

    # Auto-select top pairs if not specified
    if node_pairs is None:
        max_p: dict[tuple[str, str], float] = {}
        for f in frames:
            for pair, p in f.predictabilities.items():
                if p > max_p.get(pair, 0.0):
                    max_p[pair] = p
        sorted_pairs = sorted(max_p.items(), key=lambda x: x[1], reverse=True)
        node_pairs = [pair for pair, _ in sorted_pairs[:top_n]]

    if not node_pairs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No predictability data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))
    timestamps = [f.timestamp for f in frames]

    cmap = plt.cm.tab10
    for idx, (src, dst) in enumerate(node_pairs):
        values = [f.predictabilities.get((src, dst), 0.0) for f in frames]
        short_src = src.replace("mobile_", "m").replace("coord_", "C")
        short_dst = dst.replace("mobile_", "m").replace("coord_", "C")
        ax.plot(timestamps, values, color=cmap(idx % 10), linewidth=1.2,
                alpha=0.8, label=f"{short_src} → {short_dst}")

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Delivery Predictability P")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    title = "PRoPHET Predictability Evolution"
    if algorithm_label:
        title = f"{algorithm_label}: {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Message Journey Tracker
# ---------------------------------------------------------------------------


def find_message_journeys(
    forwarding_log: list[ForwardingEntry],
) -> dict[str, list[ForwardingEntry]]:
    """
    Group forwarding log entries by message_id into sorted hop lists.

    Returns:
        {message_id: [ForwardingEntry, ...]} sorted by timestamp.
    """
    journeys: dict[str, list[ForwardingEntry]] = defaultdict(list)
    for entry in forwarding_log:
        journeys[entry.message_id].append(entry)

    for mid in journeys:
        journeys[mid].sort(key=lambda e: e.timestamp)

    return dict(journeys)


def plot_message_journey(
    message_id: str,
    journey: list[ForwardingEntry],
    frames: list[FrameData],
    config: SimulationConfig | None = None,
    algorithm_label: str = "",
) -> plt.Figure:
    """
    Two-panel figure showing a single message's spatial path and timeline.

    Args:
        message_id: The message to track.
        journey: Sorted list of ForwardingEntry for this message.
        frames: Full list of FrameData (for node positions).
        config: SimulationConfig for zone dimensions.
        algorithm_label: Optional label for the title.

    Returns:
        matplotlib Figure with two panels.
    """
    config = config or SimulationConfig()
    apply_thesis_style()

    fig, (ax_map, ax_timeline) = plt.subplots(1, 2, figsize=(18, 7),
                                               gridspec_kw={"width_ratios": [1.2, 1]})

    # -- Left panel: spatial path --
    area = config.network.simulation_area
    ax_map.set_xlim(area.origin_x - 50, area.origin_x + area.width_m + 50)
    ax_map.set_ylim(area.origin_y - 50, area.origin_y + area.height_m + 50)
    ax_map.set_aspect("equal")
    _draw_zones(ax_map, config)

    # Find node positions from the nearest frame for each hop
    def _pos_at_time(node_id: str, t: float) -> tuple[float, float] | None:
        best = None
        best_dt = float("inf")
        for f in frames:
            dt = abs(f.timestamp - t)
            if dt < best_dt:
                best_dt = dt
                pos = f.node_positions.get(node_id)
                if pos:
                    best = pos
        return best

    # Determine source and destination from the journey entries
    if not journey:
        ax_map.text(0.5, 0.5, "No hops recorded", ha="center", va="center",
                    transform=ax_map.transAxes, fontsize=14)
        fig.tight_layout()
        return fig

    source_node = journey[0].from_node
    # Destination is the to_node of a "delivered" entry, or from the first entry
    dest_node = None
    delivered = False
    for entry in journey:
        if entry.reason == "delivered":
            dest_node = entry.to_node
            delivered = True
            break
    if dest_node is None:
        dest_node = journey[-1].to_node

    # Draw hop arrows
    cmap = plt.cm.Blues
    n_hops = len(journey)
    for idx, entry in enumerate(journey):
        pos_from = _pos_at_time(entry.from_node, entry.timestamp)
        pos_to = _pos_at_time(entry.to_node, entry.timestamp)
        if not pos_from or not pos_to:
            continue

        color_val = 0.3 + 0.7 * (idx / max(n_hops - 1, 1))
        color = cmap(color_val)
        dx = pos_to[0] - pos_from[0]
        dy = pos_to[1] - pos_from[1]

        ax_map.annotate(
            "", xy=pos_to, xytext=pos_from,
            arrowprops=dict(
                arrowstyle="->", color=color, lw=1.5 + idx * 0.3,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=3,
        )

    # Mark source and destination
    src_pos = _pos_at_time(source_node, journey[0].timestamp)
    dst_pos = _pos_at_time(dest_node, journey[-1].timestamp)

    if src_pos:
        ax_map.scatter(*src_pos, s=200, c=COORDINATION_COLOR, marker="*",
                       zorder=5, edgecolors="k", linewidths=1, label="Source")
    if dst_pos:
        marker_color = "#2CA02C" if delivered else "#D62728"
        ax_map.scatter(*dst_pos, s=200, c=marker_color, marker="D",
                       zorder=5, edgecolors="k", linewidths=1,
                       label="Destination (delivered)" if delivered else "Destination (not delivered)")

    ax_map.legend(loc="upper left", fontsize=8)
    ax_map.set_xlabel("X (metres)")
    ax_map.set_ylabel("Y (metres)")
    ax_map.set_title("Spatial Path", fontsize=12, fontweight="bold")

    # -- Right panel: timeline --
    # Collect unique nodes involved
    involved_nodes = []
    seen = set()
    for entry in journey:
        for n in (entry.from_node, entry.to_node):
            if n not in seen:
                involved_nodes.append(n)
                seen.add(n)

    node_y_pos = {n: i for i, n in enumerate(involved_nodes)}

    for idx, entry in enumerate(journey):
        y_from = node_y_pos[entry.from_node]
        y_to = node_y_pos[entry.to_node]
        color = "#2CA02C" if entry.reason == "delivered" else (
            "#2171B5" if entry.reason == "forwarded" else "#D62728"
        )
        ax_timeline.annotate(
            "", xy=(entry.timestamp, y_to), xytext=(entry.timestamp, y_from),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
        ax_timeline.scatter(entry.timestamp, y_from, c=color, s=30, zorder=4)
        ax_timeline.scatter(entry.timestamp, y_to, c=color, s=30, zorder=4)

    ax_timeline.set_yticks(range(len(involved_nodes)))
    short_names = [n.replace("mobile_", "m").replace("coord_", "C") for n in involved_nodes]
    ax_timeline.set_yticklabels(short_names, fontsize=8)
    ax_timeline.set_xlabel("Simulation Time (s)")
    ax_timeline.set_title("Message Timeline", fontsize=12, fontweight="bold")
    ax_timeline.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2171B5", lw=2, label="Forwarded"),
        Line2D([0], [0], color="#2CA02C", lw=2, label="Delivered"),
        Line2D([0], [0], color="#D62728", lw=2, label="Failed"),
    ]
    ax_timeline.legend(handles=legend_elements, loc="upper right", fontsize=8)

    title = f"Message Journey: {message_id[:12]}..."
    if algorithm_label:
        title = f"{algorithm_label}: {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_all_message_paths(
    frames: list[FrameData],
    forwarding_log: list[ForwardingEntry],
    config: SimulationConfig | None = None,
    algorithm_label: str = "",
) -> plt.Figure:
    """
    Overview map showing all message paths coloured by delivery status.

    Green = delivered, red = not delivered, arrow thickness = hop count.

    Args:
        frames: Full list of FrameData.
        forwarding_log: Complete forwarding log.
        config: SimulationConfig for zone dimensions.
        algorithm_label: Optional label for the title.

    Returns:
        matplotlib Figure.
    """
    config = config or SimulationConfig()
    apply_thesis_style()

    fig, ax = plt.subplots(figsize=(14, 7))
    area = config.network.simulation_area
    ax.set_xlim(area.origin_x - 50, area.origin_x + area.width_m + 50)
    ax.set_ylim(area.origin_y - 50, area.origin_y + area.height_m + 50)
    ax.set_aspect("equal")
    _draw_zones(ax, config)

    journeys = find_message_journeys(forwarding_log)

    # Use the last frame for node positions
    last_frame = frames[-1] if frames else None
    if not last_frame:
        return fig

    def _pos(node_id: str) -> tuple[float, float] | None:
        return last_frame.node_positions.get(node_id)

    delivered_segments = []
    undelivered_segments = []

    for mid, hops in journeys.items():
        was_delivered = any(h.reason == "delivered" for h in hops)
        for entry in hops:
            p1 = _pos(entry.from_node)
            p2 = _pos(entry.to_node)
            if p1 and p2:
                if was_delivered:
                    delivered_segments.append([p1, p2])
                else:
                    undelivered_segments.append([p1, p2])

    if delivered_segments:
        lc_del = LineCollection(delivered_segments, colors="#2CA02C", alpha=0.3,
                                linewidths=0.8, zorder=1)
        ax.add_collection(lc_del)

    if undelivered_segments:
        lc_undel = LineCollection(undelivered_segments, colors="#D62728", alpha=0.2,
                                  linewidths=0.5, zorder=1)
        ax.add_collection(lc_undel)

    # Draw nodes
    coord_ids = [nid for nid, nt in last_frame.node_types.items()
                 if nt == NodeType.COORDINATION.value]
    mobile_ids = [nid for nid, nt in last_frame.node_types.items()
                  if nt == NodeType.MOBILE_RESPONDER.value]

    mx = [last_frame.node_positions[n][0] for n in mobile_ids if n in last_frame.node_positions]
    my = [last_frame.node_positions[n][1] for n in mobile_ids if n in last_frame.node_positions]
    ax.scatter(mx, my, s=15, c="#555555", alpha=0.6, zorder=3)

    cx = [last_frame.node_positions[n][0] for n in coord_ids if n in last_frame.node_positions]
    cy = [last_frame.node_positions[n][1] for n in coord_ids if n in last_frame.node_positions]
    ax.scatter(cx, cy, s=100, c=COORDINATION_COLOR, marker="s", zorder=4, edgecolors="k")

    n_del = sum(1 for hops in journeys.values() if any(h.reason == "delivered" for h in hops))
    n_total = len(journeys)
    ax.text(
        0.02, 0.02,
        f"Messages: {n_del}/{n_total} delivered ({n_del/n_total:.0%})" if n_total else "No messages",
        transform=ax.transAxes, fontsize=10, fontfamily="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round,pad=0.3"},
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2CA02C", lw=2, alpha=0.6, label="Delivered path"),
        Line2D([0], [0], color="#D62728", lw=2, alpha=0.4, label="Undelivered path"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    title = "All Message Paths"
    if algorithm_label:
        title = f"{algorithm_label}: {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    fig.tight_layout()
    return fig
