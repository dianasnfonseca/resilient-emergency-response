"""
ERCS Configuration Parameters.

This module defines all simulation parameters organized by implementation phase.
Each parameter includes its value and description.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Enumerations
# =============================================================================


class UrgencyLevel(str, Enum):
    """
    Task urgency classification levels.

    Based on emergency medical triage conventions and DTN prioritisation
    research.
    """

    HIGH = "H"
    MEDIUM = "M"
    LOW = "L"


class AlgorithmType(str, Enum):
    """
    Coordination algorithm variants for experimental comparison.

    ADAPTIVE: Network-aware assignment with urgency-first ordering
    BASELINE: Proximity-only assignment with FCFS ordering
    """

    ADAPTIVE = "adaptive"
    BASELINE = "baseline"


class BufferDropPolicy(str, Enum):
    """
    Buffer management policy when capacity is exceeded.

    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


class MobilityModel(str, Enum):
    """
    Node mobility models for DTN simulation.

    """

    RANDOM_WAYPOINT = "random_waypoint"
    ROLE_BASED_WAYPOINT = "role_based_waypoint"


class ResponderRole(str, Enum):
    """
    Responder role determining mobility pattern.

    Different roles produce heterogeneous encounter patterns that allow
    PRoPHET to build differentiated delivery predictability values.

    """

    RESCUE = "rescue"  # ~60% — localised in incident zone
    TRANSPORT = "transport"  # ~25% — shuttle between incident and coordination zones
    LIAISON = "liaison"  # ~15% — free movement across entire area


# =============================================================================
# Phase 1: Network Topology Parameters
# =============================================================================


class ZoneConfig(BaseModel):
    """
    Geographic zone configuration.

    Defines the dimensions and position of a zone within the simulation area.
    Zone dimensions are defined in ``configs/default.yaml``.
    """

    width_m: float = Field(..., gt=0, description="Zone width in metres")
    height_m: float = Field(..., gt=0, description="Zone height in metres")
    origin_x: float = Field(0.0, description="Zone origin X coordinate in metres")
    origin_y: float = Field(0.0, description="Zone origin Y coordinate in metres")

    @property
    def area_m2(self) -> float:
        """Calculate zone area in square metres."""
        return self.width_m * self.height_m


class NetworkParameters(BaseModel):
    """
    Phase 1: Network Topology Generation Parameters.

    """

    # Network Scale
    primary_node_count: int = Field(
        default=50,
        ge=30,
        le=100,
        description="Primary node count.",
    )

    sensitivity_node_counts: list[int] = Field(
        default=[30, 50, 70],
        description="Node counts for sensitivity analysis.",
    )

    # Geographic Configuration
    simulation_area: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=3000.0, height_m=1500.0, origin_x=0.0, origin_y=0.0
        ),
        description="Total simulation area (3000x1500 m).",
    )

    incident_zone: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=700.0, height_m=600.0, origin_x=0.0, origin_y=450.0
        ),
        description="Incident zone.",
    )

    coordination_zone: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=50.0, height_m=50.0, origin_x=800.0, origin_y=300.0
        ),
        description="Coordination zone.",
    )

    # Node Types
    coordination_node_count: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Fixed coordination nodes.",
    )

    mobile_responder_count: int = Field(
        default=48,
        ge=1,
        description="Mobile responder nodes (50 - 2).",
    )

    # Communication Infrastructure
    radio_range_m: float = Field(
        default=100.0,
        gt=0,
        description="Radio range in metres.",
    )

    buffer_size_bytes: int = Field(
        default=26_214_400,  # 25 MB
        gt=0,
        description="Node buffer capacity (25 MB). Max tested value, avoids congestion-dominated regime.",
    )

    message_size_bytes: int = Field(
        default=512_000,  # 500 kB
        gt=0,
        description="Message size (500 kB).",
    )

    # Connectivity Scenarios
    connectivity_scenarios: list[float] = Field(
        default=[0.75, 0.40, 0.20],
        description=(
            "Link availability percentages: "
            "75% post-earthquake, 40% at 48h, 20% severe degradation."
        ),
    )

    @field_validator("connectivity_scenarios")
    @classmethod
    def validate_connectivity(cls, v: list[float]) -> list[float]:
        """Validate connectivity values are valid percentages."""
        for conn in v:
            if not 0.0 < conn <= 1.0:
                raise ValueError(f"Connectivity {conn} must be between 0 and 1")
        return v

    # Mobility Model
    mobility_model: MobilityModel = Field(
        default=MobilityModel.RANDOM_WAYPOINT,
        description="Mobility model.",
    )

    speed_min_mps: float = Field(
        default=0.0,
        ge=0,
        description="Min speed (0 m/s = stationary).",
    )

    speed_max_mps: float = Field(
        default=20.0,
        gt=0,
        description="Max speed (20 m/s = vehicle).",
    )

    # Mobility timing
    mobility_update_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        description="Interval between mobility position updates (seconds). "
        "1s granularity for smooth movement.",
    )

    encounter_check_interval_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Interval between encounter checks (seconds). "
        "Balances PRoPHET update frequency vs performance.",
    )

    # Mobility pause times
    pause_min_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Minimum pause duration at waypoints (seconds).",
    )

    pause_max_seconds: float = Field(
        default=30.0,
        ge=0,
        description="Maximum pause duration at waypoints (seconds). "
        "Brief pauses for realism.",
    )

    # Role-based mobility configuration
    role_rescue_fraction: float = Field(
        default=0.60,
        ge=0,
        le=1,
        description="Fraction of mobile nodes assigned RESCUE role (60%).",
    )
    role_transport_fraction: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Fraction of mobile nodes assigned TRANSPORT role (25%).",
    )
    # LIAISON fraction = 1 - rescue - transport (remainder)

    role_rescue_speed_min: float = Field(default=1.0, ge=0)
    role_rescue_speed_max: float = Field(default=5.0, gt=0)
    role_rescue_pause_min: float = Field(default=10.0, ge=0)
    role_rescue_pause_max: float = Field(default=60.0, ge=0)

    role_transport_speed_min: float = Field(default=5.0, ge=0)
    role_transport_speed_max: float = Field(default=20.0, gt=0)
    role_transport_pause_min: float = Field(default=30.0, ge=0)
    role_transport_pause_max: float = Field(default=120.0, ge=0)

    role_liaison_speed_min: float = Field(default=1.0, ge=0)
    role_liaison_speed_max: float = Field(default=10.0, gt=0)
    role_liaison_pause_min: float = Field(default=0.0, ge=0)
    role_liaison_pause_max: float = Field(default=30.0, ge=0)

    @field_validator("speed_max_mps")
    @classmethod
    def validate_speed_range(cls, v: float, info) -> float:
        """Validate max speed is greater than min speed."""
        min_speed = info.data.get("speed_min_mps", 0.0)
        if v <= min_speed:
            raise ValueError("speed_max_mps must be greater than speed_min_mps")
        return v


# =============================================================================
# Phase 2: Communication Layer Parameters
# =============================================================================


class PRoPHETParameters(BaseModel):
    """
    PRoPHETv2 protocol parameters.

    PRoPHETv2 replaces the original PRoPHET encounter equation with a
    time-based P_enc calculation and uses MAX-based transitivity to prevent
    delivery predictability saturation.

    """

    p_enc_max: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Maximum encounter probability.",
    )

    i_typ: float = Field(
        default=1800.0,
        gt=0,
        description="Typical inter-encounter interval in seconds.",
    )

    beta: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Transitivity constant (β).",
    )

    gamma: float = Field(
        default=0.999885791,
        gt=0,
        lt=1,
        description="Aging constant (γ).",
    )


class CommunicationParameters(BaseModel):
    """
    Phase 2: Communication Layer Parameters.

    """

    routing_protocol: Literal["prophet"] = Field(
        default="prophet",
        description="DTN routing protocol.",
    )

    prophet: PRoPHETParameters = Field(
        default_factory=PRoPHETParameters,
        description="PRoPHETv2 protocol parameters",
    )

    message_ttl_seconds: int = Field(
        default=18000,  # 300 minutes
        gt=0,
        description="Message TTL (300 min).",
    )

    transmit_speed_bps: int = Field(
        default=2_000_000,  # 2 Mbps
        gt=0,
        description="Transmit speed (2 Mbps).",
    )

    update_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Predictability aging time unit in seconds. "
        "γ^k applied where k = elapsed / update_interval. "
        "30s aligns aging rate with NODE_ENCOUNTER interval (10s) "
        "so ~3 encounters per aging unit.",
    )

    buffer_drop_policy: BufferDropPolicy = Field(
        default=BufferDropPolicy.DROP_OLDEST,
        description="Buffer policy.",
    )

    min_predictability_threshold: float = Field(
        default=0.001,
        ge=0,
        description="Predictabilities below this value are pruned to save memory.",
    )


# =============================================================================
# Phase 3: Scenario Generation Parameters
# =============================================================================


class UrgencyDistribution(BaseModel):
    """
    Task urgency distribution.

    """

    high: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="High-urgency proportion (20%).",
    )

    medium: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Medium-urgency proportion (50%).",
    )

    low: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Low-urgency proportion (30%).",
    )

    @field_validator("low")
    @classmethod
    def validate_distribution_sum(cls, v: float, info) -> float:
        """Validate that urgency proportions sum to 1.0."""
        high = info.data.get("high", 0.0)
        medium = info.data.get("medium", 0.0)
        total = high + medium + v
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Urgency distribution must sum to 1.0, got {total:.3f}")
        return v


class ScenarioParameters(BaseModel):
    """
    Phase 3: Scenario Generation Parameters.

    """

    scenario_type: Literal["generic_emergency"] = Field(
        default="generic_emergency",
        description="Scenario type.",
    )

    message_generation_model: Literal["poisson"] = Field(
        default="poisson",
        description="Arrival model.",
    )

    message_rate_per_minute: float = Field(
        default=2.0,
        gt=0,
        description="Message rate (2/min).",
    )

    urgency_distribution: UrgencyDistribution = Field(
        default_factory=UrgencyDistribution,
        description="Urgency distribution",
    )

    simulation_duration_seconds: int = Field(
        default=6000,  # ~100 minutes
        gt=0,
        description="Duration (6000s).",
    )

    warmup_period_seconds: int = Field(
        default=0,
        ge=0,
        description="Warm-up period (seconds). Set to 0 for cold-start evaluation: "
        "emergency coordination systems activate from zero encounter history, "
        "and initialisation variability is addressed through multiple "
        "replications.",
    )

    runs_per_configuration: int = Field(
        default=30,
        ge=1,
        description="Runs per config (30).",
    )


# =============================================================================
# Phase 4: Coordination Layer Parameters
# =============================================================================


class CoordinationParameters(BaseModel):
    """
    Phase 4: Coordination Layer Parameters.

    """

    update_interval_seconds: int = Field(
        default=1800,  # 30 minutes
        gt=0,
        description="Update interval (30 min).",
    )

    priority_levels: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Priority levels (3).",
    )

    available_path_threshold: float = Field(
        default=0.3,
        ge=0,
        lt=1,
        description="Path threshold (P > 0.3). Nodes with genuine coord-zone "
        "encounter history converge to P ≈ 0.45–0.50 while marginal transitivity "
        "nodes stabilise at P ≈ 0.05–0.20.",
    )

    # Static scoring weights
    predictability_weight: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="α — predictability weight in scoring formula. ",
    )

    recency_weight: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="γ_r — encounter recency weight in scoring formula. "
        "R_norm = 1 − min(Δt / 1800, 1.0).",
    )

    proximity_weight: float = Field(
        default=0.6, ge=0, le=1, description="β — proximity weight in scoring formula. "
    )

    workload_penalty_weight: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="λ — workload penalty weight. Discourages re-assigning the "
        "same responder across consecutive coordination cycles. ",
    )

    recency_reference_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="T_REF for encounter recency normalisation: "
        "R_norm = 1 − min(Δt / T_REF, 1.0). Matches PRoPHET I_typ.",
    )

    proximity_method: Literal["euclidean"] = Field(
        default="euclidean",
        description="Distance method.",
    )

    adaptive_task_order: Literal["urgency_first"] = Field(
        default="urgency_first",
        description="Adaptive ordering.",
    )

    baseline_task_order: Literal["fcfs"] = Field(
        default="fcfs",
        description="Baseline ordering (FCFS).",
    )


# =============================================================================
# Complete Simulation Configuration
# =============================================================================


class SimulationConfig(BaseModel):
    """
    Complete simulation configuration combining all phase parameters.

    This is the single source of truth for all experimental parameters.
    """

    network: NetworkParameters = Field(
        default_factory=NetworkParameters,
        description="Phase 1: Network topology parameters",
    )

    communication: CommunicationParameters = Field(
        default_factory=CommunicationParameters,
        description="Phase 2: Communication layer parameters",
    )

    scenario: ScenarioParameters = Field(
        default_factory=ScenarioParameters,
        description="Phase 3: Scenario generation parameters",
    )

    coordination: CoordinationParameters = Field(
        default_factory=CoordinationParameters,
        description="Phase 4: Coordination layer parameters",
    )

    experiment_name: str = Field(
        default="ercs_experiment",
        description="Experiment name for outputs and logs",
    )

    random_seed: int | None = Field(
        default=None,
        description="Base random seed for reproducibility",
    )

    output_directory: str = Field(
        default="./outputs",
        description="Directory for simulation outputs",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    @property
    def total_simulation_duration(self) -> float:
        """Total duration including warm-up period."""
        return (
            self.scenario.warmup_period_seconds
            + self.scenario.simulation_duration_seconds
        )

    @property
    def total_experimental_runs(self) -> int:
        """Total runs: 2 algorithms × 3 connectivity × 30 runs = 180."""
        num_algorithms = 2
        num_scenarios = len(self.network.connectivity_scenarios)
        runs_per_config = self.scenario.runs_per_configuration
        return num_algorithms * num_scenarios * runs_per_config

    @property
    def total_nodes(self) -> int:
        """Total node count."""
        return (
            self.network.coordination_node_count + self.network.mobile_responder_count
        )

    @property
    def simulation_area_diagonal_m(self) -> float:
        """Diagonal of the simulation area (metres), for distance normalisation."""
        import math

        w = self.network.simulation_area.width_m
        h = self.network.simulation_area.height_m
        return math.sqrt(w * w + h * h)

    def get_message_transmission_time_seconds(self) -> float:
        """Time to transmit one message in seconds."""
        message_bits = self.network.message_size_bytes * 8
        return message_bits / self.communication.transmit_speed_bps

    model_config = {"validate_assignment": True, "extra": "forbid"}
