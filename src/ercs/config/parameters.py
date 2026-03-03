"""
ERCS Configuration Parameters.

This module defines all simulation parameters organized by implementation phase.
Each parameter includes its value, literature source, and description.

Sources:
    - Ullah & Qayyum (2022): Post-disaster DTN routing simulation
    - Kumar et al. (2023): PRoPHET protocol optimization
    - Karaman et al. (2026): Turkey earthquake infrastructure analysis
    - Rosas et al. (2023): DTN message prioritisation
    - Kaji et al. (2025): Distributed emergency logistics
    - Cabral et al. (2018): EMS response time standards
    - Li et al. (2025): Emergency facility location
    - Oksuz & Satoglu (2024): Casualty allocation planning
    - Keykhaei et al. (2024): Multi-agent evacuation
    - Law (2015): Simulation methodology
    - Pu et al. (2025): Disaster response modelling
    - Castillo et al. (2024): DTN systematic review
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
    research (Rosas et al., 2023).
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

    Source: Ullah & Qayyum (2022) - drop oldest messages standard practice
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


class MobilityModel(str, Enum):
    """
    Node mobility models for DTN simulation.

    Sources:
        - Ullah & Qayyum (2022): Random Waypoint commonly used in DTN research
        - Aschenbruck et al. (2009): Disaster Area mobility model
    """

    RANDOM_WAYPOINT = "random_waypoint"
    ROLE_BASED_WAYPOINT = "role_based_waypoint"


class ResponderRole(str, Enum):
    """
    Responder role determining mobility pattern.

    Different roles produce heterogeneous encounter patterns that allow
    PRoPHET to build differentiated delivery predictability values.

    Sources:
        - Aschenbruck et al. (2009): Disaster Area mobility model
        - Uddin et al. (2011): Post-Disaster Mobility Model
        - Stute et al. (2017): Natural Disaster Mobility Model
    """

    RESCUE = "rescue"        # ~60% — localised in incident zone
    TRANSPORT = "transport"  # ~25% — shuttle between incident and coordination zones
    LIAISON = "liaison"      # ~15% — free movement across entire area


# =============================================================================
# Phase 1: Network Topology Parameters
# =============================================================================


class ZoneConfig(BaseModel):
    """
    Geographic zone configuration.

    Defines the dimensions and position of a zone within the simulation area.
    Zone dimensions are adapted from Ullah & Qayyum (2022).
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

    Sources:
        - Ullah & Qayyum (2022): Node count, area, radio range, buffer, mobility
        - Kumar et al. (2023): Message size
        - Karaman et al. (2026): Connectivity scenarios from Turkey earthquake data
    """

    # Network Scale
    primary_node_count: int = Field(
        default=50,
        ge=30,
        le=100,
        description="Primary node count. Source: Ullah & Qayyum (2022)",
    )

    sensitivity_node_counts: list[int] = Field(
        default=[30, 50, 70],
        description="Node counts for sensitivity analysis. Source: Design decision",
    )

    # Geographic Configuration
    simulation_area: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=3000.0, height_m=1500.0, origin_x=0.0, origin_y=0.0
        ),
        description="Total simulation area. Source: Ullah & Qayyum (2022) - 3000×1500 m²",
    )

    incident_zone: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=700.0, height_m=600.0, origin_x=0.0, origin_y=450.0
        ),
        description="Incident zone. Source: Adapted from Ullah & Qayyum (2022)",
    )

    coordination_zone: ZoneConfig = Field(
        default_factory=lambda: ZoneConfig(
            width_m=50.0, height_m=50.0, origin_x=800.0, origin_y=300.0
        ),
        description="Coordination zone. Source: Adapted from Ullah & Qayyum (2022)",
    )

    # Node Types
    coordination_node_count: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Fixed coordination nodes. Source: Design decision",
    )

    mobile_responder_count: int = Field(
        default=48,
        ge=1,
        description="Mobile responder nodes. Source: Design decision (50 - 2)",
    )

    # Communication Infrastructure
    radio_range_m: float = Field(
        default=100.0,
        gt=0,
        description="Radio range in metres. Source: Ullah & Qayyum (2022)",
    )

    buffer_size_bytes: int = Field(
        default=26_214_400,  # 25 MB
        gt=0,
        description="Node buffer capacity (25 MB). Source: Ullah & Qayyum (2022) — max tested value, avoids congestion-dominated regime",
    )

    message_size_bytes: int = Field(
        default=512_000,  # 500 kB
        gt=0,
        description="Message size (500 kB). Source: Kumar et al. (2023)",
    )

    # Connectivity Scenarios
    connectivity_scenarios: list[float] = Field(
        default=[0.75, 0.40, 0.20],
        description=(
            "Link availability percentages. Source: Karaman et al. (2026) - "
            "75% post-earthquake, 40% at 48h, 20% severe degradation"
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
        description="Mobility model. Source: Ullah & Qayyum (2022)",
    )

    speed_min_mps: float = Field(
        default=0.0,
        ge=0,
        description="Min speed (0 m/s = stationary). Source: Ullah & Qayyum (2022)",
    )

    speed_max_mps: float = Field(
        default=20.0,
        gt=0,
        description="Max speed (20 m/s = vehicle). Source: Ullah & Qayyum (2022)",
    )

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

    Sources:
        - Grasic et al. (2011): PRoPHETv2 specification (CHANTS '11)
        - Lindgren et al. (2012): RFC 6693
    """

    p_enc_max: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Maximum encounter probability. Source: Grasic et al. (2011)",
    )

    i_typ: float = Field(
        default=1800.0,
        gt=0,
        description="Typical inter-encounter interval in seconds. "
        "Source: Grasic et al. (2011)",
    )

    beta: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Transitivity constant (β). Source: Grasic et al. (2011)",
    )

    gamma: float = Field(
        default=0.999885791,
        gt=0,
        lt=1,
        description="Aging constant (γ). Source: Grasic et al. (2011)",
    )


class CommunicationParameters(BaseModel):
    """
    Phase 2: Communication Layer Parameters.

    Sources:
        - Grasic et al. (2011): PRoPHETv2 protocol parameters
        - Ullah & Qayyum (2022): Message TTL, transmit speed, buffer policy
        - Castillo et al. (2024): Protocol selection rationale
    """

    routing_protocol: Literal["prophet"] = Field(
        default="prophet",
        description="DTN routing protocol. Source: Castillo et al. (2024)",
    )

    prophet: PRoPHETParameters = Field(
        default_factory=PRoPHETParameters,
        description="PRoPHETv2 protocol parameters",
    )

    message_ttl_seconds: int = Field(
        default=18000,  # 300 minutes
        gt=0,
        description="Message TTL (300 min). Source: Ullah & Qayyum (2022)",
    )

    transmit_speed_bps: int = Field(
        default=2_000_000,  # 2 Mbps
        gt=0,
        description="Transmit speed (2 Mbps). Source: Ullah & Qayyum (2022)",
    )

    update_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Predictability aging time unit in seconds. "
        "γ^k applied where k = elapsed / update_interval. "
        "30s aligns aging rate with NODE_ENCOUNTER interval (10s) "
        "so ~3 encounters per aging unit. Source: Kumar et al. (2023)",
    )

    buffer_drop_policy: BufferDropPolicy = Field(
        default=BufferDropPolicy.DROP_OLDEST,
        description="Buffer policy. Source: Ullah & Qayyum (2022)",
    )


# =============================================================================
# Phase 3: Scenario Generation Parameters
# =============================================================================


class UrgencyDistribution(BaseModel):
    """
    Task urgency distribution.

    Sources:
        - Li et al. (2025): Emergency facility demand patterns
        - Oksuz & Satoglu (2024): Casualty allocation prioritisation
    """

    high: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="High-urgency proportion (20%). Source: Li et al. (2025)",
    )

    medium: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Medium-urgency proportion (50%). Source: Li et al. (2025)",
    )

    low: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Low-urgency proportion (30%). Source: Li et al. (2025)",
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

    Sources:
        - Kumar et al. (2023): Message generation rate
        - Pu et al. (2025): Poisson process for arrivals
        - Ullah & Qayyum (2022): Simulation duration
        - Law (2015): Statistical design (30 runs)
    """

    scenario_type: Literal["generic_emergency"] = Field(
        default="generic_emergency",
        description="Scenario type. Source: FEMA (2024)",
    )

    message_generation_model: Literal["poisson"] = Field(
        default="poisson",
        description="Arrival model. Source: Pu et al. (2025)",
    )

    message_rate_per_minute: float = Field(
        default=2.0,
        gt=0,
        description="Message rate (2/min). Source: Kumar et al. (2023)",
    )

    urgency_distribution: UrgencyDistribution = Field(
        default_factory=UrgencyDistribution,
        description="Urgency distribution",
    )

    simulation_duration_seconds: int = Field(
        default=6000,  # ~100 minutes
        gt=0,
        description="Duration (6000s). Source: Ullah & Qayyum (2022)",
    )

    warmup_period_seconds: int = Field(
        default=1800,
        ge=0,
        description="Warm-up period for PRoPHET encounter history (seconds). "
        "Allows delivery predictability to build through actual node "
        "encounters before task generation begins. "
        "Source: Law (2015) — warm-up period methodology",
    )

    runs_per_configuration: int = Field(
        default=30,
        ge=1,
        description="Runs per config (30). Source: Law (2015)",
    )


# =============================================================================
# Phase 4: Coordination Layer Parameters
# =============================================================================


class CoordinationParameters(BaseModel):
    """
    Phase 4: Coordination Layer Parameters.

    Sources:
        - Kaji et al. (2025): Update interval, urgency-first ordering
        - Cabral et al. (2018): 30-minute response time standard
        - Rosas et al. (2023): Priority levels
        - Ullah & Qayyum (2022): Path availability threshold
    """

    update_interval_seconds: int = Field(
        default=1800,  # 30 minutes
        gt=0,
        description="Update interval (30 min). Source: Kaji et al. (2025)",
    )

    priority_levels: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Priority levels (3). Source: Rosas et al. (2023)",
    )

    available_path_threshold: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="Path threshold (P > 0). Source: Ullah & Qayyum (2022)",
    )

    proximity_method: Literal["euclidean"] = Field(
        default="euclidean",
        description="Distance method. Source: Keykhaei et al. (2024)",
    )

    adaptive_task_order: Literal["urgency_first"] = Field(
        default="urgency_first",
        description="Adaptive ordering. Source: Kaji et al. (2025)",
    )

    baseline_task_order: Literal["fcfs"] = Field(
        default="fcfs",
        description="Baseline ordering (FCFS). Source: Design decision",
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
        return self.scenario.warmup_period_seconds + self.scenario.simulation_duration_seconds

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
        return self.network.coordination_node_count + self.network.mobile_responder_count

    def get_message_transmission_time_seconds(self) -> float:
        """Time to transmit one message in seconds."""
        message_bits = self.network.message_size_bytes * 8
        return message_bits / self.communication.transmit_speed_bps

    model_config = {"validate_assignment": True, "extra": "forbid"}