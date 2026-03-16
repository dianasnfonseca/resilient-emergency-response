"""
Centralised test constants derived from the ERCS configuration.

All values are derived from ``SimulationConfig`` defaults — the single source
of truth.  If the spec changes, update ``parameters.py`` and ``default.yaml``;
these constants will follow automatically.

Usage in tests::

    from conftest import SIMULATION_DURATION_S, P_ENC_MAX, RADIO_RANGE_M
"""

from ercs.config.parameters import SimulationConfig

# Build the canonical default config once
_CONFIG = SimulationConfig()

# =============================================================================
# Phase 1: Network Topology
# =============================================================================

NODE_COUNT = _CONFIG.total_nodes
COORDINATION_NODE_COUNT = _CONFIG.network.coordination_node_count
MOBILE_RESPONDER_COUNT = _CONFIG.network.mobile_responder_count

# Simulation area (metres)
SIMULATION_AREA_WIDTH_M = _CONFIG.network.simulation_area.width_m
SIMULATION_AREA_HEIGHT_M = _CONFIG.network.simulation_area.height_m

# Incident zone (metres)
INCIDENT_ZONE_WIDTH_M = _CONFIG.network.incident_zone.width_m
INCIDENT_ZONE_HEIGHT_M = _CONFIG.network.incident_zone.height_m

# Coordination zone (metres)
COORDINATION_ZONE_WIDTH_M = _CONFIG.network.coordination_zone.width_m
COORDINATION_ZONE_HEIGHT_M = _CONFIG.network.coordination_zone.height_m
COORDINATION_ZONE_ORIGIN_X = _CONFIG.network.coordination_zone.origin_x

# Communication infrastructure
RADIO_RANGE_M = _CONFIG.network.radio_range_m
BUFFER_SIZE_BYTES = _CONFIG.network.buffer_size_bytes
MESSAGE_SIZE_BYTES = _CONFIG.network.message_size_bytes

# Connectivity scenarios
CONNECTIVITY_SCENARIOS = _CONFIG.network.connectivity_scenarios
CONNECTIVITY_MILD = CONNECTIVITY_SCENARIOS[0]  # 0.75
CONNECTIVITY_MODERATE = CONNECTIVITY_SCENARIOS[1]  # 0.40
CONNECTIVITY_SEVERE = CONNECTIVITY_SCENARIOS[2]  # 0.20

# Mobility
SPEED_MAX_MPS = _CONFIG.network.speed_max_mps

# =============================================================================
# Phase 2: Communication / PRoPHETv2
# =============================================================================

P_ENC_MAX = _CONFIG.communication.prophet.p_enc_max
I_TYP = _CONFIG.communication.prophet.i_typ
BETA = _CONFIG.communication.prophet.beta
GAMMA = _CONFIG.communication.prophet.gamma
MESSAGE_TTL_S = _CONFIG.communication.message_ttl_seconds
TRANSMIT_SPEED_BPS = _CONFIG.communication.transmit_speed_bps
AGING_INTERVAL_S = _CONFIG.communication.update_interval_seconds

# Legacy alias kept for tests that still reference P_INIT
P_INIT = P_ENC_MAX

# =============================================================================
# Phase 3: Scenario Generation
# =============================================================================

MESSAGE_RATE_PER_MIN = _CONFIG.scenario.message_rate_per_minute
SIMULATION_DURATION_S = _CONFIG.scenario.simulation_duration_seconds
WARMUP_PERIOD_S = _CONFIG.scenario.warmup_period_seconds
TOTAL_DURATION_S = SIMULATION_DURATION_S + WARMUP_PERIOD_S
RUNS_PER_CONFIG = _CONFIG.scenario.runs_per_configuration

# Urgency distribution
URGENCY_HIGH_PROP = _CONFIG.scenario.urgency_distribution.high
URGENCY_MEDIUM_PROP = _CONFIG.scenario.urgency_distribution.medium
URGENCY_LOW_PROP = _CONFIG.scenario.urgency_distribution.low

# =============================================================================
# Phase 4: Coordination
# =============================================================================

COORDINATION_INTERVAL_S = _CONFIG.coordination.update_interval_seconds
PRIORITY_LEVELS = _CONFIG.coordination.priority_levels
PATH_THRESHOLD = _CONFIG.coordination.available_path_threshold

# Static scoring weights
PREDICTABILITY_WEIGHT = _CONFIG.coordination.predictability_weight
RECENCY_WEIGHT = _CONFIG.coordination.recency_weight
PROXIMITY_WEIGHT = _CONFIG.coordination.proximity_weight

# =============================================================================
# Experiment
# =============================================================================

TOTAL_CONFIGURATIONS = 6  # 2 algorithms × 3 connectivity levels
TOTAL_EXPERIMENTAL_RUNS = _CONFIG.total_experimental_runs
ALGORITHMS = ["adaptive", "baseline"]
