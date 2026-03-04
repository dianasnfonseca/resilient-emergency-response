"""
Centralised test constants derived from the ERCS specification.

These mirror the default values in ``src/ercs/config/parameters.py``.
If the spec changes, update HERE — every test file imports from conftest.

Usage in tests::

    from conftest import SIMULATION_DURATION_S, P_INIT, RADIO_RANGE_M
"""

# =============================================================================
# Phase 1: Network Topology (Ullah & Qayyum, 2022)
# =============================================================================

NODE_COUNT = 50
COORDINATION_NODE_COUNT = 2
MOBILE_RESPONDER_COUNT = 48

# Simulation area (metres)
SIMULATION_AREA_WIDTH_M = 3000.0
SIMULATION_AREA_HEIGHT_M = 1500.0

# Incident zone (metres)
INCIDENT_ZONE_WIDTH_M = 700.0
INCIDENT_ZONE_HEIGHT_M = 600.0

# Coordination zone (metres)
COORDINATION_ZONE_WIDTH_M = 50.0
COORDINATION_ZONE_HEIGHT_M = 50.0
COORDINATION_ZONE_ORIGIN_X = 800.0

# Communication infrastructure
RADIO_RANGE_M = 100.0
BUFFER_SIZE_BYTES = 26_214_400  # 25 MB
MESSAGE_SIZE_BYTES = 512_000  # 500 kB

# Connectivity scenarios (Karaman et al., 2026)
CONNECTIVITY_SCENARIOS = [0.75, 0.40, 0.20]
CONNECTIVITY_MILD = 0.75
CONNECTIVITY_MODERATE = 0.40
CONNECTIVITY_SEVERE = 0.20

# Mobility (Ullah & Qayyum, 2022)
SPEED_MAX_MPS = 20.0

# =============================================================================
# Phase 2: Communication / PRoPHETv2 (Grasic et al., 2011)
# =============================================================================

P_ENC_MAX = 0.5
I_TYP = 1800.0
BETA = 0.9
GAMMA = 0.999885791
MESSAGE_TTL_S = 18_000  # 300 minutes
TRANSMIT_SPEED_BPS = 2_000_000
AGING_INTERVAL_S = 30.0

# Legacy alias kept for tests that still reference P_INIT
P_INIT = P_ENC_MAX

# =============================================================================
# Phase 3: Scenario Generation
# =============================================================================

MESSAGE_RATE_PER_MIN = 2.0
SIMULATION_DURATION_S = 6_000  # ~100 minutes (Ullah & Qayyum, 2022)
WARMUP_PERIOD_S = 0  # Cold-start (Grassmann, 2008)
TOTAL_DURATION_S = 6_000  # No warm-up: equals simulation_duration_seconds
RUNS_PER_CONFIG = 30  # (Law, 2015)

# Urgency distribution (Li et al., 2025)
URGENCY_HIGH_PROP = 0.20
URGENCY_MEDIUM_PROP = 0.50
URGENCY_LOW_PROP = 0.30

# =============================================================================
# Phase 4: Coordination
# =============================================================================

COORDINATION_INTERVAL_S = 1_800  # 30 minutes (Kaji et al., 2025)
PRIORITY_LEVELS = 3
PATH_THRESHOLD = 0.3

# Static scoring weights (Boondirek et al., 2014; Nelson et al., 2009)
PREDICTABILITY_WEIGHT = 0.2
RECENCY_WEIGHT = 0.2
PROXIMITY_WEIGHT = 0.6

# =============================================================================
# Experiment 
# =============================================================================

TOTAL_CONFIGURATIONS = 6  # 2 algorithms × 3 connectivity levels
TOTAL_EXPERIMENTAL_RUNS = 180  # 6 × 30 runs
ALGORITHMS = ["adaptive", "baseline"]
