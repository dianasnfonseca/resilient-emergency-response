"""
Config consistency tests.

Ensures that all parameter values used across the codebase come from
``SimulationConfig`` defaults, preventing hardcoded overrides from
drifting out of sync with ``configs/default.yaml``.
"""

import math
import yaml
from pathlib import Path

import pytest

from ercs.config.parameters import (
    CoordinationParameters,
    NetworkParameters,
    SimulationConfig,
)
from ercs.network.mobility import (
    _build_role_configs,
    _build_role_distribution,
    _assign_roles,
    RoleConfig,
)
from ercs.config.parameters import ResponderRole


# Load the canonical default config
CONFIG = SimulationConfig()

# Path to default.yaml
YAML_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class TestYamlMatchesPydanticDefaults:
    """Verify that default.yaml values match SimulationConfig() defaults."""

    @pytest.fixture(scope="class")
    def yaml_config(self):
        with open(YAML_PATH) as f:
            return yaml.safe_load(f)

    def test_network_params(self, yaml_config):
        net = yaml_config["network"]
        assert net["primary_node_count"] == CONFIG.total_nodes
        assert net["radio_range_m"] == CONFIG.network.radio_range_m
        assert net["buffer_size_bytes"] == CONFIG.network.buffer_size_bytes
        assert net["message_size_bytes"] == CONFIG.network.message_size_bytes
        assert net["mobile_responder_count"] == CONFIG.network.mobile_responder_count
        assert net["coordination_node_count"] == CONFIG.network.coordination_node_count

    def test_simulation_area(self, yaml_config):
        area = yaml_config["network"]["simulation_area"]
        assert area["width_m"] == CONFIG.network.simulation_area.width_m
        assert area["height_m"] == CONFIG.network.simulation_area.height_m

    def test_incident_zone(self, yaml_config):
        iz = yaml_config["network"]["incident_zone"]
        assert iz["width_m"] == CONFIG.network.incident_zone.width_m
        assert iz["height_m"] == CONFIG.network.incident_zone.height_m
        assert iz["origin_x"] == CONFIG.network.incident_zone.origin_x
        assert iz["origin_y"] == CONFIG.network.incident_zone.origin_y

    def test_coordination_zone(self, yaml_config):
        cz = yaml_config["network"]["coordination_zone"]
        assert cz["width_m"] == CONFIG.network.coordination_zone.width_m
        assert cz["height_m"] == CONFIG.network.coordination_zone.height_m
        assert cz["origin_x"] == CONFIG.network.coordination_zone.origin_x
        assert cz["origin_y"] == CONFIG.network.coordination_zone.origin_y

    def test_prophet_params(self, yaml_config):
        p = yaml_config["communication"]["prophet"]
        assert p["p_enc_max"] == CONFIG.communication.prophet.p_enc_max
        assert p["i_typ"] == CONFIG.communication.prophet.i_typ
        assert p["beta"] == CONFIG.communication.prophet.beta
        assert p["gamma"] == pytest.approx(CONFIG.communication.prophet.gamma)

    def test_communication_params(self, yaml_config):
        c = yaml_config["communication"]
        assert c["message_ttl_seconds"] == CONFIG.communication.message_ttl_seconds
        assert c["transmit_speed_bps"] == CONFIG.communication.transmit_speed_bps
        assert c["update_interval_seconds"] == CONFIG.communication.update_interval_seconds
        assert c["min_predictability_threshold"] == CONFIG.communication.min_predictability_threshold

    def test_scenario_params(self, yaml_config):
        s = yaml_config["scenario"]
        assert s["message_rate_per_minute"] == CONFIG.scenario.message_rate_per_minute
        assert s["simulation_duration_seconds"] == CONFIG.scenario.simulation_duration_seconds
        assert s["warmup_period_seconds"] == CONFIG.scenario.warmup_period_seconds
        assert s["runs_per_configuration"] == CONFIG.scenario.runs_per_configuration

    def test_urgency_distribution(self, yaml_config):
        u = yaml_config["scenario"]["urgency_distribution"]
        assert u["high"] == CONFIG.scenario.urgency_distribution.high
        assert u["medium"] == CONFIG.scenario.urgency_distribution.medium
        assert u["low"] == CONFIG.scenario.urgency_distribution.low

    def test_coordination_params(self, yaml_config):
        c = yaml_config["coordination"]
        assert c["update_interval_seconds"] == CONFIG.coordination.update_interval_seconds
        assert c["available_path_threshold"] == CONFIG.coordination.available_path_threshold
        assert c["workload_penalty_weight"] == CONFIG.coordination.workload_penalty_weight
        assert c["recency_reference_seconds"] == CONFIG.coordination.recency_reference_seconds

    def test_mobility_intervals(self, yaml_config):
        net = yaml_config["network"]
        assert net["mobility_update_interval_seconds"] == CONFIG.network.mobility_update_interval_seconds
        assert net["encounter_check_interval_seconds"] == CONFIG.network.encounter_check_interval_seconds
        assert net["pause_min_seconds"] == CONFIG.network.pause_min_seconds
        assert net["pause_max_seconds"] == CONFIG.network.pause_max_seconds

    def test_role_configs_in_yaml(self, yaml_config):
        net = yaml_config["network"]
        assert net["role_rescue_fraction"] == CONFIG.network.role_rescue_fraction
        assert net["role_transport_fraction"] == CONFIG.network.role_transport_fraction
        assert net["role_rescue_speed_min"] == CONFIG.network.role_rescue_speed_min
        assert net["role_rescue_speed_max"] == CONFIG.network.role_rescue_speed_max
        assert net["role_transport_speed_min"] == CONFIG.network.role_transport_speed_min
        assert net["role_transport_speed_max"] == CONFIG.network.role_transport_speed_max
        assert net["role_liaison_speed_min"] == CONFIG.network.role_liaison_speed_min
        assert net["role_liaison_speed_max"] == CONFIG.network.role_liaison_speed_max


class TestComputedProperties:
    """Verify computed values are consistent."""

    def test_area_diagonal(self):
        expected = math.sqrt(3000.0**2 + 1500.0**2)
        assert CONFIG.simulation_area_diagonal_m == pytest.approx(expected, rel=1e-6)

    def test_total_nodes(self):
        assert CONFIG.total_nodes == 50

    def test_total_runs(self):
        assert CONFIG.total_experimental_runs == 180

    def test_role_fractions_sum_to_one(self):
        total = (
            CONFIG.network.role_rescue_fraction
            + CONFIG.network.role_transport_fraction
            + (1.0 - CONFIG.network.role_rescue_fraction - CONFIG.network.role_transport_fraction)
        )
        assert total == pytest.approx(1.0)


class TestMobilityConfigConsistency:
    """Verify mobility module reads from config, not hardcoded values."""

    def test_role_configs_match_parameters(self):
        params = NetworkParameters()
        configs = _build_role_configs(params)

        assert configs[ResponderRole.RESCUE].speed_min == params.role_rescue_speed_min
        assert configs[ResponderRole.RESCUE].speed_max == params.role_rescue_speed_max
        assert configs[ResponderRole.TRANSPORT].speed_min == params.role_transport_speed_min
        assert configs[ResponderRole.TRANSPORT].speed_max == params.role_transport_speed_max
        assert configs[ResponderRole.LIAISON].speed_min == params.role_liaison_speed_min
        assert configs[ResponderRole.LIAISON].speed_max == params.role_liaison_speed_max

    def test_role_distribution_matches_parameters(self):
        params = NetworkParameters()
        dist = _build_role_distribution(params)

        assert dist[ResponderRole.RESCUE] == params.role_rescue_fraction
        assert dist[ResponderRole.TRANSPORT] == params.role_transport_fraction
        liaison_expected = 1.0 - params.role_rescue_fraction - params.role_transport_fraction
        assert dist[ResponderRole.LIAISON] == pytest.approx(liaison_expected)

    def test_assign_roles_count(self):
        roles = _assign_roles(48)
        assert len(roles) == 48
        rescue_count = roles.count(ResponderRole.RESCUE)
        transport_count = roles.count(ResponderRole.TRANSPORT)
        liaison_count = roles.count(ResponderRole.LIAISON)
        assert rescue_count == 29  # round(48 * 0.60)
        assert transport_count == 12  # round(48 * 0.25)
        assert liaison_count == 7   # remainder


class TestCoordinationConfigConsistency:
    """Verify coordination algorithms use config values."""

    def test_adaptive_uses_config_weights(self):
        from ercs.coordination.algorithms import AdaptiveCoordinator
        params = CoordinationParameters()
        coord = AdaptiveCoordinator(params=params)
        assert coord.params.predictability_weight == 0.2
        assert coord.params.recency_weight == 0.2
        assert coord.params.proximity_weight == 0.6
        assert coord.params.workload_penalty_weight == 0.2
        assert coord.params.recency_reference_seconds == 1800.0

    def test_area_diagonal_passed_through(self):
        from ercs.coordination.algorithms import create_coordinator
        coord = create_coordinator("adaptive", area_diagonal_m=1234.5)
        assert coord.area_diagonal_m == 1234.5

    def test_default_diagonal_computed(self):
        from ercs.coordination.algorithms import create_coordinator
        coord = create_coordinator("adaptive")
        expected = math.sqrt(3000.0**2 + 1500.0**2)
        assert coord.area_diagonal_m == pytest.approx(expected, rel=1e-6)
