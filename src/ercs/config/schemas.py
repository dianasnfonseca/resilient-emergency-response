"""
ERCS Validation Schemas.

Provides validation for experiment configuration files (YAML).
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ercs.config.parameters import (
    AlgorithmType,
    CommunicationParameters,
    CoordinationParameters,
    NetworkParameters,
    ScenarioParameters,
    SimulationConfig,
)


class ConnectivityScenarioConfig(BaseModel):
    """Configuration for a single connectivity scenario."""

    connectivity_level: float = Field(..., gt=0, le=1)
    num_runs: int = Field(default=30, ge=1)
    random_seed_base: int | None = Field(default=None)


class AlgorithmConfig(BaseModel):
    """Configuration for an algorithm variant."""

    algorithm_type: AlgorithmType
    enabled: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration for ERCS."""

    experiment_name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)

    algorithms: list[AlgorithmConfig] = Field(
        default_factory=lambda: [
            AlgorithmConfig(algorithm_type=AlgorithmType.ADAPTIVE),
            AlgorithmConfig(algorithm_type=AlgorithmType.BASELINE),
        ],
        min_length=1,
    )

    connectivity_scenarios: list[ConnectivityScenarioConfig] = Field(
        default_factory=lambda: [
            ConnectivityScenarioConfig(connectivity_level=0.75),
            ConnectivityScenarioConfig(connectivity_level=0.40),
            ConnectivityScenarioConfig(connectivity_level=0.20),
        ],
        min_length=1,
    )

    network: NetworkParameters = Field(default_factory=NetworkParameters)
    communication: CommunicationParameters = Field(default_factory=CommunicationParameters)
    scenario: ScenarioParameters = Field(default_factory=ScenarioParameters)
    coordination: CoordinationParameters = Field(default_factory=CoordinationParameters)

    output_directory: str = Field(default="./outputs")
    save_intermediate_results: bool = Field(default=True)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    master_random_seed: int | None = Field(default=42)

    @field_validator("algorithms")
    @classmethod
    def validate_at_least_one_enabled(cls, v: list[AlgorithmConfig]) -> list[AlgorithmConfig]:
        """Ensure at least one algorithm is enabled."""
        if not any(alg.enabled for alg in v):
            raise ValueError("At least one algorithm must be enabled")
        return v

    @model_validator(mode="after")
    def validate_cross_parameters(self) -> "ExperimentConfig":
        """Validate cross-parameter constraints."""
        total_nodes = self.network.coordination_node_count + self.network.mobile_responder_count
        if total_nodes != self.network.primary_node_count:
            raise ValueError(
                f"Node counts inconsistent: {self.network.coordination_node_count} + "
                f"{self.network.mobile_responder_count} != {self.network.primary_node_count}"
            )
        return self

    @property
    def enabled_algorithms(self) -> list[AlgorithmType]:
        """Get list of enabled algorithm types."""
        return [alg.algorithm_type for alg in self.algorithms if alg.enabled]

    @property
    def total_runs(self) -> int:
        """Calculate total number of simulation runs."""
        num_algorithms = len(self.enabled_algorithms)
        total_scenario_runs = sum(s.num_runs for s in self.connectivity_scenarios)
        return num_algorithms * total_scenario_runs

    def to_simulation_config(
        self,
        algorithm: AlgorithmType,
        connectivity: float,
        run_number: int,
    ) -> SimulationConfig:
        """Generate a SimulationConfig for a specific run."""
        seed = None
        if self.master_random_seed is not None:
            alg_offset = 0 if algorithm == AlgorithmType.ADAPTIVE else 10000
            conn_offset = int(connectivity * 1000) * 100
            seed = self.master_random_seed + alg_offset + conn_offset + run_number

        network = self.network.model_copy()
        network.connectivity_scenarios = [connectivity]

        return SimulationConfig(
            network=network,
            communication=self.communication,
            scenario=self.scenario,
            coordination=self.coordination,
            experiment_name=f"{self.experiment_name}_{algorithm.value}_conn{int(connectivity*100)}_run{run_number}",
            random_seed=seed,
            output_directory=self.output_directory,
            log_level=self.log_level,
        )

    model_config = {"validate_assignment": True, "extra": "forbid"}


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate experiment configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)


def validate_experiment_config(config_dict: dict[str, Any]) -> ExperimentConfig:
    """Validate experiment configuration from dictionary."""
    return ExperimentConfig(**config_dict)
