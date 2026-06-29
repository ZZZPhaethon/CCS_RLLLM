"""Factory helpers that wire the real public scenarios into :class:`CCSEnv`.

These build a ready-to-train RL environment on the calibrated Northern Lights
network (real emitters, the four 7,500 t Phase 1 ships, the single-berth Oygarden
terminal, the Aurora reservoir and - when available - the real hourly capture
profiles) instead of a toy network, so metrics are research-meaningful.
"""

from __future__ import annotations

from ..physical.economics import CostModel
from ..physical.scenario import ScenarioGenerator
from ..physical.scenarios import (
    _load_phase1_plus_yara_data,
    build_northern_lights_phase1_plus_yara_demo,
)
from .env import CCSEnv, CCSEnvConfig

Coordinate = tuple[float, float]


def _scenario_locations(data: dict) -> dict[str, Coordinate]:
    return {
        location_id: (float(values[0]), float(values[1]))
        for location_id, values in data["locations"].items()
    }


def build_phase1_plus_yara_env(
    scenario_generator: ScenarioGenerator | None = None,
    cost_model: CostModel | None = None,
    config: CCSEnvConfig | None = None,
) -> CCSEnv:
    """A ``CCSEnv`` on the real Phase 1 + Yara commercial ramp-up network."""
    network, _state = build_northern_lights_phase1_plus_yara_demo()
    locations = _scenario_locations(_load_phase1_plus_yara_data())
    return CCSEnv(
        network,
        locations,
        scenario_generator=scenario_generator,
        cost_model=cost_model,
        config=config,
    )
