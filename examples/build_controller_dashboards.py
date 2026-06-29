from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.actions import ActionFrame
from sim.entities import Emitter, InjectionWell, PhysicalState, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.env import CCSEnv, CCSEnvConfig
from sim.metrics import greedy_shuttle_policy
from sim.network import PhysicalNetwork
from sim.rolling_milp import RollingMilpController
from sim.scenario import ScenarioConfig, ScenarioGenerator
from sim.visualization import build_trajectory, render_dashboard_html


HOURS = 720
VESSEL_CAPACITY_T = 7_500.0

LOCATIONS = {
    "brevik": {"lat": 59.05, "lon": 9.70, "label": "Brevik"},
    "oslo": {"lat": 59.86, "lon": 10.84, "label": "Celsio Oslo"},
    "oygarden": {"lat": 60.58, "lon": 4.84, "label": "Oygarden Terminal"},
    "pipeline": {"lat": 60.56, "lon": 4.30, "label": "Offshore CO2 Pipeline"},
    "manifold": {"lat": 60.55, "lon": 3.65, "label": "Subsea Manifold"},
    "well_1": {"lat": 60.55, "lon": 3.60, "label": "Injection Well 1"},
    "well_2": {"lat": 60.52, "lon": 3.68, "label": "Injection Well 2"},
    "aurora": {"lat": 60.55, "lon": 3.46, "label": "Aurora Reservoir"},
}


def build_dashboard_network() -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("brevik", nominal_capture_tph=80.0, buffer_capacity_t=15_000.0))
    network.add_entity(Emitter("oslo", nominal_capture_tph=60.0, buffer_capacity_t=15_000.0))
    network.add_entity(
        Vessel(
            "ship_1",
            capacity_t=VESSEL_CAPACITY_T,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(
        Vessel(
            "ship_2",
            capacity_t=VESSEL_CAPACITY_T,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(Terminal("oygarden", storage_capacity_t=15_000.0, berth_count=2))
    network.add_entity(Pipeline("pipeline", max_flow_tph=400.0, ramp_tph=400.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=400.0))
    network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
    network.add_entity(InjectionWell("well_2", max_injection_tph=200.0))
    network.add_entity(
        Reservoir(
            "aurora",
            storage_capacity_t=1e7,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    network.connect("brevik", "ship_1")
    network.connect("oslo", "ship_2")
    network.connect("ship_1", "oygarden")
    network.connect("ship_2", "oygarden")
    network.connect("oygarden", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well_1")
    network.connect("manifold", "well_2")
    network.connect("well_1", "aurora")
    network.connect("well_2", "aurora")
    return network


def _tuple_locations() -> dict[str, tuple[float, float]]:
    return {
        entity_id: (float(location["lat"]), float(location["lon"]))
        for entity_id, location in LOCATIONS.items()
    }


class EnvPolicyActionGenerator:
    """Adapter from ``CCSEnv`` policies to dashboard physical action frames."""

    def __init__(
        self,
        network,
        routes,
        policy_factory: Callable[[CCSEnv], Callable[[CCSEnv], list[int]]],
    ) -> None:
        self.env = CCSEnv(
            network,
            _tuple_locations(),
            scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=HOURS)),
            config=CCSEnvConfig(episode_hours=HOURS),
            routes=routes,
        )
        self.policy = policy_factory(self.env)

    def attach_simulator(self, simulator) -> None:
        self.env.simulator = simulator

    def next_action_frame(self, state) -> ActionFrame:
        self.env.simulator.state = state
        action = self.policy(self.env)
        proposals = self.env._build_proposals(action)
        return ActionFrame(time_h=state.time_h, proposals=proposals)


def _write_controller_dashboard(
    output_path: Path,
    title: str,
    policy_factory: Callable[[CCSEnv], Callable[[CCSEnv], list[int]]],
) -> Path:
    network = build_dashboard_network()
    payload = build_trajectory(
        network,
        state=PhysicalState(),
        locations=LOCATIONS,
        hours=HOURS,
        action_generator_factory=lambda network, routes: EnvPolicyActionGenerator(
            network, routes, policy_factory
        ),
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    return output_path


def main() -> None:
    outputs = [
        _write_controller_dashboard(
            Path("docs") / "controller_greedy_shuttle_720h_dashboard.html",
            "Greedy Shuttle Controller - 720 h CCS Dispatch",
            lambda _env: greedy_shuttle_policy,
        ),
        _write_controller_dashboard(
            Path("docs") / "controller_rolling_milp_720h_dashboard.html",
            "Rolling MILP Controller - 720 h CCS Dispatch",
            lambda env: RollingMilpController(
                env,
                replan_every=168,
                plan_target_t=10_000.0,
            ),
        ),
    ]
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
