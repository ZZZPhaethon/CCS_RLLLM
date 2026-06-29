import unittest

try:
    import pulp  # noqa: F401

    from sim.control.milp import extract_params, solve_min_makespan
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim.environment import CCSEnv, CCSEnvConfig
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.network import PhysicalNetwork
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env(cap_hours: int = 400) -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=cap_hours)),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


def _unbalanced_source_env() -> CCSEnv:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("slow_source", nominal_capture_tph=5.0, buffer_capacity_t=2_000.0))
    network.add_entity(Emitter("fast_source", nominal_capture_tph=100.0, buffer_capacity_t=2_000.0))
    network.add_entity(Vessel("short_ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Vessel("long_ship", capacity_t=1_000.0, loading_rate_tph=1_000.0, unloading_rate_tph=1_000.0, speed_knots=1.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=5_000.0, berth_count=2))
    network.add_entity(Pipeline("pipeline", max_flow_tph=1_000.0, ramp_tph=1_000.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=1_000.0))
    network.add_entity(InjectionWell("well", max_injection_tph=1_000.0))
    network.add_entity(Reservoir("reservoir", storage_capacity_t=1e7, initial_pressure_bar=100.0, pressure_at_capacity_bar=200.0, max_pressure_bar=200.0))
    network.connect("slow_source", "short_ship")
    network.connect("fast_source", "long_ship")
    network.connect("short_ship", "terminal")
    network.connect("long_ship", "terminal")
    network.connect("terminal", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well")
    network.connect("well", "reservoir")
    return CCSEnv(
        network,
        {"slow_source": (0.0, 0.0), "fast_source": (0.0, 1.0), "terminal": (0.0, 2.0)},
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=300, randomize_initial_inventory=False)),
        config=CCSEnvConfig(episode_hours=300),
        routes={
            "short_ship": {"origin": "slow_source", "destination": "terminal", "distance_km": 1.852, "speed_knots": 1.0},
            "long_ship": {"origin": "fast_source", "destination": "terminal", "distance_km": 92.6, "speed_knots": 1.0},
        },
    )


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class MilpTests(unittest.TestCase):
    def test_extract_params_from_network(self):
        env = _env()
        vessels, inj_cap, capture_rate, term_cap = extract_params(env)
        self.assertEqual(len(vessels), 2)
        self.assertGreater(inj_cap, 0.0)
        self.assertEqual(term_cap, 6_000.0)
        for v in vessels:
            self.assertGreater(v.round_trip_h, v.startup_h)

    def test_solves_and_reaches_target(self):
        result = solve_min_makespan(_env(), target_t=1_000.0)
        self.assertEqual(result.status, "Optimal")
        self.assertTrue(result.reached)
        self.assertGreater(result.makespan_h, 0.0)
        self.assertGreaterEqual(result.deliveries, 1)
        self.assertGreater(result.operating_cost, 0.0)

    def test_makespan_respects_injection_lower_bound(self):
        # Cannot finish faster than first-arrival + target/injection-capacity.
        env = _env()
        vessels, inj_cap, _cap, _term = extract_params(env)
        target = 1_500.0
        result = solve_min_makespan(env, target_t=target)
        min_startup = min(v.startup_h for v in vessels)
        lower_bound = min_startup + target / inj_cap
        self.assertGreaterEqual(result.makespan_h + 1e-6, lower_bound)

    def test_larger_target_takes_longer(self):
        small = solve_min_makespan(_env(), target_t=800.0)
        large = solve_min_makespan(_env(), target_t=2_400.0)
        self.assertGreaterEqual(large.makespan_h, small.makespan_h)

    def test_fixed_route_milp_cannot_borrow_capture_from_other_emitters(self):
        result = solve_min_makespan(_unbalanced_source_env(), target_t=1_000.0)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.schedule["short_ship"], [])
        self.assertEqual(result.schedule["long_ship"], [60])


if __name__ == "__main__":
    unittest.main()
