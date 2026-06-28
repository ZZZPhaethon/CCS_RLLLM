import unittest

try:
    import pulp  # noqa: F401

    from sim.milp import extract_params, solve_min_makespan
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim.env import CCSEnv, CCSEnvConfig
from sim.scenario import ScenarioConfig, ScenarioGenerator
from test_env import _LOCATIONS, _network


def _env(cap_hours: int = 400) -> CCSEnv:
    return CCSEnv(
        _network(),
        _LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=cap_hours)),
        config=CCSEnvConfig(episode_hours=cap_hours),
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


if __name__ == "__main__":
    unittest.main()
