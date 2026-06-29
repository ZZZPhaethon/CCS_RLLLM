import unittest

try:
    import pulp  # noqa: F401

    from sim.control.milp import solve_min_makespan
    from sim.control.rolling_milp import RollingMilpController
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim.control.baselines import greedy_shuttle_policy
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import run_episode
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _cold_env(goal_t: float, cap_hours: int = 600) -> CCSEnv:
    # Cold start (no initial inventory) so the MILP bound and the controllers face
    # the same empty-system task.
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=cap_hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=cap_hours, storage_goal_t=goal_t),
    )


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class RollingMilpTests(unittest.TestCase):
    def test_controller_runs_and_reaches_goal(self):
        env = _cold_env(goal_t=1_600.0)
        controller = RollingMilpController(env, replan_every=12)
        metrics = run_episode(env, controller, seed=1)
        self.assertTrue(metrics.reached_target)
        self.assertGreaterEqual(metrics.stored_t, 1_600.0)

    def test_controller_resets_between_episodes(self):
        env = _cold_env(goal_t=1_600.0)
        controller = RollingMilpController(env, replan_every=12)
        a = run_episode(env, controller, seed=1).elapsed_hours
        b = run_episode(env, controller, seed=1).elapsed_hours  # reused controller
        self.assertEqual(a, b)  # stale plan would make the second run differ

    def test_respects_milp_lower_bound_when_nominal(self):
        # On a cold, nominal run the controller cannot beat the open-loop optimum.
        env = _cold_env(goal_t=1_600.0)
        quiet = ScenarioConfig(
            episode_hours=600, randomize_initial_inventory=False,
            capture_noise_std=0.0, capture_outage_rate_per_week=0.0, enable_weather=False,
            well_maintenance_rate_per_week=0.0, injectivity_max_decline=0.0,
            injectivity_noise_std=0.0, berth_outage_rate_per_week=0.0,
        )
        env.scenario_generator = ScenarioGenerator(config=quiet)
        bound = solve_min_makespan(env, target_t=1_600.0).makespan_h
        metrics = run_episode(env, RollingMilpController(env, replan_every=12), seed=1)
        self.assertGreaterEqual(metrics.elapsed_hours + 1e-6, bound)


if __name__ == "__main__":
    unittest.main()
