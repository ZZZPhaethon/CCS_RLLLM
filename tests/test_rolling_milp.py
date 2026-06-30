import unittest

try:
    import pulp  # noqa: F401

    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

from sim.control.baselines import greedy_shuttle_policy
from sim.control.rolling_milp import RollingMilpController
from sim.environment import CCSEnv, CCSEnvConfig
from sim.metrics import run_episode
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _cold_env(cap_hours: int = 600) -> CCSEnv:
    # Cold start (no initial inventory) so the MILP bound and the controllers face
    # the same empty-system task.
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=cap_hours, randomize_initial_inventory=False)
        ),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


class RollingMilpInterfaceTests(unittest.TestCase):
    def test_controller_accepts_progress_and_lookahead_options(self):
        messages: list[str] = []
        progress = messages.append
        controller = RollingMilpController(
            _cold_env(),
            progress=progress,
            planning_horizon_h=96,
        )
        self.assertEqual(controller.planning_horizon_h, 96)
        self.assertIs(controller.progress, progress)


@unittest.skipUnless(HAVE_PULP, "pulp/CBC not installed")
class RollingMilpTests(unittest.TestCase):
    def test_controller_runs_to_horizon_and_stores_co2(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=12)
        metrics = run_episode(env, controller, seed=1)
        self.assertEqual(metrics.elapsed_hours, 96)
        self.assertGreater(metrics.stored_t, 0.0)

    def test_controller_resets_between_episodes(self):
        env = _cold_env(cap_hours=96)
        controller = RollingMilpController(env, replan_every=12)
        a = run_episode(env, controller, seed=1).stored_t
        b = run_episode(env, controller, seed=1).stored_t  # reused controller
        self.assertEqual(a, b)  # stale plan would make the second run differ

    def test_controller_uses_fixed_horizon_plan_without_storage_goal(self):
        env = _cold_env(cap_hours=96)
        env.reset(seed=1)
        controller = RollingMilpController(env, replan_every=12, planning_horizon_h=48)
        action = controller.policy(env)
        self.assertEqual(len(action), len(env.vessel_ids) + len(env.well_ids))

    def test_empty_vessel_returns_to_best_available_emitter_not_fixed_home(self):
        env = _cold_env(cap_hours=48)
        env.reset(seed=1)
        vessel_id = env.vessel_ids[0]
        home = str(env._routes[vessel_id]["origin"])
        other = next(eid for eid in env.emitter_ids if eid != home)
        terminal = str(env._routes[vessel_id]["destination"])
        env.simulator.state.vessel_berths[vessel_id] = terminal
        env.simulator.vessel_states[vessel_id] = {
            "mode": "berthed",
            "berth": terminal,
            "destination": None,
            "progress": 0.0,
        }
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0
        env.simulator.state.entity_inventory_t[home] = 0.0
        env.simulator.state.entity_inventory_t[other] = 5_000.0

        action = RollingMilpController(env, replan_every=12, planning_horizon_h=48).policy(env)

        self.assertEqual(action[0], env.vessel_go_emitter_action(other))


if __name__ == "__main__":
    unittest.main()
