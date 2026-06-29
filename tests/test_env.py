import unittest

from sim.environment import (
    VESSEL_GO_HOME,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_ACTIONS,
    CCSEnv,
    CCSEnvConfig,
)
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _env(**config) -> CCSEnv:
    return CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=48)),
        config=CCSEnvConfig(episode_hours=48, **config),
    )


class EnvSpaceTests(unittest.TestCase):
    def test_action_and_observation_dimensions(self):
        env = _env()
        self.assertEqual(env.action_dims, [3, 3, WELL_ACTIONS, WELL_ACTIONS])  # 2 vessels, 2 wells
        obs = env.reset(seed=0)
        self.assertEqual(len(obs), env.observation_size)
        self.assertEqual(len(obs), len(env.feature_names))

    def test_reset_returns_finite_normalized_observation(self):
        obs = _env().reset(seed=0)
        self.assertTrue(all(isinstance(x, float) for x in obs))
        self.assertTrue(all(-1e6 < x < 1e6 for x in obs))

    def test_action_mask_shape_matches_action_dims(self):
        env = _env()
        env.reset(seed=0)
        mask = env.action_mask()
        self.assertEqual([len(m) for m in mask], env.action_dims)


class EnvDynamicsTests(unittest.TestCase):
    def test_vessels_start_home_and_can_only_go_to_terminal_or_wait(self):
        env = _env()
        env.reset(seed=0)
        for i, _vid in enumerate(env.vessel_ids):
            self.assertEqual(env.action_mask()[i], [True, False, True])  # WAIT, not GO_HOME, GO_TERMINAL

    def test_sailing_vessel_can_only_wait(self):
        env = _env()
        env.reset(seed=0)
        action = [VESSEL_GO_TERMINAL, VESSEL_GO_TERMINAL, 0, 0]
        env.step(action)
        # Both ships are now sailing -> mask permits WAIT only.
        for i in range(len(env.vessel_ids)):
            self.assertEqual(env.action_mask()[i], [True, False, False])

    def test_well_maintenance_masks_all_but_off(self):
        env = _env()
        env.reset(seed=0)
        well_index = len(env.vessel_ids)  # first well dim
        env.simulator.state.well_available[env.well_ids[0]] = False
        self.assertEqual(env.action_mask()[well_index], [True] + [False] * (WELL_ACTIONS - 1))

    def test_episode_runs_to_done(self):
        env = _env()
        env.reset(seed=1)
        steps = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step([VESSEL_WAIT, VESSEL_WAIT, 0, 0])
            done = terminated or truncated
            self.assertIsInstance(reward, float)
            steps += 1
        self.assertEqual(steps, env.n_steps)
        self.assertIn("storage_rate", info)

    def test_horizon_end_is_truncation_not_termination(self):
        env = _env()
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(env.n_steps):
            _, _, terminated, truncated, _ = env.step([VESSEL_WAIT, VESSEL_WAIT, 0, 0])
        # The operation never truly ends; the horizon is only a time limit, so the
        # trainer must bootstrap the value of the leftover state.
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_observation_has_no_horizon_relative_features(self):
        env = _env()
        self.assertIn("hour_of_week", env.feature_names)
        self.assertIn("backlog_fill", env.feature_names)
        self.assertNotIn("time_fraction", env.feature_names)

    def test_deterministic_for_seed_and_policy(self):
        def rollout():
            env = _env()
            env.reset(seed=123)
            rewards = []
            for _ in range(env.n_steps):
                _, r, _terminated, _truncated, _ = env.step([VESSEL_WAIT, VESSEL_WAIT, 3, 3])
                rewards.append(r)
            return rewards

        self.assertEqual(rollout(), rollout())

    def test_shuttle_policy_actually_stores_co2(self):
        env = _env()
        env.reset(seed=2)
        done = False
        while not done:
            action = []
            for i, _vid in enumerate(env.vessel_ids):
                mask = env.action_mask()[i]
                if mask[VESSEL_GO_TERMINAL]:
                    action.append(VESSEL_GO_TERMINAL)
                elif mask[VESSEL_GO_HOME]:
                    action.append(VESSEL_GO_HOME)
                else:
                    action.append(VESSEL_WAIT)
            action += [WELL_ACTIONS - 1] * len(env.well_ids)  # wells HIGH
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        self.assertGreater(env.cumulative_stored_t, 0.0)
        self.assertGreater(env.ledger.revenue_storage, 0.0)
        self.assertTrue(0.0 <= info["storage_rate"] <= 1.0)


class EnvGuardTests(unittest.TestCase):
    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            _env().step([VESSEL_WAIT, VESSEL_WAIT, 0, 0])

    def test_wrong_action_length_raises(self):
        env = _env()
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            env.step([VESSEL_WAIT])


if __name__ == "__main__":
    unittest.main()
