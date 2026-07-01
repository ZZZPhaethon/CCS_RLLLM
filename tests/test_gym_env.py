import unittest

try:
    import numpy as np
    import gymnasium  # noqa: F401

    from sim.environment.gym_adapter import CCSGymEnv, flat_vessel_action_mask
    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False

from sim.environment import CCSEnv, CCSEnvConfig
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator
from tests.fixtures.toy_networks import TOY_TWO_SOURCE_LOCATIONS, make_toy_two_source_network


def _gym_env() -> "CCSGymEnv":
    native = CCSEnv(
        make_toy_two_source_network(),
        TOY_TWO_SOURCE_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=24)),
        config=CCSEnvConfig(episode_hours=24),
    )
    return CCSGymEnv(native)


@unittest.skipUnless(HAVE_GYM, "gymnasium/numpy not installed")
class GymWrapperTests(unittest.TestCase):
    def test_spaces_match_native_env(self):
        env = _gym_env()
        self.assertEqual(list(env.action_space["vessels"].nvec), env.env.vessel_action_dims)
        self.assertEqual(env.action_space["wells"].shape, (len(env.env.well_ids),))
        self.assertEqual(env.observation_space.shape, (env.env.observation_size,))

    def test_reset_returns_array_and_info(self):
        env = _gym_env()
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.env.observation_size,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIsInstance(info, dict)

    def test_vessel_action_masks_flatten_in_multidiscrete_order(self):
        env = _gym_env()
        env.reset(seed=0)
        masks = env.action_masks()
        self.assertEqual(masks.shape, (sum(env.env.vessel_action_dims),))
        self.assertEqual(masks.dtype, bool)

    def test_step_returns_five_tuple_with_truncation(self):
        env = _gym_env()
        env.reset(seed=0)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            legal = {
                "vessels": np.zeros(len(env.env.vessel_ids), dtype=np.int64),
                "wells": np.zeros(len(env.env.well_ids), dtype=np.float32),
            }
            _obs, reward, terminated, truncated, _info = env.step(legal)
            steps += 1
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(steps, env.env.n_steps)

    def test_flat_vessel_action_mask_helper(self):
        flat = flat_vessel_action_mask([[True, False, True], [True, True]])
        self.assertEqual(list(flat), [True, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
