import unittest

from sim.environment import (
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_ACTIONS,
    WELL_MODE_RATES_MTPA,
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
        self.assertEqual(env.action_dims, [4, 4, WELL_ACTIONS, WELL_ACTIONS])  # 2 vessels, 2 emitters, 2 wells
        self.assertEqual(WELL_MODE_RATES_MTPA, (0.0, 0.5, 1.0, 1.5, 2.0))
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
    def test_vessels_start_at_emitters_and_can_choose_terminal_or_other_emitters(self):
        env = _env()
        env.reset(seed=0)
        source_a_action = env.vessel_go_emitter_action("source_a")
        source_b_action = env.vessel_go_emitter_action("source_b")

        vessel_a_mask = env.action_mask()[env.vessel_ids.index("vessel_a")]
        vessel_b_mask = env.action_mask()[env.vessel_ids.index("vessel_b")]

        self.assertTrue(vessel_a_mask[VESSEL_WAIT])
        self.assertTrue(vessel_a_mask[VESSEL_GO_TERMINAL])
        self.assertFalse(vessel_a_mask[source_a_action])
        self.assertTrue(vessel_a_mask[source_b_action])
        self.assertTrue(vessel_b_mask[VESSEL_GO_TERMINAL])
        self.assertTrue(vessel_b_mask[source_a_action])
        self.assertFalse(vessel_b_mask[source_b_action])

    def test_sailing_vessel_can_only_wait(self):
        env = _env()
        env.reset(seed=0)
        action = [VESSEL_GO_TERMINAL, VESSEL_GO_TERMINAL, 0, 0]
        env.step(action)
        # Both ships are now sailing -> mask permits WAIT only.
        for i in range(len(env.vessel_ids)):
            self.assertEqual(env.action_mask()[i], [True, False, False, False])

    def test_well_actions_are_stable_rate_modes_with_off_only_for_maintenance(self):
        env = _env()
        env.reset(seed=0)
        well_index = len(env.vessel_ids)  # first well dim
        self.assertEqual(env.action_mask()[well_index], [False, True, True, True, True])

        env.simulator.state.well_available[env.well_ids[0]] = False
        self.assertEqual(env.action_mask()[well_index], [True] + [False] * (WELL_ACTIONS - 1))

    def test_lowest_available_well_rate_injects_half_mtpa_per_year(self):
        env = _env()
        env.reset(seed=0)
        terminal_id = env.terminal_ids[0]
        reservoir_id = env.reservoir_ids[0]
        env.simulator.state.entity_inventory_t[terminal_id] = 1_000.0
        before = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)

        low_rate_action = WELL_MODE_RATES_MTPA.index(0.5)
        env.step([VESSEL_WAIT, VESSEL_WAIT, low_rate_action, low_rate_action])

        expected_per_well_tph = 0.5 * 1_000_000.0 / (365.25 * 24.0)
        after = env.simulator.state.entity_inventory_t.get(reservoir_id, 0.0)
        self.assertAlmostEqual(after - before, 2 * expected_per_well_tph)

    def test_vessel_can_milk_run_between_emitters_before_terminal(self):
        env = _env()
        env.reset(seed=0)
        vessel_id = "vessel_a"
        vessel = env.network.entities[vessel_id]
        env.simulator.state.entity_inventory_t["source_a"] = 200.0
        env.simulator.state.entity_inventory_t["source_b"] = 1_000.0
        env.simulator.state.entity_inventory_t[vessel_id] = 0.0
        env.simulator.state.entity_inventory_t["vessel_b"] = env.network.entities["vessel_b"].capacity_t
        env._routes[vessel_id]["speed_knots"] = 10_000.0

        low_rate_action = WELL_MODE_RATES_MTPA.index(0.5)
        env.step([VESSEL_WAIT, VESSEL_WAIT, low_rate_action, low_rate_action])
        first_load_t = env.simulator.state.entity_inventory_t[vessel_id]
        expected_first_load_t = 200.0 + env.simulator.state.last_capture_tph["source_a"]
        self.assertAlmostEqual(first_load_t, expected_first_load_t)

        env.step([env.vessel_go_emitter_action("source_b"), VESSEL_WAIT, low_rate_action, low_rate_action])
        self.assertEqual(env.simulator.state.vessel_berths[vessel_id], "source_b")

        env.step([VESSEL_WAIT, VESSEL_WAIT, low_rate_action, low_rate_action])
        final_load_t = env.simulator.state.entity_inventory_t[vessel_id]
        self.assertGreater(final_load_t, first_load_t)
        self.assertLessEqual(final_load_t, vessel.capacity_t)

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
