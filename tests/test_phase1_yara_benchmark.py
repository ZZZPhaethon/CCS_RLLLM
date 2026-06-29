import unittest

from experiments.benchmark_phase1_yara_milp import (
    make_nominal_scenario_config,
    make_phase1_yara_goal_env,
)


class Phase1YaraBenchmarkHelpersTests(unittest.TestCase):
    def test_goal_env_uses_real_phase1_yara_network(self):
        env = make_phase1_yara_goal_env(target_t=15_000.0, episode_hours=720)

        self.assertEqual(env.config.storage_goal_t, 15_000.0)
        self.assertIn("yara_sluiskil", env.emitter_ids)
        self.assertEqual(len(env.emitter_ids), 3)
        self.assertEqual(len(env.vessel_ids), 4)
        self.assertGreaterEqual(
            min(env.network.entities[vid].capacity_t for vid in env.vessel_ids),
            7_500.0,
        )

    def test_nominal_scenario_config_disables_random_disturbances(self):
        config = make_nominal_scenario_config(episode_hours=720)

        self.assertEqual(config.episode_hours, 720)
        self.assertFalse(config.randomize_initial_inventory)
        self.assertFalse(config.enable_weather)
        self.assertEqual(config.capture_noise_std, 0.0)
        self.assertEqual(config.well_maintenance_rate_per_week, 0.0)
        self.assertEqual(config.berth_outage_rate_per_week, 0.0)


if __name__ == "__main__":
    unittest.main()
