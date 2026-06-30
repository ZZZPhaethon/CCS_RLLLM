import unittest

from experiments import benchmark_phase1_yara_milp as benchmark


class Phase1YaraBenchmarkHelpersTests(unittest.TestCase):
    def test_env_uses_real_phase1_yara_network(self):
        self.assertFalse(hasattr(benchmark, "make_phase1_yara_goal_env"))
        env = benchmark.make_phase1_yara_env(episode_hours=720)

        self.assertIn("yara_sluiskil", env.emitter_ids)
        self.assertEqual(len(env.emitter_ids), 3)
        self.assertEqual(len(env.vessel_ids), 4)
        self.assertGreaterEqual(
            min(env.network.entities[vid].capacity_t for vid in env.vessel_ids),
            7_500.0,
        )

    def test_nominal_scenario_config_disables_random_disturbances(self):
        config = benchmark.make_nominal_scenario_config(episode_hours=720)

        self.assertEqual(config.episode_hours, 720)
        self.assertFalse(config.randomize_initial_inventory)
        self.assertFalse(config.enable_weather)
        self.assertEqual(config.capture_noise_std, 0.0)
        self.assertEqual(config.well_maintenance_rate_per_week, 0.0)


if __name__ == "__main__":
    unittest.main()
