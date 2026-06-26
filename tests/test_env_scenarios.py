import unittest

from sim.env import CCSEnvConfig
from sim.env_scenarios import build_phase1_plus_yara_env
from sim.metrics import greedy_shuttle_policy, idle_policy, run_episode
from sim.scenario import ScenarioConfig, ScenarioGenerator


class Phase1PlusYaraEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Building the env runs searoute once per vessel; do it a single time.
        cls.env = build_phase1_plus_yara_env(
            scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=72)),
            config=CCSEnvConfig(episode_hours=72, storage_target_rate=0.9),
        )

    def test_real_network_topology(self):
        env = self.env
        self.assertEqual(len(env.vessel_ids), 4)       # four Phase 1 ships
        self.assertEqual(len(env.emitter_ids), 3)      # Brevik, Celsio, Yara
        self.assertEqual(len(env.well_ids), 2)         # two Aurora wells
        self.assertEqual(env.action_dims, [3, 3, 3, 3, 4, 4])

    def test_routes_use_real_distances(self):
        # Yara (NL) -> Oygarden is far longer than Brevik (Norway) -> Oygarden.
        routes = self.env._routes
        yara_vessel = next(v for v, r in routes.items() if r["origin"] == "yara_sluiskil")
        brevik_vessel = next(v for v, r in routes.items() if r["origin"] == "brevik")
        self.assertGreater(routes[yara_vessel]["distance_km"], routes[brevik_vessel]["distance_km"])
        self.assertGreater(routes[brevik_vessel]["distance_km"], 300.0)

    def test_reset_returns_full_observation(self):
        obs = self.env.reset(seed=0)
        self.assertEqual(len(obs), self.env.observation_size)

    def test_idle_episode_stores_nothing(self):
        metrics = run_episode(self.env, idle_policy, seed=1)
        self.assertEqual(metrics.stored_t, 0.0)
        self.assertEqual(metrics.storage_rate, 0.0)

    def test_shuttle_stores_co2_without_overflow_in_a_week(self):
        metrics = run_episode(self.env, greedy_shuttle_policy, seed=2)
        self.assertGreater(metrics.stored_t, 0.0)
        # Real buffers give ~7 days of autonomy, so a short run should not vent.
        self.assertEqual(metrics.vented_t, 0.0)


if __name__ == "__main__":
    unittest.main()
