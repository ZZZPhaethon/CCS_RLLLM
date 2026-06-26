import unittest

from sim.env import CCSEnv, CCSEnvConfig
from sim.metrics import (
    EpisodeMetrics,
    aggregate_metrics,
    evaluate,
    greedy_shuttle_policy,
    idle_policy,
    run_episode,
)
from sim.scenario import ScenarioConfig, ScenarioGenerator
from test_env import _LOCATIONS, _network


def _env(episode_hours: int = 48, **config) -> CCSEnv:
    return CCSEnv(
        _network(),
        _LOCATIONS,
        scenario_generator=ScenarioGenerator(config=ScenarioConfig(episode_hours=episode_hours)),
        config=CCSEnvConfig(episode_hours=episode_hours, **config),
    )


class RunEpisodeTests(unittest.TestCase):
    def test_returns_metrics_consistent_with_env(self):
        env = _env()
        metrics = run_episode(env, greedy_shuttle_policy, seed=1)
        self.assertIsInstance(metrics, EpisodeMetrics)
        self.assertAlmostEqual(metrics.storage_rate, env.storage_rate())
        self.assertAlmostEqual(metrics.stored_t, env.cumulative_stored_t)
        self.assertAlmostEqual(metrics.net, env.ledger.net)
        self.assertEqual(metrics.horizon_hours, 48)

    def test_kpis_are_in_sensible_ranges(self):
        metrics = run_episode(_env(), greedy_shuttle_policy, seed=3)
        self.assertTrue(0.0 <= metrics.storage_rate <= 1.0)
        self.assertGreaterEqual(metrics.vented_t, 0.0)
        self.assertGreater(metrics.operating_cost, 0.0)
        self.assertGreaterEqual(metrics.throttle_hours, 0)
        self.assertTrue(0.0 <= metrics.min_pressure_margin_fraction <= 1.0)
        self.assertGreaterEqual(metrics.longest_venting_streak_hours, 0)

    def test_idle_policy_stores_nothing_and_vents(self):
        # A long horizon guarantees the emitter buffers overflow under idling.
        metrics = run_episode(_env(episode_hours=168), idle_policy, seed=5)
        self.assertAlmostEqual(metrics.stored_t, 0.0)
        self.assertEqual(metrics.storage_rate, 0.0)
        self.assertGreater(metrics.vented_t, 0.0)
        self.assertIsNone(metrics.cost_per_stored_t)

    def test_shuttle_beats_idle_on_storage(self):
        idle = run_episode(_env(episode_hours=168), idle_policy, seed=7)
        shuttle = run_episode(_env(episode_hours=168), greedy_shuttle_policy, seed=7)
        self.assertGreater(shuttle.stored_t, idle.stored_t)
        self.assertLess(shuttle.vented_t, idle.vented_t)

    def test_deterministic_for_seed_and_policy(self):
        a = run_episode(_env(), greedy_shuttle_policy, seed=11).as_dict()
        b = run_episode(_env(), greedy_shuttle_policy, seed=11).as_dict()
        self.assertEqual(a, b)

    def test_backlog_growth_obeys_mass_balance(self):
        # backlog grows by exactly captured - stored - vented (in-transit identity).
        m = run_episode(_env(), greedy_shuttle_policy, seed=4)
        self.assertAlmostEqual(m.backlog_growth_t, m.captured_t - m.stored_t - m.vented_t, places=3)

    def test_idle_accumulates_backlog_without_losing_co2(self):
        # Idling stores nothing, so captured CO2 piles up in buffers but is not lost
        # in a short episode (loss rate ~ 0): the new signal, not a contractual miss.
        m = run_episode(_env(), idle_policy, seed=6)
        self.assertGreater(m.backlog_growth_t, 0.0)
        self.assertEqual(m.loss_rate, 0.0)
        self.assertGreater(m.backlog_penalty, 0.0)

    def test_shuttle_grows_backlog_less_than_idle(self):
        idle = run_episode(_env(), idle_policy, seed=8)
        shuttle = run_episode(_env(), greedy_shuttle_policy, seed=8)
        self.assertLess(shuttle.backlog_growth_t, idle.backlog_growth_t)


class AggregateTests(unittest.TestCase):
    def test_evaluate_returns_per_episode_and_summary(self):
        episodes, summary = evaluate(_env(), greedy_shuttle_policy, seeds=[1, 2, 3])
        self.assertEqual(len(episodes), 3)
        self.assertIn("storage_rate", summary)
        self.assertIn("mean", summary["storage_rate"])
        self.assertIn("std", summary["storage_rate"])

    def test_aggregate_handles_single_episode(self):
        episodes = [run_episode(_env(), greedy_shuttle_policy, seed=1)]
        summary = aggregate_metrics(episodes)
        self.assertEqual(summary["storage_rate"]["std"], 0.0)

    def test_aggregate_skips_none_valued_fields(self):
        # idle policy yields cost_per_stored_t = None; aggregation must not crash.
        summary = aggregate_metrics([run_episode(_env(), idle_policy, seed=1)])
        self.assertNotIn("cost_per_stored_t", summary)

    def test_report_renders_all_sections(self):
        text = run_episode(_env(), greedy_shuttle_policy, seed=2).report()
        for token in ("storage rate", "operating cost", "throttle hours", "pressure-risk"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
