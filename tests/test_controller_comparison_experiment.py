import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ControllerComparisonExperimentTests(unittest.TestCase):
    def test_controller_comparison_imports_with_current_layout(self):
        from experiments import compare_controllers_same_scenarios as compare

        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_plan_target_t=800.0,
        )
        self.assertEqual(set(factories), {"idle", "greedy_shuttle", "rule_based", "rolling_milp"})

    def test_rule_based_controller_factory_translates_to_discrete_env_action(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.environment import VESSEL_WAIT, WELL_ACTIONS
        from sim.scenario_generation import ScenarioConfig

        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_plan_target_t=None,
        )
        env = compare.make_env(
            target_t=None,
            cap_hours=2,
            scenario_seed_config=ScenarioConfig(episode_hours=2, randomize_initial_inventory=False),
        )
        env.reset(seed=1)

        action = factories["rule_based"](env)(env)

        self.assertEqual(len(action), len(env.action_dims))
        self.assertEqual(action[: len(env.vessel_ids)], [VESSEL_WAIT] * len(env.vessel_ids))
        self.assertEqual(action[len(env.vessel_ids):], [WELL_ACTIONS - 1, 0])

    def test_controller_comparison_cli_smoke_writes_outputs(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            argv = [
                "compare_controllers_same_scenarios.py",
                "--controllers",
                "idle",
                "--seeds",
                "1",
                "--cap-hours",
                "2",
                "--skip-static-milp",
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                compare.main()

            self.assertTrue((output_dir / "controller_comparison_by_seed.csv").exists())
            self.assertTrue((output_dir / "controller_comparison_summary.csv").exists())
            self.assertTrue((output_dir / "static_milp_nominal_benchmark.csv").exists())
            self.assertTrue((output_dir / "controller_comparison_report.md").exists())

    def test_static_milp_time_limit_default_is_long_enough_for_fixed_horizon_runs(self):
        from experiments import compare_controllers_same_scenarios as compare

        with patch.object(sys, "argv", ["compare_controllers_same_scenarios.py"]):
            args = compare.parse_args()

        self.assertEqual(args.static_milp_time_limit_s, 600.0)

    def test_fixed_horizon_cli_forwards_static_milp_solver_controls(self):
        from experiments import compare_controllers_same_scenarios as compare

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            argv = [
                "compare_controllers_same_scenarios.py",
                "--mode",
                "fixed-horizon",
                "--controllers",
                "idle",
                "--seeds",
                "1",
                "--cap-hours",
                "2",
                "--static-milp-time-limit-s",
                "123",
                "--static-milp-gap-rel",
                "0.05",
                "--static-milp-gap-abs",
                "10",
                "--output-dir",
                str(output_dir),
            ]
            benchmark = {
                "case": "static_milp_fixed_horizon_nominal",
                "scenario_signature": "sig",
                "horizon_h": 2,
                "status": "mocked",
                "stored_t": 0.0,
                "deliveries": 0,
                "operating_cost": 0.0,
                "cost_per_stored_t": 0.0,
                "time_limit_s": 123.0,
                "mip_gap_rel": 0.05,
                "mip_gap_abs": 10.0,
            }
            with (
                patch.object(sys, "argv", argv),
                patch.object(compare, "scenario_signature", return_value="sig"),
                patch.object(compare, "static_fixed_horizon_milp_benchmark", return_value=benchmark) as fixed,
            ):
                compare.main()

            fixed.assert_called_once()
            self.assertEqual(fixed.call_args.args, (2, 123.0, 0.05, 10.0))
            self.assertEqual(fixed.call_args.kwargs["seed"], 1)
            self.assertIsNotNone(fixed.call_args.kwargs["scenario_seed_config"])

    def test_static_fixed_horizon_benchmark_passes_mip_gap_controls(self):
        from experiments import compare_controllers_same_scenarios as compare
        from sim.scenario_generation import ScenarioConfig

        result = SimpleNamespace(
            status="mocked",
            stored_t=1.0,
            deliveries=1,
            operating_cost=2.0,
            cost_per_stored_t=2.0,
        )
        with patch.object(compare, "solve_max_storage_fixed_horizon", return_value=result) as solve:
            row = compare.static_fixed_horizon_milp_benchmark(
                720,
                600.0,
                0.01,
                100.0,
                scenario_seed_config=ScenarioConfig(episode_hours=720, randomize_initial_inventory=False),
                seed=7,
            )

        self.assertEqual(row["case"], "static_milp_fixed_horizon_scenario_oracle")
        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["scenario_signature"], compare.scenario_signature(solve.call_args.kwargs["scenario"]))
        self.assertEqual(row["mip_gap_rel"], 0.01)
        self.assertEqual(row["mip_gap_abs"], 100.0)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["horizon_h"], 720)
        self.assertEqual(solve.call_args.kwargs["time_limit_s"], 600.0)
        self.assertEqual(solve.call_args.kwargs["mip_gap_rel"], 0.01)
        self.assertEqual(solve.call_args.kwargs["mip_gap_abs"], 100.0)
        self.assertEqual(solve.call_args.kwargs["scenario"].seed, 7)


if __name__ == "__main__":
    unittest.main()
