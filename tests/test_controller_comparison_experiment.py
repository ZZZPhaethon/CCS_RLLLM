import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ControllerComparisonExperimentTests(unittest.TestCase):
    def test_controller_comparison_imports_with_current_layout(self):
        from experiments import compare_controllers_same_scenarios as compare

        factories = compare.controller_factories(
            replan_every=12,
            progress=lambda _message: None,
            rolling_plan_target_t=800.0,
        )
        self.assertEqual(set(factories), {"idle", "greedy_shuttle", "rolling_milp"})

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


if __name__ == "__main__":
    unittest.main()
