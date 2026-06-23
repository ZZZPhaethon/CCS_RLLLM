import unittest

from examples.build_physical_dashboard import DASHBOARD_HOURS, build_demo_action_frames
from sim.visualization import build_demo_trajectory


class DashboardDemoTests(unittest.TestCase):
    def test_dashboard_demo_runs_for_72h_with_visible_logistics_activity(self):
        action_frames = build_demo_action_frames(DASHBOARD_HOURS)
        payload = build_demo_trajectory(hours=DASHBOARD_HOURS, action_frames=action_frames)

        self.assertEqual(DASHBOARD_HOURS, 72)
        self.assertEqual(len(payload["frames"]), 73)
        self.assertEqual(payload["frames"][-1]["time_h"], 72)
        self.assertTrue(any(frame["actions"] for frame in payload["frames"]))
        self.assertTrue(
            any(
                position["leg"] == "outbound_to_terminal"
                for frame in payload["frames"]
                for position in frame["vessel_positions"].values()
            )
        )
        self.assertTrue(
            any(
                position["leg"] == "return_to_origin"
                for frame in payload["frames"]
                for position in frame["vessel_positions"].values()
            )
        )
        self.assertGreater(payload["frames"][-1]["entities"]["aurora_reservoir"]["inventory_t"], 0.0)


if __name__ == "__main__":
    unittest.main()
