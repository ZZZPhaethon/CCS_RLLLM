import unittest

from sim.actions import ActionFrame, ActionProposal
from sim.entities import InjectionWell, PhysicalState, Pipeline, Terminal, Vessel
from sim.network import PhysicalNetwork
from sim.simulator import PhysicalSimulator


class PhysicalSimulatorTests(unittest.TestCase):
    def test_sail_to_moves_vessel_off_berth_and_blocks_loading_until_arrival(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0, speed_knots=10.0))
        state = PhysicalState(vessel_berths={"ship_1": "origin"})
        simulator = PhysicalSimulator(
            network,
            state,
            routes={
                "ship_1": {
                    "origin": "origin",
                    "destination": "terminal",
                    "distance_km": 100.0,
                    "speed_knots": 10.0,
                    "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                    "return_coordinates": [(0.0, 1.0), (0.0, 0.0)],
                }
            },
            locations={
                "origin": (0.0, 0.0),
                "terminal": (0.0, 1.0),
            },
        )

        record = simulator.step(
            ActionFrame(
                time_h=0.0,
                proposals=[
                    ActionProposal(
                        agent_id="ship_agent",
                        entity_id="ship_1",
                        verb="sail_to",
                        params={"destination_id": "terminal"},
                    )
                ],
            )
        )

        self.assertNotIn("ship_1", record.step_result.state.vessel_berths)
        self.assertFalse(record.vessel_positions["ship_1"]["at_berth"])
        self.assertGreater(record.vessel_positions["ship_1"]["lon"], 0.0)
        self.assertLess(record.vessel_positions["ship_1"]["lon"], 1.0)

    def test_sail_to_updates_berth_when_vessel_arrives(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0, speed_knots=100.0))
        state = PhysicalState(vessel_berths={"ship_1": "origin"})
        simulator = PhysicalSimulator(
            network,
            state,
            routes={
                "ship_1": {
                    "origin": "origin",
                    "destination": "terminal",
                    "distance_km": 10.0,
                    "speed_knots": 100.0,
                    "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                    "return_coordinates": [(0.0, 1.0), (0.0, 0.0)],
                }
            },
            locations={
                "origin": (0.0, 0.0),
                "terminal": (0.0, 1.0),
            },
        )

        record = simulator.step(
            ActionFrame(
                time_h=0.0,
                proposals=[
                    ActionProposal(
                        agent_id="ship_agent",
                        entity_id="ship_1",
                        verb="sail_to",
                        params={"destination_id": "terminal"},
                    )
                ],
            )
        )

        self.assertEqual(record.step_result.state.vessel_berths["ship_1"], "terminal")
        self.assertTrue(record.vessel_positions["ship_1"]["at_berth"])

    def test_step_record_preserves_proposed_committed_and_executed_actions(self):
        network = PhysicalNetwork(time_step_hours=1.0)
        network.add_entity(Terminal("terminal", storage_capacity_t=1_000.0, berth_count=1))
        network.add_entity(Pipeline("pipeline", max_flow_tph=500.0, ramp_tph=500.0))
        network.add_entity(InjectionWell("well", max_injection_tph=300.0))
        network.connect("terminal", "pipeline")
        network.connect("pipeline", "well")
        state = PhysicalState(entity_inventory_t={"terminal": 500.0})
        simulator = PhysicalSimulator(network, state)
        action_frame = ActionFrame(
            time_h=0.0,
            proposals=[
                ActionProposal(
                    agent_id="pipeline_agent",
                    entity_id="pipeline",
                    verb="set_flow",
                    params={"flow_tph": 500.0},
                )
            ],
        )

        record = simulator.step(action_frame)

        self.assertIs(record.action_frame, action_frame)
        self.assertEqual(record.committed_action_frame.actions, {"pipeline": {"flow_tph": 500.0}})
        self.assertEqual(record.step_result.as_dict()["flows_t"]["terminal->pipeline"], 300.0)
        self.assertTrue(any(violation.violation_type == "flow_clipped" for violation in record.step_result.violations))
        self.assertEqual(record.as_dict()["executed"]["flows_t"]["terminal->pipeline"], 300.0)


if __name__ == "__main__":
    unittest.main()
