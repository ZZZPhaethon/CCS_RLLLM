import unittest


class OperationsModuleTests(unittest.TestCase):
    def test_action_operations_are_split_by_domain(self):
        import sim.operations.capture
        import sim.operations.injection
        import sim.operations.loading
        import sim.operations.snapshot
        import sim.operations.transport
        import sim.operations.unloading

        self.assertTrue(hasattr(sim.operations.capture, "apply_capture"))
        self.assertTrue(hasattr(sim.operations.injection, "inject_to_well"))
        self.assertTrue(hasattr(sim.operations.loading, "apply_loading"))
        self.assertTrue(hasattr(sim.operations.snapshot, "snapshot_network"))
        self.assertTrue(hasattr(sim.operations.transport, "project_pipeline_outflow"))
        self.assertTrue(hasattr(sim.operations.unloading, "project_terminal_unload"))


if __name__ == "__main__":
    unittest.main()
