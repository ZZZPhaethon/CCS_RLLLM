import unittest


class EntityImportTests(unittest.TestCase):
    def test_entity_submodules_are_available_with_legacy_exports(self):
        from sim.entities import Emitter as LegacyEmitter
        from sim.entities import PhysicalState as LegacyPhysicalState
        from sim.entities import SubseaManifold as LegacySubseaManifold
        from sim.entities.emitter import Emitter
        from sim.entities.manifold import SubseaManifold
        from sim.entities.pipeline import Pipeline
        from sim.entities.state import PhysicalState
        from sim.entities.storage import InjectionWell, Reservoir
        from sim.entities.terminal import Terminal
        from sim.entities.vessel import Vessel

        self.assertIs(LegacyEmitter, Emitter)
        self.assertIs(LegacyPhysicalState, PhysicalState)
        self.assertIs(LegacySubseaManifold, SubseaManifold)
        self.assertEqual(Vessel.__name__, "Vessel")
        self.assertEqual(Terminal.__name__, "Terminal")
        self.assertEqual(Pipeline.__name__, "Pipeline")
        self.assertEqual(InjectionWell.__name__, "InjectionWell")
        self.assertEqual(Reservoir.__name__, "Reservoir")


if __name__ == "__main__":
    unittest.main()
