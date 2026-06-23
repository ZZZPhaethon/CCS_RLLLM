import json
import unittest

from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.routes import route_distance_km
from sim.scenarios import (
    EOS_SUBSEA_TEMPLATE_LOCATION,
    NATURGASSPARKEN,
    NORTHERN_LIGHTS_PHASE1_DATA_PATH,
    NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH,
    NORTHERN_LIGHTS_PHASE2_DATA_PATH,
    build_northern_lights_phase2_demo,
    build_northern_lights_phase1_plus_yara_demo,
    build_northern_lights_phase1_demo,
)


class ScenarioTests(unittest.TestCase):
    def test_phase1_demo_has_one_hour_step_and_expected_chain(self):
        network, state = build_northern_lights_phase1_demo()

        self.assertEqual(network.time_step_hours, 1.0)
        self.assertIn("brevik", network.entities)
        self.assertIn("northern_pioneer", network.entities)
        self.assertIn("aurora_reservoir", network.entities)
        self.assertIn("aurora_subsea_manifold", network.entities)
        self.assertNotIn("yara", network.entities)
        self.assertEqual(network.downstream_of("oygarden_terminal"), ["oygarden_pipeline"])
        self.assertEqual(network.downstream_of("oygarden_pipeline"), ["aurora_subsea_manifold"])
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a", "aurora_well_c"])
        self.assertEqual(network.downstream_of("aurora_well_a"), ["aurora_reservoir"])
        self.assertEqual(network.downstream_of("aurora_well_c"), ["aurora_reservoir"])
        self.assertEqual(state.entity_inventory_t["northern_pioneer"], 0.0)

    def test_phase1_demo_includes_reference_physical_parameters(self):
        network, _ = build_northern_lights_phase1_demo()
        brevik = network.entities["brevik"]
        celsio = network.entities["celsio"]
        vessel = network.entities["northern_pioneer"]
        terminal = network.entities["oygarden_terminal"]
        pipeline = network.entities["oygarden_pipeline"]
        manifold = network.entities["aurora_subsea_manifold"]
        reservoir = network.entities["aurora_reservoir"]

        self.assertIsInstance(brevik, Emitter)
        self.assertAlmostEqual(brevik.annual_target_export_tpy, 400_000.0)
        self.assertAlmostEqual(brevik.max_production_tph, 56.0)
        self.assertAlmostEqual(brevik.nominal_capture_tph, 400_000.0 / 8760.0)
        self.assertIsInstance(celsio, Emitter)
        self.assertAlmostEqual(celsio.annual_target_export_tpy, 400_000.0)
        self.assertAlmostEqual(celsio.max_production_tph, 56.0)
        self.assertAlmostEqual(celsio.nominal_capture_tph, 400_000.0 / 8760.0)
        self.assertIsInstance(vessel, Vessel)
        self.assertAlmostEqual(vessel.volume_capacity_m3, 7_500.0)
        self.assertAlmostEqual(vessel.speed_knots, 14.0)
        self.assertIsInstance(terminal, Terminal)
        self.assertEqual(terminal.berth_count, 1)
        self.assertEqual(terminal.site_name, "Naturgassparken (Northern Lights Carbon Capture Plant Site)")
        self.assertIsInstance(pipeline, Pipeline)
        self.assertAlmostEqual(pipeline.annual_capacity_tpy, 5_000_000.0)
        self.assertAlmostEqual(pipeline.max_flow_tph, 5_000_000.0 / 8760.0)
        self.assertAlmostEqual(pipeline.length_km, 100.4)
        self.assertEqual(pipeline.route_color, "#ff0000")
        self.assertGreaterEqual(len(pipeline.route_coordinates), 4)
        self.assertEqual(pipeline.route_coordinates[0], NATURGASSPARKEN)
        self.assertEqual(pipeline.route_coordinates[-1], EOS_SUBSEA_TEMPLATE_LOCATION)
        self.assertAlmostEqual(route_distance_km(pipeline.route_coordinates), 100.4, delta=0.5)
        self.assertIsInstance(manifold, SubseaManifold)
        self.assertAlmostEqual(manifold.max_flow_tph, 5_000_000.0 / 8760.0)
        self.assertIsInstance(reservoir, Reservoir)
        self.assertAlmostEqual(reservoir.depth_m, 2_600.0)

    def test_phase1_demo_emitters_can_be_fully_curtailed(self):
        network, _ = build_northern_lights_phase1_demo()

        for entity in network.entities.values():
            if isinstance(entity, Emitter):
                self.assertEqual(entity.min_utilization, 0.0)

    def test_phase1_demo_marks_line_source_assumptions(self):
        network, _ = build_northern_lights_phase1_demo()
        reservoir = network.entities["aurora_reservoir"]

        self.assertIsInstance(reservoir, Reservoir)
        self.assertIsNotNone(reservoir.line_source_parameters)
        self.assertAlmostEqual(reservoir.line_source_parameters.permeability_md, 100.0)
        self.assertEqual(reservoir.line_source_parameter_status["well_radius_m"], "derived_from_concept_report")
        self.assertEqual(reservoir.line_source_parameter_status["viscosity_pa_s"], "assumed")
        self.assertEqual(reservoir.line_source_parameter_status["co2_density_kg_m3"], "assumed")
        self.assertEqual(reservoir.line_source_observation_radii_m, (100.0, 1000.0))

    def test_phase1_demo_loads_reference_values_from_external_data_file(self):
        network, _ = build_northern_lights_phase1_demo()
        with NORTHERN_LIGHTS_PHASE1_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        reservoir = network.entities["aurora_reservoir"]
        pipeline = network.entities["oygarden_pipeline"]

        self.assertIsInstance(reservoir, Reservoir)
        self.assertAlmostEqual(
            reservoir.line_source_parameters.permeability_md,
            payload["line_source_parameters"]["permeability_md"],
        )
        self.assertAlmostEqual(pipeline.length_km, payload["pipeline"]["length_km"])

    def test_phase2_demo_loads_public_scenario_with_current_two_wells(self):
        network, state = build_northern_lights_phase2_demo()
        with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        emitters = [entity for entity in network.entities.values() if isinstance(entity, Emitter)]
        vessels = [entity for entity in network.entities.values() if isinstance(entity, Vessel)]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertEqual(len(emitters), 5)
        self.assertEqual(len(vessels), 8)
        self.assertEqual(len(wells), 2)
        self.assertNotIn("aurora_phase2_well_1", network.entities)
        self.assertAlmostEqual(
            sum(emitter.annual_target_export_tpy or 0.0 for emitter in emitters),
            payload["contracted_annual_target_export_tpy"],
        )
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah", "aurora_well_c1_h"])
        self.assertEqual(network.downstream_of("aurora_well_a7_ah"), ["aurora_reservoir"])
        self.assertEqual(network.downstream_of("aurora_well_c1_h"), ["aurora_reservoir"])
        self.assertEqual(state.entity_inventory_t["stockholm_exergi"], 0.0)

    def test_phase1_plus_yara_demo_has_three_emitters_four_ships_and_two_wells(self):
        network, state = build_northern_lights_phase1_plus_yara_demo()
        with NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        emitters = [entity for entity in network.entities.values() if isinstance(entity, Emitter)]
        vessels = [entity for entity in network.entities.values() if isinstance(entity, Vessel)]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertEqual(len(emitters), 3)
        self.assertEqual(len(vessels), 4)
        self.assertEqual(len(wells), 2)
        self.assertIn("yara_sluiskil", network.entities)
        self.assertAlmostEqual(
            sum(emitter.annual_target_export_tpy or 0.0 for emitter in emitters),
            payload["contracted_annual_target_export_tpy"],
        )
        self.assertEqual(network.downstream_of("aurora_subsea_manifold"), ["aurora_well_a7_ah", "aurora_well_c1_h"])
        self.assertEqual(state.entity_inventory_t["yara_sluiskil"], 0.0)

    def test_phase1_plus_yara_pipeline_and_wells_use_1_5_mtpa_capacity(self):
        network, _ = build_northern_lights_phase1_plus_yara_demo()
        expected_tph = 1_500_000.0 / 8760.0

        pipeline = network.entities["oygarden_pipeline"]
        wells = [entity for entity in network.entities.values() if isinstance(entity, InjectionWell)]

        self.assertIsInstance(pipeline, Pipeline)
        self.assertAlmostEqual(pipeline.max_flow_tph, expected_tph)
        for well in wells:
            self.assertAlmostEqual(well.max_injection_tph, expected_tph)


if __name__ == "__main__":
    unittest.main()
