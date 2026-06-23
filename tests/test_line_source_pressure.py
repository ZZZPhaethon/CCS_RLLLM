import unittest
import json

from sim.line_source import (
    LineSourceParameters,
    annual_mt_to_kg_s,
    bottomhole_pressure_bar,
    load_line_source_parameters,
    multiwell_bottomhole_pressures_bar,
    pressure_at_radius_bar,
    variable_rate_pressure_at_radius_bar,
)


class LineSourcePressureTests(unittest.TestCase):
    def setUp(self):
        self.params = LineSourceParameters(
            initial_pressure_bar=260.0,
            permeability_md=500.0,
            thickness_m=173.0,
            porosity_fraction=0.22,
            total_compressibility_1_pa=7.0e-10,
            viscosity_pa_s=6.0e-5,
            co2_density_kg_m3=630.0,
            well_radius_m=0.1,
            skin=0.0,
        )

    def test_annual_mt_to_kg_s_uses_calendar_year(self):
        self.assertAlmostEqual(annual_mt_to_kg_s(1.0), 31.688, places=3)

    def test_pressure_response_scales_linearly_with_injection_rate(self):
        pressure_1x = pressure_at_radius_bar(self.params, 1.5, elapsed_days=365.25, radius_m=100.0)
        pressure_2x = pressure_at_radius_bar(self.params, 3.0, elapsed_days=365.25, radius_m=100.0)

        self.assertAlmostEqual(pressure_2x - self.params.initial_pressure_bar, 2.0 * (pressure_1x - self.params.initial_pressure_bar))

    def test_variable_rate_pressure_uses_rate_change_superposition(self):
        pressure = variable_rate_pressure_at_radius_bar(
            self.params,
            [(0.0, 1.0), (1.0, 2.0)],
            elapsed_days=2.0,
            radius_m=100.0,
        )

        expected = self.params.initial_pressure_bar
        expected += pressure_at_radius_bar(self.params, 1.0, elapsed_days=2.0, radius_m=100.0) - self.params.initial_pressure_bar
        expected += pressure_at_radius_bar(self.params, 1.0, elapsed_days=1.0, radius_m=100.0) - self.params.initial_pressure_bar
        constant_current_rate = pressure_at_radius_bar(self.params, 2.0, elapsed_days=2.0, radius_m=100.0)

        self.assertAlmostEqual(pressure, expected)
        self.assertNotAlmostEqual(pressure, constant_current_rate)

    def test_bottomhole_pressure_exceeds_far_field_pressure(self):
        bottomhole = bottomhole_pressure_bar(self.params, 1.5, elapsed_days=365.25)
        far_field = pressure_at_radius_bar(self.params, 1.5, elapsed_days=365.25, radius_m=1000.0)

        self.assertGreater(bottomhole, far_field)
        self.assertGreater(far_field, self.params.initial_pressure_bar)

    def test_multiwell_bottomhole_pressure_superposes_interference(self):
        pressures = multiwell_bottomhole_pressures_bar(
            self.params,
            {"well_a": 0.8, "well_b": 0.4},
            elapsed_days=365.25,
            well_distances_m={
                "well_a": {"well_b": 500.0},
                "well_b": {"well_a": 500.0},
            },
        )

        expected_a = bottomhole_pressure_bar(self.params, 0.8, elapsed_days=365.25) + (
            pressure_at_radius_bar(self.params, 0.4, elapsed_days=365.25, radius_m=500.0)
            - self.params.initial_pressure_bar
        )
        expected_b = bottomhole_pressure_bar(self.params, 0.4, elapsed_days=365.25) + (
            pressure_at_radius_bar(self.params, 0.8, elapsed_days=365.25, radius_m=500.0)
            - self.params.initial_pressure_bar
        )

        self.assertAlmostEqual(pressures["well_a"], expected_a)
        self.assertAlmostEqual(pressures["well_b"], expected_b)

    def test_can_load_northern_lights_parameter_file(self):
        loaded = load_line_source_parameters("data/northern_lights_line_source_parameters.json")

        self.assertEqual(loaded.initial_pressure_bar, 260.0)
        self.assertEqual(loaded.permeability_md, 100.0)
        self.assertEqual(loaded.thickness_m, 173.0)

    def test_parameter_file_audits_sources_and_confidence(self):
        with open("data/northern_lights_line_source_parameters.json", encoding="utf-8") as handle:
            payload = json.load(handle)

        audit_rows = payload["line_source_input_audit"]
        audited_parameters = {row["parameter"] for row in audit_rows}

        self.assertEqual(audited_parameters, set(payload["line_source_inputs"]) - {"source_status"})
        for row in audit_rows:
            self.assertIn(row["confidence"], {"high", "medium", "low"})
            self.assertTrue(row["source"])


if __name__ == "__main__":
    unittest.main()
