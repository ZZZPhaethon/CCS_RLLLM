import unittest

from sim.ship_speed import BJERKETVEDT_2020_SHIPS, NORTHERN_LIGHTS_SHIP, speed_knots


class ShipSpeedTests(unittest.TestCase):
    def test_bjerketvedt_ship_returns_design_speed_in_calm_water(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        self.assertAlmostEqual(speed_knots(0.0, vessel), 12.0, places=6)

    def test_bjerketvedt_ship_slows_as_significant_wave_height_increases(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        calm = speed_knots(0.0, vessel)
        moderate = speed_knots(4.0, vessel)
        storm = speed_knots(8.0, vessel)

        self.assertGreater(calm, moderate)
        self.assertGreater(moderate, storm)
        self.assertAlmostEqual(moderate, 9.97, delta=0.05)
        self.assertAlmostEqual(storm, 6.62, delta=0.05)

    def test_northern_lights_ship_uses_reported_design_parameters(self):
        self.assertEqual(NORTHERN_LIGHTS_SHIP.power_kw, 5500.0)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.beam_m, 21.2)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.waterline_length_m, 130.0)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.design_speed_knots, 14.0)
        self.assertAlmostEqual(speed_knots(0.0, NORTHERN_LIGHTS_SHIP), 14.0, places=6)


if __name__ == "__main__":
    unittest.main()
