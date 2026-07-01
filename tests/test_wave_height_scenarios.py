import unittest

from sim.entities import Emitter, InjectionWell, PhysicalState, Terminal, Vessel
from sim.network import PhysicalNetwork
from sim.scenario_generation import ScenarioConfig
from sim.scenario_generation.wave_height import (
    WaveHeightScenarioGenerator,
    aggregate_wave_heights,
    densify_route,
)
from sim.ship_speed import BJERKETVEDT_2020_SHIPS, speed_factor_series


class FakeWaveReader:
    def __init__(self, total_records=20):
        self.total_records = total_records
        self.calls = []

    def route_wave_height_series(self, route_coordinates, *, start_record=0, hours=None):
        self.calls.append((tuple(route_coordinates), start_record, hours))
        base = [0.0, 2.0, 4.0, 6.0, 3.0, 1.0]
        return [base[(start_record + i) % len(base)] for i in range(hours)]


def _network() -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("source", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
    network.add_entity(Vessel("ship", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Terminal("terminal", storage_capacity_t=2_000.0, berth_count=1))
    network.add_entity(InjectionWell("well", max_injection_tph=200.0))
    network.connect("source", "ship")
    network.connect("ship", "terminal")
    return network


def _quiet_config() -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=3,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
        randomize_initial_inventory=False,
    )


class RouteWaveHelpersTests(unittest.TestCase):
    def test_aggregate_wave_heights_supports_mean_max_and_percentile(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(aggregate_wave_heights(values, "mean"), 2.5)
        self.assertEqual(aggregate_wave_heights(values, "max"), 4.0)
        self.assertAlmostEqual(aggregate_wave_heights(values, "p75"), 3.25)

    def test_densify_route_adds_intermediate_points(self):
        route = densify_route([(0.0, 0.0), (0.0, 1.0)], spacing_km=25.0)

        self.assertEqual(route[0], (0.0, 0.0))
        self.assertEqual(route[-1], (0.0, 1.0))
        self.assertGreater(len(route), 2)


class WaveHeightScenarioGeneratorTests(unittest.TestCase):
    def test_wave_height_generator_replaces_vessel_speed_factor(self):
        reader = FakeWaveReader()
        routes = {
            "ship": {
                "coordinates": [(0.0, 0.0), (0.0, 1.0)],
                "speed_knots": 12.0,
            }
        }
        parameters = BJERKETVEDT_2020_SHIPS[5000]
        generator = WaveHeightScenarioGenerator(
            routes=routes,
            reader=reader,
            default_ship_parameters=parameters,
            config=_quiet_config(),
            seed=1,
        )

        scenario = generator.sample(_network(), seed=2)

        self.assertEqual(set(scenario.vessel_speed_factor), {"ship"})
        self.assertEqual(len(scenario.vessel_speed_factor["ship"]), 3)
        wave_heights = reader.route_wave_height_series(
            routes["ship"]["coordinates"],
            start_record=generator.last_start_record,
            hours=3,
        )
        expected = speed_factor_series(wave_heights, parameters, nominal_speed_knots=12.0)
        self.assertEqual(scenario.vessel_speed_factor["ship"], expected)

    def test_wave_height_scenario_drives_state_at_apply_time(self):
        reader = FakeWaveReader(total_records=3)
        generator = WaveHeightScenarioGenerator(
            routes={"ship": {"coordinates": [(0.0, 0.0), (0.0, 1.0)], "speed_knots": 12.0}},
            reader=reader,
            default_ship_parameters=BJERKETVEDT_2020_SHIPS[5000],
            config=_quiet_config(),
            seed=1,
        )
        scenario = generator.sample(_network(), seed=2)
        state = PhysicalState()

        scenario.apply_to_state(state, time_h=1.0)

        self.assertEqual(state.vessel_speed_factor["ship"], scenario.vessel_speed_factor["ship"][1])

    def test_same_seed_samples_same_weather_window(self):
        routes = {"ship": {"coordinates": [(0.0, 0.0), (0.0, 1.0)], "speed_knots": 12.0}}
        a = WaveHeightScenarioGenerator(routes=routes, reader=FakeWaveReader(), config=_quiet_config())
        b = WaveHeightScenarioGenerator(routes=routes, reader=FakeWaveReader(), config=_quiet_config())

        scenario_a = a.sample(_network(), seed=123)
        scenario_b = b.sample(_network(), seed=123)

        self.assertEqual(a.last_start_record, b.last_start_record)
        self.assertEqual(scenario_a.vessel_speed_factor, scenario_b.vessel_speed_factor)


if __name__ == "__main__":
    unittest.main()
