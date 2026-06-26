import unittest

from sim.economics import CostModel, EconomicLedger, EconomicParameters, StepEconomics
from sim.entities import (
    Emitter,
    InjectionWell,
    PhysicalState,
    Reservoir,
    Terminal,
    Vessel,
)
from sim.entities.state import StepResult
from sim.network import PhysicalNetwork


def _priced_network() -> PhysicalNetwork:
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("brevik", nominal_capture_tph=100.0, buffer_capacity_t=1_000.0))
    network.add_entity(Vessel("ship_1", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Vessel("ship_2", capacity_t=800.0, loading_rate_tph=800.0, unloading_rate_tph=800.0))
    network.add_entity(Terminal("oygarden", storage_capacity_t=2_000.0, berth_count=1))
    network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
    network.add_entity(
        Reservoir(
            "aurora",
            storage_capacity_t=1_000_000.0,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    return network


class EconomicParameterTests(unittest.TestCase):
    def test_injection_cost_combines_energy_and_storage_opex(self):
        params = EconomicParameters()
        # 120 kWh/t * 0.06 EUR/kWh + 5 EUR/t = 12.2 EUR/t.
        self.assertAlmostEqual(params.injection_cost_eur_per_t, 12.2)

    def test_as_dict_exposes_derived_injection_cost(self):
        self.assertIn("injection_cost_eur_per_t", EconomicParameters().as_dict())


class CostModelStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = _priced_network()
        self.model = CostModel()

    def _step_result(self) -> StepResult:
        # ship_1 berthed at the terminal, ship_2 at sea; one well injecting 100 t/h;
        # the emitter venting 10 t/h; 50 t loaded and 30 t unloaded this hour.
        state = PhysicalState(
            vessel_berths={"ship_1": "oygarden"},
            last_injection_flow_tph={"well_1": 100.0},
            last_vent_tph={"brevik": 10.0},
        )
        flows_t = {
            ("brevik", "ship_1"): 50.0,   # loading (emitter -> vessel)
            ("ship_1", "oygarden"): 30.0,  # unloading (vessel -> terminal)
            ("oygarden", "pipeline"): 100.0,  # not handling, ignored
        }
        return StepResult(state=state, flows_t=flows_t, violations=[], mass_balance_error_t=0.0)

    def test_step_breakdown_matches_calibrated_rates(self):
        econ = self.model.evaluate_step(self.network, self._step_result())

        self.assertAlmostEqual(econ.vessel_charter, 2 * 800.0)   # both vessels in service
        self.assertAlmostEqual(econ.vessel_fuel, 1 * 600.0)      # only ship_2 sailing
        self.assertAlmostEqual(econ.handling, (50.0 + 30.0) * 0.7)
        self.assertAlmostEqual(econ.injection, 100.0 * 12.2)
        self.assertAlmostEqual(econ.vent_penalty, 10.0 * 75.0)
        self.assertAlmostEqual(econ.revenue_storage, 100.0 * 40.0)
        self.assertAlmostEqual(econ.stored_t, 100.0)
        self.assertAlmostEqual(econ.vented_t, 10.0)
        self.assertAlmostEqual(econ.handled_t, 80.0)

    def test_net_is_revenue_minus_costs_and_penalties(self):
        econ = self.model.evaluate_step(self.network, self._step_result())
        expected_operating = 1600.0 + 600.0 + 56.0 + 1220.0
        self.assertAlmostEqual(econ.operating_cost, expected_operating)
        self.assertAlmostEqual(econ.total_cost, expected_operating + 750.0)
        self.assertAlmostEqual(econ.net, 4000.0 - (expected_operating + 750.0))

    def test_time_step_scales_time_based_costs(self):
        network = _priced_network()
        network.time_step_hours = 2.0
        econ = CostModel().evaluate_step(network, self._step_result())
        # Charter, fuel and tonnage (rate * 2 h) all double.
        self.assertAlmostEqual(econ.vessel_charter, 2 * 800.0 * 2.0)
        self.assertAlmostEqual(econ.vessel_fuel, 1 * 600.0 * 2.0)
        self.assertAlmostEqual(econ.stored_t, 200.0)


class ShortfallPenaltyTests(unittest.TestCase):
    def test_no_penalty_when_target_met(self):
        model = CostModel()
        penalty = model.storage_shortfall_penalty(
            cumulative_captured_t=1_000.0,
            cumulative_stored_t=950.0,
            target_rate=0.9,
        )
        self.assertEqual(penalty, 0.0)

    def test_penalty_prices_the_tonnage_gap(self):
        model = CostModel()
        # Need 900 stored, only 800 -> 100 t short * 100 EUR/t.
        penalty = model.storage_shortfall_penalty(
            cumulative_captured_t=1_000.0,
            cumulative_stored_t=800.0,
            target_rate=0.9,
        )
        self.assertAlmostEqual(penalty, 10_000.0)


class EconomicLedgerTests(unittest.TestCase):
    def test_ledger_accumulates_steps(self):
        ledger = EconomicLedger()
        ledger.add(StepEconomics(injection=12.2, revenue_storage=40.0, stored_t=1.0))
        ledger.add(StepEconomics(injection=12.2, revenue_storage=40.0, stored_t=1.0, vent_penalty=75.0, vented_t=1.0))
        ledger.storage_shortfall_penalty = 500.0

        self.assertAlmostEqual(ledger.stored_t, 2.0)
        self.assertAlmostEqual(ledger.vented_t, 1.0)
        self.assertAlmostEqual(ledger.operating_cost, 24.4)
        self.assertAlmostEqual(ledger.total_cost, 24.4 + 75.0 + 500.0)
        self.assertAlmostEqual(ledger.net, 80.0 - (24.4 + 75.0 + 500.0))


if __name__ == "__main__":
    unittest.main()
