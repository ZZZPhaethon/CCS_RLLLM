"""Operational cost model for the ship-based CCS network.

The research treats the infrastructure as fixed (capital is sunk), so this module
prices only *operational* flows: vessel charter and fuel while sailing, cargo
handling, injection energy, geological storage opex, the economic loss from
venting, and the revenue from permanently stored CO2. It is deliberately
decoupled from the physics: it reads a :class:`StepResult` (plus the network for
entity typing) and returns a per-step economic breakdown, so the same model
feeds both the RL reward (section 8 of the research note) and the evaluation
KPIs (section 13).

Default rates are calibrated to publicly reported 2026 figures for the Northern
Lights value chain; see the project notes for sources. They are intentionally
held in a plain dataclass so a ScenarioGenerator can randomise them per episode
(fuel-price / carbon-price uncertainty, section 6.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .entities.emitter import Emitter
from .entities.state import PhysicalState, StepResult
from .entities.storage import InjectionWell
from .entities.terminal import Terminal
from .entities.vessel import Vessel


@dataclass(frozen=True)
class EconomicParameters:
    """Operational cost/revenue rates in EUR (capex excluded).

    Defaults reflect 2026 Northern Lights-scale figures:
    - EU ETS price ~EUR 60-95/t (used as the venting loss basis);
    - Northern Lights T&S tariff ~EUR 35-50/t (storage revenue);
    - ~7,500 m3 LCO2 carrier charter/fuel order of magnitude;
    - CO2 compression ~100-200 kWh/t at ~EUR 0.06/kWh Norwegian power;
    - offshore storage technical opex ~EUR 2-20/t.
    """

    currency: str = "EUR"

    # Carbon value and storage revenue.
    ets_price_eur_per_t: float = 75.0
    storage_tariff_eur_per_t: float = 40.0

    # Vessels (time-based opex).
    vessel_charter_eur_per_h: float = 800.0
    vessel_fuel_eur_per_h_sailing: float = 600.0

    # Cargo handling. EUR 5,000 per port call over a ~7,500 t parcel each way is
    # ~EUR 0.7/t handled; modelled per-tonne so the step cost is well defined.
    handling_eur_per_t: float = 0.7

    # Injection energy and storage.
    compression_kwh_per_t: float = 120.0
    electricity_eur_per_kwh: float = 0.06
    storage_opex_eur_per_t: float = 5.0

    # Penalties (constraints priced above pure cost, section 8).
    vent_penalty_eur_per_t: float = 75.0
    # Annual storage-target shortfall: only meaningful over a long (>= multi-month)
    # horizon, where in-transit CO2 is negligible. Used by storage_shortfall_penalty.
    storage_shortfall_eur_per_t: float = 100.0
    # Backlog (in-transit, captured-but-not-yet-stored) growth price. This is the
    # horizon-appropriate short-episode signal: it penalizes the fleet/injection
    # falling behind capture, without mislabelling recoverable in-transit CO2 as a
    # contractual miss. Modest because backlog is recoverable, unlike vented CO2.
    backlog_penalty_eur_per_t: float = 20.0

    @property
    def injection_cost_eur_per_t(self) -> float:
        """All-in cost to inject and store one tonne (energy + storage opex)."""
        return self.compression_kwh_per_t * self.electricity_eur_per_kwh + self.storage_opex_eur_per_t

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["injection_cost_eur_per_t"] = self.injection_cost_eur_per_t
        return data


@dataclass
class StepEconomics:
    """Per-step economic breakdown in EUR (costs positive, revenue positive)."""

    vessel_charter: float = 0.0
    vessel_fuel: float = 0.0
    handling: float = 0.0
    injection: float = 0.0
    vent_penalty: float = 0.0
    revenue_storage: float = 0.0

    # Tonnage bookkeeping for downstream KPIs.
    stored_t: float = 0.0
    vented_t: float = 0.0
    handled_t: float = 0.0

    @property
    def operating_cost(self) -> float:
        """Cash operating cost excluding penalties and revenue."""
        return self.vessel_charter + self.vessel_fuel + self.handling + self.injection

    @property
    def total_cost(self) -> float:
        """Operating cost plus penalties."""
        return self.operating_cost + self.vent_penalty

    @property
    def net(self) -> float:
        """Revenue minus all costs and penalties (the economic reward term)."""
        return self.revenue_storage - self.total_cost

    def as_dict(self) -> dict[str, object]:
        return {
            "vessel_charter": self.vessel_charter,
            "vessel_fuel": self.vessel_fuel,
            "handling": self.handling,
            "injection": self.injection,
            "vent_penalty": self.vent_penalty,
            "revenue_storage": self.revenue_storage,
            "operating_cost": self.operating_cost,
            "total_cost": self.total_cost,
            "net": self.net,
            "stored_t": self.stored_t,
            "vented_t": self.vented_t,
            "handled_t": self.handled_t,
        }


class CostModel:
    """Prices a single simulation step from its :class:`StepResult`."""

    def __init__(self, parameters: EconomicParameters | None = None) -> None:
        self.parameters = parameters or EconomicParameters()

    def evaluate_step(self, network, step_result: StepResult) -> StepEconomics:
        params = self.parameters
        state = step_result.state
        hours = network.time_step_hours

        vessel_ids = list(network._entities_of_type(Vessel))
        sailing_count = sum(1 for vid in vessel_ids if vid not in state.vessel_berths)

        stored_t = sum(state.last_injection_flow_tph.values()) * hours
        vented_t = sum(state.last_vent_tph.values()) * hours
        handled_t = self._handled_tonnes(network, step_result.flows_t)

        return StepEconomics(
            vessel_charter=len(vessel_ids) * params.vessel_charter_eur_per_h * hours,
            vessel_fuel=sailing_count * params.vessel_fuel_eur_per_h_sailing * hours,
            handling=handled_t * params.handling_eur_per_t,
            injection=stored_t * params.injection_cost_eur_per_t,
            vent_penalty=vented_t * params.vent_penalty_eur_per_t,
            revenue_storage=stored_t * params.storage_tariff_eur_per_t,
            stored_t=stored_t,
            vented_t=vented_t,
            handled_t=handled_t,
        )

    def storage_shortfall_penalty(
        self,
        cumulative_captured_t: float,
        cumulative_stored_t: float,
        target_rate: float,
    ) -> float:
        """Penalty for falling below a required storage-rate obligation.

        ``target_rate`` is the contracted fraction of captured CO2 that must be
        safely stored (e.g. 0.9). The shortfall is the tonnage gap to that
        obligation, priced above the venting penalty so the constraint dominates
        pure cost minimisation.
        """
        required_t = max(0.0, target_rate) * max(0.0, cumulative_captured_t)
        shortfall_t = max(0.0, required_t - cumulative_stored_t)
        return shortfall_t * self.parameters.storage_shortfall_eur_per_t

    def _handled_tonnes(self, network, flows_t: dict[tuple[str, str], float]) -> float:
        handled_t = 0.0
        for (source_id, target_id), amount_t in flows_t.items():
            source = network.entities.get(source_id)
            target = network.entities.get(target_id)
            is_loading = isinstance(source, Emitter) and isinstance(target, Vessel)
            is_unloading = isinstance(source, Vessel) and isinstance(target, Terminal)
            if is_loading or is_unloading:
                handled_t += amount_t
        return handled_t


@dataclass
class EconomicLedger:
    """Accumulates per-step economics across an episode for reporting."""

    vessel_charter: float = 0.0
    vessel_fuel: float = 0.0
    handling: float = 0.0
    injection: float = 0.0
    vent_penalty: float = 0.0
    revenue_storage: float = 0.0
    storage_shortfall_penalty: float = 0.0
    stored_t: float = 0.0
    vented_t: float = 0.0
    handled_t: float = 0.0

    def add(self, step: StepEconomics) -> None:
        self.vessel_charter += step.vessel_charter
        self.vessel_fuel += step.vessel_fuel
        self.handling += step.handling
        self.injection += step.injection
        self.vent_penalty += step.vent_penalty
        self.revenue_storage += step.revenue_storage
        self.stored_t += step.stored_t
        self.vented_t += step.vented_t
        self.handled_t += step.handled_t

    @property
    def operating_cost(self) -> float:
        return self.vessel_charter + self.vessel_fuel + self.handling + self.injection

    @property
    def total_cost(self) -> float:
        return self.operating_cost + self.vent_penalty + self.storage_shortfall_penalty

    @property
    def net(self) -> float:
        return self.revenue_storage - self.total_cost

    def as_dict(self) -> dict[str, object]:
        return {
            "vessel_charter": self.vessel_charter,
            "vessel_fuel": self.vessel_fuel,
            "handling": self.handling,
            "injection": self.injection,
            "vent_penalty": self.vent_penalty,
            "storage_shortfall_penalty": self.storage_shortfall_penalty,
            "revenue_storage": self.revenue_storage,
            "operating_cost": self.operating_cost,
            "total_cost": self.total_cost,
            "net": self.net,
            "stored_t": self.stored_t,
            "vented_t": self.vented_t,
            "handled_t": self.handled_t,
        }
