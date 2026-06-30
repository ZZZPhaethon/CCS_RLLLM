"""Fixed-horizon MILP benchmarks for CCS shipping and injection.

This is a perfect-information benchmark for "store as much CO2 as possible in a
given horizon", against which online controllers can be compared. It is an
idealized upper bound: the terminal is modelled as a fluid buffer and unloads as
scheduled deliveries, so the schedule is optimistic and not necessarily
executable step-for-step in the simulator.

Binding constraints captured (all derived from the same network the env uses):
- injection capacity (pipeline / manifold / wells), the sustained-throughput cap;
- single-berth unloading (one delivery per unload duration);
- per-vessel round-trip cadence and first-arrival startup delay;
- terminal buffer capacity;
- aggregate emitter capture inflow under flexible vessel-to-emitter assignment.

Solved with PuLP + CBC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..economics import EconomicParameters
from ..entities.emitter import Emitter
from ..entities.manifold import SubseaManifold
from ..entities.pipeline import Pipeline
from ..entities.storage import InjectionWell
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel
from ..scenario_generation import Scenario

KNOTS_TO_KMH = 1.852


@dataclass
class VesselParams:
    vessel_id: str
    source_id: str
    capacity_t: float
    load_dur_h: int
    unload_dur_h: int
    unloading_rate_tph: float
    sail_h: int

    @property
    def round_trip_h(self) -> int:
        return self.load_dur_h + 2 * self.sail_h + self.unload_dur_h

    @property
    def startup_h(self) -> int:
        return self.load_dur_h + self.sail_h


@dataclass
class FixedHorizonMilpResult:
    status: str
    horizon_h: int
    stored_t: float
    deliveries: int
    operating_cost: float
    cost_per_stored_t: float
    schedule: dict[str, list[int]]


def extract_params(env) -> tuple[list[VesselParams], float, float, float]:
    """Pull MILP parameters from a CCSEnv (injection cap, capture rate, term cap)."""
    network = env.network

    pipelines = [e.max_flow_tph for e in network._entities_of_type(Pipeline).values()]
    manifolds = [e.max_flow_tph for e in network._entities_of_type(SubseaManifold).values()]
    well_sum = sum(e.max_injection_tph for e in network._entities_of_type(InjectionWell).values())
    inj_cap = min([well_sum] + pipelines + manifolds)

    capture_rate = sum(e.nominal_capture_tph for e in network._entities_of_type(Emitter).values())
    term_cap = sum(e.storage_capacity_t for e in network._entities_of_type(Terminal).values())

    vessels: list[VesselParams] = []
    for vid, route in env._routes.items():
        vessel = network.entities[vid]
        assert isinstance(vessel, Vessel)
        sail_h = math.ceil(route["distance_km"] / (route["speed_knots"] * KNOTS_TO_KMH))
        vessels.append(
            VesselParams(
                vessel_id=vid,
                source_id=str(route["origin"]),
                capacity_t=vessel.capacity_t,
                load_dur_h=math.ceil(vessel.capacity_t / vessel.loading_rate_tph),
                unload_dur_h=math.ceil(vessel.capacity_t / vessel.unloading_rate_tph),
                unloading_rate_tph=vessel.unloading_rate_tph,
                sail_h=sail_h,
            )
        )
    return vessels, inj_cap, capture_rate, term_cap


def solve_max_storage_fixed_horizon(
    env,
    horizon_h: int,
    economics: EconomicParameters | None = None,
    initial_buffer_t: float = 0.0,
    time_limit_s: float | None = None,
    mip_gap_rel: float | None = None,
    mip_gap_abs: float | None = None,
    scenario: Scenario | None = None,
    msg: bool = False,
) -> FixedHorizonMilpResult:
    """Maximum safely stored CO2 over a fixed nominal horizon."""
    import pulp

    params = economics or EconomicParameters()
    if scenario is not None:
        return _solve_max_storage_fixed_horizon_with_scenario(
            env,
            horizon_h=int(horizon_h),
            scenario=scenario,
            params=params,
            time_limit_s=time_limit_s,
            mip_gap_rel=mip_gap_rel,
            mip_gap_abs=mip_gap_abs,
            msg=msg,
        )

    vessels, inj_cap, capture_rate, term_cap = extract_params(env)
    H = int(horizon_h)
    hours = range(H)

    prob = pulp.LpProblem("max_storage_fixed_horizon", pulp.LpMaximize)
    d = {
        (v.vessel_id, t): pulp.LpVariable(f"d_{v.vessel_id}_{t}", cat="Binary")
        for v in vessels
        for t in hours
    }
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=inj_cap) for t in hours}

    for v in vessels:
        for t in hours:
            if t < v.startup_h:
                prob += d[(v.vessel_id, t)] == 0

        for t in hours:
            window = [d[(v.vessel_id, k)] for k in range(t, min(t + v.round_trip_h, H))]
            prob += pulp.lpSum(window) <= 1

    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        window = [d[(v.vessel_id, k)] for v in vessels for k in range(t, min(t + unload_dur, H))]
        prob += pulp.lpSum(window) <= 1

    for t in hours:
        cum_deliv = pulp.lpSum(v.capacity_t * d[(v.vessel_id, k)] for v in vessels for k in range(t + 1))
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        prob += cum_inj <= cum_deliv
        prob += cum_deliv - cum_inj <= term_cap
        prob += cum_deliv <= initial_buffer_t + capture_rate * (t + 1)

    prob += pulp.lpSum(inj.values()) - 1e-4 * pulp.lpSum(d.values())
    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=1 if msg else 0,
            timeLimit=time_limit_s,
            gapRel=mip_gap_rel,
            gapAbs=mip_gap_abs,
        )
    )

    status = pulp.LpStatus[prob.status]
    schedule = {
        v.vessel_id: [t for t in hours if round(d[(v.vessel_id, t)].value() or 0) == 1]
        for v in vessels
    }
    n_deliveries = sum(len(s) for s in schedule.values())
    stored_t = float(sum(inj[t].value() or 0.0 for t in hours))
    cost = _schedule_cost(vessels, schedule, float(H), stored_t, params)
    return FixedHorizonMilpResult(
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        deliveries=n_deliveries,
        operating_cost=cost,
        cost_per_stored_t=cost / stored_t if stored_t > 0 else float("nan"),
        schedule=schedule,
    )


def _solve_max_storage_fixed_horizon_with_scenario(
    env,
    *,
    horizon_h: int,
    scenario: Scenario,
    params: EconomicParameters,
    time_limit_s: float | None,
    mip_gap_rel: float | None,
    mip_gap_abs: float | None,
    msg: bool,
) -> FixedHorizonMilpResult:
    """Perfect-foresight fixed-horizon MILP for one sampled disturbance scenario."""
    import pulp

    vessels, nominal_inj_cap, _capture_rate, term_cap = extract_params(env)
    H = int(horizon_h)
    hours = range(H)
    if H <= 0:
        return FixedHorizonMilpResult(
            status="Empty horizon",
            horizon_h=H,
            stored_t=0.0,
            deliveries=0,
            operating_cost=0.0,
            cost_per_stored_t=float("nan"),
            schedule={v.vessel_id: [] for v in vessels},
        )

    initial_inventory = scenario.initial_inventory_t
    term_init_t = sum(float(initial_inventory.get(tid, 0.0)) for tid in env.terminal_ids)
    source_initial_t = {eid: float(initial_inventory.get(eid, 0.0)) for eid in env.emitter_ids}
    source_supply = {
        eid: _cumulative_capture_supply(env, scenario, eid, H, source_initial_t[eid])
        for eid in env.emitter_ids
    }
    total_supply = [
        sum(source_supply[eid][t] for eid in env.emitter_ids)
        for t in hours
    ]
    inj_cap_by_hour = [_scenario_injection_cap(env, scenario, t, nominal_inj_cap) for t in hours]
    berth_count_by_hour = [_scenario_berth_count(env, scenario, t) for t in hours]

    departures = _scenario_departure_options(env, vessels, scenario, H)

    prob = pulp.LpProblem("max_storage_fixed_horizon_scenario", pulp.LpMaximize)
    d = {
        key: pulp.LpVariable(f"depart_{key[0]}_{key[1]}", cat="Binary")
        for key in departures
    }
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=nominal_inj_cap) for t in hours}

    for t in hours:
        prob += inj[t] <= inj_cap_by_hour[t]

    for v in vessels:
        vessel_keys = [(vid, depart_t) for (vid, depart_t) in departures if vid == v.vessel_id]
        for left_index, left_key in enumerate(vessel_keys):
            left_ready = departures[left_key]["next_depart_h"]
            for right_key in vessel_keys[left_index + 1:]:
                if int(right_key[1]) < left_ready:
                    prob += d[left_key] + d[right_key] <= 1
                else:
                    break

    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        active_unloads = [
            d[key]
            for key, option in departures.items()
            if option["arrival_h"] <= t < option["arrival_h"] + unload_dur
        ]
        if active_unloads:
            prob += pulp.lpSum(active_unloads) <= berth_count_by_hour[t]

    by_id = {v.vessel_id: v for v in vessels}
    unload_profiles = {v.vessel_id: _unload_profile(v) for v in vessels}
    for t in hours:
        unloaded_by_t = [
            _cumulative_unloaded_at(unload_profiles[key[0]], t - option["arrival_h"]) * d[key]
            for key, option in departures.items()
            if t >= option["arrival_h"]
        ]
        loaded_by_t = [
            by_id[key[0]].capacity_t * d[key]
            for key in departures
            if key[1] <= t
        ]
        cum_unload = pulp.lpSum(unloaded_by_t)
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        prob += cum_inj <= term_init_t + cum_unload
        prob += term_init_t + cum_unload - cum_inj <= term_cap
        prob += pulp.lpSum(loaded_by_t) <= total_supply[t]

    prob += pulp.lpSum(inj.values()) - 1e-4 * pulp.lpSum(d.values())
    prob.solve(
        pulp.PULP_CBC_CMD(
            msg=1 if msg else 0,
            timeLimit=time_limit_s,
            gapRel=mip_gap_rel,
            gapAbs=mip_gap_abs,
        )
    )

    status = pulp.LpStatus[prob.status]
    schedule = {v.vessel_id: [] for v in vessels}
    for key, option in departures.items():
        if round(d[key].value() or 0) == 1:
            schedule[key[0]].append(int(option["arrival_h"]))
    for times in schedule.values():
        times.sort()
    stored_t = float(sum(inj[t].value() or 0.0 for t in hours))
    n_deliveries = sum(len(times) for times in schedule.values())
    cost = _schedule_cost(vessels, schedule, float(H), stored_t, params)
    return FixedHorizonMilpResult(
        status=status,
        horizon_h=H,
        stored_t=stored_t,
        deliveries=n_deliveries,
        operating_cost=cost,
        cost_per_stored_t=cost / stored_t if stored_t > 0 else float("nan"),
        schedule=schedule,
    )


def _cumulative_capture_supply(
    env,
    scenario: Scenario,
    emitter_id: str,
    horizon_h: int,
    initial_t: float,
) -> list[float]:
    emitter = env.network.entities[emitter_id]
    assert isinstance(emitter, Emitter)
    cumulative = initial_t
    out: list[float] = []
    for t in range(horizon_h):
        availability = _scenario_series_value(scenario.emitter_availability, emitter_id, t, emitter.availability)
        cumulative += emitter.capture_rate_tph_at(float(t)) * max(0.0, float(availability))
        out.append(cumulative)
    return out


def _scenario_injection_cap(env, scenario: Scenario, t: int, nominal_inj_cap: float) -> float:
    well_sum = 0.0
    for well_id, well in env.network._entities_of_type(InjectionWell).items():
        available = bool(_scenario_series_value(scenario.well_available, well_id, t, well.available))
        if not available:
            continue
        injectivity = max(0.0, float(_scenario_series_value(scenario.injectivity_factor, well_id, t, 1.0)))
        well_sum += well.max_injection_tph * injectivity
    return min(nominal_inj_cap, well_sum)


def _scenario_berth_count(env, scenario: Scenario, t: int) -> int:
    return sum(
        max(0, int(terminal.berth_count))
        for terminal in env.network._entities_of_type(Terminal).values()
    )


def _scenario_departure_options(
    env,
    vessels: list[VesselParams],
    scenario: Scenario,
    horizon_h: int,
) -> dict[tuple[str, int], dict[str, int]]:
    options: dict[tuple[str, int], dict[str, int]] = {}
    for v in vessels:
        earliest_depart = v.load_dur_h
        for depart_h in range(earliest_depart, horizon_h):
            arrival_h = _arrival_hour(env, scenario, v.vessel_id, depart_h, horizon_h)
            if arrival_h is None or arrival_h >= horizon_h:
                continue
            return_depart_h = arrival_h + v.unload_dur_h
            home_h = _arrival_hour(env, scenario, v.vessel_id, return_depart_h, horizon_h)
            next_depart_h = horizon_h if home_h is None else home_h + v.load_dur_h
            options[(v.vessel_id, depart_h)] = {
                "arrival_h": arrival_h,
                "next_depart_h": next_depart_h,
            }
    return options


def _unload_profile(vessel: VesselParams) -> list[float]:
    profile = []
    remaining = vessel.capacity_t
    for _ in range(vessel.unload_dur_h):
        amount = min(vessel.unloading_rate_tph, remaining)
        profile.append(amount)
        remaining -= amount
    return profile


def _cumulative_unloaded_at(profile: list[float], offset_h: int) -> float:
    if offset_h < 0:
        return 0.0
    return sum(profile[: min(offset_h + 1, len(profile))])


def _arrival_hour(env, scenario: Scenario, vessel_id: str, depart_h: int, horizon_h: int) -> int | None:
    route = env._routes[vessel_id]
    speed_knots = float(route["speed_knots"])
    distance_km = float(route["distance_km"])
    remaining_km = distance_km
    for t in range(max(0, depart_h), horizon_h):
        factor = max(0.0, float(_scenario_series_value(scenario.vessel_speed_factor, vessel_id, t, 1.0)))
        remaining_km -= speed_knots * KNOTS_TO_KMH * factor
        if remaining_km <= 1e-9:
            return t + 1
    return None


def _scenario_series_value(series_by_id, entity_id: str, t: int, default):
    series = series_by_id.get(entity_id)
    if not series:
        return default
    index = max(0, min(len(series) - 1, t))
    return series[index]


def _schedule_cost(
    vessels: list[VesselParams],
    schedule: dict[str, list[int]],
    horizon_h: float,
    stored_t: float,
    params: EconomicParameters,
) -> float:
    charter = len(vessels) * params.vessel_charter_eur_per_h * horizon_h
    sail_hours = sum(2 * v.sail_h * len(schedule[v.vessel_id]) for v in vessels)
    fuel = sail_hours * params.vessel_fuel_eur_per_h_sailing
    handled_t = sum(2 * v.capacity_t * len(schedule[v.vessel_id]) for v in vessels)
    handling = handled_t * params.handling_eur_per_t
    injection = stored_t * params.injection_cost_eur_per_t
    return charter + fuel + handling + injection
