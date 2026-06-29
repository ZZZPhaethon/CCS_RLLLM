"""MILP benchmark: minimum time (and cost) to store a target amount of CO2.

This is the perfect-information, nominal-conditions optimum the research note
(section 10) uses as a benchmark - the *best achievable* makespan for the
iso-storage task "store T tonnes", against which heuristic and RL policies are
measured. It is an idealized lower bound: the terminal is modelled as a fluid
buffer and unloads as instantaneous deliveries, so the schedule is optimistic
and not necessarily executable step-for-step in the simulator.

Binding constraints captured (all derived from the same network the env uses):
- injection capacity (pipeline / manifold / wells), the sustained-throughput cap;
- single-berth unloading (one delivery per unload duration);
- per-vessel round-trip cadence and first-arrival startup delay;
- terminal buffer capacity;
- capture inflow (cannot deliver CO2 faster than it is captured).

Solved with PuLP + CBC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..physical.economics import EconomicParameters
from ..physical.entities.emitter import Emitter
from ..physical.entities.manifold import SubseaManifold
from ..physical.entities.pipeline import Pipeline
from ..physical.entities.storage import InjectionWell
from ..physical.entities.terminal import Terminal
from ..physical.entities.vessel import Vessel

KNOTS_TO_KMH = 1.852


@dataclass
class VesselParams:
    vessel_id: str
    capacity_t: float
    load_dur_h: int
    unload_dur_h: int
    sail_h: int

    @property
    def round_trip_h(self) -> int:
        return self.load_dur_h + 2 * self.sail_h + self.unload_dur_h

    @property
    def startup_h(self) -> int:
        return self.load_dur_h + self.sail_h


@dataclass
class MilpResult:
    status: str
    reached: bool
    makespan_h: float
    deliveries: int
    operating_cost: float
    cost_per_stored_t: float
    target_t: float
    schedule: dict[str, list[int]]  # vessel_id -> delivery completion hours


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
                capacity_t=vessel.capacity_t,
                load_dur_h=math.ceil(vessel.capacity_t / vessel.loading_rate_tph),
                unload_dur_h=math.ceil(vessel.capacity_t / vessel.unloading_rate_tph),
                sail_h=sail_h,
            )
        )
    return vessels, inj_cap, capture_rate, term_cap


def solve_min_makespan(
    env,
    target_t: float,
    horizon_h: int | None = None,
    economics: EconomicParameters | None = None,
    initial_buffer_t: float = 0.0,
    msg: bool = False,
) -> MilpResult:
    """Minimum makespan (and its cost) to safely store ``target_t`` tonnes."""
    import pulp

    params = economics or EconomicParameters()
    vessels, inj_cap, capture_rate, term_cap = extract_params(env)
    cap = vessels[0].capacity_t  # uniform fleet

    if horizon_h is None:
        startup_max = max(v.startup_h for v in vessels)
        fleet_rate = sum(v.capacity_t / v.round_trip_h for v in vessels)
        throughput_hours = max(target_t / inj_cap, target_t / capture_rate, target_t / fleet_rate)
        horizon_h = int(startup_max + math.ceil(throughput_hours) + max(v.round_trip_h for v in vessels) + 30)
    H = horizon_h
    hours = range(H)

    prob = pulp.LpProblem("min_makespan_to_target", pulp.LpMinimize)
    d = {
        (v.vessel_id, t): pulp.LpVariable(f"d_{v.vessel_id}_{t}", cat="Binary")
        for v in vessels
        for t in hours
    }
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=inj_cap) for t in hours}
    done = {t: pulp.LpVariable(f"done_{t}", cat="Binary") for t in hours}

    # No delivery before a vessel could physically complete its first trip.
    for v in vessels:
        for t in hours:
            if t < v.startup_h:
                prob += d[(v.vessel_id, t)] == 0

    # Per-vessel round-trip cadence: at most one delivery per round_trip window.
    for v in vessels:
        for t in hours:
            window = [d[(v.vessel_id, k)] for k in range(t, min(t + v.round_trip_h, H))]
            prob += pulp.lpSum(window) <= 1

    # Single berth: at most one delivery (across all vessels) per unload duration.
    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        window = [d[(v.vessel_id, k)] for v in vessels for k in range(t, min(t + unload_dur, H))]
        prob += pulp.lpSum(window) <= 1

    # Cumulative balances.
    for t in hours:
        cum_deliv = cap * pulp.lpSum(d[(v.vessel_id, k)] for v in vessels for k in range(t + 1))
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        prob += cum_inj <= cum_deliv                                  # inject only what arrived
        prob += cum_deliv - cum_inj <= term_cap                       # terminal buffer limit
        prob += cum_deliv <= initial_buffer_t + capture_rate * (t + 1)  # capture inflow limit
        prob += cum_inj >= target_t * done[t]                          # target reached flag
        if t > 0:
            prob += done[t] >= done[t - 1]                            # done stays done

    # Maximize hours spent "done" == minimize makespan; tiny tie-break for fewer trips.
    prob += (H - pulp.lpSum(done[t] for t in hours)) + 1e-4 * pulp.lpSum(d.values())
    prob.solve(pulp.PULP_CBC_CMD(msg=1 if msg else 0))

    status = pulp.LpStatus[prob.status]
    done_count = sum(round(done[t].value() or 0) for t in hours)
    reached = done_count > 0
    makespan = float(H - done_count) if reached else float(H)
    schedule = {
        v.vessel_id: [t for t in hours if round(d[(v.vessel_id, t)].value() or 0) == 1]
        for v in vessels
    }
    n_deliveries = sum(len(s) for s in schedule.values())

    cost = _schedule_cost(vessels, schedule, makespan, target_t, params)
    return MilpResult(
        status=status,
        reached=reached,
        makespan_h=makespan,
        deliveries=n_deliveries,
        operating_cost=cost,
        cost_per_stored_t=cost / target_t if target_t > 0 else float("nan"),
        target_t=target_t,
        schedule=schedule,
    )


def solve_max_storage_fixed_horizon(
    env,
    horizon_h: int,
    economics: EconomicParameters | None = None,
    initial_buffer_t: float = 0.0,
    time_limit_s: float | None = None,
    msg: bool = False,
) -> FixedHorizonMilpResult:
    """Maximum safely stored CO2 over a fixed nominal horizon.

    This is the fixed-horizon companion to :func:`solve_min_makespan`: instead
    of asking "how fast can we store T tonnes?", it asks "within H hours, what is
    the best nominal perfect-information storage throughput?".
    """
    import pulp

    params = economics or EconomicParameters()
    vessels, inj_cap, capture_rate, term_cap = extract_params(env)
    cap = vessels[0].capacity_t  # uniform fleet
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

    for v in vessels:
        for t in hours:
            window = [d[(v.vessel_id, k)] for k in range(t, min(t + v.round_trip_h, H))]
            prob += pulp.lpSum(window) <= 1

    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:
        window = [d[(v.vessel_id, k)] for v in vessels for k in range(t, min(t + unload_dur, H))]
        prob += pulp.lpSum(window) <= 1

    for t in hours:
        cum_deliv = cap * pulp.lpSum(d[(v.vessel_id, k)] for v in vessels for k in range(t + 1))
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        prob += cum_inj <= cum_deliv
        prob += cum_deliv - cum_inj <= term_cap
        prob += cum_deliv <= initial_buffer_t + capture_rate * (t + 1)

    # Maximize stored tonnes; tiny tie-break discourages unnecessary deliveries.
    prob += pulp.lpSum(inj.values()) - 1e-4 * pulp.lpSum(d.values())
    prob.solve(pulp.PULP_CBC_CMD(msg=1 if msg else 0, timeLimit=time_limit_s))

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


def _schedule_cost(
    vessels: list[VesselParams],
    schedule: dict[str, list[int]],
    makespan_h: float,
    target_t: float,
    params: EconomicParameters,
) -> float:
    charter = len(vessels) * params.vessel_charter_eur_per_h * makespan_h
    sail_hours = sum(2 * v.sail_h * len(schedule[v.vessel_id]) for v in vessels)
    fuel = sail_hours * params.vessel_fuel_eur_per_h_sailing
    handled_t = sum(2 * v.capacity_t * len(schedule[v.vessel_id]) for v in vessels)
    handling = handled_t * params.handling_eur_per_t
    injection = target_t * params.injection_cost_eur_per_t
    return charter + fuel + handling + injection
