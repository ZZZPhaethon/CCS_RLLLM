"""Rolling-horizon MILP controller (MPC) for the iso-storage task.

Unlike the open-loop :mod:`sim.milp` benchmark (one offline solve, perfect
information), this re-plans from the *current* simulator state every few hours,
so it adapts to disturbances - weather delays, well maintenance, capture
fluctuation - by re-optimising against whatever actually happened. It is the
adaptive baseline that tells us whether an RL policy is even needed: if re-planning
MILP stays near the optimum under disturbance and is cheap enough, RL has little
room; if it degrades or is too slow, that is where RL earns its keep.

It is exposed as a metrics ``policy(env) -> action`` so it is scored by the same
iso-storage harness as idle / greedy_shuttle / PPO. Injection is always run flat
out (injection is the throughput bottleneck, never worth throttling), so the only
real decision is *when to dispatch each home vessel* - which the MILP paces to
keep the terminal continuously fed without flooding it.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from ..physical.economics import EconomicParameters
from ..rl.env import VESSEL_GO_HOME, VESSEL_GO_TERMINAL, VESSEL_WAIT, WELL_ACTIONS, CCSEnv
from .milp import KNOTS_TO_KMH, extract_params


def _plan_delivery_times(
    env: CCSEnv,
    remaining_target_t: float,
    vessel_ready_h: dict[str, int],
    term_init_t: float,
    source_buffer_t: float,
    economics: EconomicParameters,
    time_limit_s: float = 5.0,
) -> dict[str, list[int]]:
    """Plan future delivery hours (from now) per vessel to hit the remaining target."""
    import pulp

    vessels, inj_cap, capture_rate, term_cap = extract_params(env)
    by_id = {v.vessel_id: v for v in vessels}
    cap = vessels[0].capacity_t

    startup_max = max(vessel_ready_h.values())
    fleet_rate = sum(v.capacity_t / v.round_trip_h for v in vessels)
    throughput_hours = max(
        max(0.0, remaining_target_t) / inj_cap,
        max(0.0, remaining_target_t - term_init_t - source_buffer_t) / capture_rate,
        max(0.0, remaining_target_t) / fleet_rate,
    )
    H = int(startup_max + math.ceil(throughput_hours) + max(v.round_trip_h for v in vessels) + 20)
    hours = range(H)

    prob = pulp.LpProblem("rolling_plan", pulp.LpMinimize)
    d = {(vid, t): pulp.LpVariable(f"d_{vid}_{t}", cat="Binary") for vid in by_id for t in hours}
    inj = {t: pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=inj_cap) for t in hours}
    done = {t: pulp.LpVariable(f"done_{t}", cat="Binary") for t in hours}

    for vid, v in by_id.items():
        ready = vessel_ready_h[vid]
        for t in hours:
            if t < ready:
                prob += d[(vid, t)] == 0
        for t in hours:  # round-trip cadence
            prob += pulp.lpSum(d[(vid, k)] for k in range(t, min(t + v.round_trip_h, H))) <= 1

    unload_dur = max(v.unload_dur_h for v in vessels)
    for t in hours:  # single berth
        prob += pulp.lpSum(d[(vid, k)] for vid in by_id for k in range(t, min(t + unload_dur, H))) <= 1

    for t in hours:
        cum_deliv = cap * pulp.lpSum(d[(vid, k)] for vid in by_id for k in range(t + 1))
        cum_inj = pulp.lpSum(inj[k] for k in range(t + 1))
        prob += cum_inj <= term_init_t + cum_deliv
        prob += term_init_t + cum_deliv - cum_inj <= term_cap
        prob += cum_deliv <= source_buffer_t + capture_rate * (t + 1)
        prob += cum_inj >= remaining_target_t * done[t]
        if t > 0:
            prob += done[t] >= done[t - 1]

    prob += (H - pulp.lpSum(done.values())) + 1e-4 * pulp.lpSum(d.values())
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s))

    return {
        vid: [t for t in hours if round(d[(vid, t)].value() or 0) == 1]
        for vid in by_id
    }


class RollingMilpController:
    """Re-planning MILP controller, usable as a metrics ``policy(env)``."""

    def __init__(
        self,
        env: CCSEnv,
        replan_every: int = 12,
        economics: EconomicParameters | None = None,
        progress: Callable[[str], None] | None = None,
        plan_target_t: float | None = None,
    ):
        self.replan_every = replan_every
        self.economics = economics or EconomicParameters()
        self.progress = progress
        self.plan_target_t = plan_target_t
        self._fallback_greedy = False
        vessels, *_ = extract_params(env)
        self._vp = {v.vessel_id: v for v in vessels}
        self._sail_h = {
            vid: math.ceil(env._routes[vid]["distance_km"] / (env._routes[vid]["speed_knots"] * KNOTS_TO_KMH))
            for vid in self._vp
        }
        self._plan: dict[str, list[int]] = {}
        self._plan_origin_h: float = -1e9

    def __call__(self, env: CCSEnv) -> list[int]:
        return self.policy(env)

    def policy(self, env: CCSEnv) -> list[int]:
        now = env.simulator.state.time_h
        new_episode = now < self._plan_origin_h  # time went backwards -> env was reset
        if new_episode or now - self._plan_origin_h >= self.replan_every or not self._plan:
            self._replan(env, now)
        if self._fallback_greedy:
            return self._greedy_action(env)

        action: list[int] = []
        for i, vid in enumerate(env.vessel_ids):
            action.append(self._vessel_action(env, vid, now, env.action_mask()[i]))
        action += [WELL_ACTIONS - 1] * len(env.well_ids)  # inject flat out
        return action

    def _replan(self, env: CCSEnv, now: float) -> None:
        goal = env.config.storage_goal_t
        if goal is None:
            remaining = self.plan_target_t if self.plan_target_t is not None and self.plan_target_t > 0.0 else 10_000.0
        else:
            remaining = max(0.0, goal - env.cumulative_stored_t)
        if remaining <= 0.0:
            self._plan = {vid: [] for vid in self._vp}
            self._plan_origin_h = now
            self._fallback_greedy = False
            return
        state = env.simulator.state
        term_init = sum(state.entity_inventory_t.get(t, 0.0) for t in env.terminal_ids)
        source_buffer = sum(state.entity_inventory_t.get(e, 0.0) for e in env.emitter_ids)
        ready = {vid: self._ready_hours(env, vid) for vid in self._vp}
        planning_target = (
            min(remaining, self.plan_target_t)
            if self.plan_target_t is not None and self.plan_target_t > 0.0
            else remaining
        )
        start = time.perf_counter()
        if self.progress is not None:
            self.progress(
                f"  rolling_milp replan at t={now:.0f} h; remaining={remaining:,.1f} t; "
                f"planning_target={planning_target:,.1f} t; "
                f"terminal={term_init:,.1f} t; source_buffer={source_buffer:,.1f} t"
            )
        self._plan = _plan_delivery_times(
            env, planning_target, ready, term_init, source_buffer, self.economics
        )
        self._plan_origin_h = now
        if self.progress is not None:
            planned_deliveries = sum(len(times) for times in self._plan.values())
            self.progress(
                f"  rolling_milp plan ready in {time.perf_counter() - start:.1f}s; "
                f"planned_deliveries={planned_deliveries}"
            )
        planned_deliveries = sum(len(times) for times in self._plan.values())
        self._fallback_greedy = planned_deliveries == 0
        if self._fallback_greedy and self.progress is not None:
            self.progress("  rolling_milp fallback: no MILP deliveries planned; using greedy shuttle until next replan")

    def _greedy_action(self, env: CCSEnv) -> list[int]:
        state = env.simulator.state
        action: list[int] = []
        for i, vessel_id in enumerate(env.vessel_ids):
            mask = env.action_mask()[i]
            cargo = state.entity_inventory_t.get(vessel_id, 0.0)
            if mask[VESSEL_GO_TERMINAL] and cargo > 1e-9:
                action.append(VESSEL_GO_TERMINAL)
            elif mask[VESSEL_GO_HOME] and cargo <= 1e-9:
                action.append(VESSEL_GO_HOME)
            else:
                action.append(VESSEL_WAIT)
        action += [WELL_ACTIONS - 1] * len(env.well_ids)
        return action

    def _ready_hours(self, env: CCSEnv, vid: str) -> int:
        """Hours from now until this vessel could next complete a delivery."""
        v = self._vp[vid]
        state = env.simulator.state
        vstate = env.simulator.vessel_states[vid]
        route = env._routes[vid]
        sail = self._sail_h[vid]
        cap = v.capacity_t
        cargo = state.entity_inventory_t.get(vid, 0.0)
        vessel = env.network.entities[vid]

        if vstate["mode"] == "sailing":
            remaining_sail = math.ceil(sail * (1.0 - float(vstate["progress"])))
            if vstate["destination"] == route["destination"]:
                return max(0, remaining_sail)                       # inbound: about to deliver
            return remaining_sail + v.load_dur_h + sail             # returning: reload then go
        # berthed
        if vstate["berth"] == route["destination"]:
            return 0 if cargo > 1e-9 else v.load_dur_h + 2 * sail   # at terminal
        # at home
        if cargo >= cap - 1e-9:
            return sail                                             # full: depart now, deliver in sail
        load_remaining = math.ceil(max(0.0, cap - cargo) / vessel.loading_rate_tph)
        return load_remaining + sail

    def _vessel_action(self, env: CCSEnv, vid: str, now: float, mask: list[bool]) -> int:
        vstate = env.simulator.vessel_states[vid]
        route = env._routes[vid]
        state = env.simulator.state
        cargo = state.entity_inventory_t.get(vid, 0.0)
        cap = self._vp[vid].capacity_t

        if vstate["mode"] != "berthed":
            return VESSEL_WAIT
        if vstate["berth"] == route["destination"]:
            # at terminal: head home once empty, else wait for auto-unload
            return VESSEL_GO_HOME if (cargo <= 1e-9 and mask[VESSEL_GO_HOME]) else VESSEL_WAIT
        # at home: depart now iff the plan wants this vessel's next delivery imminently
        if cargo >= cap - 1e-9 and mask[VESSEL_GO_TERMINAL]:
            planned = self._plan.get(vid, [])
            elapsed = now - self._plan_origin_h
            next_delivery = min((t for t in planned if t >= elapsed), default=None)
            if next_delivery is not None and next_delivery - elapsed <= self._sail_h[vid] + 1:
                return VESSEL_GO_TERMINAL
        return VESSEL_WAIT
