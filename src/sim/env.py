"""``CCSEnv`` - a reinforcement-learning environment over the CCS physics.

This is the layer that turns the deterministic physical twin into something an
RL agent can train against. Each episode :meth:`reset` samples a
:class:`~sim.scenario.Scenario` (the exogenous disturbances), and each
:meth:`step` maps a discrete control vector into physical action proposals, runs
one hour of physics through the :class:`~sim.simulator.PhysicalSimulator`,
applies the scenario disturbances, and prices the outcome with the
:class:`~sim.economics.CostModel` to produce the reward.

The interface is gym-style (``reset`` / ``step`` returning
``(obs, reward, done, info)``) but intentionally has **no numpy or gymnasium
dependency** so it stays importable anywhere. Observations are flat ``list[float]``
and the action space is a small ``MultiDiscrete`` described by
:attr:`CCSEnv.action_dims`; wrapping it for SB3/gymnasium later is trivial.

Controls (section 7.2 of the research note):
- per vessel: ``WAIT`` / ``GO_HOME`` / ``GO_TERMINAL``;
- per well: ``OFF`` / ``LOW`` / ``MEDIUM`` / ``HIGH`` injection mode.

Loading at the home emitter and unloading at the terminal are issued
automatically (they are never the interesting decision); the agent chooses where
to send vessels and how hard to inject. An action mask exposes which choices are
physically legal so the policy only selects feasible actions (section 7.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionFrame, ActionProposal
from .economics import CostModel, EconomicLedger
from .entities.emitter import Emitter
from .entities.manifold import SubseaManifold
from .entities.pipeline import Pipeline
from .entities.state import PhysicalState
from .entities.storage import InjectionWell, Reservoir
from .entities.terminal import Terminal
from .entities.vessel import Vessel
from .routes import route_distance_km, sea_route
from .scenario import Scenario, ScenarioGenerator
from .simulator import PhysicalSimulator

# Vessel action ids.
VESSEL_WAIT, VESSEL_GO_HOME, VESSEL_GO_TERMINAL = 0, 1, 2
VESSEL_ACTIONS = 3

# Well injection modes mapped to a fraction of nominal max injection.
WELL_MODE_FRACTIONS = (0.0, 0.34, 0.67, 1.0)
WELL_ACTIONS = len(WELL_MODE_FRACTIONS)

Coordinate = tuple[float, float]


@dataclass
class CCSEnvConfig:
    episode_hours: int = 168
    storage_target_rate: float = 0.9
    reward_scale: float = 1e-3
    default_speed_knots: float = 12.0
    # Goal mode: if set, the episode genuinely terminates once this many tonnes
    # have been safely stored. This turns the episode into an iso-storage task -
    # "store T tonnes, how long and at what cost?" - and makes goal-reaching a
    # true `terminated` (not a time-limit truncation). episode_hours is the cap.
    storage_goal_t: float | None = None


class CCSEnv:
    """Single-agent (centralized) RL environment over the CCS network."""

    def __init__(
        self,
        network,
        locations: dict[str, Coordinate],
        scenario_generator: ScenarioGenerator | None = None,
        cost_model: CostModel | None = None,
        config: CCSEnvConfig | None = None,
        *,
        routes: dict[str, dict] | None = None,
    ) -> None:
        self.network = network
        self.config = config or CCSEnvConfig()
        self.scenario_generator = scenario_generator or ScenarioGenerator()
        self.cost_model = cost_model or CostModel()
        self.locations = locations
        self._routes = routes or self._build_routes(locations)

        self.emitter_ids = sorted(network._entities_of_type(Emitter))
        self.vessel_ids = sorted(network._entities_of_type(Vessel))
        self.terminal_ids = sorted(network._entities_of_type(Terminal))
        self.well_ids = sorted(network._entities_of_type(InjectionWell))
        self.reservoir_ids = sorted(network._entities_of_type(Reservoir))

        # Total capacity that can hold in-transit (backlog) CO2, used to normalise
        # the backlog observation into a horizon-invariant [0, 1] fill signal.
        self._backlog_capacity_t = (
            sum(network.entities[e].buffer_capacity_t for e in self.emitter_ids)
            + sum(network.entities[v].capacity_t for v in self.vessel_ids)
            + sum(network.entities[t].storage_capacity_t for t in self.terminal_ids)
        )

        self.n_steps = max(1, int(round(self.config.episode_hours / network.time_step_hours)))

        # Episode state, populated by reset().
        self.scenario: Scenario | None = None
        self.simulator: PhysicalSimulator | None = None
        self.t = 0
        self.ledger = EconomicLedger()
        self.cumulative_captured_t = 0.0
        self.cumulative_stored_t = 0.0
        self.last_info: dict = {}

    # -- spaces -----------------------------------------------------------
    @property
    def action_dims(self) -> list[int]:
        return [VESSEL_ACTIONS] * len(self.vessel_ids) + [WELL_ACTIONS] * len(self.well_ids)

    @property
    def observation_size(self) -> int:
        return len(self.feature_names)

    @property
    def feature_names(self) -> list[str]:
        # Horizon-invariant globals only: a weekly clock (weather/ops cycle) and
        # the instantaneous system backlog fill. No episode-relative features, so a
        # policy trained on short episodes transfers to a long evaluation rollout.
        names = ["hour_of_week", "backlog_fill"]
        for eid in self.emitter_ids:
            names += [f"{eid}.fill", f"{eid}.capture_norm", f"{eid}.availability"]
        for vid in self.vessel_ids:
            names += [f"{vid}.cargo", f"{vid}.berthed", f"{vid}.at_home", f"{vid}.at_terminal", f"{vid}.progress"]
        for tid in self.terminal_ids:
            names += [f"{tid}.fill", f"{tid}.berth_frac"]
        for wid in self.well_ids:
            names += [f"{wid}.inject_norm", f"{wid}.injectivity", f"{wid}.available"]
        for rid in self.reservoir_ids:
            names += [f"{rid}.pressure_margin"]
        return names

    # -- episode lifecycle ------------------------------------------------
    def reset(self, seed: int | None = None) -> list[float]:
        self.scenario = self.scenario_generator.sample(self.network, seed=seed)
        state = PhysicalState()
        self.scenario.apply_initial(state)
        self.simulator = PhysicalSimulator(
            self.network, state, routes=self._routes, locations=self.locations
        )
        self.t = 0
        self.ledger = EconomicLedger()
        self.cumulative_captured_t = 0.0
        self.cumulative_stored_t = 0.0
        self.initial_backlog_t = self._backlog()
        self._prev_backlog_t = self.initial_backlog_t
        self.cumulative_backlog_penalty = 0.0
        self._apply_disturbances()
        self.last_info = {"action_mask": self.action_mask()}
        return self._observation()

    def step(self, action: list[int]) -> tuple[list[float], float, bool, bool, dict]:
        if self.simulator is None or self.scenario is None:
            raise RuntimeError("Call reset() before step().")
        if len(action) != len(self.action_dims):
            raise ValueError(f"Expected action of length {len(self.action_dims)}, got {len(action)}.")

        hours = self.network.time_step_hours
        current_time = self.simulator.state.time_h
        proposals = self._build_proposals(action)
        record = self.simulator.step(
            ActionFrame(time_h=current_time, proposals=proposals), compute_observation=False
        )
        step_result = record.step_result

        economics = self.cost_model.evaluate_step(self.network, step_result)
        # Gross captured = CO2 that entered the buffer plus CO2 the plant captured
        # but had to vent for lack of buffer/logistics. Venting therefore lowers
        # the storage rate, matching the section 8 definition (stored / captured).
        captured_step = (
            sum(step_result.state.last_capture_tph.values())
            + sum(step_result.state.last_vent_tph.values())
        ) * hours
        self.cumulative_captured_t += captured_step
        self.cumulative_stored_t += economics.stored_t
        self.ledger.add(economics)

        # Horizon-appropriate shaping: penalize backlog (in-transit CO2) *growth*
        # instead of the absolute annual-target gap. Capture adds to backlog every
        # hour; storing drains it. A positive penalty means the system fell behind
        # this step, a negative one (reward) means it caught up. Summed over the
        # episode this telescopes to (backlog_end - backlog_start), so there is no
        # reward farming. Recoverable in-transit CO2 is no longer mislabelled as a
        # contractual miss; venting (true loss) is still penalized via economics.
        backlog_now = self._backlog()
        backlog_growth = backlog_now - self._prev_backlog_t
        backlog_penalty = backlog_growth * self.cost_model.parameters.backlog_penalty_eur_per_t
        self._prev_backlog_t = backlog_now
        self.cumulative_backlog_penalty += backlog_penalty
        reward = (economics.net - backlog_penalty) * self.config.reward_scale

        self.t += 1
        # Without a goal the operation never truly "ends" - reaching the horizon is
        # a time limit (truncation), not a terminal state, so terminated stays False
        # and the trainer bootstraps V(s_T). In goal mode, storing the target amount
        # IS a genuine terminal (the task is done), so terminated becomes True.
        goal = self.config.storage_goal_t
        terminated = goal is not None and self.cumulative_stored_t >= goal
        truncated = (not terminated) and self.t >= self.n_steps
        if not (terminated or truncated):
            self._apply_disturbances()

        info = {
            "time_h": step_result.state.time_h,
            "economics": economics.as_dict(),
            "backlog_t": backlog_now,
            "backlog_growth_t": backlog_growth,
            "backlog_penalty": backlog_penalty,
            "storage_rate": self.storage_rate(),
            "loss_rate": self.loss_rate(),
            "violations": [v.violation_type for v in step_result.violations],
            "action_mask": self.action_mask(),
        }
        self.last_info = info
        return self._observation(), reward, terminated, truncated, info

    def storage_rate(self) -> float:
        """Stored / gross-captured. Only meaningful over a long horizon, where
        in-transit CO2 is negligible relative to total captured."""
        if self.cumulative_captured_t <= 0.0:
            return 1.0
        return self.cumulative_stored_t / self.cumulative_captured_t

    def loss_rate(self) -> float:
        """Vented / gross-captured: the share of captured CO2 truly lost. This is
        the short-horizon truth - it ignores recoverable in-transit inventory."""
        if self.cumulative_captured_t <= 0.0:
            return 0.0
        return self.ledger.vented_t / self.cumulative_captured_t

    def _backlog(self) -> float:
        """In-transit CO2: captured but not yet stored (everything but reservoirs)."""
        reservoirs = set(self.reservoir_ids)
        return sum(
            inventory
            for entity_id, inventory in self.simulator.state.entity_inventory_t.items()
            if entity_id not in reservoirs
        )

    # -- action mask ------------------------------------------------------
    def action_mask(self) -> list[list[bool]]:
        mask: list[list[bool]] = []
        for vid in self.vessel_ids:
            mask.append(self._vessel_mask(vid))
        for wid in self.well_ids:
            mask.append(self._well_mask(wid))
        return mask

    def _vessel_mask(self, vessel_id: str) -> list[bool]:
        vstate = self.simulator.vessel_states[vessel_id]
        route = self._routes[vessel_id]
        if vstate["mode"] != "berthed":
            return [True, False, False]  # mid-voyage: can only WAIT
        berth = vstate["berth"]
        at_home = berth == route["origin"]
        at_terminal = berth == route["destination"]
        return [True, not at_home, not at_terminal]

    def _well_mask(self, well_id: str) -> list[bool]:
        available = self.simulator.state.well_available.get(well_id, True)
        if not available:
            return [True] + [False] * (WELL_ACTIONS - 1)  # maintenance: only OFF
        return [True] * WELL_ACTIONS

    # -- action translation ----------------------------------------------
    def _build_proposals(self, action: list[int]) -> list[ActionProposal]:
        vessel_actions = action[: len(self.vessel_ids)]
        well_actions = action[len(self.vessel_ids):]
        proposals: list[ActionProposal] = []

        # Always capture at full rate (capture is not an RL control here).
        for emitter_id in self.emitter_ids:
            proposals.append(self._proposal(emitter_id, "set_capture_utilization", {"utilization": 1.0}))

        departing = self._vessel_dispatch_proposals(vessel_actions, proposals)
        self._auto_loading_proposals(proposals, departing)
        self._auto_unloading_proposals(proposals, departing)
        self._injection_proposals(well_actions, proposals)
        return proposals

    def _vessel_dispatch_proposals(self, vessel_actions, proposals) -> set[str]:
        departing: set[str] = set()
        for vessel_id, choice in zip(self.vessel_ids, vessel_actions):
            vstate = self.simulator.vessel_states[vessel_id]
            if vstate["mode"] != "berthed":
                continue
            route = self._routes[vessel_id]
            berth = vstate["berth"]
            destination = None
            if choice == VESSEL_GO_HOME and berth != route["origin"]:
                destination = route["origin"]
            elif choice == VESSEL_GO_TERMINAL and berth != route["destination"]:
                destination = route["destination"]
            if destination is not None:
                proposals.append(self._proposal(vessel_id, "sail_to", {"destination_id": destination}))
                departing.add(vessel_id)
        return departing

    def _auto_loading_proposals(self, proposals, departing) -> None:
        loaded_emitters: set[str] = set()
        for vessel_id in self.vessel_ids:
            if vessel_id in departing:
                continue
            vstate = self.simulator.vessel_states[vessel_id]
            route = self._routes[vessel_id]
            home = route["origin"]
            if vstate["mode"] != "berthed" or vstate["berth"] != home or home in loaded_emitters:
                continue
            vessel = self.network.entities[vessel_id]
            cargo = self.simulator.state.entity_inventory_t.get(vessel_id, 0.0)
            if cargo < vessel.capacity_t - 1e-9:
                proposals.append(self._proposal(home, "load_vessel", {"vessel_id": vessel_id}))
                loaded_emitters.add(home)

    def _auto_unloading_proposals(self, proposals, departing) -> None:
        for terminal_id in self.terminal_ids:
            head = self._terminal_unload_head(terminal_id, departing)
            if head is not None:
                proposals.append(self._proposal(terminal_id, "unload_vessel", {"vessel_id": head}))

    def _terminal_unload_head(self, terminal_id: str, departing) -> str | None:
        candidates = []
        for vessel_id in self.network._upstream_of_type(terminal_id, Vessel):
            if vessel_id in departing:
                continue
            if self.simulator.state.vessel_berths.get(vessel_id) != terminal_id:
                continue
            if self.simulator.state.entity_inventory_t.get(vessel_id, 0.0) > 1e-9:
                candidates.append(vessel_id)
        return sorted(candidates)[0] if candidates else None

    def _injection_proposals(self, well_actions, proposals) -> None:
        desired: dict[str, float] = {}
        for well_id, choice in zip(self.well_ids, well_actions):
            well = self.network.entities[well_id]
            available = self.simulator.state.well_available.get(well_id, True)
            fraction = WELL_MODE_FRACTIONS[choice] if available else 0.0
            desired[well_id] = fraction * well.max_injection_tph

        for pipeline_id in self.network._entities_of_type(Pipeline):
            wells = self._pipeline_wells(pipeline_id)
            total = sum(desired.get(w, 0.0) for w in wells)
            proposals.append(self._proposal(pipeline_id, "set_flow", {"flow_tph": total}))
            self._manifold_split_proposals(pipeline_id, desired, proposals)

    def _manifold_split_proposals(self, pipeline_id, desired, proposals) -> None:
        for manifold_id in self.network._downstream_of_type(pipeline_id, SubseaManifold):
            wells = self.network._downstream_of_type(manifold_id, InjectionWell)
            total = sum(desired.get(w, 0.0) for w in wells)
            if total <= 1e-9:
                continue  # all OFF: no split needed, pipeline flow already excludes them
            splits = {w: desired.get(w, 0.0) / total for w in wells}
            proposals.append(self._proposal(manifold_id, "set_well_split", {"well_splits": splits}))

    def _pipeline_wells(self, pipeline_id: str) -> list[str]:
        wells = list(self.network._downstream_of_type(pipeline_id, InjectionWell))
        for manifold_id in self.network._downstream_of_type(pipeline_id, SubseaManifold):
            wells += self.network._downstream_of_type(manifold_id, InjectionWell)
        return wells

    # -- observation ------------------------------------------------------
    def _observation(self) -> list[float]:
        state = self.simulator.state
        hour_of_week = (state.time_h % 168.0) / 168.0
        backlog_fill = _safe_div(self._backlog(), self._backlog_capacity_t)
        obs: list[float] = [hour_of_week, backlog_fill]
        for eid in self.emitter_ids:
            emitter = self.network.entities[eid]
            inv = state.entity_inventory_t.get(eid, 0.0)
            obs += [
                _safe_div(inv, emitter.buffer_capacity_t),
                _safe_div(state.last_capture_tph.get(eid, 0.0), emitter.nominal_capture_tph),
                state.emitter_availability.get(eid, emitter.availability),
            ]
        for vid in self.vessel_ids:
            vessel = self.network.entities[vid]
            vstate = self.simulator.vessel_states[vid]
            route = self._routes[vid]
            berthed = vstate["mode"] == "berthed"
            obs += [
                _safe_div(state.entity_inventory_t.get(vid, 0.0), vessel.capacity_t),
                1.0 if berthed else 0.0,
                1.0 if berthed and vstate["berth"] == route["origin"] else 0.0,
                1.0 if berthed and vstate["berth"] == route["destination"] else 0.0,
                float(vstate["progress"]),
            ]
        for tid in self.terminal_ids:
            terminal = self.network.entities[tid]
            berth_override = state.berth_count_override.get(tid, terminal.berth_count)
            obs += [
                _safe_div(state.entity_inventory_t.get(tid, 0.0), terminal.storage_capacity_t),
                _safe_div(berth_override, max(1, terminal.berth_count)),
            ]
        for wid in self.well_ids:
            well = self.network.entities[wid]
            obs += [
                _safe_div(state.last_injection_flow_tph.get(wid, 0.0), well.max_injection_tph),
                state.injectivity_factor.get(wid, 1.0),
                1.0 if state.well_available.get(wid, True) else 0.0,
            ]
        for rid in self.reservoir_ids:
            reservoir = self.network.entities[rid]
            inv = state.entity_inventory_t.get(rid, 0.0)
            span = reservoir.max_pressure_bar - reservoir.initial_pressure_bar
            obs += [_safe_div(reservoir.pressure_margin_bar(inv), span) if span > 0 else 1.0]
        return obs

    # -- helpers ----------------------------------------------------------
    def _apply_disturbances(self) -> None:
        self.scenario.apply_to_state(self.simulator.state, self.simulator.state.time_h)

    def _proposal(self, entity_id: str, verb: str, params: dict) -> ActionProposal:
        return ActionProposal(agent_id="ccs_env", entity_id=entity_id, verb=verb, params=params)

    def _build_routes(self, locations: dict[str, Coordinate]) -> dict[str, dict]:
        routes: dict[str, dict] = {}
        for vessel_id in sorted(self.network._entities_of_type(Vessel)):
            origin_id = self._upstream_id(vessel_id, Emitter)
            destination_id = self._downstream_id(vessel_id, Terminal)
            if origin_id is None or destination_id is None:
                raise ValueError(
                    f"Vessel {vessel_id} needs an upstream emitter and downstream terminal to build a route."
                )
            for location_id in (origin_id, destination_id):
                if location_id not in locations:
                    raise ValueError(f"Missing location coordinate for {location_id}.")
            route = sea_route(locations[origin_id], locations[destination_id])
            vessel = self.network.entities[vessel_id]
            routes[vessel_id] = {
                "origin": origin_id,
                "destination": destination_id,
                "distance_km": route.distance_km,
                "speed_knots": vessel.speed_knots or self.config.default_speed_knots,
                "coordinates": route.coordinates,
                "return_coordinates": list(reversed(route.coordinates)),
            }
        return routes

    def _upstream_id(self, entity_id: str, entity_type: type) -> str | None:
        matches = self.network._upstream_of_type(entity_id, entity_type)
        return matches[0] if matches else None

    def _downstream_id(self, entity_id: str, entity_type: type) -> str | None:
        matches = self.network._downstream_of_type(entity_id, entity_type)
        return matches[0] if matches else None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
