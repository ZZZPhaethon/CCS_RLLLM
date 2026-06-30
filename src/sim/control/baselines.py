"""Simple baseline controllers for CCS environments."""

from __future__ import annotations

from ..environment import (
    CCSEnv,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_ACTIONS,
)

_EPS = 1e-9


def idle_policy(env: CCSEnv) -> list[int]:
    """Do nothing with vessels while holding wells at their minimum stable mode."""
    return [VESSEL_WAIT] * len(env.vessel_ids) + [1] * len(env.well_ids)


def greedy_shuttle_policy(env: CCSEnv) -> list[int]:
    """Send full vessels to terminal, otherwise serve the best buffered emitter."""
    state = env.simulator.state
    action: list[int] = []
    for i, vessel_id in enumerate(env.vessel_ids):
        mask = env.action_mask()[i]
        cargo = state.entity_inventory_t.get(vessel_id, 0.0)
        vessel = env.network.entities[vessel_id]
        berth = state.vessel_berths.get(vessel_id)
        best_emitter_action = _best_emitter_action(env, mask)

        if berth in env.terminal_ids and cargo > _EPS:
            action.append(VESSEL_WAIT)
        elif mask[VESSEL_GO_TERMINAL] and cargo >= vessel.capacity_t - _EPS:
            action.append(VESSEL_GO_TERMINAL)
        elif (
            berth in env.emitter_ids
            and cargo < vessel.capacity_t - _EPS
            and _emitter_supply_score(env, str(berth)) > _EPS
        ):
            action.append(VESSEL_WAIT)
        elif best_emitter_action is not None:
            action.append(best_emitter_action)
        elif mask[VESSEL_GO_TERMINAL] and cargo > _EPS:
            action.append(VESSEL_GO_TERMINAL)
        else:
            action.append(VESSEL_WAIT)
    action += [WELL_ACTIONS - 1] * len(env.well_ids)
    return action


def _best_emitter_action(env: CCSEnv, mask: list[bool]) -> int | None:
    best: tuple[float, int] | None = None
    for emitter_id in env.emitter_ids:
        action = env.vessel_go_emitter_action(emitter_id)
        if not mask[action]:
            continue
        score = _emitter_supply_score(env, emitter_id)
        if best is None or score > best[0]:
            best = (score, action)
    return None if best is None else best[1]


def _emitter_supply_score(env: CCSEnv, emitter_id: str) -> float:
    emitter = env.network.entities[emitter_id]
    state = env.simulator.state
    availability = state.emitter_availability.get(emitter_id, emitter.availability)
    return state.entity_inventory_t.get(emitter_id, 0.0) + emitter.nominal_capture_tph * max(0.0, availability)
