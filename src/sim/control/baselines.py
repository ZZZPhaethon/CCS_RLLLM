"""Simple baseline controllers for CCS environments."""

from __future__ import annotations

from ..environment import (
    CCSEnv,
    VESSEL_GO_HOME,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_ACTIONS,
)

_EPS = 1e-9


def idle_policy(env: CCSEnv) -> list[int]:
    """Do nothing: never dispatch a vessel, never inject."""
    return [VESSEL_WAIT] * len(env.vessel_ids) + [0] * len(env.well_ids)


def greedy_shuttle_policy(env: CCSEnv) -> list[int]:
    """Send loaded vessels to the terminal, empties home; inject at full rate."""
    state = env.simulator.state
    action: list[int] = []
    for i, vessel_id in enumerate(env.vessel_ids):
        mask = env.action_mask()[i]
        cargo = state.entity_inventory_t.get(vessel_id, 0.0)
        if mask[VESSEL_GO_TERMINAL] and cargo > _EPS:
            action.append(VESSEL_GO_TERMINAL)
        elif mask[VESSEL_GO_HOME] and cargo <= _EPS:
            action.append(VESSEL_GO_HOME)
        else:
            action.append(VESSEL_WAIT)
    action += [WELL_ACTIONS - 1] * len(env.well_ids)
    return action
