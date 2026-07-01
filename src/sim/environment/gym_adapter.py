"""Gymnasium adapter so RL libraries can train against :class:`CCSEnv`.

``CCSGymEnv`` exposes the native env as a standard ``gymnasium.Env`` with a
``Dict`` action space: ``MultiDiscrete`` vessel destinations plus normalized
continuous well rates. ``action_masks()`` exposes the vessel mask for hybrid
policies that support discrete masking.

The episode boundary is reported as ``truncated`` (never ``terminated``), which
tells the trainer to bootstrap ``V(s_T)`` instead of zeroing the future - the
operation continues past the 168 h training window, it is not a true terminal.

This module is the only place that imports numpy/gymnasium; the simulation core
stays dependency-free.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("CCSGymEnv requires gymnasium. Install with `pip install gymnasium`.") from exc

from .env import CCSEnv


def flat_vessel_action_mask(mask: list[list[bool]]) -> np.ndarray:
    """Flatten the per-vessel legality mask into MultiDiscrete order."""
    return np.array([legal for dimension in mask for legal in dimension], dtype=bool)


def well_unit_to_rates(unit_rates, bounds: list[tuple[float, float]]) -> list[float]:
    """Map normalized [0, 1] well controls to Mt/y under current env bounds."""
    rates: list[float] = []
    clipped = np.clip(np.asarray(unit_rates, dtype=np.float32), 0.0, 1.0)
    for unit, (lower, upper) in zip(clipped, bounds):
        if upper <= 0.0:
            rates.append(0.0)
        else:
            rates.append(float(lower + unit * (upper - lower)))
    return rates


class CCSGymEnv(gym.Env):
    """A ``gymnasium.Env`` view over a :class:`CCSEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv) -> None:
        super().__init__()
        self.env = env
        self.action_space = spaces.Dict(
            {
                "vessels": spaces.MultiDiscrete(env.vessel_action_dims),
                "wells": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(env.well_ids),),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(env.observation_size,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Draw a fresh per-episode scenario seed from the (optionally seeded)
        # np_random, so episodes are varied yet reproducible: domain randomization.
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        obs = self.env.reset(seed=episode_seed)
        return self._to_array(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self._native_action(action))
        return self._to_array(obs), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return flat_vessel_action_mask(self.env.vessel_action_mask())

    def _to_array(self, obs: list[float]) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def _native_action(self, action) -> dict[str, list]:
        return {
            "vessels": [int(a) for a in action["vessels"]],
            "wells": well_unit_to_rates(action["wells"], self.env.well_rate_bounds()),
        }


def make_ppo_policy(model):
    """Wrap a trained hybrid-action PPO model as a metrics ``policy(env) -> action``.

    Lets the trained policy be scored by the same ``sim.metrics`` harness as the
    heuristic baselines, on the native :class:`CCSEnv`.
    """

    def policy(env: CCSEnv) -> dict[str, list]:
        obs = np.asarray(env._observation(), dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return {
            "vessels": [int(a) for a in action["vessels"]],
            "wells": well_unit_to_rates(action["wells"], env.well_rate_bounds()),
        }

    return policy
