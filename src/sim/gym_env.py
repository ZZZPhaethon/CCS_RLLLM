"""Gymnasium adapter so RL libraries can train against :class:`CCSEnv`.

``CCSGymEnv`` exposes the native env as a standard ``gymnasium.Env`` with a
``MultiDiscrete`` action space, a ``Box`` observation space, and an
``action_masks()`` method consumed by ``sb3_contrib.MaskablePPO`` so the policy
only ever samples physically legal actions.

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


def flat_action_mask(mask: list[list[bool]]) -> np.ndarray:
    """Flatten the per-dimension legality mask into the MultiDiscrete layout."""
    return np.array([legal for dimension in mask for legal in dimension], dtype=bool)


class CCSGymEnv(gym.Env):
    """A ``gymnasium.Env`` view over a :class:`CCSEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, env: CCSEnv) -> None:
        super().__init__()
        self.env = env
        self.action_space = spaces.MultiDiscrete(env.action_dims)
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
        obs, reward, terminated, truncated, info = self.env.step([int(a) for a in action])
        return self._to_array(obs), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return flat_action_mask(self.env.action_mask())

    def _to_array(self, obs: list[float]) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


def make_ppo_policy(model):
    """Wrap a trained MaskablePPO model as a metrics ``policy(env) -> action``.

    Lets the trained policy be scored by the same ``sim.metrics`` harness as the
    heuristic baselines, on the native :class:`CCSEnv`.
    """

    def policy(env: CCSEnv) -> list[int]:
        obs = np.asarray(env._observation(), dtype=np.float32)
        masks = flat_action_mask(env.action_mask())
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        return [int(a) for a in action]

    return policy
