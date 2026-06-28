"""Behavior cloning to rescue PPO from the do-nothing collapse.

Vanilla PPO never explores a full ~100 h delivery cycle, so it sinks into the
"barely operate" local optimum. Behavior cloning sidesteps the exploration wall:
we collect ``(observation, action)`` pairs from a demonstrator that already knows
how to ship (greedy_shuttle, or the rolling-MILP controller), then supervise the
PPO policy network to reproduce those actions. The cloned policy starts out
"knowing how to run vessels", and PPO fine-tuning takes over from there.

Because we train the *same* ``MaskablePPO`` policy object, the hand-off to RL is
seamless: maximize the log-probability of the demonstrator's action under the
policy distribution, then call ``model.learn()`` to fine-tune.
"""

from __future__ import annotations

import numpy as np

from .gym_env import CCSGymEnv
from .metrics import Policy


def collect_demonstrations(
    gym_env: CCSGymEnv,
    demo_policy: Policy,
    n_episodes: int,
    seed0: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the demonstrator and record ``(obs, action)`` pairs."""
    obs_rows: list[np.ndarray] = []
    act_rows: list[list[int]] = []
    for i in range(n_episodes):
        obs, _ = gym_env.reset(seed=seed0 + i)
        done = False
        while not done:
            action = demo_policy(gym_env.env)  # demonstrator acts on the native env
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            act_rows.append([int(a) for a in action])
            obs, _reward, terminated, truncated, _info = gym_env.step(action)
            done = terminated or truncated
    return np.asarray(obs_rows, dtype=np.float32), np.asarray(act_rows, dtype=np.int64)


def behavior_clone(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    """Supervise ``model.policy`` to imitate the demonstrator's actions in place."""
    import torch

    policy = model.policy
    policy.set_training_mode(True)
    device = policy.device
    obs_t = torch.as_tensor(observations, device=device)
    act_t = torch.as_tensor(actions, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n = obs_t.shape[0]
    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            distribution = policy.get_distribution(obs_t[idx])
            # log_prob of the demonstrator action (= negative cross-entropy).
            log_prob = distribution.log_prob(act_t[idx])
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    policy.set_training_mode(False)
