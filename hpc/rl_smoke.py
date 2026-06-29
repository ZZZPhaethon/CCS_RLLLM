"""Minimal RL smoke test for the Borg HPC environment."""

from __future__ import annotations

import gymnasium
import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib import MaskablePPO

from sim.environment.gym_adapter import CCSGymEnv, make_ppo_policy
from sim.metrics import evaluate
from sim.train import make_native_env


def main() -> None:
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
    print("gymnasium", gymnasium.__version__)
    print("stable_baselines3", stable_baselines3.__version__)
    print("sb3_contrib", sb3_contrib.__version__)

    env = CCSGymEnv(make_native_env(episode_hours=24, warm_start=True))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=0,
        gamma=0.999,
        n_steps=16,
        batch_size=8,
        verbose=0,
        device="auto",
    )
    print("model_device", model.device)
    model.learn(total_timesteps=16)

    eval_env = make_native_env(episode_hours=24, warm_start=False)
    _episodes, summary = evaluate(eval_env, make_ppo_policy(model), seeds=[101])
    print("storage_rate", summary["storage_rate"]["mean"])
    print("loss_rate", summary["loss_rate"]["mean"])
    print("RL_SMOKE_OK")


if __name__ == "__main__":
    main()
