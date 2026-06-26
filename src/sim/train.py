"""Train a centralized RL policy on the CCS network and score it vs baselines.

Uses ``MaskablePPO`` (action masking + automatic ``V(s_T)`` bootstrapping at the
truncated horizon). ``gamma`` defaults to 0.999 (~6-week lookahead in hourly
steps) so the value function can see the week-scale backlog -> venting
consequence the reward shaping points at; slower (yearly) dynamics are handled by
warm-start state randomization, not by the discount.

Run as a script:
    PYTHONPATH=src python -m sim.train --timesteps 200000
"""

from __future__ import annotations

import argparse

from .env import CCSEnvConfig
from .env_scenarios import build_phase1_plus_yara_env
from .gym_env import CCSGymEnv, make_ppo_policy
from .metrics import evaluate, greedy_shuttle_policy, idle_policy
from .scenario import ScenarioConfig, ScenarioGenerator


def make_native_env(
    episode_hours: int = 168,
    storage_target_rate: float = 0.9,
    warm_start: bool = True,
):
    """A native CCSEnv on the real Phase 1 + Yara network configured for RL."""
    return build_phase1_plus_yara_env(
        scenario_generator=ScenarioGenerator(
            config=ScenarioConfig(episode_hours=episode_hours, warm_start=warm_start)
        ),
        config=CCSEnvConfig(episode_hours=episode_hours, storage_target_rate=storage_target_rate),
    )


def train_ppo(
    total_timesteps: int = 200_000,
    seed: int = 0,
    gamma: float = 0.999,
    episode_hours: int = 168,
    warm_start: bool = True,
    verbose: int = 1,
):
    from sb3_contrib import MaskablePPO

    gym_env = CCSGymEnv(make_native_env(episode_hours=episode_hours, warm_start=warm_start))
    model = MaskablePPO(
        "MlpPolicy",
        gym_env,
        gamma=gamma,
        seed=seed,
        n_steps=2048,
        batch_size=256,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def compare(model, seeds: list[int], episode_hours: int = 168, warm_start: bool = False):
    """Score idle / greedy_shuttle / PPO on the same scenarios."""
    policies = {
        "idle": idle_policy,
        "greedy_shuttle": greedy_shuttle_policy,
        "ppo": make_ppo_policy(model),
    }
    rows = {}
    for name, policy in policies.items():
        env = make_native_env(episode_hours=episode_hours, warm_start=warm_start)
        _episodes, summary = evaluate(env, policy, seeds=seeds)
        rows[name] = summary
    return rows


def _format_comparison(rows: dict) -> str:
    header = f"{'policy':16} {'storage%':>9} {'loss%':>7} {'backlog+':>10} {'net EUR':>16}"
    lines = [header, "-" * len(header)]
    for name, s in rows.items():
        lines.append(
            f"{name:16} {s['storage_rate']['mean'] * 100:8.1f}% "
            f"{s['loss_rate']['mean'] * 100:6.1f}% "
            f"{s['backlog_growth_t']['mean']:10,.0f} "
            f"{s['net']['mean']:16,.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on the CCS network.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = parser.parse_args()

    model = train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
        episode_hours=args.episode_hours,
    )
    rows = compare(model, seeds=args.eval_seeds, episode_hours=args.episode_hours)
    print("\n=== PPO vs baselines (Phase 1 + Yara, evaluation seeds) ===")
    print(_format_comparison(rows))


if __name__ == "__main__":
    main()
