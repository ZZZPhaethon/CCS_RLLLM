"""Phase 1 fixed-horizon MILP benchmark entry point."""

from __future__ import annotations

import argparse

from sim.control.baselines import greedy_shuttle_policy
from sim.environment import CCSEnv, CCSEnvConfig, build_phase1_env
from sim.metrics import evaluate, format_fixed_horizon
from sim.control.milp import FixedHorizonMilpResult, solve_max_storage_fixed_horizon
from sim.control.rolling_milp import RollingMilpController
from sim.scenario_generation import ScenarioConfig, ScenarioGenerator


def make_nominal_scenario_config(episode_hours: int) -> ScenarioConfig:
    """Deterministic cold-start scenario config for benchmark comparability."""
    return ScenarioConfig(
        episode_hours=episode_hours,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
        randomize_initial_inventory=False,
        warm_start=False,
    )


def make_phase1_yara_env(
    episode_hours: int,
    *,
    nominal: bool = True,
) -> CCSEnv:
    """Build the real Phase 1 environment in fixed-horizon mode."""
    scenario_config = (
        make_nominal_scenario_config(episode_hours)
        if nominal
        else ScenarioConfig(episode_hours=episode_hours)
    )
    return build_phase1_env(
        scenario_generator=ScenarioGenerator(config=scenario_config),
        config=CCSEnvConfig(episode_hours=episode_hours),
    )


def run_static_milp(
    episode_hours: int,
    *,
    time_limit_s: float | None = None,
    msg: bool = False,
) -> FixedHorizonMilpResult:
    """Solve the idealized fixed-horizon upper bound on the real Phase 1 case."""
    env = make_phase1_yara_env(episode_hours=episode_hours)
    return solve_max_storage_fixed_horizon(
        env,
        horizon_h=episode_hours,
        time_limit_s=time_limit_s,
        msg=msg,
    )


def compare_policies(
    episode_hours: int,
    *,
    seeds: list[int],
    replan_every: int,
) -> str:
    """Compare executable policies on the same Phase 1 fixed-horizon task."""
    rows = {}
    for name, policy_factory in {
        "greedy_shuttle": lambda env: greedy_shuttle_policy,
        "rolling_milp": lambda env: RollingMilpController(env, replan_every=replan_every),
    }.items():
        env = make_phase1_yara_env(episode_hours=episode_hours)
        _episodes, summary = evaluate(env, policy_factory(env), seeds=seeds)
        rows[name] = summary
    return format_fixed_horizon(rows)


def _format_static_milp(result: FixedHorizonMilpResult) -> str:
    lines = [
        "=== Static MILP fixed-horizon oracle ===",
        f"status             : {result.status}",
        f"horizon            : {result.horizon_h:,.0f} h",
        f"stored             : {result.stored_t:,.0f} t",
        f"deliveries         : {result.deliveries}",
        f"operating cost     : EUR {result.operating_cost:,.0f}",
        f"cost / stored t    : EUR {result.cost_per_stored_t:,.1f}",
        "schedule:",
    ]
    for vessel_id, hours in result.schedule.items():
        rendered = ", ".join(str(h) for h in hours) if hours else "-"
        lines.append(f"  {vessel_id:20} {rendered}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MILP baselines on Phase 1.")
    parser.add_argument("--episode-hours", type=int, default=720)
    parser.add_argument("--static-milp-time-limit-s", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--replan-every", type=int, default=12)
    parser.add_argument("--skip-policies", action="store_true")
    parser.add_argument("--solver-msg", action="store_true")
    args = parser.parse_args()

    result = run_static_milp(
        episode_hours=args.episode_hours,
        time_limit_s=args.static_milp_time_limit_s,
        msg=args.solver_msg,
    )
    print(_format_static_milp(result))

    if not args.skip_policies:
        print("\n=== Executable policy comparison ===")
        print(
            compare_policies(
                episode_hours=args.episode_hours,
                seeds=args.seeds,
                replan_every=args.replan_every,
            )
        )


if __name__ == "__main__":
    main()
