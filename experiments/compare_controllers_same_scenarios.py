"""Compare controllers on identical disturbance scenarios.

This script runs each episode controller against the same scenario seeds. For a
given seed, each controller gets a fresh environment, but the
``ScenarioGenerator`` receives the same seed, so capture noise/outages, weather,
well maintenance, injectivity changes, berth outages, and optional initial
inventory are identical across controllers.

The static MILP is reported separately as a nominal lower-bound benchmark,
because it is not an episode controller and does not execute the sampled
disturbance trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from sim.env import CCSEnv, CCSEnvConfig
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.metrics import EpisodeMetrics, greedy_shuttle_policy, idle_policy, run_episode
from sim.milp import solve_max_storage_fixed_horizon, solve_min_makespan
from sim.network import PhysicalNetwork
from sim.rolling_milp import RollingMilpController
from sim.scenario import Scenario, ScenarioConfig, ScenarioGenerator

ProgressLogger = Callable[[str], None]
PolicyFactory = Callable[[CCSEnv], Callable[[CCSEnv], list[int]]]

TOY_LOCATIONS = {
    "brevik": (59.05, 9.70),
    "oslo": (59.86, 10.84),
    "oygarden": (60.58, 4.84),
}

VESSEL_CAPACITY_T = 7_500.0
EMITTER_BUFFER_CAPACITY_T = 15_000.0
TERMINAL_STORAGE_CAPACITY_T = 15_000.0


def build_toy_network() -> PhysicalNetwork:
    """Small two-emitter/two-ship network used by the MILP regression tests."""
    network = PhysicalNetwork(time_step_hours=1.0)
    network.add_entity(Emitter("brevik", nominal_capture_tph=80.0, buffer_capacity_t=EMITTER_BUFFER_CAPACITY_T))
    network.add_entity(Emitter("oslo", nominal_capture_tph=60.0, buffer_capacity_t=EMITTER_BUFFER_CAPACITY_T))
    network.add_entity(
        Vessel(
            "ship_1",
            capacity_t=VESSEL_CAPACITY_T,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(
        Vessel(
            "ship_2",
            capacity_t=VESSEL_CAPACITY_T,
            loading_rate_tph=800.0,
            unloading_rate_tph=800.0,
            speed_knots=12.0,
        )
    )
    network.add_entity(Terminal("oygarden", storage_capacity_t=TERMINAL_STORAGE_CAPACITY_T, berth_count=2))
    network.add_entity(Pipeline("pipeline", max_flow_tph=400.0, ramp_tph=400.0))
    network.add_entity(SubseaManifold("manifold", max_flow_tph=400.0))
    network.add_entity(InjectionWell("well_1", max_injection_tph=200.0))
    network.add_entity(InjectionWell("well_2", max_injection_tph=200.0))
    network.add_entity(
        Reservoir(
            "aurora",
            storage_capacity_t=1e7,
            initial_pressure_bar=100.0,
            pressure_at_capacity_bar=200.0,
            max_pressure_bar=200.0,
        )
    )
    network.connect("brevik", "ship_1")
    network.connect("oslo", "ship_2")
    network.connect("ship_1", "oygarden")
    network.connect("ship_2", "oygarden")
    network.connect("oygarden", "pipeline")
    network.connect("pipeline", "manifold")
    network.connect("manifold", "well_1")
    network.connect("manifold", "well_2")
    network.connect("well_1", "aurora")
    network.connect("well_2", "aurora")
    return network


def make_env(
    *,
    target_t: float | None,
    cap_hours: int,
    scenario_seed_config: ScenarioConfig,
) -> CCSEnv:
    return CCSEnv(
        build_toy_network(),
        TOY_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=scenario_seed_config),
        config=CCSEnvConfig(episode_hours=cap_hours, storage_goal_t=target_t),
    )


def scenario_signature(scenario: Scenario) -> str:
    payload = {
        "initial_inventory_t": scenario.initial_inventory_t,
        "emitter_availability": scenario.emitter_availability,
        "vessel_speed_factor": scenario.vessel_speed_factor,
        "well_available": scenario.well_available,
        "injectivity_factor": scenario.injectivity_factor,
        "berth_count_override": scenario.berth_count_override,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def controller_factories(
    replan_every: int,
    progress: ProgressLogger,
    rolling_plan_target_t: float | None,
) -> dict[str, PolicyFactory]:
    return {
        "idle": lambda _env: idle_policy,
        "greedy_shuttle": lambda _env: greedy_shuttle_policy,
        "rolling_milp": lambda env: RollingMilpController(
            env,
            replan_every=replan_every,
            progress=progress,
            plan_target_t=rolling_plan_target_t,
        ),
    }


def metric_row(
    *,
    seed: int,
    controller: str,
    metrics: EpisodeMetrics,
    signature: str,
) -> dict[str, object]:
    row = metrics.as_dict()
    return {
        "seed": seed,
        "controller": controller,
        "scenario_signature": signature,
        **row,
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_controller: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_controller[str(row["controller"])].append(row)

    metrics = [
        "reached_target",
        "elapsed_hours",
        "captured_t",
        "stored_t",
        "vented_t",
        "backlog_growth_t",
        "loss_rate",
        "storage_rate",
        "operating_cost",
        "cost_per_stored_t",
        "throttle_hours",
        "well_switch_count",
        "berth_wait_vessel_hours",
        "pressure_risk_hours",
        "longest_venting_streak_hours",
    ]
    summary_rows: list[dict[str, object]] = []
    for controller, controller_rows in sorted(by_controller.items()):
        out: dict[str, object] = {"controller": controller, "episodes": len(controller_rows)}
        for metric in metrics:
            values = [row[metric] for row in controller_rows if row.get(metric) is not None]
            if not values:
                continue
            floats = [float(value) for value in values]
            mean = sum(floats) / len(floats)
            variance = sum((value - mean) ** 2 for value in floats) / len(floats)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = variance**0.5
        summary_rows.append(out)
    return summary_rows


def static_milp_benchmark(target_t: float, cap_hours: int) -> dict[str, object]:
    env = make_env(
        target_t=target_t,
        cap_hours=cap_hours,
        scenario_seed_config=ScenarioConfig(
            episode_hours=cap_hours,
            randomize_initial_inventory=False,
            capture_noise_std=0.0,
            capture_outage_rate_per_week=0.0,
            enable_weather=False,
            well_maintenance_rate_per_week=0.0,
            injectivity_max_decline=0.0,
            injectivity_noise_std=0.0,
            berth_outage_rate_per_week=0.0,
        ),
    )
    result = solve_min_makespan(env, target_t=target_t)
    return {
        "case": "static_milp_nominal_lower_bound",
        "target_t": target_t,
        "status": result.status,
        "reached": result.reached,
        "makespan_h": result.makespan_h,
        "deliveries": result.deliveries,
        "operating_cost": result.operating_cost,
        "cost_per_stored_t": result.cost_per_stored_t,
    }


def static_fixed_horizon_milp_benchmark(cap_hours: int, time_limit_s: float | None) -> dict[str, object]:
    env = make_env(
        target_t=None,
        cap_hours=cap_hours,
        scenario_seed_config=ScenarioConfig(
            episode_hours=cap_hours,
            randomize_initial_inventory=False,
            capture_noise_std=0.0,
            capture_outage_rate_per_week=0.0,
            enable_weather=False,
            well_maintenance_rate_per_week=0.0,
            injectivity_max_decline=0.0,
            injectivity_noise_std=0.0,
            berth_outage_rate_per_week=0.0,
        ),
    )
    result = solve_max_storage_fixed_horizon(env, horizon_h=cap_hours, time_limit_s=time_limit_s)
    return {
        "case": "static_milp_fixed_horizon_nominal",
        "horizon_h": cap_hours,
        "status": result.status,
        "stored_t": result.stored_t,
        "deliveries": result.deliveries,
        "operating_cost": result.operating_cost,
        "cost_per_stored_t": result.cost_per_stored_t,
        "time_limit_s": "" if time_limit_s is None else time_limit_s,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, *, summary_rows: list[dict[str, object]], benchmark: dict[str, object]) -> None:
    lines = [
        "# Same-Scenario Controller Comparison",
        "",
        "All episode controllers are evaluated on identical disturbance trajectories for each seed.",
        "The static MILP row is a separate nominal lower-bound benchmark, not a disturbed episode rollout.",
        "",
        "## Static MILP Benchmark",
        "",
    ]
    if benchmark.get("case") == "static_milp_fixed_horizon_nominal":
        lines.append(
            f"- horizon: {float(benchmark['horizon_h']):.0f} h; "
            f"stored: {float(benchmark['stored_t']):,.1f} t; "
            f"deliveries: {benchmark['deliveries']}; "
            f"operating cost: EUR {float(benchmark['operating_cost']):,.0f}; "
            f"cost/t: EUR {float(benchmark['cost_per_stored_t']):,.2f}"
        )
    elif benchmark.get("status") == "not_applicable_fixed_horizon":
        lines.append("- not applicable in fixed-horizon mode.")
    elif benchmark.get("status") == "skipped":
        lines.append(f"- target: {float(benchmark['target_t']):.0f} t; static MILP solve skipped.")
    else:
        lines.append(
            f"- target: {float(benchmark['target_t']):.0f} t; "
            f"makespan: {float(benchmark['makespan_h']):.0f} h; "
            f"deliveries: {benchmark['deliveries']}; "
            f"operating cost: EUR {float(benchmark['operating_cost']):,.0f}; "
            f"cost/t: EUR {float(benchmark['cost_per_stored_t']):,.2f}"
        )
    lines += [
        "",
        "## Episode Controller Summary",
        "",
        "| Controller | Success | Hours mean | Stored t mean | Vented t mean | Op cost mean | Cost/t mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        success = float(row.get("reached_target_mean", 0.0))
        cost_per_t = row.get("cost_per_stored_t_mean", "")
        cost_per_t_text = "" if cost_per_t == "" else f"{float(cost_per_t):,.2f}"
        lines.append(
            f"| {row['controller']} | {success:.0%} | "
            f"{float(row.get('elapsed_hours_mean', 0.0)):,.1f} | "
            f"{float(row.get('stored_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('vented_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('operating_cost_mean', 0.0)):,.0f} | "
            f"{cost_per_t_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixed-target", "fixed-horizon"], default="fixed-target")
    parser.add_argument("--target-t", type=float, default=1_600.0)
    parser.add_argument("--cap-hours", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["idle", "greedy_shuttle", "rolling_milp"],
        default=["idle", "greedy_shuttle", "rolling_milp"],
    )
    parser.add_argument("--rolling-replan-every", type=int, default=12)
    parser.add_argument(
        "--rolling-plan-target-t",
        type=float,
        default=None,
        help="Optional rolling MILP chunk target. Use this for large storage goals to avoid huge MILPs.",
    )
    parser.add_argument(
        "--skip-static-milp",
        action="store_true",
        help="Skip the full static MILP lower-bound solve, useful for very large targets.",
    )
    parser.add_argument(
        "--static-milp-time-limit-s",
        type=float,
        default=60.0,
        help="CBC time limit for fixed-horizon static MILP benchmark.",
    )
    parser.add_argument(
        "--random-initial-inventory",
        action="store_true",
        help="Also randomize starting emitter/terminal inventories. Off by default for cold-start iso-storage tests.",
    )
    parser.add_argument(
        "--quiet-scenario",
        action="store_true",
        help="Disable dynamic disturbances. Useful for nominal sanity checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    if args.quiet_scenario:
        scenario_config = ScenarioConfig(
            episode_hours=args.cap_hours,
            randomize_initial_inventory=args.random_initial_inventory,
            capture_noise_std=0.0,
            capture_outage_rate_per_week=0.0,
            enable_weather=False,
            well_maintenance_rate_per_week=0.0,
            injectivity_max_decline=0.0,
            injectivity_noise_std=0.0,
            berth_outage_rate_per_week=0.0,
        )
    else:
        scenario_config = ScenarioConfig(
            episode_hours=args.cap_hours,
            randomize_initial_inventory=args.random_initial_inventory,
        )

    rows: list[dict[str, object]] = []
    total_runs = len(args.seeds) * len(args.controllers)
    run_index = 0
    script_start = time.perf_counter()
    target_t = None if args.mode == "fixed-horizon" else args.target_t
    log(
        f"Running {total_runs} episode rollouts: mode={args.mode}, "
        f"target={'none' if target_t is None else f'{target_t:.0f} t'}, "
        f"cap={args.cap_hours} h, seeds={args.seeds}, controllers={args.controllers}",
    )
    factories = controller_factories(args.rolling_replan_every, log, args.rolling_plan_target_t)
    partial_path = args.output_dir / "controller_comparison_by_seed.partial.csv"
    for seed in args.seeds:
        seed_signatures = set()
        for controller_name in args.controllers:
            factory = factories[controller_name]
            run_index += 1
            run_start = time.perf_counter()
            log(f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} ...")
            env = make_env(
                target_t=target_t,
                cap_hours=args.cap_hours,
                scenario_seed_config=scenario_config,
            )
            policy = factory(env)
            metrics = run_episode(env, policy, seed=seed)
            signature = scenario_signature(env.scenario)
            seed_signatures.add(signature)
            rows.append(
                metric_row(
                    seed=seed,
                    controller=controller_name,
                    metrics=metrics,
                    signature=signature,
                )
            )
            write_csv(partial_path, rows)
            elapsed_s = time.perf_counter() - run_start
            log(
                f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} done "
                f"in {elapsed_s:.1f}s; reached={metrics.reached_target}; "
                f"hours={metrics.elapsed_hours:.0f}; stored={metrics.stored_t:.1f} t; "
                f"vented={metrics.vented_t:.1f} t; cost={metrics.operating_cost:,.0f}",
            )
        if len(seed_signatures) != 1:
            raise RuntimeError(f"Seed {seed} did not produce identical scenarios across controllers.")

    summary_rows = summarize(rows)
    if args.mode == "fixed-horizon":
        log("Solving static MILP fixed-horizon nominal benchmark ...")
        benchmark = static_fixed_horizon_milp_benchmark(args.cap_hours, args.static_milp_time_limit_s)
    elif args.skip_static_milp:
        log("Skipping static MILP nominal lower-bound benchmark.")
        benchmark = {
            "case": "static_milp_nominal_lower_bound",
            "target_t": args.target_t,
            "status": "skipped",
            "reached": "",
            "makespan_h": "",
            "deliveries": "",
            "operating_cost": "",
            "cost_per_stored_t": "",
        }
    else:
        log("Solving static MILP nominal lower-bound benchmark ...")
        benchmark = static_milp_benchmark(args.target_t, args.cap_hours)

    per_seed_path = args.output_dir / "controller_comparison_by_seed.csv"
    summary_path = args.output_dir / "controller_comparison_summary.csv"
    benchmark_path = args.output_dir / "static_milp_nominal_benchmark.csv"
    report_path = args.output_dir / "controller_comparison_report.md"

    write_csv(per_seed_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(benchmark_path, [benchmark])
    write_report(report_path, summary_rows=summary_rows, benchmark=benchmark)

    log(f"Wrote {per_seed_path}")
    log(f"Wrote {summary_path}")
    log(f"Wrote {benchmark_path}")
    log(f"Wrote {report_path}")
    log(f"Wrote {progress_path}")
    log(f"Total wall-clock time: {time.perf_counter() - script_start:.1f}s")


if __name__ == "__main__":
    main()
