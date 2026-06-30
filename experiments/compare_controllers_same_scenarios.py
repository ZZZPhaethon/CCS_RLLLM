"""Compare controllers on identical disturbance scenarios.

Each controller gets a fresh environment for a seed, while ``ScenarioGenerator``
uses the same seed so disturbances are comparable across controllers. Static
MILP benchmarks are reported separately because they are perfect-foresight
optimizers, not episode controllers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from sim.control.baselines import greedy_shuttle_policy, idle_policy
from sim.control.milp import solve_max_storage_fixed_horizon
from sim.control.rule_based import RuleBasedActionGenerator
from sim.control.rolling_milp import RollingMilpController
from sim.entities import Emitter, InjectionWell, Pipeline, Reservoir, SubseaManifold, Terminal, Vessel
from sim.environment import (
    CCSEnv,
    CCSEnvConfig,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_ACTIONS,
)
from sim.metrics import EpisodeMetrics, run_episode
from sim.network import PhysicalNetwork
from sim.scenario_generation import Scenario, ScenarioConfig, ScenarioGenerator

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
    """Small two-emitter/two-ship network used by controller comparisons."""
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
    cap_hours: int,
    scenario_seed_config: ScenarioConfig,
) -> CCSEnv:
    return CCSEnv(
        build_toy_network(),
        TOY_LOCATIONS,
        scenario_generator=ScenarioGenerator(config=scenario_seed_config),
        config=CCSEnvConfig(episode_hours=cap_hours),
    )


def scenario_signature(scenario: Scenario) -> str:
    payload = {
        "initial_inventory_t": scenario.initial_inventory_t,
        "emitter_availability": scenario.emitter_availability,
        "vessel_speed_factor": scenario.vessel_speed_factor,
        "well_available": scenario.well_available,
        "injectivity_factor": scenario.injectivity_factor,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def rule_based_env_policy(env: CCSEnv) -> Callable[[CCSEnv], list[int]]:
    generator = RuleBasedActionGenerator(env.network, env._routes)

    def policy(current_env: CCSEnv) -> list[int]:
        frame = generator.next_action_frame(current_env.simulator.state)
        action = [VESSEL_WAIT] * len(current_env.vessel_ids) + [1] * len(current_env.well_ids)
        mask = current_env.action_mask()
        vessel_index = {vessel_id: i for i, vessel_id in enumerate(current_env.vessel_ids)}
        well_index = {well_id: i for i, well_id in enumerate(current_env.well_ids)}

        for proposal in frame.proposals:
            if proposal.verb == "sail_to" and proposal.entity_id in vessel_index:
                i = vessel_index[proposal.entity_id]
                route = current_env._routes[proposal.entity_id]
                destination = str(proposal.params["destination_id"])
                if destination in current_env.emitter_ids:
                    emitter_action = current_env.vessel_go_emitter_action(destination)
                    if mask[i][emitter_action]:
                        action[i] = emitter_action
                elif destination == route["destination"] and mask[i][VESSEL_GO_TERMINAL]:
                    action[i] = VESSEL_GO_TERMINAL
            elif proposal.verb == "set_well_split":
                for well_id, split in proposal.params["well_splits"].items():
                    if float(split) <= 0.0 or well_id not in well_index:
                        continue
                    i = len(current_env.vessel_ids) + well_index[well_id]
                    if mask[i][WELL_ACTIONS - 1]:
                        action[i] = WELL_ACTIONS - 1

        return action

    return policy


def controller_factories(
    replan_every: int,
    progress: ProgressLogger,
    rolling_planning_horizon_h: int,
) -> dict[str, PolicyFactory]:
    return {
        "idle": lambda _env: idle_policy,
        "greedy_shuttle": lambda _env: greedy_shuttle_policy,
        "rule_based": rule_based_env_policy,
        "rolling_milp": lambda env: RollingMilpController(
            env,
            replan_every=replan_every,
            progress=progress,
            planning_horizon_h=rolling_planning_horizon_h,
        ),
    }


def metric_row(
    *,
    seed: int,
    controller: str,
    metrics: EpisodeMetrics,
    signature: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "controller": controller,
        "scenario_signature": signature,
        **metrics.as_dict(),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_controller: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_controller[str(row["controller"])].append(row)

    metrics = [
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


def static_fixed_horizon_milp_benchmark(
    cap_hours: int,
    time_limit_s: float | None,
    mip_gap_rel: float | None,
    mip_gap_abs: float | None,
    *,
    scenario_seed_config: ScenarioConfig,
    seed: int,
) -> dict[str, object]:
    env = make_env(
        cap_hours=cap_hours,
        scenario_seed_config=scenario_seed_config,
    )
    scenario = env.scenario_generator.sample(env.network, seed=seed)
    result = solve_max_storage_fixed_horizon(
        env,
        horizon_h=cap_hours,
        time_limit_s=time_limit_s,
        mip_gap_rel=mip_gap_rel,
        mip_gap_abs=mip_gap_abs,
        scenario=scenario,
    )
    return {
        "case": "static_milp_fixed_horizon_scenario_oracle",
        "seed": seed,
        "scenario_signature": scenario_signature(scenario),
        "horizon_h": cap_hours,
        "status": result.status,
        "stored_t": result.stored_t,
        "deliveries": result.deliveries,
        "operating_cost": result.operating_cost,
        "cost_per_stored_t": result.cost_per_stored_t,
        "time_limit_s": "" if time_limit_s is None else time_limit_s,
        "mip_gap_rel": "" if mip_gap_rel is None else mip_gap_rel,
        "mip_gap_abs": "" if mip_gap_abs is None else mip_gap_abs,
    }


def summarize_static_fixed_horizon_benchmarks(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) == 1:
        return rows[0]
    stored_mean = sum(float(row["stored_t"]) for row in rows) / len(rows)
    cost_mean = sum(float(row["operating_cost"]) for row in rows) / len(rows)
    return {
        "case": "static_milp_fixed_horizon_scenario_oracle_summary",
        "episodes": len(rows),
        "horizon_h": rows[0]["horizon_h"],
        "status": ";".join(sorted({str(row["status"]) for row in rows})),
        "stored_t": stored_mean,
        "deliveries": sum(float(row["deliveries"]) for row in rows) / len(rows),
        "operating_cost": cost_mean,
        "cost_per_stored_t": cost_mean / stored_mean if stored_mean > 0 else float("nan"),
        "time_limit_s": rows[0].get("time_limit_s", ""),
        "mip_gap_rel": rows[0].get("mip_gap_rel", ""),
        "mip_gap_abs": rows[0].get("mip_gap_abs", ""),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_cost_per_t(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"EUR {numeric:,.2f}"


def write_report(path: Path, *, summary_rows: list[dict[str, object]], benchmark: dict[str, object]) -> None:
    lines = [
        "# Same-Scenario Controller Comparison",
        "",
        "All episode controllers are evaluated on identical disturbance trajectories for each seed.",
        "The static MILP row is a separate perfect-foresight benchmark, not an online episode controller.",
        "",
        "## Static MILP Benchmark",
        "",
    ]
    if benchmark.get("status") == "skipped":
        lines.append(f"- horizon: {float(benchmark['horizon_h']):.0f} h; static MILP solve skipped.")
    elif str(benchmark.get("case", "")).startswith("static_milp_fixed_horizon"):
        label = "same-scenario oracle" if "scenario_oracle" in str(benchmark.get("case", "")) else "nominal"
        episodes = benchmark.get("episodes")
        episodes_text = "" if episodes is None else f"; episodes: {episodes}"
        lines.append(
            f"- {label}; horizon: {float(benchmark['horizon_h']):.0f} h{episodes_text}; "
            f"status: {benchmark['status']}; "
            f"stored: {float(benchmark['stored_t']):,.1f} t; "
            f"deliveries: {benchmark['deliveries']}; "
            f"operating cost: EUR {float(benchmark['operating_cost']):,.0f}; "
            f"cost/t: {format_cost_per_t(benchmark['cost_per_stored_t'])}"
        )
    else:
        raise ValueError(f"Unsupported benchmark row: {benchmark}")
    lines += [
        "",
        "## Episode Controller Summary",
        "",
        "| Controller | Hours mean | Stored t mean | Vented t mean | Op cost mean | Cost/t mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        cost_per_t = row.get("cost_per_stored_t_mean", "")
        cost_per_t_text = "" if cost_per_t == "" else f"{float(cost_per_t):,.2f}"
        lines.append(
            f"| {row['controller']} | {float(row.get('elapsed_hours_mean', 0.0)):,.1f} | "
            f"{float(row.get('stored_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('vented_t_mean', 0.0)):,.1f} | "
            f"{float(row.get('operating_cost_mean', 0.0)):,.0f} | "
            f"{cost_per_t_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-hours", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["idle", "greedy_shuttle", "rule_based", "rolling_milp"],
        default=["idle", "greedy_shuttle", "rolling_milp"],
    )
    parser.add_argument("--rolling-replan-every", type=int, default=12)
    parser.add_argument(
        "--rolling-planning-horizon-h",
        type=int,
        default=168,
        help="Lookahead horizon for the rolling fixed-horizon MILP controller.",
    )
    parser.add_argument("--skip-static-milp", action="store_true")
    parser.add_argument("--static-milp-time-limit-s", type=float, default=600.0)
    parser.add_argument(
        "--static-milp-gap-rel",
        type=float,
        default=None,
        help="Optional relative MIP gap tolerance for the fixed-horizon static MILP.",
    )
    parser.add_argument(
        "--static-milp-gap-abs",
        type=float,
        default=None,
        help="Optional absolute MIP gap tolerance for the fixed-horizon static MILP.",
    )
    parser.add_argument("--random-initial-inventory", action="store_true")
    parser.add_argument("--quiet-scenario", action="store_true")
    return parser.parse_args()


def _quiet_config(cap_hours: int, random_initial_inventory: bool) -> ScenarioConfig:
    return ScenarioConfig(
        episode_hours=cap_hours,
        randomize_initial_inventory=random_initial_inventory,
        capture_noise_std=0.0,
        capture_outage_rate_per_week=0.0,
        enable_weather=False,
        well_maintenance_rate_per_week=0.0,
        injectivity_max_decline=0.0,
        injectivity_noise_std=0.0,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    scenario_config = (
        _quiet_config(args.cap_hours, args.random_initial_inventory)
        if args.quiet_scenario
        else ScenarioConfig(
            episode_hours=args.cap_hours,
            randomize_initial_inventory=args.random_initial_inventory,
        )
    )

    rows: list[dict[str, object]] = []
    seed_to_signature: dict[int, str] = {}
    total_runs = len(args.seeds) * len(args.controllers)
    run_index = 0
    script_start = time.perf_counter()
    log(
        f"Running {total_runs} fixed-horizon episode rollouts: "
        f"cap={args.cap_hours} h, seeds={args.seeds}, controllers={args.controllers}"
    )
    factories = controller_factories(args.rolling_replan_every, log, args.rolling_planning_horizon_h)
    partial_path = args.output_dir / "controller_comparison_by_seed.partial.csv"
    for seed in args.seeds:
        seed_signatures = set()
        for controller_name in args.controllers:
            factory = factories[controller_name]
            run_index += 1
            run_start = time.perf_counter()
            log(f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} ...")
            env = make_env(cap_hours=args.cap_hours, scenario_seed_config=scenario_config)
            metrics = run_episode(env, factory(env), seed=seed)
            signature = scenario_signature(env.scenario)
            seed_signatures.add(signature)
            rows.append(metric_row(seed=seed, controller=controller_name, metrics=metrics, signature=signature))
            write_csv(partial_path, rows)
            log(
                f"[{run_index}/{total_runs}] seed={seed} controller={controller_name} done "
                f"in {time.perf_counter() - run_start:.1f}s; "
                f"hours={metrics.elapsed_hours:.0f}; stored={metrics.stored_t:.1f} t; "
                f"vented={metrics.vented_t:.1f} t; cost={metrics.operating_cost:,.0f}"
            )
        if len(seed_signatures) != 1:
            raise RuntimeError(f"Seed {seed} did not produce identical scenarios across controllers.")
        seed_to_signature[seed] = next(iter(seed_signatures))

    summary_rows = summarize(rows)
    benchmark_rows: list[dict[str, object]]
    if args.skip_static_milp:
        log("Skipping static MILP fixed-horizon same-scenario oracle.")
        benchmark = {
            "case": "static_milp_fixed_horizon_scenario_oracle",
            "episodes": len(args.seeds),
            "horizon_h": args.cap_hours,
            "status": "skipped",
            "stored_t": "",
            "deliveries": "",
            "operating_cost": "",
            "cost_per_stored_t": "",
            "time_limit_s": "",
            "mip_gap_rel": "",
            "mip_gap_abs": "",
        }
        benchmark_rows = [benchmark]
    else:
        benchmark_rows = []
        for seed in args.seeds:
            log(f"Solving static MILP fixed-horizon same-scenario oracle for seed={seed} ...")
            row = static_fixed_horizon_milp_benchmark(
                args.cap_hours,
                args.static_milp_time_limit_s,
                args.static_milp_gap_rel,
                args.static_milp_gap_abs,
                scenario_seed_config=scenario_config,
                seed=seed,
            )
            if row["scenario_signature"] != seed_to_signature[seed]:
                raise RuntimeError(f"Static MILP scenario signature mismatch for seed {seed}.")
            benchmark_rows.append(row)
        benchmark = summarize_static_fixed_horizon_benchmarks(benchmark_rows)

    write_csv(args.output_dir / "controller_comparison_by_seed.csv", rows)
    write_csv(args.output_dir / "controller_comparison_summary.csv", summary_rows)
    write_csv(args.output_dir / "static_milp_same_scenario_benchmark.csv", benchmark_rows)
    write_report(args.output_dir / "controller_comparison_report.md", summary_rows=summary_rows, benchmark=benchmark)
    log(f"Wrote outputs to {args.output_dir}")
    log(f"Total wall-clock time: {time.perf_counter() - script_start:.1f}s")


if __name__ == "__main__":
    main()
