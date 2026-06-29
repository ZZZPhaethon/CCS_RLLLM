from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else default


def _fmt_tonnes(value: float) -> str:
    return f"{value:,.1f} t"


def _fmt_eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def _fmt_eur_t(value: float | None) -> str:
    return "n/a" if value is None else f"EUR {value:,.2f}/t"


def _summary_cards(summary: list[dict[str, str]], benchmark: dict[str, str]) -> str:
    rolling = next(row for row in summary if row["controller"] == "rolling_milp")
    greedy = next(row for row in summary if row["controller"] == "greedy_shuttle")
    cards = [
        ("Best episode stored", "rolling_milp", _fmt_tonnes(_float(rolling, "stored_t_mean"))),
        ("Lowest episode venting", "rolling_milp", _fmt_tonnes(_float(rolling, "vented_t_mean"))),
        ("Best episode EUR/t", "rolling_milp", _fmt_eur_t(_float(rolling, "cost_per_stored_t_mean"))),
        ("Nominal MILP upper bound", "static_milp", _fmt_tonnes(_float(benchmark, "stored_t"))),
        ("Greedy stored", "greedy_shuttle", _fmt_tonnes(_float(greedy, "stored_t_mean"))),
        ("Greedy venting", "greedy_shuttle", _fmt_tonnes(_float(greedy, "vented_t_mean"))),
    ]
    return "\n".join(
        f"""
        <section class="card">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
          <small>{html.escape(owner)}</small>
        </section>
        """
        for label, owner, value in cards
    )


def _bar_rows(summary: list[dict[str, str]], metric: str, label: str, formatter) -> str:
    max_value = max(_float(row, metric) for row in summary) or 1.0
    rows = []
    for row in summary:
        value = _float(row, metric)
        width = max(1.0, value / max_value * 100.0)
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(row["controller"])}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div class="bar-value">{html.escape(formatter(value))}</div>
            </div>
            """
        )
    return f"<h3>{html.escape(label)}</h3>" + "\n".join(rows)


def _summary_table(summary: list[dict[str, str]]) -> str:
    rows = []
    for row in summary:
        cost_per_t = row.get("cost_per_stored_t_mean")
        rows.append(
            f"""
            <tr>
              <td>{html.escape(row["controller"])}</td>
              <td>{_float(row, "episodes"):.0f}</td>
              <td>{_fmt_tonnes(_float(row, "stored_t_mean"))}</td>
              <td>{_fmt_tonnes(_float(row, "vented_t_mean"))}</td>
              <td>{_float(row, "storage_rate_mean"):.1%}</td>
              <td>{_fmt_eur(_float(row, "operating_cost_mean"))}</td>
              <td>{_fmt_eur_t(float(cost_per_t) if cost_per_t else None)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def _by_seed_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            f"""
            <tr>
              <td>{html.escape(row["seed"])}</td>
              <td>{html.escape(row["controller"])}</td>
              <td>{html.escape(row["scenario_signature"])}</td>
              <td>{_fmt_tonnes(_float(row, "stored_t"))}</td>
              <td>{_fmt_tonnes(_float(row, "vented_t"))}</td>
              <td>{_fmt_eur(_float(row, "operating_cost"))}</td>
              <td>{_fmt_eur_t(_float(row, "cost_per_stored_t") if row.get("cost_per_stored_t") else None)}</td>
            </tr>
            """
        )
    return "\n".join(body)


def build_html(input_dir: Path, title: str) -> str:
    summary = _read_csv(input_dir / "controller_comparison_summary.csv")
    by_seed = _read_csv(input_dir / "controller_comparison_by_seed.csv")
    benchmark = _read_csv(input_dir / "static_milp_nominal_benchmark.csv")[0]
    benchmark_cost_per_t = _float(benchmark, "cost_per_stored_t")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px 28px 48px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card span, .card small {{ display: block; color: var(--muted); }}
    .card strong {{ display: block; margin: 8px 0; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }}
    .panel {{ margin-top: 18px; overflow-x: auto; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 16px 0 10px; font-size: 14px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 130px minmax(180px, 1fr) 120px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}
    .bar-label {{ font-weight: 700; }}
    .bar-track {{ height: 12px; background: #eef2f7; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); }}
    .bar-value {{ color: var(--muted); text-align: right; }}
    .links a {{
      display: inline-block;
      margin: 6px 10px 0 0;
      padding: 8px 10px;
      border-radius: 6px;
      background: var(--accent-soft);
      color: #115e59;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .cards, .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Fixed-horizon comparison over 720 h with 7,500 t vessels. Episode controllers use identical disturbance trajectories for each seed. Static MILP is a nominal perfect-information benchmark.</p>
  </header>
  <main>
    <section class="cards">
      {_summary_cards(summary, benchmark)}
    </section>

    <section class="panel">
      <h2>Static MILP Benchmark</h2>
      <p>Nominal fixed-horizon MILP stores <strong>{_fmt_tonnes(_float(benchmark, "stored_t"))}</strong> in 720 h with {html.escape(benchmark["deliveries"])} deliveries, operating cost {_fmt_eur(_float(benchmark, "operating_cost"))}, and cost {_fmt_eur_t(benchmark_cost_per_t)}.</p>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Throughput</h2>
        {_bar_rows(summary, "stored_t_mean", "Mean stored CO2", _fmt_tonnes)}
      </div>
      <div class="panel">
        <h2>Losses And Cost</h2>
        {_bar_rows(summary, "vented_t_mean", "Mean vented CO2", _fmt_tonnes)}
        {_bar_rows(summary, "cost_per_stored_t_mean", "Mean cost per stored tonne", lambda value: _fmt_eur_t(value) if value > 0 else "n/a")}
      </div>
    </section>

    <section class="panel">
      <h2>Controller Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Controller</th><th>Episodes</th><th>Stored mean</th><th>Vented mean</th><th>Storage rate</th><th>Operating cost</th><th>Cost/t</th>
          </tr>
        </thead>
        <tbody>{_summary_table(summary)}</tbody>
      </table>
    </section>

    <section class="panel links">
      <h2>Dispatch Dashboards</h2>
      <p>Open these to inspect hour-by-hour vessel movements, terminal inventory, injection, and action logs.</p>
      <a href="controller_greedy_shuttle_720h_dashboard.html">Greedy Shuttle Dashboard</a>
      <a href="controller_rolling_milp_720h_dashboard.html">Rolling MILP Dashboard</a>
    </section>

    <section class="panel">
      <h2>Per-Seed Results</h2>
      <table>
        <thead>
          <tr>
            <th>Seed</th><th>Controller</th><th>Scenario signature</th><th>Stored</th><th>Vented</th><th>Operating cost</th><th>Cost/t</th>
          </tr>
        </thead>
        <tbody>{_by_seed_table(by_seed)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("output/fixed_horizon_720h_vessel7500t"))
    parser.add_argument("--output", type=Path, default=Path("docs/controller_comparison_720h_vessel7500t_results.html"))
    parser.add_argument("--title", default="Controller Comparison - 720 h, 7,500 t Vessels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(args.input_dir, args.title), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
