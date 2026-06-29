from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from ..physical.actions import ActionFrame, ActionProposal
from ..physical.entities.emitter import Emitter
from ..physical.entities.pipeline import Pipeline
from ..physical.entities.storage import InjectionWell, Reservoir
from ..physical.entities.terminal import Terminal
from ..physical.entities.vessel import Vessel
from ..physical.routes import route_distance_km, sea_route
from ..physical.scenarios import (
    C1_INJECTION_WELL_LOCATION,
    EOS_SUBSEA_TEMPLATE_LOCATION,
    NATURGASSPARKEN,
    NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH,
    NORTHERN_LIGHTS_PHASE2_DATA_PATH,
    build_northern_lights_phase1_demo,
    build_northern_lights_phase1_plus_yara_demo,
    build_northern_lights_phase2_demo,
)
from ..physical.simulator import PhysicalSimulator

Coordinate = tuple[float, float]
MAX_DISPLAY_ROUTE_SEGMENT_KM = 25.0
CORNER_SMOOTHING_FRACTION = 0.12
CORNER_SMOOTHING_STEPS = 5

LOCATIONS: dict[str, dict[str, float | str]] = {
    "brevik": {"lat": 59.05, "lon": 9.70, "label": "Brevik"},
    "celsio": {"lat": 59.86, "lon": 10.84, "label": "Celsio Oslo"},
    "oygarden_terminal": {
        "lat": NATURGASSPARKEN[0],
        "lon": NATURGASSPARKEN[1],
        "label": "Naturgassparken (Northern Lights Carbon Capture Plant Site)",
    },
    "oygarden_pipeline": {"lat": 60.58, "lon": 3.65, "label": "Offshore CO2 Pipeline"},
    "aurora_subsea_manifold": {
        "lat": EOS_SUBSEA_TEMPLATE_LOCATION[0],
        "lon": EOS_SUBSEA_TEMPLATE_LOCATION[1],
        "label": "31/5-7 EOS Subsea Distribution Manifold",
    },
    "aurora_well_a": {
        "lat": EOS_SUBSEA_TEMPLATE_LOCATION[0],
        "lon": EOS_SUBSEA_TEMPLATE_LOCATION[1],
        "label": "31/5-A-7 AH Injection Well",
    },
    "aurora_well_c": {
        "lat": C1_INJECTION_WELL_LOCATION[0],
        "lon": C1_INJECTION_WELL_LOCATION[1],
        "label": "31/5-C-1 H Injection Well",
    },
    "aurora_reservoir": {"lat": 60.55, "lon": 3.46, "label": "Aurora Reservoir"},
}

VESSEL_ROUTES = {
    "northern_pioneer": ("brevik", "oygarden_terminal", 0.05),
    "northern_pathfinder": ("celsio", "oygarden_terminal", 0.45),
}

def build_demo_trajectory(
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase1_demo()
    return build_trajectory(
        network,
        state,
        locations=LOCATIONS,
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="CCS Physical Layer Dashboard",
    )


def build_northern_lights_phase2_trajectory(
    hours: int = 240,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase2_demo()
    with NORTHERN_LIGHTS_PHASE2_DATA_PATH.open(encoding="utf-8") as handle:
        scenario_data = json.load(handle)
    return build_trajectory(
        network,
        state,
        locations=_locations_from_phase2_data(scenario_data),
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="Northern Lights Phase 2 Dashboard",
    )


def build_northern_lights_phase1_plus_yara_trajectory(
    hours: int = 72,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> dict[str, Any]:
    network, state = build_northern_lights_phase1_plus_yara_demo()
    with NORTHERN_LIGHTS_PHASE1_PLUS_YARA_DATA_PATH.open(encoding="utf-8") as handle:
        scenario_data = json.load(handle)
    return build_trajectory(
        network,
        state,
        locations=_locations_from_scenario_data(scenario_data),
        hours=hours,
        action_frames=action_frames,
        action_generator_factory=action_generator_factory,
        title="Northern Lights Phase 1 + Yara Dashboard",
    )


def _locations_from_phase2_data(data: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    return _locations_from_scenario_data(data)


def _locations_from_scenario_data(data: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    locations: dict[str, dict[str, float | str]] = {}

    for emitter in data["emitters"]:
        lat, lon = emitter["coordinates"]
        locations[emitter["entity_id"]] = {
            "lat": float(lat),
            "lon": float(lon),
            "label": emitter["reference_name"],
        }

    terminal = data["terminal"]
    terminal_lat, terminal_lon = data["locations"][terminal["entity_id"]]
    locations[terminal["entity_id"]] = {
        "lat": float(terminal_lat),
        "lon": float(terminal_lon),
        "label": terminal["site_name"],
    }

    route = data["offshore_pipeline_route"]
    midpoint = route[len(route) // 2]
    locations[data["pipeline"]["entity_id"]] = {
        "lat": float(midpoint[0]),
        "lon": float(midpoint[1]),
        "label": "Offshore CO2 Pipeline",
    }

    manifold = data["manifold"]
    manifold_lat, manifold_lon = data["locations"]["eos_subsea_template"]
    locations[manifold["entity_id"]] = {
        "lat": float(manifold_lat),
        "lon": float(manifold_lon),
        "label": "31/5-7 EOS Subsea Distribution Manifold",
    }

    for well in data["injection_wells"]:
        coordinates = well.get("coordinates")
        if coordinates is None:
            continue
        lat, lon = coordinates
        locations[well["entity_id"]] = {
            "lat": float(lat),
            "lon": float(lon),
            "label": well.get("public_name") or well["entity_id"],
        }

    locations[data["reservoir"]["entity_id"]] = {
        "lat": 60.55,
        "lon": 3.46,
        "label": "Aurora Reservoir",
    }
    return locations


def build_trajectory(
    network: Any,
    state: Any,
    locations: dict[str, dict[str, float | str]],
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
    title: str = "CCS Physical Layer Dashboard",
) -> dict[str, Any]:
    if hours < 0:
        raise ValueError("hours must be non-negative")

    routes = _build_routes(network, locations)
    simulator = PhysicalSimulator(network, state, routes, _dashboard_locations(locations))
    action_generator = action_generator_factory(network, routes) if action_generator_factory else None
    if action_generator is not None and hasattr(action_generator, "attach_simulator"):
        action_generator.attach_simulator(simulator)
    frames = [
        _frame_from_state(
            network,
            simulator.state,
            flows={},
            violations=[],
            actions={},
            vessel_positions=simulator.vessel_positions(),
        )
    ]

    for hour in range(hours):
        if action_generator is not None:
            action_frame = action_generator.next_action_frame(simulator.state)
        else:
            actions = _actions_at(action_frames, hour)
            action_frame = ActionFrame(time_h=simulator.state.time_h, proposals=_legacy_action_proposals(actions))
        record = simulator.step(action_frame)
        frames.append(
            _frame_from_state(
                network,
                record.step_result.state,
                flows=record.step_result.as_dict()["flows_t"],
                violations=record.step_result.as_dict()["violations"],
                actions=record.committed_action_frame.actions,
                vessel_positions=record.vessel_positions,
            )
        )

    first_route = next(iter(routes.values()), {})
    return {
        "title": title,
        "time_step_hours": network.time_step_hours,
        "route": {
            "provider": first_route.get("provider"),
            "distance_km": first_route.get("distance_km"),
            "coordinates": first_route.get("coordinates", []),
        },
        "map": {
            "bbox": _map_bbox(routes, network, locations),
            "locations": locations,
            "routes": routes,
            "pipeline_segments": _build_pipeline_segments(network, locations),
            "injection_links": _build_injection_links(network, locations),
        },
        "connections": [
            {"source": connection.source, "target": connection.target}
            for connection in network.connections
        ],
        "storage_targets": _storage_targets(network),
        "frames": frames,
    }


def render_dashboard_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, indent=2)
    escaped_title = html.escape(str(payload.get("title", "CCS Physical Layer Dashboard")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d8dee6;
      --sea: #d8eef7;
      --land: #edf3e8;
      --accent: #167c80;
      --accent-2: #2563eb;
      --ship: #b45309;
      --warn: #b7791f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }}
    #timeLabel {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(290px, 360px) 1fr minmax(300px, 380px);
      gap: 16px;
      padding: 16px;
      height: calc(100vh - 76px);
      min-height: 0;
      overflow: hidden;
    }}
    section, aside {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
      overflow: hidden;
    }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    button {{
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    input[type="range"] {{ width: min(440px, 36vw); }}
    .playback-control {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    #playbackDelayMs {{
      width: 78px;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 8px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }}
    .panel-title {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    #componentList {{
      display: grid;
      gap: 10px;
      padding: 12px;
      max-height: calc(100vh - 150px);
      overflow: auto;
    }}
    .component-group {{
      display: grid;
      gap: 8px;
    }}
    .component-group-toggle {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
      min-height: 34px;
      padding: 0 10px;
      border-color: var(--line);
      background: #f8fafc;
      color: var(--ink);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .component-group-count {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0;
      text-transform: none;
    }}
    .component-group-items {{
      display: grid;
      gap: 8px;
    }}
    .component {{
      display: grid;
      gap: 7px;
      width: 100%;
      min-height: 88px;
      text-align: left;
      border: 1px solid var(--line);
      border-left: 5px solid var(--component-color, var(--accent));
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      cursor: pointer;
      overflow: hidden;
    }}
    .component.active {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(22,124,128,.12); }}
    .component-name {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: start;
      font-weight: 700;
      line-height: 1.25;
    }}
    .component-name span:first-child {{
      overflow-wrap: anywhere;
      min-width: 0;
    }}
    .component-type-row {{
      display: block;
      width: 100%;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .component-type-row {{
      display: flex;
      gap: 7px;
      align-items: center;
    }}
    .component-summary {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    .ship-color-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--ship-color);
      box-shadow: 0 0 0 2px #fff, 0 0 0 3px rgba(31,41,51,.16);
      flex: 0 0 auto;
    }}
    .component-capacity {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      white-space: nowrap;
      text-align: right;
    }}
    .bar {{ height: 8px; border-radius: 999px; background: #eef2f6; overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; background: var(--component-color, var(--accent)); width: 0%; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(5, minmax(110px, 1fr));
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ font-size: 20px; font-weight: 750; margin-top: 4px; }}
    .workspace {{ display: grid; grid-template-rows: minmax(0, 1fr) 300px; min-height: 0; }}
    .map-panel {{ display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }}
    #networkCanvas {{ width: 100%; height: 100%; min-height: 0; background: var(--sea); }}
    .leaflet-container {{ font: 12px Inter, Segoe UI, Arial, sans-serif; }}
    .map-legend {{
      display: grid;
      gap: 5px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.94);
      color: var(--ink);
      box-shadow: 0 2px 8px rgba(31,41,51,.14);
    }}
    .legend-row {{ display: flex; gap: 7px; align-items: center; white-space: nowrap; }}
    .legend-line {{ width: 28px; height: 0; border-top: 3px solid #2563eb; }}
    .legend-line.return {{ border-top-color: #ca8a04; border-top-style: dashed; }}
    .legend-line.pipeline {{ border-top-color: #ff0000; border-top-width: 5px; }}
    .legend-line.subsea {{ border-top-color: #7c3aed; border-top-style: dashed; }}
    .legend-line.injection {{ border-top-color: #64748b; border-top-style: dotted; }}
    .coord-readout {{
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.96);
      color: var(--ink);
      box-shadow: 0 2px 8px rgba(31,41,51,.14);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }}
    .grid-label {{
      padding: 2px 5px;
      border: 1px solid rgba(71,84,103,.26);
      border-radius: 4px;
      background: rgba(255,255,255,.92);
      color: #1f2933;
      font-size: 11px;
      font-weight: 700;
      box-shadow: 0 1px 4px rgba(31,41,51,.14);
      white-space: nowrap;
    }}
    .facility-tooltip, .vessel-tooltip, .pipeline-tooltip {{ color: var(--ink); font-weight: 700; }}
    .vessel-marker {{ background: transparent; border: 0; }}
    .ship-icon {{
      display: grid;
      place-items: center;
      width: 34px;
      height: 26px;
      border: 2px solid var(--ship-color);
      border-radius: 8px 16px 16px 8px;
      background: #ffffff;
      color: var(--ship-color);
      font-weight: 800;
      font-size: 11px;
      box-shadow: 0 2px 7px rgba(31,41,51,.22);
    }}
    .ship-icon.selected {{ background: color-mix(in srgb, var(--ship-color) 16%, #ffffff); transform: scale(1.14); }}
    .chart-panel {{ display: grid; grid-template-rows: 38px minmax(0, 1fr); border-top: 1px solid var(--line); min-height: 0; }}
    .chart-toolbar {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      padding: 6px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    #chartMetric {{
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    #chartCanvas {{ width: 100%; height: 100%; min-height: 0; display: block; }}
    #statusPanel {{ padding: 14px 16px; display: grid; gap: 12px; }}
    .kv {{ display: grid; grid-template-columns: 130px 1fr; gap: 8px; font-size: 14px; }}
    .kv span:first-child {{ color: var(--muted); }}
    pre {{
      margin: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      overflow: auto;
      max-height: 230px;
      font-size: 12px;
    }}
    .violations {{ display: grid; gap: 8px; }}
    .violation {{ border-left: 3px solid var(--warn); padding: 8px 10px; background: #fff8eb; border-radius: 6px; }}
    @media (max-width: 1120px) {{
      main {{ grid-template-columns: 1fr; }}
      input[type="range"] {{ width: 100%; }}
      .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{escaped_title}</h1>
      <div id="timeLabel">Hour 0</div>
    </div>
    <div class="toolbar">
      <button id="playPause">Play</button>
      <button id="prevStep">Prev</button>
      <input id="timeline" type="range" min="0" max="0" step="1" value="0" aria-label="Simulation hour">
      <button id="nextStep">Next</button>
      <label class="playback-control" for="playbackDelayMs">
        <input id="playbackDelayMs" type="number" min="100" max="5000" step="100" value="200">
        ms/frame
      </label>
    </div>
  </header>
  <main>
    <aside>
      <div class="panel-title">Components</div>
      <div id="componentList"></div>
    </aside>
    <section class="workspace">
      <div class="map-panel">
        <div class="summary">
          <div class="metric"><div class="metric-label">Total inventory</div><div id="totalInventory" class="metric-value">0 t</div></div>
          <div class="metric"><div class="metric-label">Stored</div><div id="injectedTotal" class="metric-value">0 t</div></div>
          <div class="metric"><div class="metric-label">Vented</div><div id="ventedTotal" class="metric-value">0 t</div></div>
          <div class="metric"><div class="metric-label">Active flows</div><div id="activeFlows" class="metric-value">0</div></div>
          <div class="metric"><div class="metric-label">Violations</div><div id="violationCount" class="metric-value">0</div></div>
        </div>
        <div id="networkCanvas" role="img" aria-label="Interactive geographic CCS network map"></div>
      </div>
      <div class="chart-panel">
        <label class="chart-toolbar" for="chartMetric">
          Metric
          <select id="chartMetric">
            <option value="inventory">Inventory</option>
            <option value="pressure">Pressure</option>
            <option value="pressure_margin">Pressure margin</option>
            <option value="injection_rate">Injection rate</option>
          </select>
        </label>
        <canvas id="chartCanvas" width="980" height="220"></canvas>
      </div>
    </section>
    <aside>
      <div class="panel-title">Selected Status</div>
      <div id="statusPanel"></div>
    </aside>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    window.__SIM_DATA__ = {data_json};
  </script>
  <script>
    const data = window.__SIM_DATA__;
    const frames = data.frames;
    const entityOrder = Object.keys(frames[0].entities);
    const timeline = document.getElementById("timeline");
    const playPause = document.getElementById("playPause");
    const prevStep = document.getElementById("prevStep");
    const nextStep = document.getElementById("nextStep");
    const playbackDelayMs = document.getElementById("playbackDelayMs");
    const componentList = document.getElementById("componentList");
    const statusPanel = document.getElementById("statusPanel");
    const mapElement = document.getElementById("networkCanvas");
    const chartMetric = document.getElementById("chartMetric");
    const chart = document.getElementById("chartCanvas");
    const ctx = chart.getContext("2d");
    const componentGroups = [
      {{id: "emitter", label: "Emitter"}},
      {{id: "ship", label: "Ship"}},
      {{id: "pipeline", label: "Pipeline"}},
      {{id: "intermediateStorage", label: "Intermediate Storage"}},
      {{id: "storage", label: "Storage"}},
    ];
    const collapsedGroups = new Set();
    const vesselPalette = ["#0f766e", "#b42318", "#2563eb", "#7c2d12", "#6d28d9", "#15803d"];
    const vesselIds = entityOrder.filter(id => frames[0].entities[id]?.type === "Vessel");
    const vesselColors = Object.fromEntries(vesselIds.map((id, index) => [id, vesselPalette[index % vesselPalette.length]]));
    const metricDefinitions = {{
      inventory: "Inventory",
      capture_rate: "Capture rate",
      vent_rate: "Vent rate",
      flow_rate: "Flow rate",
      pressure: "Pressure",
      pressure_margin: "Pressure margin",
      injection_rate: "Injection rate"
    }};
    let frameIndex = 0;
    let selected = "brevik";
    let timer = null;
    let leafletMap = null;
    let graticuleLayer = null;
    let routeLayer = null;
    let pipelineLayer = null;
    let injectionLayer = null;
    let facilityLayer = null;
    let vesselLayer = null;
    let legendControl = null;
    let coordinateReadoutControl = null;
    let hoveredChartIndex = null;

    timeline.max = String(frames.length - 1);

    function capacityOf(entity) {{
      return limitBasisOf(entity).value;
    }}

    function limitBasisOf(entity) {{
      const p = entity.parameters || {{}};
      if (p.buffer_capacity_t) return {{label: "Buffer", value: p.buffer_capacity_t, unit: "t", showInventory: true}};
      if (p.capacity_t) return {{label: "Cargo", value: p.capacity_t, unit: "t", showInventory: true}};
      if (p.storage_capacity_t) return {{label: "Storage", value: p.storage_capacity_t, unit: "t", showInventory: true}};
      if (p.max_flow_tph) return {{label: "Flow limit", value: p.max_flow_tph, unit: "t/h", showInventory: false}};
      if (p.max_injection_tph) return {{label: "Injection", value: p.max_injection_tph, unit: "t/h", showInventory: false}};
      return {{label: "Inventory", value: 1, unit: "t", showInventory: true}};
    }}

    function formatTonnes(value) {{
      return `${{formatTonneNumber(value)}} t`;
    }}

    function formatTonneNumber(value) {{
      return Math.round(value).toLocaleString();
    }}

    function formatBar(value) {{
      return `${{Number(value).toFixed(2)}} bar`;
    }}

    function formatSignedBar(value) {{
      const numeric = Number(value);
      const sign = numeric >= 0 ? "+" : "";
      return `${{sign}}${{numeric.toFixed(2)}} bar`;
    }}

    function formatRate(value) {{
      return `${{Number(value || 0).toFixed(1)}} t/h`;
    }}

    function formatPercent(value) {{
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function formatInventoryLimit(entity) {{
      const basis = limitBasisOf(entity);
      if (!basis.showInventory) {{
        return `${{basis.label}} ${{formatTonneNumber(basis.value)}} ${{basis.unit}}`;
      }}
      return `${{basis.label}} ${{formatTonneNumber(entity.inventory_t || 0)}} / ${{formatTonneNumber(basis.value)}} ${{basis.unit}}`;
    }}

    function componentSummary(entity) {{
      if (entity.type === "Emitter") {{
        return {{
          primary: formatInventoryLimit(entity),
          secondary: `Capture ${{formatRate(entity.capture_rate_tph || 0)}} / Vent ${{formatRate(entity.vent_rate_tph || 0)}}`
        }};
      }}
      const pressureSummary = pressureSummaryForEntity(entity);
      if (pressureSummary) return pressureSummary;
      return {{
        primary: formatInventoryLimit(entity),
        secondary: ""
      }};
    }}

    function componentSecondaryLine(entity, storageTargetText) {{
      return `${{entity.type}}${{storageTargetText}}`;
    }}

    function pressureSummaryForEntity(entity) {{
      if (entity.type === "InjectionWell" && entity.bottomhole_pressure_bar !== undefined) {{
        const delta = entity.bottomhole_pressure_delta_bar ?? 0;
        const rate = entity.line_source_rate_tph ?? 0;
        return {{
          primary: `BHP ${{formatBar(entity.bottomhole_pressure_bar)}}`,
          secondary: `Delta ${{formatSignedBar(delta)}} / Rate ${{formatRate(rate)}}`
        }};
      }}
      if (entity.type === "Reservoir" && entity.pressure_bar !== undefined) {{
        const margin = entity.pressure_margin_bar ?? 0;
        const fill = entity.fill_fraction ?? 0;
        return {{
          primary: `P ${{formatBar(entity.pressure_bar)}}`,
          secondary: `Margin ${{formatBar(margin)}} / Fill ${{formatPercent(fill)}}`
        }};
      }}
      return null;
    }}

    function categoryForEntity(entity) {{
      if (entity.type === "Emitter") return "emitter";
      if (entity.type === "Vessel") return "ship";
      if (entity.type === "Terminal") return "intermediateStorage";
      if (["Pipeline", "SubseaManifold"].includes(entity.type)) return "pipeline";
      if (entity.type === "InjectionWell") return "storage";
      return "storage";
    }}

    function storageTargetForEntity(id) {{
      return data.storage_targets?.[id] || "";
    }}

    function vesselColor(id) {{
      return vesselColors[id] || "#0f766e";
    }}

    function availableMetricsForEntity(entity) {{
      if (entity.type === "Emitter") return ["inventory", "capture_rate", "vent_rate"];
      if (entity.type === "Pipeline") return ["inventory", "flow_rate"];
      if (entity.type === "InjectionWell") return ["inventory", "pressure", "injection_rate"];
      if (entity.type === "Reservoir") return ["inventory", "pressure", "pressure_margin", "injection_rate"];
      return ["inventory"];
    }}

    function syncChartMetricOptions(entity) {{
      const availableMetrics = availableMetricsForEntity(entity || {{}});
      const currentMetric = chartMetric.value;
      chartMetric.innerHTML = "";
      availableMetrics.forEach(metric => {{
        const option = document.createElement("option");
        option.value = metric;
        option.textContent = metricDefinitions[metric] || metric;
        chartMetric.appendChild(option);
      }});
      chartMetric.value = availableMetrics.includes(currentMetric) ? currentMetric : availableMetrics[0];
    }}

    function componentColor(id, entity) {{
      if (entity.type === "Vessel") return vesselColor(id);
      if (entity.type === "Emitter") return "#167c80";
      if (["Pipeline", "SubseaManifold"].includes(entity.type)) return "#dc2626";
      if (entity.type === "Terminal") return "#b45309";
      return "#64748b";
    }}

    function vesselInitials(id) {{
      return id.split("_").map(part => part.slice(0, 1).toUpperCase()).slice(0, 2).join("");
    }}

    function playbackDelay() {{
      const value = Number(playbackDelayMs.value);
      return Number.isFinite(value) ? Math.max(100, Math.min(5000, value)) : 700;
    }}

    function render() {{
      const frame = frames[frameIndex];
      timeline.value = String(frameIndex);
      document.getElementById("timeLabel").textContent = `Hour ${{frame.time_h}} / ${{frames[frames.length - 1].time_h}}`;
      renderMetrics(frame);
      renderComponentList(frame);
      renderStatus(frame);
      syncChartMetricOptions(frame.entities[selected]);
      renderChart();
      renderMap(frame);
    }}

    function renderMetrics(frame) {{
      const entities = Object.values(frame.entities);
      const total = entities.reduce((sum, e) => sum + (e.inventory_t || 0), 0);
      const injected = entities.filter(e => e.type === "Reservoir" || e.type === "InjectionWell").reduce((sum, e) => sum + (e.inventory_t || 0), 0);
      const vented = entities.filter(e => e.type === "Emitter").reduce((sum, e) => sum + (e.cumulative_vent_t || 0), 0);
      const active = Object.values(frame.flows_t || {{}}).filter(v => v > 0).length;
      document.getElementById("totalInventory").textContent = formatTonnes(total);
      document.getElementById("injectedTotal").textContent = formatTonnes(injected);
      document.getElementById("ventedTotal").textContent = formatTonnes(vented);
      document.getElementById("activeFlows").textContent = String(active);
      document.getElementById("violationCount").textContent = String((frame.violations || []).length);
    }}

    function renderComponentList(frame) {{
      componentList.innerHTML = "";
      componentGroups.forEach(group => {{
        const ids = entityOrder.filter(id => frame.entities[id] && categoryForEntity(frame.entities[id]) === group.id);
        if (!ids.length) return;
        const groupElement = document.createElement("div");
        groupElement.className = "component-group";
        const toggle = document.createElement("button");
        toggle.className = "component-group-toggle";
        toggle.type = "button";
        toggle.onclick = () => {{
          if (collapsedGroups.has(group.id)) collapsedGroups.delete(group.id); else collapsedGroups.add(group.id);
          renderComponentList(frame);
        }};
        toggle.innerHTML = `<span>${{collapsedGroups.has(group.id) ? "+" : "-"}} ${{group.label}}</span><span class="component-group-count">${{ids.length}}</span>`;
        groupElement.appendChild(toggle);
        if (!collapsedGroups.has(group.id)) {{
          const items = document.createElement("div");
          items.className = "component-group-items";
          ids.forEach(id => {{
            const entity = frame.entities[id];
            const pct = Math.max(0, Math.min(100, ((entity.inventory_t || 0) / capacityOf(entity)) * 100));
            const color = componentColor(id, entity);
            const summary = componentSummary(entity);
            const shipSwatch = entity.type === "Vessel" ? `<span class="ship-color-swatch" style="--ship-color:${{vesselColor(id)}}"></span>` : "";
            const storageTarget = storageTargetForEntity(id);
            const storageTargetText = storageTarget ? ` · Storage: ${{storageTarget}}` : "";
            const button = document.createElement("button");
            button.className = `component${{id === selected ? " active" : ""}}`;
            button.type = "button";
            button.style.setProperty("--component-color", color);
            button.onclick = () => {{ selected = id; render(); }};
            button.innerHTML = `
              <div class="component-name"><span>${{id}}</span><span class="component-capacity">${{summary.primary}}</span></div>
              <div class="component-type-row">${{shipSwatch}}<span>${{componentSecondaryLine(entity, storageTargetText)}}</span></div>
              ${{summary.secondary ? `<div class="component-summary">${{summary.secondary}}</div>` : ""}}
              <div class="bar"><span style="width:${{pct}}%"></span></div>
            `;
            items.appendChild(button);
          }});
          groupElement.appendChild(items);
        }}
        componentList.appendChild(groupElement);
      }});
    }}

    function renderMap(frame) {{
      initMap();
      routeLayer.clearLayers();
      pipelineLayer.clearLayers();
      injectionLayer.clearLayers();
      facilityLayer.clearLayers();
      vesselLayer.clearLayers();
      drawRoutes(frame);
      drawReturnRoutes(frame);
      drawPipelines(frame);
      drawInjectionLinks(frame);
      drawFacilities(frame);
      drawVessels(frame);
    }}

    function initMap() {{
      if (leafletMap) {{
        leafletMap.invalidateSize();
        return;
      }}
      leafletMap = L.map("networkCanvas", {{
        zoomControl: true,
        scrollWheelZoom: true,
        attributionControl: true
      }});
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 12,
        attribution: "&copy; OpenStreetMap contributors"
      }}).addTo(leafletMap);
      leafletMap.createPane("graticulePane");
      leafletMap.getPane("graticulePane").style.zIndex = 430;
      leafletMap.getPane("graticulePane").style.pointerEvents = "none";
      graticuleLayer = L.layerGroup([], {{pane: "graticulePane"}}).addTo(leafletMap);
      routeLayer = L.layerGroup().addTo(leafletMap);
      pipelineLayer = L.layerGroup().addTo(leafletMap);
      injectionLayer = L.layerGroup().addTo(leafletMap);
      facilityLayer = L.layerGroup().addTo(leafletMap);
      vesselLayer = L.layerGroup().addTo(leafletMap);
      const bounds = L.latLngBounds([]);
      Object.values(data.map.locations).forEach(location => bounds.extend([location.lat, location.lon]));
      Object.values(data.map.routes).forEach(route => {{
        route.coordinates.forEach(([lat, lon]) => bounds.extend([lat, lon]));
      }});
      (data.map.pipeline_segments || []).forEach(segment => {{
        segment.coordinates.forEach(([lat, lon]) => bounds.extend([lat, lon]));
      }});
      (data.map.injection_links || []).forEach(link => {{
        link.coordinates.forEach(([lat, lon]) => bounds.extend([lat, lon]));
      }});
      leafletMap.fitBounds(bounds.pad(0.15));
      drawLatLonGrid();
      leafletMap.on("moveend zoomend", drawLatLonGrid);
      addMapLegend();
      addCoordinateReadout();
      setTimeout(() => leafletMap.invalidateSize(), 0);
    }}

    function addMapLegend() {{
      if (legendControl) return;
      legendControl = L.control({{position: "bottomleft"}});
      legendControl.onAdd = () => {{
        const div = L.DomUtil.create("div", "map-legend");
        div.innerHTML = `
          <div class="legend-row"><span class="legend-line"></span><span>Outbound voyage</span></div>
          <div class="legend-row"><span class="legend-line return"></span><span>Return leg, same corridor assumed</span></div>
          <div class="legend-row"><span class="legend-line pipeline"></span><span>Offshore CO2 pipeline</span></div>
          <div class="legend-row"><span class="legend-line subsea"></span><span>Subsea connection topology</span></div>
          <div class="legend-row"><span class="legend-line injection"></span><span>Geologic injection target</span></div>
        `;
        return div;
      }};
      legendControl.addTo(leafletMap);
    }}

    function addCoordinateReadout() {{
      if (coordinateReadoutControl) return;
      coordinateReadoutControl = L.control({{position: "bottomright"}});
      coordinateReadoutControl.onAdd = () => {{
        const div = L.DomUtil.create("div", "coord-readout");
        div.textContent = "Lat/Lon grid";
        leafletMap.on("mousemove", event => {{
          div.textContent = `${{event.latlng.lat.toFixed(5)}}°N, ${{event.latlng.lng.toFixed(5)}}°E`;
        }});
        leafletMap.on("mouseout", () => {{
          div.textContent = "Lat/Lon grid";
        }});
        return div;
      }};
      coordinateReadoutControl.addTo(leafletMap);
    }}

    function drawRoutes(frame) {{
      Object.values(data.map.routes).forEach(route => {{
        const active = route.vessel_id === selected;
        L.polyline(route.coordinates, {{
          color: active ? "#167c80" : "#2563eb",
          weight: active ? 5 : 3,
          opacity: active ? 0.95 : 0.55
        }})
          .bindTooltip(`${{route.vessel_id}} outbound<br>${{route.distance_km}} km`, {{className: "vessel-tooltip"}})
          .addTo(routeLayer);
      }});
    }}

    function drawReturnRoutes(frame) {{
      Object.values(data.map.routes).forEach(route => {{
        const active = route.vessel_id === selected;
        L.polyline(route.return_coordinates || route.coordinates.slice().reverse(), {{
          color: active ? "#b7791f" : "#ca8a04",
          weight: active ? 4 : 3,
          opacity: active ? 0.9 : 0.62,
          dashArray: "5 9",
          dashOffset: "6"
        }})
          .bindTooltip(`${{route.vessel_id}} Return leg<br>same corridor assumed<br>${{route.distance_km}} km`, {{className: "vessel-tooltip"}})
          .addTo(routeLayer);
      }});
    }}

    function drawPipelines(frame) {{
      const hasPipelineFlow = Object.entries(frame.flows_t || {{}})
        .some(([key, value]) => key.includes("oygarden_pipeline") && value > 0);
      (data.map.pipeline_segments || []).forEach(segment => {{
        const componentId = segment.component_id || segment.pipeline_id;
        const segmentEntity = frame.entities[componentId] || {{inventory_t: 0, parameters: {{max_flow_tph: 0}}}};
        const active = selected === componentId || selected === segment.source || selected === segment.target;
        L.polyline(segment.coordinates, {{
          color: segment.color,
          weight: segment.style === "subsea_connection" ? (active ? 5 : 3) : (active ? 7 : 5),
          opacity: hasPipelineFlow || active ? 0.95 : 0.72,
          dashArray: segment.style === "subsea_connection" ? "7 7" : null
        }})
          .bindTooltip(`${{segment.label}}<br>${{formatInventoryLimit(segmentEntity)}}`, {{className: "pipeline-tooltip"}})
          .on("click", () => {{ selected = componentId; render(); }})
          .addTo(pipelineLayer);
      }});
    }}

    function drawInjectionLinks(frame) {{
      (data.map.injection_links || []).forEach(link => {{
        const active = selected === link.component_id || selected === link.source || selected === link.target;
        L.polyline(link.coordinates, {{
          color: link.color,
          weight: active ? 3 : 2,
          opacity: active ? 0.86 : 0.48,
          dashArray: "2 8",
          lineCap: "round"
        }})
          .bindTooltip(`${{escapeHtml(link.label)}}<br>Geologic injection relation, not a pipeline`, {{className: "pipeline-tooltip"}})
          .on("click", () => {{ selected = link.source; render(); }})
          .addTo(injectionLayer);
      }});
    }}

    function drawLatLonGrid() {{
      if (!leafletMap || !graticuleLayer) return;
      graticuleLayer.clearLayers();
      const bounds = leafletMap.getBounds();
      const south = bounds.getSouth();
      const north = bounds.getNorth();
      const west = bounds.getWest();
      const east = bounds.getEast();
      const step = gridStep(Math.max(north - south, east - west));
      const decimals = step < 1 ? 2 : 0;
      const latitudeLabelLon = west + (east - west) * 0.012;
      const longitudeLabelLat = south + (north - south) * 0.035;
      for (let lat = Math.ceil(south / step) * step; lat <= north; lat += step) {{
        const value = Number(lat.toFixed(decimals));
        L.polyline([[value, west], [value, east]], {{
          color: "#334155",
          weight: 1.4,
          opacity: 0.62,
          dashArray: "4 6",
          interactive: false
        }}).addTo(graticuleLayer);
        L.marker([value, latitudeLabelLon], {{
          interactive: false,
          icon: L.divIcon({{
            className: "grid-label",
            html: `${{value.toFixed(decimals)}}°N`,
            iconSize: [58, 18],
            iconAnchor: [0, 9]
          }})
        }}).addTo(graticuleLayer);
      }}
      for (let lon = Math.ceil(west / step) * step; lon <= east; lon += step) {{
        const value = Number(lon.toFixed(decimals));
        L.polyline([[south, value], [north, value]], {{
          color: "#334155",
          weight: 1.4,
          opacity: 0.62,
          dashArray: "4 6",
          interactive: false
        }}).addTo(graticuleLayer);
        L.marker([longitudeLabelLat, value], {{
          interactive: false,
          icon: L.divIcon({{
            className: "grid-label",
            html: `${{value.toFixed(decimals)}}°E`,
            iconSize: [58, 18],
            iconAnchor: [29, 20]
          }})
        }}).addTo(graticuleLayer);
      }}
    }}

    function gridStep(spanDegrees) {{
      if (spanDegrees <= 1) return 0.1;
      if (spanDegrees <= 3) return 0.25;
      if (spanDegrees <= 8) return 0.5;
      if (spanDegrees <= 16) return 1;
      return 2;
    }}

    function drawFacilities(frame) {{
      Object.entries(data.map.locations).forEach(([id, location]) => {{
        if (id === "oygarden_pipeline") return;
        const entity = frame.entities[id];
        if (!entity) {{
          L.circleMarker([location.lat, location.lon], {{
            radius: id === selected ? 8 : 6,
            color: "#7c3aed",
            weight: id === selected ? 3 : 2,
            fillColor: "#ffffff",
            fillOpacity: 0.95,
            dashArray: "3 5"
          }})
            .bindTooltip(`${{escapeHtml(location.label || id)}}<br>${{Number(location.lat).toFixed(5)}}, ${{Number(location.lon).toFixed(5)}}<br>Reference coordinate`, {{className: "facility-tooltip"}})
            .on("click", () => {{ selected = id; render(); }})
            .addTo(facilityLayer);
          return;
        }}
        const pct = Math.max(0, Math.min(1, (entity.inventory_t || 0) / capacityOf(entity)));
        const marker = L.circleMarker([location.lat, location.lon], {{
          radius: id === selected ? 10 : 7,
          color: id === selected ? "#167c80" : "#475467",
          weight: id === selected ? 4 : 2,
          fillColor: pct > .85 ? "#fff0e6" : "#ffffff",
          fillOpacity: 0.95
        }})
          .bindTooltip(`${{location.label || id}}<br>${{location.lat.toFixed(3)}}, ${{location.lon.toFixed(3)}}`, {{className: "facility-tooltip"}})
          .on("click", () => {{ selected = id; render(); }})
          .addTo(facilityLayer);
      }});
    }}

    function drawVessels(frame) {{
      Object.entries(frame.vessel_positions || {{}}).forEach(([id, position]) => {{
        const entity = frame.entities[id];
        const cargo = entity ? entity.inventory_t || 0 : 0;
        const selectedShip = id === selected;
        const color = vesselColor(id);
        const berthLine = position.at_berth ? `<br>${{escapeHtml(position.berth_label || "At berth")}}` : "";
        const icon = L.divIcon({{
          className: "vessel-marker",
          html: `<div class="ship-icon${{selectedShip ? " selected" : ""}}" style="--ship-color:${{color}}">${{vesselInitials(id)}}</div>`,
          iconSize: [38, 30],
          iconAnchor: [19, 15]
        }});
        L.marker([position.lat, position.lon], {{icon}})
          .bindTooltip(`${{escapeHtml(id)}}<br>${{escapeHtml(position.leg_label || "Voyage")}}${{berthLine}}<br>Cargo ${{formatTonnes(cargo)}}<br>${{Number(position.lat).toFixed(3)}}, ${{Number(position.lon).toFixed(3)}}`, {{className: "vessel-tooltip"}})
          .on("click", () => {{ selected = id; render(); }})
          .addTo(vesselLayer);
      }});
    }}

    function renderStatus(frame) {{
      const entity = frame.entities[selected] || Object.values(frame.entities)[0];
      selected = Object.keys(frame.entities).find(id => frame.entities[id] === entity) || selected;
      const inbound = Object.entries(frame.flows_t || {{}}).filter(([k]) => k.endsWith(`->${{selected}}`));
      const outbound = Object.entries(frame.flows_t || {{}}).filter(([k]) => k.startsWith(`${{selected}}->`));
      const entityViolations = (frame.violations || []).filter(v => v.entity_id === selected);
      const location = data.map.locations[selected] || frame.vessel_positions?.[selected];
      const coordLine = location ? `${{Number(location.lat).toFixed(3)}}, ${{Number(location.lon).toFixed(3)}}` : "n/a";
      const pressureLine = entity.pressure_bar === undefined ? "" : `<div class="kv"><span>Pressure</span><strong>${{Number(entity.pressure_bar).toFixed(2)}} bar</strong></div>`;
      const storageTarget = storageTargetForEntity(selected);
      const storageTargetLine = storageTarget ? `<div class="kv"><span>Storage</span><strong>${{escapeHtml(storageTarget)}}</strong></div>` : "";
      const pressureStatus = renderPressureStatus(entity);
      statusPanel.innerHTML = `
        <div class="kv"><span>Component</span><strong>${{selected}}</strong></div>
        <div class="kv"><span>Type</span><strong>${{entity.type}}</strong></div>
        <div class="kv"><span>Coordinate</span><strong>${{coordLine}}</strong></div>
        <div class="kv"><span>Inventory</span><strong>${{formatTonnes(entity.inventory_t || 0)}}</strong></div>
        <div class="kv"><span>Limit</span><strong>${{formatInventoryLimit(entity)}}</strong></div>
        ${{storageTargetLine}}
        ${{pressureLine}}
        ${{pressureStatus}}
        <div class="kv"><span>Inbound</span><strong>${{inbound.length ? inbound.map(([k,v]) => `${{k}}: ${{formatTonnes(v)}}`).join("<br>") : "None"}}</strong></div>
        <div class="kv"><span>Outbound</span><strong>${{outbound.length ? outbound.map(([k,v]) => `${{k}}: ${{formatTonnes(v)}}`).join("<br>") : "None"}}</strong></div>
        <div>
          <div class="panel-title" style="padding-left:0;border:0">Parameters</div>
          <pre>${{escapeHtml(JSON.stringify(entity.parameters, null, 2))}}</pre>
        </div>
        <div>
          <div class="panel-title" style="padding-left:0;border:0">Violations</div>
          <div class="violations">${{entityViolations.length ? entityViolations.map(v => `<div class="violation"><strong>${{escapeHtml(v.violation_type)}}</strong><br>${{escapeHtml(v.message)}}<br>${{formatTonnes(v.magnitude_t)}} clipped</div>`).join("") : "None"}}</div>
        </div>
      `;
    }}

    function renderPressureStatus(entity) {{
      if (entity.type === "InjectionWell" && entity.bottomhole_pressure_bar !== undefined) {{
        const interference = entity.line_source_interference_delta_bar === undefined ? "" : `<div class="kv"><span>Interference delta</span><strong>${{formatSignedBar(entity.line_source_interference_delta_bar)}}</strong></div>`;
        return `
          <div class="kv"><span>Bottomhole pressure</span><strong>${{formatBar(entity.bottomhole_pressure_bar)}}</strong></div>
          <div class="kv"><span>BHP delta</span><strong>${{formatSignedBar(entity.bottomhole_pressure_delta_bar || 0)}}</strong></div>
          <div class="kv"><span>Injection rate</span><strong>${{formatRate(entity.line_source_rate_tph || 0)}}</strong></div>
          ${{interference}}
        `;
      }}
      if (entity.type === "Reservoir" && entity.pressure_bar !== undefined) {{
        const radiusLines = Object.entries(entity.line_source_pressure_bar_by_radius_m || {{}})
          .map(([radius, pressure]) => `<div class="kv"><span>Pressure @${{Number(radius).toLocaleString()}}m</span><strong>${{formatBar(pressure)}}</strong></div>`)
          .join("");
        return `
          <div class="kv"><span>Reservoir pressure</span><strong>${{formatBar(entity.pressure_bar)}}</strong></div>
          <div class="kv"><span>Pressure margin</span><strong>${{formatBar(entity.pressure_margin_bar || 0)}}</strong></div>
          <div class="kv"><span>Fill fraction</span><strong>${{formatPercent(entity.fill_fraction || 0)}}</strong></div>
          ${{radiusLines}}
        `;
      }}
      return "";
    }}

    function resizeChartCanvas() {{
      const ratio = window.devicePixelRatio || 1;
      const rect = chart.getBoundingClientRect();
      const width = Math.max(360, Math.floor(rect.width));
      const height = Math.max(220, Math.floor(rect.height));
      if (chart.width !== Math.floor(width * ratio) || chart.height !== Math.floor(height * ratio)) {{
        chart.width = Math.floor(width * ratio);
        chart.height = Math.floor(height * ratio);
      }}
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {{width, height}};
    }}

    function renderChart() {{
      const {{width, height}} = resizeChartCanvas();
      const selectedEntity = frames[frameIndex].entities[selected] || {{}};
      const series = metricSeries(selected, chartMetric.value);
      const values = series.values;
      const capacity = series.referenceValue;
      const maxValue = niceChartMax(Math.max(1, capacity || 0, ...values));
      const plot = {{left: 64, top: 34, right: width - 28, bottom: height - 42}};
      const color = componentColor(selected, selectedEntity);

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      drawChartHeader(width, series);
      drawChartGrid(plot, maxValue, series.formatter);
      drawCapacityReference(plot, maxValue, capacity, series.referenceLabel, series.formatter);
      drawInventoryArea(plot, values, maxValue, color);
      const markerIndex = hoveredChartIndex ?? frameIndex;
      drawCurrentValueMarker(plot, values, maxValue, markerIndex, color);
      if (hoveredChartIndex !== null) drawChartTooltip(plot, series, values, maxValue, hoveredChartIndex, width);
      drawChartAxes(plot, maxValue);
    }}

    function metricSeries(entityId, metric) {{
      const values = frames.map(frame => metricValueForEntity(frame.entities[entityId], metric));
      const selectedEntity = frames[frameIndex].entities[entityId] || {{}};
      return {{
        metric,
        label: metricDefinitions[metric] || "Inventory",
        values,
        currentValue: values[frameIndex] || 0,
        formatter: value => formatMetricValue(value, metric),
        referenceValue: metric === "inventory" ? capacityOf(selectedEntity) : pressureReferenceValue(selectedEntity, metric),
        referenceLabel: metric === "inventory" ? "capacity" : pressureReferenceLabel(metric)
      }};
    }}

    function metricValueForEntity(entity, metric) {{
      if (!entity) return 0;
      if (metric === "pressure") {{
        if (entity.bottomhole_pressure_bar !== undefined) return Number(entity.bottomhole_pressure_bar);
        if (entity.pressure_bar !== undefined) return Number(entity.pressure_bar);
        return 0;
      }}
      if (metric === "pressure_margin") {{
        if (entity.pressure_margin_bar !== undefined) return Number(entity.pressure_margin_bar);
        return 0;
      }}
      if (metric === "injection_rate") {{
        if (entity.line_source_rate_tph !== undefined) return Number(entity.line_source_rate_tph);
        if (entity.line_source_well_rates_tph) {{
          return Object.values(entity.line_source_well_rates_tph).reduce((sum, value) => sum + Number(value || 0), 0);
        }}
        return 0;
      }}
      if (metric === "capture_rate") return Number(entity.capture_rate_tph || 0);
      if (metric === "vent_rate") return Number(entity.vent_rate_tph || 0);
      if (metric === "flow_rate") return Number(entity.pipeline_flow_rate_tph || 0);
      return Number(entity.inventory_t || 0);
    }}

    function pressureReferenceValue(entity, metric) {{
      const params = entity.parameters || {{}};
      if (metric === "pressure" && params.max_pressure_bar) return Number(params.max_pressure_bar);
      if ((metric === "capture_rate" || metric === "vent_rate") && params.nominal_capture_tph) return Number(params.nominal_capture_tph);
      if (metric === "flow_rate" && params.max_flow_tph) return Number(params.max_flow_tph);
      return null;
    }}

    function pressureReferenceLabel(metric) {{
      if (metric === "pressure") return "max pressure";
      if (metric === "capture_rate" || metric === "vent_rate") return "nominal capture";
      if (metric === "flow_rate") return "max flow";
      return "";
    }}

    function formatMetricValue(value, metric) {{
      if (metric === "pressure" || metric === "pressure_margin") return formatBar(value);
      if (metric === "injection_rate" || metric === "capture_rate" || metric === "vent_rate" || metric === "flow_rate") return formatRate(value);
      return formatChartValue(value);
    }}

    function drawChartHeader(width, series) {{
      ctx.fillStyle = "#1f2933";
      ctx.font = "700 13px Inter, Arial";
      ctx.fillText(`${{selected}} ${{series.label}}`, 18, 20);
      ctx.fillStyle = "#667085";
      ctx.font = "12px Inter, Arial";
      ctx.textAlign = "right";
      ctx.fillText(`Current ${{series.formatter(series.currentValue)}}`, width - 18, 20);
      ctx.textAlign = "left";
    }}

    function drawChartGrid(plot, maxValue, formatter) {{
      ctx.save();
      ctx.strokeStyle = "#e6ebf1";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#667085";
      ctx.font = "11px Inter, Arial";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let tick = 0; tick <= 4; tick += 1) {{
        const value = (maxValue / 4) * tick;
        const y = yForValue(value, plot, maxValue);
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.right, y);
        ctx.stroke();
        ctx.fillText(formatter(value), plot.left - 10, y);
      }}
      ctx.restore();
    }}

    function drawCapacityReference(plot, maxValue, referenceValue, referenceLabel, formatter) {{
      if (!referenceValue || referenceValue >= maxValue) return;
      const y = yForValue(referenceValue, plot, maxValue);
      ctx.save();
      ctx.strokeStyle = "#b7791f";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8a5a12";
      ctx.font = "11px Inter, Arial";
      ctx.fillText(referenceLabel || formatter(referenceValue), plot.right - 82, y - 6);
      ctx.restore();
    }}

    function drawInventoryArea(plot, values, maxValue, color) {{
      const points = chartPoints(plot, values, maxValue);
      const gradient = ctx.createLinearGradient(0, plot.top, 0, plot.bottom);
      gradient.addColorStop(0, "rgba(22, 124, 128, 0.22)");
      gradient.addColorStop(1, "rgba(22, 124, 128, 0.02)");

      ctx.save();
      ctx.beginPath();
      points.forEach(([x, y], index) => {{
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.lineTo(plot.right, plot.bottom);
      ctx.lineTo(plot.left, plot.bottom);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      points.forEach(([x, y], index) => {{
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.strokeStyle = color || "#167c80";
      ctx.lineWidth = 2.5;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.restore();
    }}

    function drawCurrentValueMarker(plot, values, maxValue, index, color) {{
      const x = xForIndex(index, values.length, plot);
      const y = yForValue(values[index] || 0, plot, maxValue);
      ctx.save();
      ctx.strokeStyle = "#b42318";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = color || "#167c80";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 4.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }}

    function drawChartTooltip(plot, series, values, maxValue, index, width) {{
      const x = xForIndex(index, values.length, plot);
      const y = yForValue(values[index] || 0, plot, maxValue);
      const title = `Hour ${{frames[index].time_h}}`;
      const value = series.formatter(values[index] || 0);
      ctx.save();
      ctx.font = "700 12px Inter, Arial";
      const labelWidth = Math.max(ctx.measureText(title).width, ctx.measureText(value).width);
      const boxWidth = labelWidth + 22;
      const boxHeight = 48;
      const boxX = Math.min(Math.max(x + 10, plot.left), width - boxWidth - 8);
      const boxY = Math.max(plot.top + 6, y - boxHeight - 12);
      ctx.fillStyle = "rgba(31, 41, 51, 0.92)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(title, boxX + 11, boxY + 8);
      ctx.font = "12px Inter, Arial";
      ctx.fillText(value, boxX + 11, boxY + 27);
      ctx.restore();
    }}

    function drawChartAxes(plot, maxValue) {{
      ctx.save();
      ctx.strokeStyle = "#cfd7e2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, plot.top);
      ctx.lineTo(plot.left, plot.bottom);
      ctx.lineTo(plot.right, plot.bottom);
      ctx.stroke();
      ctx.fillStyle = "#667085";
      ctx.font = "11px Inter, Arial";
      ctx.textAlign = "left";
      ctx.fillText("Hour 0", plot.left, plot.bottom + 22);
      ctx.textAlign = "right";
      ctx.fillText(`Hour ${{frames[frames.length - 1].time_h}}`, plot.right, plot.bottom + 22);
      ctx.restore();
    }}

    function chartPoints(plot, values, maxValue) {{
      return values.map((value, index) => [
        xForIndex(index, values.length, plot),
        yForValue(value, plot, maxValue)
      ]);
    }}

    function xForIndex(index, count, plot) {{
      return plot.left + (index / Math.max(1, count - 1)) * (plot.right - plot.left);
    }}

    function chartHoverIndex(event, plot) {{
      const rect = chart.getBoundingClientRect();
      const x = event.clientX - rect.left;
      if (x < plot.left || x > plot.right) return null;
      const ratio = (x - plot.left) / (plot.right - plot.left);
      return Math.max(0, Math.min(frames.length - 1, Math.round(ratio * (frames.length - 1))));
    }}

    function yForValue(value, plot, maxValue) {{
      return plot.bottom - (value / maxValue) * (plot.bottom - plot.top);
    }}

    function niceChartMax(value) {{
      const exponent = Math.floor(Math.log10(value));
      const base = Math.pow(10, exponent);
      const scaled = value / base;
      const nice = scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
      return nice * base;
    }}

    function formatChartValue(value) {{
      if (value >= 1_000_000) return `${{(value / 1_000_000).toFixed(1)}} Mt`;
      if (value >= 1_000) return `${{(value / 1_000).toFixed(1)}} kt`;
      return `${{Math.round(value).toLocaleString()}} t`;
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#039;"}}[ch]));
    }}

    timeline.addEventListener("input", event => {{ frameIndex = Number(event.target.value); render(); }});
    chartMetric.addEventListener("change", () => renderChart());
    chart.addEventListener("mousemove", event => {{
      const rect = chart.getBoundingClientRect();
      const width = Math.max(360, Math.floor(rect.width));
      const height = Math.max(220, Math.floor(rect.height));
      const plot = {{left: 64, top: 34, right: width - 28, bottom: height - 42}};
      const nextIndex = chartHoverIndex(event, plot);
      if (nextIndex !== hoveredChartIndex) {{
        hoveredChartIndex = nextIndex;
        renderChart();
      }}
    }});
    chart.addEventListener("mouseleave", () => {{
      hoveredChartIndex = null;
      renderChart();
    }});
    prevStep.addEventListener("click", () => {{ frameIndex = Math.max(0, frameIndex - 1); render(); }});
    nextStep.addEventListener("click", () => {{ frameIndex = Math.min(frames.length - 1, frameIndex + 1); render(); }});
    window.addEventListener("resize", () => {{
      if (leafletMap) leafletMap.invalidateSize();
      renderChart();
      drawLatLonGrid();
    }});
    playbackDelayMs.addEventListener("change", () => {{
      playbackDelayMs.value = String(playbackDelay());
      if (timer) {{
        clearInterval(timer);
        timer = setInterval(() => {{
          frameIndex = frameIndex >= frames.length - 1 ? 0 : frameIndex + 1;
          render();
        }}, playbackDelay());
      }}
    }});
    playPause.addEventListener("click", () => {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playPause.textContent = "Play";
      }} else {{
        playPause.textContent = "Pause";
        timer = setInterval(() => {{
          frameIndex = frameIndex >= frames.length - 1 ? 0 : frameIndex + 1;
          render();
        }}, playbackDelay());
      }}
    }});
    render();
  </script>
</body>
</html>
"""


def write_dashboard(
    path: str | Path,
    hours: int = 48,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_demo_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


def write_northern_lights_phase2_dashboard(
    path: str | Path,
    hours: int = 240,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_northern_lights_phase2_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


def write_northern_lights_phase1_plus_yara_dashboard(
    path: str | Path,
    hours: int = 72,
    action_frames: list[dict[str, dict[str, Any]]] | None = None,
    action_generator_factory: Any | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard_html(build_northern_lights_phase1_plus_yara_trajectory(hours, action_frames, action_generator_factory)),
        encoding="utf-8",
    )
    return output_path


def _build_routes(network: Any, locations: dict[str, dict[str, float | str]]) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    origin_usage: dict[str, int] = {}
    for vessel_id, entity in network.entities.items():
        if not isinstance(entity, Vessel):
            continue
        origin_id = _route_origin_for_vessel(network, vessel_id, origin_usage)
        destination_id = _route_destination_for_vessel(network, vessel_id)
        if origin_id is None or destination_id is None:
            continue
        origin_usage[origin_id] = origin_usage.get(origin_id, 0) + 1
        origin = _location_tuple(origin_id, locations)
        destination = _location_tuple(destination_id, locations)
        route = sea_route(origin, destination)
        if route.provider != "searoute":
            raise RuntimeError("Dashboard route generation requires searoute. Install it with `python -m pip install searoute`.")
        display_coordinates = _connect_route_to_facilities(route.coordinates, origin, destination)
        routes[vessel_id] = {
            "vessel_id": vessel_id,
            "origin": origin_id,
            "destination": destination_id,
            "provider": route.provider,
            "distance_km": round(route.distance_km, 2),
            "speed_knots": _vessel_speed_knots(network, vessel_id),
            "sea_coordinates": route.coordinates,
            "coordinates": display_coordinates,
            "return_coordinates": list(reversed(display_coordinates)),
            "return_policy": "same_corridor_reverse",
        }
    return routes


def _route_origin_for_vessel(network: Any, vessel_id: str, origin_usage: dict[str, int]) -> str | None:
    emitter_ids = [
        upstream_id
        for upstream_id in network.upstream_of(vessel_id)
        if isinstance(network.entities.get(upstream_id), Emitter)
    ]
    if not emitter_ids:
        return None
    return min(emitter_ids, key=lambda emitter_id: (origin_usage.get(emitter_id, 0), emitter_ids.index(emitter_id)))


def _route_destination_for_vessel(network: Any, vessel_id: str) -> str | None:
    for downstream_id in network.downstream_of(vessel_id):
        if isinstance(network.entities.get(downstream_id), Terminal):
            return downstream_id
    return None


def _vessel_speed_knots(network: Any, vessel_id: str) -> float | None:
    vessel = network.entities.get(vessel_id)
    if isinstance(vessel, Vessel) and vessel.speed_knots:
        return float(vessel.speed_knots)
    return None


def _connect_route_to_facilities(
    sea_coordinates: list[Coordinate],
    origin: Coordinate,
    destination: Coordinate,
) -> list[Coordinate]:
    coordinates = list(sea_coordinates)
    if not coordinates:
        return [origin, destination]
    if coordinates[0] != origin:
        coordinates.insert(0, origin)
    if coordinates[-1] != destination:
        coordinates.append(destination)
    coordinates = _smooth_route_corners(coordinates)
    return _densify_route(coordinates, MAX_DISPLAY_ROUTE_SEGMENT_KM)


def _smooth_route_corners(coordinates: list[Coordinate]) -> list[Coordinate]:
    if len(coordinates) < 3:
        return list(coordinates)

    smoothed = [coordinates[0]]
    for previous, current, following in zip(coordinates, coordinates[1:], coordinates[2:]):
        entry = _interpolate_coordinate(current, previous, CORNER_SMOOTHING_FRACTION)
        exit_point = _interpolate_coordinate(current, following, CORNER_SMOOTHING_FRACTION)
        smoothed.append(entry)
        for step in range(1, CORNER_SMOOTHING_STEPS + 1):
            t = step / CORNER_SMOOTHING_STEPS
            smoothed.append(_quadratic_bezier(entry, current, exit_point, t))
    smoothed.append(coordinates[-1])
    return smoothed


def _quadratic_bezier(start: Coordinate, control: Coordinate, end: Coordinate, t: float) -> Coordinate:
    lat_start, lon_start = start
    lat_control, lon_control = control
    lat_end, lon_end = end
    inverse = 1.0 - t
    return (
        inverse * inverse * lat_start + 2 * inverse * t * lat_control + t * t * lat_end,
        inverse * inverse * lon_start + 2 * inverse * t * lon_control + t * t * lon_end,
    )


def _interpolate_coordinate(start: Coordinate, end: Coordinate, fraction: float) -> Coordinate:
    lat_start, lon_start = start
    lat_end, lon_end = end
    return (lat_start + (lat_end - lat_start) * fraction, lon_start + (lon_end - lon_start) * fraction)


def _densify_route(coordinates: list[Coordinate], max_segment_km: float) -> list[Coordinate]:
    if len(coordinates) < 2:
        return list(coordinates)

    densified = [coordinates[0]]
    for start, end in zip(coordinates, coordinates[1:]):
        segment_km = route_distance_km([start, end])
        steps = max(1, math.ceil(segment_km / max_segment_km))
        lat_a, lon_a = start
        lat_b, lon_b = end
        for step in range(1, steps + 1):
            fraction = step / steps
            densified.append((lat_a + (lat_b - lat_a) * fraction, lon_a + (lon_b - lon_a) * fraction))
    return densified


def _build_pipeline_segments(network: Any, locations: dict[str, dict[str, float | str]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for pipeline_id, pipeline in network.entities.items():
        if not isinstance(pipeline, Pipeline) or not pipeline.route_coordinates:
            continue
        upstream_terminal = _first_upstream_of_type(network, pipeline_id, Terminal)
        downstream_manifold = network.downstream_of(pipeline_id)[0] if network.downstream_of(pipeline_id) else None
        if upstream_terminal is None or downstream_manifold is None:
            continue
        segments.append(
            {
                "id": "naturgassparken_to_eos_subsea_manifold",
                "pipeline_id": pipeline.entity_id,
                "component_id": pipeline.entity_id,
                "source": upstream_terminal,
                "target": downstream_manifold,
                "label": "Offshore pipeline from Naturgassparken to 31/5-7 EOS subsea manifold",
                "color": pipeline.route_color or "#ff0000",
                "coordinates": pipeline.route_coordinates,
                "length_km": pipeline.length_km,
            }
        )
    for manifold_id, entity in network.entities.items():
        if entity.__class__.__name__ != "SubseaManifold":
            continue
        for well_id in network.downstream_of(manifold_id):
            if not isinstance(network.entities.get(well_id), InjectionWell):
                continue
            source = _location_tuple(manifold_id, locations)
            target = _location_tuple(well_id, locations)
            segments.append(
                {
                    "id": f"{manifold_id}_to_{well_id}",
                    "pipeline_id": "oygarden_pipeline",
                    "component_id": manifold_id,
                    "source": manifold_id,
                    "target": well_id,
                    "label": f"{manifold_id} to {well_id}",
                    "color": "#7c3aed",
                    "style": "subsea_connection",
                    "coordinates": [source, target],
                }
            )
    return segments


def _first_upstream_of_type(network: Any, entity_id: str, entity_type: type) -> str | None:
    for upstream_id in network.upstream_of(entity_id):
        if isinstance(network.entities.get(upstream_id), entity_type):
            return upstream_id
    return None


def _build_injection_links(network: Any, locations: dict[str, dict[str, float | str]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for connection in network.connections:
        source_entity = network.entities.get(connection.source)
        target_entity = network.entities.get(connection.target)
        if not isinstance(source_entity, InjectionWell) or not isinstance(target_entity, Reservoir):
            continue
        source = _location_tuple(connection.source, locations)
        target = _location_tuple(connection.target, locations)
        links.append(
            {
                "id": f"{connection.source}_to_{connection.target}",
                "component_id": connection.target,
                "source": connection.source,
                "target": connection.target,
                "label": f"{connection.source} injection target: {connection.target}",
                "color": "#64748b",
                "relation": "injection_target",
                "style": "geologic",
                "coordinates": [source, target],
            }
        )
    return links


def _storage_targets(network: Any) -> dict[str, str]:
    targets: dict[str, str] = {}
    for connection in network.connections:
        source = network.entities.get(connection.source)
        target = network.entities.get(connection.target)
        if isinstance(source, InjectionWell) and isinstance(target, Reservoir):
            targets[connection.source] = connection.target
    return targets


def _frame_from_state(
    network: Any,
    state: Any,
    flows: dict[str, float],
    violations: list[dict[str, Any]],
    actions: dict[str, Any],
    vessel_positions: dict[str, dict[str, object]],
) -> dict[str, Any]:
    snapshot = network.snapshot(state)
    return {
        "time_h": state.time_h,
        "entities": snapshot["entities"],
        "flows_t": flows,
        "violations": violations,
        "actions": actions,
        "vessel_positions": vessel_positions,
    }


def _actions_at(
    action_frames: list[dict[str, dict[str, Any]]] | None,
    hour: int,
) -> dict[str, dict[str, Any]]:
    if action_frames is None or hour >= len(action_frames):
        return {}
    return {
        entity_id: dict(action)
        for entity_id, action in action_frames[hour].items()
    }


def _legacy_action_proposals(actions: dict[str, dict[str, Any]]) -> list[ActionProposal]:
    proposals: list[ActionProposal] = []
    for entity_id, action in actions.items():
        if "utilization" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="set_capture_utilization",
                    params={"utilization": action["utilization"]},
                )
            )
        if "load_vessel" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="load_vessel",
                    params={"vessel_id": action["load_vessel"]},
                )
            )
        if "sail_to" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="sail_to",
                    params={"destination_id": action["sail_to"]},
                )
            )
        if "unload_vessel" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="unload_vessel",
                    params={"vessel_id": action["unload_vessel"]},
                )
            )
        if "flow_tph" in action:
            proposals.append(
                ActionProposal(
                    agent_id=f"{entity_id}_legacy_agent",
                    entity_id=entity_id,
                    verb="set_flow",
                    params={"flow_tph": action["flow_tph"]},
                )
            )
    return proposals


def _dashboard_locations(locations: dict[str, dict[str, float | str]]) -> dict[str, Coordinate]:
    return {
        location_id: (float(location["lat"]), float(location["lon"]))
        for location_id, location in locations.items()
    }


def _physical_actions(
    actions: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        entity_id: {
            key: value
            for key, value in action.items()
            if key != "sail_to"
        }
        for entity_id, action in actions.items()
        if entity_id not in routes
    }


def _initial_vessel_states(routes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        vessel_id: {
            "mode": "berthed",
            "berth": route["origin"],
            "origin": route["origin"],
            "destination": route["origin"],
            "progress": 0.0,
        }
        for vessel_id, route in routes.items()
    }


def _start_vessel_voyages(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    actions: dict[str, dict[str, Any]],
) -> None:
    for vessel_id, route in routes.items():
        action = actions.get(vessel_id, {})
        destination = action.get("sail_to")
        state = vessel_states[vessel_id]
        if not destination or state["mode"] != "berthed" or destination == state["berth"]:
            continue
        if destination not in {route["origin"], route["destination"]}:
            continue
        state.update(
            {
                "mode": "sailing",
                "origin": state["berth"],
                "destination": destination,
                "berth": None,
                "progress": 0.0,
            }
        )


def _advance_vessel_voyages(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    hours: float,
) -> None:
    for vessel_id, state in vessel_states.items():
        if state["mode"] != "sailing":
            continue
        route = routes[vessel_id]
        speed_knots = route.get("speed_knots")
        distance_km = float(route.get("distance_km") or 0.0)
        if not speed_knots or distance_km <= 0.0:
            state["progress"] = 1.0
        else:
            state["progress"] = min(1.0, state["progress"] + (float(speed_knots) * 1.852 * hours) / distance_km)
        if state["progress"] >= 1.0:
            state.update(
                {
                    "mode": "berthed",
                    "berth": state["destination"],
                    "origin": state["destination"],
                    "progress": 0.0,
                }
            )


def _vessel_berths_from_states(vessel_states: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        vessel_id: str(state["berth"])
        for vessel_id, state in vessel_states.items()
        if state["mode"] == "berthed" and state.get("berth")
    }


def _vessel_positions_from_states(
    vessel_states: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, object]]:
    positions: dict[str, dict[str, object]] = {}
    for vessel_id, state in vessel_states.items():
        route = routes[vessel_id]
        if state["mode"] == "berthed":
            berth = str(state["berth"])
            lat, lon = _location_tuple(berth)
            if berth == route["destination"]:
                leg = "unloading_at_terminal"
                berth_id = f"{berth}_unloading_berth_1"
                berth_label = f"Unloading berth 1 at {berth}"
            else:
                leg = "loading_at_origin"
                berth_id = f"{berth}_loading_berth"
                berth_label = f"Loading berth at {berth}"
            leg_label = f"Berthed at {berth}"
            at_berth = True
        else:
            origin = str(state["origin"])
            destination = str(state["destination"])
            coordinates = _route_coordinates_for_leg(route, origin, destination)
            leg = "outbound_to_terminal" if destination == route["destination"] else "return_to_origin"
            leg_label = "Outbound voyage to terminal" if leg == "outbound_to_terminal" else "Return voyage to emitter"
            lat, lon = _interpolate_route(coordinates, float(state["progress"]))
            at_berth = False
            berth_id = None
            berth_label = None
        positions[vessel_id] = {
            "lat": lat,
            "lon": lon,
            "route_id": vessel_id,
            "leg": leg,
            "leg_label": leg_label,
            "at_berth": at_berth,
            "berth_id": berth_id,
            "berth_label": berth_label,
        }
    return positions


def _route_coordinates_for_leg(route: dict[str, Any], origin: str, destination: str) -> list[Coordinate]:
    if origin == route["origin"] and destination == route["destination"]:
        return route["coordinates"]
    if origin == route["destination"] and destination == route["origin"]:
        return route["return_coordinates"]
    return [_location_tuple(origin), _location_tuple(destination)]


def _interpolate_route(coordinates: list[Coordinate], progress: float) -> Coordinate:
    if len(coordinates) == 1:
        return coordinates[0]
    distances = [
        route_distance_km([a, b])
        for a, b in zip(coordinates, coordinates[1:])
    ]
    total = sum(distances) or 1.0
    target = max(0.0, min(1.0, progress)) * total
    covered = 0.0
    for index, segment in enumerate(distances):
        if covered + segment >= target:
            local = (target - covered) / (segment or 1.0)
            lat_a, lon_a = coordinates[index]
            lat_b, lon_b = coordinates[index + 1]
            return (lat_a + (lat_b - lat_a) * local, lon_a + (lon_b - lon_a) * local)
        covered += segment
    return coordinates[-1]


def _map_bbox(
    routes: dict[str, dict[str, Any]],
    network: Any,
    locations: dict[str, dict[str, float | str]],
) -> dict[str, float]:
    coordinates: list[Coordinate] = []
    for location in locations.values():
        coordinates.append((float(location["lat"]), float(location["lon"])))
    for route in routes.values():
        coordinates.extend(route["coordinates"])
        coordinates.extend(route["return_coordinates"])
    for segment in _build_pipeline_segments(network, locations):
        coordinates.extend(segment["coordinates"])
    for link in _build_injection_links(network, locations):
        coordinates.extend(link["coordinates"])
    lats = [lat for lat, _ in coordinates]
    lons = [lon for _, lon in coordinates]
    return {
        "min_lat": min(lats) - 0.7,
        "max_lat": max(lats) + 0.7,
        "min_lon": min(lons) - 0.9,
        "max_lon": max(lons) + 0.9,
    }


def _location_tuple(location_id: str, locations: dict[str, dict[str, float | str]] = LOCATIONS) -> Coordinate:
    location = locations[location_id]
    return (float(location["lat"]), float(location["lon"]))
