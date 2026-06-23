import tempfile
import unittest
from pathlib import Path

from sim.routes import route_distance_km
from sim.visualization import (
    _connect_route_to_facilities,
    _interpolate_route,
    build_demo_trajectory,
    render_dashboard_html,
    write_dashboard,
)


class VisualizationTests(unittest.TestCase):
    def test_demo_trajectory_uses_one_hour_frames(self):
        payload = build_demo_trajectory(hours=8)

        self.assertEqual(payload["time_step_hours"], 1.0)
        self.assertEqual(len(payload["frames"]), 9)
        frame_times = [frame["time_h"] for frame in payload["frames"]]
        self.assertEqual(frame_times, list(range(0, 9)))
        self.assertIn("brevik", payload["frames"][0]["entities"])
        self.assertIn("northern_pioneer", payload["frames"][0]["entities"])

    def test_trajectory_does_not_execute_actions_without_agent_input(self):
        payload = build_demo_trajectory(hours=8)

        for frame in payload["frames"]:
            self.assertEqual(frame["flows_t"], {})
            self.assertEqual(frame["actions"], {})
            self.assertAlmostEqual(frame["entities"]["northern_pioneer"]["inventory_t"], 0.0)
            self.assertEqual(frame["vessel_positions"]["northern_pioneer"]["leg"], "loading_at_origin")

    def test_trajectory_executes_supplied_agent_actions_only(self):
        payload = build_demo_trajectory(
            hours=2,
            action_frames=[
                {"brevik": {"load_vessel": "northern_pioneer"}},
                {},
            ],
        )

        self.assertEqual(payload["frames"][1]["actions"], {"brevik": {"load_vessel": "northern_pioneer"}})
        self.assertAlmostEqual(payload["frames"][1]["entities"]["northern_pioneer"]["inventory_t"], 45.662100456621)
        self.assertEqual(payload["frames"][2]["flows_t"], {})
        self.assertAlmostEqual(payload["frames"][2]["entities"]["northern_pioneer"]["inventory_t"], 45.662100456621)

    def test_agent_sail_action_moves_vessel_without_loading_or_unloading(self):
        payload = build_demo_trajectory(
            hours=2,
            action_frames=[
                {"northern_pioneer": {"sail_to": "oygarden_terminal"}},
                {},
            ],
        )

        self.assertEqual(payload["frames"][1]["vessel_positions"]["northern_pioneer"]["leg"], "outbound_to_terminal")
        self.assertFalse(payload["frames"][1]["vessel_positions"]["northern_pioneer"]["at_berth"])
        self.assertEqual(payload["frames"][1]["flows_t"], {})

    def test_demo_trajectory_includes_map_coordinates_routes_and_multiple_vessels(self):
        payload = build_demo_trajectory(hours=4)

        self.assertIn("map", payload)
        self.assertIn("locations", payload["map"])
        self.assertIn("routes", payload["map"])
        self.assertIn("brevik", payload["map"]["locations"])
        self.assertIn("aurora_well_a", payload["map"]["locations"])
        self.assertIn("aurora_subsea_manifold", payload["map"]["locations"])
        self.assertIn("lat", payload["map"]["locations"]["brevik"])
        self.assertIn("lon", payload["map"]["locations"]["brevik"])

        vessel_ids = [
            entity_id
            for entity_id, entity in payload["frames"][0]["entities"].items()
            if entity["type"] == "Vessel"
        ]
        self.assertGreaterEqual(len(vessel_ids), 2)
        self.assertIn("vessel_positions", payload["frames"][1])
        for vessel_id in vessel_ids:
            self.assertIn(vessel_id, payload["frames"][1]["vessel_positions"])
            self.assertIn("lat", payload["frames"][1]["vessel_positions"][vessel_id])
            self.assertIn("lon", payload["frames"][1]["vessel_positions"][vessel_id])

    def test_demo_trajectory_marks_vessel_return_legs(self):
        action_frames = [{"northern_pioneer": {"sail_to": "oygarden_terminal"}}]
        action_frames.extend({} for _ in range(27))
        action_frames.append({"northern_pioneer": {"sail_to": "brevik"}})
        payload = build_demo_trajectory(hours=45, action_frames=action_frames)

        route = payload["map"]["routes"]["northern_pioneer"]
        return_frame = payload["frames"][45]["vessel_positions"]["northern_pioneer"]

        self.assertIn("return_coordinates", route)
        self.assertEqual(route["return_coordinates"][0], route["coordinates"][-1])
        self.assertEqual(route["return_policy"], "same_corridor_reverse")
        self.assertEqual(return_frame["leg"], "return_to_origin")
        self.assertIn("Return", return_frame["leg_label"])

    def test_vessel_cargo_is_constant_while_sailing(self):
        payload = build_demo_trajectory(hours=48)
        vessel_ids = [
            entity_id
            for entity_id, entity in payload["frames"][0]["entities"].items()
            if entity["type"] == "Vessel"
        ]

        for index in range(1, len(payload["frames"])):
            previous_frame = payload["frames"][index - 1]
            frame = payload["frames"][index]
            for vessel_id in vessel_ids:
                previous_cargo_t = previous_frame["entities"][vessel_id]["inventory_t"]
                cargo_t = frame["entities"][vessel_id]["inventory_t"]
                leg = frame["vessel_positions"][vessel_id]["leg"]
                if leg in {"outbound_to_terminal", "return_to_origin"}:
                    self.assertAlmostEqual(cargo_t, previous_cargo_t, msg=f"{vessel_id} cargo changed during {leg} at frame {index}")
                elif abs(cargo_t - previous_cargo_t) > 1e-9:
                    self.assertIn(leg, {"loading_at_origin", "unloading_at_terminal"})

    def test_vessel_map_progress_uses_speed_knots(self):
        payload = build_demo_trajectory(
            hours=10,
            action_frames=[{"northern_pioneer": {"sail_to": "oygarden_terminal"}}],
        )
        route = payload["map"]["routes"]["northern_pioneer"]
        position = payload["frames"][10]["vessel_positions"]["northern_pioneer"]
        speed_knots = payload["frames"][0]["entities"]["northern_pioneer"]["parameters"]["speed_knots"]

        expected_progress = (speed_knots * 1.852 * 10) / route["distance_km"]
        expected_lat, expected_lon = _interpolate_route(route["coordinates"], expected_progress)

        self.assertEqual(position["leg"], "outbound_to_terminal")
        self.assertAlmostEqual(position["lat"], expected_lat)
        self.assertAlmostEqual(position["lon"], expected_lon)

    def test_demo_does_not_unload_before_speed_based_arrival(self):
        action_frames = [{"brevik": {"load_vessel": "northern_pioneer"}}]
        action_frames.append({"northern_pioneer": {"sail_to": "oygarden_terminal"}})
        action_frames.extend({} for _ in range(22))
        action_frames.append(
            {
                "oygarden_terminal": {"unload_vessel": "northern_pioneer"},
                "oygarden_pipeline": {"flow_tph": 300.0},
            }
        )
        payload = build_demo_trajectory(hours=25, action_frames=action_frames)
        frame = payload["frames"][25]

        self.assertEqual(frame["vessel_positions"]["northern_pioneer"]["leg"], "outbound_to_terminal")
        self.assertNotIn("northern_pioneer->oygarden_terminal", frame["flows_t"])
        self.assertTrue(any(v["violation_type"] == "berth_required" for v in frame["violations"]))

    def test_vessels_stop_at_berths_during_loading_and_unloading(self):
        payload = build_demo_trajectory(hours=48)
        locations = payload["map"]["locations"]
        routes = payload["map"]["routes"]

        for frame in payload["frames"]:
            for vessel_id, position in frame["vessel_positions"].items():
                route = routes[vessel_id]
                if position["leg"] == "loading_at_origin":
                    berth_location = locations[route["origin"]]
                    self.assertTrue(position["at_berth"])
                    self.assertEqual(position["berth_id"], f"{route['origin']}_loading_berth")
                    self.assertEqual((position["lat"], position["lon"]), (berth_location["lat"], berth_location["lon"]))
                elif position["leg"] == "unloading_at_terminal":
                    berth_location = locations[route["destination"]]
                    self.assertTrue(position["at_berth"])
                    self.assertEqual(position["berth_id"], f"{route['destination']}_unloading_berth_1")
                    self.assertEqual((position["lat"], position["lon"]), (berth_location["lat"], berth_location["lon"]))
                else:
                    self.assertFalse(position["at_berth"])

    def test_vessel_routes_visually_connect_to_facility_coordinates(self):
        payload = build_demo_trajectory(hours=2)

        for route in payload["map"]["routes"].values():
            origin = payload["map"]["locations"][route["origin"]]
            destination = payload["map"]["locations"][route["destination"]]

            self.assertEqual(route["coordinates"][0], (origin["lat"], origin["lon"]))
            self.assertEqual(route["coordinates"][-1], (destination["lat"], destination["lon"]))
            self.assertIn("sea_coordinates", route)
            self.assertGreater(len(route["sea_coordinates"]), 2)

    def test_vessel_routes_are_densified_for_smoother_display(self):
        payload = build_demo_trajectory(hours=2)

        for route in payload["map"]["routes"].values():
            segment_lengths = [
                route_distance_km([start, end])
                for start, end in zip(route["coordinates"], route["coordinates"][1:])
            ]

            self.assertGreater(len(route["coordinates"]), len(route["sea_coordinates"]))
            self.assertLessEqual(max(segment_lengths), 25.0)

    def test_display_route_rounds_corners_without_large_distance_change(self):
        original = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

        route = _connect_route_to_facilities(original, original[0], original[-1])
        original_distance = route_distance_km(original)
        display_distance = route_distance_km(route)

        self.assertEqual(route[0], original[0])
        self.assertEqual(route[-1], original[-1])
        self.assertNotIn(original[1], route)
        self.assertTrue(any(lat > 0.0 and lon < 1.0 for lat, lon in route))
        self.assertLess(abs(display_distance - original_distance) / original_distance, 0.03)

    def test_dashboard_html_contains_embedded_data_and_controls(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("id=\"timeline\"", html)
        self.assertIn("id=\"playbackDelayMs\"", html)
        self.assertIn("id=\"componentList\"", html)
        self.assertIn("id=\"statusPanel\"", html)
        self.assertIn("window.__SIM_DATA__", html)
        self.assertIn("\"time_step_hours\": 1.0", html)
        self.assertIn("leaflet.css", html)
        self.assertIn("L.map", html)
        self.assertIn("L.tileLayer", html)
        self.assertIn("L.polyline", html)
        self.assertIn("L.marker", html)
        self.assertIn("renderMap", html)
        self.assertIn("component-type-row", html)

    def test_dashboard_groups_components_and_uses_ship_colors(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("const componentGroups", html)
        self.assertIn("function categoryForEntity", html)
        self.assertIn("component-group-toggle", html)
        self.assertIn("Emitter", html)
        self.assertIn("Ship", html)
        self.assertIn("Pipeline", html)
        self.assertIn("Storage", html)
        self.assertIn("const vesselColors", html)
        self.assertIn("function vesselColor", html)
        self.assertIn("ship-color-swatch", html)
        self.assertIn("--ship-color", html)

    def test_dashboard_classifies_wells_and_terminal_storage_roles(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("Intermediate Storage", html)
        self.assertIn('entity.type === "Terminal"', html)
        self.assertIn('entity.type === "InjectionWell"', html)
        self.assertIn('return "storage"', html)
        self.assertIn("function storageTargetForEntity", html)
        self.assertIn("Storage:", html)
        self.assertIn("\"aurora_well_a\": \"aurora_reservoir\"", html)

    def test_dashboard_playback_rate_is_user_configurable(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("playbackDelayMs", html)
        self.assertIn('id="playbackDelayMs" type="number" min="100" max="5000" step="100" value="200"', html)
        self.assertIn("function playbackDelay", html)
        self.assertIn("playbackDelayMs.addEventListener", html)
        self.assertIn("setInterval", html)
        self.assertIn("playbackDelay()", html)

    def test_write_dashboard_accepts_agent_action_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_dashboard(
                Path(tmpdir) / "dashboard.html",
                hours=1,
                action_frames=[{"northern_pioneer": {"sail_to": "oygarden_terminal"}}],
            )

            html = output.read_text(encoding="utf-8")

        self.assertIn("\"sail_to\": \"oygarden_terminal\"", html)
        self.assertIn("outbound_to_terminal", html)

    def test_map_uses_searoute_and_pipeline_segments(self):
        payload = build_demo_trajectory(hours=2)

        providers = {route["provider"] for route in payload["map"]["routes"].values()}

        self.assertEqual(providers, {"searoute"})
        self.assertIn("pipeline_segments", payload["map"])
        self.assertGreaterEqual(len(payload["map"]["pipeline_segments"]), 2)
        for segment in payload["map"]["pipeline_segments"]:
            self.assertIn("coordinates", segment)
            self.assertGreaterEqual(len(segment["coordinates"]), 2)
            self.assertIn("color", segment)

    def test_map_routes_pipeline_to_subsea_manifold(self):
        payload = build_demo_trajectory(hours=2)

        self.assertNotIn("johansen_pipeline_endpoint", payload["map"]["locations"])
        manifold = payload["map"]["locations"]["aurora_subsea_manifold"]

        offshore_segments = [
            segment for segment in payload["map"]["pipeline_segments"]
            if segment["id"] == "naturgassparken_to_eos_subsea_manifold"
        ]

        self.assertEqual(len(offshore_segments), 1)
        segment = offshore_segments[0]
        self.assertEqual(segment["source"], "oygarden_terminal")
        self.assertEqual(segment["target"], "aurora_subsea_manifold")
        self.assertEqual(segment["coordinates"][-1], (manifold["lat"], manifold["lon"]))

    def test_map_separates_injection_links_from_pipeline_segments(self):
        payload = build_demo_trajectory(hours=2)

        pipeline_targets = {segment["target"] for segment in payload["map"]["pipeline_segments"]}
        self.assertNotIn("aurora_reservoir", pipeline_targets)
        self.assertIn("injection_links", payload["map"])

        links = payload["map"]["injection_links"]
        self.assertEqual({link["target"] for link in links}, {"aurora_reservoir"})
        self.assertEqual({link["relation"] for link in links}, {"injection_target"})
        for link in links:
            self.assertIn(link["source"], {"aurora_well_a", "aurora_well_c"})
            self.assertIn("coordinates", link)
            self.assertEqual(link["style"], "geologic")

    def test_dashboard_draws_injection_links_and_lat_lon_grid(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("injectionLayer", html)
        self.assertIn("drawInjectionLinks", html)
        self.assertIn("data.map.injection_links", html)
        self.assertIn('dashArray: "2 8"', html)
        self.assertIn("graticuleLayer", html)
        self.assertIn("drawLatLonGrid", html)
        self.assertIn("leafletMap.on(\"moveend zoomend\", drawLatLonGrid)", html)
        self.assertIn("createPane(\"graticulePane\")", html)
        self.assertIn("coordinateReadoutControl", html)
        self.assertIn("Lat/Lon grid", html)
        self.assertIn("°N", html)
        self.assertIn("longitudeLabelLat", html)
        self.assertIn("latitudeLabelLon", html)
        self.assertIn("iconAnchor: [29, 20]", html)

    def test_dashboard_does_not_draw_map_halo_for_pressure_or_reservoir(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertNotIn("Geologic storage extent marker", html)
        self.assertNotIn("L.circle([targetLocation.lat, targetLocation.lon]", html)

    def test_dashboard_chart_uses_polished_responsive_renderer(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("resizeChartCanvas", html)
        self.assertIn("drawChartGrid", html)
        self.assertIn("drawCapacityReference", html)
        self.assertIn("drawCurrentValueMarker", html)
        self.assertIn("devicePixelRatio", html)
        self.assertIn("createLinearGradient", html)
        self.assertIn("window.addEventListener(\"resize\"", html)

    def test_dashboard_component_list_summarizes_well_and_reservoir_pressures(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("function componentSummary", html)
        self.assertIn("function pressureSummaryForEntity", html)
        self.assertIn("BHP", html)
        self.assertIn("Delta", html)
        self.assertIn("Rate", html)
        self.assertIn("Margin", html)
        self.assertIn("Fill", html)
        self.assertIn("component-summary", html)

    def test_dashboard_component_list_does_not_duplicate_plain_entity_type(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("function componentSecondaryLine", html)
        self.assertNotIn("secondary: entity.type", html)
        self.assertIn("${componentSecondaryLine(entity, storageTargetText)}", html)

    def test_dashboard_chart_layout_reserves_full_canvas_area(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("height: calc(100vh - 76px);", html)
        self.assertIn("overflow: hidden;", html)
        self.assertIn(".map-panel { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }", html)
        self.assertIn(".workspace { display: grid; grid-template-rows: minmax(0, 1fr) 300px; min-height: 0; }", html)
        self.assertIn("#networkCanvas { width: 100%; height: 100%; min-height: 0;", html)
        self.assertIn("<div class=\"map-panel\">", html)
        self.assertIn("const height = Math.max(220, Math.floor(rect.height));", html)
        self.assertIn(".chart-panel { display: grid; grid-template-rows: 38px minmax(0, 1fr);", html)
        self.assertIn("#chartCanvas { width: 100%; height: 100%; min-height: 0; display: block; }", html)

    def test_dashboard_status_shows_pressure_diagnostics(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("function renderPressureStatus", html)
        self.assertIn("Bottomhole pressure", html)
        self.assertIn("BHP delta", html)
        self.assertIn("Injection rate", html)
        self.assertIn("Interference delta", html)
        self.assertIn("Reservoir pressure", html)
        self.assertIn("Pressure margin", html)
        self.assertIn("Fill fraction", html)
        self.assertIn("Pressure @", html)

    def test_dashboard_chart_metric_selector_supports_pressure_metrics(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("id=\"chartMetric\"", html)
        self.assertIn("Inventory", html)
        self.assertIn("Pressure", html)
        self.assertIn("Pressure margin", html)
        self.assertIn("Injection rate", html)
        self.assertIn("function metricSeries", html)
        self.assertIn("function metricValueForEntity", html)
        self.assertIn("chartMetric.addEventListener", html)

    def test_dashboard_chart_shows_hover_time_and_value(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("let hoveredChartIndex = null;", html)
        self.assertIn("function chartHoverIndex(event, plot)", html)
        self.assertIn("function drawChartTooltip(plot, series, values, maxValue, index, width)", html)
        self.assertIn("const markerIndex = hoveredChartIndex ?? frameIndex;", html)
        self.assertIn("Hour ${frames[index].time_h}", html)
        self.assertIn("chart.addEventListener(\"mousemove\"", html)
        self.assertIn("chart.addEventListener(\"mouseleave\"", html)

    def test_dashboard_filters_metric_options_by_selected_entity_type(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("function availableMetricsForEntity", html)
        self.assertIn("function syncChartMetricOptions", html)
        self.assertIn("if (entity.type === \"InjectionWell\") return [\"inventory\", \"pressure\", \"injection_rate\"]", html)
        self.assertIn("if (entity.type === \"Reservoir\") return [\"inventory\", \"pressure\", \"pressure_margin\", \"injection_rate\"]", html)
        self.assertIn("return [\"inventory\"]", html)
        self.assertIn("syncChartMetricOptions(frame.entities[selected])", html)

    def test_dashboard_html_draws_capacity_limits_and_pipelines(self):
        payload = build_demo_trajectory(hours=2)

        html = render_dashboard_html(payload)

        self.assertIn("formatInventoryLimit", html)
        self.assertIn("formatInventoryLimit(entity)", html)
        self.assertIn("Cargo", html)
        self.assertIn("Return leg", html)
        self.assertIn("same corridor assumed", html)
        self.assertIn("drawReturnRoutes", html)
        self.assertIn("pipelineLayer", html)
        self.assertIn("drawPipelines", html)

    def test_dashboard_embeds_reference_physical_parameters(self):
        payload = build_demo_trajectory(hours=2)
        html = render_dashboard_html(payload)

        brevik_params = payload["frames"][0]["entities"]["brevik"]["parameters"]
        vessel_params = payload["frames"][0]["entities"]["northern_pioneer"]["parameters"]
        pipeline_params = payload["frames"][0]["entities"]["oygarden_pipeline"]["parameters"]
        reservoir_params = payload["frames"][0]["entities"]["aurora_reservoir"]["parameters"]

        self.assertEqual(brevik_params["annual_target_export_tpy"], 400_000.0)
        self.assertEqual(brevik_params["max_production_tph"], 56.0)
        self.assertEqual(vessel_params["volume_capacity_m3"], 7_500.0)
        self.assertEqual(vessel_params["speed_knots"], 14.0)
        self.assertEqual(pipeline_params["annual_capacity_tpy"], 5_000_000.0)
        self.assertEqual(pipeline_params["length_km"], 100.4)
        self.assertEqual(pipeline_params["route_color"], "#ff0000")
        self.assertEqual(reservoir_params["depth_m"], 2_600.0)
        self.assertIn("\"annual_target_export_tpy\": 400000.0", html)
        self.assertIn("\"volume_capacity_m3\": 7500.0", html)
        self.assertIn("\"speed_knots\": 14.0", html)
        self.assertIn("\"annual_capacity_tpy\": 5000000.0", html)
        self.assertIn("\"depth_m\": 2600.0", html)

    def test_offshore_pipeline_route_is_red_and_starts_at_naturgassparken(self):
        payload = build_demo_trajectory(hours=2)

        terminal = payload["map"]["locations"]["oygarden_terminal"]
        manifold = payload["map"]["locations"]["aurora_subsea_manifold"]
        offshore_segments = [
            segment for segment in payload["map"]["pipeline_segments"]
            if segment["id"] == "naturgassparken_to_eos_subsea_manifold"
        ]

        self.assertEqual(len(offshore_segments), 1)
        segment = offshore_segments[0]
        self.assertEqual(segment["color"], "#ff0000")
        self.assertEqual(segment["source"], "oygarden_terminal")
        self.assertEqual(segment["target"], "aurora_subsea_manifold")
        self.assertEqual(segment["coordinates"][0], (terminal["lat"], terminal["lon"]))
        self.assertEqual(segment["coordinates"][-1], (manifold["lat"], manifold["lon"]))
        self.assertGreaterEqual(len(segment["coordinates"]), 4)
        self.assertIn("Naturgassparken", terminal["label"])
        self.assertAlmostEqual(terminal["lat"], 60.553892)
        self.assertAlmostEqual(terminal["lon"], 4.882342)
        self.assertAlmostEqual(manifold["lat"], 60.575913400339694)
        self.assertAlmostEqual(manifold["lon"], 3.4416777066688846)
        self.assertAlmostEqual(segment["length_km"], 100.4)

        pipeline_segments = [
            segment for segment in payload["map"]["pipeline_segments"]
            if segment["component_id"] == "oygarden_pipeline"
        ]
        self.assertEqual(
            [segment["id"] for segment in pipeline_segments],
            ["naturgassparken_to_eos_subsea_manifold"],
        )


if __name__ == "__main__":
    unittest.main()
