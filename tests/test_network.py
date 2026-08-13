from __future__ import annotations

import unittest

from vehicular_offloading.config import EnergyConfig, NetworkConfig
from vehicular_offloading.domain import OffloadAction, ServiceQuote, Task, VehicleState
from vehicular_offloading.network import (
    LazyNeighborGraph,
    ReverseParetoV2VIndex,
    ServiceSpatialIndex,
    best_v2v_path,
    build_neighbor_map,
    build_neighbor_map_with_throughput,
    build_compact_neighbor_graph,
    effective_v2v_throughput_mbps,
    effective_v2i_throughput_mbps,
    spectral_efficiency_bps_hz,
    transmission_delay_s,
)


class NetworkTests(unittest.TestCase):
    def test_throughput_uses_channel_mhz_times_spectral_efficiency(self):
        config = NetworkConfig(channel_capacity_model="distance_only")
        self.assertAlmostEqual(
            effective_v2v_throughput_mbps(500.0, config),
            config.v2v_channel_bandwidth_mhz * config.v2v_resource_efficiency,
        )
        self.assertAlmostEqual(
            effective_v2i_throughput_mbps(500.0, config),
            config.v2i_channel_bandwidth_mhz * config.v2i_resource_efficiency,
        )

    def test_log_distance_path_loss_has_ten_n_db_factor(self):
        config = NetworkConfig(
            v2v_reference_snr_db=20.0,
            v2v_reference_distance_m=100.0,
            v2v_path_loss_exponent=2.0,
        )
        # One distance decade loses 10*n = 20 dB, leaving 0 dB SNR.
        self.assertAlmostEqual(
            spectral_efficiency_bps_hz(1_000.0, OffloadAction.V2V, config),
            1.0,
            places=6,
        )

    def test_reviewed_v2v_edge_rate_meets_mean_task_budget_and_v2i_is_stronger(self):
        config = NetworkConfig()
        v2v = effective_v2v_throughput_mbps(500.0, config)
        v2i = effective_v2i_throughput_mbps(500.0, config)
        required_mbps = 8.0 * 50.5 / (2.5 - 3.0e9 / 2.0e9)
        self.assertGreaterEqual(v2v, required_mbps)
        self.assertGreater(v2i, v2v)

    def test_v2i_transmission_formula_uses_data_bits_over_throughput(self):
        config = NetworkConfig(channel_capacity_model="distance_only")
        delay = transmission_delay_s(10.0, 0.0, OffloadAction.V2I, config)
        self.assertAlmostEqual(delay, 80.0 / 170.0)

    def test_shannon_capacity_restores_snr_multiplier(self):
        config = NetworkConfig()
        distance = 135.3
        efficiency = spectral_efficiency_bps_hz(distance, OffloadAction.V2I, config)
        capacity = effective_v2i_throughput_mbps(distance, config)
        self.assertAlmostEqual(efficiency, 10.0, places=3)
        self.assertAlmostEqual(capacity, 1700.0, places=1)
        self.assertLess(
            transmission_delay_s(100.0, distance, OffloadAction.V2I, config),
            0.525,
        )

    def test_shannon_efficiency_is_bounded_at_zero_distance(self):
        config = NetworkConfig(v2i_max_spectral_efficiency_bps_hz=8.0)
        self.assertEqual(
            spectral_efficiency_bps_hz(0.0, OffloadAction.V2I, config),
            8.0,
        )

    def test_three_hop_path_is_found_deterministically(self):
        vehicles = {
            "a": VehicleState("a", (0.0, 0.0), 0.0, 2e9),
            "b": VehicleState("b", (90.0, 0.0), 0.0, 2e9),
            "c": VehicleState("c", (180.0, 0.0), 0.0, 2e9),
            "service": VehicleState("service", (270.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=100.0, max_hops=3)
        graph = build_neighbor_map(vehicles, config.neighbor_radius_m)
        task = Task("t", "a", 1e9, 1.0, 2.0, 0.5, 0)
        result = best_v2v_path(
            "a", vehicles, (ServiceQuote("service", 0.05, 10e9, 0.03),), task, config, graph
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.path, ("a", "b", "c", "service"))
        weighted_graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        optimized = best_v2v_path(
            "a",
            vehicles,
            {"service": ServiceQuote("service", 0.05, 10e9, 0.03)},
            task,
            config,
            weighted_graph,
            bandwidths,
            10e9,
        )
        self.assertEqual(optimized, result)

    def test_bounded_dynamic_program_matches_exhaustive_paths(self):
        vehicles = {
            "a": VehicleState("a", (0.0, 0.0), 0.0, 2e9),
            "b": VehicleState("b", (60.0, 0.0), 0.0, 2e9),
            "c": VehicleState("c", (120.0, 20.0), 0.0, 2e9),
            "s1": VehicleState("s1", (180.0, 0.0), 0.0, 10e9, is_service=True),
            "s2": VehicleState("s2", (100.0, 80.0), 0.0, 8e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=125.0, max_hops=3)
        graph = build_neighbor_map(vehicles, config.neighbor_radius_m)
        task = Task("t", "a", 1e9, 2.0, 2.0, 0.5, 0)
        quotes = (
            ServiceQuote("s1", 0.04, 10e9, 0.02),
            ServiceQuote("s2", 0.03, 8e9, 0.02),
        )
        result = best_v2v_path("a", vehicles, quotes, task, config, graph)
        weighted_graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        weighted_result = best_v2v_path(
            "a", vehicles, quotes, task, config, weighted_graph, bandwidths
        )

        quote_by_id = {quote.vehicle_id: quote for quote in quotes}
        candidates = []

        def visit(current, path, transmission):
            if 0 < len(path) - 1 <= config.max_hops and current in quote_by_id:
                quote = quote_by_id[current]
                candidates.append(
                    (
                        transmission + task.compute_cycles / quote.compute_hz,
                        tuple(path),
                        current,
                    )
                )
            if len(path) - 1 == config.max_hops:
                return
            for neighbor in graph[current]:
                if neighbor in path:
                    continue
                distance = ((vehicles[current].position[0] - vehicles[neighbor].position[0]) ** 2
                            + (vehicles[current].position[1] - vehicles[neighbor].position[1]) ** 2) ** 0.5
                link = transmission_delay_s(task.data_size_mb, distance, OffloadAction.V2V, config)
                visit(neighbor, path + [neighbor], transmission + link)

        visit("a", ["a"], 0.0)
        expected = min(candidates)
        self.assertAlmostEqual(result.delay_s, expected[0])
        self.assertEqual(result.path, expected[1])
        self.assertEqual(result.target_id, expected[2])
        self.assertEqual(weighted_result, result)

    def test_uniform_service_fast_path_matches_generic_search(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "relay": VehicleState("relay", (30.0, 0.0), 0.0, 2e9),
            "far": VehicleState("far", (90.0, 0.0), 0.0, 10e9, is_service=True),
            "near": VehicleState("near", (20.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=100.0, max_hops=3)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        task = Task("t", "source", 1e9, 10.0, 2.0, 0.5, 0)
        quotes = {
            quote.vehicle_id: quote
            for quote in (
                ServiceQuote("far", 0.04, 10e9, 0.02),
                ServiceQuote("near", 0.03, 10e9, 0.02),
            )
        }
        generic = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths
        )
        optimized = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 10e9
        )
        self.assertEqual(optimized, generic)
        # Both short links hit the configured spectral-efficiency ceiling, so
        # deterministic target-ID tie breaking selects "far".
        self.assertEqual(optimized.target_id, "far")

    def test_uniform_service_search_accounts_for_persistent_queue(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "near_busy": VehicleState(
                "near_busy", (10.0, 0.0), 0.0, 2e9, workload_cycles=4e9, is_service=True
            ),
            "far_idle": VehicleState(
                "far_idle", (80.0, 0.0), 0.0, 2e9, workload_cycles=0.0, is_service=True
            ),
        }
        config = NetworkConfig(neighbor_radius_m=100.0, max_hops=3)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        quotes = {
            name: ServiceQuote(name, 0.05, 2e9, 0.03)
            for name in ("near_busy", "far_idle")
        }
        task = Task("t", "source", 1e9, 1.0, 2.0, 0.5, 0)
        generic = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths
        )
        optimized = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 2e9
        )
        self.assertEqual(optimized, generic)
        self.assertEqual(optimized.target_id, "far_idle")

    def test_service_spatial_index_is_exact_and_respects_dynamic_quotes(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "near": VehicleState("near", (20.0, 0.0), 0.0, 10e9, is_service=True),
            "far": VehicleState("far", (80.0, 0.0), 0.0, 10e9, is_service=True),
            "outside": VehicleState("outside", (101.0, 0.0), 0.0, 10e9, is_service=True),
        }
        index = ServiceSpatialIndex(vehicles, {"near", "far", "outside"})
        quotes = {
            name: ServiceQuote(name, 0.05, 10e9, 0.03)
            for name in ("near", "far", "outside")
        }
        self.assertEqual(index.nearest("source", quotes, 100.0), "near")
        quotes.pop("near")
        self.assertEqual(index.nearest("source", quotes, 100.0), "far")
        quotes.pop("far")
        self.assertIsNone(index.nearest("source", quotes, 100.0))

    def test_batched_spatial_index_remains_exact_after_candidate_removal(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            **{
                f"service-{index:02d}": VehicleState(
                    f"service-{index:02d}", (float(index + 1), 0.0), 0.0, 10e9, is_service=True
                )
                for index in range(20)
            },
        }
        service_ids = {vehicle_id for vehicle_id in vehicles if vehicle_id != "source"}
        index = ServiceSpatialIndex(vehicles, service_ids)
        index.prepare_sources(
            ["source"], radius_m=100.0, candidate_count=4, minimum_pair_count=0
        )
        quotes = {
            vehicle_id: ServiceQuote(vehicle_id, 0.05, 10e9, 0.03)
            for vehicle_id in service_ids
        }
        self.assertEqual(index.nearest("source", quotes, 100.0), "service-00")
        for removed in ("service-00", "service-01", "service-02", "service-03"):
            quotes.pop(removed)
        self.assertEqual(index.nearest("source", quotes, 100.0), "service-04")

    def test_spatial_direct_path_avoids_neighbor_materialization(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "relay": VehicleState("relay", (10.0, 10.0), 0.0, 2e9),
            "near": VehicleState("near", (20.0, 0.0), 0.0, 10e9, is_service=True),
            "far": VehicleState("far", (80.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=100.0, max_hops=3)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        quotes = {
            name: ServiceQuote(name, 0.05, 10e9, 0.03)
            for name in ("near", "far")
        }
        index = ServiceSpatialIndex(vehicles, quotes)
        task = Task("t", "source", 1e9, 10.0, 2.0, 0.5, 0)
        result = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 10e9, index
        )
        self.assertEqual(result.target_id, "near")
        self.assertEqual(graph.materialized_nodes, 0)

    def test_uniform_compute_bound_does_not_scan_all_quotes(self):
        class NoIterationQuotes(dict):
            def values(self):
                raise AssertionError("uniform CPU bound must not scan every service quote")

        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "service": VehicleState("service", (10.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=100.0, max_hops=3)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        quotes = NoIterationQuotes(
            {"service": ServiceQuote("service", 0.05, 10e9, 0.03)}
        )
        task = Task("t", "source", 1e9, 1.0, 2.0, 0.5, 0)
        result = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 10e9
        )
        self.assertEqual(result.target_id, "service")

    def test_uniform_dijkstra_matches_layered_search_when_service_is_sparse(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "a": VehicleState("a", (40.0, 0.0), 0.0, 2e9),
            "b": VehicleState("b", (0.0, 45.0), 0.0, 2e9),
            "c": VehicleState("c", (80.0, 0.0), 0.0, 2e9),
            "service": VehicleState("service", (120.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=55.0, max_hops=3)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        quotes = {"service": ServiceQuote("service", 0.05, 10e9, 0.03)}
        task = Task("t", "source", 5e9, 50.0, 2.0, 0.5, 0)
        layered = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths
        )
        optimized = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 10e9
        )
        self.assertEqual(optimized, layered)
        self.assertEqual(optimized.path, ("source", "a", "c", "service"))

    def test_reverse_pareto_index_matches_exact_per_task_search(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "a": VehicleState("a", (40.0, 0.0), 0.0, 2e9),
            "b": VehicleState("b", (40.0, 35.0), 0.0, 2e9),
            "idle": VehicleState(
                "idle", (115.0, 0.0), 0.0, 2e9, workload_cycles=0.0, is_service=True
            ),
            "busy": VehicleState(
                "busy", (70.0, 40.0), 0.0, 2e9, workload_cycles=3e9, is_service=True
            ),
        }
        config = NetworkConfig(neighbor_radius_m=80.0, max_hops=3)
        quotes = {
            target: ServiceQuote(target, 0.05, 2e9, 0.03)
            for target in ("idle", "busy")
        }
        lazy_graph, throughputs = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        compact_graph = build_compact_neighbor_graph(
            vehicles, config.neighbor_radius_m, config
        )
        index = ReverseParetoV2VIndex(
            compact_graph,
            vehicles,
            quotes,
            config.max_hops,
            2e9,
            EnergyConfig(),
        )
        for data_size_mb in (1.0, 25.0, 100.0):
            task = Task("t", "source", 1e9, data_size_mb, 5.0, 0.5, 0)
            expected = best_v2v_path(
                "source",
                vehicles,
                quotes,
                task,
                config,
                lazy_graph,
                throughputs,
                2e9,
            )
            actual = index.estimate("source", task)
            self.assertAlmostEqual(actual.delay_s, expected.delay_s, places=12)
            self.assertEqual(actual.target_id, expected.target_id)
            self.assertEqual(actual.path, expected.path)

    def test_equal_delay_prefers_fewer_hops_deterministically(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "direct": VehicleState("direct", (500.0, 0.0), 0.0, 10e9, is_service=True),
            "relay": VehicleState("relay", (0.0, 0.0), 0.0, 2e9),
            "deep": VehicleState("deep", (0.0, 0.0), 0.0, 10e9, is_service=True),
        }
        config = NetworkConfig(neighbor_radius_m=500.0, max_hops=2)
        graph, bandwidths = build_neighbor_map_with_throughput(
            vehicles, config.neighbor_radius_m, config
        )
        quotes = {
            name: ServiceQuote(name, 0.05, 10e9, 0.03)
            for name in ("direct", "deep")
        }
        task = Task("t", "source", 1e9, 10.0, 2.0, 0.5, 0)
        result = best_v2v_path(
            "source", vehicles, quotes, task, config, graph, bandwidths, 10e9
        )
        self.assertEqual(result.path, ("source", "deep"))

    def test_lazy_graph_materializes_only_requested_nodes(self):
        vehicles = {
            f"v{index}": VehicleState(f"v{index}", (float(index * 10), 0.0), 0.0, 2e9)
            for index in range(20)
        }
        graph, bandwidths = build_neighbor_map_with_throughput(vehicles, 55.0, NetworkConfig())
        self.assertIsInstance(graph, LazyNeighborGraph)
        self.assertEqual(graph.materialized_nodes, 0)
        self.assertEqual(graph.neighbor_count("v0", limit=3), 3)
        self.assertEqual(graph.materialized_nodes, 0)
        expected = tuple(f"v{index}" for index in (1, 2, 3, 4, 5))
        self.assertEqual(set(graph["v0"]), set(expected))
        self.assertEqual(graph.materialized_nodes, 1)
        self.assertEqual(set(bandwidths["v0"]), set(expected))
        self.assertEqual(graph.materialized_nodes, 1)

    def test_lazy_reachability_reuses_service_set(self):
        vehicles = {
            "source": VehicleState("source", (0.0, 0.0), 0.0, 2e9),
            "relay": VehicleState("relay", (40.0, 0.0), 0.0, 2e9),
            "service": VehicleState("service", (80.0, 0.0), 0.0, 10e9),
            "isolated": VehicleState("isolated", (500.0, 0.0), 0.0, 2e9),
        }
        graph, _ = build_neighbor_map_with_throughput(vehicles, 50.0, NetworkConfig())
        self.assertTrue(graph.can_reach_any("source", {"service"}, 2))
        materialized = graph.materialized_nodes
        self.assertFalse(graph.can_reach_any("isolated", {"service"}, 2))
        self.assertEqual(graph.materialized_nodes, materialized)


if __name__ == "__main__":
    unittest.main()
