from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import sumolib

from vehicular_offloading.routes import generate_exact_routes


class RouteGenerationTests(unittest.TestCase):
    def test_requested_vehicle_count_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "routes.rou.xml"
            summary = generate_exact_routes(
                Path("scenarios/wakaba/wakaba.net.xml"), output, vehicle_count=10, simulation_end_s=100, seed=3
            )
            root = ET.parse(output).getroot()
            self.assertEqual(summary.written_vehicles, 10)
            self.assertEqual(len(root.findall("vehicle")), 10)
            network = sumolib.net.readNet("scenarios/wakaba/wakaba.net.xml")
            for vehicle in root.findall("vehicle"):
                route_ids = vehicle.find("route").attrib["edges"].split()
                route = tuple(network.getEdge(edge_id) for edge_id in route_ids)
                expected, _ = network.getShortestPath(route[0], route[-1], vClass="passenger")
                self.assertEqual(tuple(edge.getID() for edge in expected), tuple(route_ids))
                self.assertTrue(all(edge.allows("passenger") for edge in route))


if __name__ == "__main__":
    unittest.main()
