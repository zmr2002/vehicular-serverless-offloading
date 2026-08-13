from __future__ import annotations

import unittest

from serverless_function.app import app


class ServerlessFunctionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_and_task_contract(self):
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        response = self.client.post(
            "/v1/tasks",
            json={
                "task_id": "test-task",
                "compute_cycles": 1e9,
                "data_size_mb": 1.0,
                "deadline_ms": 1000.0,
                "work_units": 10,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("processing_ms", body)
        self.assertIn("cold_start", body)
        self.assertIn("checksum", body)

    def test_invalid_payload_is_rejected(self):
        self.assertEqual(self.client.post("/v1/tasks", json={}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
