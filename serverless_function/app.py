from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from threading import Lock
from time import perf_counter

from flask import Flask, jsonify, request


app = Flask(__name__)
INSTANCE_ID = os.getenv("HOSTNAME", f"local-{os.getpid()}")
_first_request = True
_first_request_lock = Lock()


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "instance_id": INSTANCE_ID})


@app.post("/v1/tasks")
def process_task():
    global _first_request
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    required = ("task_id", "compute_cycles", "data_size_mb", "deadline_ms", "work_units")
    missing = [field for field in required if field not in payload]
    if missing:
        return jsonify({"error": "missing required fields", "fields": missing}), 400
    try:
        task_id = str(payload["task_id"])
        compute_cycles = float(payload["compute_cycles"])
        data_size_mb = float(payload["data_size_mb"])
        deadline_ms = float(payload["deadline_ms"])
        work_units = int(payload["work_units"])
    except (TypeError, ValueError):
        return jsonify({"error": "numeric fields contain invalid values"}), 400
    if compute_cycles <= 0 or data_size_mb < 0 or deadline_ms <= 0 or not 1 <= work_units <= 1_000_000:
        return jsonify({"error": "numeric fields are outside their allowed ranges"}), 422

    with _first_request_lock:
        cold_start = _first_request
        _first_request = False
    received_at = datetime.now(timezone.utc)
    started = perf_counter()
    digest = task_id.encode("utf-8")
    for index in range(work_units):
        digest = sha256(digest + index.to_bytes(4, "little", signed=False)).digest()
    processing_ms = (perf_counter() - started) * 1_000.0
    completed_at = datetime.now(timezone.utc)
    return jsonify(
        {
            "task_id": task_id,
            "instance_id": INSTANCE_ID,
            "cold_start": cold_start,
            "processing_ms": processing_ms,
            "received_at": received_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "checksum": digest.hex()[:16],
            "input": {
                "compute_cycles": compute_cycles,
                "data_size_mb": data_size_mb,
                "deadline_ms": deadline_ms,
                "work_units": work_units,
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
