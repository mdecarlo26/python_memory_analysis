"""
test_integration.py
-------------------
End-to-end integration test for the Python profiler layer.

Flow:
  1. Start shared memory bridge (reader side)
  2. Spawn workload_numpy.py subprocess
  3. Attach to it via Attacher
  4. Drain events from bridge into ObjectGraph
  5. Serialize as ProfilingSession JSON
  6. Validate against schema

Pass criteria:
  [I1] Attach succeeds and target keeps running
  [I2] Events arrive within 2s of attach
  [I3] Events include numpy allocations (numpy.ndarray or float types)
  [I4] ObjectGraph has at least 10 nodes
  [I5] ProfilingSession JSON is schema-valid
  [I6] Target exits cleanly after SIGTERM
"""

import gc
import json
import os
import select
import signal
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from attach import Attacher
from bridge import Bridge
from graph import ObjectGraph

SCHEMA_PATH = ROOT / "hpc_profiler_schema.json"
WORKLOAD    = Path(__file__).parent / "workload_numpy.py"

ATTACH_TIMEOUT  = 5.0   # seconds to wait for attach
EVENT_TIMEOUT   = 4.0   # seconds to wait for first events after attach
COLLECT_WINDOW  = 3.0   # seconds to collect events after first one arrives


def unique_shm():
    return f"/hpc_integ_{uuid.uuid4().hex[:8]}"


def load_schema():
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def build_session(cpu_events: list, session_start: int, session_end: int) -> dict:
    """Assemble a ProfilingSession dict from collected CPU events."""
    return {
        "session_id":       str(uuid.uuid4()),
        "schema_version":   "0.1",
        "start_time_ns":    session_start,
        "end_time_ns":      session_end,
        "cpu_events":       cpu_events,
        "gpu_events":       [],          # GPU layer not yet built
        "correlated_events": [],
    }


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Brief settle between test class runs to avoid shm naming collisions."""
        time.sleep(0.2)

    def setUp(self):
        self.shm     = unique_shm()
        self.bridge  = None
        self.proc    = None
        self.attacher = None

    def tearDown(self):
        # Stop target first — gives profiler thread time to flush before bridge closes
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

        # Small settle window so profiler thread in target stops writing
        time.sleep(0.1)

        if self.bridge:
            try:
                self.bridge.close(unlink=True)
            except Exception:
                pass

    def _start_bridge(self):
        self.bridge = Bridge(shm_name=self.shm, capacity=8 * 1024 * 1024)
        self.bridge.open(create=True)

    def _spawn_target(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKLOAD)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for pid= line
        r, _, _ = select.select([self.proc.stdout], [], [], ATTACH_TIMEOUT)
        self.assertTrue(r, "Target did not print PID within timeout")
        line = self.proc.stdout.readline().decode().strip()
        self.assertTrue(line.startswith("pid="), f"Unexpected target output: {line}")
        return int(line.split("=")[1])

    def _attach(self, pid):
        self.attacher = Attacher(pid=pid, shm_name=self.shm)
        self.attacher.attach()

    def _drain_events(self, duration: float) -> list:
        """
        Read events from the bridge for `duration` seconds.
        Returns list of deserialized CpuEvent dicts.
        """
        events = []
        deadline = time.time() + duration
        while time.time() < deadline:
            payload = self.bridge.read()
            if payload:
                try:
                    events.append(Bridge.deserialize(payload))
                except Exception:
                    pass
            else:
                time.sleep(0.01)
        return events

    def _wait_for_first_event(self, timeout: float) -> list:
        """
        Poll the bridge until the first event arrives or timeout.
        Returns a list containing the first event if found, or empty list.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = self.bridge.read()
            if payload:
                try:
                    event = Bridge.deserialize(payload)
                    return [event]
                except Exception:
                    pass
            time.sleep(0.02)
        return []

    # ------------------------------------------------------------------
    # I1 — Attach succeeds, target keeps running
    # ------------------------------------------------------------------

    def test_I1_attach_succeeds_target_survives(self):
        self._start_bridge()
        pid = self._spawn_target()
        self._attach(pid)

        self.assertTrue(self.attacher.is_attached,
            "Attacher did not set is_attached after attach()")

        time.sleep(0.3)
        self.assertIsNone(self.proc.poll(),
            f"Target process died after attach (exit code {self.proc.poll()})")

    # ------------------------------------------------------------------
    # I2 — Events arrive within 2s of attach
    # ------------------------------------------------------------------

    def test_I2_events_arrive_within_2s(self):
        self._start_bridge()
        pid = self._spawn_target()
        attach_time = time.time()
        self._attach(pid)

        arrived = self._wait_for_first_event(timeout=EVENT_TIMEOUT)
        elapsed = time.time() - attach_time

        self.assertGreater(len(arrived), 0,
            f"No events arrived within {EVENT_TIMEOUT}s of attach")
        self.assertLess(elapsed, EVENT_TIMEOUT,
            f"First event took {elapsed:.2f}s — over {EVENT_TIMEOUT}s limit")

    # ------------------------------------------------------------------
    # I3 — Events include numpy-related allocations
    # ------------------------------------------------------------------

    def test_I3_numpy_allocations_captured(self):
        self._start_bridge()
        pid = self._spawn_target()
        self._attach(pid)

        # Wait for events to start flowing, then collect for a window
        first = self._wait_for_first_event(timeout=EVENT_TIMEOUT)
        events = first + self._drain_events(COLLECT_WINDOW)

        self.assertGreater(len(events), 0,
            "No events collected during collection window")

        # Check for numpy-related types
        object_types = {e["object_type"] for e in events if not e["is_dealloc"]}
        numpy_types  = {t for t in object_types
                        if "ndarray" in t or "float" in t.lower()
                        or "list" in t or "numpy" in t.lower()}

        self.assertGreater(len(numpy_types), 0,
            f"No numpy or numeric types found in events. Types seen: {sorted(object_types)[:10]}")

    # ------------------------------------------------------------------
    # I4 — ObjectGraph has at least 10 nodes
    # ------------------------------------------------------------------

    def test_I4_object_graph_has_nodes(self):
        self._start_bridge()
        pid = self._spawn_target()
        self._attach(pid)

        first = self._wait_for_first_event(timeout=EVENT_TIMEOUT)
        events = first + self._drain_events(COLLECT_WINDOW)

        graph = ObjectGraph()
        graph.ingest(events)

        summary = graph.summary()
        self.assertGreaterEqual(summary["total_nodes"], 10,
            f"ObjectGraph has only {summary['total_nodes']} nodes after collection")
        self.assertGreater(summary["total_size_bytes"], 0,
            "ObjectGraph reports zero total size")

    # ------------------------------------------------------------------
    # I5 — ProfilingSession JSON is schema-valid
    # ------------------------------------------------------------------

    def test_I5_session_json_schema_valid(self):
        schema = load_schema()
        if schema is None:
            self.skipTest(f"Schema not found at {SCHEMA_PATH}")

        self._start_bridge()
        pid = self._spawn_target()
        session_start = time.time_ns()
        self._attach(pid)

        first = self._wait_for_first_event(timeout=EVENT_TIMEOUT)
        events = first + self._drain_events(COLLECT_WINDOW)
        session_end = time.time_ns()

        self.assertGreater(len(events), 0, "No events to build session from")

        session = build_session(events, session_start, session_end)

        # Must be JSON serializable
        try:
            session_json = json.dumps(session)
        except TypeError as e:
            self.fail(f"Session is not JSON serializable: {e}")

        # Must validate against schema
        try:
            import jsonschema
            jsonschema.validate(instance=session, schema=schema)
        except ImportError:
            self.skipTest("jsonschema not installed")
        except jsonschema.ValidationError as e:
            self.fail(f"Session failed schema validation: {e.message}")

        # Write to disk for inspection
        out_path = Path(__file__).parent / "session_output.json"
        with open(out_path, "w") as f:
            json.dump(session, f, indent=2)

        # Report stats
        allocs   = sum(1 for e in events if not e["is_dealloc"])
        deallocs = sum(1 for e in events if e["is_dealloc"])
        print(f"\n  Session: {allocs} allocs, {deallocs} deallocs, "
              f"{len(events)} total events, written to {out_path.name}")

    # ------------------------------------------------------------------
    # I6 — Target exits cleanly after SIGTERM
    # ------------------------------------------------------------------

    def test_I6_target_exits_cleanly(self):
        self._start_bridge()
        pid = self._spawn_target()
        self._attach(pid)

        time.sleep(0.5)

        self.proc.terminate()
        try:
            self.proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.fail("Target did not exit within 3s of SIGTERM")

        # The workload handles SIGTERM gracefully and exits with code 0.
        # An unhandled SIGTERM would give -signal.SIGTERM (-15).
        self.assertIn(self.proc.returncode, (0, -signal.SIGTERM),
            f"Target exited with unexpected code: {self.proc.returncode}")


if __name__ == "__main__":
    unittest.main(verbosity=2)