"""
test_correlator_integration.py
-------------------------------
End-to-end integration test for the correlator.

Uses mock_event_generator to produce a realistic ProfilingSession,
strips the pre-computed correlated_events, runs the correlator, and
validates that the correlator re-discovers the known HARD matches.

Pass criteria:
  [CI-1] Correlator recovers >= 80% of ADDRESS_MATCH pairs from mock session
  [CI-2] All CorrelatedEvent dicts have required keys and correct types
  [CI-3] No CorrelatedEvent references a non-existent cpu_event_id or gpu_event_id
  [CI-4] Output session is schema-valid (if jsonschema is available)
  [CI-5] Correlator.ingest() + build_session() round-trip preserves event counts
"""

import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "correlator"))
sys.path.insert(0, str(ROOT / "tools"))

SCHEMA_PATH = ROOT / "hpc_profiler_schema.json"


def load_schema():
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH) as f:
        return json.load(f)


class TestCorrelatorIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate a mock session once for all tests in this class."""
        from mock_event_generator import generate_session
        from correlator import Correlator

        # 100 alloc pairs → ~60 ADDRESS_MATCH pairs (mock generator uses 60% rate)
        cls.mock_session = generate_session(n_alloc_pairs=100)
        cls.cpu_events   = cls.mock_session["cpu_events"]
        cls.gpu_events   = cls.mock_session["gpu_events"]

        # Ground-truth HARD pairs from the mock generator
        cls.ground_truth = {
            (c["cpu_event_id"], c["gpu_event_id"])
            for c in cls.mock_session["correlated_events"]
            if c["confidence"] == "HARD"
        }

        # Run correlator on stripped session (no pre-computed correlated_events)
        c = Correlator(time_window_ns=10_000_000)   # 10 ms — generous for mock data
        c.ingest(cls.cpu_events, cls.gpu_events)
        cls.result_session   = c.build_session()
        cls.correlated       = cls.result_session["correlated_events"]

    # ------------------------------------------------------------------
    # CI-1 — Recovery rate
    # ------------------------------------------------------------------

    def test_CI1_hard_match_recovery_rate(self):
        """Correlator must recover >= 80% of known HARD pairs."""
        recovered = {
            (c["cpu_event_id"], c["gpu_event_id"])
            for c in self.correlated
            if c["confidence"] == "HARD"
        }
        if not self.ground_truth:
            self.skipTest("Mock session has no HARD ground truth pairs")

        overlap = len(recovered & self.ground_truth)
        rate = overlap / len(self.ground_truth)
        self.assertGreaterEqual(rate, 0.80,
            f"Recovery rate {rate:.1%} below 80% "
            f"({overlap}/{len(self.ground_truth)} pairs found)")

    # ------------------------------------------------------------------
    # CI-2 — CorrelatedEvent structure
    # ------------------------------------------------------------------

    def test_CI2_correlated_event_keys_and_types(self):
        """Every CorrelatedEvent must have correct keys and value types."""
        required = {
            "cpu_event_id": int,
            "gpu_event_id": int,
            "confidence":   str,
            "match_reason": str,
            "latency_ns":   int,
        }
        valid_confidences  = {"HARD", "WEAK"}
        valid_reasons      = {"ADDRESS_MATCH", "SIZE_AND_TIMESTAMP", "TIMESTAMP_ONLY"}

        for i, c in enumerate(self.correlated):
            for key, typ in required.items():
                self.assertIn(key, c, f"[event {i}] missing key '{key}'")
                self.assertIsInstance(c[key], typ,
                    f"[event {i}] '{key}' is {type(c[key]).__name__}, expected {typ.__name__}")
            self.assertIn(c["confidence"],  valid_confidences,
                f"[event {i}] unknown confidence '{c['confidence']}'")
            self.assertIn(c["match_reason"], valid_reasons,
                f"[event {i}] unknown match_reason '{c['match_reason']}'")

    # ------------------------------------------------------------------
    # CI-3 — No dangling references
    # ------------------------------------------------------------------

    def test_CI3_no_dangling_event_ids(self):
        """CorrelatedEvents must only reference event_ids that actually exist."""
        cpu_ids = {e["base"]["event_id"] for e in self.cpu_events}
        gpu_ids = {e["base"]["event_id"] for e in self.gpu_events}

        for i, c in enumerate(self.correlated):
            self.assertIn(c["cpu_event_id"], cpu_ids,
                f"[event {i}] cpu_event_id {c['cpu_event_id']} not in cpu_events")
            self.assertIn(c["gpu_event_id"], gpu_ids,
                f"[event {i}] gpu_event_id {c['gpu_event_id']} not in gpu_events")

    # ------------------------------------------------------------------
    # CI-4 — Schema validation
    # ------------------------------------------------------------------

    def test_CI4_schema_valid(self):
        """build_session() output must validate against the JSON schema."""
        schema = load_schema()
        if schema is None:
            self.skipTest(f"Schema not found at {SCHEMA_PATH}")
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")

        try:
            jsonschema.validate(instance=self.result_session, schema=schema)
        except jsonschema.ValidationError as e:
            self.fail(f"Schema validation failed: {e.message}")

    # ------------------------------------------------------------------
    # CI-5 — Round-trip event count preservation
    # ------------------------------------------------------------------

    def test_CI5_event_counts_preserved(self):
        """build_session() must include all original cpu/gpu events unchanged."""
        self.assertEqual(
            len(self.result_session["cpu_events"]),
            len(self.cpu_events),
            "CPU event count changed after correlation"
        )
        self.assertEqual(
            len(self.result_session["gpu_events"]),
            len(self.gpu_events),
            "GPU event count changed after correlation"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)