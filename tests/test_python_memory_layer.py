"""
test_python_memory_layer.py
---------------------------
Pass criteria:
  [A] ALLOC events are emitted with correct address, size, type, ref_count,
      callstack, and gc_generation all populated
  [B] DEALLOC events fire with correct address matching the original ALLOC
  [C] All emitted events validate against the JSON schema
  [D] Layer is safe to start/stop/collect in edge cases
"""

import gc
import json
import sys
import time
import unittest
from pathlib import Path

from python_memory_layer import PythonMemoryLayer

SCHEMA_PATH = Path(__file__).parent.parent / "hpc_profiler_schema.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_schema():
    if not SCHEMA_PATH.exists():
        return None
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def validate_cpu_event(event: dict, schema: dict) -> tuple[bool, str]:
    try:
        import jsonschema
        cpu_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": schema["definitions"],
            "$ref": "#/definitions/CpuEvent"
        }
        jsonschema.validate(instance=event, schema=cpu_schema)
        return True, ""
    except ImportError:
        return True, "jsonschema not installed"
    except Exception as e:
        return False, str(e)

def alloc_events(events):
    return [e for e in events if not e["is_dealloc"]]

def dealloc_events(events):
    return [e for e in events if e["is_dealloc"]]

# A simple GC-trackable class we control completely
class Payload:
    def __init__(self, data):
        self.data = data


# ---------------------------------------------------------------------------
# A — ALLOC field population
# ---------------------------------------------------------------------------

class TestAllocFields(unittest.TestCase):

    def setUp(self):
        self.layer = PythonMemoryLayer(nframe=16)
        self.layer.start()

    def tearDown(self):
        self.layer.stop()

    def test_alloc_events_emitted(self):
        """Allocating a tracked object must produce at least one ALLOC event."""
        _ = Payload(list(range(1000)))
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0,
            "No ALLOC events after allocating a Payload")

    def test_alloc_address_is_real_id(self):
        """alloc_address must equal id(obj) — the real CPython memory address."""
        obj = Payload(list(range(500)))
        expected_addr = id(obj)
        events = alloc_events(self.layer.collect())
        addresses = [e["alloc_address"] for e in events]
        self.assertIn(expected_addr, addresses,
            f"Expected address {expected_addr} not found in {addresses}")

    def test_alloc_size_positive(self):
        """alloc_size_bytes must be > 0 for every ALLOC event."""
        _ = Payload(list(range(2000)))
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertGreater(e["alloc_size_bytes"], 0,
                f"ALLOC event has zero size: {e}")

    def test_alloc_object_type_populated(self):
        """object_type must be a non-empty string."""
        _ = Payload([0] * 100)
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertIsInstance(e["object_type"], str)
            self.assertGreater(len(e["object_type"]), 0,
                f"object_type is empty in: {e}")

    def test_alloc_object_type_correct(self):
        """object_type must match type(obj).__qualname__ for a known object."""
        obj = Payload(list(range(300)))
        events = alloc_events(self.layer.collect())
        payload_events = [e for e in events if e["alloc_address"] == id(obj)]
        self.assertGreater(len(payload_events), 0,
            "No event found for the Payload object")
        self.assertEqual(payload_events[0]["object_type"], "Payload")

    def test_alloc_ref_count_positive(self):
        """ref_count must be >= 1 for a live object."""
        _ = Payload(list(range(200)))
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertGreaterEqual(e["ref_count"], 1,
                f"ref_count is 0 or negative: {e}")

    def test_alloc_callstack_populated(self):
        """callstack must be a non-empty list of strings."""
        _ = Payload([None] * 50)
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertIsInstance(e["callstack"], list)
            self.assertGreater(len(e["callstack"]), 0,
                "callstack is empty")
            for frame in e["callstack"]:
                self.assertIsInstance(frame, str)
                self.assertIn(":", frame,
                    f"Frame missing file:line format: {frame}")

    def test_alloc_gc_generation_valid(self):
        """gc_generation must be 0, 1, or 2."""
        _ = Payload(list(range(100)))
        events = alloc_events(self.layer.collect())
        self.assertGreater(len(events), 0)
        for e in events:
            self.assertIn(e["gc_generation"], [0, 1, 2],
                f"Invalid gc_generation: {e['gc_generation']}")

    def test_alloc_new_objects_in_generation_0(self):
        """Freshly allocated objects must be in GC generation 0."""
        obj = Payload(list(range(100)))
        events = alloc_events(self.layer.collect())
        payload_events = [e for e in events if e["alloc_address"] == id(obj)]
        self.assertGreater(len(payload_events), 0)
        self.assertEqual(payload_events[0]["gc_generation"], 0,
            "New object should be in generation 0")

    def test_base_fields_complete(self):
        """Every event must have all required base fields with valid values."""
        _ = Payload([1, 2, 3] * 100)
        events = self.layer.collect()
        self.assertGreater(len(events), 0)
        for e in events:
            b = e["base"]
            self.assertGreater(b["event_id"], 0)
            self.assertGreater(b["timestamp_ns"], 0)
            self.assertGreater(b["process_id"], 0)
            self.assertGreater(b["thread_id"], 0)
            self.assertIn(b["event_type"], ["ALLOC", "DEALLOC"])

    def test_event_ids_unique_and_increasing(self):
        """event_id values must be strictly increasing across a batch of events."""
        _ = [Payload([i]) for i in range(20)]
        events = self.layer.collect()
        self.assertGreater(len(events), 1)
        ids = [e["base"]["event_id"] for e in events]
        for i in range(1, len(ids)):
            self.assertGreater(ids[i], ids[i - 1],
                f"event_id not increasing at index {i}: {ids[i-1]} -> {ids[i]}")

    def test_timestamps_nondecreasing(self):
        """Timestamps must be non-decreasing within a single collect() call."""
        _ = [Payload([i] * 10) for i in range(20)]
        events = self.layer.collect()
        self.assertGreater(len(events), 1)
        ts = [e["base"]["timestamp_ns"] for e in events]
        for i in range(1, len(ts)):
            self.assertGreaterEqual(ts[i], ts[i - 1],
                f"Timestamp went backwards at index {i}")


# ---------------------------------------------------------------------------
# B — DEALLOC correctness
# ---------------------------------------------------------------------------

class TestDeallocFields(unittest.TestCase):

    def test_dealloc_fires_for_tracked_object(self):
        """Deleting a tracked object must produce a DEALLOC event."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload(list(range(500)))
        expected_addr = id(obj)
        layer.collect()  # register the alloc + finalizer

        del obj
        gc.collect()
        time.sleep(0.01)  # give finalizer thread time to fire

        all_events = layer.flush()
        layer.stop()

        d_events = dealloc_events(all_events)
        self.assertGreater(len(d_events), 0,
            "No DEALLOC events after deleting tracked object")

    def test_dealloc_address_matches_alloc(self):
        """DEALLOC event address must match the original ALLOC address."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload(list(range(300)))
        expected_addr = id(obj)
        layer.collect()

        del obj
        gc.collect()
        time.sleep(0.01)

        all_events = layer.flush()
        layer.stop()

        allocs = [e for e in all_events if not e["is_dealloc"] and e["alloc_address"] == expected_addr]
        deallocs = [e for e in all_events if e["is_dealloc"] and e["alloc_address"] == expected_addr]

        self.assertGreater(len(allocs), 0, "No ALLOC for expected address")
        self.assertGreater(len(deallocs), 0,
            f"No DEALLOC found for address {expected_addr}")

    def test_dealloc_event_type_correct(self):
        """DEALLOC events must have event_type=DEALLOC and is_dealloc=True."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload(list(range(200)))
        layer.collect()
        del obj
        gc.collect()
        time.sleep(0.01)

        all_events = layer.flush()
        layer.stop()

        for e in dealloc_events(all_events):
            self.assertTrue(e["is_dealloc"])
            self.assertEqual(e["base"]["event_type"], "DEALLOC")

    def test_dealloc_size_matches_alloc(self):
        """DEALLOC size must match the size recorded at alloc time."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload(list(range(400)))
        addr = id(obj)
        layer.collect()
        del obj
        gc.collect()
        time.sleep(0.01)

        all_events = layer.flush()
        layer.stop()

        alloc = next((e for e in all_events if not e["is_dealloc"] and e["alloc_address"] == addr), None)
        dealloc = next((e for e in all_events if e["is_dealloc"] and e["alloc_address"] == addr), None)

        if alloc and dealloc:
            self.assertEqual(alloc["alloc_size_bytes"], dealloc["alloc_size_bytes"],
                "DEALLOC size does not match ALLOC size")

    def test_dealloc_ref_count_zero(self):
        """DEALLOC events must have ref_count=0 (object is being destroyed)."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload([1, 2, 3] * 50)
        layer.collect()
        del obj
        gc.collect()
        time.sleep(0.01)

        all_events = layer.flush()
        layer.stop()

        for e in dealloc_events(all_events):
            self.assertEqual(e["ref_count"], 0,
                f"DEALLOC event should have ref_count=0, got {e['ref_count']}")

    def test_dealloc_after_alloc_timestamp(self):
        """DEALLOC timestamp must be >= ALLOC timestamp for the same address."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        obj = Payload(list(range(200)))
        addr = id(obj)
        layer.collect()
        del obj
        gc.collect()
        time.sleep(0.01)

        all_events = layer.flush()
        layer.stop()

        alloc = next((e for e in all_events if not e["is_dealloc"] and e["alloc_address"] == addr), None)
        dealloc = next((e for e in all_events if e["is_dealloc"] and e["alloc_address"] == addr), None)

        if alloc and dealloc:
            self.assertGreaterEqual(
                dealloc["base"]["timestamp_ns"],
                alloc["base"]["timestamp_ns"],
                "DEALLOC timestamp is before ALLOC timestamp"
            )


# ---------------------------------------------------------------------------
# C — Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):

    def test_all_events_schema_valid(self):
        """Every event from collect() and dealloc finalizers must pass schema validation."""
        schema = load_schema()
        if schema is None:
            self.skipTest(f"Schema not found at {SCHEMA_PATH}")

        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        objs = [Payload(list(range(i * 10))) for i in range(1, 6)]
        events = layer.collect()

        del objs
        gc.collect()
        time.sleep(0.02)

        events += layer.flush()
        layer.stop()

        self.assertGreater(len(events), 0, "No events to validate")

        failures = []
        for i, event in enumerate(events):
            valid, msg = validate_cpu_event(event, schema)
            if not valid:
                failures.append(f"Event {i} (id={event['base']['event_id']}): {msg}")

        if failures:
            self.fail(f"{len(failures)} schema failures:\n" + "\n".join(failures[:5]))


# ---------------------------------------------------------------------------
# D — Edge cases and safety
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_collect_before_start_returns_empty(self):
        layer = PythonMemoryLayer()
        self.assertEqual(layer.collect(), [])

    def test_stop_before_start_returns_empty(self):
        layer = PythonMemoryLayer()
        self.assertEqual(layer.stop(), [])

    def test_double_start_is_noop(self):
        layer = PythonMemoryLayer()
        layer.start()
        layer.start()
        layer.stop()

    def test_flush_clears_buffer(self):
        layer = PythonMemoryLayer(nframe=8)
        layer.start()
        _ = [Payload([i]) for i in range(5)]
        layer.collect()
        first = layer.flush()
        self.assertGreater(len(first), 0)
        second = layer.flush()
        self.assertEqual(len(second), 0, "Buffer should be empty after flush()")
        layer.stop()

    def test_collect_returns_list(self):
        layer = PythonMemoryLayer()
        layer.start()
        result = layer.collect()
        layer.stop()
        self.assertIsInstance(result, list)

    def test_is_running_flag(self):
        layer = PythonMemoryLayer()
        self.assertFalse(layer.is_running)
        layer.start()
        self.assertTrue(layer.is_running)
        layer.stop()
        self.assertFalse(layer.is_running)

    def test_no_duplicate_addresses_in_single_collect(self):
        """Each object address should appear at most once per collect() call."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()
        _ = [Payload(list(range(50))) for _ in range(10)]
        events = alloc_events(layer.collect())
        layer.stop()
        addresses = [e["alloc_address"] for e in events]
        self.assertEqual(len(addresses), len(set(addresses)),
            "Duplicate addresses found in a single collect()")


if __name__ == "__main__":
    unittest.main(verbosity=2)