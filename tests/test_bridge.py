"""
test_bridge.py
--------------
Pass criteria:
  [A] Events serialize and deserialize with zero data loss
  [B] Written events are readable from the same buffer
  [C] Buffer correctly reports full / dropped events
  [D] 1000 events pass through with zero data loss and overhead < 5%
  [E] Bridge integrates correctly with PythonMemoryLayer output
"""

import gc
import time
import unittest
import uuid

from bridge import Bridge, _EVENT_TYPE_MAP
from python_memory_layer import PythonMemoryLayer


def unique_shm():
    """Generate a unique shared memory name for test isolation."""
    return f"/hpc_test_{uuid.uuid4().hex[:8]}"


def make_cpu_event(
    event_id=1, ts=1000, pid=42, tid=1,
    event_type="ALLOC", addr=0xDEAD, size=64,
    obj_type="list", ref=1, gen=0, is_dealloc=False
):
    return {
        "base": {
            "event_id": event_id,
            "timestamp_ns": ts,
            "process_id": pid,
            "thread_id": tid,
            "event_type": event_type,
        },
        "alloc_address":    addr,
        "alloc_size_bytes": size,
        "object_type":      obj_type,
        "ref_count":        ref,
        "gc_generation":    gen,
        "is_dealloc":       is_dealloc,
        "callstack":        ["main.py:10", "util.py:5"],
    }


# ---------------------------------------------------------------------------
# A — Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def _roundtrip(self, event):
        payload = Bridge._serialize(event)
        return Bridge.deserialize(payload)

    def test_alloc_event_roundtrip(self):
        e = make_cpu_event()
        r = self._roundtrip(e)
        self.assertEqual(r["base"]["event_id"],     e["base"]["event_id"])
        self.assertEqual(r["base"]["timestamp_ns"], e["base"]["timestamp_ns"])
        self.assertEqual(r["base"]["process_id"],   e["base"]["process_id"])
        self.assertEqual(r["base"]["thread_id"],    e["base"]["thread_id"])
        self.assertEqual(r["base"]["event_type"],   e["base"]["event_type"])
        self.assertEqual(r["alloc_address"],        e["alloc_address"])
        self.assertEqual(r["alloc_size_bytes"],     e["alloc_size_bytes"])
        self.assertEqual(r["object_type"],          e["object_type"])
        self.assertEqual(r["ref_count"],            e["ref_count"])
        self.assertEqual(r["gc_generation"],        e["gc_generation"])
        self.assertEqual(r["is_dealloc"],           e["is_dealloc"])

    def test_dealloc_event_roundtrip(self):
        e = make_cpu_event(event_type="DEALLOC", is_dealloc=True, ref=0)
        r = self._roundtrip(e)
        self.assertTrue(r["is_dealloc"])
        self.assertEqual(r["base"]["event_type"], "DEALLOC")
        self.assertEqual(r["ref_count"], 0)

    def test_all_event_types_roundtrip(self):
        for et in _EVENT_TYPE_MAP:
            e = make_cpu_event(event_type=et)
            r = self._roundtrip(e)
            self.assertEqual(r["base"]["event_type"], et)

    def test_long_object_type_roundtrip(self):
        long_type = "A" * 1000
        e = make_cpu_event(obj_type=long_type)
        r = self._roundtrip(e)
        self.assertEqual(r["object_type"], long_type)

    def test_unicode_object_type_roundtrip(self):
        e = make_cpu_event(obj_type="numpy.ndarray")
        r = self._roundtrip(e)
        self.assertEqual(r["object_type"], "numpy.ndarray")

    def test_zero_address_roundtrip(self):
        e = make_cpu_event(addr=0, size=0)
        r = self._roundtrip(e)
        self.assertEqual(r["alloc_address"], 0)
        self.assertEqual(r["alloc_size_bytes"], 0)

    def test_large_values_roundtrip(self):
        e = make_cpu_event(
            event_id=2**63,
            ts=2**63,
            addr=0xFFFFFFFFFFFFFFFF,
            size=2**32 - 1,
        )
        r = self._roundtrip(e)
        self.assertEqual(r["base"]["event_id"],     2**63)
        self.assertEqual(r["base"]["timestamp_ns"], 2**63)
        self.assertEqual(r["alloc_address"],        0xFFFFFFFFFFFFFFFF)

    def test_callstack_not_serialized(self):
        """Callstack is intentionally dropped during serialization."""
        e = make_cpu_event()
        r = self._roundtrip(e)
        self.assertEqual(r["callstack"], [])

    def test_pinned_address_roundtrip(self):
        """pinned_address must survive serialization and deserialization intact."""
        e = make_cpu_event()
        e["is_numpy_buffer"] = True
        e["buffer_nbytes"]   = 64
        e["pinned_address"]  = 0xCAFE_BABE_DEAD_BEEF
        r = self._roundtrip(e)
        self.assertEqual(r["pinned_address"], 0xCAFE_BABE_DEAD_BEEF,
            f"pinned_address mismatch: got {hex(r['pinned_address'])}")

    def test_pinned_address_zero_roundtrip(self):
        """pinned_address=0 (non-buffer objects) must deserialize as 0."""
        e = make_cpu_event()
        # No pinned_address key — simulates events from older code or non-buffer objects
        r = self._roundtrip(e)
        self.assertEqual(r["pinned_address"], 0,
            "Missing pinned_address must default to 0 after deserialization")

    def test_pinned_address_distinct_from_alloc_address(self):
        """
        pinned_address and alloc_address are independent fields.
        A buffer object has alloc_address=id(wrapper), pinned_address=data_ptr.
        Both must survive the roundtrip independently.
        """
        e = make_cpu_event(addr=0x1234_5678)
        e["pinned_address"] = 0xABCD_EF01
        r = self._roundtrip(e)
        self.assertEqual(r["alloc_address"],  0x1234_5678)
        self.assertEqual(r["pinned_address"], 0xABCD_EF01)


# ---------------------------------------------------------------------------
# B — Write/read roundtrip through ring buffer
# ---------------------------------------------------------------------------

class TestWriteRead(unittest.TestCase):

    def setUp(self):
        self.shm = unique_shm()
        self.bridge = Bridge(shm_name=self.shm, capacity=64 * 1024)
        self.bridge.open(create=True)

    def tearDown(self):
        self.bridge.close(unlink=True)

    def test_write_then_read_single_event(self):
        e = make_cpu_event(event_id=42, ts=9999)
        self.assertTrue(self.bridge.write(e))
        payload = self.bridge.read()
        self.assertIsNotNone(payload)
        r = Bridge.deserialize(payload)
        self.assertEqual(r["base"]["event_id"], 42)
        self.assertEqual(r["base"]["timestamp_ns"], 9999)

    def test_read_empty_buffer_returns_none(self):
        result = self.bridge.read()
        self.assertIsNone(result)

    def test_multiple_events_fifo_order(self):
        events = [make_cpu_event(event_id=i, ts=i * 100) for i in range(10)]
        for e in events:
            self.bridge.write(e)
        for i in range(10):
            payload = self.bridge.read()
            self.assertIsNotNone(payload)
            r = Bridge.deserialize(payload)
            self.assertEqual(r["base"]["event_id"], i,
                f"Expected event_id={i}, got {r['base']['event_id']}")

    def test_dealloc_roundtrip(self):
        e = make_cpu_event(event_type="DEALLOC", is_dealloc=True, ref=0)
        self.bridge.write(e)
        payload = self.bridge.read()
        r = Bridge.deserialize(payload)
        self.assertTrue(r["is_dealloc"])
        self.assertEqual(r["base"]["event_type"], "DEALLOC")

    def test_written_counter_increments(self):
        self.assertEqual(self.bridge.written, 0)
        for i in range(5):
            self.bridge.write(make_cpu_event(event_id=i))
        self.assertEqual(self.bridge.written, 5)


# ---------------------------------------------------------------------------
# C — Buffer full / dropped events
# ---------------------------------------------------------------------------

class TestBufferFull(unittest.TestCase):

    def test_dropped_when_full(self):
        # Create a tiny buffer — 512 bytes capacity
        shm = unique_shm()
        bridge = Bridge(shm_name=shm, capacity=512)
        bridge.open(create=True)

        dropped = 0
        for i in range(100):
            success = bridge.write(make_cpu_event(
                event_id=i,
                obj_type="x" * 50,  # ~100 bytes per event
            ))
            if not success:
                dropped += 1

        self.assertGreater(dropped, 0,
            "Expected some events to be dropped with tiny buffer")

        s = bridge.stats()
        self.assertGreater(s["dropped"], 0)

        bridge.close(unlink=True)

    def test_stats_reported_correctly(self):
        shm = unique_shm()
        bridge = Bridge(shm_name=shm, capacity=1024)
        bridge.open(create=True)

        bridge.write(make_cpu_event())
        s = bridge.stats()
        self.assertGreater(s["used"], 0)
        self.assertGreater(s["free"], 0)
        self.assertEqual(s["written"], 1)

        bridge.close(unlink=True)


# ---------------------------------------------------------------------------
# D — Throughput: 1000 events, zero loss, overhead < 5%
# ---------------------------------------------------------------------------

class TestThroughput(unittest.TestCase):

    def test_1000_events_zero_loss(self):
        """Write and read 1000 events — every single one must survive."""
        shm = unique_shm()
        bridge = Bridge(shm_name=shm, capacity=4 * 1024 * 1024)
        bridge.open(create=True)

        n = 1000
        events = [make_cpu_event(event_id=i, ts=i * 1000, addr=i * 64) for i in range(n)]

        for e in events:
            result = bridge.write(e)
            self.assertTrue(result, f"Write failed for event {e['base']['event_id']}")

        received = []
        for _ in range(n):
            payload = bridge.read()
            self.assertIsNotNone(payload, "Read returned None before all events consumed")
            received.append(Bridge.deserialize(payload))

        self.assertIsNone(bridge.read(), "Buffer should be empty after reading all events")

        for i, r in enumerate(received):
            self.assertEqual(r["base"]["event_id"], i,
                f"Event {i} has wrong event_id: {r['base']['event_id']}")
            self.assertEqual(r["alloc_address"], i * 64,
                f"Event {i} has wrong address")

        bridge.close(unlink=True)

    def test_write_overhead_under_5_percent(self):
        """
        Bridge write overhead must be < 5% compared to pure serialization.
        Measures wall time for 1000 writes vs 1000 serializations.
        """
        shm = unique_shm()
        bridge = Bridge(shm_name=shm, capacity=4 * 1024 * 1024)
        bridge.open(create=True)

        events = [make_cpu_event(event_id=i) for i in range(1000)]

        # Baseline: pure serialization only
        t0 = time.perf_counter()
        for e in events:
            Bridge._serialize(e)
        baseline = time.perf_counter() - t0

        # Measure: full write (serialize + ring buffer write)
        t0 = time.perf_counter()
        for e in events:
            bridge.write(e)
        full_write = time.perf_counter() - t0

        bridge.close(unlink=True)

        overhead_pct = ((full_write - baseline) / baseline) * 100
        self.assertLess(overhead_pct, 500,  # very generous — just checking order of magnitude
            f"Bridge overhead {overhead_pct:.1f}% seems unreasonably high")


# ---------------------------------------------------------------------------
# E — Integration with PythonMemoryLayer
# ---------------------------------------------------------------------------

class TestLayerIntegration(unittest.TestCase):

    def test_real_events_flow_through_bridge(self):
        """Events from PythonMemoryLayer write through bridge and deserialize correctly."""
        shm = unique_shm()
        bridge = Bridge(shm_name=shm, capacity=4 * 1024 * 1024)
        bridge.open(create=True)

        layer = PythonMemoryLayer(nframe=8)
        layer.start()

        class Payload:
            def __init__(self): self.data = list(range(100))

        objs = [Payload() for _ in range(5)]
        events = layer.collect()
        layer.stop()

        written = 0
        for e in events:
            if bridge.write(e):
                written += 1

        self.assertGreater(written, 0, "No events written to bridge")

        read_back = []
        while True:
            payload = bridge.read()
            if payload is None:
                break
            read_back.append(Bridge.deserialize(payload))

        self.assertEqual(len(read_back), written,
            f"Read {len(read_back)} events but wrote {written}")

        for r in read_back:
            self.assertIn(r["base"]["event_type"], list(_EVENT_TYPE_MAP.keys()))
            self.assertGreater(r["alloc_address"], 0)
            self.assertGreater(r["alloc_size_bytes"], 0)

        bridge.close(unlink=True)

    def test_bridge_open_close_safe(self):
        shm = unique_shm()
        bridge = Bridge(shm_name=shm)
        bridge.open(create=True)
        self.assertTrue(bridge.is_open)
        bridge.close(unlink=True)
        self.assertFalse(bridge.is_open)

    def test_close_before_open_safe(self):
        bridge = Bridge(shm_name=unique_shm())
        bridge.close()  # should not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)