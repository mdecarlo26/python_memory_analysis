"""
test_correlator.py
------------------
Pass criteria:
  [C1] ADDRESS_MATCH — GPU transfer src/dst within a CPU alloc range → HARD
  [C2] SIZE_AND_TIMESTAMP — matching size + timestamp window → HARD
  [C3] TIMESTAMP_ONLY — kernel events correlate weakly by proximity → WEAK
  [C4] No false positives — unrelated events produce no match
  [C5] HARD match priority — ADDRESS_MATCH wins over SIZE_AND_TIMESTAMP
       for the same CPU/GPU event pair
  [C6] Each CPU alloc consumed by at most one HARD match
  [C7] GPU events not in exclude set after WEAK match (multiple GPU → same CPU allowed)
  [C8] latency_ns = gpu_ts - cpu_ts (may be negative if GPU timestamp precedes alloc)
  [C9] Ingest path works: Correlator.ingest() + build_session() → valid ProfilingSession
  [C10] Empty inputs produce empty correlated list
  [C11] Large batch: 1000 CPU allocs × 800 GPU transfers, correct match count, < 1s
"""

import time
import uuid
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "correlator"))
sys.path.insert(0, str(ROOT / "src" / "python"))

from correlator import Correlator, _run_correlation, _binary_search_interval, _LiveAlloc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_eid = 0
def _next_eid():
    global _eid
    _eid += 1
    return _eid


def cpu_alloc(address, size, ts=None, obj_type="numpy.ndarray"):
    global _eid
    eid = _next_eid()
    return {
        "base": {
            "event_id":    eid,
            "timestamp_ns": ts if ts is not None else eid * 1_000,
            "process_id":  1,
            "thread_id":   1,
            "event_type":  "ALLOC",
        },
        "alloc_address":    address,
        "alloc_size_bytes": size,
        "object_type":      obj_type,
        "ref_count":        1,
        "gc_generation":    0,
        "is_dealloc":       False,
        "callstack":        [],
        "pinned_address":   0,
        "tracemalloc_size": size,
        "array_shape":      [],
        "array_dtype":      "",
    }


def cpu_dealloc(address, size, ts=None):
    eid = _next_eid()
    return {
        "base": {
            "event_id":    eid,
            "timestamp_ns": ts if ts is not None else eid * 1_000,
            "process_id":  1,
            "thread_id":   1,
            "event_type":  "DEALLOC",
        },
        "alloc_address":    address,
        "alloc_size_bytes": size,
        "object_type":      "numpy.ndarray",
        "ref_count":        0,
        "gc_generation":    0,
        "is_dealloc":       True,
        "callstack":        [],
        "pinned_address":   0,
        "tracemalloc_size": 0,
        "array_shape":      [],
        "array_dtype":      "",
    }


def gpu_transfer(src=0, dst=0, size=0, ts=None, kind="HOST_TO_DEVICE"):
    eid = _next_eid()
    return {
        "base": {
            "event_id":    eid,
            "timestamp_ns": ts if ts is not None else eid * 1_000,
            "process_id":  1,
            "thread_id":   0,
            "event_type":  "TRANSFER",
        },
        "device_id":           0,
        "src_address":         src,
        "dst_address":         dst,
        "transfer_size_bytes": size,
        "transfer_kind":       kind,
        "um_page_faults":      0,
        "kernel_name":         "",
        "kernel_duration_ns":  0,
        "stream_id":           0,
        "device_mem_used_mb":  0,
        "um_bytes_htod":       size if kind == "HOST_TO_DEVICE" else 0,
        "um_bytes_dtoh":       size if kind == "DEVICE_TO_HOST" else 0,
        "grid":                {"x": 0, "y": 0, "z": 0},
        "block":               {"x": 0, "y": 0, "z": 0},
        "registers_per_thread": 0,
        "shared_mem_bytes":    0,
        "correlation_id":      eid,
    }


def gpu_kernel(ts=None, name="volta_sgemm"):
    eid = _next_eid()
    return {
        "base": {
            "event_id":    eid,
            "timestamp_ns": ts if ts is not None else eid * 1_000,
            "process_id":  1,
            "thread_id":   0,
            "event_type":  "KERNEL",
        },
        "device_id":           0,
        "src_address":         0,
        "dst_address":         0,
        "transfer_size_bytes": 0,
        "transfer_kind":       "HOST_TO_DEVICE",
        "um_page_faults":      0,
        "kernel_name":         name,
        "kernel_duration_ns":  50_000,
        "stream_id":           0,
        "device_mem_used_mb":  0,
        "um_bytes_htod":       0,
        "um_bytes_dtoh":       0,
        "grid":                {"x": 128, "y": 1, "z": 1},
        "block":               {"x": 256, "y": 1, "z": 1},
        "registers_per_thread": 32,
        "shared_mem_bytes":    4096,
        "correlation_id":      eid,
    }


WINDOW = 5_000_000   # 5 ms


# ---------------------------------------------------------------------------
# C1 — ADDRESS_MATCH
# ---------------------------------------------------------------------------

class TestAddressMatch(unittest.TestCase):

    def test_src_address_within_alloc_range(self):
        """GPU transfer src falls inside CPU alloc → HARD / ADDRESS_MATCH."""
        alloc = cpu_alloc(address=0x1000_0000, size=4096, ts=100)
        # src_address is the base address of the allocation
        transfer = gpu_transfer(src=0x1000_0000, size=4096, ts=200)

        result = _run_correlation([alloc], [transfer], WINDOW)

        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertEqual(c["confidence"],   "HARD")
        self.assertEqual(c["match_reason"], "ADDRESS_MATCH")
        self.assertEqual(c["cpu_event_id"], alloc["base"]["event_id"])
        self.assertEqual(c["gpu_event_id"], transfer["base"]["event_id"])

    def test_src_address_mid_range(self):
        """GPU transfer src is mid-allocation — still ADDRESS_MATCH."""
        alloc = cpu_alloc(address=0x2000_0000, size=65536, ts=100)
        transfer = gpu_transfer(src=0x2000_0000 + 1024, size=65536, ts=200)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["match_reason"], "ADDRESS_MATCH")

    def test_dst_address_match(self):
        """GPU transfer dst falls inside CPU alloc range → ADDRESS_MATCH."""
        alloc = cpu_alloc(address=0x3000_0000, size=1024, ts=100)
        transfer = gpu_transfer(src=0x9000_0000, dst=0x3000_0000 + 512, size=1024, ts=200)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["match_reason"], "ADDRESS_MATCH")

    def test_address_just_outside_range_no_hard_match(self):
        """
        Address exactly at end_address (exclusive) must NOT produce an
        ADDRESS_MATCH. A WEAK or SIZE_AND_TIMESTAMP match is acceptable
        since size and timestamp may still align.
        """
        alloc = cpu_alloc(address=0x4000_0000, size=1024, ts=100_000)
        outside = gpu_transfer(src=0x4000_0000 + 1024, size=64, ts=100_000 + 200_000)  # == end_address, different size

        result = _run_correlation([alloc], [outside], WINDOW)
        # If there's a result, it must NOT be ADDRESS_MATCH
        for r in result:
            self.assertNotEqual(r["match_reason"], "ADDRESS_MATCH",
                "end_address (exclusive boundary) must not produce ADDRESS_MATCH")

    def test_multiple_allocs_correct_one_matched(self):
        """Only the alloc whose range contains the GPU address matches."""
        a1 = cpu_alloc(address=0x1000, size=256, ts=100)
        a2 = cpu_alloc(address=0x2000, size=256, ts=200)
        a3 = cpu_alloc(address=0x3000, size=256, ts=300)

        transfer = gpu_transfer(src=0x2080, size=64, ts=400)   # inside a2

        result = _run_correlation([a1, a2, a3], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cpu_event_id"], a2["base"]["event_id"])


# ---------------------------------------------------------------------------
# C2 — SIZE_AND_TIMESTAMP
# ---------------------------------------------------------------------------

class TestSizeAndTimestamp(unittest.TestCase):

    def test_size_and_time_match(self):
        """Same size + GPU ts within window → HARD / SIZE_AND_TIMESTAMP."""
        ts_alloc   = 1_000_000
        ts_transfer = ts_alloc + 1_000_000   # 1 ms later — well within 5 ms window
        alloc    = cpu_alloc(address=0xAAAA_0000, size=16384, ts=ts_alloc)
        transfer = gpu_transfer(src=0xBBBB_0000, dst=0x7000_0000, size=16384, ts=ts_transfer)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertEqual(c["confidence"],   "HARD")
        self.assertEqual(c["match_reason"], "SIZE_AND_TIMESTAMP")

    def test_size_match_but_outside_window(self):
        """Same size but GPU ts too far from alloc ts → no HARD match."""
        ts_alloc   = 1_000_000
        ts_transfer = ts_alloc + 100_000_000   # 100 ms — outside 5 ms window
        alloc    = cpu_alloc(address=0xAAAA_0000, size=16384, ts=ts_alloc)
        transfer = gpu_transfer(src=0xCCCC_0000, size=16384, ts=ts_transfer)

        result = _run_correlation([alloc], [transfer], WINDOW)
        # May get a WEAK match from pass 3, but NOT a SIZE_AND_TIMESTAMP HARD
        hard = [r for r in result if r["match_reason"] == "SIZE_AND_TIMESTAMP"]
        self.assertEqual(len(hard), 0)

    def test_size_zero_not_matched(self):
        """Zero-size GPU transfer must not match on size."""
        alloc    = cpu_alloc(address=0x5000, size=0,  ts=100)
        transfer = gpu_transfer(src=0x9000, size=0, ts=200)

        result = _run_correlation([alloc], [transfer], WINDOW)
        hard = [r for r in result if r["match_reason"] == "SIZE_AND_TIMESTAMP"]
        self.assertEqual(len(hard), 0)


# ---------------------------------------------------------------------------
# C3 — TIMESTAMP_ONLY (WEAK)
# ---------------------------------------------------------------------------

class TestTimestampOnly(unittest.TestCase):

    def test_kernel_event_gets_weak_match(self):
        """A KERNEL event near a CPU alloc → WEAK / TIMESTAMP_ONLY."""
        ts_alloc  = 1_000_000
        ts_kernel = ts_alloc + 500_000   # 0.5 ms after alloc
        alloc  = cpu_alloc(address=0xDEAD_0000, size=1024, ts=ts_alloc)
        kernel = gpu_kernel(ts=ts_kernel)

        result = _run_correlation([alloc], [kernel], WINDOW)
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertEqual(c["confidence"],   "WEAK")
        self.assertEqual(c["match_reason"], "TIMESTAMP_ONLY")

    def test_multiple_gpu_events_can_weakly_match_same_cpu(self):
        """Multiple GPU events (kernels) may weakly associate with one CPU alloc."""
        ts_alloc = 1_000_000
        alloc = cpu_alloc(address=0x1234_0000, size=512, ts=ts_alloc)
        k1 = gpu_kernel(ts=ts_alloc + 100_000)
        k2 = gpu_kernel(ts=ts_alloc + 200_000)
        k3 = gpu_kernel(ts=ts_alloc + 300_000)

        result = _run_correlation([alloc], [k1, k2, k3], WINDOW)
        weak = [r for r in result if r["confidence"] == "WEAK"]
        # All three kernels may weakly point at the same CPU alloc
        self.assertEqual(len(weak), 3)
        cpu_ids = {r["cpu_event_id"] for r in weak}
        self.assertEqual(cpu_ids, {alloc["base"]["event_id"]})

    def test_kernel_outside_window_no_match(self):
        """Kernel too far in time → no WEAK match either."""
        ts_alloc  = 1_000_000
        ts_kernel = ts_alloc + 50_000_000   # 50 ms — way outside 5 ms window
        alloc  = cpu_alloc(address=0x1000, size=256, ts=ts_alloc)
        kernel = gpu_kernel(ts=ts_kernel)

        result = _run_correlation([alloc], [kernel], WINDOW)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# C4 — No false positives
# ---------------------------------------------------------------------------

class TestNoFalsePositives(unittest.TestCase):

    def test_address_mismatch_no_hard_match(self):
        """
        GPU address completely outside any CPU alloc and timestamp outside
        window → no match at all. Use a distant timestamp to also rule out
        WEAK/SIZE_AND_TIMESTAMP matches.
        """
        alloc    = cpu_alloc(address=0x1000, size=64, ts=100)
        # Unique size (99999) so Pass 2 won't hit; timestamp 100ms away so Pass 3 won't hit
        transfer = gpu_transfer(src=0xFFFF_0000, size=99999, ts=100_000_100)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 0)

    def test_address_mismatch_no_hard_match_with_size_collision(self):
        """
        GPU address outside alloc range but size matches — Pass 2 may produce
        a SIZE_AND_TIMESTAMP HARD match if within window. This is expected
        correct behavior (zero-copy scenario). Ensure it is HARD, not ADDRESS.
        """
        alloc    = cpu_alloc(address=0x1000, size=64, ts=100_000)
        transfer = gpu_transfer(src=0xFFFF_0000, size=64, ts=100_000 + 1_000_000)

        result = _run_correlation([alloc], [transfer], WINDOW)
        if result:
            self.assertEqual(result[0]["match_reason"], "SIZE_AND_TIMESTAMP")
            self.assertEqual(result[0]["confidence"], "HARD")

    def test_dealloc_events_not_used_as_allocs(self):
        """Dealloc events must not participate in correlation as alloc side."""
        dealloc  = cpu_dealloc(address=0x1000, size=256, ts=100)
        transfer = gpu_transfer(src=0x1000, size=256, ts=200)

        # Only pass dealloc — no allocs
        result = _run_correlation([dealloc], [transfer], WINDOW)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# C5 — ADDRESS_MATCH wins over SIZE_AND_TIMESTAMP for same pair
# ---------------------------------------------------------------------------

class TestHardPriority(unittest.TestCase):

    def test_address_match_wins(self):
        """
        When an alloc+transfer pair qualifies for both ADDRESS_MATCH and
        SIZE_AND_TIMESTAMP, only ADDRESS_MATCH is emitted.
        """
        ts_alloc   = 1_000_000
        ts_transfer = ts_alloc + 500_000
        alloc    = cpu_alloc(address=0x8000_0000, size=4096, ts=ts_alloc)
        transfer = gpu_transfer(
            src=0x8000_0000,   # address match
            size=4096,         # also size match
            ts=ts_transfer,
        )

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["match_reason"], "ADDRESS_MATCH")


# ---------------------------------------------------------------------------
# C6 — Each CPU alloc consumed by at most one HARD match
# ---------------------------------------------------------------------------

class TestHardExclusivity(unittest.TestCase):

    def test_one_cpu_alloc_not_doubly_hard_matched(self):
        """Two GPU transfers both address-match same alloc → only first is HARD."""
        ts_alloc = 1_000_000
        alloc = cpu_alloc(address=0x6000_0000, size=8192, ts=ts_alloc)

        t1 = gpu_transfer(src=0x6000_0000,        size=8192, ts=ts_alloc + 100_000)
        t2 = gpu_transfer(src=0x6000_0000 + 100,  size=8192, ts=ts_alloc + 200_000)

        result = _run_correlation([alloc], [t1, t2], WINDOW)
        hard = [r for r in result if r["confidence"] == "HARD"]
        self.assertEqual(len(hard), 1)

    def test_two_allocs_two_transfers_both_matched(self):
        """Two distinct allocs with address-matching transfers → 2 HARD matches."""
        a1 = cpu_alloc(address=0x1000_0000, size=1024, ts=100_000)
        a2 = cpu_alloc(address=0x2000_0000, size=2048, ts=200_000)
        t1 = gpu_transfer(src=0x1000_0000, size=1024, ts=150_000)
        t2 = gpu_transfer(src=0x2000_0000, size=2048, ts=250_000)

        result = _run_correlation([a1, a2], [t1, t2], WINDOW)
        hard = [r for r in result if r["confidence"] == "HARD"]
        self.assertEqual(len(hard), 2)


# ---------------------------------------------------------------------------
# C7 — WEAK match does not block other GPU events
# ---------------------------------------------------------------------------

class TestWeakNonExclusive(unittest.TestCase):

    def test_weak_match_gpu_id_not_reused(self):
        """Each GPU event appears in at most one CorrelatedEvent."""
        ts_alloc = 1_000_000
        alloc  = cpu_alloc(address=0xAAAA, size=64, ts=ts_alloc)
        kernel = gpu_kernel(ts=ts_alloc + 1_000)

        result = _run_correlation([alloc], [kernel], WINDOW)
        gpu_ids = [r["gpu_event_id"] for r in result]
        self.assertEqual(len(gpu_ids), len(set(gpu_ids)), "Duplicate GPU event_id in result")


# ---------------------------------------------------------------------------
# C8 — latency_ns
# ---------------------------------------------------------------------------

class TestLatency(unittest.TestCase):

    def test_latency_positive(self):
        """Normal case: GPU event after CPU alloc → positive latency."""
        alloc    = cpu_alloc(address=0x1000, size=64, ts=1_000_000)
        transfer = gpu_transfer(src=0x1000, size=64, ts=1_500_000)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["latency_ns"], 500_000)

    def test_latency_skew_clamped_to_zero(self):
        """
        GPU timestamp slightly before CPU alloc (clock skew / async flush)
        → raw delta is negative but latency_ns is clamped to 0 for schema
        compliance. Match is still emitted.
        """
        alloc    = cpu_alloc(address=0x2000, size=128, ts=2_000_000)
        transfer = gpu_transfer(src=0x2000, size=128, ts=1_999_000)

        result = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["latency_ns"], 0)


# ---------------------------------------------------------------------------
# C9 — Correlator.ingest + build_session
# ---------------------------------------------------------------------------

class TestIngestAndBuildSession(unittest.TestCase):

    def test_build_session_shape(self):
        """build_session() returns a dict with all required ProfilingSession keys."""
        alloc    = cpu_alloc(address=0x9000_0000, size=4096, ts=1_000_000)
        transfer = gpu_transfer(src=0x9000_0000, size=4096, ts=1_500_000)

        c = Correlator()
        c.ingest([alloc], [transfer])
        session = c.build_session()

        for key in ("session_id", "schema_version", "start_time_ns", "end_time_ns",
                    "cpu_events", "gpu_events", "correlated_events"):
            self.assertIn(key, session, f"Missing key: {key}")

        self.assertEqual(session["schema_version"], "0.1")
        self.assertIsInstance(session["session_id"], str)
        self.assertEqual(len(session["correlated_events"]), 1)

    def test_session_json_serializable(self):
        """The session dict must be JSON-serializable."""
        import json
        alloc    = cpu_alloc(address=0x7000, size=256, ts=500_000)
        transfer = gpu_transfer(src=0x7000, size=256, ts=600_000)

        c = Correlator()
        c.ingest([alloc], [transfer])
        session = c.build_session()

        try:
            json.dumps(session)
        except TypeError as e:
            self.fail(f"Session not JSON serializable: {e}")


# ---------------------------------------------------------------------------
# C10 — Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs(unittest.TestCase):

    def test_no_cpu_events(self):
        transfer = gpu_transfer(src=0x1000, size=64, ts=100)
        result = _run_correlation([], [transfer], WINDOW)
        self.assertEqual(result, [])

    def test_no_gpu_events(self):
        alloc = cpu_alloc(address=0x1000, size=64, ts=100)
        result = _run_correlation([alloc], [], WINDOW)
        self.assertEqual(result, [])

    def test_both_empty(self):
        result = _run_correlation([], [], WINDOW)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# C11 — Performance: 1000 allocs × 800 transfers < 1 second
# ---------------------------------------------------------------------------

class TestPerformance(unittest.TestCase):

    def test_large_batch_performance(self):
        import random
        rng = random.Random(42)

        base_addr = 0x1000_0000
        alloc_size = 4096

        cpu_events = []
        for i in range(1000):
            addr = base_addr + i * alloc_size * 2   # non-overlapping
            cpu_events.append(cpu_alloc(address=addr, size=alloc_size, ts=i * 10_000))

        gpu_events = []
        # 500 address-matched transfers
        for i in range(0, 500):
            addr = base_addr + i * alloc_size * 2
            gpu_events.append(gpu_transfer(src=addr, size=alloc_size, ts=i * 10_000 + 1_000))
        # 300 kernel events (WEAK candidates)
        for i in range(300):
            gpu_events.append(gpu_kernel(ts=i * 10_000 + 500))

        t0 = time.perf_counter()
        result = _run_correlation(cpu_events, gpu_events, WINDOW)
        elapsed = time.perf_counter() - t0

        hard = [r for r in result if r["confidence"] == "HARD"]
        self.assertEqual(len(hard), 500, f"Expected 500 HARD matches, got {len(hard)}")
        self.assertLess(elapsed, 1.0, f"Correlation took {elapsed:.3f}s — too slow")


# ---------------------------------------------------------------------------
# Binary search helper tests
# ---------------------------------------------------------------------------

class TestBinarySearch(unittest.TestCase):

    def _make_alloc(self, address, size):
        """Create a _LiveAlloc directly for unit testing the search."""
        ev = cpu_alloc(address=address, size=size, ts=0)
        return _LiveAlloc(ev)

    def test_exact_base_address(self):
        allocs = [self._make_alloc(0x1000, 256)]
        result = _binary_search_interval(allocs, 0x1000)
        self.assertIsNotNone(result)
        self.assertEqual(result.address, 0x1000)

    def test_mid_range(self):
        allocs = [self._make_alloc(0x1000, 256)]
        result = _binary_search_interval(allocs, 0x1080)
        self.assertIsNotNone(result)

    def test_end_exclusive(self):
        allocs = [self._make_alloc(0x1000, 256)]
        result = _binary_search_interval(allocs, 0x1100)   # == end_address
        self.assertIsNone(result)

    def test_empty_list(self):
        result = _binary_search_interval([], 0x1000)
        self.assertIsNone(result)

    def test_multiple_allocs_correct_one_found(self):
        allocs = sorted([
            self._make_alloc(0x1000, 256),
            self._make_alloc(0x2000, 256),
            self._make_alloc(0x3000, 256),
        ], key=lambda a: a.address)
        result = _binary_search_interval(allocs, 0x2080)
        self.assertIsNotNone(result)
        self.assertEqual(result.address, 0x2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ---------------------------------------------------------------------------
# C1b — Pass 1b: ADDRESS_MATCH via pinned_address
# ---------------------------------------------------------------------------

def cpu_alloc_pinned(address, size, pinned_address, ts=None):
    """cpu_alloc with pinned_address set (the C-level buffer pointer)."""
    ev = cpu_alloc(address=address, size=size, ts=ts)
    ev["pinned_address"] = pinned_address
    return ev


class TestPinnedAddressMatch(unittest.TestCase):
    """Pass 1b — GPU src/dst matches cpu_event.pinned_address exactly."""

    def test_pinned_address_hard_match(self):
        """GPU src == pinned_address → HARD / ADDRESS_MATCH (Pass 1b)."""
        alloc    = cpu_alloc_pinned(address=0xAAAA_0000, size=64,
                                    pinned_address=0xBEEF_0000, ts=1_000_000)
        transfer = gpu_transfer(src=0xBEEF_0000, size=64, ts=1_500_000)
        result   = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        c = result[0]
        self.assertEqual(c["confidence"],   "HARD")
        self.assertEqual(c["match_reason"], "ADDRESS_MATCH")
        self.assertEqual(c["cpu_event_id"], alloc["base"]["event_id"])
        self.assertEqual(c["gpu_event_id"], transfer["base"]["event_id"])

    def test_pinned_address_dst_match(self):
        """GPU dst == pinned_address → HARD / ADDRESS_MATCH (D2H case)."""
        alloc    = cpu_alloc_pinned(address=0xCCCC_0000, size=128,
                                    pinned_address=0xDEAD_C0DE, ts=500_000)
        transfer = gpu_transfer(src=0x9999_0000, dst=0xDEAD_C0DE,
                                size=128, ts=600_000)
        result   = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"],   "HARD")
        self.assertEqual(result[0]["match_reason"], "ADDRESS_MATCH")

    def test_pinned_address_zero_does_not_match(self):
        """pinned_address=0 (sentinel) must never fire Pass 1b."""
        alloc    = cpu_alloc_pinned(address=0x1111_0000, size=64,
                                    pinned_address=0, ts=1_000_000)
        transfer = gpu_transfer(src=0, dst=0, size=999, ts=1_200_000)
        result   = _run_correlation([alloc], [transfer], WINDOW)
        hard = [r for r in result if r["confidence"] == "HARD"]
        self.assertEqual(len(hard), 0,
            "pinned_address=0 should not produce a HARD match")

    def test_pass1a_wins_no_duplicate_when_both_could_match(self):
        """When alloc_address AND pinned_address both match, only one event emitted."""
        alloc    = cpu_alloc_pinned(address=0x5000_0000, size=4096,
                                    pinned_address=0x5000_0000, ts=1_000_000)
        transfer = gpu_transfer(src=0x5000_0000, size=4096, ts=1_500_000)
        result   = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1,
            "Event matched twice (Pass 1a and 1b duplicate)")
        self.assertEqual(result[0]["confidence"], "HARD")

    def test_pinned_address_correct_latency(self):
        """latency_ns uses CPU alloc timestamp regardless of which pass matched."""
        cpu_ts   = 2_000_000
        gpu_ts   = cpu_ts + 750_000
        alloc    = cpu_alloc_pinned(address=0x7777_0000, size=256,
                                    pinned_address=0xF000_F000, ts=cpu_ts)
        transfer = gpu_transfer(src=0xF000_F000, size=256, ts=gpu_ts)
        result   = _run_correlation([alloc], [transfer], WINDOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["latency_ns"], 750_000)

    def test_multiple_allocs_pinned_correct_one_matched(self):
        """Only the alloc whose pinned_address matches GPU src is correlated."""
        a1 = cpu_alloc_pinned(0x1000, 64, pinned_address=0xAA00, ts=100_000)
        a2 = cpu_alloc_pinned(0x2000, 64, pinned_address=0xBB00, ts=200_000)
        a3 = cpu_alloc_pinned(0x3000, 64, pinned_address=0xCC00, ts=300_000)
        transfer = gpu_transfer(src=0xBB00, size=64, ts=400_000)
        result   = _run_correlation([a1, a2, a3], [transfer], WINDOW)
        hard     = [r for r in result if r["confidence"] == "HARD"]
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0]["cpu_event_id"], a2["base"]["event_id"])


# ---------------------------------------------------------------------------
# GPU new fields: um_bytes, grid, block, registers, shared_mem, correlation_id
# ---------------------------------------------------------------------------

class TestGpuNewFields(unittest.TestCase):
    """Verify that new GPU event fields are present and have correct types
    in the helpers and that the correlator build_session includes them."""

    def test_gpu_transfer_has_um_bytes(self):
        """gpu_transfer helper populates um_bytes_htod for H2D transfers."""
        t = gpu_transfer(src=0x1000, size=8192, kind="HOST_TO_DEVICE")
        self.assertEqual(t["um_bytes_htod"], 8192)
        self.assertEqual(t["um_bytes_dtoh"], 0)

    def test_gpu_transfer_dtoh_um_bytes(self):
        """gpu_transfer helper populates um_bytes_dtoh for D2H transfers."""
        t = gpu_transfer(src=0x1000, size=4096, kind="DEVICE_TO_HOST")
        self.assertEqual(t["um_bytes_dtoh"], 4096)
        self.assertEqual(t["um_bytes_htod"], 0)

    def test_gpu_kernel_has_grid_block_fields(self):
        """gpu_kernel helper populates grid, block, registers, shared_mem."""
        k = gpu_kernel(name="test_kernel")
        self.assertIn("grid",  k)
        self.assertIn("block", k)
        self.assertIn("x", k["grid"])
        self.assertIn("y", k["grid"])
        self.assertIn("z", k["grid"])
        self.assertIn("registers_per_thread", k)
        self.assertIn("shared_mem_bytes",     k)
        self.assertIn("correlation_id",       k)
        # Values should be plausible
        self.assertGreater(k["grid"]["x"],            0)
        self.assertGreater(k["block"]["x"],           0)
        self.assertGreater(k["registers_per_thread"], 0)
        self.assertGreater(k["shared_mem_bytes"],     0)

    def test_build_session_gpu_events_include_new_fields(self):
        """build_session() passes GPU events through with all new fields intact."""
        alloc    = cpu_alloc(address=0x9000_0000, size=4096, ts=1_000_000)
        transfer = gpu_transfer(src=0x9000_0000, size=4096, ts=1_500_000)

        c = Correlator()
        c.ingest([alloc], [transfer])
        session = c.build_session()

        self.assertEqual(len(session["gpu_events"]), 1)
        gpu_ev = session["gpu_events"][0]

        for field in ("um_bytes_htod", "um_bytes_dtoh", "grid", "block",
                      "registers_per_thread", "shared_mem_bytes", "correlation_id"):
            self.assertIn(field, gpu_ev,
                f"GPU event missing field: {field}")

    def test_build_session_cpu_events_include_new_fields(self):
        """build_session() passes CPU events through with all new fields intact."""
        alloc = cpu_alloc(address=0x1000, size=64, ts=100_000)
        alloc["tracemalloc_size"] = 9999
        alloc["array_shape"]      = [128, 64]
        alloc["array_dtype"]      = "float32"

        c = Correlator()
        c.ingest([alloc], [])
        session = c.build_session()

        cpu_ev = session["cpu_events"][0]
        self.assertEqual(cpu_ev["tracemalloc_size"], 9999)
        self.assertEqual(cpu_ev["array_shape"],      [128, 64])
        self.assertEqual(cpu_ev["array_dtype"],      "float32")


if __name__ == "__main__":
    unittest.main(verbosity=2)