"""
mock_event_generator.py
-----------------------
Generates synthetic ProfilingSession data for integration tests.

Produces realistic CPU and GPU events with known HARD correlations so
tests can verify recovery rate, schema validity, and structural correctness
without requiring a GPU or ptrace permissions.

Usage:
    from mock_event_generator import generate_session
    session = generate_session(n_alloc_pairs=100)
"""

import random
import time
import uuid

# ~60% of CPU allocs have a matching GPU transfer (ADDRESS_MATCH)
_HARD_MATCH_RATE = 0.60


def _base(eid, ts, pid=1, tid=1, etype="ALLOC"):
    return {
        "event_id":     eid,
        "timestamp_ns": ts,
        "process_id":   pid,
        "thread_id":    tid,
        "event_type":   etype,
    }


def _cpu_alloc(eid, ts, addr, size, pinned=0):
    return {
        "base":             _base(eid, ts, etype="ALLOC"),
        "alloc_address":    addr,
        "alloc_size_bytes": size,
        "object_type":      "numpy.ndarray",
        "module_name":      "numpy",
        "ref_count":        2,
        "gc_generation":    0,
        "is_dealloc":       False,
        "callstack":        [f"workload.py:{eid % 50 + 1}"],
        "peak_rss_kb":      1024,
        "lifetime_ns":      0,
        "is_numpy_buffer":  bool(pinned),
        "buffer_nbytes":    size if pinned else 0,
        "parent_address":   0,
        "pinned_address":   pinned,
    }


def _gpu_transfer(eid, ts, src, size, kind="HOST_TO_DEVICE"):
    return {
        "base":                _base(eid, ts, etype="TRANSFER"),
        "device_id":           0,
        "src_address":         src,
        "dst_address":         0,
        "transfer_size_bytes": size,
        "transfer_kind":       kind,
        "um_page_faults":      0,
        "kernel_name":         "",
        "kernel_duration_ns":  0,
        "stream_id":           0,
        "device_mem_used_mb":  0,
    }


def _gpu_kernel(eid, ts, name="volta_sgemm"):
    return {
        "base":                _base(eid, ts, etype="KERNEL"),
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
    }


def generate_session(
    n_alloc_pairs: int = 100,
    hard_match_rate: float = _HARD_MATCH_RATE,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic ProfilingSession with known HARD correlations.

    Parameters
    ----------
    n_alloc_pairs   : Number of CPU alloc events to generate.
    hard_match_rate : Fraction that get a matching GPU TRANSFER.
    seed            : RNG seed for reproducibility.

    Returns
    -------
    ProfilingSession dict (schema v0.1).
    """
    rng = random.Random(seed)
    eid = 1

    cpu_events = []
    gpu_events = []
    correlated_events = []

    base_addr  = 0x1000_0000
    alloc_size = 4096
    ts_cpu     = 1_000_000   # 1 ms

    for i in range(n_alloc_pairs):
        # CPU alloc — pinned_address set to the C-buffer ptr for matched allocs
        addr        = base_addr + i * alloc_size * 2   # non-overlapping
        pinned      = addr + 0x4000_0000  # synthetic buffer ptr ≠ alloc_address
        cpu_ev      = _cpu_alloc(eid, ts_cpu, addr=addr, size=alloc_size, pinned=pinned)
        cpu_eid     = eid
        eid        += 1
        ts_cpu     += 5_000   # 5 µs apart

        cpu_events.append(cpu_ev)

        # 60% of allocs get a matching GPU TRANSFER
        if rng.random() < hard_match_rate:
            ts_gpu  = ts_cpu + rng.randint(100_000, 2_000_000)  # 0.1–2 ms later
            # Use the CPU alloc_address as src so Pass 1a fires (address range match)
            gpu_ev  = _gpu_transfer(eid, ts_gpu, src=addr, size=alloc_size)
            gpu_eid = eid
            eid    += 1
            gpu_events.append(gpu_ev)

            correlated_events.append({
                "cpu_event_id": cpu_eid,
                "gpu_event_id": gpu_eid,
                "confidence":   "HARD",
                "match_reason": "ADDRESS_MATCH",
                "latency_ns":   ts_gpu - ts_cpu,
            })

    # Add some kernel events (WEAK match candidates)
    ts_k = 1_000_000
    kernels = ["volta_sgemm_128x64", "relu_forward", "batch_norm_bwd"]
    for i in range(min(20, n_alloc_pairs // 5)):
        gpu_events.append(_gpu_kernel(eid, ts_k + i * 100_000, name=kernels[i % 3]))
        eid += 1

    now = time.time_ns()
    return {
        "session_id":        str(uuid.uuid4()),
        "schema_version":    "0.1",
        "start_time_ns":     now,
        "end_time_ns":       now + 10_000_000,
        "cpu_events":        cpu_events,
        "gpu_events":        gpu_events,
        "correlated_events": correlated_events,
    }