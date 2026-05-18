"""
correlator.py
-------------
Phase 3 correlator: reads CpuEvents and GpuEvents from their respective
POSIX ring buffers and emits CorrelatedEvent records.

Correlation algorithm (three-pass, in priority order):
  Pass 1 — ADDRESS_MATCH (HARD)
      GPU transfer src_address falls within a live CPU allocation's
      [alloc_address, alloc_address + alloc_size_bytes).
      Highest confidence: the GPU is operating on memory we tracked.

  Pass 2 — SIZE_AND_TIMESTAMP (HARD)
      GPU transfer_size_bytes == cpu alloc_size_bytes AND the GPU event
      timestamp falls within [cpu_alloc_ts, cpu_alloc_ts + time_window_ns].
      Used when address ranges don't overlap (e.g. pinned / zero-copy memory
      that was re-mapped) but size+time strongly imply the same buffer.

  Pass 3 — TIMESTAMP_ONLY (WEAK)
      GPU event timestamp falls within time_window_ns of a CPU alloc and
      no HARD match exists. Records the closest CPU alloc by timestamp.
      Useful for kernel events that have no address information.

Each CPU alloc event produces at most one CorrelatedEvent (first HARD match
wins). GPU events are not deduplicated across passes — a TRANSFER event that
matches on address AND size will only appear once (ADDRESS_MATCH wins).

Output format (CorrelatedEvent dict, schema v0.1):
  {
    "cpu_event_id":  int,
    "gpu_event_id":  int,
    "confidence":    "HARD" | "WEAK",
    "match_reason":  "ADDRESS_MATCH" | "SIZE_AND_TIMESTAMP" | "TIMESTAMP_ONLY",
    "latency_ns":    int   # gpu_timestamp_ns - cpu_timestamp_ns
  }

Usage:
    from correlator import Correlator

    c = Correlator(cpu_shm="/hpc_cpu_xyz", gpu_shm="/hpc_gpu_xyz")
    c.start()
    # ... let it collect ...
    c.stop()
    session = c.build_session()   # returns ProfilingSession dict
"""

import threading
import time
import uuid
from typing import List, Dict, Optional, Tuple

import sys
from pathlib import Path

# Bridge lives in src/python — imported lazily so the correlator module
# can be imported without the full project tree (e.g. during unit tests).
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIME_WINDOW_NS = 5_000_000   # 5 ms — GPU ops are usually within this of the CPU alloc
DEFAULT_CPU_SHM        = "/hpc_profiler_cpu"
DEFAULT_GPU_SHM        = "/hpc_profiler_gpu"
DEFAULT_CAPACITY       = 8 * 1024 * 1024   # 8 MiB per ring buffer
DRAIN_POLL_INTERVAL    = 0.005             # 5 ms poll cadence


# ---------------------------------------------------------------------------
# Internal event index structures
# ---------------------------------------------------------------------------

class _LiveAlloc:
    """Tracks a single CPU allocation that hasn't been deallocated yet."""
    __slots__ = ("event_id", "timestamp_ns", "address", "size", "end_address")

    def __init__(self, event: dict):
        base               = event["base"]
        self.event_id      = base["event_id"]
        self.timestamp_ns  = base["timestamp_ns"]
        self.address       = event["alloc_address"]
        self.size          = event["alloc_size_bytes"]
        self.end_address   = self.address + self.size   # exclusive upper bound

    def contains(self, addr: int) -> bool:
        """True if addr falls within [address, end_address)."""
        return self.address <= addr < self.end_address


# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

class Correlator:
    """
    Reads from two ring buffers (CPU and GPU) in background threads,
    then correlates events on demand or at stop().

    Parameters
    ----------
    cpu_shm : str
        POSIX shared memory name for the CPU ring buffer.
    gpu_shm : str
        POSIX shared memory name for the GPU ring buffer.
    capacity : int
        Ring buffer capacity in bytes (used when opening consumer handles).
    time_window_ns : int
        Maximum nanoseconds between a CPU alloc and a GPU event for a
        SIZE_AND_TIMESTAMP or TIMESTAMP_ONLY match to be valid.
    """

    def __init__(
        self,
        cpu_shm: str = DEFAULT_CPU_SHM,
        gpu_shm: str = DEFAULT_GPU_SHM,
        capacity: int = DEFAULT_CAPACITY,
        time_window_ns: int = DEFAULT_TIME_WINDOW_NS,
    ):
        self._cpu_shm       = cpu_shm
        self._gpu_shm       = gpu_shm
        self._capacity      = capacity
        self._time_window   = time_window_ns

        self._cpu_bridge: Optional[object] = None
        self._gpu_bridge: Optional[object] = None

        self._cpu_events: List[dict] = []
        self._gpu_events: List[dict] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._cpu_thread: Optional[threading.Thread] = None
        self._gpu_thread: Optional[threading.Thread] = None

        self._session_start_ns: int = 0
        self._session_end_ns:   int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open ring buffer consumer handles and begin draining both buffers."""
        from bridge import Bridge  # lazy import — only needed for live ring buffer use

        self._session_start_ns = time.time_ns()
        self._stop_event.clear()

        self._cpu_bridge = Bridge(shm_name=self._cpu_shm, capacity=self._capacity)
        self._cpu_bridge.open(create=False)   # consumer — buffer was created by profiler

        self._gpu_bridge = Bridge(shm_name=self._gpu_shm, capacity=self._capacity)
        self._gpu_bridge.open(create=False)

        self._cpu_thread = threading.Thread(
            target=self._drain_loop,
            args=(self._cpu_bridge, self._cpu_events, "CPU"),
            daemon=True,
            name="correlator-cpu-drain",
        )
        self._gpu_thread = threading.Thread(
            target=self._drain_loop,
            args=(self._gpu_bridge, self._gpu_events, "GPU"),
            daemon=True,
            name="correlator-gpu-drain",
        )

        self._cpu_thread.start()
        self._gpu_thread.start()

    def stop(self) -> None:
        """Signal drain threads to stop and close ring buffers."""
        self._stop_event.set()

        if self._cpu_thread:
            self._cpu_thread.join(timeout=2.0)
        if self._gpu_thread:
            self._gpu_thread.join(timeout=2.0)

        self._session_end_ns = time.time_ns()

        if self._cpu_bridge:
            self._cpu_bridge.close(unlink=False)   # correlator is a consumer; don't unlink
        if self._gpu_bridge:
            self._gpu_bridge.close(unlink=False)

    # ------------------------------------------------------------------
    # Drain loop (runs in background thread)
    # ------------------------------------------------------------------

    def _drain_loop(self, bridge: "object", sink: list, label: str) -> None:
        from bridge import Bridge  # lazy import
        while not self._stop_event.is_set():
            payload = bridge.read()
            if payload is None:
                time.sleep(DRAIN_POLL_INTERVAL)
                continue
            try:
                event = Bridge.deserialize(payload)
                with self._lock:
                    sink.append(event)
            except Exception:
                pass   # malformed payload — skip silently

        # Drain any remaining events after stop signal
        while True:
            payload = bridge.read()
            if payload is None:
                break
            try:
                event = Bridge.deserialize(payload)
                with self._lock:
                    sink.append(event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Correlation engine
    # ------------------------------------------------------------------

    def correlate(self) -> List[dict]:
        """
        Run the three-pass correlation algorithm over the currently
        collected CPU and GPU events.

        Returns a list of CorrelatedEvent dicts.
        """
        with self._lock:
            cpu_snapshot = list(self._cpu_events)
            gpu_snapshot = list(self._gpu_events)

        return _run_correlation(cpu_snapshot, gpu_snapshot, self._time_window)

    # ------------------------------------------------------------------
    # Session builder
    # ------------------------------------------------------------------

    def build_session(self) -> dict:
        """
        Correlate all collected events and return a complete
        ProfilingSession dict ready for JSON serialization.
        """
        with self._lock:
            cpu_events = list(self._cpu_events)
            gpu_events = list(self._gpu_events)

        correlated = _run_correlation(cpu_events, gpu_events, self._time_window)

        return {
            "session_id":        str(uuid.uuid4()),
            "schema_version":    "0.1",
            "start_time_ns":     self._session_start_ns,
            "end_time_ns":       self._session_end_ns or time.time_ns(),
            "cpu_events":        cpu_events,
            "gpu_events":        gpu_events,
            "correlated_events": correlated,
        }

    # ------------------------------------------------------------------
    # Convenience: ingest pre-collected event lists (no ring buffers)
    # ------------------------------------------------------------------

    def ingest(self, cpu_events: List[dict], gpu_events: List[dict]) -> None:
        """
        Load events directly (e.g. from a JSON session or mock generator)
        instead of draining live ring buffers. Useful for offline analysis
        and testing.
        """
        with self._lock:
            self._cpu_events = list(cpu_events)
            self._gpu_events = list(gpu_events)
        if not self._session_start_ns:
            self._session_start_ns = time.time_ns()


# ---------------------------------------------------------------------------
# Core algorithm (pure function — easy to unit test)
# ---------------------------------------------------------------------------

def _run_correlation(
    cpu_events: List[dict],
    gpu_events: List[dict],
    time_window_ns: int,
) -> List[dict]:
    """
    Three-pass correlation.

    Returns list of CorrelatedEvent dicts.
    """
    # Separate CPU allocs from deallocs
    cpu_allocs: List[dict] = [e for e in cpu_events if not e.get("is_dealloc", False)]
    # GPU TRANSFER events have src/dst addresses; KERNEL events don't
    gpu_transfers = [e for e in gpu_events if e["base"]["event_type"] == "TRANSFER"]
    gpu_kernels   = [e for e in gpu_events if e["base"]["event_type"] != "TRANSFER"]

    correlated: List[dict] = []
    # Track which cpu_event_ids and gpu_event_ids have been matched (HARD)
    matched_cpu:  set = set()
    matched_gpu:  set = set()

    # ------------------------------------------------------------------
    # Pass 1 — ADDRESS_MATCH (HARD)
    # ------------------------------------------------------------------
    # Build interval index: list of _LiveAlloc sorted by address
    live_allocs = [_LiveAlloc(e) for e in cpu_allocs]
    live_allocs.sort(key=lambda a: a.address)

    for gpu_ev in gpu_transfers:
        gpu_id   = gpu_ev["base"]["event_id"]
        gpu_ts   = gpu_ev["base"]["timestamp_ns"]
        src_addr = gpu_ev.get("src_address", 0)
        dst_addr = gpu_ev.get("dst_address", 0)

        match = _find_address_match(live_allocs, src_addr, dst_addr)
        if match is None:
            continue
        if match.event_id in matched_cpu or gpu_id in matched_gpu:
            continue

        correlated.append(_make_correlated(
            cpu_event_id=match.event_id,
            gpu_event_id=gpu_id,
            confidence="HARD",
            match_reason="ADDRESS_MATCH",
            latency_ns=gpu_ts - match.timestamp_ns,
        ))
        matched_cpu.add(match.event_id)
        matched_gpu.add(gpu_id)

    # ------------------------------------------------------------------
    # Pass 2 — SIZE_AND_TIMESTAMP (HARD)
    # Applied to GPU transfers not already matched by address in Pass 1.
    # Handles zero-copy, pinned, and unified memory transfers where CUPTI
    # reports a GPU-side address that doesn't overlap with the CPU alloc
    # virtual address space, but size + timestamp proximity strongly imply
    # the same buffer.
    # ------------------------------------------------------------------
    # Build size → alloc mapping for fast lookup
    size_index: Dict[int, List[_LiveAlloc]] = {}
    for alloc in live_allocs:
        size_index.setdefault(alloc.size, []).append(alloc)

    for gpu_ev in gpu_transfers:
        gpu_id        = gpu_ev["base"]["event_id"]
        gpu_ts        = gpu_ev["base"]["timestamp_ns"]
        transfer_size = gpu_ev.get("transfer_size_bytes", 0)

        if gpu_id in matched_gpu:
            continue
        if transfer_size == 0:
            continue

        candidates = size_index.get(transfer_size, [])
        best = _find_closest_in_window(candidates, gpu_ts, time_window_ns, matched_cpu)
        if best is None:
            continue

        correlated.append(_make_correlated(
            cpu_event_id=best.event_id,
            gpu_event_id=gpu_id,
            confidence="HARD",
            match_reason="SIZE_AND_TIMESTAMP",
            latency_ns=gpu_ts - best.timestamp_ns,
        ))
        matched_cpu.add(best.event_id)
        matched_gpu.add(gpu_id)

    # ------------------------------------------------------------------
    # Pass 3 — TIMESTAMP_ONLY (WEAK)
    # ------------------------------------------------------------------
    # All remaining GPU events (transfers without a HARD match + kernels)
    unmatched_gpu = [
        e for e in gpu_events
        if e["base"]["event_id"] not in matched_gpu
    ]

    for gpu_ev in unmatched_gpu:
        gpu_id = gpu_ev["base"]["event_id"]
        gpu_ts = gpu_ev["base"]["timestamp_ns"]

        best = _find_closest_in_window(live_allocs, gpu_ts, time_window_ns, matched_cpu)
        if best is None:
            continue

        correlated.append(_make_correlated(
            cpu_event_id=best.event_id,
            gpu_event_id=gpu_id,
            confidence="WEAK",
            match_reason="TIMESTAMP_ONLY",
            latency_ns=gpu_ts - best.timestamp_ns,
        ))
        # WEAK matches do NOT consume the cpu_event_id — multiple GPU events
        # can weakly associate with the same CPU alloc
        matched_gpu.add(gpu_id)

    return correlated


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------

def _find_address_match(
    sorted_allocs: List["_LiveAlloc"],
    src_addr: int,
    dst_addr: int,
) -> Optional["_LiveAlloc"]:
    """
    Binary-search the sorted alloc list for an alloc whose range contains
    src_addr or dst_addr. Returns the first hit.
    """
    for addr in (src_addr, dst_addr):
        if addr == 0:
            continue
        result = _binary_search_interval(sorted_allocs, addr)
        if result is not None:
            return result
    return None


def _binary_search_interval(
    sorted_allocs: List["_LiveAlloc"],
    addr: int,
) -> Optional["_LiveAlloc"]:
    """
    Find an alloc whose [address, end_address) contains addr.
    sorted_allocs must be sorted by .address ascending.
    O(log n) search: find rightmost alloc with .address <= addr,
    then check if addr < end_address.
    """
    lo, hi = 0, len(sorted_allocs) - 1
    best_idx = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_allocs[mid].address <= addr:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_idx == -1:
        return None
    candidate = sorted_allocs[best_idx]
    return candidate if candidate.contains(addr) else None


def _find_closest_in_window(
    allocs: List["_LiveAlloc"],
    gpu_ts: int,
    window_ns: int,
    exclude_ids: set,
) -> Optional["_LiveAlloc"]:
    """
    Among allocs NOT in exclude_ids, find the one whose timestamp_ns is
    closest to gpu_ts and within window_ns. Returns None if no match.
    """
    best: Optional[_LiveAlloc] = None
    best_delta = window_ns + 1

    for alloc in allocs:
        if alloc.event_id in exclude_ids:
            continue
        delta = abs(gpu_ts - alloc.timestamp_ns)
        if delta <= window_ns and delta < best_delta:
            best_delta = delta
            best = alloc

    return best


def _make_correlated(
    cpu_event_id: int,
    gpu_event_id: int,
    confidence: str,
    match_reason: str,
    latency_ns: int,
) -> dict:
    return {
        "cpu_event_id": cpu_event_id,
        "gpu_event_id": gpu_event_id,
        "confidence":   confidence,
        "match_reason": match_reason,
        # Schema requires latency_ns >= 0. Negative values arise from GPU
        # clock skew or async CUPTI flush ordering; clamp rather than discard.
        "latency_ns":   max(0, latency_ns),
    }