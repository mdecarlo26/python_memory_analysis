"""
python_memory_layer.py
----------------------
Combined Python memory profiler. Merges tracemalloc (size, callstack)
with gc introspection (address, type, ref_count, generation) into a
single fully-populated CpuEvent per allocation.

Clock domain:
  All timestamps use CLOCK_MONOTONIC (nanoseconds since boot), NOT
  time.time_ns() (epoch-based). This is critical: the GPU profiler
  (cupti_layer.cpp TimestampSync) calibrates CUPTI hardware timestamps
  against CLOCK_MONOTONIC. Using CLOCK_REALTIME here would place CPU
  timestamps ~56 years ahead of GPU timestamps, making every correlator
  time-window comparison fail. We read CLOCK_MONOTONIC via ctypes directly
  so no C extension is required; time.monotonic_ns() is the fallback.

New fields vs schema v0.1 original:
  peak_rss_kb      — process RSS at collect() time (resource.getrusage)
  lifetime_ns      — ns between alloc and dealloc (set on DEALLOC events)
  module_name      — type(obj).__module__ (free at alloc time)
  is_numpy_buffer  — True when obj has __array_interface__ (numpy/cupy/torch)
  buffer_nbytes    — obj.nbytes when is_numpy_buffer is True, else 0
  parent_address   — id() of first tracked GC referrer (track_refs=True only)

Dealloc fix:
  _dealloc_queue is a separate list from _events. _on_dealloc() always
  appends to _dealloc_queue. Both collect() and flush() drain it so
  deallocs are never stranded regardless of which method the caller uses.
"""

import gc
import os
import resource
import sys
import time
import threading
import tracemalloc
import weakref
from itertools import count
from typing import Optional


# ---------------------------------------------------------------------------
# CLOCK_MONOTONIC timestamp — must match the GPU profiler's clock domain.
#
# The GPU profiler (cupti_layer.cpp TimestampSync) calibrates CUPTI hardware
# timestamps against CLOCK_MONOTONIC via clock_gettime(CLOCK_MONOTONIC).
# Python's time.time_ns() uses CLOCK_REALTIME (epoch-based), which is ~56
# years ahead of CLOCK_MONOTONIC on a typical system. Using the wrong clock
# here makes every timestamp comparison in the correlator off by ~1.78×10¹⁸ns,
# which is larger than any time window — nothing would ever correlate.
#
# time.monotonic_ns() (Python 3.7+) reads CLOCK_MONOTONIC directly with no
# subprocess forks. Do NOT use ctypes.util.find_library() here — it calls
# subprocess.Popen(['/sbin/ldconfig']) which forks a child process. When that
# child exits it delivers SIGCHLD to the target process during ptrace attach,
# causing inject() to receive SIGCHLD instead of SIGTRAP and crash.
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    """Nanoseconds since system boot (CLOCK_MONOTONIC). Matches GPU clock domain."""
    return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Module-level event ID counter — thread-safe, never resets
# ---------------------------------------------------------------------------

_id_counter = count(1)
_id_lock = threading.Lock()

def _next_id() -> int:
    with _id_lock:
        return next(_id_counter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tb_key(tb) -> tuple:
    """Stable hashable key from a tracemalloc Traceback."""
    return tuple((f.filename, f.lineno) for f in tb)

def _tb_to_callstack(tb) -> list:
    """Convert tracemalloc Traceback to list of 'file:line' strings, outermost first."""
    return [f"{f.filename}:{f.lineno}" for f in reversed(tb)]

def _current_rss_kb() -> int:
    """Process RSS in kilobytes. Linux: ru_maxrss is in KB. macOS: bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024
    return rss

def _numpy_info(obj) -> tuple:
    """
    Returns (is_numpy_buffer: bool, buffer_nbytes: int).
    Detects numpy/cupy/torch tensors whose real data lives in a C-level
    buffer that tracemalloc cannot see because it bypasses Python's allocator.
    """
    try:
        if hasattr(obj, '__array_interface__') or hasattr(obj, '__cuda_array_interface__'):
            nbytes = getattr(obj, 'nbytes', 0)
            return True, int(nbytes)
    except Exception:
        pass
    return False, 0

# Types that can't be weakly referenced
_NO_WEAKREF_TYPES = (int, float, str, bytes, bool, type(None))


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class PythonMemoryLayer:
    """
    Combined tracemalloc + GC memory profiler.

    Emits fully-populated CpuEvent dicts:
      alloc_address    — id(obj), the real CPython memory address
      alloc_size_bytes — from tracemalloc snapshot diff
      object_type      — type(obj).__qualname__
      module_name      — type(obj).__module__
      ref_count        — sys.getrefcount(obj) - 1
      callstack        — from tracemalloc.get_object_traceback(obj)
      gc_generation    — from gc.get_objects(gen)
      is_dealloc       — False on alloc, True on dealloc
      peak_rss_kb      — process RSS at collect() time
      lifetime_ns      — ns from alloc to dealloc (DEALLOC events only, else 0)
      is_numpy_buffer  — True if obj exposes __array_interface__
      buffer_nbytes    — obj.nbytes when is_numpy_buffer, else 0
      parent_address   — id() of first tracked GC referrer (track_refs=True only)
    """

    def __init__(self, nframe: int = 16, track_refs: bool = False):
        """
        nframe:     stack frames to capture per allocation (default 16).
        track_refs: if True, record parent_address via gc.get_referrers().
                    Expensive — leave False for high-frequency workloads.
        """
        self._nframe = nframe
        self._track_refs = track_refs
        self._running = False
        self._events: list = []
        self._events_lock = threading.Lock()
        self._last_snapshot: Optional[tracemalloc.Snapshot] = None

        # Map: alloc_address → (size, obj_type, alloc_timestamp_ns)
        self._live: dict = {}
        self._live_lock = threading.Lock()

        # Set of currently tracked addresses (for parent_address lookup)
        self._tracked_addrs: set = set()

        # Separate queue for deallocs arriving asynchronously via weakref finalizers.
        # collect() drains this UNCONDITIONALLY so deallocs are never stranded
        # by the early-return path that fires when the snapshot diff is empty.
        self._dealloc_queue: list = []
        self._dealloc_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start profiling. Takes a baseline snapshot."""
        if self._running:
            return
        tracemalloc.start(self._nframe)
        self._last_snapshot = tracemalloc.take_snapshot()
        self._running = True

    def collect(self) -> list:
        """
        Snapshot diff since last collect() or start(), plus any pending deallocs.

        The dealloc drain is UNCONDITIONAL — it runs even when the snapshot
        diff is empty. This is the key fix: previously deallocs were stranded
        whenever no new allocs appeared in the same tick.

        Returns new events (allocs + deallocs) since last call.
        Does not clear the buffer — call flush() to drain it.
        """
        if not self._running:
            # Still drain any queued deallocs even if stopped
            with self._dealloc_lock:
                pending = self._dealloc_queue
                self._dealloc_queue = []
            if pending:
                with self._events_lock:
                    self._events.extend(pending)
            return pending

        # ── Snapshot diff ──────────────────────────────────────────────
        current = tracemalloc.take_snapshot()
        stats   = current.compare_to(self._last_snapshot, key_type="traceback")
        self._last_snapshot = current

        # RSS snapshot — once per tick, not per object
        rss_kb = _current_rss_kb()

        stat_by_key: dict = {}
        for stat in stats:
            if stat.size_diff != 0:
                stat_by_key[_tb_key(stat.traceback)] = stat

        new_allocs: list = []

        if stat_by_key:
            seen_addresses: set = set()

            for gen in range(3):
                try:
                    live_objects = gc.get_objects(gen)
                except Exception:
                    continue

                for obj in live_objects:
                    if obj is self or obj is self._events or obj is self._live:
                        continue
                    if isinstance(obj, _NO_WEAKREF_TYPES):
                        continue

                    try:
                        tb = tracemalloc.get_object_traceback(obj)
                    except Exception:
                        continue
                    if tb is None:
                        continue

                    key = _tb_key(tb)
                    if key not in stat_by_key:
                        continue

                    stat = stat_by_key[key]
                    if stat.size_diff <= 0:
                        continue

                    addr = id(obj)
                    if addr in seen_addresses:
                        continue
                    seen_addresses.add(addr)

                    try:
                        obj_type    = type(obj).__qualname__
                        module_name = type(obj).__module__ or ""
                        ref_count   = sys.getrefcount(obj) - 1
                        callstack   = _tb_to_callstack(tb)
                        generation  = gen
                        size        = stat.size_diff
                        is_numpy, buf_nbytes = _numpy_info(obj)
                    except Exception:
                        continue

                    parent_addr = 0
                    if self._track_refs:
                        try:
                            with self._live_lock:
                                tracked_snap = frozenset(self._tracked_addrs)
                            for referrer in gc.get_referrers(obj):
                                r_addr = id(referrer)
                                if r_addr in tracked_snap and r_addr != addr:
                                    parent_addr = r_addr
                                    break
                        except Exception:
                            pass

                    alloc_ts = _now_ns()
                    event = {
                        "base": {
                            "event_id":     _next_id(),
                            "timestamp_ns": alloc_ts,
                            "process_id":   os.getpid(),
                            "thread_id":    threading.get_ident(),
                            "event_type":   "ALLOC",
                        },
                        "alloc_address":    addr,
                        "alloc_size_bytes": size,
                        "object_type":      obj_type,
                        "module_name":      module_name,
                        "ref_count":        ref_count,
                        "callstack":        callstack,
                        "gc_generation":    generation,
                        "is_dealloc":       False,
                        "peak_rss_kb":      rss_kb,
                        "lifetime_ns":      0,
                        "is_numpy_buffer":  is_numpy,
                        "buffer_nbytes":    buf_nbytes,
                        "parent_address":   parent_addr,
                    }
                    new_allocs.append(event)

                    with self._live_lock:
                        self._live[addr] = (size, obj_type, alloc_ts)
                        self._tracked_addrs.add(addr)

                    self._register_finalizer(obj, addr)

        # ── Drain dealloc queue — UNCONDITIONAL ────────────────────────
        with self._dealloc_lock:
            pending_deallocs = self._dealloc_queue
            self._dealloc_queue = []

        all_new = new_allocs + pending_deallocs

        with self._events_lock:
            self._events.extend(all_new)

        return all_new

    def flush(self) -> list:
        """
        Return all accumulated events (alloc + dealloc) and clear the buffer.

        Also drains _dealloc_queue so that callers who call flush() directly
        (without going through collect() first) still receive dealloc events
        that arrived via weakref finalizers since the last collect().
        """
        # Drain async dealloc queue first, under its own lock
        with self._dealloc_lock:
            pending = self._dealloc_queue
            self._dealloc_queue = []

        with self._events_lock:
            all_events = list(self._events) + pending
            self._events.clear()

        return all_events

    def stop(self) -> list:
        """Stop profiling. Returns any remaining events from a final collect()."""
        if not self._running:
            return []
        remaining = self.collect()
        tracemalloc.stop()
        self._running = False
        return remaining

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register_finalizer(self, obj, addr: int):
        try:
            weakref.finalize(obj, self._on_dealloc, addr)
        except TypeError:
            pass  # type doesn't support weakrefs

    def _on_dealloc(self, addr: int):
        """Called by weakref finalizer when a tracked object is collected."""
        with self._live_lock:
            info = self._live.pop(addr, None)
            self._tracked_addrs.discard(addr)

        if info is None:
            return

        size, obj_type, alloc_ts = info
        dealloc_ts = _now_ns()

        event = {
            "base": {
                "event_id":     _next_id(),
                "timestamp_ns": dealloc_ts,
                "process_id":   os.getpid(),
                "thread_id":    threading.get_ident(),
                "event_type":   "DEALLOC",
            },
            "alloc_address":    addr,
            "alloc_size_bytes": size,
            "object_type":      obj_type,
            "module_name":      "",
            "ref_count":        0,
            "callstack":        [],
            "gc_generation":    0,
            "is_dealloc":       True,
            "peak_rss_kb":      0,
            "lifetime_ns":      dealloc_ts - alloc_ts,
            "is_numpy_buffer":  False,
            "buffer_nbytes":    0,
            "parent_address":   0,
        }
        with self._dealloc_lock:
            self._dealloc_queue.append(event)