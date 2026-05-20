"""
python_memory_layer.py
----------------------
Combined Python memory profiler. Merges tracemalloc (size, callstack)
with gc introspection (address, type, ref_count, generation) into a
single fully-populated CpuEvent per allocation.

Clock domain:
  All timestamps use CLOCK_MONOTONIC (time.monotonic_ns()), not
  time.time_ns() (epoch-based). This matches the GPU profiler's clock.

Non-GC-tracked object capture (e.g. numpy ndarrays):
  numpy arrays are tracked by tracemalloc but NOT by gc.get_objects().
  collect() supplements the gc scan with two additional passes:
    1. Cross-thread frame scan — iterates sys._current_frames() across
       ALL threads (including the main workload thread) to find buffer
       objects (ndarrays, tensors) held as local variables.
    2. Referent scan — for each newly found GC-tracked object, walks
       its direct gc.get_referents() to find buffer members (e.g. arrays
       stored as class attributes).
  Both passes check get_object_traceback() against the snapshot diff,
  so only genuinely NEW allocations are emitted.

Dealloc coverage:
  weakref.finalize() is registered on every tracked object. numpy arrays
  support weakrefs natively. Finalizers push to _dealloc_queue which is
  drained unconditionally in collect() and flush().

Performance:
  Traceback objects are used directly as dict keys (no tuple conversion)
  cutting the stat_by_key construction from ~50ms to <1ms per tick.

New fields vs schema v0.1 original:
  module_name      — type(obj).__module__
  peak_rss_kb      — process RSS at collect() time
  lifetime_ns      — ns between alloc and dealloc (DEALLOC only)
  is_numpy_buffer  — True when obj has __array_interface__
  buffer_nbytes    — obj.nbytes when is_numpy_buffer
  parent_address   — id() of first tracked GC referrer (track_refs=True)
  pinned_address   — actual C data pointer for buffer objects
  array_shape      — tuple of ints for numpy/torch (empty otherwise)  [NEW]
  array_dtype      — dtype string e.g. 'float32' (empty otherwise)    [NEW]
  tracemalloc_size — size as reported by tracemalloc (may differ from  [NEW]
                     alloc_size_bytes which comes from snapshot diff)
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
# CLOCK_MONOTONIC timestamp
# ---------------------------------------------------------------------------

def _now_ns() -> int:
    """Nanoseconds since system boot (CLOCK_MONOTONIC). Matches GPU clock domain."""
    return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Thread-safe event ID counter
# ---------------------------------------------------------------------------

_id_counter = count(1)
_id_lock    = threading.Lock()

def _next_id() -> int:
    with _id_lock:
        return next(_id_counter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tb_to_callstack(tb) -> list:
    """Convert tracemalloc Traceback to list of 'file:line' strings, outermost first."""
    return [f"{f.filename}:{f.lineno}" for f in reversed(tb)]

def _current_rss_kb() -> int:
    """Process RSS in kilobytes (Linux: KB, macOS: bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024
    return rss

def _numpy_info(obj) -> tuple:
    """
    Returns (is_numpy_buffer, buffer_nbytes, pinned_address, shape, dtype).

    pinned_address — the actual C-level data pointer that CUPTI reports
                     as src_address on HOST_TO_DEVICE transfers.
    shape          — tuple of ints (array dimensions), empty for non-arrays.
    dtype          — dtype string e.g. 'float32', empty for non-arrays.
    """
    try:
        # PyTorch tensors
        if hasattr(obj, 'data_ptr') and callable(obj.data_ptr):
            nbytes  = getattr(obj, 'nbytes', 0)
            shape   = tuple(int(d) for d in getattr(obj, 'shape', ()))
            dtype   = str(getattr(obj, 'dtype', ''))
            try:
                ptr = int(obj.data_ptr())
            except Exception:
                ptr = 0
            return True, int(nbytes), ptr, shape, dtype

        # NumPy / host arrays
        if hasattr(obj, '__array_interface__'):
            iface  = obj.__array_interface__
            nbytes = getattr(obj, 'nbytes', 0)
            ptr    = iface.get('data', (0,))[0] or 0
            shape  = tuple(int(d) for d in getattr(obj, 'shape', ()))
            dtype  = str(getattr(obj, 'dtype', ''))
            return True, int(nbytes), int(ptr), shape, dtype

        # CuPy / CUDA arrays
        if hasattr(obj, '__cuda_array_interface__'):
            iface  = obj.__cuda_array_interface__
            nbytes = getattr(obj, 'nbytes', 0)
            ptr    = iface.get('data', (0,))[0] or 0
            shape  = tuple(int(d) for d in getattr(obj, 'shape', ()))
            dtype  = str(getattr(obj, 'dtype', ''))
            return True, int(nbytes), int(ptr), shape, dtype

    except Exception:
        pass
    return False, 0, 0, (), ""

# Types that can't be weakly referenced and don't yield useful events
_NO_WEAKREF_TYPES = (int, float, str, bytes, bool, type(None))


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class PythonMemoryLayer:
    """
    Combined tracemalloc + GC + frame-scan memory profiler.

    Emits fully-populated CpuEvent dicts including:
      alloc_address        — id(obj), the real CPython memory address
      alloc_size_bytes     — bytes from tracemalloc snapshot diff
      tracemalloc_size     — size as directly reported by tracemalloc stat
      object_type          — type(obj).__qualname__
      module_name          — type(obj).__module__
      ref_count            — sys.getrefcount(obj) - 1
      callstack            — from tracemalloc.get_object_traceback(obj)
      gc_generation        — from gc.get_objects(gen), or -1 if not GC-tracked
      is_dealloc           — False on alloc, True on dealloc
      peak_rss_kb          — process RSS at collect() time
      lifetime_ns          — ns from alloc to dealloc (DEALLOC only, else 0)
      is_numpy_buffer      — True if obj has __array_interface__
      buffer_nbytes        — obj.nbytes when is_numpy_buffer
      parent_address       — id() of first tracked GC referrer (track_refs only)
      pinned_address       — C-level data pointer for buffer objects
      array_shape          — tuple of ints (dims) for arrays, else ()
      array_dtype          — dtype string for arrays, else ""
    """

    def __init__(self, nframe: int = 16, track_refs: bool = False):
        """
        nframe:     stack frames to capture per allocation (default 16).
        track_refs: if True, record parent_address via gc.get_referrers().
                    Expensive — leave False for high-frequency workloads.
        """
        self._nframe     = nframe
        self._track_refs = track_refs
        self._running    = False

        self._events:      list = []
        self._events_lock        = threading.Lock()
        self._last_snapshot: Optional[tracemalloc.Snapshot] = None

        # Map: alloc_address → (size, obj_type, alloc_ts, pinned_addr)
        self._live:      dict = {}
        self._live_lock        = threading.Lock()
        self._tracked_addrs: set = set()

        # Async dealloc queue — drained unconditionally in collect() and flush()
        self._dealloc_queue: list = []
        self._dealloc_lock        = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start profiling. Takes a baseline snapshot."""
        if self._running:
            return
        tracemalloc.start(self._nframe)
        self._last_snapshot = tracemalloc.take_snapshot()
        self._running = True

    def collect(self) -> list:
        """
        Snapshot diff since last collect() or start(), plus pending deallocs.

        Three-pass object discovery (all passes gated on the snapshot diff):
          Pass A — gc.get_objects() scan (GC-tracked objects: dicts, lists,
                   user-class instances)
          Pass B — Cross-thread frame scan via sys._current_frames()
                   (finds numpy arrays stored as local variables in any thread)
          Pass C — gc.get_referents() of new GC objects from Pass A
                   (finds numpy arrays stored as class / list attributes)

        Dealloc drain is unconditional so deallocs are never stranded.
        """
        # Drain deallocs unconditionally (even if not running)
        with self._dealloc_lock:
            pending_deallocs = self._dealloc_queue
            self._dealloc_queue = []

        if not self._running:
            with self._events_lock:
                self._events.extend(pending_deallocs)
            return pending_deallocs

        # ── Snapshot diff ──────────────────────────────────────────────
        current = tracemalloc.take_snapshot()
        stats   = current.compare_to(self._last_snapshot, key_type="traceback")
        self._last_snapshot = current

        # Use Traceback objects directly as dict keys — 50× faster than tuple()
        stat_by_tb: dict = {}
        for stat in stats:
            if stat.size_diff > 0:
                stat_by_tb[stat.traceback] = stat

        rss_kb = _current_rss_kb()

        new_allocs: list = []

        if stat_by_tb:
            seen_addrs: set = set()
            # Accumulate (obj, tb, stat, gc_generation) tuples across all passes
            candidates: list = []

            # ── Pass A: gc.get_objects() scan ─────────────────────────
            new_gc_objects: list = []  # (addr, obj) for Passes B→referents
            for gen in range(3):
                try:
                    for obj in gc.get_objects(gen):
                        if isinstance(obj, _NO_WEAKREF_TYPES):
                            continue
                        addr = id(obj)
                        if addr in seen_addrs:
                            continue
                        seen_addrs.add(addr)
                        try:
                            tb = tracemalloc.get_object_traceback(obj)
                        except Exception:
                            continue
                        if tb is None:
                            continue
                        if tb in stat_by_tb:
                            candidates.append((obj, tb, stat_by_tb[tb], gen))
                            new_gc_objects.append((addr, obj))
                except Exception:
                    continue

            # ── Pass B: cross-thread frame scan ──────────────────────
            # Finds buffer objects (ndarrays, tensors) stored as local
            # variables in any running thread, including the main workload.
            try:
                for _tid, frame in sys._current_frames().items():
                    f = frame
                    while f:
                        try:
                            for v in list(f.f_locals.values()):
                                addr = id(v)
                                if addr in seen_addrs:
                                    continue
                                if isinstance(v, _NO_WEAKREF_TYPES):
                                    seen_addrs.add(addr)
                                    continue
                                seen_addrs.add(addr)
                                try:
                                    tb = tracemalloc.get_object_traceback(v)
                                except Exception:
                                    continue
                                if tb and tb in stat_by_tb:
                                    candidates.append((v, tb, stat_by_tb[tb], -1))
                                # Also check one level of container contents
                                if isinstance(v, (list, tuple)):
                                    for item in list(v):
                                        iid = id(item)
                                        if iid in seen_addrs:
                                            continue
                                        if isinstance(item, _NO_WEAKREF_TYPES):
                                            seen_addrs.add(iid)
                                            continue
                                        seen_addrs.add(iid)
                                        try:
                                            tb2 = tracemalloc.get_object_traceback(item)
                                        except Exception:
                                            continue
                                        if tb2 and tb2 in stat_by_tb:
                                            candidates.append(
                                                (item, tb2, stat_by_tb[tb2], -1))
                        except Exception:
                            pass
                        f = f.f_back
            except Exception:
                pass

            # ── Pass C: referents of new GC objects ───────────────────
            # Catches ndarrays / tensors stored as class attributes.
            for _addr, obj in new_gc_objects:
                try:
                    for ref in gc.get_referents(obj):
                        rid = id(ref)
                        if rid in seen_addrs:
                            continue
                        if isinstance(ref, _NO_WEAKREF_TYPES):
                            seen_addrs.add(rid)
                            continue
                        seen_addrs.add(rid)
                        try:
                            tb = tracemalloc.get_object_traceback(ref)
                        except Exception:
                            continue
                        if tb and tb in stat_by_tb:
                            candidates.append((ref, tb, stat_by_tb[tb], -1))
                except Exception:
                    continue

            # ── Build events for all candidates ───────────────────────
            for obj, tb, stat, gen in candidates:
                addr = id(obj)
                # Final duplicate guard (candidates may overlap across passes)
                with self._live_lock:
                    if addr in self._live:
                        continue

                try:
                    obj_type    = type(obj).__qualname__
                    module_name = type(obj).__module__ or ""
                    ref_count   = sys.getrefcount(obj) - 1
                    callstack   = _tb_to_callstack(tb)
                    size        = stat.size_diff
                    tm_size     = stat.size        # tracemalloc's total size
                    is_numpy, buf_nbytes, pinned_addr, shape, dtype = _numpy_info(obj)
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
                    "tracemalloc_size": tm_size,
                    "object_type":      obj_type,
                    "module_name":      module_name,
                    "ref_count":        ref_count,
                    "callstack":        callstack,
                    "gc_generation":    gen,
                    "is_dealloc":       False,
                    "peak_rss_kb":      rss_kb,
                    "lifetime_ns":      0,
                    "is_numpy_buffer":  is_numpy,
                    "buffer_nbytes":    buf_nbytes,
                    "parent_address":   parent_addr,
                    "pinned_address":   pinned_addr,
                    "array_shape":      list(shape),
                    "array_dtype":      dtype,
                }
                new_allocs.append(event)

                with self._live_lock:
                    self._live[addr] = (size, obj_type, alloc_ts, pinned_addr)
                    self._tracked_addrs.add(addr)

                self._register_finalizer(obj, addr)

        all_new = new_allocs + pending_deallocs
        with self._events_lock:
            self._events.extend(all_new)
        return all_new

    def flush(self) -> list:
        """Return all accumulated events and clear the buffer. Also drains dealloc queue."""
        with self._dealloc_lock:
            pending = self._dealloc_queue
            self._dealloc_queue = []
        with self._events_lock:
            all_events = list(self._events) + pending
            self._events.clear()
        return all_events

    def stop(self) -> list:
        """Stop profiling. Returns remaining events from a final collect()."""
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

    def _register_finalizer(self, obj, addr: int) -> None:
        """Register a weakref finalizer to capture deallocation."""
        try:
            weakref.finalize(obj, self._on_dealloc, addr)
        except TypeError:
            pass  # type doesn't support weakrefs

    def _on_dealloc(self, addr: int) -> None:
        """Called by weakref finalizer when a tracked object is collected."""
        with self._live_lock:
            info = self._live.pop(addr, None)
            self._tracked_addrs.discard(addr)
        if info is None:
            return

        size, obj_type, alloc_ts, pinned_addr = info
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
            "tracemalloc_size": 0,
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
            "pinned_address":   pinned_addr,
            "array_shape":      [],
            "array_dtype":      "",
        }
        with self._dealloc_lock:
            self._dealloc_queue.append(event)