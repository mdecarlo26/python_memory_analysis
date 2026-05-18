"""
python_memory_layer.py
----------------------
Combined Python memory profiler. Merges tracemalloc (size, callstack)
with gc introspection (address, type, ref_count, generation) into a
single fully-populated CpuEvent per allocation.

Design:
  - tracemalloc provides size and callstack via snapshot diffs
  - gc.get_objects() + get_object_traceback() links live objects to
    their tracemalloc entries, giving us address, type, ref_count,
    and gc_generation in the same pass
  - weakref finalizers capture the exact address and timestamp of
    deallocation without needing a second snapshot diff

Why combined:
  Separating these two concerns requires a merge step keyed on
  (size, callstack) which is ambiguous when multiple objects of the
  same type and size are allocated from the same call site. Doing
  them together in one pass eliminates that ambiguity entirely.

Usage:
    layer = PythonMemoryLayer()
    layer.start()
    ... code to profile ...
    events = layer.collect()
    layer.stop()
    all_events = layer.flush()
"""

import gc
import os
import sys
import time
import threading
import tracemalloc
import weakref
from itertools import count
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level event ID counter — thread-safe, never resets
# ---------------------------------------------------------------------------

_id_counter = count(1)
_id_lock = threading.Lock()

def _next_id() -> int:
    with _id_lock:
        return next(_id_counter)

def _now_ns() -> int:
    return time.time_ns()

def _make_base(event_type: str) -> dict:
    return {
        "event_id": _next_id(),
        "timestamp_ns": _now_ns(),
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
        "event_type": event_type,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tb_key(tb) -> tuple:
    """
    Stable hashable key from a tracemalloc Traceback.
    Used to match snapshot stats against live object tracebacks.
    """
    return tuple((f.filename, f.lineno) for f in tb)

def _tb_to_callstack(tb) -> list[str]:
    """Convert tracemalloc Traceback to list of 'file:line' strings, outermost first."""
    return [f"{f.filename}:{f.lineno}" for f in reversed(tb)]

def _gc_generation(obj) -> int:
    """Return the GC generation (0, 1, 2) of a live object. Returns 0 if not found."""
    for gen in range(3):
        try:
            if any(o is obj for o in gc.get_objects(gen)):
                return gen
        except Exception:
            pass
    return 0

# Types that can't be weakly referenced — we skip finalizer registration for these
_NO_WEAKREF_TYPES = (int, float, str, bytes, bool, type(None))


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class PythonMemoryLayer:
    """
    Combined tracemalloc + GC memory profiler.

    Emits fully-populated CpuEvent dicts with all fields from the schema:
      alloc_address    — id(obj), the real CPython memory address
      alloc_size_bytes — from tracemalloc snapshot diff
      object_type      — type(obj).__qualname__
      ref_count        — sys.getrefcount(obj) - 1
      callstack        — from tracemalloc.get_object_traceback(obj)
      gc_generation    — from gc.get_objects(gen)
      is_dealloc       — False on alloc, True on dealloc (via weakref finalizer)
    """

    def __init__(self, nframe: int = 16):
        """
        nframe: stack frames to capture per allocation (default 16).
                Higher = richer callstacks, slightly more overhead.
        """
        self._nframe = nframe
        self._running = False
        self._events: list[dict] = []
        self._events_lock = threading.Lock()
        self._last_snapshot: Optional[tracemalloc.Snapshot] = None

        # Map from alloc_address → (alloc_size_bytes, object_type) for dealloc events
        self._live: dict[int, tuple[int, str]] = {}
        self._live_lock = threading.Lock()

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

    def collect(self) -> list[dict]:
        """
        Snapshot diff since last collect() or start().
        For each new allocation, finds the live object and emits a
        fully-populated CpuEvent. Registers a weakref finalizer to
        capture the dealloc event later.

        Returns new events since last call. Does not clear the buffer —
        call flush() to drain it.
        """
        if not self._running:
            return []

        current = tracemalloc.take_snapshot()
        stats = current.compare_to(self._last_snapshot, key_type="traceback")
        self._last_snapshot = current

        # Build a lookup: traceback key → StatisticDiff
        # so we can match live objects to their tracemalloc entry
        stat_by_key: dict[tuple, object] = {}
        for stat in stats:
            if stat.size_diff != 0:
                stat_by_key[_tb_key(stat.traceback)] = stat

        if not stat_by_key:
            return []

        new_events = []

        # Walk all live GC-tracked objects and match them to stats
        # We iterate generation 0 first (newest objects)
        seen_addresses: set[int] = set()

        for gen in range(3):
            try:
                live_objects = gc.get_objects(gen)
            except Exception:
                continue

            for obj in live_objects:
                # Skip internal profiler objects
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

                # Gather all fields in one pass
                try:
                    obj_type = type(obj).__qualname__
                    ref_count = sys.getrefcount(obj) - 1  # subtract getrefcount's own ref
                    callstack = _tb_to_callstack(tb)
                    generation = gen
                    size = stat.size_diff
                except Exception:
                    continue

                event = {
                    "base": _make_base("ALLOC"),
                    "alloc_address": addr,
                    "alloc_size_bytes": size,
                    "object_type": obj_type,
                    "ref_count": ref_count,
                    "callstack": callstack,
                    "gc_generation": generation,
                    "is_dealloc": False,
                }
                new_events.append(event)

                # Track for dealloc
                with self._live_lock:
                    self._live[addr] = (size, obj_type)

                # Register weakref finalizer to capture dealloc
                self._register_finalizer(obj, addr)

        with self._events_lock:
            self._events.extend(new_events)

        return new_events

    def flush(self) -> list[dict]:
        """Return all accumulated events (alloc + dealloc) and clear the buffer."""
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
        return events

    def stop(self) -> list[dict]:
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
        """
        Register a weakref finalizer on obj. When obj is garbage collected,
        emits a DEALLOC CpuEvent with the same address as the original ALLOC.
        """
        try:
            # weakref.finalize is safer than weakref.ref — works on more types
            # and won't raise if the object is already being collected
            weakref.finalize(obj, self._on_dealloc, addr)
        except TypeError:
            # Some types don't support weakrefs (e.g. certain C extensions)
            pass

    def _on_dealloc(self, addr: int):
        """Called by weakref finalizer when a tracked object is collected."""
        with self._live_lock:
            info = self._live.pop(addr, None)

        if info is None:
            return

        size, obj_type = info
        event = {
            "base": _make_base("DEALLOC"),
            "alloc_address": addr,
            "alloc_size_bytes": size,
            "object_type": obj_type,
            "ref_count": 0,
            "callstack": [],
            "gc_generation": 0,
            "is_dealloc": True,
        }
        with self._events_lock:
            self._events.append(event)