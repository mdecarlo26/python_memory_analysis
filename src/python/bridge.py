"""
bridge.py
---------
Python writer side of the shared memory bridge.

Loads ring_buffer.so via cffi, opens (or creates) a named shared memory
segment, and exposes a simple write() API that serializes CpuEvent dicts
and writes them into the ring buffer for the C++ correlator to read.

Serialization format (little-endian struct):
  event_id        uint64
  timestamp_ns    uint64
  process_id      uint32
  thread_id       uint32
  event_type      uint8   (0=ALLOC, 1=DEALLOC)
  is_dealloc      uint8
  alloc_address   uint64
  alloc_size_bytes uint64
  ref_count       uint32
  gc_generation   uint8
  object_type_len uint16  (length of following string)
  object_type     bytes   (variable, object_type_len bytes, no null terminator)

Callstack is intentionally omitted from the binary format — it is large
and not needed by the correlator. It remains available in the JSON export.

Usage:
    bridge = Bridge(shm_name="/hpc_profiler_cpu")
    bridge.open(create=True)
    bridge.write(cpu_event_dict)
    bridge.close()
"""

import struct
import os
from pathlib import Path
from typing import Optional

try:
    import cffi
    _CFFI_AVAILABLE = True
except ImportError:
    _CFFI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Event type enum mapping
# ---------------------------------------------------------------------------

_EVENT_TYPE_MAP = {
    "ALLOC":      0,
    "DEALLOC":    1,
    "TRANSFER":   2,
    "PAGE_FAULT": 3,
    "KERNEL":     4,
}

# ---------------------------------------------------------------------------
# Struct format for fixed fields (everything except object_type string)
# < = little-endian
# Q = uint64, I = uint32, B = uint8, H = uint16
# ---------------------------------------------------------------------------

_FIXED_FMT = struct.Struct("<QQIQBBQQIbH")  # event_id,ts,pid,tid(uint64),etype,is_dealloc,addr,size,ref,gen,typelen
_FIXED_SIZE = _FIXED_FMT.size  # 40 bytes

# ---------------------------------------------------------------------------
# cffi declarations for ring_buffer.so
# ---------------------------------------------------------------------------

_CDEF = """
    typedef struct RingBuffer RingBuffer;

    RingBuffer* rb_create(const char* name, uint32_t capacity, int create);
    int         rb_write(RingBuffer* rb, const uint8_t* data, uint32_t len);
    int         rb_read(RingBuffer* rb, uint8_t* buf, uint32_t buf_len,
                        uint32_t* out_len);
    void        rb_stats(RingBuffer* rb, uint32_t* out_used, uint32_t* out_free,
                         uint32_t* out_dropped);
    void        rb_destroy(RingBuffer* rb);
    void        rb_unlink(const char* name);
"""

_SO_PATH = Path(__file__).parent.parent / "cpp" / "ring_buffer.so"


class Bridge:
    """
    Python writer side of the CPU event bridge.

    Opens a POSIX shared memory ring buffer and writes serialized
    CpuEvent dicts into it for the C++ correlator to consume.
    """

    def __init__(
        self,
        shm_name: str = "/hpc_profiler_cpu",
        capacity: int = 4 * 1024 * 1024,
    ):
        self._shm_name = shm_name
        self._capacity = capacity
        self._rb = None
        self._ffi = None
        self._lib = None
        self._open = False
        self._written = 0
        self._dropped = 0

    def open(self, create: bool = True):
        """
        Open the shared memory segment.
        create=True: create it (producer side, call this first).
        create=False: attach to existing (consumer side).
        """
        if self._open:
            return

        if not _CFFI_AVAILABLE:
            raise RuntimeError("cffi is required for the bridge. pip install cffi")

        if not _SO_PATH.exists():
            raise RuntimeError(
                f"ring_buffer.so not found at {_SO_PATH}. "
                "Run: python3 src/cpp/build.py"
            )

        ffi = cffi.FFI()
        ffi.cdef(_CDEF)
        lib = ffi.dlopen(str(_SO_PATH))

        rb = lib.rb_create(
            self._shm_name.encode(),
            self._capacity,
            1 if create else 0,
        )
        if rb == ffi.NULL:
            raise RuntimeError(
                f"Failed to {'create' if create else 'open'} "
                f"shared memory segment '{self._shm_name}'"
            )

        self._ffi = ffi
        self._lib = lib
        self._rb  = rb
        self._open = True

    def write(self, event: dict) -> bool:
        """
        Serialize a CpuEvent dict and write it to the ring buffer.
        Returns True on success, False if the buffer was full (dropped).
        """
        if not self._open:
            raise RuntimeError("Bridge is not open. Call open() first.")

        payload = self._serialize(event)
        data    = self._ffi.from_buffer(payload)
        result  = self._lib.rb_write(self._rb, data, len(payload))

        if result:
            self._written += 1
        else:
            self._dropped += 1

        return bool(result)

    def read(self, max_bytes: int = 4096) -> Optional[bytes]:
        """
        Read one record from the ring buffer.
        Returns the raw payload bytes, or None if the buffer is empty.
        Primarily for testing — the C++ correlator is the normal reader.
        """
        if not self._open:
            return None

        buf     = self._ffi.new(f"uint8_t[{max_bytes}]")
        out_len = self._ffi.new("uint32_t *")
        result  = self._lib.rb_read(self._rb, buf, max_bytes, out_len)

        if not result:
            return None

        return bytes(self._ffi.buffer(buf, out_len[0]))

    def stats(self) -> dict:
        """Return current buffer usage statistics."""
        if not self._open:
            return {"used": 0, "free": 0, "dropped": 0,
                    "written": self._written, "py_dropped": self._dropped}

        used    = self._ffi.new("uint32_t *")
        free    = self._ffi.new("uint32_t *")
        dropped = self._ffi.new("uint32_t *")
        self._lib.rb_stats(self._rb, used, free, dropped)

        return {
            "used":       int(used[0]),
            "free":       int(free[0]),
            "dropped":    int(dropped[0]),
            "written":    self._written,
            "py_dropped": self._dropped,
        }

    def close(self, unlink: bool = False):
        """Close the bridge. If unlink=True, remove the shm segment."""
        if not self._open:
            return
        self._lib.rb_destroy(self._rb)
        if unlink:
            self._lib.rb_unlink(self._shm_name.encode())
        self._rb   = None
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def written(self) -> int:
        return self._written

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(event: dict) -> bytes:
        """
        Pack a CpuEvent dict into the binary wire format.
        Returns a bytes object ready to pass to rb_write.
        """
        base         = event["base"]
        event_id     = base["event_id"]
        timestamp_ns = base["timestamp_ns"]
        process_id   = base["process_id"]
        thread_id    = base["thread_id"]
        event_type   = _EVENT_TYPE_MAP.get(base["event_type"], 0)

        is_dealloc      = 1 if event["is_dealloc"] else 0
        alloc_address   = event["alloc_address"]
        alloc_size      = event["alloc_size_bytes"]
        ref_count       = event["ref_count"]
        gc_generation   = event["gc_generation"]
        object_type_enc = event["object_type"].encode("utf-8")[:65535]
        object_type_len = len(object_type_enc)

        fixed = _FIXED_FMT.pack(
            event_id,
            timestamp_ns,
            process_id,
            thread_id,
            event_type,
            is_dealloc,
            alloc_address,
            alloc_size,
            ref_count,
            gc_generation,
            object_type_len,
        )

        return fixed + object_type_enc

    @staticmethod
    def deserialize(payload: bytes) -> dict:
        """
        Unpack a binary payload back into a CpuEvent-like dict.
        Used by tests and by the Python-side reader in integration tests.
        """
        if len(payload) < _FIXED_SIZE:
            raise ValueError(f"Payload too short: {len(payload)} < {_FIXED_SIZE}")

        (event_id, timestamp_ns, process_id, thread_id,
         event_type_int, is_dealloc, alloc_address, alloc_size,
         ref_count, gc_generation, object_type_len) = _FIXED_FMT.unpack(
            payload[:_FIXED_SIZE]
        )

        object_type = payload[_FIXED_SIZE:_FIXED_SIZE + object_type_len].decode("utf-8")

        type_map_rev = {v: k for k, v in _EVENT_TYPE_MAP.items()}
        event_type_str = type_map_rev.get(event_type_int, "ALLOC")

        return {
            "base": {
                "event_id":     event_id,
                "timestamp_ns": timestamp_ns,
                "process_id":   process_id,
                "thread_id":    thread_id,
                "event_type":   event_type_str,
            },
            "alloc_address":    alloc_address,
            "alloc_size_bytes": alloc_size,
            "object_type":      object_type,
            "ref_count":        ref_count,
            "gc_generation":    gc_generation,
            "is_dealloc":       bool(is_dealloc),
            "callstack":        [],  # not transmitted over the bridge
        }