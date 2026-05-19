"""
gpu_bridge.py
-------------
Python consumer for the GPU ring buffer written by cupti_layer.so.

cupti_layer.so calls rb_create(shm_name, capacity, 1) internally when
cupti_start() is called — it owns and creates the segment. This module
opens a *consumer* handle (create=0) to the same segment and provides:

  GpuBridge.open()         — attach to existing segment
  GpuBridge.read()         — read one raw payload (bytes) or None
  GpuBridge.close()        — detach (never unlinks — cupti_layer owns it)
  GpuBridge.deserialize()  — static: bytes → GpuEvent dict

Binary wire format (little-endian, mirrors cupti_layer.cpp write order):

  Offset  Size  Field
  ------  ----  -----
  0        8    event_id              (uint64)
  8        8    timestamp_ns          (uint64)
  16       4    process_id            (uint32)
  20       8    thread_id             (uint64)  — always 0 for GPU events
  28       1    event_type            (uint8)   2=TRANSFER 3=PAGE_FAULT 4=KERNEL
  29       1    is_dealloc            (uint8)   — always 0
  30       8    alloc_address         (uint64)  — unused; 0
  38       8    alloc_size            (uint64)  — unused; 0
  46       4    ref_count             (uint32)  — unused; 0
  50       1    gc_gen                (int8)    — unused; 0
  51       2    label_len             (uint16)
  53       N    label                 (utf-8)   — kernel name or transfer kind
  53+N     4    device_id             (uint32)
  57+N     8    src_address           (uint64)
  65+N     8    dst_address           (uint64)
  73+N     8    transfer_size         (uint64)
  81+N     1    transfer_kind         (uint8)
  82+N     4    page_faults           (uint32)
  86+N     8    kernel_duration_ns    (uint64)
  94+N     4    stream_id             (uint32)
  98+N     4    device_mem_used_mb    (uint32)

Header struct "<QQIQBBQQIBH"  = 8+8+4+8+1+1+8+8+4+1+2 = 53 bytes
Extension struct "<IQQQBIQii" = 4+8+8+8+1+4+8+4+4     = 49 bytes
Minimum total (label_len=0)  = 53 + 49                 = 102 bytes
"""

import struct
import ctypes
import sys
from pathlib import Path
from typing import Optional

ROOT    = Path(__file__).parent.parent.parent
_RB_SO  = str(ROOT / "src" / "cpp" / "ring_buffer.so")

_TRANSFER_KIND = {
    0: "HOST_TO_DEVICE",
    1: "DEVICE_TO_HOST",
    2: "DEVICE_TO_DEVICE",
    3: "PREFETCH",
    4: "UNIFIED_MEMORY",
}

_EVENT_TYPE = {
    2: "TRANSFER",
    3: "PAGE_FAULT",
    4: "KERNEL",
}

_HDR_FMT = struct.Struct("<QQIQBBQQIBH")   # 53 bytes
_EXT_FMT = struct.Struct("<IQQQBIQII")    # 4+8+8+8+1+4+8+4+4 = 45 bytes
#  device_id uint32, src uint64, dst uint64, xfer_size uint64,
#  transfer_kind uint8, page_faults uint32,
#  kernel_duration_ns uint64, stream_id uint32, device_mem_used_mb uint32
_MIN_LEN  = _HDR_FMT.size + _EXT_FMT.size   # 53 + 45 = 98 (label_len=0)


class GpuBridge:
    """
    Consumer-side reader for the GPU ring buffer created by cupti_layer.so.
    """

    def __init__(
        self,
        shm_name: str = "/hpc_profiler_gpu",
        capacity: int = 8 * 1024 * 1024,
        rb_so_path: str = _RB_SO,
    ):
        self._shm_name = shm_name.encode() if isinstance(shm_name, str) else shm_name
        self._capacity = capacity
        self._so_path  = rb_so_path
        self._lib      = None
        self._rb       = None
        self._buf      = ctypes.create_string_buffer(65536)
        self.is_open   = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        lib = ctypes.CDLL(self._so_path)

        lib.rb_create.restype  = ctypes.c_void_p
        lib.rb_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int]

        lib.rb_read.restype  = ctypes.c_int
        lib.rb_read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]

        lib.rb_destroy.restype  = None
        lib.rb_destroy.argtypes = [ctypes.c_void_p]

        self._lib = lib
        rb = lib.rb_create(self._shm_name, self._capacity, 0)
        if not rb:
            raise RuntimeError(
                f"GpuBridge: failed to open ring buffer '{self._shm_name.decode()}'. "
                "Is cupti_start() running?"
            )
        self._rb     = rb
        self.is_open = True

    def read(self) -> Optional[bytes]:
        if not self.is_open:
            return None
        out_len = ctypes.c_uint32(0)
        ok = self._lib.rb_read(
            self._rb,
            self._buf,
            ctypes.c_uint32(len(self._buf)),
            ctypes.byref(out_len),
        )
        if not ok or out_len.value == 0:
            return None
        return bytes(self._buf.raw[: out_len.value])

    def close(self) -> None:
        """Detach from ring buffer. Does NOT unlink — cupti_layer owns the segment."""
        if self._rb and self._lib:
            self._lib.rb_destroy(self._rb)
        self._rb     = None
        self.is_open = False

    # ------------------------------------------------------------------
    # Deserializer
    # ------------------------------------------------------------------

    @staticmethod
    def deserialize(payload: bytes) -> dict:
        """
        Parse a raw GPU event payload into a GpuEvent dict.
        Format mirrors the write order in cupti_layer.cpp and the read
        order in test_gpu.cu's deserialize() function.
        """
        n = len(payload)
        if n < _MIN_LEN:
            raise ValueError(f"GPU payload too short: {n} < {_MIN_LEN}")

        # ── Header (53 bytes) ──────────────────────────────────────────
        (
            event_id,        # Q uint64 offset  0
            timestamp_ns,    # Q uint64 offset  8
            process_id,      # I uint32 offset 16
            _thread_id,      # Q uint64 offset 20
            event_type_byte, # B uint8  offset 28
            _is_dealloc,     # B uint8  offset 29
            _alloc_addr,     # Q uint64 offset 30
            _alloc_size,     # Q uint64 offset 38
            _ref_count,      # I uint32 offset 46
            _gc_gen,         # B uint8  offset 50
            label_len,       # H uint16 offset 51
        ) = _HDR_FMT.unpack_from(payload, 0)

        # ── Variable-length label ──────────────────────────────────────
        off = 53
        if n < off + label_len:
            raise ValueError(
                f"GPU payload truncated: label needs {off+label_len} bytes, have {n}"
            )
        label = payload[off: off + label_len].decode("utf-8", errors="replace")
        off  += label_len

        # ── Extension fields ───────────────────────────────────────────
        if n < off + _EXT_FMT.size:
            raise ValueError(
                f"GPU payload truncated in extension fields at offset {off}, have {n}"
            )
        (
            device_id,            # I uint32
            src_address,          # Q uint64
            dst_address,          # Q uint64
            transfer_size,        # Q uint64
            transfer_kind_byte,   # B uint8
            page_faults,          # I uint32
            kernel_duration_ns,   # Q uint64
            stream_id,            # I uint32
            device_mem_used_mb,   # I uint32
        ) = _EXT_FMT.unpack_from(payload, off)

        return {
            "base": {
                "event_id":    event_id,
                "timestamp_ns": timestamp_ns,
                "process_id":  process_id,
                "thread_id":   0,
                "event_type":  _EVENT_TYPE.get(event_type_byte, f"UNKNOWN_{event_type_byte}"),
            },
            "device_id":              device_id,
            "src_address":            src_address,
            "dst_address":            dst_address,
            "transfer_size_bytes":    transfer_size,
            "transfer_kind":          _TRANSFER_KIND.get(transfer_kind_byte, "UNKNOWN"),
            "um_page_faults":         page_faults,
            "kernel_name":            label,
            "kernel_duration_ns":     kernel_duration_ns,
            "stream_id":              stream_id,
            "device_mem_used_mb":     device_mem_used_mb,
        }