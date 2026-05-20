"""
gpu_bridge.py
-------------
Python consumer for the GPU ring buffer written by cupti_layer.so.

Binary wire format (little-endian, mirrors cupti_layer.cpp serialise_event):

  ── Base header (53 bytes, struct "<QQIQBBQQIBH") ──────────────────────────
  Offset  Size  Field
  0        8    event_id              uint64
  8        8    timestamp_ns          uint64   CLOCK_MONOTONIC ns
  16       4    process_id            uint32
  20       8    thread_id             uint64   always 0 for GPU events
  28       1    event_type            uint8    2=TRANSFER 3=PAGE_FAULT 4=KERNEL
  29       1    is_dealloc            uint8    always 0
  30       8    alloc_address         uint64   unused, 0
  38       8    alloc_size_bytes      uint64   unused, 0
  46       4    ref_count             uint32   unused, 0
  50       1    gc_generation         int8     unused, 0
  51       2    label_len             uint16
  53       N    label                 utf-8    kernel name or transfer kind

  ── GPU extension (after label) ─────────────────────────────────────────────
  +0       4    device_id             uint32
  +4       8    src_address           uint64
  +12      8    dst_address           uint64
  +20      8    transfer_size         uint64
  +28      1    transfer_kind         uint8    0=H2D 1=D2H 2=D2D 3=PREFETCH
  +29      4    page_faults           uint32
  +33      8    kernel_duration_ns    uint64
  +41      4    stream_id             uint32
  +45      4    device_mem_used_mb    uint32
  +49      8    um_bytes_htod         uint64   [NEW] UM migration bytes H→D
  +57      8    um_bytes_dtoh         uint64   [NEW] UM migration bytes D→H
  +65      4    grid_x                uint32   [NEW] kernel grid dims
  +69      4    grid_y                uint32
  +73      4    grid_z                uint32
  +77      4    block_x               uint32   [NEW] kernel block dims
  +81      4    block_y               uint32
  +85      4    block_z               uint32
  +89      4    registers_per_thread  uint32   [NEW]
  +93      4    shared_mem_bytes      uint32   [NEW] static + dynamic shared mem
  +97      4    correlation_id        uint32   [NEW] CUPTI correlation ID

  Header = 53 bytes
  Extension = 4+8+8+8+1+4+8+4+4+8+8+4+4+4+4+4+4+4+4+4 = 101 bytes
  Minimum total (label_len=0) = 53 + 101 = 154 bytes
"""

import struct
import ctypes
from pathlib import Path
from typing import Optional

ROOT   = Path(__file__).parent.parent.parent
_RB_SO = str(ROOT / "src" / "cpp" / "ring_buffer.so")

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

# Base header: event_id, ts, pid, tid, etype, is_dealloc,
#              alloc_addr, alloc_size, ref_count, gc_gen, label_len
_HDR_FMT = struct.Struct("<QQIQBBQQIBH")   # 53 bytes

# GPU extension block: all fields in serialisation order
# Original fields (49 bytes):
#   device_id uint32, src uint64, dst uint64, xfer_size uint64,
#   xfer_kind uint8, page_faults uint32,
#   kernel_duration_ns uint64, stream_id uint32, device_mem_used_mb uint32
# New fields (52 bytes):
#   um_bytes_htod uint64, um_bytes_dtoh uint64,
#   grid_x/y/z uint32x3, block_x/y/z uint32x3,
#   registers_per_thread uint32, shared_mem_bytes uint32,
#   correlation_id uint32
_EXT_FMT = struct.Struct("<IQQQBIQIIQQIIIIIIIIi")
# Breakdown: I=device_id, Q=src, Q=dst, Q=xfer_size,
#            B=xfer_kind, I=page_faults, Q=kern_dur, I=stream, I=dev_mem,
#            Q=um_htod, Q=um_dtoh,
#            I=gx, I=gy, I=gz, I=bx, I=by, I=bz,
#            I=regs, I=shmem, i=corr_id (signed to handle CUPTI's uint32)
# Sizes: 4+8+8+8+1+4+8+4+4+8+8+4+4+4+4+4+4+4+4+4 = 101 bytes

_MIN_LEN = _HDR_FMT.size + _EXT_FMT.size   # 53 + 101 = 154


class GpuBridge:
    """Consumer-side reader for the GPU ring buffer created by cupti_layer.so."""

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
        lib.rb_read.restype    = ctypes.c_int
        lib.rb_read.argtypes   = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
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
            self._rb, self._buf, ctypes.c_uint32(len(self._buf)),
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
    # Deserialiser
    # ------------------------------------------------------------------

    @staticmethod
    def deserialize(payload: bytes) -> dict:
        """
        Parse a raw GPU event payload into a GpuEvent dict.
        Backwards-compatible: payloads shorter than the new minimum are
        parsed with zero-filled new fields so old cupti_layer.so binaries
        still work until rebuilt.
        """
        n = len(payload)
        if n < _HDR_FMT.size:
            raise ValueError(f"GPU payload too short for header: {n} < {_HDR_FMT.size}")

        # ── Header ────────────────────────────────────────────────────────
        (
            event_id, timestamp_ns, process_id, _thread_id,
            event_type_byte, _is_dealloc,
            _alloc_addr, _alloc_size, _ref_count, _gc_gen,
            label_len,
        ) = _HDR_FMT.unpack_from(payload, 0)

        # ── Variable-length label ─────────────────────────────────────────
        off = _HDR_FMT.size
        if n < off + label_len:
            raise ValueError(f"GPU payload truncated at label: need {off+label_len}, have {n}")
        label = payload[off: off + label_len].decode("utf-8", errors="replace")
        off  += label_len

        # ── Extension fields (with backwards-compat zero-fill) ─────────────
        ext_available = n - off

        # Defaults for all extension fields
        device_id = src_address = dst_address = transfer_size = 0
        transfer_kind_byte = page_faults = stream_id = device_mem_used_mb = 0
        kernel_duration_ns = 0
        um_bytes_htod = um_bytes_dtoh = 0
        grid_x = grid_y = grid_z = 0
        block_x = block_y = block_z = 0
        registers_per_thread = shared_mem_bytes = correlation_id = 0

        if ext_available >= _EXT_FMT.size:
            # Full new-format payload
            (
                device_id, src_address, dst_address, transfer_size,
                transfer_kind_byte, page_faults,
                kernel_duration_ns, stream_id, device_mem_used_mb,
                um_bytes_htod, um_bytes_dtoh,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                registers_per_thread, shared_mem_bytes,
                correlation_id,
            ) = _EXT_FMT.unpack_from(payload, off)
        elif ext_available >= 49:
            # Old format — original 9 fields only (49 bytes)
            _old = struct.Struct("<IQQQBIQii")
            (
                device_id, src_address, dst_address, transfer_size,
                transfer_kind_byte, page_faults,
                kernel_duration_ns, stream_id, device_mem_used_mb,
            ) = _old.unpack_from(payload, off)
        # else: payload has no extension at all — all zeros

        return {
            "base": {
                "event_id":    event_id,
                "timestamp_ns": timestamp_ns,
                "process_id":  process_id,
                "thread_id":   0,
                "event_type":  _EVENT_TYPE.get(event_type_byte,
                                               f"UNKNOWN_{event_type_byte}"),
            },
            # Original fields
            "device_id":           device_id,
            "src_address":         src_address,
            "dst_address":         dst_address,
            "transfer_size_bytes": transfer_size,
            "transfer_kind":       _TRANSFER_KIND.get(transfer_kind_byte, "UNKNOWN"),
            "um_page_faults":      page_faults,
            "kernel_name":         label,
            "kernel_duration_ns":  kernel_duration_ns,
            "stream_id":           stream_id,
            "device_mem_used_mb":  device_mem_used_mb,
            # New fields
            "um_bytes_htod":          um_bytes_htod,
            "um_bytes_dtoh":          um_bytes_dtoh,
            "grid":                   {"x": grid_x, "y": grid_y, "z": grid_z},
            "block":                  {"x": block_x, "y": block_y, "z": block_z},
            "registers_per_thread":   registers_per_thread,
            "shared_mem_bytes":       shared_mem_bytes,
            "correlation_id":         correlation_id,
        }