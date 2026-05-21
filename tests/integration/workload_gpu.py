#!/usr/bin/env python3
"""
workload_gpu.py
---------------
GPU workload for the HPC profiler end-to-end test.

This script self-starts cupti_layer inside its own process so CUPTI
activity callbacks fire for its own CUDA context. The runner passes
two environment variables:

  HPC_GPU_SHM        — POSIX shm name for the GPU ring buffer
  HPC_CUPTI_SO       — path to cupti_layer.so

If those env vars are set, cupti_start() is called before any CUDA work.
If not set (standalone run), GPU profiling is skipped silently.

Prints "pid=<PID>" on stdout immediately so run_profiler.py can attach
the CPU profiler. Loops until SIGTERM.

Requirements:
    pip install cupy-cuda12x   (match your CUDA version)
"""

import ctypes
import os
import signal
import sys
import time
from pathlib import Path

# ── Announce PID immediately ──────────────────────────────────────────────
print(f"pid={os.getpid()}", flush=True)

# ── Self-start CPU profiler if env vars are set ───────────────────────────
# Do this BEFORE importing cupy so no CUDA threads exist yet.
# The profiler thread starts, then cupy loads — no GIL contention.
_cpu_shm = os.environ.get("HPC_CPU_SHM", "")
_rb_so   = os.environ.get("HPC_RB_SO", "")
_src_py  = os.environ.get("HPC_SRC_PYTHON", "")

if _cpu_shm and _rb_so and _src_py:
    try:
        sys.path.insert(0, _src_py)
        from python_memory_layer import PythonMemoryLayer
        from bridge import Bridge
        import threading as _threading

        _hpc_layer  = PythonMemoryLayer(nframe=16)
        _hpc_bridge = Bridge(shm_name=_cpu_shm)
        _hpc_bridge.open(create=False)
        _hpc_layer.start()
        sys._hpc_profiler_active = True

        def _hpc_profiler_thread():
            while getattr(sys, "_hpc_profiler_active", False):
                for e in _hpc_layer.collect():
                    _hpc_bridge.write(e)
                time.sleep(0.005)  # 5 ms — workload loop is 50 ms, giving ~10 polls per iteration
            _hpc_layer.stop()
            _hpc_bridge.close()

        _threading.Thread(target=_hpc_profiler_thread,
                          name="hpc_cpu_profiler", daemon=True).start()
        print(f"[workload_gpu] CPU profiler started  shm={_cpu_shm}", flush=True)
    except Exception as _e:
        print(f"[workload_gpu] CPU profiler failed: {_e}", flush=True)

# ── Try importing cupy and init CUDA context ────────────────────────────
try:
    import cupy as cp
    _ = cp.zeros(1)
    cp.cuda.Stream.null.synchronize()
    HAS_GPU = True
    print(f"[workload_gpu] cupy ready, device={cp.cuda.Device().id}", flush=True)
except ImportError:
    HAS_GPU = False
    print("[workload_gpu] cupy not found — CPU-only mode", flush=True)
except Exception as _e:
    HAS_GPU = False
    print(f"[workload_gpu] cupy init failed ({_e}) — CPU-only mode", flush=True)

# ── Self-start CUPTI AFTER CUDA context exists ────────────────────────────
# Must start AFTER cupy/CUDA init so CUPTI attaches to the live context.
_cupti_lib    = None
_cupti_started = False

_gpu_shm  = os.environ.get("HPC_GPU_SHM", "")
_cupti_so = os.environ.get("HPC_CUPTI_SO", "")

if _gpu_shm and _cupti_so and Path(_cupti_so).exists():
    try:
        _cupti_lib = ctypes.CDLL(_cupti_so)
        _cupti_lib.cupti_start.restype  = ctypes.c_int
        _cupti_lib.cupti_start.argtypes = [ctypes.c_char_p]
        _cupti_lib.cupti_stop.restype   = None
        _cupti_lib.cupti_stop.argtypes  = []

        ok = _cupti_lib.cupti_start(_gpu_shm.encode())
        if ok:
            _cupti_started = True
            print(f"[workload_gpu] cupti_start OK  shm={_gpu_shm}", flush=True)
        else:
            print(f"[workload_gpu] cupti_start returned 0", flush=True)
    except Exception as e:
        print(f"[workload_gpu] cupti load failed: {e}", flush=True)
else:
    print(f"[workload_gpu] HPC_GPU_SHM/HPC_CUPTI_SO not set — GPU events skipped", flush=True)

import numpy as np

# ── Signal handling ───────────────────────────────────────────────────────
_running = True

def _stop(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT,  _stop)

# ── Workload loop ─────────────────────────────────────────────────────────
iteration = 0

while _running:
    iteration += 1

    # CPU work — generates CPU alloc events
    size = 512 * (1 + (iteration % 4))
    a_cpu = np.random.randn(size, 64).astype(np.float32)
    b_cpu = np.random.randn(64, 128).astype(np.float32)
    c_cpu = a_cpu @ b_cpu

    # Python object churn
    data   = [float(x) for x in range(64)]
    lookup = {i: v for i, v in enumerate(data)}
    del data, lookup

    if HAS_GPU:
        # H2D — CUPTI TRANSFER event
        a_gpu = cp.asarray(a_cpu)
        b_gpu = cp.asarray(b_cpu)

        # Kernel — CUPTI KERNEL event
        c_gpu = cp.matmul(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()

        # D2H — CUPTI TRANSFER event
        _ = cp.asnumpy(c_gpu)
        cp.cuda.Stream.null.synchronize()

        del a_gpu, b_gpu, c_gpu
        cp.get_default_memory_pool().free_all_blocks()

    time.sleep(0.05)

# ── Cleanup ───────────────────────────────────────────────────────────────
if _cupti_started and _cupti_lib:
    _cupti_lib.cupti_stop()
    print(f"[workload_gpu] cupti_stop called", flush=True)

print(f"[workload_gpu] exiting after {iteration} iterations", flush=True)