import os
import time
import json
import threading
import ctypes
import gc
import tracemalloc

import numpy as np
import cupy as cp
import psutil


def jwrite(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def start_host_sampler(out_path: str, period_ms: int = 25, enable_tracemalloc: bool = True):
    """
    Periodically appends host-side statistics to the JSONL file.
    This is your CPU/host memory + cleanup story.
    """
    proc = psutil.Process()
    stop = threading.Event()

    if enable_tracemalloc:
        tracemalloc.start()

    # Prime CPU percent measurement
    proc.cpu_percent(interval=None)

    # Baselines (so you can graph deltas)
    t0 = time.time_ns()
    try:
        base_mem = proc.memory_info()
    except Exception:
        base_mem = None

    # Linux-only: page faults available via memory_info() fields on many distros
    def get_faults(mi):
        # psutil returns platform-specific fields; guard everything.
        minor = getattr(mi, "pfaults", None) or getattr(mi, "minflt", None)
        major = getattr(mi, "pageins", None) or getattr(mi, "majflt", None)
        return minor, major

    def run():
        last_gc = gc.get_count()
        while not stop.is_set():
            t_ns = time.time_ns()
            dt_ns = t_ns - t0

            evt = {"type": "host_stats", "t_ns": t_ns, "dt_ns": dt_ns}

            # Process memory
            try:
                mi = proc.memory_info()
                evt["rss_bytes"] = int(getattr(mi, "rss", 0))
                evt["vms_bytes"] = int(getattr(mi, "vms", 0))
                evt["shared_bytes"] = int(getattr(mi, "shared", 0)) if hasattr(mi, "shared") else None
                evt["data_bytes"] = int(getattr(mi, "data", 0)) if hasattr(mi, "data") else None
                evt["text_bytes"] = int(getattr(mi, "text", 0)) if hasattr(mi, "text") else None

                minor, major = get_faults(mi)
                if minor is not None:
                    evt["minor_faults"] = int(minor)
                if major is not None:
                    evt["major_faults"] = int(major)

                if base_mem is not None:
                    evt["rss_delta_bytes"] = int(evt["rss_bytes"] - getattr(base_mem, "rss", 0))
            except Exception as e:
                evt["mem_err"] = str(e)

            # CPU times + utilization
            try:
                ct = proc.cpu_times()
                evt["cpu_user_s"] = float(getattr(ct, "user", 0.0))
                evt["cpu_system_s"] = float(getattr(ct, "system", 0.0))
                evt["cpu_percent"] = float(proc.cpu_percent(interval=None))
            except Exception as e:
                evt["cpu_err"] = str(e)

            # Threads
            try:
                evt["num_threads"] = int(proc.num_threads())
            except Exception:
                pass

            # Python GC (cleanup behavior)
            try:
                gc_counts = gc.get_count()
                evt["py_gc_count_gen0"] = int(gc_counts[0])
                evt["py_gc_count_gen1"] = int(gc_counts[1])
                evt["py_gc_count_gen2"] = int(gc_counts[2])
                evt["py_gc_delta_gen0"] = int(gc_counts[0] - last_gc[0])
                evt["py_gc_delta_gen1"] = int(gc_counts[1] - last_gc[1])
                evt["py_gc_delta_gen2"] = int(gc_counts[2] - last_gc[2])
                last_gc = gc_counts
            except Exception:
                pass

            # Python allocation pressure (not RSS; Python-heap behavior)
            if enable_tracemalloc:
                try:
                    cur, peak = tracemalloc.get_traced_memory()
                    evt["py_alloc_current_bytes"] = int(cur)
                    evt["py_alloc_peak_bytes"] = int(peak)
                except Exception:
                    pass

            jwrite(out_path, evt)
            time.sleep(period_ms / 1000.0)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return stop


def mark(out_path: str, msg: str, extra: dict | None = None):
    evt = {"type": "marker", "msg": msg, "t_ns": time.time_ns()}
    if extra:
        evt.update(extra)
    jwrite(out_path, evt)


def maybe_force_gc(out_path: str, reason: str):
    """
    Optional explicit cleanup marker: tells you what happens when you force collection.
    """
    mark(out_path, f"gc_collect_begin:{reason}")
    before = gc.get_count()
    collected = gc.collect()
    after = gc.get_count()
    mark(out_path, f"gc_collect_end:{reason}", {
        "gc_collected": int(collected),
        "gc_before": list(before),
        "gc_after": list(after),
    })


def main():
    out_cuda = "events_cuda_stats.jsonl"
    out_python = "events_python_stats.jsonl"
    out_host = "events_host_stats.jsonl"

    # CUPTI tracer writes device/cuda events here
    os.environ["CUPTI_TRACE_OUT"] = out_cuda

    # Load CUPTI tracer before first CUDA call
    ctypes.CDLL("./libcupti_trace.so", mode=ctypes.RTLD_GLOBAL)

    # Start rich host stats sampler
    stop_host = start_host_sampler(out_host, period_ms=1, enable_tracemalloc=True)

    # --- Host allocations ---
    M, N, K = 4096, 4096, 4096
    mark(out_host, "host_alloc_begin", {"shape_A": [M, K], "shape_B": [K, N], "dtype": "float32"})
    A_h = np.random.rand(M, K).astype(np.float32)
    B_h = np.random.rand(K, N).astype(np.float32)
    mark(out_host, "host_alloc_end")

    # Optional: show cleanup behavior (uncomment if you want forced GC markers)
    # maybe_force_gc(out_host, "after_host_alloc")

    # --- Push to GPU ---
    mark(out_cuda, "htoD_begin")
    A_d = cp.asarray(A_h)
    B_d = cp.asarray(B_h)
    C_d = cp.empty((M, N), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    mark(out_cuda, "htoD_end")

    # --- Kernel ---
    kernel_src = r"""
    extern "C" __global__
    void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
      int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
      int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
      if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) sum += A[row*K + i] * B[i*N + col];
        C[row*N + col] = sum;
      }
    }
    """
    matmul = cp.RawKernel(kernel_src, "matmul")
    block = (16, 16, 1)
    grid = ((N + block[0] - 1) // block[0], (M + block[1] - 1) // block[1], 1)

    mark(out_cuda, "kernel_begin", {"grid": list(grid), "block": list(block)})
    matmul(grid, block, (A_d, B_d, C_d, M, N, K))
    cp.cuda.runtime.deviceSynchronize()
    mark(out_cuda, "kernel_end")

    # --- Pull back ---
    mark(out_cuda, "DtoH_begin")
    C_h = cp.asnumpy(C_d)
    mark(out_cuda, "DtoH_end")

    # --- Host cleanup signals ---
    mark(out_host, "host_cleanup_begin")
    # Drop references so Python can free them eventually
    del A_h, B_h
    # Optional explicit GC to see "cleaning" effect
    maybe_force_gc(out_host, "after_del_host_arrays")
    mark(out_host, "host_cleanup_end")

    # --- Device cleanup signals ---
    mark(out_cuda, "device_cleanup_begin")
    # Free device arrays (CuPy will release memory; allocator may cache)
    del A_d, B_d, C_d
    cp.cuda.runtime.deviceSynchronize()
    mark(out_cuda, "device_cleanup_end")
    # Optional: force another GC since CuPy objects are Python objects too
    maybe_force_gc(out_cuda, "after_del_device_arrays")

    stop_host.set()

    print("C[0,0] =", float(C_h[0, 0]))
    print("C[-1,-1] =", float(C_h[-1, -1]))
    print("Wrote JSONL trace to:", out_host, out_cuda, out_python) 


if __name__ == "__main__":
    main()