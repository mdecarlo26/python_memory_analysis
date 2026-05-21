#!/usr/bin/env python3
"""
workload_ml.py
--------------
ML training workload for the HPC profiler demo.

Simulates a realistic deep-learning training loop:
  Phase 0 — Dataset load      : allocate raw data batch (large float32 arrays)
  Phase 1 — Preprocessing     : normalise, one-hot encode, build feature matrix
  Phase 2 — Forward pass      : linear layers + ReLU activations
  Phase 3 — Loss computation  : cross-entropy loss, softmax
  Phase 4 — Backward pass     : gradient buffers (same shape as weights)
  Phase 5 — Optimiser step    : SGD momentum update, weight decay
  Phase 6 — Evaluation        : validation forward pass + accuracy

Each phase allocates arrays with realistic shapes/dtypes so the profiler
captures array_shape, array_dtype, is_numpy_buffer, and (on GPU) H2D/D2H
transfer events with matching sizes.

The self-start CPU/GPU profiler protocol is identical to workload_gpu.py:
  HPC_CPU_SHM, HPC_RB_SO, HPC_SRC_PYTHON  → CPU ring buffer
  HPC_GPU_SHM, HPC_CUPTI_SO               → GPU ring buffer (optional)

Usage (standalone, no profiler):
    python3 workload_ml.py

Usage (via run_profiler.py):
    python3 run_profiler.py --target tests/integration/workload_ml.py

Prints phase progress to stdout so the runner's log shows what the workload
is doing at each memory event.
"""

import ctypes
import os
import signal
import sys
import time
from pathlib import Path

# ── Announce PID immediately ──────────────────────────────────────────────
print(f"pid={os.getpid()}", flush=True)

# ─────────────────────────────────────────────────────────────────────────
# Self-start CPU profiler BEFORE any heavy imports so no CUDA threads exist
# ─────────────────────────────────────────────────────────────────────────
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
                time.sleep(0.005)   # 5 ms poll — catches short-lived arrays
            _hpc_layer.stop()
            _hpc_bridge.close()

        _threading.Thread(
            target=_hpc_profiler_thread,
            name="hpc_cpu_profiler",
            daemon=True,
        ).start()
        print(f"[workload_ml] CPU profiler started  shm={_cpu_shm}", flush=True)
    except Exception as _e:
        print(f"[workload_ml] CPU profiler failed: {_e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────
# Import numpy (always present) then try cupy for GPU path
# ─────────────────────────────────────────────────────────────────────────
import numpy as np
import numpy.random  # force lazy submodule import now to avoid races

try:
    import cupy as cp
    _warm = cp.zeros(1)               # initialise CUDA context
    cp.cuda.Stream.null.synchronize()
    HAS_GPU = True
    print(f"[workload_ml] cupy ready, device={cp.cuda.Device().id}", flush=True)
except ImportError:
    HAS_GPU = False
    print("[workload_ml] cupy not found — CPU-only mode", flush=True)
except Exception as _e:
    HAS_GPU = False
    print(f"[workload_ml] cupy init failed ({_e}) — CPU-only mode", flush=True)

# ─────────────────────────────────────────────────────────────────────────
# Self-start CUPTI AFTER CUDA context is live
# ─────────────────────────────────────────────────────────────────────────
_cupti_lib     = None
_cupti_started = False
_gpu_shm       = os.environ.get("HPC_GPU_SHM", "")
_cupti_so      = os.environ.get("HPC_CUPTI_SO", "")

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
            print(f"[workload_ml] cupti_start OK  shm={_gpu_shm}", flush=True)
        else:
            print("[workload_ml] cupti_start returned 0", flush=True)
    except Exception as _e:
        print(f"[workload_ml] cupti load failed: {_e}", flush=True)
else:
    if HAS_GPU:
        print("[workload_ml] HPC_GPU_SHM/HPC_CUPTI_SO not set — GPU events skipped",
              flush=True)

# ─────────────────────────────────────────────────────────────────────────
# Model hyperparameters  (kept small enough to run on any machine quickly)
# ─────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 128    # samples per mini-batch
INPUT_DIM     = 784    # e.g. 28×28 flattened image
HIDDEN_DIM    = 256    # first hidden layer width
HIDDEN_DIM2   = 128    # second hidden layer width
OUTPUT_DIM    = 10     # number of classes
LR            = 0.01   # SGD learning rate
MOMENTUM      = 0.9    # SGD momentum coefficient
WEIGHT_DECAY  = 1e-4   # L2 regularisation factor
VAL_BATCH     = 256    # validation set size (larger than train batch)

# ─────────────────────────────────────────────────────────────────────────
# Weight initialisation  (Xavier uniform, stored as float32)
# ─────────────────────────────────────────────────────────────────────────
def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

# Weights and biases for a 3-layer MLP
W1 = _xavier(INPUT_DIM,   HIDDEN_DIM)
b1 = np.zeros(HIDDEN_DIM,  dtype=np.float32)
W2 = _xavier(HIDDEN_DIM,   HIDDEN_DIM2)
b2 = np.zeros(HIDDEN_DIM2, dtype=np.float32)
W3 = _xavier(HIDDEN_DIM2,  OUTPUT_DIM)
b3 = np.zeros(OUTPUT_DIM,  dtype=np.float32)

# SGD momentum buffers (same shapes as weights)
vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)

# ─────────────────────────────────────────────────────────────────────────
# Pure-numpy forward / backward helpers (CPU path)
# ─────────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=np.empty_like(x))

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    n = labels.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-9).mean()

def forward_cpu(X, W1, b1, W2, b2, W3, b3):
    """Returns (logits, cache) where cache holds intermediate activations."""
    z1 = X  @ W1 + b1          # (B, H1)
    a1 = relu(z1)
    z2 = a1 @ W2 + b2          # (B, H2)
    a2 = relu(z2)
    z3 = a2 @ W3 + b3          # (B, C)
    return z3, (X, z1, a1, z2, a2, z3)

def backward_cpu(cache, labels, W2, W3):
    """Returns gradient dict. Allocates one array per parameter."""
    X, z1, a1, z2, a2, z3 = cache
    n = labels.shape[0]

    # Output layer
    probs   = softmax(z3)
    dz3     = probs.copy()
    dz3[np.arange(n), labels] -= 1.0
    dz3    /= n

    dW3 = a2.T @ dz3                        # (H2, C)
    db3 = dz3.sum(axis=0)                   # (C,)
    da2 = dz3 @ W3.T                        # (B, H2)

    # Second hidden layer
    dz2 = da2 * relu_grad(z2)
    dW2 = a1.T @ dz2                        # (H1, H2)
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T                        # (B, H1)

    # First hidden layer
    dz1 = da1 * relu_grad(z1)
    dW1 = X.T  @ dz1                        # (I, H1)
    db1 = dz1.sum(axis=0)

    return dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)

# ─────────────────────────────────────────────────────────────────────────
# Signal handling
# ─────────────────────────────────────────────────────────────────────────
_running = True

def _stop(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT,  _stop)

# ─────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────
iteration = 0
print("[workload_ml] starting training loop", flush=True)

while _running:
    iteration += 1
    t_iter = time.perf_counter()

    # ── Phase 0: Dataset load ─────────────────────────────────────────────
    # Simulate loading a raw mini-batch from disk: uint8 pixels + int64 labels.
    # uint8 → float32 cast is the classic memory spike every ML pipeline has.
    raw_pixels = np.random.randint(0, 256,
                                   (BATCH_SIZE, INPUT_DIM),
                                   dtype=np.uint8)           # (128, 784) uint8
    labels_int = np.random.randint(0, OUTPUT_DIM,
                                   BATCH_SIZE,
                                   dtype=np.int64)           # (128,)     int64

    # ── Phase 1: Preprocessing ────────────────────────────────────────────
    # Normalise to [-1, 1] float32 — new allocation, different dtype.
    X_batch = (raw_pixels.astype(np.float32) / 127.5) - 1.0  # (128, 784) float32
    del raw_pixels   # raw pixels no longer needed — triggers DEALLOC

    # Augment: random horizontal flip (column-reverse) for half the batch
    flip_mask = np.random.rand(BATCH_SIZE) > 0.5              # (128,)     bool
    X_batch[flip_mask] = X_batch[flip_mask, ::-1]

    # Per-feature mean subtraction using a running mean buffer
    feature_mean = X_batch.mean(axis=0)                       # (784,)     float32
    X_batch -= feature_mean

    # ── Phase 2: Forward pass ─────────────────────────────────────────────
    logits, cache = forward_cpu(X_batch, W1, b1, W2, b2, W3, b3)
    # cache holds: X (128,784), z1 (128,256), a1 (128,256),
    #              z2 (128,128), a2 (128,128), z3 (128,10)

    # ── Phase 3: Loss computation ─────────────────────────────────────────
    probs = softmax(logits)                                    # (128, 10)  float32
    loss  = cross_entropy(probs, labels_int)

    # ── Phase 4: Backward pass ────────────────────────────────────────────
    grads = backward_cpu(cache, labels_int, W2, W3)
    # grads: 6 arrays — dW1 (784,256), db1 (256,), dW2 (256,128),
    #                    db2 (128,),   dW3 (128,10), db3 (10,)

    # ── Phase 5: Optimiser step  (SGD with momentum + weight decay) ───────
    for (W, b, vW, vb, gW, gb) in [
        (W1, b1, vW1, vb1, grads["dW1"], grads["db1"]),
        (W2, b2, vW2, vb2, grads["dW2"], grads["db2"]),
        (W3, b3, vW3, vb3, grads["dW3"], grads["db3"]),
    ]:
        # Weight decay — adds L2 gradient in-place
        gW = gW + WEIGHT_DECAY * W
        # Momentum update (in-place, no extra alloc after first iteration)
        vW[:] = MOMENTUM * vW - LR * gW
        vb[:] = MOMENTUM * vb - LR * gb
        W += vW
        b += vb

    # ── GPU path (when cupy is available) ─────────────────────────────────
    # Mirrors the CPU computation on-device to generate CUPTI transfer events.
    # Each cp.asarray triggers an H2D TRANSFER; cp.asnumpy triggers D2H.
    if HAS_GPU:
        # H2D — upload this iteration's batch + weights
        X_gpu  = cp.asarray(X_batch)   # (128, 784) float32  H2D
        W1_gpu = cp.asarray(W1)        # (784, 256) float32  H2D
        b1_gpu = cp.asarray(b1)        # (256,)     float32  H2D
        W2_gpu = cp.asarray(W2)        # (256, 128) float32  H2D
        b2_gpu = cp.asarray(b2)        # (128,)     float32  H2D
        W3_gpu = cp.asarray(W3)        # (128,  10) float32  H2D
        b3_gpu = cp.asarray(b3)        # (10,)      float32  H2D

        # Forward pass on GPU
        z1_gpu = X_gpu  @ W1_gpu + b1_gpu
        a1_gpu = cp.maximum(z1_gpu, 0.0)
        z2_gpu = a1_gpu @ W2_gpu + b2_gpu
        a2_gpu = cp.maximum(z2_gpu, 0.0)
        z3_gpu = a2_gpu @ W3_gpu + b3_gpu

        # Softmax + loss
        ex_gpu    = cp.exp(z3_gpu - z3_gpu.max(axis=1, keepdims=True))
        probs_gpu = ex_gpu / ex_gpu.sum(axis=1, keepdims=True)

        # D2H — bring loss and predictions back for logging
        preds_cpu = cp.asnumpy(probs_gpu.argmax(axis=1))   # (128,)  D2H
        cp.cuda.Stream.null.synchronize()

        # Free GPU buffers each iteration so the memory pool turns over
        del (X_gpu, W1_gpu, b1_gpu, W2_gpu, b2_gpu, W3_gpu, b3_gpu,
             z1_gpu, a1_gpu, z2_gpu, a2_gpu, z3_gpu, ex_gpu, probs_gpu)
        cp.get_default_memory_pool().free_all_blocks()

    # ── Phase 6: Evaluation (every 5 iterations) ─────────────────────────
    # Larger validation batch — distinct allocation footprint from training.
    if iteration % 5 == 0:
        X_val   = (np.random.randint(0, 256,
                                     (VAL_BATCH, INPUT_DIM),
                                     dtype=np.uint8
                                     ).astype(np.float32) / 127.5) - 1.0
        y_val   = np.random.randint(0, OUTPUT_DIM, VAL_BATCH, dtype=np.int64)

        val_logits, _ = forward_cpu(X_val, W1, b1, W2, b2, W3, b3)
        val_preds     = val_logits.argmax(axis=1)
        val_acc       = (val_preds == y_val).mean()

        del X_val, y_val, val_logits, val_preds   # explicit frees → DEALLOCs

        print(
            f"[workload_ml] iter={iteration:4d}  "
            f"loss={loss:.4f}  val_acc={val_acc:.3f}  "
            f"iter_ms={(time.perf_counter()-t_iter)*1000:.1f}",
            flush=True,
        )

    # Brief pause BEFORE explicit cleanup so the profiler's frame scan (Pass B)
    # has a window to see this iteration's local arrays while they are still
    # referenced as local variables in this frame.
    time.sleep(0.030)

    # Explicit cleanup of per-iteration temporaries so dealloc events flow
    del X_batch, logits, cache, probs, grads, feature_mean, flip_mask, labels_int

# ─────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────
if _cupti_started and _cupti_lib:
    _cupti_lib.cupti_stop()
    print("[workload_ml] cupti_stop called", flush=True)

print(f"[workload_ml] done after {iteration} iterations", flush=True)