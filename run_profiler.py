#!/usr/bin/env python3
"""
run_profiler.py
---------------
End-to-end runner: attaches both the Python memory profiler AND the GPU
profiler into a running target process, collects events from both ring
buffers simultaneously, correlates them, and writes a ProfilingSession
JSON ready to load in the UI.

Architecture:
  - CPU profiling: attach.py injects PythonMemoryLayer via ptrace
  - GPU profiling: gpu_attach.py injects cupti_layer.so via ptrace +
    PyRun_SimpleString(ctypes.CDLL(...).cupti_start(...))
    CUPTI runs inside the target's process so it captures that process's
    own CUDA context — the only correct way to profile arbitrary processes.

Usage:
    python run_profiler.py                          # default GPU workload, 5s
    python run_profiler.py --target my_script.py   # any Python script
    python run_profiler.py --duration 10            # longer collection
    python run_profiler.py --no-gpu                 # CPU-only
    python run_profiler.py --output out.json        # custom output path

Prerequisites:
    src/cpp/ring_buffer.so   (python3 src/cpp/build.py)
    src/gpu/cupti_layer.so   (python3 src/gpu/build_gpu.py)
    pip install cffi cupy-cuda12x
"""

import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SRC_PYTHON = ROOT / "src" / "python"
SRC_GPU    = ROOT / "src" / "gpu"
SRC_CPP    = ROOT / "src" / "cpp"
CORRELATOR = ROOT / "src" / "correlator"
WORKLOAD   = ROOT / "tests" / "integration" / "workload_gpu.py"

sys.path.insert(0, str(SRC_PYTHON))
sys.path.insert(0, str(CORRELATOR))

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_DURATION = 5
DEFAULT_OUTPUT   = "session.json"
DEFAULT_CPU_SHM  = "/hpc_runner_cpu"
DEFAULT_GPU_SHM  = "/hpc_runner_gpu"
DEFAULT_CAPACITY = 8 * 1024 * 1024
ATTACH_TIMEOUT   = 6.0
EVENT_TIMEOUT    = 4.0
POLL_INTERVAL    = 0.005


# ── Logging ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def die(msg: str) -> None:
    print(f"\n  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Drain threads ─────────────────────────────────────────────────────────

def _drain_cpu(bridge, sink: list, stop_evt: threading.Event) -> None:
    from bridge import Bridge
    while not stop_evt.is_set():
        payload = bridge.read()
        if payload is None:
            time.sleep(POLL_INTERVAL)
            continue
        try:
            sink.append(Bridge.deserialize(payload))
        except Exception:
            pass
    while True:
        payload = bridge.read()
        if payload is None:
            break
        try:
            sink.append(Bridge.deserialize(payload))
        except Exception:
            pass


def _drain_gpu(gpu_bridge, sink: list, stop_evt: threading.Event) -> None:
    from gpu_bridge import GpuBridge
    while not stop_evt.is_set():
        payload = gpu_bridge.read()
        if payload is None:
            time.sleep(POLL_INTERVAL)
            continue
        try:
            sink.append(GpuBridge.deserialize(payload))
        except Exception:
            pass
    while True:
        payload = gpu_bridge.read()
        if payload is None:
            break
        try:
            sink.append(GpuBridge.deserialize(payload))
        except Exception:
            pass


# ── Main runner ───────────────────────────────────────────────────────────

def run(
    target:   Path,
    duration: float,
    output:   Path,
    use_gpu:  bool,
    cpu_shm:  str,
    gpu_shm:  str,
    capacity: int,
) -> None:

    from bridge     import Bridge
    from attach     import Attacher
    from correlator import Correlator

    cpu_events: list = []
    gpu_events: list = []
    proc             = None
    cpu_bridge       = None
    gpu_bridge       = None
    cpu_attacher     = None
    gpu_attacher     = None
    stop_evt         = threading.Event()

    cupti_so = str(SRC_GPU / "cupti_layer.so")
    rb_so    = str(SRC_CPP / "ring_buffer.so")

    if use_gpu and not Path(cupti_so).exists():
        log(f"WARNING: {cupti_so} not found — running CPU-only.")
        use_gpu = False

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     HPC Memory Profiler — Runner     ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    try:
        # ── 1. CPU ring buffer (runner creates; profiler writes into it) ──────
        log("Opening CPU ring buffer...")
        cpu_bridge = Bridge(shm_name=cpu_shm, capacity=capacity)
        cpu_bridge.open(create=True)
        log(f"CPU bridge ready  ({cpu_shm})")

        # ── 2. Spawn the target workload ──────────────────────────────────────
        # Pass GPU shm name and cupti_layer.so path via env vars so the
        # workload can call cupti_start() inside its own process.
        # This avoids ptrace injection into a CUDA-initialised process,
        # which deadlocks on the CUDA driver mutex.
        child_env = os.environ.copy()
        # CPU profiler env vars — workload self-starts profiler before cupy loads,
        # avoiding GIL contention from CUDA threads during ptrace injection.
        child_env["HPC_CPU_SHM"]    = cpu_shm
        child_env["HPC_RB_SO"]      = rb_so
        child_env["HPC_SRC_PYTHON"] = str(SRC_PYTHON)
        log(f"CPU env vars set for target (shm={cpu_shm})")

        if use_gpu:
            child_env["HPC_GPU_SHM"]  = gpu_shm
            child_env["HPC_CUPTI_SO"] = cupti_so
            log(f"GPU env vars set for target (shm={gpu_shm})")

        log(f"Spawning target: {target}")
        proc = subprocess.Popen(
            [sys.executable, str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
        )

        # Wait for pid= handshake line
        r, _, _ = select.select([proc.stdout], [], [], ATTACH_TIMEOUT)
        if not r:
            die(f"Target did not print PID within {ATTACH_TIMEOUT}s")
        line = proc.stdout.readline().decode().strip()
        if not line.startswith("pid="):
            die(f"Unexpected target output: {line!r}")
        target_pid = int(line.split("=")[1])
        log(f"Target PID: {target_pid}")

        # Drain startup lines. Wait specifically for:
        #   "CPU profiler started" — confirms profiler thread is up
        #   "cupti_start OK"       — confirms CUPTI ring buffer is created
        # cupy takes several seconds to init, so use a generous 30s timeout
        # and only stop early if we've seen both confirmations.
        log("Waiting for target to finish startup...")
        cpu_confirmed  = False
        cupti_confirmed = False
        drain_deadline  = time.time() + 30.0

        while time.time() < drain_deadline:
            r2, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not r2:
                # 1s quiet — check if we have what we need
                if cpu_confirmed and (not use_gpu or cupti_confirmed):
                    break
                if proc.poll() is not None:
                    break
                continue
            info = proc.stdout.readline().decode().strip()
            if not info:
                break
            log(f"  target: {info}")
            if "CPU profiler started" in info:
                cpu_confirmed = True
            if "cupti_start OK" in info:
                cupti_confirmed = True
            if proc.poll() is not None:
                die(f"Target exited during startup (code={proc.returncode}). "
                    "Check stderr for errors.")
            # Stop draining once we have both confirmations
            if cpu_confirmed and (not use_gpu or cupti_confirmed):
                # Give a brief moment for cupti to fully attach to CUDA context
                time.sleep(0.3)
                break

        if not cpu_confirmed:
            log("WARNING: CPU profiler start not confirmed — events may be missing")

        # Verify process is still alive
        if proc.poll() is not None:
            stderr_out = proc.stderr.read().decode()[:500]
            die(f"Target exited before attach (code={proc.returncode}). stderr: {stderr_out}")

        # ── 3. CPU profiler already running in target (self-started via env vars)
        log("CPU profiler self-started in target ✓")

        # ── 4. Open GPU ring buffer consumer ─────────────────────────────────
        # The workload called cupti_start() inside its own process (via the
        # HPC_GPU_SHM / HPC_CUPTI_SO env vars we passed at spawn time).
        # cupti_start creates the shm segment; we open the consumer side here.
        # We rely on the startup drain above having already seen the
        # "cupti_start OK" confirmation line from the workload.
        if use_gpu:
            from gpu_bridge import GpuBridge

            # Check the shm segment exists (cupti_start creates it)
            shm_path = f"/dev/shm{gpu_shm}"
            if not os.path.exists(shm_path):
                log(f"WARNING: GPU shm {shm_path} not found — CUPTI may not have started.")
                log("  Check workload stderr. Is cupti_layer.so on LD_LIBRARY_PATH?")
                use_gpu = False
            else:
                gpu_bridge = GpuBridge(
                    shm_name=gpu_shm,
                    capacity=capacity,
                    rb_so_path=rb_so,
                )
                try:
                    gpu_bridge.open()
                    log(f"GPU ring buffer consumer ready  ({gpu_shm}) ✓")
                except RuntimeError as e:
                    log(f"WARNING: GPU ring buffer open failed: {e}")
                    gpu_bridge = None
                    use_gpu    = False

        # ── 5. Start drain threads ────────────────────────────────────────────
        cpu_thread = threading.Thread(
            target=_drain_cpu,
            args=(cpu_bridge, cpu_events, stop_evt),
            daemon=True, name="drain-cpu",
        )
        cpu_thread.start()

        gpu_thread = None
        if use_gpu and gpu_bridge:
            gpu_thread = threading.Thread(
                target=_drain_gpu,
                args=(gpu_bridge, gpu_events, stop_evt),
                daemon=True, name="drain-gpu",
            )
            gpu_thread.start()

        # ── 6. Wait for first CPU event ───────────────────────────────────────
        log(f"Waiting for first events (timeout={EVENT_TIMEOUT}s)...")
        deadline = time.time() + EVENT_TIMEOUT
        while time.time() < deadline and len(cpu_events) == 0:
            time.sleep(0.05)
        if cpu_events:
            log("First CPU event received ✓")
        else:
            log("WARNING: No CPU events — profiler may not have attached cleanly.")

        # ── 7. Collect for duration ───────────────────────────────────────────
        session_start = time.time_ns()
        log(f"Collecting for {duration}s...")
        bar = 30
        t0  = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration:
                break
            filled = int(elapsed / duration * bar)
            print(
                f"\r  [{'█'*filled}{'░'*(bar-filled)}] "
                f"{elapsed:.1f}s  CPU:{len(cpu_events)}  GPU:{len(gpu_events)}   ",
                end="", flush=True,
            )
            time.sleep(0.1)
        print()
        session_end = time.time_ns()

        # ── 8. Stop drain threads ─────────────────────────────────────────────
        log("Stopping drain threads...")
        stop_evt.set()
        cpu_thread.join(timeout=2.0)
        if gpu_thread:
            gpu_thread.join(timeout=2.0)

        # ── 9. Stop target (workload calls cupti_stop() in its own cleanup) ──
        log("Terminating target...")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log(f"Target exited (code={proc.returncode})")

        # Drain any remaining GPU events flushed by cupti_stop in the workload
        if gpu_bridge:
            time.sleep(0.2)
            from gpu_bridge import GpuBridge
            while True:
                payload = gpu_bridge.read()
                if payload is None:
                    break
                try:
                    gpu_events.append(GpuBridge.deserialize(payload))
                except Exception:
                    pass

    finally:
        stop_evt.set()
        if cpu_bridge:
            cpu_bridge.close(unlink=True)
        if gpu_bridge:
            gpu_bridge.close()

    # ── 11. Report ────────────────────────────────────────────────────────────
    cpu_allocs   = sum(1 for e in cpu_events if not e.get("is_dealloc", False))
    cpu_deallocs = len(cpu_events) - cpu_allocs
    print()
    log(f"CPU events:  {cpu_allocs} allocs + {cpu_deallocs} deallocs = {len(cpu_events)} total")
    log(f"GPU events:  {len(gpu_events)}")

    if len(cpu_events) == 0:
        log("WARNING: zero CPU events — check ptrace permissions (ptrace_scope=0)")

    if len(gpu_events) == 0 and use_gpu:
        log("WARNING: zero GPU events. Possible causes:")
        log("  1. Target has no CUDA operations (use workload_gpu.py with cupy)")
        log("  2. CUPTI injection failed — check stderr for [CUPTI] errors")
        log("  3. ptrace_scope > 0 — run: echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope")

    # ── 12. Correlate ─────────────────────────────────────────────────────────
    log("Running correlator...")
    correlator = Correlator()
    correlator.ingest(cpu_events, gpu_events)
    correlator._session_start_ns = session_start
    correlator._session_end_ns   = session_end
    session = correlator.build_session()

    hard = sum(1 for c in session["correlated_events"] if c["confidence"] == "HARD")
    weak = sum(1 for c in session["correlated_events"] if c["confidence"] == "WEAK")
    log(f"Correlated:  {hard} HARD + {weak} WEAK = {len(session['correlated_events'])} total")

    # ── 13. Write output ──────────────────────────────────────────────────────
    output_path = Path(output)
    with open(output_path, "w") as f:
        json.dump(session, f, indent=2)
    log(f"Session written → {output_path}  ({output_path.stat().st_size/1024:.1f} KB)")

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print(f"  │  Open src/ui/index.html and drag in:             │")
    print(f"  │    {str(output_path):<48s}│")
    print("  └──────────────────────────────────────────────────┘")
    print()


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="HPC Memory Profiler — end-to-end runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--target",   type=Path,  default=WORKLOAD,
                   help="Python script to profile (default: workload_gpu.py)")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"Seconds to collect (default: {DEFAULT_DURATION})")
    p.add_argument("--output",   type=Path,  default=DEFAULT_OUTPUT,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--no-gpu",   action="store_true",
                   help="CPU-only profiling (skip GPU injection)")
    p.add_argument("--cpu-shm",  default=DEFAULT_CPU_SHM)
    p.add_argument("--gpu-shm",  default=DEFAULT_GPU_SHM)
    p.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    args = p.parse_args()

    if not args.target.exists():
        die(f"Target not found: {args.target}")

    run(
        target   = args.target,
        duration = args.duration,
        output   = args.output,
        use_gpu  = not args.no_gpu,
        cpu_shm  = args.cpu_shm,
        gpu_shm  = args.gpu_shm,
        capacity = args.capacity,
    )


if __name__ == "__main__":
    main()