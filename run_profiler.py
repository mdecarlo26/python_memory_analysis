#!/usr/bin/env python3
"""
run_profiler.py
---------------
End-to-end runner: attaches the HPC profiler to a live workload,
collects CPU and GPU events simultaneously, runs the correlator,
and writes a ProfilingSession JSON ready to load in the UI.

Usage:
    # Profile the built-in numpy workload for 5 seconds:
    python run_profiler.py

    # Profile a custom script:
    python run_profiler.py --target path/to/your_script.py --duration 10

    # Custom output path:
    python run_profiler.py --output my_session.json

    # Skip GPU (CPU-only profiling):
    python run_profiler.py --no-gpu

Prerequisites:
    src/cpp/ring_buffer.so   (built by src/cpp/build.py)
    src/gpu/cupti_layer.so   (built by src/gpu/build_gpu.py)
    Python packages: cffi (for bridge.py)

Output:
    session.json  — drag into src/ui/index.html to visualize
"""

import argparse
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SRC_PYTHON = ROOT / "src" / "python"
SRC_GPU    = ROOT / "src" / "gpu"
SRC_CPP    = ROOT / "src" / "cpp"
CORRELATOR = ROOT / "src" / "correlator"
WORKLOAD   = ROOT / "tests" / "integration" / "workload_numpy.py"

sys.path.insert(0, str(SRC_PYTHON))
sys.path.insert(0, str(CORRELATOR))

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_DURATION   = 5       # seconds to collect
DEFAULT_OUTPUT     = "session.json"
DEFAULT_CPU_SHM    = "/hpc_runner_cpu"
DEFAULT_GPU_SHM    = "/hpc_runner_gpu"
DEFAULT_CAPACITY   = 8 * 1024 * 1024   # 8 MiB per ring buffer
ATTACH_TIMEOUT     = 6.0
EVENT_TIMEOUT      = 4.0
POLL_INTERVAL      = 0.005


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# GPU layer loader (cupti_layer.so via ctypes)
# ═══════════════════════════════════════════════════════════════════════════

class CuptiLayer:
    """
    Thin Python wrapper around cupti_layer.so C API:
      int      cupti_start(const char* shm_name)
      void     cupti_stop()
      uint64_t cupti_events_written()
    """

    def __init__(self, so_path: str):
        self._lib     = ctypes.CDLL(so_path)
        self._started = False
        self._shm_name: bytes = b""

        self._lib.cupti_start.restype  = ctypes.c_int
        self._lib.cupti_start.argtypes = [ctypes.c_char_p]

        self._lib.cupti_stop.restype  = None
        self._lib.cupti_stop.argtypes = []

        self._lib.cupti_events_written.restype  = ctypes.c_uint64
        self._lib.cupti_events_written.argtypes = []

    def start(self, shm_name: str) -> bool:
        self._shm_name = shm_name.encode()
        ok = self._lib.cupti_start(self._shm_name)
        self._started = bool(ok)
        return self._started

    def stop(self) -> None:
        if self._started:
            self._lib.cupti_stop()
            self._started = False

    def events_written(self) -> int:
        return int(self._lib.cupti_events_written())


# ═══════════════════════════════════════════════════════════════════════════
# CPU drain thread
# ═══════════════════════════════════════════════════════════════════════════

import threading

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
    # flush remainder
    while True:
        payload = bridge.read()
        if payload is None:
            break
        try:
            from bridge import Bridge
            sink.append(Bridge.deserialize(payload))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# GPU drain thread
# ═══════════════════════════════════════════════════════════════════════════

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
            from gpu_bridge import GpuBridge
            sink.append(GpuBridge.deserialize(payload))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def run(
    target: Path,
    duration: float,
    output: Path,
    use_gpu: bool,
    cpu_shm: str,
    gpu_shm: str,
    capacity: int,
) -> None:

    from bridge import Bridge
    from attach import Attacher
    from correlator import Correlator

    cpu_events: list = []
    gpu_events: list = []
    proc       = None
    cupti      = None
    cpu_bridge = None
    gpu_bridge = None
    stop_evt   = threading.Event()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     HPC Memory Profiler — Runner     ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    try:
        # ── 1. Start CPU ring buffer (we create it; profiler writes to it) ──
        log("Opening CPU ring buffer...")
        cpu_bridge = Bridge(shm_name=cpu_shm, capacity=capacity)
        cpu_bridge.open(create=True)
        log(f"CPU bridge ready  ({cpu_shm})")

        # ── 2. Start GPU layer ──────────────────────────────────────────────
        if use_gpu:
            cupti_so = str(SRC_GPU / "cupti_layer.so")
            if not Path(cupti_so).exists():
                log(f"WARNING: {cupti_so} not found — running CPU-only.")
                use_gpu = False
            else:
                log("Loading cupti_layer.so...")
                cupti = CuptiLayer(cupti_so)
                if not cupti.start(gpu_shm):
                    log("WARNING: cupti_start() returned 0 — GPU profiling disabled.")
                    use_gpu = False
                    cupti = None
                else:
                    log(f"CUPTI profiler running  ({gpu_shm})")

                    # Give cupti_layer a moment to create its ring buffer
                    time.sleep(0.2)

                    from gpu_bridge import GpuBridge
                    gpu_bridge = GpuBridge(shm_name=gpu_shm, capacity=capacity,
                                           rb_so_path=str(SRC_CPP / "ring_buffer.so"))
                    gpu_bridge.open()
                    log("GPU ring buffer consumer ready")

        # ── 3. Spawn target workload ────────────────────────────────────────
        log(f"Spawning target: {target}")
        proc = subprocess.Popen(
            [sys.executable, str(target)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for pid= handshake
        r, _, _ = select.select([proc.stdout], [], [], ATTACH_TIMEOUT)
        if not r:
            die(f"Target did not print PID within {ATTACH_TIMEOUT}s")
        line = proc.stdout.readline().decode().strip()
        if not line.startswith("pid="):
            die(f"Unexpected target output: {line!r}")
        target_pid = int(line.split("=")[1])
        log(f"Target PID: {target_pid}")

        # ── 4. Attach Python profiler ───────────────────────────────────────
        log("Attaching Python memory profiler...")
        attacher = Attacher(pid=target_pid, shm_name=cpu_shm)
        attacher.attach()
        log("Profiler attached ✓")

        # ── 5. Start drain threads ──────────────────────────────────────────
        cpu_thread = threading.Thread(
            target=_drain_cpu,
            args=(cpu_bridge, cpu_events, stop_evt),
            daemon=True,
            name="drain-cpu",
        )
        cpu_thread.start()

        gpu_thread = None
        if use_gpu and gpu_bridge:
            gpu_thread = threading.Thread(
                target=_drain_gpu,
                args=(gpu_bridge, gpu_events, stop_evt),
                daemon=True,
                name="drain-gpu",
            )
            gpu_thread.start()

        # ── 6. Wait for first CPU event ─────────────────────────────────────
        log(f"Waiting for first events (timeout={EVENT_TIMEOUT}s)...")
        deadline = time.time() + EVENT_TIMEOUT
        while time.time() < deadline and len(cpu_events) == 0:
            time.sleep(0.05)
        if len(cpu_events) == 0:
            log("WARNING: No CPU events arrived — profiler may not have attached cleanly.")
        else:
            log(f"First CPU event received ✓")

        # ── 7. Collect for duration ─────────────────────────────────────────
        session_start = time.time_ns()
        log(f"Collecting for {duration}s...")

        bar_width = 30
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration:
                break
            frac  = elapsed / duration
            filled = int(frac * bar_width)
            bar   = "█" * filled + "░" * (bar_width - filled)
            ncpu  = len(cpu_events)
            ngpu  = len(gpu_events)
            print(
                f"\r  [{bar}] {elapsed:.1f}s  "
                f"CPU:{ncpu}  GPU:{ngpu}   ",
                end="", flush=True,
            )
            time.sleep(0.1)
        print()  # newline after progress bar

        session_end = time.time_ns()

        # ── 8. Stop collection ──────────────────────────────────────────────
        log("Stopping drain threads...")
        stop_evt.set()
        cpu_thread.join(timeout=2.0)
        if gpu_thread:
            gpu_thread.join(timeout=2.0)

        # ── 9. Stop GPU profiler ────────────────────────────────────────────
        if cupti:
            log(f"Stopping CUPTI (events written: {cupti.events_written()})...")
            cupti.stop()
            # Drain any remaining GPU events flushed by cupti_stop()
            if gpu_bridge:
                time.sleep(0.1)
                from gpu_bridge import GpuBridge
                while True:
                    payload = gpu_bridge.read()
                    if payload is None:
                        break
                    try:
                        gpu_events.append(GpuBridge.deserialize(payload))
                    except Exception:
                        pass

        # ── 10. Stop target ─────────────────────────────────────────────────
        log("Terminating target workload...")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log(f"Target exited (code={proc.returncode})")

    finally:
        # Clean up ring buffers regardless of errors
        stop_evt.set()
        if cpu_bridge:
            cpu_bridge.close(unlink=True)
        if gpu_bridge:
            gpu_bridge.close()

    # ── 11. Report collection stats ─────────────────────────────────────────
    cpu_allocs   = sum(1 for e in cpu_events if not e.get("is_dealloc", False))
    cpu_deallocs = sum(1 for e in cpu_events if e.get("is_dealloc", False))
    print()
    log(f"Collected  CPU: {cpu_allocs} allocs + {cpu_deallocs} deallocs = {len(cpu_events)} total")
    log(f"Collected  GPU: {len(gpu_events)} events")

    if len(cpu_events) == 0:
        log("WARNING: Zero CPU events — session will be empty. Check profiler attach.")

    # ── 12. Correlate ────────────────────────────────────────────────────────
    log("Running correlator...")
    correlator = Correlator()
    correlator.ingest(cpu_events, gpu_events)
    # Override session timestamps with real wall-clock values
    correlator._session_start_ns = session_start
    correlator._session_end_ns   = session_end
    session = correlator.build_session()

    hard = sum(1 for c in session["correlated_events"] if c["confidence"] == "HARD")
    weak = sum(1 for c in session["correlated_events"] if c["confidence"] == "WEAK")
    log(f"Correlated: {hard} HARD + {weak} WEAK = {len(session['correlated_events'])} total")

    # ── 13. Write output ─────────────────────────────────────────────────────
    output_path = Path(output)
    with open(output_path, "w") as f:
        json.dump(session, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    log(f"Session written → {output_path}  ({size_kb:.1f} KB)")

    print()
    print("  ┌──────────────────────────────────────────┐")
    print(f"  │  Open src/ui/index.html in your browser  │")
    print(f"  │  and drag in:                            │")
    print(f"  │    {str(output_path):<40s}│")
    print("  └──────────────────────────────────────────┘")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPC Memory Profiler — end-to-end runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target", type=Path, default=WORKLOAD,
        help="Python script to profile (default: tests/integration/workload_numpy.py)",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help=f"Seconds to collect (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU profiling (CPU events only)",
    )
    parser.add_argument(
        "--cpu-shm", default=DEFAULT_CPU_SHM,
        help=f"CPU shared memory name (default: {DEFAULT_CPU_SHM})",
    )
    parser.add_argument(
        "--gpu-shm", default=DEFAULT_GPU_SHM,
        help=f"GPU shared memory name (default: {DEFAULT_GPU_SHM})",
    )
    parser.add_argument(
        "--capacity", type=int, default=DEFAULT_CAPACITY,
        help=f"Ring buffer capacity in bytes (default: {DEFAULT_CAPACITY})",
    )
    args = parser.parse_args()

    if not args.target.exists():
        die(f"Target script not found: {args.target}")

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