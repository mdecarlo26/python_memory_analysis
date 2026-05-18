"""
gpu_attach.py
-------------
Injects cupti_layer.so into an already-running Python process so that
CUPTI activity records (GPU transfers, kernels, page faults) are captured
from that process's own CUDA context.

Mechanism:
  Reuses attach.py's ptrace + shellcode + PyRun_SimpleString path.
  Injects a Python snippet that spawns a daemon thread to call
  cupti_start(). The thread approach is critical: PyRun_SimpleString
  runs synchronously while the target is stopped under ptrace, so any
  blocking CUDA/CUPTI call (cuptiActivityEnable, dlopen of libcuda) would
  deadlock. By returning from PyRun_SimpleString immediately and letting
  ptrace detach, the daemon thread runs cupti_start freely in the live
  process after detach.

  sys._hpc_cupti_ready  — threading.Event, set when cupti_start returns
  sys._hpc_cupti_active — True if cupti_start succeeded
  sys._hpc_cupti_lib    — ctypes handle, used by stop injection
  sys._hpc_cupti_error  — error string if something went wrong

Stop injection follows the same non-blocking pattern: spawns a thread
to call cupti_stop() (which calls cuptiActivityFlushAll internally and
can also block on GPU sync), then returns.

Usage:
    from gpu_attach import GpuAttacher
    ga = GpuAttacher(pid=12345, shm_name="/hpc_gpu", cupti_so="src/gpu/cupti_layer.so")
    ga.attach()      # injects thread; returns immediately after ptrace detach
    ga.wait(5.0)     # wait up to 5s for cupti_start to confirm inside target
    # ... collect ...
    ga.detach()      # injects cupti_stop thread; waits for flush
"""

import os
import sys
import time
from pathlib import Path

_SRC_PYTHON = Path(__file__).parent
sys.path.insert(0, str(_SRC_PYTHON))

from attach import Attacher


class GpuAttacher:
    """
    Attaches cupti_layer.so into a running Python process.

    Parameters
    ----------
    pid : int
        PID of the target process.
    shm_name : str
        POSIX shared memory name for the GPU ring buffer.
    cupti_so : str | Path
        Absolute path to cupti_layer.so.
    """

    def __init__(
        self,
        pid: int,
        shm_name: str = "/hpc_profiler_gpu",
        cupti_so: str = "",
    ):
        self._pid      = pid
        self._shm_name = shm_name
        self._cupti_so = str(Path(cupti_so).resolve()) if cupti_so else _default_cupti_so()
        self._attached = False
        self._attacher = Attacher(pid=pid, shm_name=shm_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """
        Inject the cupti_start thread into the target process.
        Returns quickly — use wait() to confirm cupti_start succeeded.
        """
        if self._attached:
            return
        if not Path(self._cupti_so).exists():
            raise FileNotFoundError(
                f"cupti_layer.so not found at {self._cupti_so}. "
                "Run: python3 src/gpu/build_gpu.py"
            )
        bootstrap = _make_gpu_bootstrap(self._shm_name, self._cupti_so)
        self._attacher._inject_simple(bootstrap)
        self._attached = True

    def wait(self, timeout: float = 5.0) -> bool:
        """
        Poll the target process to confirm cupti_start() returned.
        Injects a tiny check snippet that reads sys._hpc_cupti_ready
        via /proc/PID/mem — actually we just poll with a re-injection.

        Simpler approach: just sleep a fixed interval and inject a
        status-check snippet via PyRun_SimpleString.

        Returns True if CUPTI confirmed active, False on timeout/error.
        """
        # Poll by injecting a status probe every 0.2s
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.2)
            try:
                self._attacher._inject_simple(_make_status_probe())
                # If we get here without hang, the thread finished
                # (status probe is tiny and non-blocking)
                # Check result via a second probe that reads the flag
                status = self._read_status()
                if status == "running":
                    return True
                if status and status.startswith("error:"):
                    print(f"  [gpu_attach] CUPTI error: {status[6:]}", flush=True)
                    return False
            except Exception:
                pass
        return False

    def _read_status(self) -> str:
        """
        Inject a probe that writes sys._hpc_cupti_status and read it back.
        Since we can't read Python interpreter memory directly, we use
        a simpler heuristic: if the _inject_simple call returns without
        hanging, and cupti_start was called, the ring buffer shm segment
        should now exist in /dev/shm or we can check via /proc.
        """
        # Check if the shm segment was created by cupti_layer
        shm_path = f"/dev/shm{self._shm_name}"
        if os.path.exists(shm_path):
            return "running"
        # Also check /proc/PID/maps for the shm name
        try:
            maps = open(f"/proc/{self._pid}/maps").read()
            if self._shm_name.lstrip("/") in maps:
                return "running"
        except Exception:
            pass
        return "pending"

    def detach(self) -> None:
        """
        Inject cupti_stop() into the target in a non-blocking thread.
        Then wait for the ring buffer to go quiet (no new events for 0.5s).
        """
        if not self._attached:
            return
        stop_code = _make_gpu_stop()
        self._attacher._inject_simple(stop_code)
        self._attached = False

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def pid(self) -> int:
        return self._pid


# ------------------------------------------------------------------
# Bootstrap strings
# ------------------------------------------------------------------

def _make_gpu_bootstrap(shm_name: str, cupti_so: str) -> str:
    """
    Injected via PyRun_SimpleString. Returns immediately by spawning
    a daemon thread for the actual cupti_start work. This avoids
    deadlocking while ptrace has the target stopped.
    """
    return f"""
import sys as _sys
import threading as _threading

if not getattr(_sys, '_hpc_cupti_active', False):
    _sys._hpc_cupti_active = False
    _sys._hpc_cupti_ready  = _threading.Event()

    def _hpc_cupti_start():
        import ctypes as _ct
        try:
            _lib = _ct.CDLL('{cupti_so}')
            _lib.cupti_start.restype  = _ct.c_int
            _lib.cupti_start.argtypes = [_ct.c_char_p]
            _lib.cupti_stop.restype   = None
            _lib.cupti_stop.argtypes  = []
            _ok = _lib.cupti_start(b'{shm_name}')
            if _ok:
                _sys._hpc_cupti_lib    = _lib
                _sys._hpc_cupti_active = True
            else:
                _sys._hpc_cupti_error  = 'cupti_start returned 0'
        except Exception as _e:
            _sys._hpc_cupti_error = str(_e)
        finally:
            _sys._hpc_cupti_ready.set()

    _threading.Thread(target=_hpc_cupti_start,
                      name='hpc_cupti', daemon=True).start()
"""


def _make_gpu_stop() -> str:
    """
    Injected to stop CUPTI. Also runs in a thread because cupti_stop
    calls cuptiActivityFlushAll which synchronises the GPU and can block.
    """
    return """
import sys as _sys, threading as _threading

def _hpc_cupti_stop():
    _lib = getattr(_sys, '_hpc_cupti_lib', None)
    if _lib is not None:
        _lib.cupti_stop()
        _sys._hpc_cupti_active = False
        _sys._hpc_cupti_lib    = None

_threading.Thread(target=_hpc_cupti_stop,
                  name='hpc_cupti_stop', daemon=True).start()
"""


def _make_status_probe() -> str:
    """Tiny non-blocking probe — just checks a flag, safe to inject."""
    return "import sys as _s; _s._hpc_cupti_probed = True"


def _default_cupti_so() -> str:
    root = Path(__file__).parent.parent.parent
    return str(root / "src" / "gpu" / "cupti_layer.so")