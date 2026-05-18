#!/usr/bin/env python3
"""
test_gpu_attach.py
------------------
Diagnostic test suite for the GPU attach injection mechanism.
Run from project root: python tests/test_gpu_attach.py

Tests are ordered from lowest-level to highest-level so the first
failure pinpoints the exact broken layer.
"""

import os
import select
import subprocess
import sys
import time
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))

CUPTI_SO = str(ROOT / "src" / "gpu" / "cupti_layer.so")
GPU_SHM  = "/hpc_test_gpu_attach"


def _spawn_simple(test_case, extra_code="", timeout=10.0):
    """
    Spawn a minimal Python process that prints pid= then loops.
    Uses a temp file to avoid -c multiline parsing issues.
    extra_code runs BEFORE printing pid= (for setup like cupy init).
    """
    import tempfile
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = (
        "import os, time\n"
        + (extra_code + "\n" if extra_code else "")
        + "print(f'pid={os.getpid()}', flush=True)\n"
        + "import time as _t\n"
        + "for _ in range(400): _t.sleep(0.05)\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        script = f.name

    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    r, _, _ = select.select([proc.stdout], [], [], timeout)
    if not r:
        proc.kill()
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
        os.unlink(script)
        test_case.fail(f"Target never printed pid= within {timeout}s")
    line = proc.stdout.readline().decode().strip()
    os.unlink(script)
    if not line.startswith("pid="):
        proc.kill()
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
        test_case.fail(f"Expected pid=<N>, got: {line!r}")
    return proc, int(line.split("=")[1])


# ---------------------------------------------------------------------------
# T0 — environment sanity (no ptrace at all)
# ---------------------------------------------------------------------------

class T0_Environment(unittest.TestCase):
    """Verify basic subprocess + Python symbol resolution works."""

    def test_subprocess_spawn(self):
        """A subprocess that prints pid= must be readable within 3s."""
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import os,sys; print(f'pid={os.getpid()}',flush=True); sys.stdout.flush()"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            r, _, _ = select.select([proc.stdout], [], [], 5.0)
            self.assertTrue(r, "No output within 5s — subprocess spawn broken")
            line = proc.stdout.readline().decode().strip()
            self.assertTrue(line.startswith("pid="), f"Got: {line!r}")
            pid = int(line.split("=")[1])
            self.assertGreater(pid, 0)
            print(f"  T0: subprocess spawn OK (pid={pid})")
        finally:
            proc.terminate()
            proc.wait(timeout=3)

    def test_symbol_resolution(self):
        """Resolve PyGILState_Ensure from the current Python binary."""
        python_bin = os.path.realpath(f"/proc/{os.getpid()}/exe")
        print(f"\n  T0: Python binary: {python_bin}")

        from attach import _resolve_symbols, _get_load_base, _resolve_for_pid

        offsets = _resolve_symbols(python_bin)
        print(f"  T0: nm offsets: { {k: hex(v) for k,v in offsets.items()} }")

        base = _get_load_base(os.getpid(), python_bin)
        print(f"  T0: load base: {hex(base)}")

        resolved = _resolve_for_pid(os.getpid())
        print(f"  T0: resolved VMAs: { {k: hex(v) for k,v in resolved.items()} }")

        # Verify the resolved address actually matches where the symbol is
        import ctypes
        real_ensure = ctypes.cast(
            ctypes.pythonapi.PyGILState_Ensure,
            ctypes.c_void_p
        ).value
        print(f"  T0: ctypes PyGILState_Ensure: {hex(real_ensure)}")

        resolved_ensure = resolved.get("PyGILState_Ensure", 0)
        diff = abs(resolved_ensure - real_ensure)
        print(f"  T0: diff between resolved and ctypes: {diff} bytes")

        self.assertLess(diff, 0x1000,
            f"Symbol resolution wrong by {diff} bytes — PIE base mismatch?\n"
            f"  resolved={hex(resolved_ensure)}\n"
            f"  ctypes  ={hex(real_ensure)}")
        print(f"  T0: symbol resolution correct ✓")

    def test_ptrace_scope(self):
        """Check ptrace_scope — must be 0 or 1 for same-user ptrace."""
        scope_path = "/proc/sys/kernel/yama/ptrace_scope"
        if not os.path.exists(scope_path):
            print("  T0: ptrace_scope not found (no Yama LSM) — OK")
            return
        scope = int(open(scope_path).read().strip())
        print(f"  T0: ptrace_scope = {scope}")
        self.assertLessEqual(scope, 1,
            f"ptrace_scope={scope} blocks injection. Run:\n"
            f"  echo 0 | sudo tee {scope_path}")


# ---------------------------------------------------------------------------
# T1 — _inject_simple with trivial code
# ---------------------------------------------------------------------------

class T1_TrivialInject(unittest.TestCase):

    def _spawn(self, timeout=10.0):
        """Spawn a minimal target process and return (proc, pid)."""
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Write target code to a temp file to avoid -c multiline issues
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write(
                "import os, time\n"
                "print(f'pid={os.getpid()}', flush=True)\n"
                "import time as _t\n"
                "for _ in range(400): _t.sleep(0.05)\n"
            )
            script = f.name
        self._tmpfiles = getattr(self, '_tmpfiles', [])
        self._tmpfiles.append(script)

        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        r, _, _ = select.select([proc.stdout], [], [], timeout)
        if not r:
            proc.kill()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
            self.fail("Target never printed pid= within timeout")
        line = proc.stdout.readline().decode().strip()
        self.assertTrue(line.startswith("pid="), f"Unexpected output: {line!r}")
        return proc, int(line.split("=")[1])

    def tearDown(self):
        for f in getattr(self, '_tmpfiles', []):
            try: os.unlink(f)
            except: pass

    def test_trivial_inject(self):
        """_inject_simple with a no-import assignment must complete < 5s."""
        from attach import Attacher
        proc, pid = self._spawn()
        try:
            a = Attacher(pid=pid)
            t0 = time.perf_counter()
            a._inject_simple("import sys; sys._t1_done = True")
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 5.0,
                f"_inject_simple took {elapsed:.2f}s — hung on waitpid. "
                "Symbol resolution is likely wrong (see T0).")
            print(f"  T1: trivial inject in {elapsed*1000:.0f}ms ✓")
        finally:
            proc.terminate(); proc.wait(timeout=3)

    def test_inject_thread(self):
        """Inject a threading.Thread bootstrap — must return < 5s."""
        from attach import Attacher
        proc, pid = self._spawn()
        try:
            a = Attacher(pid=pid)
            t0 = time.perf_counter()
            a._inject_simple(
                "import threading,sys\n"
                "threading.Thread(target=lambda:None,daemon=True).start()\n"
            )
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 5.0,
                f"Thread inject took {elapsed:.2f}s — hung")
            print(f"  T1: thread inject in {elapsed*1000:.0f}ms ✓")
        finally:
            proc.terminate(); proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# T2 — GpuAttacher bootstrap (no GPU required)
# ---------------------------------------------------------------------------

class T2_GpuAttacher(unittest.TestCase):

    def _spawn(self):
        return _spawn_simple(self)

    def test_nonexistent_so_raises(self):
        from gpu_attach import GpuAttacher
        proc, pid = self._spawn()
        try:
            ga = GpuAttacher(pid=pid, cupti_so="/nonexistent.so")
            with self.assertRaises(FileNotFoundError):
                ga.attach()
            print("  T2: FileNotFoundError on missing .so ✓")
        finally:
            proc.terminate(); proc.wait(timeout=3)

    def test_thread_bootstrap_returns_quickly(self):
        """Thread-based bootstrap should return well under 5s."""
        from gpu_attach import GpuAttacher
        import tempfile, stat

        # Write a minimal valid-looking .so (exists on disk but dlopen will fail)
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8)
            fake_so = f.name

        proc, pid = self._spawn()
        try:
            ga = GpuAttacher(pid=pid, shm_name="/hpc_t2", cupti_so=fake_so)
            t0 = time.perf_counter()
            ga.attach()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 5.0,
                f"GpuAttacher.attach() took {elapsed:.2f}s — hung")
            print(f"  T2: GpuAttacher.attach() in {elapsed*1000:.0f}ms ✓")
        finally:
            proc.terminate(); proc.wait(timeout=3)
            os.unlink(fake_so)


# ---------------------------------------------------------------------------
# T3 — Full cupti attach (GPU required)
# ---------------------------------------------------------------------------

class T3_FullCupti(unittest.TestCase):

    def setUp(self):
        if not Path(CUPTI_SO).exists():
            self.skipTest(f"cupti_layer.so not found at {CUPTI_SO}")

    def _spawn_idle(self):
        return _spawn_simple(self)

    def _spawn_cupy(self):
        return _spawn_simple(self,
            extra_code=(
                "import cupy as cp\n"
                "_ = cp.zeros(1); cp.cuda.Stream.null.synchronize()"
            ),
            timeout=30.0,
        )

    def test_attach_idle_process(self):
        """Inject cupti into an idle (no CUDA) process."""
        from gpu_attach import GpuAttacher

        shm_path = f"/dev/shm{GPU_SHM}"
        if os.path.exists(shm_path):
            os.unlink(shm_path)

        proc, pid = self._spawn_idle()
        try:
            ga = GpuAttacher(pid=pid, shm_name=GPU_SHM, cupti_so=CUPTI_SO)
            t0 = time.perf_counter()
            ga.attach()
            attach_t = time.perf_counter() - t0
            self.assertLess(attach_t, 5.0, f"attach() hung for {attach_t:.2f}s")
            print(f"  T3: attach() in {attach_t*1000:.0f}ms")

            # cupti_start may succeed or fail (no CUDA context) — either is fine
            confirmed = ga.wait(timeout=5.0)
            print(f"  T3: CUPTI confirmed={confirmed}")

            ga.detach()
        finally:
            proc.terminate(); proc.wait(timeout=3)
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_attach_cupy_process(self):
        """Inject cupti into a process that already has a CUDA context."""
        try:
            import cupy  # noqa
        except ImportError:
            self.skipTest("cupy not installed")

        from gpu_attach import GpuAttacher

        shm_path = f"/dev/shm{GPU_SHM}"
        if os.path.exists(shm_path):
            os.unlink(shm_path)

        proc, pid = self._spawn_cupy()
        try:
            ga = GpuAttacher(pid=pid, shm_name=GPU_SHM, cupti_so=CUPTI_SO)
            t0 = time.perf_counter()
            ga.attach()
            attach_t = time.perf_counter() - t0
            self.assertLess(attach_t, 5.0, f"attach() into cupy hung for {attach_t:.2f}s")
            print(f"  T3: attach() into cupy in {attach_t*1000:.0f}ms")

            confirmed = ga.wait(timeout=8.0)
            self.assertTrue(confirmed, "CUPTI shm not created after cupy attach")
            print(f"  T3: CUPTI confirmed in cupy process ✓")

            ga.detach()
        finally:
            proc.terminate(); proc.wait(timeout=3)
            if os.path.exists(shm_path):
                os.unlink(shm_path)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nRunning gpu_attach diagnostics (runner PID={os.getpid()})")
    print(f"Python:    {sys.executable}")
    print(f"cupti_so:  {'FOUND' if Path(CUPTI_SO).exists() else 'NOT FOUND (T3 skipped)'}")
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [T0_Environment, T1_TrivialInject, T2_GpuAttacher, T3_FullCupti]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)