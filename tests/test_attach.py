"""
test_attach.py
--------------
Pass criteria:
  [A] Attacher can attach to a running Python process without crashing it
  [B] Bootstrap code executes in the target (sys._hpc_profiler_active = True)
  [C] Profiler thread starts in target and events flow within 500ms of attach
  [D] Target process continues running normally after attach
  [E] Attacher handles invalid PID gracefully
"""

import os
import select
import signal
import subprocess
import sys
import time
import unittest
import uuid

from attach import Attacher
from bridge import Bridge


def unique_shm():
    return f"/hpc_test_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# A — Attach without crashing target
# ---------------------------------------------------------------------------

class TestAttach(unittest.TestCase):

    def test_attach_does_not_crash_target(self):
        """Target process must still be alive after attach."""
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import time, os
print(f"pid={os.getpid()}", flush=True)
for i in range(100):
    time.sleep(0.1)
print("done", flush=True)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        a = Attacher(pid=pid)
        a.attach()
        self.assertTrue(a.is_attached)

        # Target should still be running
        time.sleep(0.2)
        self.assertIsNone(proc.poll(), "Target process died after attach")

        proc.terminate()
        proc.wait()

    def test_attach_sets_is_attached(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(10)"],
            stdout=subprocess.PIPE,
        )
        proc.stdout.readline()  # wait for ready signal
        pid = proc.pid

        a = Attacher(pid=pid)
        self.assertFalse(a.is_attached)
        a.attach()
        self.assertTrue(a.is_attached)

        proc.terminate()
        proc.wait()

    def test_double_attach_is_safe(self):
        """Calling attach() twice must not raise or double-inject."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(10)"],
            stdout=subprocess.PIPE,
        )
        proc.stdout.readline()
        pid = proc.pid

        a = Attacher(pid=pid)
        a.attach()
        a.attach()  # should be a no-op
        self.assertTrue(a.is_attached)

        proc.terminate()
        proc.wait()


# ---------------------------------------------------------------------------
# B — Bootstrap executes in target
# ---------------------------------------------------------------------------

class TestBootstrapExecution(unittest.TestCase):

    def test_bootstrap_sets_flag_in_target(self):
        """
        After attach, sys._hpc_profiler_active must be True in the target.
        We verify by having the target print its value after a delay.
        """
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import sys, time, os
print(f"pid={os.getpid()}", flush=True)
time.sleep(1.0)
val = getattr(sys, '_hpc_profiler_active', False)
print(f"active={val}", flush=True)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        a = Attacher(pid=pid)
        a.attach()

        # Wait for target to print the flag value
        r, _, _ = select.select([proc.stdout], [], [], 3.0)
        self.assertTrue(r, "Target did not produce output in time")
        line = proc.stdout.readline().decode().strip()
        self.assertEqual(line, "active=True",
            f"Expected 'active=True', got '{line}'")

        proc.wait()

    def test_profiler_thread_starts_in_target(self):
        """After attach, target must have a thread named 'hpc_profiler'."""
        shm = unique_shm()

        # Bridge must exist before attaching so the bootstrap can open it
        bridge = Bridge(shm_name=shm, capacity=4 * 1024 * 1024)
        bridge.open(create=True)

        proc = subprocess.Popen(
            [sys.executable, "-c", """
import sys, time, os, threading
print(f"pid={os.getpid()}", flush=True)
time.sleep(1.5)
names = [t.name for t in threading.enumerate()]
has_profiler = 'hpc_profiler' in names
print(f"has_thread={has_profiler}", flush=True)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        a = Attacher(pid=pid, shm_name=shm)
        a.attach()

        r, _, _ = select.select([proc.stdout], [], [], 4.0)
        bridge.close(unlink=True)
        self.assertTrue(r, "Target did not produce output")
        line = proc.stdout.readline().decode().strip()
        self.assertEqual(line, "has_thread=True",
            f"Expected 'has_thread=True', got '{line}'")

        proc.wait()


# ---------------------------------------------------------------------------
# C — Events flow within 500ms of attach
# ---------------------------------------------------------------------------

class TestEventFlow(unittest.TestCase):

    def test_events_arrive_within_500ms(self):
        """
        After attach, CpuEvents must appear in the shared memory bridge
        within 500ms.
        """
        shm = unique_shm()

        # Start the bridge on the reader side (create=True here so target can open it)
        reader = Bridge(shm_name=shm, capacity=4 * 1024 * 1024)
        reader.open(create=True)

        proc = subprocess.Popen(
            [sys.executable, "-c", f"""
import sys, time, os
print(f"pid={{os.getpid()}}", flush=True)
# Keep allocating so the profiler has something to capture
data = []
for i in range(500):
    data.append(list(range(100)))
    time.sleep(0.01)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        # Override shm name in attacher
        a = Attacher(pid=pid, shm_name=shm)
        attach_time = time.time()
        a.attach()

        # Poll the bridge for events, with 500ms timeout
        deadline = attach_time + 0.5
        events_received = 0
        while time.time() < deadline:
            payload = reader.read()
            if payload:
                events_received += 1
                break
            time.sleep(0.01)

        reader.close(unlink=True)
        proc.terminate()
        proc.wait()

        elapsed = time.time() - attach_time
        self.assertGreater(events_received, 0,
            f"No events received within 500ms of attach (elapsed={elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# D — Target continues running normally after attach
# ---------------------------------------------------------------------------

class TestTargetContinuity(unittest.TestCase):

    def test_target_output_continues_after_attach(self):
        """Target must continue producing output after being attached to."""
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import time, os
print(f"pid={os.getpid()}", flush=True)
for i in range(20):
    time.sleep(0.1)
    print(f"tick={i}", flush=True)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        # Let a few ticks happen, then attach
        time.sleep(0.3)
        a = Attacher(pid=pid)
        a.attach()

        # Read ticks after attach — should keep coming
        ticks_after = []
        deadline = time.time() + 1.0
        while time.time() < deadline:
            r, _, _ = select.select([proc.stdout], [], [], 0.2)
            if r:
                line = proc.stdout.readline().decode().strip()
                if line.startswith("tick="):
                    ticks_after.append(int(line.split("=")[1]))

        proc.wait()

        self.assertGreater(len(ticks_after), 0,
            "Target produced no output after attach")
        # Ticks should be sequential (no corruption)
        for i in range(1, len(ticks_after)):
            self.assertEqual(ticks_after[i], ticks_after[i-1] + 1,
                f"Tick sequence broken: {ticks_after[i-1]} -> {ticks_after[i]}")

    def test_target_exit_code_normal(self):
        """Target must exit normally (code 0) after being attached to and completing."""
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import time, os
print(f"pid={os.getpid()}", flush=True)
time.sleep(0.5)
print("done", flush=True)
"""],
            stdout=subprocess.PIPE,
        )
        pid = int(proc.stdout.readline().decode().split("=")[1])

        a = Attacher(pid=pid)
        a.attach()

        r, _, _ = select.select([proc.stdout], [], [], 3.0)
        self.assertTrue(r)
        line = proc.stdout.readline().decode().strip()
        self.assertEqual(line, "done")

        proc.wait()
        self.assertEqual(proc.returncode, 0,
            f"Target exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# E — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):

    def test_invalid_pid_raises(self):
        """Attaching to a non-existent PID must raise OSError."""
        a = Attacher(pid=999999)
        with self.assertRaises(OSError):
            a.attach()

    def test_attacher_pid_property(self):
        a = Attacher(pid=12345)
        self.assertEqual(a.pid, 12345)

    def test_detach_before_attach_is_safe(self):
        """detach() before attach() must not raise."""
        a = Attacher(pid=12345)
        a.detach()  # should be a no-op


if __name__ == "__main__":
    unittest.main(verbosity=2)