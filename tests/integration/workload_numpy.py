"""
workload_numpy.py
-----------------
Target workload for the integration test. Runs a numpy matrix multiply
in a loop, allocating and deallocating arrays continuously.

Prints its PID on stdout immediately so the test harness knows it is
ready to be attached to. Runs until stdin is closed or SIGTERM.

Usage:
    python3 workload_numpy.py
"""

import os
import sys
import signal
import time
import numpy as np
import numpy.random  # force lazy submodule import before signaling ready

# Signal readiness AFTER all imports complete — this is the attach-safe point.
# numpy.random is lazily imported on first use; forcing it here prevents a
# race condition where the profiler thread triggers the import concurrently.
print(f"pid={os.getpid()}", flush=True)

_running = True

def _stop(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGTERM, _stop)

N = 256  # matrix dimension — large enough to produce meaningful allocations

iteration = 0
while _running:
    # Each iteration allocates several numpy arrays
    a = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N, N).astype(np.float32)
    c = np.dot(a, b)
    result = c.sum()

    # Mix in some Python-native allocations too
    data = [float(i) for i in range(100)]
    total = sum(data)

    iteration += 1
    time.sleep(0.02)  # ~50 iterations/sec — enough events without flooding

print(f"iterations={iteration}", flush=True)