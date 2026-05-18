"""
build.py
--------
Compiles ring_buffer.cpp into ring_buffer.so.
Run from the project root:

    python3 src/cpp/build.py

Output: src/cpp/ring_buffer.so
"""

import subprocess
import sys
from pathlib import Path

SRC  = Path(__file__).parent / "ring_buffer.cpp"
OUT  = Path(__file__).parent / "ring_buffer.so"

CMD = [
    "g++",
    "-O2",
    "-shared",
    "-fPIC",
    "-std=c++17",
    str(SRC),
    "-lrt",
    "-o", str(OUT),
]

def build():
    print(f"Compiling {SRC.name} -> {OUT.name}")
    result = subprocess.run(CMD, capture_output=True, text=True)
    if result.returncode != 0:
        print("Build FAILED:")
        print(result.stderr)
        sys.exit(1)
    print(f"Done: {OUT}")

if __name__ == "__main__":
    build()