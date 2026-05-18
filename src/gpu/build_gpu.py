"""
build_gpu.py
------------
Compiles src/gpu/cupti_layer.cpp into src/gpu/cupti_layer.so.
Supports both the Ubuntu system CUDA layout (/usr/lib/cuda, /usr/include)
and the upstream NVIDIA toolkit layout (/usr/local/cuda).

Run from project root:
    python3 src/gpu/build_gpu.py
"""

import subprocess
import sys
import glob
from pathlib import Path

ROOT  = Path(__file__).parent.parent.parent
SRC   = ROOT / "src" / "gpu" / "cupti_layer.cpp"
OUT   = ROOT / "src" / "gpu" / "cupti_layer.so"
RB_SO = ROOT / "src" / "cpp" / "ring_buffer.so"


def find_layout():
    candidates = [
        # Ubuntu system package layout (apt install nvidia-cuda-toolkit libcupti-dev)
        {
            "cuda_inc":  Path("/usr/include"),
            "cupti_inc": Path("/usr/include"),
            "cuda_lib":  Path("/usr/lib/x86_64-linux-gnu"),
            "cupti_lib": Path("/usr/lib/x86_64-linux-gnu"),
        },
        # Ubuntu /usr/lib/cuda layout
        {
            "cuda_inc":  Path("/usr/lib/cuda/include"),
            "cupti_inc": Path("/usr/lib/cuda/include"),
            "cuda_lib":  Path("/usr/lib/cuda/lib64"),
            "cupti_lib": Path("/usr/lib/cuda/lib64"),
        },
        # Upstream NVIDIA toolkit layout
        {
            "cuda_inc":  Path("/usr/local/cuda/include"),
            "cupti_inc": Path("/usr/local/cuda/extras/CUPTI/include"),
            "cuda_lib":  Path("/usr/local/cuda/lib64"),
            "cupti_lib": Path("/usr/local/cuda/extras/CUPTI/lib64"),
        },
    ]
    for p in sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        candidates.append({
            "cuda_inc":  Path(p) / "include",
            "cupti_inc": Path(p) / "extras" / "CUPTI" / "include",
            "cuda_lib":  Path(p) / "lib64",
            "cupti_lib": Path(p) / "extras" / "CUPTI" / "lib64",
        })
    for c in candidates:
        if (c["cupti_inc"] / "cupti.h").exists():
            return c
    return None


def build():
    layout = find_layout()
    if not layout:
        print("ERROR: cupti.h not found. Install: sudo apt install libcupti-dev")
        sys.exit(1)

    if not RB_SO.exists():
        print(f"ERROR: ring_buffer.so not found at {RB_SO}")
        print("Run: python3 src/cpp/build.py")
        sys.exit(1)

    cuda_inc  = layout["cuda_inc"]
    cupti_inc = layout["cupti_inc"]
    cuda_lib  = layout["cuda_lib"]
    cupti_lib = layout["cupti_lib"]

    includes  = [f"-I{cuda_inc}"] + ([f"-I{cupti_inc}"] if cupti_inc != cuda_inc else [])
    lib_paths = [f"-L{cuda_lib}"] + ([f"-L{cupti_lib}"] if cupti_lib != cuda_lib else [])
    rpaths    = [f"-Wl,-rpath,{cuda_lib}"] + ([f"-Wl,-rpath,{cupti_lib}"] if cupti_lib != cuda_lib else [])

    cmd = (
        ["g++", "-O2", "-std=c++17", "-shared", "-fPIC"]
        + includes
        + [str(SRC), str(RB_SO)]
        + lib_paths
        + ["-lcuda", "-lcupti"]
        + rpaths
        + ["-o", str(OUT)]
    )

    print(f"cuda_inc:  {cuda_inc}")
    print(f"cupti_inc: {cupti_inc}")
    print(f"Compiling {SRC.name} -> {OUT.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED:"); print(result.stderr); sys.exit(1)
    print(f"OK: {OUT}")


if __name__ == "__main__":
    build()