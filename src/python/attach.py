"""
attach.py
---------
Attaches to an already-running Python process and injects the HPC
profiler without requiring any source code changes to the target.

Mechanism (Linux x86_64):
  1. PTRACE_ATTACH to stop the target process
  2. Save all registers
  3. Inject a mmap(rwx) syscall to allocate executable memory in the target
  4. Write shellcode + bootstrap Python string into the rwx region
  5. The shellcode calls PyGILState_Ensure -> PyRun_SimpleString -> PyGILState_Release
  6. PyRun_SimpleString executes the bootstrap which starts PythonMemoryLayer
     and connects it to the shared memory bridge
  7. PTRACE_CONT until INT3 trap (shellcode finished)
  8. Restore registers, PTRACE_DETACH
  9. Target resumes normally with the profiler running in a background thread

Requirements:
  - Linux x86_64
  - Python 3.12 (non-PIE build — symbol addresses are absolute VMAs)
  - CAP_SYS_PTRACE or ptrace_scope=0
  - ring_buffer.so must be compiled (run src/cpp/build.py first)

Usage:
    attacher = Attacher(pid=12345)
    attacher.attach()
    # profiler is now running in the target process
    attacher.detach()   # optional — target keeps profiling after detach
"""

import ctypes
import ctypes.util
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Platform check
# ---------------------------------------------------------------------------

if sys.platform != "linux":
    raise ImportError("attach.py is Linux-only")

if struct.calcsize("P") != 8:
    raise ImportError("attach.py requires x86_64 (64-bit)")


# ---------------------------------------------------------------------------
# ptrace constants and structures
# ---------------------------------------------------------------------------

PTRACE_ATTACH     = 16
PTRACE_DETACH     = 17
PTRACE_GETREGS    = 12
PTRACE_SETREGS    = 13
PTRACE_CONT       = 7
PTRACE_SINGLESTEP = 9

PROT_RWX      = 7    # PROT_READ | PROT_WRITE | PROT_EXEC
MAP_ANON_PRIV = 34   # MAP_ANONYMOUS | MAP_PRIVATE
MMAP_SYSCALL  = 9    # sys_mmap on x86_64


class UserRegs(ctypes.Structure):
    """x86_64 user_regs_struct as defined in sys/user.h"""
    _fields_ = [
        ("r15", ctypes.c_ulong), ("r14", ctypes.c_ulong),
        ("r13", ctypes.c_ulong), ("r12", ctypes.c_ulong),
        ("rbp", ctypes.c_ulong), ("rbx", ctypes.c_ulong),
        ("r11", ctypes.c_ulong), ("r10", ctypes.c_ulong),
        ("r9",  ctypes.c_ulong), ("r8",  ctypes.c_ulong),
        ("rax", ctypes.c_ulong), ("rcx", ctypes.c_ulong),
        ("rdx", ctypes.c_ulong), ("rsi", ctypes.c_ulong),
        ("rdi", ctypes.c_ulong), ("orig_rax", ctypes.c_ulong),
        ("rip", ctypes.c_ulong), ("cs",  ctypes.c_ulong),
        ("eflags", ctypes.c_ulong), ("rsp", ctypes.c_ulong),
        ("ss",  ctypes.c_ulong), ("fs_base", ctypes.c_ulong),
        ("gs_base", ctypes.c_ulong), ("ds", ctypes.c_ulong),
        ("es",  ctypes.c_ulong), ("fs",  ctypes.c_ulong),
        ("gs",  ctypes.c_ulong),
    ]


# ---------------------------------------------------------------------------
# Python symbol resolution — discovers addresses at runtime
# Works for both non-PIE (absolute VMA) and PIE (base + offset) builds
# ---------------------------------------------------------------------------

def _resolve_symbols(python_binary: str) -> dict[str, int]:
    """
    Resolve PyGILState_Ensure, PyGILState_Release, PyRun_SimpleString
    from the given Python binary using nm.

    Returns a dict of {symbol_name: absolute_vma} for non-PIE binaries,
    or {symbol_name: file_offset} for PIE binaries (caller adds load base).

    Raises RuntimeError if any symbol cannot be found.
    """
    import subprocess as _sp

    required = {"PyGILState_Ensure", "PyGILState_Release", "PyRun_SimpleString"}
    found: dict[str, int] = {}

    for flags in [[], ["-D"]]:
        r = _sp.run(["nm"] + flags + [python_binary],
                    capture_output=True, text=True)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            addr_str, sym_type, sym_name = parts[0], parts[1], parts[2]
            if sym_name in required and sym_type.upper() in ("T", "W"):
                try:
                    found[sym_name] = int(addr_str, 16)
                except ValueError:
                    pass
        if found:
            break

    missing = required - set(found)
    if missing:
        raise RuntimeError(
            f"Could not resolve symbols {missing} from {python_binary}. "
            "Ensure nm is installed and the binary has symbols."
        )

    return found


def _get_load_base(pid: int, python_binary: str) -> int:
    """
    For PIE binaries: return the base address the binary was loaded at.
    For non-PIE (EXEC type): return 0 (symbols are already absolute VMAs).
    """
    import subprocess as _sp

    r = _sp.run(["readelf", "-h", python_binary], capture_output=True, text=True)
    is_pie = "DYN" in r.stdout

    if not is_pie:
        return 0

    # PIE: find the load base from /proc/PID/maps
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if python_binary in line and "r--p 00000000" in line:
                return int(line.split("-")[0], 16)

    raise RuntimeError(f"Could not find load base for {python_binary} in /proc/{pid}/maps")


def _resolve_for_pid(pid: int) -> dict[str, int]:
    """
    Resolve symbol addresses as they appear in the address space of pid.
    Handles both PIE and non-PIE Python builds.
    """
    import os as _os
    python_binary = _os.path.realpath(f"/proc/{pid}/exe")
    offsets = _resolve_symbols(python_binary)
    base = _get_load_base(pid, python_binary)
    return {name: base + offset for name, offset in offsets.items()}

# Path to the shared memory bridge .so — used in bootstrap
_SO_PATH = Path(__file__).parent.parent / "cpp" / "ring_buffer.so"

# Shared memory name the bootstrap will open
_SHM_NAME = "/hpc_profiler_cpu"


# ---------------------------------------------------------------------------
# Bootstrap Python code injected into the target process
# ---------------------------------------------------------------------------

def _make_bootstrap(shm_name: str, so_path: str) -> bytes:
    """
    Build the Python source string that will be executed in the target
    via PyRun_SimpleString. Starts PythonMemoryLayer in a background
    thread and connects it to the shared memory bridge.
    """
    code = f"""
import sys
import threading

if not getattr(sys, '_hpc_profiler_active', False):
    sys._hpc_profiler_active = True
    try:
        import sys as _sys
        _sys.path.insert(0, '{Path(__file__).parent}')
        from python_memory_layer import PythonMemoryLayer
        from bridge import Bridge

        _hpc_layer  = PythonMemoryLayer(nframe=16)
        _hpc_bridge = Bridge(shm_name='{shm_name}')
        _hpc_bridge.open(create=False)

        def _hpc_profiler_thread():
            _hpc_layer.start()
            import time
            while getattr(_sys, '_hpc_profiler_active', False):
                events = _hpc_layer.collect()
                for e in events:
                    _hpc_bridge.write(e)
                time.sleep(0.05)
            _hpc_layer.stop()
            _hpc_bridge.close()

        _hpc_thread = threading.Thread(
            target=_hpc_profiler_thread,
            name='hpc_profiler',
            daemon=True,
        )
        _hpc_thread.start()
        _sys._hpc_profiler_thread = _hpc_thread
    except Exception as _e:
        import traceback
        _sys._hpc_inject_error = traceback.format_exc()
"""
    return code.encode("utf-8") + b"\x00"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _libc():
    lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    return lib


def _ptrace(lib, request: int, pid: int, addr: int, data) -> int:
    result = lib.ptrace(request, pid, addr, data)
    err = ctypes.get_errno()
    if result == -1 and err != 0:
        raise OSError(err, f"ptrace({request}, {pid}): {os.strerror(err)}")
    return result


def _get_regs(lib, pid: int) -> UserRegs:
    regs = UserRegs()
    _ptrace(lib, PTRACE_GETREGS, pid, 0, ctypes.byref(regs))
    return regs


def _set_regs(lib, pid: int, regs: UserRegs):
    _ptrace(lib, PTRACE_SETREGS, pid, 0, ctypes.byref(regs))


def _copy_regs(src: UserRegs) -> UserRegs:
    dst = UserRegs()
    ctypes.memmove(ctypes.byref(dst), ctypes.byref(src), ctypes.sizeof(UserRegs))
    return dst


def _mem_write(pid: int, addr: int, data: bytes):
    with open(f"/proc/{pid}/mem", "r+b") as f:
        f.seek(addr)
        f.write(data)


def _find_syscall_in_vdso(pid: int) -> int:
    """Find the address of a 'syscall' (0f 05) instruction in the vdso."""
    vdso_start = vdso_end = None
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if "[vdso]" in line:
                parts = line.split()[0].split("-")
                vdso_start = int(parts[0], 16)
                vdso_end   = int(parts[1], 16)
                break
    if vdso_start is None:
        raise RuntimeError("Could not find vdso in target process maps")

    with open(f"/proc/{pid}/mem", "rb") as f:
        f.seek(vdso_start)
        data = f.read(vdso_end - vdso_start)

    idx = data.find(b"\x0f\x05")
    if idx < 0:
        raise RuntimeError("Could not find syscall instruction in vdso")
    return vdso_start + idx


def _inject_mmap(lib, pid: int, syscall_addr: int, saved_regs: UserRegs) -> int:
    """
    Inject a mmap(0, 4096, RWX, ANON|PRIVATE, -1, 0) syscall into the target.
    Returns the address of the allocated rwx region.
    """
    regs = _copy_regs(saved_regs)
    regs.rax = MMAP_SYSCALL
    regs.rdi = 0
    regs.rsi = 4096
    regs.rdx = PROT_RWX
    regs.r10 = MAP_ANON_PRIV
    regs.r8  = ctypes.c_ulong(-1).value
    regs.r9  = 0
    regs.rip = syscall_addr
    _set_regs(lib, pid, regs)

    _ptrace(lib, PTRACE_SINGLESTEP, pid, 0, 0)
    os.waitpid(pid, 0)

    post_regs = _get_regs(lib, pid)
    rwx_addr = post_regs.rax

    if rwx_addr > 2**63:
        raise RuntimeError(f"mmap syscall failed in target, errno={2**64 - rwx_addr}")

    return rwx_addr


def _make_shellcode(str_addr: int, symbols: dict) -> bytes:
    """
    x86_64 shellcode:
      call PyGILState_Ensure()
      mov rbx, rax              ; save gilstate
      mov rdi, str_addr
      call PyRun_SimpleString(str_addr)
      mov rdi, rbx              ; restore gilstate
      call PyGILState_Release(gilstate)
      int3                      ; trap back to debugger
    """
    def call_abs(addr: int) -> bytes:
        return b"\x48\xb8" + struct.pack("<Q", addr) + b"\xff\xd0"

    def movabs_rdi(addr: int) -> bytes:
        return b"\x48\xbf" + struct.pack("<Q", addr)

    code  = call_abs(symbols["PyGILState_Ensure"])
    code += b"\x48\x89\xc3"                          # mov rbx, rax
    code += movabs_rdi(str_addr)
    code += call_abs(symbols["PyRun_SimpleString"])
    code += b"\x48\x89\xdf"                          # mov rdi, rbx
    code += call_abs(symbols["PyGILState_Release"])
    code += b"\xcc"                                   # int3
    return code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Attacher:
    """
    Attaches to a running Python process and starts the HPC profiler.

    The profiler runs as a daemon thread inside the target process,
    collecting CpuEvents and writing them to the shared memory bridge.
    The C++ correlator reads from the other end of the bridge.
    """

    def __init__(
        self,
        pid: int,
        shm_name: str = _SHM_NAME,
        timeout: float = 5.0,
    ):
        self._pid      = pid
        self._shm_name = shm_name
        self._timeout  = timeout
        self._attached = False
        self._lib      = _libc()

    def attach(self):
        """
        Perform the full attach + inject sequence.
        Blocks until the bootstrap has been injected and the profiler
        thread has started in the target (confirmed via SIGTRAP).
        """
        if self._attached:
            return

        pid = self._pid
        lib = self._lib

        # 1. Attach and stop the target
        _ptrace(lib, PTRACE_ATTACH, pid, 0, 0)
        os.waitpid(pid, 0)

        # 2. Save registers
        saved = _get_regs(lib, pid)

        # If the process was stopped mid-syscall (orig_rax holds the syscall number
        # and is not 0xffffffffffffffff), we must let the syscall complete first
        # before redirecting rip, otherwise the kernel will corrupt our injection.
        if saved.orig_rax != 0xFFFFFFFFFFFFFFFF:
            # Single-step past the syscall exit point
            _ptrace(lib, PTRACE_SINGLESTEP, pid, 0, 0)
            os.waitpid(pid, 0)
            saved = _get_regs(lib, pid)

        try:
            # 3. Resolve Python symbol addresses for this specific target binary
            symbols = _resolve_for_pid(pid)

            # 4. Find syscall instruction in vdso
            syscall_addr = _find_syscall_in_vdso(pid)

            # 5. Inject mmap to get rwx memory
            rwx_addr = _inject_mmap(lib, pid, syscall_addr, saved)

            # 6. Build bootstrap + shellcode
            bootstrap = _make_bootstrap(self._shm_name, str(_SO_PATH))
            str_addr  = rwx_addr + 512
            shellcode = _make_shellcode(str_addr, symbols)

            # 6. Write both into the rwx region
            _mem_write(pid, rwx_addr, shellcode)
            _mem_write(pid, str_addr, bootstrap)

            # 7. Redirect execution to shellcode
            # x86_64 ABI: rsp must be 16-byte aligned BEFORE a call instruction,
            # meaning (rsp - 8) % 16 == 0 at the call site (call pushes 8 bytes).
            # We use a fresh stack region well below the current rsp to avoid
            # clobbering data, and align it correctly.
            inject_regs = _copy_regs(saved)
            inject_regs.rip = rwx_addr
            # Align rsp: pick a safe area, ensure (rsp % 16 == 0) for function entry
            safe_rsp = (saved.rsp - 4096) & ~0xF  # 16-byte aligned, 4KB below original
            inject_regs.rsp = safe_rsp
            _set_regs(lib, pid, inject_regs)

            # 8. Continue until INT3 trap
            _ptrace(lib, PTRACE_CONT, pid, 0, 0)
            _, status = os.waitpid(pid, 0)

            if os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                if sig != signal.SIGTRAP:
                    raise RuntimeError(
                        f"Unexpected signal {signal.Signals(sig).name} "
                        f"during injection into pid {pid}"
                    )

            # 9. Restore original registers
            _set_regs(lib, pid, saved)

        except Exception:
            # Always restore registers and detach on failure
            try:
                _set_regs(lib, pid, saved)
            except Exception:
                pass
            _ptrace(lib, PTRACE_DETACH, pid, 0, 0)
            raise

        # 10. Detach — target resumes with profiler thread running
        _ptrace(lib, PTRACE_DETACH, pid, 0, 0)
        self._attached = True

    def detach(self):
        """
        Signal the profiler thread in the target to stop.
        Does not use ptrace — sets sys._hpc_profiler_active = False
        via a second injection, which the profiler thread polls.
        """
        if not self._attached:
            return
        # The profiler thread polls sys._hpc_profiler_active every 50ms.
        # A second inject sets it to False, causing clean shutdown.
        self._inject_simple("import sys; sys._hpc_profiler_active = False")
        self._attached = False

    def _inject_simple(self, code: str):
        """Inject a short Python expression into the target."""
        pid = self._pid
        lib = self._lib

        _ptrace(lib, PTRACE_ATTACH, pid, 0, 0)
        os.waitpid(pid, 0)
        saved = _get_regs(lib, pid)

        if saved.orig_rax != 0xFFFFFFFFFFFFFFFF:
            _ptrace(lib, PTRACE_SINGLESTEP, pid, 0, 0)
            os.waitpid(pid, 0)
            saved = _get_regs(lib, pid)

        try:
            symbols = _resolve_for_pid(pid)
            syscall_addr = _find_syscall_in_vdso(pid)
            rwx_addr = _inject_mmap(lib, pid, syscall_addr, saved)

            payload = code.encode("utf-8") + b"\x00"
            str_addr = rwx_addr + 512
            shellcode = _make_shellcode(str_addr, symbols)

            _mem_write(pid, rwx_addr, shellcode)
            _mem_write(pid, str_addr, payload)

            inject_regs = _copy_regs(saved)
            inject_regs.rip = rwx_addr
            safe_rsp = (saved.rsp - 4096) & ~0xF
            inject_regs.rsp = safe_rsp
            _set_regs(lib, pid, inject_regs)

            _ptrace(lib, PTRACE_CONT, pid, 0, 0)
            os.waitpid(pid, 0)
            _set_regs(lib, pid, saved)

        finally:
            _ptrace(lib, PTRACE_DETACH, pid, 0, 0)

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def pid(self) -> int:
        return self._pid