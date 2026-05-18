"""
conftest.py
-----------
Adds src/python to sys.path so all tests can import profiler modules
without needing per-file sys.path hacks.

Run all tests from project root:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))