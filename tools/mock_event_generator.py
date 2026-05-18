"""
mock_event_generator.py
-----------------------
Generates a fake but schema-valid ProfilingSession JSON file.
Used during Phase 0 and Phase 1/2 development so the correlator
and web UI can be built before real profiler data exists.

Usage:
    python mock_event_generator.py --events 200 --output session.json

Requirements:
    pip install jsonschema
"""

import json
import random
import time
import uuid
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_event_id = 0

def next_id():
    global _event_id
    _event_id += 1
    return _event_id

def base_event(timestamp_ns, event_type, process_id=12345, thread_id=1):
    return {
        "event_id": next_id(),
        "timestamp_ns": timestamp_ns,
        "process_id": process_id,
        "thread_id": thread_id,
        "event_type": event_type
    }

PYTHON_TYPES = [
    "numpy.ndarray",
    "torch.Tensor",
    "list",
    "dict",
    "bytes",
    "bytearray",
]

KERNEL_NAMES = [
    "cudnn_infer_engine",
    "volta_sgemm_128x64_nt",
    "elementwise_kernel",
    "reduce_sum_kernel",
    "",
]

CALLSTACKS = [
    ["<module>", "train", "forward", "numpy.dot"],
    ["<module>", "preprocess", "numpy.array"],
    ["<module>", "train", "backward", "torch.autograd"],
    ["<module>", "load_data", "pandas.read_csv"],
]


# ---------------------------------------------------------------------------
# Event generators
# ---------------------------------------------------------------------------

def make_cpu_alloc(timestamp_ns, address, size, object_type=None):
    return {
        "base": base_event(timestamp_ns, "ALLOC"),
        "alloc_address": address,
        "alloc_size_bytes": size,
        "object_type": object_type or random.choice(PYTHON_TYPES),
        "ref_count": random.randint(1, 5),
        "callstack": random.choice(CALLSTACKS),
        "gc_generation": random.randint(0, 2),
        "is_dealloc": False
    }

def make_cpu_dealloc(timestamp_ns, address, size, object_type):
    return {
        "base": base_event(timestamp_ns, "DEALLOC"),
        "alloc_address": address,
        "alloc_size_bytes": size,
        "object_type": object_type,
        "ref_count": 0,
        "callstack": random.choice(CALLSTACKS),
        "gc_generation": random.randint(0, 2),
        "is_dealloc": True
    }

def make_gpu_transfer(timestamp_ns, src_address, dst_address, size, kind):
    return {
        "base": base_event(timestamp_ns, "TRANSFER"),
        "device_id": 0,
        "src_address": src_address,
        "dst_address": dst_address,
        "transfer_size_bytes": size,
        "transfer_kind": kind,
        "um_page_faults": 0,
        "kernel_name": ""
    }

def make_gpu_kernel(timestamp_ns):
    return {
        "base": base_event(timestamp_ns, "KERNEL"),
        "device_id": 0,
        "src_address": 0,
        "dst_address": 0,
        "transfer_size_bytes": 0,
        "transfer_kind": "HOST_TO_DEVICE",
        "um_page_faults": random.randint(0, 50),
        "kernel_name": random.choice([k for k in KERNEL_NAMES if k])
    }

def make_correlated(cpu_event_id, gpu_event_id, cpu_ts, gpu_ts, hard=True):
    return {
        "cpu_event_id": cpu_event_id,
        "gpu_event_id": gpu_event_id,
        "confidence": "HARD" if hard else "WEAK",
        "match_reason": "ADDRESS_MATCH" if hard else "SIZE_AND_TIMESTAMP",
        "latency_ns": gpu_ts - cpu_ts
    }


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

def generate_session(n_alloc_pairs=50):
    """
    Generates a realistic profiling session:
      - n_alloc_pairs CPU alloc/dealloc pairs
      - A subset of those get a matching GPU transfer (HARD correlation)
      - Additional GPU kernel events (no CPU match — WEAK or no correlation)
    """
    cpu_events = []
    gpu_events = []
    correlated_events = []

    session_start = int(time.time_ns())
    cursor = session_start

    # GPU device memory base address (simulated)
    gpu_base = 0x700000000

    for _ in range(n_alloc_pairs):
        # CPU allocation
        cpu_address = random.randint(0x100000, 0x9FFFFFFF)
        size = random.choice([64, 256, 1024, 4096, 16384, 65536, 1048576])
        obj_type = random.choice(PYTHON_TYPES)
        cursor += random.randint(1_000, 100_000)     # ~1–100 µs between events
        alloc_event = make_cpu_alloc(cursor, cpu_address, size, obj_type)
        cpu_events.append(alloc_event)

        # ~60% of allocations get transferred to GPU (HARD correlation)
        if random.random() < 0.6:
            cursor += random.randint(5_000, 500_000)  # transfer latency
            gpu_dst = gpu_base + random.randint(0, 0x0FFFFFFF)
            transfer = make_gpu_transfer(cursor, cpu_address, gpu_dst, size, "HOST_TO_DEVICE")
            gpu_events.append(transfer)

            corr = make_correlated(
                alloc_event["base"]["event_id"],
                transfer["base"]["event_id"],
                alloc_event["base"]["timestamp_ns"],
                transfer["base"]["timestamp_ns"],
                hard=True
            )
            correlated_events.append(corr)

            # Optional: kernel fires after transfer
            if random.random() < 0.5:
                cursor += random.randint(10_000, 200_000)
                gpu_events.append(make_gpu_kernel(cursor))

        # CPU dealloc (later)
        cursor += random.randint(100_000, 10_000_000)
        dealloc_event = make_cpu_dealloc(cursor, cpu_address, size, obj_type)
        cpu_events.append(dealloc_event)

    # A few extra GPU events with no CPU match (WEAK / unmatched)
    for _ in range(n_alloc_pairs // 5):
        cursor += random.randint(50_000, 500_000)
        gpu_events.append(make_gpu_kernel(cursor))

    session_end = cursor + random.randint(1_000_000, 10_000_000)

    return {
        "session_id": str(uuid.uuid4()),
        "schema_version": "0.1",
        "start_time_ns": session_start,
        "end_time_ns": session_end,
        "cpu_events": cpu_events,
        "gpu_events": gpu_events,
        "correlated_events": correlated_events
    }


# ---------------------------------------------------------------------------
# Validation (optional — requires jsonschema)
# ---------------------------------------------------------------------------

def validate(session, schema_path):
    try:
        import jsonschema
    except ImportError:
        print("jsonschema not installed — skipping validation. Run: pip install jsonschema")
        return True

    with open(schema_path) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=session, schema=schema)
        print("Schema validation passed.")
        return True
    except jsonschema.ValidationError as e:
        print(f"Schema validation FAILED: {e.message}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a mock HPC profiler session JSON.")
    parser.add_argument("--events", type=int, default=50, help="Number of CPU alloc/dealloc pairs to generate.")
    parser.add_argument("--output", type=str, default="mock_session.json", help="Output file path.")
    parser.add_argument("--schema", type=str, default="../hpc_profiler_schema.json", help="Path to schema file for validation.")
    args = parser.parse_args()

    print(f"Generating session with {args.events} allocation pairs...")
    session = generate_session(n_alloc_pairs=args.events)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(session, f, indent=2)

    cpu_count = len(session["cpu_events"])
    gpu_count = len(session["gpu_events"])
    corr_count = len(session["correlated_events"])
    print(f"Generated: {cpu_count} CPU events, {gpu_count} GPU events, {corr_count} correlated pairs.")
    print(f"Output written to: {output_path}")

    schema_path = Path(args.schema)
    if schema_path.exists():
        validate(session, schema_path)
    else:
        print(f"Schema file not found at {schema_path} — skipping validation.")