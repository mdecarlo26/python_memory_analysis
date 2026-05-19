"""
mock_event_generator.py
-----------------------
Generates synthetic profiling sessions for testing and UI development.
Produces schema-valid CpuEvent, GpuEvent, and CorrelatedEvent dicts.
"""

import argparse
import json
import random
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PYTHON_TYPES = [
    "numpy.ndarray", "list", "dict", "tuple", "torch.Tensor",
    "pandas.DataFrame", "bytes", "set", "str", "float",
]

KERNEL_NAMES = [
    "void gemm_kernel<float>",
    "elementwise_add",
    "batch_norm_forward",
    "conv2d_winograd",
    "softmax_kernel",
]

CALLSTACKS = [
    ["train.py:42", "model.py:118", "torch/nn/modules/linear.py:87"],
    ["inference.py:15", "numpy/core/fromnumeric.py:86"],
    ["data_loader.py:201", "numpy/lib/stride_tricks.py:119"],
    [],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_event_id_counter = 0

def _next_id():
    global _event_id_counter
    _event_id_counter += 1
    return _event_id_counter


def base_event(timestamp_ns, event_type):
    return {
        "event_id": _next_id(),
        "timestamp_ns": timestamp_ns,
        "process_id": 12345,
        "thread_id": 67890,
        "event_type": event_type,
    }


def make_cpu_alloc(timestamp_ns, address, size, object_type=None):
    otype = object_type or random.choice(PYTHON_TYPES)
    is_numpy = "ndarray" in otype or "Tensor" in otype
    return {
        "base": base_event(timestamp_ns, "ALLOC"),
        "alloc_address": address,
        "alloc_size_bytes": size,
        "object_type": otype,
        "module_name": "numpy" if "ndarray" in otype else ("torch" if "Tensor" in otype else "builtins"),
        "ref_count": random.randint(1, 5),
        "callstack": random.choice(CALLSTACKS),
        "gc_generation": random.randint(0, 2),
        "is_dealloc": False,
        "peak_rss_kb": random.randint(50000, 200000),
        "lifetime_ns": 0,
        "is_numpy_buffer": is_numpy,
        "buffer_nbytes": size if is_numpy else 0,
        "parent_address": 0,
    }


def make_cpu_dealloc(timestamp_ns, address, size, object_type, alloc_ts=None):
    lifetime = (timestamp_ns - alloc_ts) if alloc_ts else random.randint(1_000_000, 500_000_000)
    return {
        "base": base_event(timestamp_ns, "DEALLOC"),
        "alloc_address": address,
        "alloc_size_bytes": size,
        "object_type": object_type,
        "module_name": "",
        "ref_count": 0,
        "callstack": [],
        "gc_generation": 0,
        "is_dealloc": True,
        "peak_rss_kb": 0,
        "lifetime_ns": lifetime,
        "is_numpy_buffer": False,
        "buffer_nbytes": 0,
        "parent_address": 0,
    }


def make_gpu_transfer(timestamp_ns, src_address, dst_address, size, kind, duration_ns=None):
    return {
        "base": base_event(timestamp_ns, "TRANSFER"),
        "device_id": 0,
        "src_address": src_address,
        "dst_address": dst_address,
        "transfer_size_bytes": size,
        "transfer_kind": kind,
        "um_page_faults": 0,
        "kernel_name": "",
        "kernel_duration_ns": duration_ns if duration_ns is not None else random.randint(10_000, 5_000_000),
        "stream_id": random.randint(0, 4),
        "device_mem_used_mb": random.randint(100, 8000),
    }


def make_gpu_kernel(timestamp_ns):
    return {
        "base": base_event(timestamp_ns, "KERNEL"),
        "device_id": 0,
        "src_address": 0,
        "dst_address": 0,
        "transfer_size_bytes": 0,
        "transfer_kind": "DEVICE_TO_DEVICE",
        "um_page_faults": random.randint(0, 50),
        "kernel_name": random.choice(KERNEL_NAMES),
        "kernel_duration_ns": random.randint(50_000, 20_000_000),
        "stream_id": random.randint(0, 4),
        "device_mem_used_mb": random.randint(100, 8000),
    }


def make_correlated(cpu_event_id, gpu_event_id, cpu_ts, gpu_ts, hard=True):
    return {
        "cpu_event_id": cpu_event_id,
        "gpu_event_id": gpu_event_id,
        "confidence": "HARD" if hard else "WEAK",
        "match_reason": "ADDRESS_MATCH" if hard else "SIZE_AND_TIMESTAMP",
        "latency_ns": max(0, gpu_ts - cpu_ts),
    }


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

def generate_session(n_alloc_pairs=50):
    """
    Generates a realistic profiling session:
      - n_alloc_pairs CPU alloc/dealloc pairs
      - ~60% get a matching GPU transfer (HARD correlation)
      - Additional GPU kernel events (WEAK or no correlation)
    """
    global _event_id_counter
    _event_id_counter = 0

    cpu_events = []
    gpu_events = []
    correlated_events = []

    session_start = int(time.time_ns())
    cursor = session_start
    gpu_base = 0x700000000

    for _ in range(n_alloc_pairs):
        cpu_address = random.randint(0x100000, 0x9FFFFFFF)
        size = random.choice([64, 256, 1024, 4096, 16384, 65536, 1048576])
        obj_type = random.choice(PYTHON_TYPES)
        cursor += random.randint(1_000, 100_000)
        alloc_ts = cursor
        alloc_event = make_cpu_alloc(cursor, cpu_address, size, obj_type)
        cpu_events.append(alloc_event)

        if random.random() < 0.6:
            cursor += random.randint(5_000, 500_000)
            gpu_dst = gpu_base + random.randint(0, 0x0FFFFFFF)
            transfer = make_gpu_transfer(cursor, cpu_address, gpu_dst, size, "HOST_TO_DEVICE")
            gpu_events.append(transfer)

            corr = make_correlated(
                alloc_event["base"]["event_id"],
                transfer["base"]["event_id"],
                alloc_event["base"]["timestamp_ns"],
                transfer["base"]["timestamp_ns"],
                hard=True,
            )
            correlated_events.append(corr)

            if random.random() < 0.5:
                cursor += random.randint(10_000, 200_000)
                gpu_events.append(make_gpu_kernel(cursor))

        cursor += random.randint(100_000, 10_000_000)
        dealloc_event = make_cpu_dealloc(cursor, cpu_address, size, obj_type, alloc_ts)
        cpu_events.append(dealloc_event)

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
        "correlated_events": correlated_events,
    }


# ---------------------------------------------------------------------------
# Validation
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
    parser.add_argument("--events", type=int, default=50)
    parser.add_argument("--output", type=str, default="mock_session.json")
    parser.add_argument("--schema", type=str, default="../hpc_profiler_schema.json")
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
    print(f"Output: {output_path}")

    schema_path = Path(args.schema)
    if schema_path.exists():
        validate(session, schema_path)
    else:
        print(f"Schema not found at {schema_path} — skipping validation.")