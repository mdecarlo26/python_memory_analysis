#!/usr/bin/env python3
"""
Merge two JSONL trace files (CPU + GPU) into a single time-ordered JSONL.

Usage:
    python merge_jsonl_traces.py \
        --cpu events_host.jsonl \
        --gpu events_device.jsonl \
        --out merged_events.jsonl
"""

import argparse
import json
from typing import Optional, Dict, Any


def extract_timestamp(evt: Dict[str, Any]) -> Optional[int]:
    """
    Try to extract a nanosecond timestamp from an event.
    Preference order:
      1) t_ns
      2) recorder_t_ns
      3) gpu_start
      4) start
    """

    if "t_ns" in evt:
        return int(evt["t_ns"])

    if "recorder_t_ns" in evt:
        return int(evt["recorder_t_ns"])

    if "gpu_start" in evt:
        return int(evt["gpu_start"])

    if "start" in evt:
        return int(evt["start"])

    return None


def load_jsonl(path: str, source: str):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                # Skip corrupt lines but keep going
                continue

            ts = extract_timestamp(evt)

            if ts is None:
                # Skip events without usable timestamps
                continue

            evt["merged_t_ns"] = ts
            evt["source"] = source
            events.append(evt)

    return events


def main():
    parser = argparse.ArgumentParser(description="Merge CPU and GPU JSONL traces.")
    parser.add_argument("--cpu", required=True, help="CPU-side JSONL file")
    parser.add_argument("--gpu", required=True, help="GPU-side JSONL file")
    parser.add_argument("--out", default="merged_events.jsonl", help="Output merged JSONL")
    args = parser.parse_args()

    print("Loading CPU events...")
    cpu_events = load_jsonl(args.cpu, "cpu")

    print("Loading GPU events...")
    gpu_events = load_jsonl(args.gpu, "gpu")

    print(f"CPU events: {len(cpu_events)}")
    print(f"GPU events: {len(gpu_events)}")

    all_events = cpu_events + gpu_events

    print("Sorting by timestamp...")
    all_events.sort(key=lambda e: e["merged_t_ns"])

    print(f"Writing merged file: {args.out}")
    with open(args.out, "w", encoding="utf-8") as out:
        for evt in all_events:
            out.write(json.dumps(evt) + "\n")

    print("Done.")
    print(f"Total merged events: {len(all_events)}")


if __name__ == "__main__":
    main()