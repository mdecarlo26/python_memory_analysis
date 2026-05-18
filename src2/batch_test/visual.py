#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_jsonl(path: str, events: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def median_int(xs: List[int]) -> Optional[int]:
    if not xs:
        return None
    xs = sorted(xs)
    m = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[m]
    return (xs[m - 1] + xs[m]) // 2


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def bytes_to_mb(b: int) -> float:
    return b / (1024.0 * 1024.0)


def classify_source(e: Dict[str, Any]) -> str:
    # Your merge sample is wrong: CPU callbacks are labeled source="gpu".
    # Fix source deterministically from fields.
    if e.get("type") == "host_stats" or (e.get("type") == "marker" and e.get("source") == "cpu"):
        return "cpu"
    if e.get("type") == "callback" and e.get("side") == "cpu":
        return "cpu"
    if e.get("type") == "activity" and e.get("side") == "gpu":
        return "gpu"
    # fallback to existing
    return e.get("source", "unknown")


def get_t_ns(e: Dict[str, Any]) -> Optional[int]:
    # prefer explicit t_ns
    if "t_ns" in e:
        try:
            return int(e["t_ns"])
        except Exception:
            return None
    # older keys
    for k in ("cpu_t_ns", "recorder_t_ns", "merged_t_ns"):
        if k in e:
            try:
                return int(e[k])
            except Exception:
                pass
    return None


def get_gpu_start_end(e: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    s = None
    t = None
    for k in ("start", "gpu_start"):
        if k in e:
            try:
                s = int(e[k])
                break
            except Exception:
                pass
    for k in ("end", "gpu_end"):
        if k in e:
            try:
                t = int(e[k])
                break
            except Exception:
                pass
    return s, t


def cpu_kind_from_api(api: str) -> str:
    if not api:
        return "other"
    if api.startswith("cudaMemcpy"):
        return "memcpy"
    if api == "cudaLaunchKernel":
        return "kernel_launch"
    return "other"


def build_cpu_enter_index(cpu_callbacks: List[Dict[str, Any]]) -> Dict[Tuple[int, str], int]:
    """
    Map (corr, kind) -> cpu_enter_t_ns (in whatever CPU domain the callbacks currently are).
    """
    idx: Dict[Tuple[int, str], int] = {}
    for e in cpu_callbacks:
        if e.get("type") != "callback" or e.get("side") != "cpu":
            continue
        if e.get("site") != "enter":
            continue
        corr = e.get("corr")
        api = e.get("api", "")
        t = get_t_ns(e)
        if corr is None or t is None:
            continue
        idx[(int(corr), cpu_kind_from_api(str(api)))] = int(t)
    return idx


def estimate_cb_to_host_offset_ns(cpu_callbacks: List[Dict[str, Any]], host_events: List[Dict[str, Any]]) -> int:
    """
    Align callback clock domain to host epoch domain.

    We assume:
      - host_events t_ns are epoch (~1e18)
      - cpu_callbacks t_ns are monotonic (~1e12)
    Use earliest timestamp in each domain as anchor.
    """
    cb_ts = [get_t_ns(e) for e in cpu_callbacks]
    cb_ts = [t for t in cb_ts if t is not None]
    host_ts = [get_t_ns(e) for e in host_events]
    host_ts = [t for t in host_ts if t is not None]

    if not cb_ts or not host_ts:
        return 0

    cb0 = min(cb_ts)
    host0 = min(host_ts)
    return int(host0 - cb0)


def estimate_gpu_to_cpu_offset_ns(cpu_callbacks_aligned: List[Dict[str, Any]], gpu_acts: List[Dict[str, Any]]) -> int:
    """
    Estimate offset so that: gpu_start + offset ~= cpu_enter_t_ns  (CPU callbacks already aligned to host epoch)
    Uses corr-based matching, median for robustness.
    """
    idx = build_cpu_enter_index(cpu_callbacks_aligned)
    offsets: List[int] = []

    for g in gpu_acts:
        if g.get("type") != "activity" or g.get("side") != "gpu":
            continue
        corr = g.get("corr")
        kind = g.get("kind")
        s, _ = get_gpu_start_end(g)
        if corr is None or s is None:
            continue

        if kind == "memcpy":
            ck = "memcpy"
        elif kind == "kernel":
            ck = "kernel_launch"
        else:
            ck = "other"

        cpu_enter = idx.get((int(corr), ck))
        if cpu_enter is None:
            continue

        offsets.append(int(cpu_enter - int(s)))

    m = median_int(offsets)
    return int(m) if m is not None else 0


def align_events(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Returns:
      aligned_events, cb_to_host_offset_ns, gpu_to_cpu_offset_ns
    Adds: aligned_t_ns for everything (host epoch domain)
    """
    # Re-classify sources (fix your current merge artifact)
    for e in events:
        e["source"] = classify_source(e)

    host_events = [e for e in events if e["source"] == "cpu" and e.get("type") in ("host_stats", "marker", "gc")]
    cpu_callbacks = [e for e in events if e.get("type") == "callback" and e.get("side") == "cpu"]
    gpu_acts = [e for e in events if e.get("type") == "activity" and e.get("side") == "gpu"]

    cb_to_host = estimate_cb_to_host_offset_ns(cpu_callbacks, host_events)

    # First: align CPU callback timestamps into host epoch domain
    for e in cpu_callbacks:
        t = get_t_ns(e)
        if t is not None:
            e["aligned_t_ns"] = int(t + cb_to_host)

    # Host events already in epoch domain
    for e in host_events:
        t = get_t_ns(e)
        if t is not None:
            e["aligned_t_ns"] = int(t)

    # Estimate GPU->CPU offset using aligned CPU callbacks
    cpu_callbacks_aligned = [e for e in cpu_callbacks if "aligned_t_ns" in e]
    # Build a temporary copy where callback enter uses aligned time in t_ns for matching
    tmp_callbacks = []
    for e in cpu_callbacks_aligned:
        c = dict(e)
        c["t_ns"] = c["aligned_t_ns"]
        tmp_callbacks.append(c)

    gpu_to_cpu = estimate_gpu_to_cpu_offset_ns(tmp_callbacks, gpu_acts)

    # Align GPU activity into host epoch domain:
    # gpu_start/end are in GPU domain; shift by gpu_to_cpu then also *implicitly* into host domain because CPU is host-aligned.
    for g in gpu_acts:
        s, t = get_gpu_start_end(g)
        if s is None:
            continue
        g["aligned_start_ns"] = int(s + gpu_to_cpu)
        if t is not None:
            g["aligned_end_ns"] = int(t + gpu_to_cpu)
        g["aligned_t_ns"] = g["aligned_start_ns"]

    # For any event not covered, fall back to merged_t_ns if it looks epoch-ish
    for e in events:
        if "aligned_t_ns" in e:
            continue
        t = get_t_ns(e)
        if t is not None:
            e["aligned_t_ns"] = int(t)

    # Sort by aligned time
    events_sorted = sorted([e for e in events if "aligned_t_ns" in e], key=lambda x: x["aligned_t_ns"])
    return events_sorted, cb_to_host, gpu_to_cpu


def plot_timeline(events: List[Dict[str, Any]], out_png: str, title: str, show_cpu_bars: bool = True):
    # Establish t0/t1
    ts = [e["aligned_t_ns"] for e in events if "aligned_t_ns" in e]
    if not ts:
        raise RuntimeError("No aligned timestamps to plot.")
    t0 = min(ts)
    t1 = max(ts)
    span_ms = ns_to_ms(t1 - t0)

    # Memory series
    host_t, host_rss = [], []
    for e in events:
        if e.get("type") == "host_stats" and "rss_bytes" in e:
            host_t.append(ns_to_ms(e["aligned_t_ns"] - t0))
            host_rss.append(bytes_to_mb(int(e["rss_bytes"])))

    dev_t, dev_used = [], []
    for e in events:
        if e.get("type") == "meminfo" and e.get("side") in ("device", "cpu"):
            if "used_mb" in e and "aligned_t_ns" in e:
                dev_t.append(ns_to_ms(e["aligned_t_ns"] - t0))
                dev_used.append(float(e["used_mb"]))

    # GPU bars
    memcpy_bars = []
    kernel_bars = []
    for e in events:
        if e.get("type") == "activity" and e.get("side") == "gpu":
            s = e.get("aligned_start_ns")
            en = e.get("aligned_end_ns")
            if s is None or en is None:
                continue
            x = ns_to_ms(int(s) - t0)
            w = ns_to_ms(int(en) - int(s))
            if e.get("kind") == "memcpy":
                memcpy_bars.append((x, w))
            elif e.get("kind") == "kernel":
                kernel_bars.append((x, w))

    # CPU API bars (from callback exits if duration present)
    cpu_bars = []
    if show_cpu_bars:
        for e in events:
            if e.get("type") == "callback" and e.get("side") == "cpu" and e.get("site") == "exit":
                dur_us = e.get("dur_us")
                if dur_us is None:
                    continue
                end_ns = e.get("aligned_t_ns")
                start_ns = int(end_ns - float(dur_us) * 1000.0)
                cpu_bars.append((ns_to_ms(start_ns - t0), float(dur_us) / 1000.0))

    # Plot
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.5], hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    fig.suptitle(title, fontsize=14)

    if host_t:
        ax1.plot(host_t, host_rss, label="Host RSS (MB)")
    if dev_t:
        ax1.plot(dev_t, dev_used, label="Device used (MB)")
    ax1.set_xlim(0, span_ms)
    ax1.set_ylabel("Memory")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # lanes
    y_cpu, y_kernel, y_memcpy = 10, 20, 30
    if show_cpu_bars:
        for x, w in cpu_bars:
            ax2.broken_barh([(x, w)], (y_cpu - 3, 6))
    for x, w in kernel_bars:
        ax2.broken_barh([(x, w)], (y_kernel - 3, 6))
    for x, w in memcpy_bars:
        ax2.broken_barh([(x, w)], (y_memcpy - 3, 6))

    ax2.set_yticks([y_cpu, y_kernel, y_memcpy])
    ax2.set_yticklabels(["CPU API", "GPU Kernel", "GPU Memcpy"])
    ax2.set_xlim(0, span_ms)
    ax2.set_xlabel("Time (ms), aligned to host epoch domain")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print("Wrote:", out_png)


def main():
    ap = argparse.ArgumentParser(description="Align mixed clock domains and generate a time-scaled plot.")
    ap.add_argument("--in", dest="inp", required=True, help="Merged JSONL input")
    ap.add_argument("--out-jsonl", default="aligned_events.jsonl", help="Aligned JSONL output")
    ap.add_argument("--out-png", default="timeline.png", help="Timeline PNG output")
    ap.add_argument("--no-cpu-bars", action="store_true", help="Disable CPU API duration bars")
    args = ap.parse_args()

    events = read_jsonl(args.inp)
    aligned, cb_to_host, gpu_to_cpu = align_events(events)

    # annotate offsets in a meta record at start
    meta = {
        "type": "meta",
        "msg": "alignment_offsets",
        "cb_to_host_offset_ns": int(cb_to_host),
        "gpu_to_cpu_offset_ns": int(gpu_to_cpu),
    }
    aligned_with_meta = [meta] + aligned

    write_jsonl(args.out_jsonl, aligned_with_meta)
    print("Wrote aligned JSONL:", args.out_jsonl)
    print("Applied cb_to_host_offset_ns:", cb_to_host)
    print("Applied gpu_to_cpu_offset_ns:", gpu_to_cpu)

    plot_timeline(
        aligned,
        out_png=args.out_png,
        title="Time-scaled CPU/GPU timeline (clock-normalized + corr-aligned)",
        show_cpu_bars=not args.no_cpu_bars,
    )


if __name__ == "__main__":
    main()