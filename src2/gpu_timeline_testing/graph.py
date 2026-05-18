#!/usr/bin/env python3
import argparse
import re
import math
import json
import pandas as pd
from pyvis.network import Network


# -----------------------------
# Regex for your log format
# -----------------------------
CB_RE = re.compile(
    r"^\[CB\]\[(ENTER|EXIT )\]\[CPU\]\s+t=(\d+)\s+ns\s+corr=(\d+)\s+(\S+)(?:\s+dur=([\d.]+)us)?\s*$"
)
MEMINFO_RE = re.compile(
    r"^\[MEMINFO\]\[CPU\]\s+(?P<tag>.+?)\s+free=(\d+)MB\s+total=(\d+)MB\s+used=(\d+)MB\s*$"
)
ACT_MEMCPY_RE = re.compile(
    r"^\[ACT\]\[MEMCPY\]\[GPU\]\s+(\S+)\s+bytes=(\d+)\s+start=(\d+)\s+end=(\d+)\s+dur=([\d.]+)us\s+stream=(\d+)\s+corr=(\d+)\s*$"
)
ACT_KERNEL_RE = re.compile(
    r"^\[ACT\]\[KERNEL\]\[GPU\]\s+(\S+)\s+start=(\d+)\s+end=(\d+)\s+dur=([\d.]+)us\s+grid=\(([^)]+)\)\s+block=\(([^)]+)\)\s+stream=(\d+)\s+corr=(\d+)\s*$"
)


# -----------------------------
# Helpers
# -----------------------------
def fmt_bytes(n: int) -> str:
    if n is None:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    if u == 0:
        return f"{int(x)} {units[u]}"
    return f"{x:.2f} {units[u]}"

def scale_size_us(dur_us: float) -> int:
    # compressed sizing; readable but not overwhelming
    if dur_us is None or (isinstance(dur_us, float) and math.isnan(dur_us)):
        return 14
    return int(14 + 7 * math.log10(1.0 + max(dur_us, 0.0)))

def html_kv_table(title: str, kv: dict) -> str:
    # Pretty hover tooltip; HTML is supported by pyvis "title"
    rows = "".join(
        f"<tr><td style='padding:2px 8px; color:#555;'><b>{k}</b></td>"
        f"<td style='padding:2px 8px;'>{v}</td></tr>"
        for k, v in kv.items()
    )
    return (
        f"<div style='font-family:Arial; font-size:13px;'>"
        f"<div style='font-size:14px; margin-bottom:6px;'><b>{title}</b></div>"
        f"<table style='border-collapse:collapse;'>{rows}</table>"
        f"</div>"
    )

def sanitize_kernel_name(mangled: str) -> str:
    # Keep it short; still show full name in tooltip.
    if mangled is None:
        return "kernel"
    if mangled.startswith("_Z"):
        return "matmulKernel"  # heuristic for demo; replace with demangle if you want
    return mangled if len(mangled) <= 24 else (mangled[:21] + "...")

def parse(log_path: str):
    cpu_events = []
    gpu_events = []
    mem_events = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = MEMINFO_RE.match(line)
            if m:
                mem_events.append({
                    "tag": m.group("tag"),
                    "free_mb": int(m.group(2)),
                    "total_mb": int(m.group(3)),
                    "used_mb": int(m.group(4)),
                })
                continue

            m = CB_RE.match(line)
            if m:
                cpu_events.append({
                    "site": m.group(1).strip(),
                    "t_cpu_ns": int(m.group(2)),
                    "corr": int(m.group(3)),
                    "api": m.group(4),
                    "dur_us_reported": float(m.group(5)) if m.group(5) else None,
                })
                continue

            m = ACT_MEMCPY_RE.match(line)
            if m:
                gpu_events.append({
                    "kind": "MEMCPY",
                    "op": m.group(1),
                    "bytes": int(m.group(2)),
                    "t_gpu_start": int(m.group(3)),
                    "t_gpu_end": int(m.group(4)),
                    "dur_us": float(m.group(5)),
                    "stream": int(m.group(6)),
                    "corr": int(m.group(7)),
                })
                continue

            m = ACT_KERNEL_RE.match(line)
            if m:
                gpu_events.append({
                    "kind": "KERNEL",
                    "name": m.group(1),
                    "t_gpu_start": int(m.group(2)),
                    "t_gpu_end": int(m.group(3)),
                    "dur_us": float(m.group(4)),
                    "grid": m.group(5),
                    "block": m.group(6),
                    "stream": int(m.group(7)),
                    "corr": int(m.group(8)),
                })
                continue

    cpu_df = pd.DataFrame(cpu_events)
    gpu_df = pd.DataFrame(gpu_events)
    mem_df = pd.DataFrame(mem_events)

    # Build CPU intervals (ENTER->EXIT) per corr/api
    if not cpu_df.empty:
        enters = cpu_df[cpu_df["site"] == "ENTER"][["corr", "api", "t_cpu_ns"]].rename(columns={"t_cpu_ns":"t_cpu_start"})
        exits  = cpu_df[cpu_df["site"] == "EXIT"][["corr", "api", "t_cpu_ns", "dur_us_reported"]].rename(columns={"t_cpu_ns":"t_cpu_end"})
        cpu_iv = pd.merge(enters, exits, on=["corr","api"], how="outer")
        cpu_iv["cpu_dur_us"] = (cpu_iv["t_cpu_end"] - cpu_iv["t_cpu_start"]) / 1000.0
        cpu_iv.loc[cpu_iv["cpu_dur_us"].isna(), "cpu_dur_us"] = cpu_iv["dur_us_reported"]
    else:
        cpu_iv = pd.DataFrame()

    return cpu_iv, gpu_df, mem_df


# -----------------------------
# Graph construction
# -----------------------------
def build_graph(cpu_iv: pd.DataFrame, gpu_df: pd.DataFrame, mem_df: pd.DataFrame, out_html: str):
    net = Network(height="850px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=170, spring_strength=0.02, damping=0.25)

    # Legend
    net.add_node("LEGEND", label="Legend", shape="box", color="#f5f5f5", physics=False, x=-750, y=-280,
                 title=html_kv_table("Legend", {
                     "corr hub": "Causal join point (CPU request → GPU work)",
                     "CPU call node": "From CUPTI Callback API (CUpti_CallbackData)",
                     "GPU activity node": "From CUPTI Activity API (CUpti_ActivityMemcpy / CUpti_ActivityKernel4)",
                     "stream node": "CUDA streamId hub",
                 }))

    # Collect unique corr IDs
    corrs = set()
    if not cpu_iv.empty:
        corrs |= set(cpu_iv["corr"].dropna().astype(int).tolist())
    if not gpu_df.empty:
        corrs |= set(gpu_df["corr"].dropna().astype(int).tolist())

    # Corr hub nodes
    for corr in sorted(corrs):
        nid = f"CORR:{corr}"
        net.add_node(
            nid,
            label=f"corr {corr}",
            shape="dot",
            size=18,
            color="#e9e9ff",
            title=html_kv_table("Correlation hub", {
                "corr": corr,
                "meaning": "Causal ID joining CPU API call(s) to GPU activity record(s)",
                "why it matters": "Lets you attribute GPU work to the exact CPU request that triggered it"
            })
        )

    # Stream hub nodes
    streams = set()
    if not gpu_df.empty and "stream" in gpu_df.columns:
        streams |= set(gpu_df["stream"].dropna().astype(int).tolist())
    for s in sorted(streams):
        nid = f"STREAM:{s}"
        net.add_node(
            nid,
            label=f"stream {s}",
            shape="dot",
            size=16,
            color="#e6ffe6",
            title=html_kv_table("CUDA Stream", {
                "streamId": s,
                "meaning": "Execution/ordering queue on the GPU",
                "why it matters": "Events on the same stream are ordered; different streams may overlap"
            })
        )

    # CPU nodes (CUpti_CallbackData -> interval)
    # Create ONE node per (corr, api, row) to stay accurate even if multiple per corr.
    if not cpu_iv.empty:
        for idx, r in cpu_iv.iterrows():
            corr = int(r["corr"])
            api = str(r["api"])
            dur = float(r["cpu_dur_us"]) if pd.notna(r["cpu_dur_us"]) else None

            nid = f"CPU:{corr}:{idx}"
            label = f"CPU {api}\n{dur:.3f} µs" if dur is not None else f"CPU {api}"
            title = html_kv_table("CPU CUDA API call (CUPTI Callback API)", {
                "CUPTI struct": "CUpti_CallbackData",
                "functionName": api,
                "correlationId": corr,
                "cpu_duration_us": f"{dur:.3f}" if dur is not None else "?",
                "note": "This is CPU time inside the API call, not GPU execution time."
            })

            net.add_node(
                nid,
                label=label,
                shape="box",
                color="#dff0ff",
                size=scale_size_us(dur),
                title=title,
            )

            # Corr hub -> CPU call
            net.add_edge(f"CORR:{corr}", nid, label="requests", arrows="to")

    # GPU activity nodes (CUpti_ActivityMemcpy / CUpti_ActivityKernel4)
    if not gpu_df.empty:
        for idx, r in gpu_df.iterrows():
            corr = int(r["corr"])
            kind = r["kind"]
            dur = float(r["dur_us"]) if pd.notna(r["dur_us"]) else None
            stream = int(r["stream"]) if pd.notna(r.get("stream", None)) else -1

            if kind == "MEMCPY":
                op = r.get("op", "memcpy")
                b = int(r.get("bytes", 0))
                label = f"GPU MEMCPY {op}\n{fmt_bytes(b)} • {dur:.3f} µs"
                title = html_kv_table("GPU MEMCPY (CUPTI Activity API)", {
                    "CUPTI struct": "CUpti_ActivityMemcpy",
                    "copyKind": op,
                    "bytes": b,
                    "streamId": stream,
                    "correlationId": corr,
                    "gpu_duration_us": f"{dur:.3f}" if dur is not None else "?",
                    "start/end": "GPU timestamp domain",
                })
                shape = "ellipse"
                color = "#ffe8d6"
            else:
                full = r.get("name", "kernel")
                short = sanitize_kernel_name(full)
                grid = r.get("grid", "?")
                block = r.get("block", "?")
                label = f"GPU KERNEL {short}\n{dur:.3f} µs"
                title = html_kv_table("GPU KERNEL (CUPTI Activity API)", {
                    "CUPTI struct": "CUpti_ActivityKernel4",
                    "name": full,
                    "grid": grid,
                    "block": block,
                    "streamId": stream,
                    "correlationId": corr,
                    "gpu_duration_us": f"{dur:.3f}" if dur is not None else "?",
                    "start/end": "GPU timestamp domain",
                })
                shape = "box"
                color = "#ffe8d6"

            nid = f"GPU:{corr}:{idx}"
            net.add_node(
                nid,
                label=label,
                title=title,
                shape=shape,
                color=color,
                size=scale_size_us(dur),
            )

            # Corr hub -> GPU work
            net.add_edge(f"CORR:{corr}", nid, label="causes", arrows="to")

            # GPU work -> stream hub
            if stream != -1:
                net.add_edge(nid, f"STREAM:{stream}", label="runs on", arrows="to")

    # Memory snapshot nodes (state)
    if not mem_df.empty:
        for i, r in mem_df.iterrows():
            used = int(r["used_mb"])
            nid = f"MEM:{i}"
            net.add_node(
                nid,
                label=f"MEM used\n{used} MB",
                shape="diamond",
                color="#f1fff1",
                size=16,
                title=html_kv_table("Device memory snapshot", {
                    "source": "cudaMemGetInfo",
                    "tag": str(r["tag"]),
                    "used_mb": used,
                    "free_mb": int(r["free_mb"]),
                    "total_mb": int(r["total_mb"]),
                    "note": "Snapshot only. If you want time-alignment, log a timestamp with MEMINFO."
                })
            )
            if i > 0:
                net.add_edge(f"MEM:{i-1}", nid, label="next snapshot", dashes=True, color="#2a9d8f")

    # Visual options
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1,
        "font": {"size": 14, "face": "arial"},
        "shadow": true
      },
      "edges": {
        "smooth": {"type": "dynamic"},
        "font": {"size": 11, "align": "middle"},
        "arrows": {"to": {"enabled": true}},
        "color": {"inherit": false}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "enabled": true
      }
    }
    """)

    # Use write_html (more reliable than show())
    net.write_html(out_html, open_browser=True)
    print(f"Wrote interactive graph to: {out_html}")


def main():
    ap = argparse.ArgumentParser(description="Build an informative corr/CPU/GPU/stream graph from CUPTI logs.")
    ap.add_argument("logfile", help="Path to log file (e.g., run.log)")
    ap.add_argument("--out", default="cupti_graph_informative.html", help="Output HTML file")
    args = ap.parse_args()

    cpu_iv, gpu_df, mem_df = parse(args.logfile)
    build_graph(cpu_iv, gpu_df, mem_df, args.out)


if __name__ == "__main__":
    main()
