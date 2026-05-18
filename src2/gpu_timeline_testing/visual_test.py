#!/usr/bin/env python3
import argparse
import re
import math
import pandas as pd
from pyvis.network import Network

# Parses the enriched lines we added.
CB_ENTER_RE = re.compile(
    r"^\[CB\]\[ENTER\]\[CPU\]\s+t=(\d+)\s+ns\s+corr=(\d+)\s+(\S+)(.*)$"
)
CB_EXIT_RE = re.compile(
    r"^\[CB\]\[EXIT\s*\]\[CPU\]\s+t=(\d+)\s+ns\s+corr=(\d+)\s+(\S+)\s+dur=([\d.]+)us(.*)$"
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

def fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    x = float(n)
    u = 0
    while x >= 1024 and u < len(units)-1:
        x /= 1024
        u += 1
    return f"{x:.2f} {units[u]}" if u else f"{int(x)} B"

def scale_size(val: float) -> int:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 14
    return int(14 + 7 * math.log10(1 + max(val, 0.0)))

def parse_kv_tail(tail: str) -> dict:
    # tail looks like: " dev=0x... size=123 dst=0x... ..."
    out = {}
    for tok in tail.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def parse_log(path: str):
    cpu_rows = []
    gpu_rows = []
    mem_rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = MEMINFO_RE.match(line)
            if m:
                mem_rows.append({
                    "tag": m.group("tag"),
                    "free_mb": int(m.group(2)),
                    "total_mb": int(m.group(3)),
                    "used_mb": int(m.group(4)),
                })
                continue

            m = CB_ENTER_RE.match(line)
            if m:
                t_ns = int(m.group(1)); corr = int(m.group(2)); api = m.group(3); tail = m.group(4)
                kv = parse_kv_tail(tail)
                cpu_rows.append({
                    "site": "ENTER", "t_cpu_ns": t_ns, "corr": corr, "api": api, **kv
                })
                continue

            m = CB_EXIT_RE.match(line)
            if m:
                t_ns = int(m.group(1)); corr = int(m.group(2)); api = m.group(3); dur_us = float(m.group(4)); tail = m.group(5)
                kv = parse_kv_tail(tail)
                cpu_rows.append({
                    "site": "EXIT", "t_cpu_ns": t_ns, "corr": corr, "api": api, "dur_us": dur_us, **kv
                })
                continue

            m = ACT_MEMCPY_RE.match(line)
            if m:
                gpu_rows.append({
                    "kind": "MEMCPY",
                    "op": m.group(1),
                    "bytes": int(m.group(2)),
                    "start": int(m.group(3)),
                    "end": int(m.group(4)),
                    "dur_us": float(m.group(5)),
                    "stream": int(m.group(6)),
                    "corr": int(m.group(7)),
                })
                continue

            m = ACT_KERNEL_RE.match(line)
            if m:
                gpu_rows.append({
                    "kind": "KERNEL",
                    "name": m.group(1),
                    "start": int(m.group(2)),
                    "end": int(m.group(3)),
                    "dur_us": float(m.group(4)),
                    "grid": m.group(5),
                    "block": m.group(6),
                    "stream": int(m.group(7)),
                    "corr": int(m.group(8)),
                })
                continue

    cpu_df = pd.DataFrame(cpu_rows)
    gpu_df = pd.DataFrame(gpu_rows)
    mem_df = pd.DataFrame(mem_rows)

    # Build CPU intervals for display
    if not cpu_df.empty:
        enters = cpu_df[cpu_df["site"] == "ENTER"][["corr","api","t_cpu_ns"]].rename(columns={"t_cpu_ns":"t_start"})
        exits  = cpu_df[cpu_df["site"] == "EXIT"][["corr","api","t_cpu_ns","dur_us"]].rename(columns={"t_cpu_ns":"t_end"})
        cpu_iv = pd.merge(enters, exits, on=["corr","api"], how="outer")
        cpu_iv["cpu_dur_us"] = cpu_iv["dur_us"]
    else:
        cpu_iv = pd.DataFrame()

    return cpu_df, cpu_iv, gpu_df, mem_df

def build_graph(cpu_df: pd.DataFrame, cpu_iv: pd.DataFrame, gpu_df: pd.DataFrame, mem_df: pd.DataFrame, out_html: str):
    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    net.barnes_hut(gravity=-9500, central_gravity=0.25, spring_length=170, spring_strength=0.02, damping=0.25)

    # 1) Corr hubs
    corrs = set()
    if not cpu_df.empty: corrs |= set(cpu_df["corr"].dropna().astype(int).tolist())
    if not gpu_df.empty: corrs |= set(gpu_df["corr"].dropna().astype(int).tolist())
    for c in sorted(corrs):
        net.add_node(f"CORR:{c}", label=f"corr {c}", shape="dot", size=18, color="#e9e9ff",
                     title=f"<b>Correlation hub</b><br>corr={c}<br>Join CPU request ↔ GPU work")

    # 2) Stream hubs
    streams = set()
    if not gpu_df.empty and "stream" in gpu_df.columns:
        streams |= set(gpu_df["stream"].dropna().astype(int).tolist())
    for s in sorted(streams):
        net.add_node(f"STREAM:{s}", label=f"stream {s}", shape="dot", size=16, color="#e6ffe6",
                     title=f"<b>CUDA stream</b><br>streamId={s}")

    # 3) Buffer nodes from cudaMalloc EXIT lines
    # We emitted: [CB][EXIT ] ... cudaMalloc ... dev=0x... size=...
    buffers = {}  # dev_id -> {"size":..., "node":...}
    if not cpu_df.empty:
        mallocs = cpu_df[(cpu_df["site"]=="EXIT") & (cpu_df["api"]=="cudaMalloc")]
        for _, r in mallocs.iterrows():
            dev = r.get("dev")
            size = r.get("size")
            if pd.isna(dev) or pd.isna(size):
                continue
            dev = str(dev); size = int(size)
            nid = f"BUF:{dev}"
            buffers[dev] = {"size": size, "node": nid}
            net.add_node(
                nid,
                label=f"BUFFER\n{dev}\n{fmt_bytes(size)}",
                shape="box",
                color="#fff3cd",
                size=18,
                title=f"<b>Device buffer</b><br>id={dev}<br>size={size} bytes"
            )

    # 4) CPU call nodes (make them readable)
    if not cpu_iv.empty:
        for idx, r in cpu_iv.iterrows():
            corr = int(r["corr"])
            api = str(r["api"])
            dur = float(r["cpu_dur_us"]) if pd.notna(r.get("cpu_dur_us")) else None
            nid = f"CPU:{corr}:{idx}"
            net.add_node(
                nid,
                label=f"CPU {api}\n{dur:.3f} µs" if dur is not None else f"CPU {api}",
                shape="box",
                color="#dff0ff",
                size=scale_size(dur),
                title=f"<b>CPU API</b><br>api={api}<br>corr={corr}<br>cpu_dur_us={dur}"
            )
            net.add_edge(f"CORR:{corr}", nid, label="requests")

            # If this is cudaMalloc, connect to buffer node using dev id from EXIT record
            if api == "cudaMalloc":
                # find matching EXIT row with dev/size
                exit_row = cpu_df[(cpu_df["site"]=="EXIT") & (cpu_df["corr"]==corr) & (cpu_df["api"]=="cudaMalloc")]
                if not exit_row.empty:
                    dev = str(exit_row.iloc[0].get("dev", ""))
                    if dev and dev in buffers:
                        net.add_edge(nid, buffers[dev]["node"], label="allocates")

            # If this is cudaFree, connect to buffer node (dev printed on ENTER/EXIT)
            if api == "cudaFree":
                enter_row = cpu_df[(cpu_df["site"]=="ENTER") & (cpu_df["corr"]==corr) & (cpu_df["api"]=="cudaFree")]
                if not enter_row.empty:
                    dev = str(enter_row.iloc[0].get("dev", ""))
                    if dev and dev in buffers:
                        net.add_edge(nid, buffers[dev]["node"], label="frees")

            # If this is cudaMemcpy, connect to src/dst buffers when known device pointers
            if api.startswith("cudaMemcpy"):
                enter_row = cpu_df[(cpu_df["site"]=="ENTER") & (cpu_df["corr"]==corr) & (cpu_df["api"]==api)]
                if not enter_row.empty:
                    row = enter_row.iloc[0]
                    dst = str(row.get("dst", ""))
                    src = str(row.get("src", ""))
                    bytes_ = row.get("bytes")
                    kind = str(row.get("kind", ""))
                    dst_is_dev = str(row.get("dst_is_dev", "0")) == "1"
                    src_is_dev = str(row.get("src_is_dev", "0")) == "1"
                    # reads from src buffer, writes to dst buffer (when those are known dev buffers)
                    if src_is_dev and src in buffers:
                        net.add_edge(buffers[src]["node"], nid, label=f"read {kind}")
                    if dst_is_dev and dst in buffers:
                        net.add_edge(nid, buffers[dst]["node"], label=f"write {kind}")
                    # add tooltip info if present
                    net.get_node(nid)["title"] += f"<br>kind={kind}<br>bytes={bytes_}<br>dst={dst}<br>src={src}"

    # 5) GPU activity nodes (still useful for runtime truth)
    if not gpu_df.empty:
        for idx, r in gpu_df.iterrows():
            corr = int(r["corr"])
            kind = r["kind"]
            dur = float(r["dur_us"]) if pd.notna(r.get("dur_us")) else None
            stream = int(r.get("stream", -1))

            if kind == "MEMCPY":
                op = r.get("op", "MEMCPY")
                b = int(r.get("bytes", 0))
                nid = f"GPU:{corr}:{idx}"
                net.add_node(
                    nid,
                    label=f"GPU MEMCPY {op}\n{fmt_bytes(b)} • {dur:.3f} µs",
                    shape="ellipse",
                    color="#ffe8d6",
                    size=scale_size(dur),
                    title=f"<b>GPU MEMCPY</b><br>op={op}<br>bytes={b}<br>dur_us={dur}<br>stream={stream}<br>corr={corr}"
                )
            else:
                name = r.get("name", "KERNEL")
                nid = f"GPU:{corr}:{idx}"
                net.add_node(
                    nid,
                    label=f"GPU KERNEL\n{dur:.3f} µs",
                    shape="box",
                    color="#ffe8d6",
                    size=scale_size(dur),
                    title=f"<b>GPU KERNEL</b><br>name={name}<br>dur_us={dur}<br>stream={stream}<br>corr={corr}<br>grid={r.get('grid')}<br>block={r.get('block')}"
                )

            net.add_edge(f"CORR:{corr}", nid, label="causes")
            if stream != -1:
                net.add_edge(nid, f"STREAM:{stream}", label="runs on")

    # memory snapshots as state chain (optional)
    if not mem_df.empty:
        for i, r in mem_df.iterrows():
            nid = f"MEM:{i}"
            net.add_node(nid, label=f"MEM used\n{int(r['used_mb'])} MB", shape="diamond", color="#f1fff1", size=16,
                         title=f"<b>cudaMemGetInfo</b><br>tag={r['tag']}<br>used_mb={r['used_mb']}<br>free_mb={r['free_mb']}<br>total_mb={r['total_mb']}")
            if i > 0:
                net.add_edge(f"MEM:{i-1}", nid, dashes=True, label="next snapshot", color="#2a9d8f")

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
        "arrows": {"to": {"enabled": true}}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {"enabled": true}
    }
    """)

    net.write_html(out_html, open_browser=True)
    print(f"Wrote: {out_html}")

def main():
    ap = argparse.ArgumentParser(description="Graph CUPTI log as an object graph (buffers + events).")
    ap.add_argument("logfile", help="run.log")
    ap.add_argument("--out", default="cupti_buffer_graph.html")
    args = ap.parse_args()

    cpu_df, cpu_iv, gpu_df, mem_df = parse_log(args.logfile)
    build_graph(cpu_df, cpu_iv, gpu_df, mem_df, args.out)

if __name__ == "__main__":
    main()
