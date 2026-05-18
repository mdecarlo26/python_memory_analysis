#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


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


def get_time_ns(e: Dict[str, Any]) -> Optional[int]:
    # Prefer aligned time if you created it, else fall back.
    for k in ("aligned_t_ns", "merged_t_ns", "t_ns", "cpu_t_ns", "recorder_t_ns"):
        if k in e:
            try:
                return int(e[k])
            except Exception:
                pass
    return None


def event_key(e: Dict[str, Any], i: int) -> str:
    # stable-ish id for nodes
    t = get_time_ns(e)
    base = f"{e.get('type','evt')}"
    if t is not None:
        return f"{base}:{t}:{i}"
    return f"{base}:idx:{i}"


def short_label(e: Dict[str, Any]) -> str:
    t = get_time_ns(e)
    if e.get("type") == "callback":
        api = e.get("api", "api")
        site = e.get("site", "")
        corr = e.get("corr", "")
        return f"CPU {api} {site} (corr {corr})"
    if e.get("type") == "activity":
        kind = e.get("kind", "act")
        corr = e.get("corr", "")
        if kind == "memcpy":
            op = e.get("op", "memcpy")
            b = e.get("bytes", None)
            if b is not None:
                return f"GPU memcpy {op} {int(b)//1024}KB (corr {corr})"
            return f"GPU memcpy {op} (corr {corr})"
        if kind == "kernel":
            name = e.get("name", "kernel")
            return f"GPU kernel {name} (corr {corr})"
        return f"GPU {kind} (corr {corr})"
    if e.get("type") == "host_stats":
        rss = e.get("rss_bytes", None)
        if rss is not None:
            return f"Host RSS {rss//(1024*1024)}MB"
        return "Host stats"
    if e.get("type") == "meminfo":
        used = e.get("used_mb", None)
        side = e.get("side", "device")
        if used is not None:
            return f"{side} used {used}MB"
        return f"{side} meminfo"
    if e.get("type") == "marker":
        return f"Marker: {e.get('msg','')}"
    return e.get("type", "event")


def node_group(e: Dict[str, Any]) -> str:
    if e.get("type") == "callback":
        return "cpu_api"
    if e.get("type") == "activity":
        return "gpu_activity"
    if e.get("type") == "host_stats":
        return "host_mem"
    if e.get("type") == "meminfo":
        return "device_mem"
    if e.get("type") == "marker":
        return "marker"
    return "other"


def build_graph(events: List[Dict[str, Any]], max_host_samples: int = 4000) -> Dict[str, Any]:
    """
    Graph model:
      nodes: [{id, label, group, t_ns, data}]
      edges: [{id, from, to, kind, data}]
    """
    # Sort by time (best effort)
    events_sorted = sorted(
        [(i, e) for i, e in enumerate(events)],
        key=lambda ie: (get_time_ns(ie[1]) is None, get_time_ns(ie[1]) or 0, ie[0]),
    )

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_by_id: Dict[str, Dict[str, Any]] = {}

    # corr hubs
    corr_hub_id: Dict[int, str] = {}

    def add_node(n: Dict[str, Any]):
        if n["id"] in node_by_id:
            return
        node_by_id[n["id"]] = n
        nodes.append(n)

    def add_edge(frm: str, to: str, kind: str, data: Optional[Dict[str, Any]] = None):
        eid = f"e:{kind}:{frm}->{to}:{len(edges)}"
        edges.append({"id": eid, "from": frm, "to": to, "kind": kind, "data": data or {}})

    # Optionally downsample host_stats (they can dominate)
    host_stats = [(i, e) for i, e in events_sorted if e.get("type") == "host_stats"]
    if len(host_stats) > max_host_samples:
        step = max(1, len(host_stats) // max_host_samples)
        keep = set(idx for idx, _ in host_stats[::step])
    else:
        keep = set(idx for idx, _ in host_stats)

    prev_event_node: Optional[str] = None
    prev_mem_node_host: Optional[str] = None
    prev_mem_node_dev: Optional[str] = None

    for i, e in events_sorted:
        # skip host_stats if downsampled out
        if e.get("type") == "host_stats" and i not in keep:
            continue

        eid = event_key(e, i)
        t = get_time_ns(e)

        n = {
            "id": eid,
            "label": short_label(e),
            "group": node_group(e),
            "t_ns": t,
            "data": e,  # full original JSON attached for drilldown
        }
        add_node(n)

        # Sequence edge for time ordering (optional but useful)
        if prev_event_node is not None:
            add_edge(prev_event_node, eid, "next")
        prev_event_node = eid

        # Correlation hub edges (causal)
        corr = e.get("corr", None)
        if corr is not None and isinstance(corr, int):
            if corr not in corr_hub_id:
                hid = f"corr:{corr}"
                corr_hub_id[corr] = hid
                add_node({
                    "id": hid,
                    "label": f"corr {corr}",
                    "group": "corr",
                    "t_ns": t,
                    "data": {"corr": corr},
                })
            add_edge(corr_hub_id[corr], eid, "corr")

        # Memory chaining (state timeline)
        if e.get("type") == "host_stats":
            if prev_mem_node_host is not None:
                add_edge(prev_mem_node_host, eid, "mem_next", {"side": "host"})
            prev_mem_node_host = eid

        if e.get("type") == "meminfo":
            side = e.get("side", "device")
            if side in ("device", "gpu"):
                if prev_mem_node_dev is not None:
                    add_edge(prev_mem_node_dev, eid, "mem_next", {"side": "device"})
                prev_mem_node_dev = eid

        # Link markers to nearest event (here: to itself in sequence already; optionally link to corr if present)
        # You can evolve this to link markers to a window of events.

    # Metadata
    graph = {
        "meta": {
            "num_events_in": len(events),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def main():
    ap = argparse.ArgumentParser(description="Build a graph data structure from aligned JSONL trace events.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (aligned_events.jsonl recommended)")
    ap.add_argument("--out", default="graph.json", help="Output graph JSON")
    ap.add_argument("--max-host-samples", type=int, default=3000, help="Downsample host_stats to this many nodes")
    args = ap.parse_args()

    events = read_jsonl(args.inp)
    g = build_graph(events, max_host_samples=args.max_host_samples)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(g, f)

    print("Wrote:", args.out)
    print("Nodes:", g["meta"]["num_nodes"], "Edges:", g["meta"]["num_edges"])


if __name__ == "__main__":
    main()