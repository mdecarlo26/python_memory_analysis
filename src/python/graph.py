"""
graph.py
--------
Builds an in-memory object graph from a stream of CpuEvents emitted
by PythonMemoryLayer.

Each node represents a tracked Python object. Nodes are keyed by
alloc_address (id(obj)) and track the full object lifecycle:
created at ALLOC, marked dead at DEALLOC.

Edges (references between objects) are opt-in via include_edges=True
on ingest(). Edges are captured at the moment of the ALLOC event by
calling gc.get_referents() on the live object. Edge building is skipped
by default because gc.get_referents() is O(referents) per object and
adds up quickly on large workloads.

Usage:
    graph = ObjectGraph()
    graph.ingest(events)            # add events, nodes only
    graph.ingest(events,            # add events with edges
                 include_edges=True,
                 live_objects={id(obj): obj for obj in tracked})

    node = graph.get_node(address)
    all_nodes = graph.nodes()
    edges = graph.edges()
    summary = graph.summary()
    exported = graph.to_dict()      # JSON-serializable
"""

import gc
from typing import Optional


# Types whose referents are too noisy to be useful as graph edges
_SKIP_EDGE_TYPES = (int, float, str, bytes, bool, type(None), type)


class GraphNode:
    """A single node in the object graph — one Python object."""

    __slots__ = (
        "node_id",
        "object_type",
        "size_bytes",
        "ref_count",
        "gc_generation",
        "alive",
        "first_seen_ns",
        "last_seen_ns",
        "callstack",
        "gpu_correlations",   # list of CorrelatedEvent dicts for this node
    )

    def __init__(
        self,
        node_id: int,
        object_type: str,
        size_bytes: int,
        ref_count: int,
        gc_generation: int,
        first_seen_ns: int,
        callstack: list,
    ):
        self.node_id = node_id
        self.object_type = object_type
        self.size_bytes = size_bytes
        self.ref_count = ref_count
        self.gc_generation = gc_generation
        self.alive = True
        self.first_seen_ns = first_seen_ns
        self.last_seen_ns: Optional[int] = None
        self.callstack = callstack
        self.gpu_correlations: list = []

    def mark_dead(self, timestamp_ns: int):
        self.alive = False
        self.last_seen_ns = timestamp_ns

    def lifetime_ns(self) -> Optional[int]:
        """Returns object lifetime in nanoseconds, or None if still alive."""
        if self.last_seen_ns is None:
            return None
        return self.last_seen_ns - self.first_seen_ns

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "object_type": self.object_type,
            "size_bytes": self.size_bytes,
            "ref_count": self.ref_count,
            "gc_generation": self.gc_generation,
            "alive": self.alive,
            "first_seen_ns": self.first_seen_ns,
            "last_seen_ns": self.last_seen_ns,
            "lifetime_ns": self.lifetime_ns(),
            "callstack": self.callstack,
            "gpu_correlations": self.gpu_correlations,
        }

    def __repr__(self):
        status = "alive" if self.alive else "dead"
        gpu_str = f", gpu={len(self.gpu_correlations)}" if self.gpu_correlations else ""
        return (
            f"GraphNode(id={self.node_id}, type={self.object_type}, "
            f"size={self.size_bytes}, {status}{gpu_str})"
        )


class GraphEdge:
    """A directed reference edge between two nodes (src references dst)."""

    __slots__ = ("src_id", "dst_id", "captured_at_ns")

    def __init__(self, src_id: int, dst_id: int, captured_at_ns: int):
        self.src_id = src_id
        self.dst_id = dst_id
        self.captured_at_ns = captured_at_ns

    def to_dict(self) -> dict:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "captured_at_ns": self.captured_at_ns,
        }

    def __repr__(self):
        return f"GraphEdge({self.src_id} -> {self.dst_id})"


class ObjectGraph:
    """
    In-memory object graph built from CpuEvent streams.

    Nodes are indexed by alloc_address. Edges are stored as a list of
    (src_id, dst_id) pairs and are only populated when include_edges=True
    is passed to ingest().
    """

    def __init__(self):
        self._nodes: dict[int, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        events: list[dict],
        include_edges: bool = False,
        live_objects: Optional[dict[int, object]] = None,
    ):
        """
        Process a list of CpuEvent dicts into the graph.

        Args:
            events:        CpuEvent dicts from PythonMemoryLayer.collect()
            include_edges: If True, build reference edges for each ALLOC
                           event. Requires live_objects to be provided.
            live_objects:  Dict mapping id(obj) -> obj for currently live
                           objects. Only needed when include_edges=True.
        """
        for event in events:
            addr = event["alloc_address"]
            ts = event["base"]["timestamp_ns"]

            if not event["is_dealloc"]:
                node = GraphNode(
                    node_id=addr,
                    object_type=event["object_type"],
                    size_bytes=event["alloc_size_bytes"],
                    ref_count=event["ref_count"],
                    gc_generation=event["gc_generation"],
                    first_seen_ns=ts,
                    callstack=event["callstack"],
                )
                self._nodes[addr] = node

                if include_edges and live_objects and addr in live_objects:
                    self._build_edges(addr, live_objects[addr], ts)

            else:
                if addr in self._nodes:
                    self._nodes[addr].mark_dead(ts)

    def _build_edges(self, src_id: int, obj: object, ts: int):
        """
        Walk gc.get_referents(obj) and add edges to tracked nodes.
        Only adds edges where the referent is itself a tracked node —
        we don't create nodes for previously unseen objects here.
        """
        try:
            referents = gc.get_referents(obj)
        except Exception:
            return

        for ref in referents:
            if isinstance(ref, _SKIP_EDGE_TYPES):
                continue
            dst_id = id(ref)
            if dst_id == src_id:
                continue
            if dst_id not in self._nodes:
                continue
            self._edges.append(GraphEdge(src_id, dst_id, ts))

    def gpu_correlated_nodes(self) -> list:
        """Return only nodes that have at least one GPU correlation."""
        return [n for n in self._nodes.values() if n.gpu_correlations]

    def merge_gpu_correlations(
        self,
        correlated_events: list,
        gpu_events: list,
        cpu_events: Optional[list] = None,
    ):
        """
        Annotate graph nodes with their GPU correlation data.

        After building the graph with ingest(), call this to attach
        CorrelatedEvent records (and the corresponding GpuEvent metadata)
        to each node. The UI can then display GPU transfer/kernel details
        directly on the object that triggered them.

        Args:
            correlated_events: List of CorrelatedEvent dicts from the correlator.
            gpu_events:        List of GpuEvent dicts (the full GPU event list).
            cpu_events:        Optional. If provided, used to resolve cpu_event_id
                               to alloc_address when a node isn't found by event_id.
                               Handles the case where graph was built from a subset.
        """
        # Build GPU event lookup by event_id for O(1) access
        gpu_by_id: dict = {
            e["base"]["event_id"]: e for e in gpu_events
        }

        # Build CPU event_id → alloc_address lookup if provided
        cpu_addr_by_event_id: dict = {}
        if cpu_events:
            for e in cpu_events:
                eid  = e["base"]["event_id"]
                addr = e["alloc_address"]
                cpu_addr_by_event_id[eid] = addr

        for corr in correlated_events:
            cpu_eid = corr["cpu_event_id"]
            gpu_eid = corr["gpu_event_id"]

            # Find the graph node — try by alloc_address derived from event_id
            addr = cpu_addr_by_event_id.get(cpu_eid)
            node = self._nodes.get(addr) if addr is not None else None

            # Fallback: linear scan by node_id == cpu_eid (rare)
            if node is None:
                node = self._nodes.get(cpu_eid)

            if node is None:
                continue

            gpu_ev = gpu_by_id.get(gpu_eid)
            if gpu_ev is None:
                continue

            # Attach a compact annotation to the node including all new fields
            node.gpu_correlations.append({
                "confidence":          corr["confidence"],
                "match_reason":        corr["match_reason"],
                "latency_ns":          corr["latency_ns"],
                "gpu_event_id":        gpu_eid,
                "gpu_event_type":      gpu_ev["base"]["event_type"],
                "gpu_timestamp_ns":    gpu_ev["base"]["timestamp_ns"],
                "transfer_size_bytes": gpu_ev.get("transfer_size_bytes", 0),
                "transfer_kind":       gpu_ev.get("transfer_kind", ""),
                "kernel_name":         gpu_ev.get("kernel_name", ""),
                "kernel_duration_ns":  gpu_ev.get("kernel_duration_ns", 0),
                "stream_id":           gpu_ev.get("stream_id", 0),
                "device_id":           gpu_ev.get("device_id", 0),
                # New fields from extended GPU event
                "um_bytes_htod":          gpu_ev.get("um_bytes_htod", 0),
                "um_bytes_dtoh":          gpu_ev.get("um_bytes_dtoh", 0),
                "grid":                   gpu_ev.get("grid",  {"x": 0, "y": 0, "z": 0}),
                "block":                  gpu_ev.get("block", {"x": 0, "y": 0, "z": 0}),
                "registers_per_thread":   gpu_ev.get("registers_per_thread", 0),
                "shared_mem_bytes":       gpu_ev.get("shared_mem_bytes", 0),
                "correlation_id":         gpu_ev.get("correlation_id", 0),
            })

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, address: int) -> Optional[GraphNode]:
        """Return the node for a given address, or None if not found."""
        return self._nodes.get(address)

    def nodes(self) -> list[GraphNode]:
        """Return all nodes."""
        return list(self._nodes.values())

    def alive_nodes(self) -> list[GraphNode]:
        """Return only nodes whose objects are still alive."""
        return [n for n in self._nodes.values() if n.alive]

    def dead_nodes(self) -> list[GraphNode]:
        """Return only nodes whose objects have been deallocated."""
        return [n for n in self._nodes.values() if not n.alive]

    def edges(self) -> list[GraphEdge]:
        """Return all edges. Empty unless include_edges=True was used."""
        return list(self._edges)

    def edges_from(self, src_id: int) -> list[GraphEdge]:
        """Return all edges originating from a given node."""
        return [e for e in self._edges if e.src_id == src_id]

    def edges_to(self, dst_id: int) -> list[GraphEdge]:
        """Return all edges pointing to a given node."""
        return [e for e in self._edges if e.dst_id == dst_id]

    def total_size_bytes(self) -> int:
        """Total bytes across all tracked nodes (alive + dead)."""
        return sum(n.size_bytes for n in self._nodes.values())

    def alive_size_bytes(self) -> int:
        """Total bytes across alive nodes only."""
        return sum(n.size_bytes for n in self._nodes.values() if n.alive)

    # ------------------------------------------------------------------
    # Summary + export
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """High-level stats about the graph — useful for logging and tests."""
        nodes = list(self._nodes.values())
        alive = [n for n in nodes if n.alive]
        dead = [n for n in nodes if not n.alive]

        type_counts: dict[str, int] = {}
        for n in nodes:
            type_counts[n.object_type] = type_counts.get(n.object_type, 0) + 1

        lifetimes = [n.lifetime_ns() for n in dead if n.lifetime_ns() is not None]

        return {
            "total_nodes": len(nodes),
            "alive_nodes": len(alive),
            "dead_nodes": len(dead),
            "total_edges": len(self._edges),
            "total_size_bytes": self.total_size_bytes(),
            "alive_size_bytes": self.alive_size_bytes(),
            "type_counts": type_counts,
            "avg_lifetime_ns": (
                sum(lifetimes) // len(lifetimes) if lifetimes else None
            ),
        }

    def to_dict(self) -> dict:
        """
        Export the full graph as a JSON-serializable dict.
        Suitable for embedding in a ProfilingSession or writing to disk.
        """
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "summary": self.summary(),
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self):
        s = self.summary()
        return (
            f"ObjectGraph(nodes={s['total_nodes']}, "
            f"alive={s['alive_nodes']}, "
            f"edges={s['total_edges']}, "
            f"size={s['total_size_bytes']}B)"
        )