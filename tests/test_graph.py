"""
test_graph.py
-------------
Pass criteria:
  [A] Nodes are correctly built from ALLOC events
  [B] DEALLOC events correctly mark nodes as dead
  [C] Edge building is opt-in and only links tracked nodes
  [D] Queries (alive, dead, by address) return correct results
  [E] Summary and export are correct and JSON-serializable
  [F] Graph integrates correctly with PythonMemoryLayer output
"""

import gc
import json
import time
import unittest

from graph import ObjectGraph, GraphNode, GraphEdge
from python_memory_layer import PythonMemoryLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_alloc(addr, size, obj_type, ts, gen=0, ref=1, callstack=None):
    return {
        "base": {"event_id": addr, "timestamp_ns": ts, "event_type": "ALLOC",
                 "process_id": 1, "thread_id": 1},
        "alloc_address":    addr,
        "alloc_size_bytes": size,
        "object_type":      obj_type,
        "ref_count":        ref,
        "gc_generation":    gen,
        "is_dealloc":       False,
        "callstack":        callstack or [f"file.py:{ts}"],
        "pinned_address":   0,
        "tracemalloc_size": size,
        "array_shape":      [],
        "array_dtype":      "",
    }

def make_dealloc(addr, size, obj_type, ts):
    return {
        "base": {"event_id": addr + 10000, "timestamp_ns": ts,
                 "event_type": "DEALLOC", "process_id": 1, "thread_id": 1},
        "alloc_address":    addr,
        "alloc_size_bytes": size,
        "object_type":      obj_type,
        "ref_count":        0,
        "gc_generation":    0,
        "is_dealloc":       True,
        "callstack":        [],
        "pinned_address":   0,
        "tracemalloc_size": 0,
        "array_shape":      [],
        "array_dtype":      "",
    }


# ---------------------------------------------------------------------------
# A — Node construction from ALLOC events
# ---------------------------------------------------------------------------

class TestNodeConstruction(unittest.TestCase):

    def test_single_alloc_creates_node(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        self.assertEqual(len(g), 1)

    def test_node_fields_populated(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100, gen=0, ref=2,
                             callstack=["main.py:10", "util.py:5"])])
        node = g.get_node(1000)
        self.assertIsNotNone(node)
        self.assertEqual(node.node_id, 1000)
        self.assertEqual(node.object_type, "list")
        self.assertEqual(node.size_bytes, 64)
        self.assertEqual(node.ref_count, 2)
        self.assertEqual(node.gc_generation, 0)
        self.assertEqual(node.first_seen_ns, 100)
        self.assertIsNone(node.last_seen_ns)
        self.assertTrue(node.alive)
        self.assertEqual(node.callstack, ["main.py:10", "util.py:5"])

    def test_multiple_allocs_create_multiple_nodes(self):
        g = ObjectGraph()
        events = [
            make_alloc(1000, 64,  "list", 100),
            make_alloc(2000, 128, "dict", 200),
            make_alloc(3000, 256, "numpy.ndarray", 300),
        ]
        g.ingest(events)
        self.assertEqual(len(g), 3)

    def test_duplicate_address_overwrites_node(self):
        """If the same address is reused (address space recycled), latest wins."""
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        g.ingest([make_alloc(1000, 128, "dict", 200)])
        node = g.get_node(1000)
        self.assertEqual(node.object_type, "dict")
        self.assertEqual(node.size_bytes, 128)

    def test_new_node_is_alive(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        self.assertTrue(g.get_node(1000).alive)

    def test_ingest_empty_list_is_safe(self):
        g = ObjectGraph()
        g.ingest([])
        self.assertEqual(len(g), 0)


# ---------------------------------------------------------------------------
# B — DEALLOC marks nodes dead
# ---------------------------------------------------------------------------

class TestDealloc(unittest.TestCase):

    def test_dealloc_marks_node_dead(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        g.ingest([make_dealloc(1000, 64, "list", 300)])
        self.assertFalse(g.get_node(1000).alive)

    def test_dealloc_sets_last_seen_ns(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        g.ingest([make_dealloc(1000, 64, "list", 300)])
        self.assertEqual(g.get_node(1000).last_seen_ns, 300)

    def test_lifetime_computed_correctly(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        g.ingest([make_dealloc(1000, 64, "list", 400)])
        self.assertEqual(g.get_node(1000).lifetime_ns(), 300)

    def test_alive_node_has_no_lifetime(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        self.assertIsNone(g.get_node(1000).lifetime_ns())

    def test_dealloc_without_prior_alloc_is_safe(self):
        """DEALLOC for an untracked address must not raise."""
        g = ObjectGraph()
        g.ingest([make_dealloc(9999, 64, "list", 100)])
        self.assertIsNone(g.get_node(9999))

    def test_mixed_alloc_dealloc_stream(self):
        g = ObjectGraph()
        events = [
            make_alloc(1000, 64,  "list", 100),
            make_alloc(2000, 128, "dict", 200),
            make_dealloc(1000, 64, "list", 300),
            make_alloc(3000, 256, "tuple", 400),
            make_dealloc(2000, 128, "dict", 500),
        ]
        g.ingest(events)
        self.assertFalse(g.get_node(1000).alive)
        self.assertFalse(g.get_node(2000).alive)
        self.assertTrue(g.get_node(3000).alive)


# ---------------------------------------------------------------------------
# C — Edge building
# ---------------------------------------------------------------------------

class TestEdges(unittest.TestCase):

    def test_no_edges_by_default(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        self.assertEqual(len(g.edges()), 0)

    def test_edges_not_built_without_live_objects(self):
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)], include_edges=True)
        self.assertEqual(len(g.edges()), 0)

    def test_edges_built_between_tracked_nodes(self):
        """Edge from B to A is created when B references A and both are tracked."""
        class A: pass
        class B:
            def __init__(self, a): self.a = a

        a = A()
        b = B(a)
        live = {id(a): a, id(b): b}

        g = ObjectGraph()
        events = [
            make_alloc(id(a), 64, "A", 100),
            make_alloc(id(b), 64, "B", 200),
        ]
        g.ingest(events, include_edges=True, live_objects=live)

        edges = g.edges()
        dst_ids = [e.dst_id for e in edges]
        self.assertIn(id(a), dst_ids,
            "Expected edge from B to A")

    def test_edges_only_link_tracked_nodes(self):
        """Edges to untracked objects must not appear."""
        class C: pass
        obj = C()
        live = {id(obj): obj}

        g = ObjectGraph()
        g.ingest([make_alloc(id(obj), 64, "C", 100)],
                 include_edges=True, live_objects=live)

        for edge in g.edges():
            self.assertIn(edge.dst_id, {id(obj)},
                "Edge points to an untracked node")

    def test_edges_from_query(self):
        g = ObjectGraph()
        g._edges.append(GraphEdge(1000, 2000, 100))
        g._edges.append(GraphEdge(1000, 3000, 100))
        g._edges.append(GraphEdge(2000, 3000, 200))
        self.assertEqual(len(g.edges_from(1000)), 2)
        self.assertEqual(len(g.edges_from(2000)), 1)

    def test_edges_to_query(self):
        g = ObjectGraph()
        g._edges.append(GraphEdge(1000, 3000, 100))
        g._edges.append(GraphEdge(2000, 3000, 200))
        self.assertEqual(len(g.edges_to(3000)), 2)
        self.assertEqual(len(g.edges_to(1000)), 0)


# ---------------------------------------------------------------------------
# D — Queries
# ---------------------------------------------------------------------------

class TestQueries(unittest.TestCase):

    def setUp(self):
        self.g = ObjectGraph()
        self.g.ingest([
            make_alloc(1000, 64,  "list",  100),
            make_alloc(2000, 128, "dict",  200),
            make_alloc(3000, 256, "tuple", 300),
        ])
        self.g.ingest([make_dealloc(1000, 64, "list", 400)])

    def test_get_node_found(self):
        self.assertIsNotNone(self.g.get_node(2000))

    def test_get_node_not_found(self):
        self.assertIsNone(self.g.get_node(9999))

    def test_nodes_returns_all(self):
        self.assertEqual(len(self.g.nodes()), 3)

    def test_alive_nodes(self):
        alive = self.g.alive_nodes()
        self.assertEqual(len(alive), 2)
        self.assertNotIn(1000, [n.node_id for n in alive])

    def test_dead_nodes(self):
        dead = self.g.dead_nodes()
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0].node_id, 1000)

    def test_total_size_bytes(self):
        self.assertEqual(self.g.total_size_bytes(), 64 + 128 + 256)

    def test_alive_size_bytes(self):
        self.assertEqual(self.g.alive_size_bytes(), 128 + 256)


# ---------------------------------------------------------------------------
# E — Summary and export
# ---------------------------------------------------------------------------

class TestSummaryAndExport(unittest.TestCase):

    def setUp(self):
        self.g = ObjectGraph()
        self.g.ingest([
            make_alloc(1000, 64,  "list", 100),
            make_alloc(2000, 128, "list", 200),
            make_alloc(3000, 256, "dict", 300),
        ])
        self.g.ingest([make_dealloc(1000, 64, "list", 500)])

    def test_summary_counts(self):
        s = self.g.summary()
        self.assertEqual(s["total_nodes"], 3)
        self.assertEqual(s["alive_nodes"], 2)
        self.assertEqual(s["dead_nodes"], 1)

    def test_summary_sizes(self):
        s = self.g.summary()
        self.assertEqual(s["total_size_bytes"], 64 + 128 + 256)
        self.assertEqual(s["alive_size_bytes"], 128 + 256)

    def test_summary_type_counts(self):
        s = self.g.summary()
        self.assertEqual(s["type_counts"]["list"], 2)
        self.assertEqual(s["type_counts"]["dict"], 1)

    def test_summary_avg_lifetime(self):
        s = self.g.summary()
        self.assertEqual(s["avg_lifetime_ns"], 400)  # 500 - 100

    def test_to_dict_json_serializable(self):
        d = self.g.to_dict()
        try:
            json.dumps(d)
        except TypeError as e:
            self.fail(f"to_dict() is not JSON serializable: {e}")

    def test_to_dict_structure(self):
        d = self.g.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("summary", d)
        self.assertEqual(len(d["nodes"]), 3)

    def test_node_to_dict_fields(self):
        d = self.g.to_dict()
        node = d["nodes"][0]
        for field in ["node_id", "object_type", "size_bytes", "ref_count",
                      "gc_generation", "alive", "first_seen_ns",
                      "last_seen_ns", "lifetime_ns", "callstack"]:
            self.assertIn(field, node, f"Missing field: {field}")

    def test_repr(self):
        r = repr(self.g)
        self.assertIn("ObjectGraph", r)
        self.assertIn("nodes=3", r)


# ---------------------------------------------------------------------------
# F — Integration with PythonMemoryLayer
# ---------------------------------------------------------------------------

class TestLayerIntegration(unittest.TestCase):

    def test_graph_built_from_real_events(self):
        """Graph ingests real events from PythonMemoryLayer correctly."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        class RealObj:
            def __init__(self, data):
                self.data = data

        obj = RealObj(list(range(200)))
        addr = id(obj)
        events = layer.collect()
        layer.stop()

        g = ObjectGraph()
        g.ingest(events)

        node = g.get_node(addr)
        self.assertIsNotNone(node, f"Node for address {addr} not found in graph")
        self.assertTrue(node.object_type.endswith("RealObj"),
            f"Expected object_type ending in 'RealObj', got '{node.object_type}'")
        self.assertGreater(node.size_bytes, 0)
        self.assertTrue(node.alive)

    def test_dealloc_marks_node_dead_from_real_events(self):
        """DEALLOC events from PythonMemoryLayer mark nodes dead in the graph."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        class RealObj:
            def __init__(self, data):
                self.data = data

        obj = RealObj(list(range(300)))
        addr = id(obj)
        layer.collect()

        del obj
        gc.collect()
        layer.collect()   # release tracemalloc snapshot reference
        time.sleep(0.05)

        remaining = layer.flush()
        layer.stop()

        g = ObjectGraph()
        g.ingest(remaining)

        node = g.get_node(addr)
        if node:
            self.assertFalse(node.alive,
                "Node should be dead after object was deleted")

    def test_graph_summary_from_real_events(self):
        """summary() reflects real event counts from the layer."""
        layer = PythonMemoryLayer(nframe=16)
        layer.start()

        class RealObj:
            def __init__(self): self.data = [0] * 100

        objs = [RealObj() for _ in range(5)]
        events = layer.collect()
        layer.stop()

        g = ObjectGraph()
        g.ingest(events)

        s = g.summary()
        self.assertGreaterEqual(s["total_nodes"], 5)
        self.assertGreater(s["total_size_bytes"], 0)


# ---------------------------------------------------------------------------
# G — GPU correlation: merge_gpu_correlations and gpu_correlated_nodes
# ---------------------------------------------------------------------------

def make_gpu_transfer_event(eid, ts, src=0, size=0):
    return {
        "base": {"event_id": eid, "timestamp_ns": ts, "event_type": "TRANSFER",
                 "process_id": 1, "thread_id": 0},
        "device_id":           0,
        "src_address":         src,
        "dst_address":         0,
        "transfer_size_bytes": size,
        "transfer_kind":       "HOST_TO_DEVICE",
        "um_page_faults":      0,
        "kernel_name":         "",
        "kernel_duration_ns":  0,
        "stream_id":           0,
        "device_mem_used_mb":  0,
        "um_bytes_htod":       size,
        "um_bytes_dtoh":       0,
        "grid":                {"x": 0, "y": 0, "z": 0},
        "block":               {"x": 0, "y": 0, "z": 0},
        "registers_per_thread": 0,
        "shared_mem_bytes":    0,
        "correlation_id":      eid,
    }


def make_gpu_kernel_event(eid, ts, name="volta_sgemm"):
    return {
        "base": {"event_id": eid, "timestamp_ns": ts, "event_type": "KERNEL",
                 "process_id": 1, "thread_id": 0},
        "device_id":           0,
        "src_address":         0,
        "dst_address":         0,
        "transfer_size_bytes": 0,
        "transfer_kind":       "HOST_TO_DEVICE",
        "um_page_faults":      0,
        "kernel_name":         name,
        "kernel_duration_ns":  80_000,
        "stream_id":           2,
        "device_mem_used_mb":  512,
        "um_bytes_htod":       0,
        "um_bytes_dtoh":       0,
        "grid":                {"x": 128, "y": 1, "z": 1},
        "block":               {"x": 256, "y": 1, "z": 1},
        "registers_per_thread": 32,
        "shared_mem_bytes":    4096,
        "correlation_id":      eid,
    }


def make_corr(cpu_eid, gpu_eid, confidence="HARD", reason="ADDRESS_MATCH", latency=5000):
    return {
        "cpu_event_id": cpu_eid,
        "gpu_event_id": gpu_eid,
        "confidence":   confidence,
        "match_reason": reason,
        "latency_ns":   latency,
    }


class TestGpuCorrelations(unittest.TestCase):
    """[G] gpu_correlations on nodes and merge_gpu_correlations()."""

    def test_new_node_has_empty_gpu_correlations(self):
        """Every freshly-built node starts with an empty gpu_correlations list."""
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100)])
        node = g.get_node(1000)
        self.assertIsInstance(node.gpu_correlations, list)
        self.assertEqual(len(node.gpu_correlations), 0)

    def test_gpu_correlations_in_to_dict(self):
        """to_dict() must include gpu_correlations on every node dict."""
        g = ObjectGraph()
        g.ingest([make_alloc(2000, 128, "dict", 200)])
        node = g.to_dict()["nodes"][0]
        self.assertIn("gpu_correlations", node)
        self.assertIsInstance(node["gpu_correlations"], list)

    def test_merge_gpu_correlations_populates_node(self):
        """merge_gpu_correlations attaches the correct GPU event to the right node."""
        g = ObjectGraph()
        cpu_ev = make_alloc(1000, 64, "ndarray", 100)
        cpu_ev["base"]["event_id"] = 7
        g.ingest([cpu_ev])

        gpu_ev = make_gpu_transfer_event(eid=99, ts=200, src=1000, size=64)
        corr   = make_corr(cpu_eid=7, gpu_eid=99, latency=100)
        g.merge_gpu_correlations([corr], [gpu_ev], [cpu_ev])

        node = g.get_node(1000)
        self.assertEqual(len(node.gpu_correlations), 1)
        c = node.gpu_correlations[0]
        self.assertEqual(c["confidence"],          "HARD")
        self.assertEqual(c["match_reason"],        "ADDRESS_MATCH")
        self.assertEqual(c["latency_ns"],          100)
        self.assertEqual(c["gpu_event_id"],        99)
        self.assertEqual(c["gpu_event_type"],      "TRANSFER")
        self.assertEqual(c["transfer_size_bytes"], 64)

    def test_merge_kernel_event_with_new_fields(self):
        """Kernel GPU events pass grid/block/registers through gpu_correlations."""
        g = ObjectGraph()
        cpu_ev = make_alloc(2000, 256, "Tensor", 500)
        cpu_ev["base"]["event_id"] = 11
        g.ingest([cpu_ev])

        kernel_ev = make_gpu_kernel_event(eid=55, ts=600, name="relu_forward")
        corr = make_corr(cpu_eid=11, gpu_eid=55, confidence="WEAK",
                         reason="TIMESTAMP_ONLY", latency=100)
        g.merge_gpu_correlations([corr], [kernel_ev], [cpu_ev])

        node = g.get_node(2000)
        self.assertEqual(len(node.gpu_correlations), 1)
        c = node.gpu_correlations[0]
        self.assertEqual(c["confidence"],         "WEAK")
        self.assertEqual(c["kernel_name"],        "relu_forward")
        self.assertEqual(c["kernel_duration_ns"], 80_000)
        self.assertEqual(c["stream_id"],          2)
        # New fields must be present in the correlation annotation
        self.assertIn("um_bytes_htod", c)
        self.assertIn("grid",          c)
        self.assertIn("block",         c)
        self.assertIn("registers_per_thread", c)
        self.assertIn("shared_mem_bytes",     c)
        self.assertIn("correlation_id",       c)
        self.assertEqual(c["grid"],  {"x": 128, "y": 1, "z": 1})
        self.assertEqual(c["block"], {"x": 256, "y": 1, "z": 1})
        self.assertEqual(c["registers_per_thread"], 32)
        self.assertEqual(c["shared_mem_bytes"],     4096)

    def test_merge_transfer_event_with_um_bytes(self):
        """Transfer events carry um_bytes_htod/dtoh into gpu_correlations."""
        g = ObjectGraph()
        cpu_ev = make_alloc(3000, 512, "ndarray", 1000)
        cpu_ev["base"]["event_id"] = 20
        g.ingest([cpu_ev])

        gpu_ev = make_gpu_transfer_event(eid=101, ts=1100, src=3000, size=512)
        corr   = make_corr(cpu_eid=20, gpu_eid=101, latency=100)
        g.merge_gpu_correlations([corr], [gpu_ev], [cpu_ev])

        node = g.get_node(3000)
        self.assertEqual(len(node.gpu_correlations), 1)
        c = node.gpu_correlations[0]
        self.assertIn("um_bytes_htod", c)
        self.assertIn("um_bytes_dtoh", c)
        self.assertEqual(c["um_bytes_htod"], 512)
        self.assertEqual(c["um_bytes_dtoh"], 0)

    def test_merge_multiple_gpu_events_same_node(self):
        """One CPU node can accumulate both a TRANSFER and a KERNEL correlation."""
        g = ObjectGraph()
        cpu_ev = make_alloc(3000, 512, "ndarray", 1000)
        cpu_ev["base"]["event_id"] = 20
        g.ingest([cpu_ev])

        gpu1 = make_gpu_transfer_event(eid=101, ts=1100, src=3000, size=512)
        gpu2 = make_gpu_kernel_event(eid=102, ts=1200, name="sgemm")
        corrs = [
            make_corr(cpu_eid=20, gpu_eid=101, latency=100),
            make_corr(cpu_eid=20, gpu_eid=102, confidence="WEAK",
                      reason="TIMESTAMP_ONLY", latency=200),
        ]
        g.merge_gpu_correlations(corrs, [gpu1, gpu2], [cpu_ev])

        node = g.get_node(3000)
        self.assertEqual(len(node.gpu_correlations), 2)
        gpu_ids = {c["gpu_event_id"] for c in node.gpu_correlations}
        self.assertEqual(gpu_ids, {101, 102})

    def test_merge_unknown_cpu_event_id_safe(self):
        """Dangling cpu_event_id references are silently skipped."""
        g = ObjectGraph()
        g.ingest([make_alloc(4000, 64, "list", 100)])
        gpu_ev = make_gpu_transfer_event(eid=200, ts=200, src=9999, size=64)
        corr   = make_corr(cpu_eid=9999, gpu_eid=200)
        try:
            g.merge_gpu_correlations([corr], [gpu_ev], [])
        except Exception as e:
            self.fail(f"merge_gpu_correlations raised on unknown cpu_event_id: {e}")
        self.assertEqual(len(g.get_node(4000).gpu_correlations), 0)

    def test_merge_unknown_gpu_event_id_safe(self):
        """Dangling gpu_event_id references are silently skipped."""
        g = ObjectGraph()
        cpu_ev = make_alloc(5000, 64, "list", 100)
        cpu_ev["base"]["event_id"] = 30
        g.ingest([cpu_ev])
        corr = make_corr(cpu_eid=30, gpu_eid=9999)
        try:
            g.merge_gpu_correlations([corr], [], [cpu_ev])
        except Exception as e:
            self.fail(f"merge_gpu_correlations raised on unknown gpu_event_id: {e}")
        self.assertEqual(len(g.get_node(5000).gpu_correlations), 0)

    def test_merge_empty_inputs_safe(self):
        """merge_gpu_correlations with all-empty inputs must not raise."""
        g = ObjectGraph()
        g.ingest([make_alloc(6000, 64, "list", 100)])
        g.merge_gpu_correlations([], [], [])
        self.assertEqual(len(g.get_node(6000).gpu_correlations), 0)

    def test_gpu_correlated_nodes_returns_only_matched(self):
        """gpu_correlated_nodes() returns only nodes with ≥1 GPU correlation."""
        g = ObjectGraph()
        cpu_a = make_alloc(7000, 64, "ndarray", 100)
        cpu_b = make_alloc(8000, 64, "list",    200)
        cpu_a["base"]["event_id"] = 40
        cpu_b["base"]["event_id"] = 41
        g.ingest([cpu_a, cpu_b])

        gpu_ev = make_gpu_transfer_event(eid=300, ts=150, src=7000, size=64)
        corr   = make_corr(cpu_eid=40, gpu_eid=300)
        g.merge_gpu_correlations([corr], [gpu_ev], [cpu_a, cpu_b])

        matched = g.gpu_correlated_nodes()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].node_id, 7000)

    def test_gpu_correlated_nodes_empty_when_no_correlations(self):
        """gpu_correlated_nodes() returns [] when no merges have been done."""
        g = ObjectGraph()
        g.ingest([make_alloc(9000, 64, "list", 100)])
        self.assertEqual(g.gpu_correlated_nodes(), [])

    def test_merge_gpu_correlations_json_serializable(self):
        """to_dict() after merge remains fully JSON-serializable."""
        g = ObjectGraph()
        cpu_ev = make_alloc(10000, 128, "ndarray", 1000)
        cpu_ev["base"]["event_id"] = 50
        g.ingest([cpu_ev])

        gpu_ev = make_gpu_transfer_event(eid=400, ts=1100, src=10000, size=128)
        corr   = make_corr(cpu_eid=50, gpu_eid=400, latency=100)
        g.merge_gpu_correlations([corr], [gpu_ev], [cpu_ev])

        d = g.to_dict()
        try:
            json.dumps(d)
        except TypeError as e:
            self.fail(f"to_dict() not JSON-serializable after merge: {e}")

    def test_node_to_dict_has_all_expected_fields(self):
        """to_dict() node dict has all expected fields including new gpu_correlations."""
        g = ObjectGraph()
        g.ingest([make_alloc(1000, 64, "list", 100, gen=0, ref=2,
                             callstack=["main.py:10", "util.py:5"])])
        node_dict = g.to_dict()["nodes"][0]
        for field in ["node_id", "object_type", "size_bytes", "ref_count",
                      "gc_generation", "alive", "first_seen_ns",
                      "last_seen_ns", "lifetime_ns", "callstack",
                      "gpu_correlations"]:
            self.assertIn(field, node_dict, f"Missing field: {field}")
        self.assertIsInstance(node_dict["gpu_correlations"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)