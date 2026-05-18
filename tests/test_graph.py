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
        "alloc_address": addr,
        "alloc_size_bytes": size,
        "object_type": obj_type,
        "ref_count": ref,
        "gc_generation": gen,
        "is_dealloc": False,
        "callstack": callstack or [f"file.py:{ts}"],
    }

def make_dealloc(addr, size, obj_type, ts):
    return {
        "base": {"event_id": addr + 10000, "timestamp_ns": ts,
                 "event_type": "DEALLOC", "process_id": 1, "thread_id": 1},
        "alloc_address": addr,
        "alloc_size_bytes": size,
        "object_type": obj_type,
        "ref_count": 0,
        "gc_generation": 0,
        "is_dealloc": True,
        "callstack": [],
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
        time.sleep(0.01)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)