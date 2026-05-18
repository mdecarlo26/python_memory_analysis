#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Trace Graph</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #wrap {{ display: flex; height: 100vh; }}
    #mynetwork {{ flex: 1; border-right: 1px solid #ddd; }}
    #panel {{ width: 420px; padding: 10px; overflow: auto; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    .hint {{ color: #666; font-size: 12px; }}
  </style>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
<div id="wrap">
  <div id="mynetwork"></div>
  <div id="panel">
    <h3>Node details</h3>
    <div class="hint">Click a node to inspect the underlying event record.</div>
    <pre id="details"></pre>
  </div>
</div>

<script>
const graph = __GRAPH_JSON__;

function groupStyle(group) {{
  // Use vis groups for consistent styling (no manual colors required)
  return group;
}}

const nodes = new vis.DataSet(graph.nodes.map(n => {{
  return {{
    id: n.id,
    label: n.label,
    group: groupStyle(n.group),
    title: n.label
  }};
}}));

const edges = new vis.DataSet(graph.edges.map(e => {{
  return {{
    id: e.id,
    from: e.from,
    to: e.to,
    arrows: e.kind === "corr" ? "to" : "",
    dashes: e.kind === "next" || e.kind === "mem_next"
  }};
}}));

const container = document.getElementById("mynetwork");
const data = {{ nodes, edges }};
const options = {{
  physics: {{
    stabilization: false
  }},
  interaction: {{
    hover: true
  }},
  groups: {{
    corr: {{ shape: "ellipse" }},
    cpu_api: {{ shape: "box" }},
    gpu_activity: {{ shape: "box" }},
    host_mem: {{ shape: "diamond" }},
    device_mem: {{ shape: "diamond" }},
    marker: {{ shape: "triangle" }},
    other: {{ shape: "dot" }}
  }}
}};

const network = new vis.Network(container, data, options);

const detailsEl = document.getElementById("details");
const nodeMap = new Map(graph.nodes.map(n => [n.id, n]));

network.on("click", function(params) {{
  if (!params.nodes || params.nodes.length === 0) return;
  const id = params.nodes[0];
  const node = nodeMap.get(id);
  if (!node) return;
  detailsEl.textContent = JSON.stringify(node.data, null, 2);
}});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Render graph.json into a self-contained HTML viewer.")
    ap.add_argument("--graph", required=True, help="graph.json produced by build_trace_graph.py")
    ap.add_argument("--out", default="graph.html", help="Output html")
    args = ap.parse_args()

    g = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(g))
    Path(args.out).write_text(html, encoding="utf-8")
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()