# `urbanity.topology`

High-performance network centrality computation using [NetworKit](https://networkit.github.io/), and utilities for merging NetworkX graph attributes into node GeoDataFrames.

```python
from urbanity.topology import compute_centrality, merge_nx_attr, merge_nx_property
```

---

## `compute_centrality`

```python
compute_centrality(G, G_nodes, function, colname: str, T_nodes) -> GeoDataFrame
```

Computes a centrality measure for all nodes in the network using NetworKit's fast C++ backend, then merges the result back into the node GeoDataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `G` | `nx.MultiDiGraph` | The street network graph. |
| `G_nodes` | `GeoDataFrame` | Node attribute GeoDataFrame to receive the centrality column. |
| `function` | callable | A NetworKit centrality class (e.g. `networkit.centrality.Betweenness`). |
| `colname` | `str` | Name of the output column added to `G_nodes`. |
| `T_nodes` | list | List of target node IDs for which centrality is computed. |

**Returns:** `GeoDataFrame` with a new column `colname` containing centrality values.

**Example:**

```python
import networkit as nk
from urbanity.topology import compute_centrality

nodes = compute_centrality(
    G, nodes,
    function=nk.centrality.Betweenness,
    colname="betweenness_centrality",
    T_nodes=list(G.nodes()),
)
```

!!! note
    NetworKit converts the NetworkX graph internally. For very large networks (> 1M nodes), this step can take several minutes but is significantly faster than NetworkX's pure Python equivalents.

---

## `merge_nx_attr`

```python
merge_nx_attr(G, G_nodes, nxfunction, colname: str, T_nodes, temporal: bool = False) -> GeoDataFrame
```

Computes a NetworkX graph attribute function and merges the resulting values into the node GeoDataFrame.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `G` | `nx.MultiDiGraph` | required | Urban network graph. |
| `G_nodes` | `GeoDataFrame` | required | Node GeoDataFrame to augment. |
| `nxfunction` | callable | required | NetworkX function that returns a dict of `{node_id: value}`. |
| `colname` | `str` | required | Output column name. |
| `T_nodes` | list | required | Target node IDs. |
| `temporal` | `bool` | `False` | If `True`, appends a year suffix to `colname`. |

**Returns:** Augmented `GeoDataFrame`.

**Example:**

```python
import networkx as nx
from urbanity.topology import merge_nx_attr

nodes = merge_nx_attr(G, nodes, nx.pagerank, "pagerank", list(G.nodes()))
```

---

## `merge_nx_property`

```python
merge_nx_property(G_nodes, nxproperty, colname: str, T_nodes, temporal: bool = False) -> GeoDataFrame
```

Merges a pre-computed NetworkX graph property dictionary into the node GeoDataFrame. Use this when the property dict has already been computed (e.g. `nx.betweenness_centrality(G)`) to avoid recomputation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `G_nodes` | `GeoDataFrame` | required | Node GeoDataFrame. |
| `nxproperty` | `dict` | required | Dict of `{node_id: value}` (output of a NetworkX function). |
| `colname` | `str` | required | Output column name. |
| `T_nodes` | list | required | Target node IDs to include. |
| `temporal` | `bool` | `False` | If `True`, appends a year suffix to `colname`. |

**Returns:** Augmented `GeoDataFrame`.
