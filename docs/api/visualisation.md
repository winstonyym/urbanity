# `urbanity.visualisation`

Plotting and visualisation functions, all importable directly from the `urbanity` namespace.

```python
from urbanity import (
    plot_graph,
    plot_street_network,
    plot_buildings,
    plot_urban_graph_overview,
    plot_node_attribute,
    plot_network_centrality,
)
```

---

## `plot_graph`

```python
plot_graph(
    objects,
    connections=None,
    connection_type: str = "street",
    node_type: str = "intersection",
    node_idx=None,
    figsize=(12, 12),
    title: str = "",
    **kwargs,
)
```

General-purpose graph plot. Renders nodes and edges from an `UrbanGraph` object with sensible defaults.

| Parameter | Type | Description |
|-----------|------|-------------|
| `objects` | `UrbanGraph` | The graph to plot. |
| `connections` | `GeoDataFrame` | Edge GeoDataFrame (uses `objects.edges` if `None`). |
| `connection_type` | `str` | Edge type key for heterogeneous graphs. |
| `node_type` | `str` | Node type to render (`"intersection"`, `"building"`, …). |
| `figsize` | `tuple` | Matplotlib figure size. |
| `title` | `str` | Plot title. |

---

## `plot_street_network`

```python
plot_street_network(
    G,
    edge_color: str = "#999999",
    node_color: str = "#333333",
    node_size: float = 5.0,
    figsize=(14, 14),
    bgcolor: str = "white",
    title: str = "",
    **kwargs,
)
```

Clean cartographic-style street network plot.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `G` | `UrbanGraph` | required | Network to render. |
| `edge_color` | `str` | `"#999999"` | Colour for edges. |
| `node_color` | `str` | `"#333333"` | Colour for nodes. |
| `node_size` | `float` | `5.0` | Node marker size. |
| `bgcolor` | `str` | `"white"` | Background colour. |

---

## `plot_buildings`

```python
plot_buildings(
    G,
    color_by: str = None,
    cmap: str = "YlOrRd",
    alpha: float = 0.7,
    figsize=(14, 14),
    title: str = "",
)
```

Plots building footprints, optionally coloured by a numeric attribute (e.g. height, energy use).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_by` | `str` | `None` | Column name to use for fill colour. If `None`, uses a single colour. |
| `cmap` | `str` | `"YlOrRd"` | Matplotlib colormap name. |
| `alpha` | `float` | `0.7` | Fill opacity. |

---

## `plot_urban_graph_overview`

```python
plot_urban_graph_overview(
    G,
    figsize=(16, 16),
    title: str = "",
)
```

Composite overview plot: street network, building footprints, and node context in a single figure. Useful for quick sanity checks.

---

## `plot_node_attribute`

```python
plot_node_attribute(
    G,
    attribute: str,
    cmap: str = "plasma",
    node_size: float = 10.0,
    figsize=(12, 12),
    title: str = "",
    legend: bool = True,
)
```

Colours every node by the value of a chosen attribute column, with a continuous colour gradient.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attribute` | `str` | required | Column name in `G.nodes` to visualise. |
| `cmap` | `str` | `"plasma"` | Matplotlib colormap. |
| `legend` | `bool` | `True` | Show a colour-bar legend. |

**Example:**

```python
from urbanity import plot_node_attribute
plot_node_attribute(G, attribute="building_density", cmap="viridis")
```

---

## `plot_network_centrality`

```python
plot_network_centrality(
    G,
    attribute: str = "betweenness_centrality",
    cmap: str = "magma",
    node_size: float = 8.0,
    figsize=(14, 14),
    title: str = "",
    legend: bool = True,
)
```

Renders centrality values (betweenness, closeness, degree…) as a node colour map — the higher the centrality, the brighter the node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attribute` | `str` | `"betweenness_centrality"` | Centrality column to plot. |
| `cmap` | `str` | `"magma"` | Matplotlib colormap. |

**Example:**

```python
from urbanity import plot_network_centrality
plot_network_centrality(G, attribute="closeness_centrality")
```
