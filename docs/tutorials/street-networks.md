# Street Network Analysis

This tutorial shows how to build a primal street network, inspect its attributes, compute centrality metrics, and visualise the results. Access shapefile here [Shapefile](https://figshare.com/ndownloader/files/42515773)

---

## Setup

```python
from urbanity import Map
from urbanity.visualisation import (
    plot_street_network,
    plot_network_centrality,
    plot_node_attribute,
)
import geopandas as gpd
import matplotlib.pyplot as plt

m = Map(country='Singapore')
m.add_polygon_boundary('path_to_shapefile')
print("Boundary set:", m.polygon_bounds.geometry.iloc[0].geom_type)
```

---

## Building the Street Network

`get_street_network()` works with no mandatory arguments once a boundary is set. Urbanity automatically:

- Queries Geofabrik and BBBike to find the smallest available OSM extract
- Downloads and caches it (subsequent calls skip the download)
- Constructs and enriches the NetworkX graph

```python
G, nodes, edges = m.get_street_network(
    bandwidth=100,          # buffer (m) to capture context beyond boundary
    network_type='driving',
    graph_attr=True,        # betweenness, closeness, degree centrality
    building_attr=False,    # set True to add building counts per node
    pop_attr=False,         # set True to add population counts per node
)
```

**Returns:**

- `G` — NetworkX `MultiDiGraph`
- `nodes` — node attribute `GeoDataFrame` (intersections)
- `edges` — edge attribute `GeoDataFrame` (street segments)

```python
print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
print("Node columns:", list(nodes.columns))
print("Edge columns:", list(edges.columns))
```

---

## Inspecting the Output

```python
nodes.head()
```

```python
edges.head()
```

Key edge columns include `length`, `street_num_buildings`, `street_mean_building_bid_complexity`, `PopSum`, `Men`, `Women`, `Elderly`, and street view indicator columns like `street_mean_GreenView`.

---

## Static Visualisation

`plot_street_network()` produces a high-quality matplotlib figure using a dark-mode theme by default.

```python
fig = plot_street_network(
    nodes, edges,
    title='Downtown Core Street Network',
    dark_mode=True,
)
plt.show()
```

Colour edges by any numeric attribute:

```python
fig = plot_street_network(
    nodes, edges,
    colname='length',       # colour by street segment length
    cmap='plasma',
    title='Street Segments by Length',
    dark_mode=True,
)
plt.show()
```

---

## Dual Network

The dual (or edge) network — common in Space Syntax — treats street segments as nodes and shared intersections as edges:

```python
dual_G, dual_nodes, dual_edges = m.get_street_network(
    location='Singapore',
    dual=True,
)

fig, ax = plt.subplots(figsize=(7, 7))
dual_nodes.plot(ax=ax, color='green', markersize=0.1)
dual_edges.plot(ax=ax, color='black', linewidth=0.1)
plt.show()
```

---

## Network Centrality

When `graph_attr=True`, betweenness and closeness centrality are computed via NetworKit and stored in the node GeoDataFrame.

```python
centrality_cols = [c for c in nodes.columns
                   if any(k in c for k in ['betweenness', 'closeness', 'degree'])]
print("Centrality columns:", centrality_cols)

fig = plot_network_centrality(
    nodes, edges,
    centrality_col=centrality_cols[0],
    title='Betweenness Centrality',
    dark_mode=True,
)
plt.show()
```

Colour nodes by any attribute:

```python
fig, ax = plt.subplots(figsize=(12, 12))
nodes.plot(
    column='Youth',
    ax=ax,
    cmap='turbo',
    markersize=5,
    legend=True,
    legend_kwds={'label': 'Youth Population Count'},
)
plt.show()
```

---

## Aggregate Statistics

Get area-level descriptive statistics without constructing a full network:

```python
# Single bounding box
subzone_gdf = m.get_aggregate_stats("Seattle", get_svi=True)
print(subzone_gdf.columns.tolist())
```

For multiple subzones, pass a shapefile via `add_polygon_boundary`:

```python
m4 = Map(country="United States", zoom=10)
m4.add_polygon_boundary('https://figshare.com/ndownloader/files/41053475')

# Aggregate for each of the 12 CRA subzones
aggregate_gdf = m4.get_aggregate_stats("Seattle", column='CRA_NAM', get_svi=True)
aggregate_gdf.head()
```

---

## Saving the Network

```python
nodes.to_file('output/tanjong_pagar_nodes.geojson', driver='GeoJSON')
edges.to_file('output/tanjong_pagar_edges.geojson', driver='GeoJSON')
```

---

## Summary

| Task | Code |
|------|------|
| Build street network | `G, nodes, edges = m.get_street_network()` |
| Static map | `plot_street_network(nodes, edges)` |
| Colour by attribute | `colname='length'` |
| Centrality map | `plot_network_centrality(nodes, edges, centrality_col=...)` |
| Dual graph | `get_street_network(dual=True)` |
| Area statistics | `m.get_aggregate_stats(location, column=...)` |

→ Next: [Urban Graphs](urban-graphs.md)
