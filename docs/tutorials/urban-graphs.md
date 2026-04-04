# Heterogeneous Urban Graphs

This tutorial walks through constructing and visualising an `UrbanGraph`: a rich heterogeneous graph encoding four spatial node types and their inter-layer topology. Access shapefile here [Shapefile](https://figshare.com/ndownloader/files/42515773)

| Node type | Geometry | Typical count |
|-----------|----------|---------------|
| Urban plots | Polygon | City blocks / land parcels |
| Buildings | Polygon | Individual footprints |
| Streets | Point (midpoint) | Road segments |
| Intersections | Point | Street junctions |

---

## Setup

```python
from urbanity import Map
from urbanity.visualisation import (
    plot_graph,
    plot_urban_graph_overview,
    plot_urban_graph_edges,
    plot_buildings,
    plot_node_attribute,
    plot_street_network,
)
import geopandas as gpd
import matplotlib.pyplot as plt

m = Map(country='Singapore')
m.add_polygon_boundary('path_to_shapefile',
                        layer_name='Downtown Core')
print("Area (km²):", round(
    m.polygon_bounds.to_crs(m.polygon_bounds.estimate_utm_crs()).area.values[0] / 1e6, 3
))
```

---

## Constructing the Urban Graph

```python
urban_graph = m.get_urban_graph(
    bandwidth=100,              # buffer distance (m) beyond the boundary
    minimum_area=30,            # filter plots smaller than 30 m²
    network_type='driving',
    population_layer='meta',    # use Meta 30-m resolution population data
)
```

**What `get_urban_graph()` returns:**

An `UrbanGraph` object with:
- `urban_graph.geo_store` — dict of node `GeoDataFrame`s keyed by type
- `urban_graph.edge_store` — dict of edge arrays (populated after `initialize_edges()`)

```python
# Node counts by type
for layer, gdf in urban_graph.geo_store.items():
    if hasattr(gdf, '__len__'):
        print(f"{layer:>14s}: {len(gdf):>6,} features")
```

---

## Initialising Edges

Before visualising edges or exporting to graph-ML, call `initialize_edges()`:

```python
# 3 nearest-neighbour building edges, max 100 m distance
urban_graph.initialize_edges(building_neighbours='knn', knn=3, distance=100)

# Inspect created edge types
for edge_type, arr in urban_graph.edge_store.items():
    print(f"{edge_type}: {arr.shape}")
```

---

## Four-Panel Overview

```python
fig = plot_urban_graph_overview(
    urban_graph.geo_store,
    title='Tanjong Pagar — UrbanGraph Layers',
    dark_mode=True,
)
plt.show()
```

---

## Edge Visualisation

```python
fig = plot_urban_graph_edges(
    urban_graph.geo_store,
    urban_graph.edge_store,
    edge_types=["plot_to_plot"],   # choose which edge type to show
)
plt.show()
```

Available edge types after `initialize_edges()`:

- `plot_to_plot` — plots sharing a street boundary
- `building_to_building` — KNN building neighbours
- `building_to_street` — nearest street to each building
- `intersection_to_street` — streets at each intersection
- `building_in_plot` — buildings contained within plots

---

## Building Footprint Visualisation

```python
fig = plot_buildings(
    urban_graph.geo_store['building'],
    boundary=urban_graph.geo_store['boundary'],
    title='Building Footprints — Tanjong Pagar',
    dark_mode=True,
)
plt.show()

# Colour by footprint area
fig = plot_buildings(
    urban_graph.geo_store['building'],
    colname='bid_area',
    cmap='viridis',
    boundary=urban_graph.geo_store['boundary'],
    title='Building Footprint Area (m²)',
    dark_mode=True,
)
plt.show()
```

---

## Urban Plot Attribute Map

```python
numeric_cols = (urban_graph.geo_store['plot']
                .select_dtypes(include='number').columns.tolist())
print("Numeric plot columns:", numeric_cols[:10])

fig = plot_node_attribute(
    urban_graph.geo_store['plot'],
    colname=numeric_cols[0],
    geometry_type='polygon',
    cmap='RdYlGn',
    boundary=urban_graph.geo_store['boundary'],
    title=f'{numeric_cols[0]}',
    dark_mode=True,
)
plt.show()
```

---

## Interactive 3-D View with PyDeck

`plot_graph()` renders an interactive Deck.gl 3-D visualisation inside Jupyter:

```python
# Building-centric view with height extrusion
plot_graph(
    urban_graph.geo_store,
    urban_graph.edge_store,
    node_type='building',
    colname='bid_height',
)
```

Highlight a single node and all its cross-layer neighbours:

```python
plot_graph(
    urban_graph.geo_store,
    urban_graph.edge_store,
    node_type='plot',
    node_id=0,              # highlights plot #0 in red
)
```

---

## Export to PyTorch Geometric

```python
urban_graph.to_pyg_graph()
```

See the full [Graph ML tutorial](graph-ml.md) for a complete training example.

---

## Save and Load

```python
# Save all node GeoDataFrames and edge arrays to a ZIP
urban_graph.save_graph('./data/graph.zip')

# Load in a future session
from urbanity.data_class import UrbanGraph

new_graph = UrbanGraph()
new_graph.load_graph('./data/graph.zip')
```

!!! tip "Save before initialising edges"
    Edge connectivity is cheap to recompute but the attribute-rich node GeoDataFrames are not. Save *before* calling `initialize_edges()`.

---

## Summary

| Task | Function |
|------|---------|
| Build heterogeneous graph | `m.get_urban_graph()` |
| Initialise edges | `urban_graph.initialize_edges(knn=3)` |
| Four-panel overview | `plot_urban_graph_overview(objects)` |
| Edge visualisation | `plot_urban_graph_edges(objects, edges)` |
| Building footprint map | `plot_buildings(objects['building'])` |
| Any node × attribute map | `plot_node_attribute(gdf, colname=...)` |
| Interactive 3-D view | `plot_graph(objects, edges, node_type=...)` |
| Export to PyG | `urban_graph.to_pyg_graph()` |
| Save / load | `save_graph(path)` / `load_graph(path)` |
