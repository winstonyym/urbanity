# API Reference

Complete documentation for all public classes and functions in Urbanity, extracted from source.

---

## Modules

| Module | Description |
|--------|-------------|
| [`Map`](map.md) | Main entry point — interactive map, network builder, graph generator |
| [`UrbanGraph`](data_class.md) | Heterogeneous graph container — edge init, ML export, save/load |
| [`building`](building.md) | Building fetch, geometry preprocessing, morphological shape metrics |
| [`population`](population.md) | Meta and GHS population raster fetching and processing |
| [`satellite`](satellite.md) | Mapbox tile download, building image chips, Google Earth Engine integration |
| [`topology`](topology.md) | Fast NetworKit centrality and NetworkX attribute merging |
| [`geom`](geometry.md) | Coordinate projection, buffer, graph-to-GDF, tile utilities |
| [`utils`](utils.md) | OSM extract discovery, country/city lookup, graph edge construction, data prep |
| [`visualisation`](visualisation.md) | Static matplotlib and interactive PyDeck plots |

---

## Quick Reference

### Core workflow

```python
import urbanity
from urbanity import Map

# 1. Map + boundary
m = Map(country="Singapore")
m.add_polygon_boundary("data/my_area.geojson")

# 2a. Street network
G, nodes, edges = m.get_street_network(bandwidth=100, graph_attr=True)

# 2b. Heterogeneous urban graph
urban_graph = m.get_urban_graph(bandwidth=100)
urban_graph.initialize_edges(knn=3)

# 3. Export to PyG / DGL
pyg_data = urban_graph.to_pyg_graph(target_node="building")
dgl_graph = urban_graph.to_dgl()

# 4. Save / load
urban_graph.save_graph("graph.zip")
```

### Visualisation functions

```python
from urbanity.visualisation import (
    plot_graph,               # interactive 3-D PyDeck view
    plot_street_network,      # static matplotlib street map
    plot_buildings,           # building footprint map
    plot_urban_graph_overview,# four-panel node-type overview
    plot_urban_graph_edges,   # inter-layer edge visualisation
    plot_node_attribute,      # colour any node layer by attribute
    plot_network_centrality,  # centrality heat map
)
```

### Building morphology (momepy wrappers)

```python
from urbanity.building import (
    compute_complexity, compute_squareness, compute_shape_index,
    compute_square_compactness, compute_rectangularity, compute_fractaldim,
    compute_equivalent_rectangular_index, compute_longest_axis_length,
    compute_shared_wall_ratio, compute_orientation, compute_elongation,
    compute_corners, compute_convexity, compute_circularcompactness,
)
```

### Population datasets

```python
from urbanity.population import (
    get_meta_population_data,   # Meta 30-m HDX data
    get_tiled_population_data,  # tiled version for large areas
    get_ghs_population_data,    # GHS multi-year temporal data
    raster2gdf,                 # GeoTIFF → GeoDataFrame
)
```
