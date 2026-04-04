# `urbanity.geom`

Geometric utility functions for working with urban spatial data. These are low-level helpers used internally and can also be called directly.

```python
from urbanity import geom
```

---

## Coordinate & CRS Utilities

### `get_utm_crs`

```python
geom.get_utm_crs(gdf) -> str
```

Returns the appropriate UTM CRS EPSG code for a GeoDataFrame based on its centroid location. Useful for projecting to a metric CRS before distance calculations.

---

### `project_to_utm`

```python
geom.project_to_utm(gdf) -> GeoDataFrame
```

Projects a GeoDataFrame from geographic coordinates (WGS84) into the best-fit UTM zone.

---

## Spatial Operations

### `buffer_geometry`

```python
geom.buffer_geometry(gdf, distance: float, cap_style: int = 1) -> GeoDataFrame
```

Buffers all geometries in a GeoDataFrame by `distance` metres (after projecting to UTM).

| Parameter | Type | Description |
|-----------|------|-------------|
| `gdf` | `GeoDataFrame` | Input geometries. |
| `distance` | `float` | Buffer radius in metres. |
| `cap_style` | `int` | Shapely cap style: `1` = round, `2` = flat, `3` = square. |

---

### `compute_voronoi`

```python
geom.compute_voronoi(points_gdf) -> GeoDataFrame
```

Computes Voronoi polygons for a set of point geometries, clipped to the convex hull of the input.

---

### `snap_points_to_network`

```python
geom.snap_points_to_network(points_gdf, edges_gdf, threshold: float = 50.0) -> GeoDataFrame
```

Snaps each point to the nearest edge in the network within `threshold` metres. Returns the points GeoDataFrame with added `nearest_edge_id` and `distance_to_edge` columns.

---

## Geometry Helpers

### `get_polygon_bounds`

```python
geom.get_polygon_bounds(gdf) -> tuple
```

Returns `(minx, miny, maxx, maxy)` bounding box for a GeoDataFrame in WGS84.

---

### `polygon_to_bbox`

```python
geom.polygon_to_bbox(polygon) -> dict
```

Converts a Shapely polygon to an `{"north", "south", "east", "west"}` bounding-box dict compatible with OSM query APIs.

---

### `split_polygon_grid`

```python
geom.split_polygon_grid(polygon, cell_size: float = 1000.0) -> list
```

Splits a large polygon into a regular grid of smaller cells (cell size in metres). Useful for tiling large-area data downloads.
