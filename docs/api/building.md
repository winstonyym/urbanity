# `urbanity.building`

Functions for fetching, preprocessing, and analysing building footprint data from OpenStreetMap and Overture Maps.

---

## Data Fetching

### `get_osm_buildings`

```python
get_osm_buildings(location: str = '', fp: str = '', boundary=None) -> GeoDataFrame
```

Fetches OpenStreetMap building footprints via the pyrosm API (Geofabrik mirror). Accepts a named place or a pre-downloaded `.osm.pbf` file path, and an optional GeoDataFrame to spatially filter results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `location` | `str` | `''` | Country or city name (e.g. `"Singapore"`). |
| `fp` | `str` | `''` | Path to a local `.osm.pbf` file. |
| `boundary` | `GeoDataFrame` | `None` | Spatial filter — only buildings inside this polygon are returned. |

**Returns:** `GeoDataFrame` with building polygons and OSM attributes.

---

### `get_overture_buildings`

```python
get_overture_buildings(building_data: str) -> GeoDataFrame
```

Loads pre-downloaded Overture Maps building footprints from a local file path.

| Parameter | Type | Description |
|-----------|------|-------------|
| `building_data` | `str` | Path to the Overture building footprint file (GeoParquet or GeoJSON). |

---

## Preprocessing

### `preprocess_osm_building_geometry`

```python
preprocess_osm_building_geometry(buildings, minimum_area: float = 30, prefix: str = 'osm') -> GeoDataFrame
```

Standardises raw OSM or Overture building geometries: converts all types to `Polygon`, validates geometry, projects to UTM, and filters out buildings below `minimum_area` m².

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buildings` | `GeoDataFrame` | required | Raw building footprints. |
| `minimum_area` | `float` | `30` | Minimum building footprint area in m². |
| `prefix` | `str` | `'osm'` | Prefix added to attribute column names. |

---

### `remove_overlapping_polygons`

```python
remove_overlapping_polygons(building: GeoDataFrame) -> GeoDataFrame
```

Removes overlapping building polygons, retaining the largest polygon in each overlapping group. Prevents double-counting in density calculations.

---

### `assign_numerical_id_suffix`

```python
assign_numerical_id_suffix(gdf: GeoDataFrame, prefix: str = 'osm') -> GeoDataFrame
```

Assigns unique building IDs to footprints, appending `_1`, `_2`, … suffixes to resolve duplicates. For example, two polygons sharing `osmid=12093210` become `12093210_1` and `12093210_2`.

---

### `remove_duplicate_points`

```python
remove_duplicate_points(points: GeoDataFrame) -> GeoDataFrame
```

Removes globally duplicated point geometries, keeping one occurrence of each. Used when sampling street view image locations.

---

## Morphological Shape Metrics

All morphological functions below wrap [momepy](https://docs.momepy.org/) and accept a building `GeoDataFrame` with a `geometry` column of `Polygon` objects. Each function adds a new column to the GeoDataFrame and returns it.

### `compute_complexity`

```python
compute_complexity(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Computes polygon complexity (ratio of perimeter to area). Based on FRAGSTATS landscape metrics (McGarigal & Marks, 1995).

---

### `compute_squareness`

```python
compute_squareness(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Measures how close each building polygon is to a perfect square, based on the mean deviation of corners from 90°.

---

### `compute_shape_index`

```python
compute_shape_index(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Computes the shape index (perimeter divided by the perimeter of an equivalent circle). Values near 1 indicate circular shapes.

---

### `compute_square_compactness`

```python
compute_square_compactness(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Ratio of the area of the polygon to the area of its minimum bounding square.

---

### `compute_rectangularity`

```python
compute_rectangularity(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Ratio of the polygon area to its minimum bounding rectangle area. Values near 1 indicate very rectangular buildings.

---

### `compute_fractaldim`

```python
compute_fractaldim(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Computes the fractal dimension of each building polygon boundary, capturing perimeter complexity at multiple scales.

---

### `compute_equivalent_rectangular_index`

```python
compute_equivalent_rectangular_index(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Based on Basaraner & Cetinkaya (2017). Measures how close the polygon shape is to its equivalent rectangle (same area and perimeter). Values near 1 indicate near-rectangular buildings.

---

### `compute_longest_axis_length`

```python
compute_longest_axis_length(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Computes the length of the longest axis — defined as the diameter of the minimal circumscribed circle around the convex hull.

---

### `compute_shared_wall_ratio`

```python
compute_shared_wall_ratio(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Computes the ratio of shared wall length to total perimeter for each building. Useful for characterising terraced or semi-detached typologies. Based on Hamaina et al. (2012).

---

### `compute_orientation`

```python
compute_orientation(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Orientation of each building's longest bounding rectangle axis, in the range 0–45°.

---

### `compute_elongation`

```python
compute_elongation(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Ratio of the shorter to the longer side of the minimum bounding rectangle. Values near 0 are very elongated; values near 1 are nearly square.

---

### `compute_corners`

```python
compute_corners(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Counts the number of corners in each building polygon boundary.

---

### `compute_convexity`

```python
compute_convexity(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Ratio of building polygon area to its convex hull area. Values near 1 indicate convex shapes.

---

### `compute_circularcompactness`

```python
compute_circularcompactness(building_nodes: GeoDataFrame, element: str = 'building') -> GeoDataFrame
```

Ratio of building area to the area of the circumscribed circle around it.

---

## Neighbourhood Analysis

### `building_knn_nearest`

```python
building_knn_nearest(gdf: GeoDataFrame, knn: int = 3, non_nan_col: str = None) -> GeoDataFrame
```

Computes K-nearest neighbours and distances for each building centroid. Optionally excludes buildings with `NaN` in `non_nan_col` from being considered as neighbours.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gdf` | `GeoDataFrame` | required | Building footprints with centroid geometry. |
| `knn` | `int` | `3` | Number of nearest neighbours to find. |
| `non_nan_col` | `str` | `None` | If set, excludes rows with NaN in this column from the neighbour pool. |

---

### `compute_knn_aggregate`

```python
compute_knn_aggregate(building_nodes: GeoDataFrame, attr_cols: list) -> GeoDataFrame
```

Computes mean and standard deviation for each column in `attr_cols` across each building's K-nearest neighbours. Adds `{col}_knn_mean` and `{col}_knn_std` columns.

---

### `get_minimum_bounding_rectangle`

```python
get_minimum_bounding_rectangle(building_nodes: GeoDataFrame) -> GeoDataFrame
```

Generates the minimum bounding rectangle for each building polygon and adds it as a new geometry column.

---

## Building Heights

### `get_building_heights`

```python
get_building_heights(filepath: str, target_key: str) -> dict
```

Loads a building height lookup table from a file and returns a dictionary keyed by `target_key`.

---

### `assign_building_heights`

```python
assign_building_heights(heights: dict, building_gdf: GeoDataFrame) -> GeoDataFrame
```

Merges a height lookup dictionary into a building GeoDataFrame by matching on building ID.

---

### `get_and_assign_building_heights`

```python
get_and_assign_building_heights(filepath: str, target_key: str, building_gdf: GeoDataFrame) -> GeoDataFrame
```

Convenience wrapper: loads heights from `filepath` and assigns them to `building_gdf` in one call.
