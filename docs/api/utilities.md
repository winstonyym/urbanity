# `urbanity.utils`

General helper functions and data loading utilities used across the package.

```python
from urbanity import utils
```

---

## Country & Geography Helpers

### `get_country_centroids`

```python
utils.get_country_centroids() -> dict
```

Returns a dictionary mapping country names to `(latitude, longitude)` centroid coordinates. Used internally by `Map(country=...)` to centre the initial map view.

---

### `get_city_bbox`

```python
utils.get_city_bbox(city_name: str) -> dict
```

Queries the Nominatim geocoder to retrieve a bounding box for a named city or region.

**Returns:** `{"north": float, "south": float, "east": float, "west": float}`

---

## Data Loading

### `load_building_height_data`

```python
utils.load_building_height_data(filepath: str) -> GeoDataFrame
```

Loads a building height raster or vector file from disk and returns it as a GeoDataFrame with a `height` column.

---

### `load_npz_as_raster`

```python
utils.load_npz_as_raster(filepath: str) -> tuple
```

Loads a compressed `.npz` population raster. Returns `(array, transform, crs)`.

---

## Network Helpers

### `merge_nx_property`

```python
utils.merge_nx_property(G_nx, gdf, attribute: str) -> networkx.Graph
```

Merges a column from a GeoDataFrame back into a NetworkX graph as a node attribute.

---

### `merge_nx_attr`

```python
utils.merge_nx_attr(G_nx, attr_dict: dict) -> networkx.Graph
```

Bulk-assigns attributes from a dictionary `{node_id: value}` to a NetworkX graph.

---

## HTTP Utilities

### `get_with_retry`

```python
utils.get_with_retry(url: str, params: dict = None, retries: int = 3, backoff: float = 1.0)
```

Makes an HTTP GET request with automatic exponential-backoff retry on failure. Used internally for all external API calls (Mapillary, Mapbox, population tile servers).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | required | Endpoint URL. |
| `params` | `dict` | `None` | Query parameters. |
| `retries` | `int` | `3` | Maximum retry attempts. |
| `backoff` | `float` | `1.0` | Base wait time (seconds) between retries. |

---

## Miscellaneous

### `flatten_list`

```python
utils.flatten_list(nested: list) -> list
```

Flattens an arbitrarily nested list into a single flat list.

---

### `chunk_list`

```python
utils.chunk_list(lst: list, size: int) -> list[list]
```

Splits a list into chunks of at most `size` elements. Used for batching API requests.
