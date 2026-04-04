# `urbanity.utils`

General helper functions covering OSM data discovery, country/city lookups, graph edge construction, data splitting, and other utilities used internally across the package.

---

## OSM Extract Discovery & Download

### `fetch_geofabrik_index`

```python
fetch_geofabrik_index(cache_path: str = './data/geofabrik_index.json') -> dict
```

Downloads and caches the Geofabrik extract index (a GeoJSON-like index of all available `.osm.pbf` extracts with their bounding geometries).

---

### `fetch_bbbike_index`

```python
fetch_bbbike_index(cache_dir: str = './data/bbbike') -> list
```

Builds a BBBike index of cities with their boundary geometries. Downloads the city list once, then lazily fetches `.poly` files per city.

**Returns:** List of dicts with keys: `name`, `geometry`, `pbf_url`.

---

### `find_smallest_extract`

```python
find_smallest_extract(aoi_geometry, data_dir: str = './data') -> dict
```

Searches both Geofabrik and BBBike to find the smallest `.osm.pbf` extract that fully contains the area of interest. Automatically used by `get_street_network()` to avoid downloading unnecessarily large files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `aoi_geometry` | Shapely geometry | The area of interest polygon. |
| `data_dir` | `str` | Directory to cache downloaded index files. |

**Returns:** Dict with keys `url`, `name`, `size`.

---

### `is_extract_cached`

```python
is_extract_cached(dest_path: str, expected_size: int = None, url: str = None) -> bool
```

Returns `True` if a valid `.osm.pbf` file already exists at `dest_path` (and optionally matches `expected_size` or is consistent with `url`).

---

### `download_extract`

```python
download_extract(url: str, dest_path: str, expected_size: int = None)
```

Downloads a `.osm.pbf` file from `url` to `dest_path`, skipping the download if the file is already cached.

---

## Country & City Discovery

### `get_country_centroids`

```python
get_country_centroids() -> dict
```

Returns a dictionary mapping country names to `(latitude, longitude)` centroid coordinates. Used by `Map(country=...)` to centre the initial view.

---

### `get_available_countries`

```python
get_available_countries()
```

Prints the list of countries for which centroid data is available.

---

### `get_available_pop_countries`

```python
get_available_pop_countries()
```

Prints the list of countries where Meta population data can be fetched.

---

### `get_available_precomputed_network_data`

```python
get_available_precomputed_network_data()
```

Prints the list of cities available from the [Global Urban Network Dataset](https://figshare.com/articles/dataset/Global_Urban_Network_Dataset/22124219).

---

### `get_country_from_polygon`

```python
get_country_from_polygon(polygon_geometry) -> str
```

Determines which country a polygon is in via Nominatim reverse geocoding of the centroid, then matches the result to the Geofabrik extract name list.

---

### `get_gadm`

```python
get_gadm(country: str, city: str, version: str = '4.1', max_level: int = 4, level_drop: int = 0) -> GeoDataFrame
```

Fetches GADM administrative boundary data for a city and returns its subzone GeoDataFrame.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `country` | `str` | required | Country name. |
| `city` | `str` | required | City name. |
| `version` | `str` | `'4.1'` | GADM database version. |
| `max_level` | `int` | `4` | Maximum administrative level to retrieve. |
| `level_drop` | `int` | `0` | Number of top levels to drop. |

---

## Population Links

### `get_population_data_links`

```python
get_population_data_links() -> dict
```

Returns a dictionary mapping country names to their Meta HDX population data download URLs.

---

## POI Utilities

### `finetune_poi`

```python
finetune_poi(df, target: str, relabel_dict: dict, n: int = 5, pois_data: str = 'osm') -> DataFrame
```

Relabels and trims a POI list to eight standard categories: `Civic`, `Commercial`, `Entertainment`, `Food`, `Healthcare`, `Institutional`, `Recreational`, `Social`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | POI DataFrame with full OSM/Overture amenity tags. |
| `target` | `str` | Column containing the raw POI category labels. |
| `relabel_dict` | `dict` | Mapping from raw labels to standard categories. |
| `n` | `int` | Minimum POI count threshold; categories below this are grouped into `Other`. |

---

## Graph Edge Construction

### `get_building_to_building_edges`

```python
get_building_to_building_edges(
    buildings,
    return_neighbours: str = 'knn',
    knn: int = 3,
    distance_threshold: int = 100,
    knn_threshold: int = 100,
    add_reverse: bool = True,
) -> np.ndarray
```

Generates building–building adjacency edges using K-NN or distance threshold.

---

### `get_building_to_street_edges`

```python
get_building_to_street_edges(streets, building_nodes, add_reverse: bool = True) -> np.ndarray
```

Generates edges between buildings and their nearest adjacent street segment (within 50 m).

---

### `get_intersection_to_street_edges`

```python
get_intersection_to_street_edges(intersections, streets, add_reverse: bool = True) -> np.ndarray
```

Connects street intersection nodes to the street segments they belong to.

---

### `get_buildings_in_plot_edges`

```python
get_buildings_in_plot_edges(urban_plots, add_reverse: bool = True) -> np.ndarray
```

Connects each building to the urban plot polygon it falls within.

---

### `get_plot_to_plot_edges`

```python
get_plot_to_plot_edges(urban_plots, add_reverse: bool = True) -> np.ndarray
```

Generates plot–plot adjacency edges for urban plots that share a street boundary.

---

### `get_edge_nodes`

```python
get_edge_nodes(edges: GeoDataFrame) -> GeoDataFrame
```

Converts street segment LineStrings to midpoint nodes for use in dual graph or multi-nodal representations.

---

### `boundary_to_plot`

```python
boundary_to_plot(plot, add_reverse: bool = True) -> np.ndarray
```

Creates a super-node boundary edge linking the study area boundary to all urban plots. Useful as a global context node in GNN architectures.

---

## Data Preparation

### `one_hot_encode_categorical`

```python
one_hot_encode_categorical(df, target_col: str = '', prefix: str = '') -> DataFrame
```

Converts a categorical column to one-hot encoded binary columns with a given prefix.

---

### `standardise_and_scale`

```python
standardise_and_scale(objects: dict) -> dict
```

Standardises and min-max scales all numeric columns in each GeoDataFrame within the `objects` dict. Returns a new dict with scaled DataFrames.

---

### `most_frequent`

```python
most_frequent(List: list)
```

Returns the most common element in a list. Used to determine the dominant category in a set of POI or land-use labels.

---

### `gdf_to_poly`

```python
gdf_to_poly(gdf: GeoDataFrame, poly_path: str, column: str = 'boundary_id')
```

Writes a GeoDataFrame of Polygon or MultiPolygon geometries to Osmosis `.poly` format for use with `osmium` or pyrosm.
