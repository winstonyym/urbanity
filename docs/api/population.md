# `urbanity.population`

Functions for fetching, processing, and integrating population raster datasets — Meta High Resolution Population Density and Global Human Settlement (GHS) layers.

---

## Population Data Fetching

### `get_meta_population_data`

```python
get_meta_population_data(country: str, bounding_poly=None, all_only: bool = False) -> GeoDataFrame
```

Fetches Meta's High Resolution Population Density dataset (30 m resolution) from HDX for a given country. Prioritises `.csv` format over `.tif` due to raster band sparsity in recent time periods. Values are rounded to integers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `country` | `str` | required | Country name (must be in `get_available_pop_countries()`). |
| `bounding_poly` | `GeoDataFrame` | `None` | Optional spatial filter — clips result to this polygon. |
| `all_only` | `bool` | `False` | If `True`, returns only the total population count (not sub-groups). |

**Returns:** `GeoDataFrame` with population grid cells and demographic columns: `PopSum`, `Men`, `Women`, `Elderly`, `Youth`, `Children`.

!!! tip "Check available countries"
    ```python
    from urbanity.utils import get_available_pop_countries
    get_available_pop_countries()
    ```

---

### `get_tiled_population_data`

```python
get_tiled_population_data(country: str, bounding_poly=None, all_only: bool = False) -> GeoDataFrame
```

Variant of `get_meta_population_data` optimised for large study areas. Population data are pre-tiled to allow efficient extraction across multiple geographic scales. Not available for `.tif` format.

---

### `get_ghs_population_data`

```python
get_ghs_population_data(bounding_poly, bandwidth: int = 100, temporal_years: list = range(1975, 2030, 5)) -> GeoDataFrame
```

Extracts Global Human Settlement (GHS) Population Density at 3 arcsecond (~100 m) resolution for multiple years.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bounding_poly` | `GeoDataFrame` | required | Bounding polygon defining the extraction extent. |
| `bandwidth` | `int` | `100` | Extra buffer (m) beyond the polygon edge. |
| `temporal_years` | `list` | `range(1975, 2030, 5)` | Years to extract (GHS provides data every 5 years). |

**Returns:** `GeoDataFrame` with one population column per year (e.g. `pop_1975`, `pop_1980`, …).

---

## Raster Conversion

### `raster2gdf`

```python
raster2gdf(raster_path: str, chosen_band: int = 1, get_grid: bool = True, zoom: bool = False, boundary=None, same_geometry: bool = False) -> GeoDataFrame
```

Converts a GeoTIFF raster to a GeoDataFrame. Each pixel becomes a row with its polygon geometry and raster value.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raster_path` | `str` | required | Path to a `.tiff` raster file. |
| `chosen_band` | `int` | `1` | Band index to extract (1-indexed). |
| `get_grid` | `bool` | `True` | If `True`, returns pixel polygons; if `False`, returns centroids. |
| `zoom` | `bool` | `False` | Apply zoom-level tile clipping. |
| `boundary` | `GeoDataFrame` | `None` | Clip result to this polygon. |
| `same_geometry` | `bool` | `False` | If `True`, preserves the original grid geometry. |

---

### `csv2grid`

```python
csv2grid(filepath: str, lat_col: str = 'Y', lon_col: str = 'X') -> GeoDataFrame
```

Converts a CSV file with latitude/longitude columns into a point `GeoDataFrame`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | required | Path to the local CSV file. |
| `lat_col` | `str` | `'Y'` | Column name for latitude. |
| `lon_col` | `str` | `'X'` | Column name for longitude. |

---

## Raster Utilities

### `extract_tiff_from_shapefile`

```python
extract_tiff_from_shapefile(geotiff_path: str, shapefile, output_filepath: str = None, zipped: bool = True)
```

Clips a GeoTIFF to the bounds of a shapefile and optionally writes the result to disk.

---

### `mask_raster_with_gdf`

```python
mask_raster_with_gdf(gdf, raster) -> np.ndarray
```

Masks a rasterio raster array with a GeoDataFrame polygon, setting all pixels outside the polygon to `NoData`.

---

### `mask_raster_with_gdf_large_raster`

```python
mask_raster_with_gdf_large_raster(gdf, raster, tile_size: int = 512) -> np.ndarray
```

Tile-based variant of `mask_raster_with_gdf` for large rasters that do not fit in memory. Processes the raster in `tile_size × tile_size` pixel chunks.

---

### `merge_raster_list`

```python
merge_raster_list(raster_list: list)
```

Merges a list of rasterio raster objects into a single mosaic raster.

---

### `load_npz_as_raster`

```python
load_npz_as_raster(npz_file: str) -> tuple
```

Loads a compressed `.npz` population raster file. Returns `(array, transform, crs)`.

---

### `is_bounding_box_within`

```python
is_bounding_box_within(bounds: tuple, box: tuple) -> bool
```

Returns `True` if `box` is fully contained within `bounds`. Both are `(min_x, min_y, max_x, max_y)` tuples.

---

### `find_valid_tif_files`

```python
find_valid_tif_files(filepaths: list, bounding_box: tuple) -> list
```

Filters a list of TIFF file paths, returning only those whose spatial extent overlaps or contains `bounding_box`.

---

### `download_pop_tiff_from_path`

```python
download_pop_tiff_from_path(lcz_path: str)
```

Downloads a TIFF file from a URL into a rasterio object. Handles ZIP-archived TIFFs transparently.
