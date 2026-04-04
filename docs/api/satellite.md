# `urbanity.satellite`

Functions for downloading Mapbox satellite tiles, generating building image chips, and integrating Google Earth Engine raster layers.

---

## Satellite Tile Download

### `download_satellite_tiles_from_bbox`

```python
download_satellite_tiles_from_bbox(bbox: list, api_key: str) -> GeoDataFrame
```

Downloads Mapbox satellite imagery tiles for the given bounding box and returns a GeoDataFrame of tile polygons with paths to the downloaded images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bbox` | `list` | `[west, south, east, north]` bounding box in WGS84. |
| `api_key` | `str` | Mapbox API token (set in `.env` as `MAPBOX_API_TOKEN`). |

!!! tip
    Store your Mapbox token in `.env`:
    ```
    MAPBOX_API_TOKEN=pk.XXXXXXXXXXXXXXXX
    ```

---

### `get_tiles_from_bbox`

```python
get_tiles_from_bbox(bbox: list = None, zoom: int = 18) -> list
```

Returns a list of `mercantile.Tile` objects covering the bounding box at the specified zoom level.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `list` | `None` | `[west, south, east, north]`. |
| `zoom` | `int` | `18` | Zoom level (higher = more tiles, more detail). |

---

### `get_tiles_gdf`

```python
get_tiles_gdf(tiles: list, bounds=None) -> GeoDataFrame
```

Converts a list of `mercantile.Tile` objects into a `GeoDataFrame` of tile bounding box polygons.

---

### `get_and_combine_tiles`

```python
get_and_combine_tiles(tiles_gdf_proj, building_chips, data_folder: str, output_folder: str) -> GeoDataFrame
```

Combines downloaded satellite tiles into a mosaic, then crops to building chip extents.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tiles_gdf_proj` | `GeoDataFrame` | Projected tile polygons with image paths. |
| `building_chips` | `GeoDataFrame` | Building chip geometries to crop to. |
| `data_folder` | `str` | Directory containing downloaded tile images. |
| `output_folder` | `str` | Directory to save cropped building chips. |

---

## Building Image Chips

### `get_building_image_chips`

```python
get_building_image_chips(building_proj, tiles_gdf_proj, add_context: bool = False, pad_npixels: int = 0) -> GeoDataFrame
```

Generates per-building image chip extents by intersecting building footprints with satellite tiles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `building_proj` | `GeoDataFrame` | required | Projected building footprints. |
| `tiles_gdf_proj` | `GeoDataFrame` | required | Projected tile polygons. |
| `add_context` | `bool` | `False` | If `True`, adds a contextual buffer around each chip. |
| `pad_npixels` | `int` | `0` | Extra pixel padding added to chip extents. |

---

### `get_max_img_dims`

```python
get_max_img_dims(buildings_proj) -> tuple
```

Returns `(max_width_px, max_height_px)` — the maximum pixel dimensions across all building footprints, used to set a uniform chip size.

---

### `view_satellite_building_pair`

```python
view_satellite_building_pair(gdf: GeoDataFrame, bids: list)
```

Plots satellite imagery alongside the corresponding building footprint for one or more building IDs. Useful for visual inspection of chip quality.

---

### `get_grid_size`

```python
get_grid_size(tile_gdf_proj) -> tuple
```

Returns the pixel grid dimensions `(nrows, ncols)` of a projected tile GeoDataFrame.

---

## Google Earth Engine Integration

### `gee_layer_from_boundary`

```python
gee_layer_from_boundary(boundary, layer_name: str, delta: float = 0.1, band: str = '', index: int = 0, large: bool = True) -> GeoDataFrame
```

Fetches a raster from a Google Earth Engine `ImageCollection` or `Image` over the input boundary, and converts the result to a GeoDataFrame with polygon geometries for each pixel.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boundary` | `GeoDataFrame` | required | Study area boundary. |
| `layer_name` | `str` | required | GEE asset ID (e.g. `"COPERNICUS/S2_SR"`). |
| `delta` | `float` | `0.1` | Tile size in degrees for tiling large areas. |
| `band` | `str` | `''` | Band name to extract. |
| `index` | `int` | `0` | Band index if multiple bands are returned. |
| `large` | `bool` | `True` | If `True`, tiles large areas automatically. |

!!! note "Authentication required"
    ```python
    import ee
    ee.Authenticate()
    ee.Initialize(project="your-project-id")
    ```

---

### `tiled_large_array`

```python
tiled_large_array(boundary, layer_name: str, delta: float = 0.1, band: str = '', index: int = 0) -> np.ndarray
```

Fetches and stitches a GEE raster mosaic for a large boundary by splitting it into tiles of size `delta` degrees.

---

### `merge_raster_to_gdf`

```python
merge_raster_to_gdf(raster, gdf, id_col: str = 'plot_id', raster_col: str = 'value', num_classes: int = 1, method: str = 'mean') -> GeoDataFrame
```

Overlays a raster on a GeoDataFrame of polygons and summarises pixel values per polygon using `method` (`"mean"`, `"sum"`, `"max"`, etc.).

---

## Raster Utilities

### `raster2gdf`

```python
raster2gdf(raster_path: str, chosen_band: int = 1, get_grid: bool = True, zoom: bool = False, boundary=None) -> GeoDataFrame
```

Converts a GeoTIFF raster to a GeoDataFrame. Each pixel becomes a row with its polygon geometry and raster value.

---

### `raster2gdf_with_reprojection`

```python
raster2gdf_with_reprojection(raster_path: str, boundary_gdf, chosen_band: int = 1) -> GeoDataFrame
```

Reads a raster, reprojects the entire dataset to EPSG:4326, clips to the boundary, and converts to a GeoDataFrame.

---

### `extract_tiff_from_shapefile`

```python
extract_tiff_from_shapefile(geotiff_path: str, shapefile, output_filepath: str = None, zipped: bool = True)
```

Clips a GeoTIFF to the extent of a shapefile polygon and optionally saves the result.

---

### `mosaic_and_save_tiff`

```python
mosaic_and_save_tiff(source, dest: str, name: str, fmt: str = 'tiff')
```

Merges a list of raster sources into a mosaic and saves it to `dest/name.fmt`.

---

### `split_bbox_into_grid`

```python
split_bbox_into_grid(bbox_gdf) -> GeoDataFrame
```

Splits the bounding box of `bbox_gdf` into an equal-sized grid of rectangles. Used internally for tiling large GEE requests.

---

### `concat_grid`

```python
concat_grid(tile_list: list) -> np.ndarray
```

Stitches a flat list of NumPy arrays (supplied row-major) into a single mosaic array.

---

### `download_tiff_from_path`

```python
download_tiff_from_path(lcz_path: str)
```

Downloads and reads a TIFF file from a URL into a rasterio dataset object. Handles ZIP-archived TIFFs.
