# `urbanity.Map`

The `Map` class is the primary entry point for Urbanity. It inherits from `ipyleaflet.Map` and adds all urban data methods.

```python
from urbanity import Map

m = Map(country="Singapore")
```

---

## Constructor

```python
Map(country: str = None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `country` | `str` | `None` | Name of country to centre the map view on. |
| `**kwargs` | | | Passed through to `ipyleaflet.Map`. |

---

## Boundary Methods

### `add_bbox`

```python
m.add_bbox(show: bool = False, remove: bool = True)
```

Commits the bounding box drawn on the map as the active geographic extent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show` | `bool` | `False` | If `True`, opens a second map view for confirmation. |
| `remove` | `bool` | `True` | If `True`, removes the drawn box layer from the map. |

---

### `add_polygon_boundary`

```python
m.add_polygon_boundary(
    filepath: str,
    layer_name: str = "Boundary",
    polygon_style: dict = None,
    show: bool = False,
)
```

Loads a vector boundary from disk (`.geojson` or `.shp`) and adds it as a map layer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | required | Path to the vector file. |
| `layer_name` | `str` | `"Boundary"` | Display name for the layer in the legend. |
| `polygon_style` | `dict` | `None` | `ipyleaflet` style dict (color, weight, fillOpacity…). |
| `show` | `bool` | `False` | Open a second map view after loading. |

---

### `remove_polygon_boundary`

```python
m.remove_polygon_boundary()
```

Removes the currently loaded polygon boundary from the map.

---

## Data Quality Checks

### `check_osm_buildings`

```python
m.check_osm_buildings(location: str, column: str = "name")
```

Reports OSM building attribute completeness for a city or country — percentage of buildings with height, levels, and use type.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | `str` | City or country name (e.g. `"Singapore"`, `"Netherlands"`). |
| `column` | `str` | OSM attribute column to check completeness for. |

---

### `check_population`

```python
m.check_population(year: int = 2020)
```

Compares Meta high-resolution population counts (30 m) against WorldPop UN-adjusted data for the current boundary.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year` | `int` | `2020` | Reference year for the population layer. |

---

## Network Methods

### `get_network_layer`

```python
m.get_network_layer(
    location: str,
    network_filepath: str = None,
    bandwidth: int = 0,
    network_type: str = "drive",
    add_svi: bool = False,
    segment_svi: bool = False,
    fill_svi_gaps: bool = False,
    bin_distance: int = 50,
    v_threshold: float = 0.5,
    proximity_distance: int = 20,
)
```

Assigns urban network layer information to the map. Optionally adds Mapillary Street View Imagery (SVI) data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `location` | `str` | required | City or country name, or bounding box string. |
| `network_filepath` | `str` | `None` | Path to a pre-downloaded OSM `.pbf` file. |
| `bandwidth` | `int` | `0` | Buffer radius (m) around the network for context. |
| `network_type` | `str` | `"drive"` | `"drive"`, `"walk"`, `"bike"`, or `"all"`. |
| `add_svi` | `bool` | `False` | Download and attach Mapillary SVI to nodes. |
| `segment_svi` | `bool` | `False` | Run semantic segmentation on SVI images. |
| `fill_svi_gaps` | `bool` | `False` | Interpolate SVI features for nodes without imagery. |
| `bin_distance` | `int` | `50` | Sampling interval (m) for SVI along edges. |
| `v_threshold` | `float` | `0.5` | Visibility score threshold for filtering SVI. |
| `proximity_distance` | `int` | `20` | Max distance (m) to snap SVI points to the network. |

---

### `get_street_network`

```python
m.get_street_network(
    location: str,
    filepath: str = None,
    bandwidth: int = 0,
    network_type: str = "drive",
    graph_attr: bool = True,
    catchment_attr: bool = True,
    building_attr: bool = False,
    pop_attr: bool = False,
    poi_attr: bool = False,
    svi_attr: bool = False,
    edge_attr: bool = False,
    pois_data=None,
    building_data=None,
    get_precomputed: bool = False,
    temporal_network: bool = False,
    temporal_years: list = None,
    built_up_threshold: float = 0.0,
    dual: bool = False,
)
```

Generates a feature-rich primal planar or dual street network. The primary method for full network construction with all indicators.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `location` | `str` | required | City or country name. |
| `filepath` | `str` | `None` | Path to OSM `.pbf` file (optional). |
| `bandwidth` | `int` | `0` | Buffer radius (m). |
| `network_type` | `str` | `"drive"` | Network mode. |
| `graph_attr` | `bool` | `True` | Include graph-level topological attributes. |
| `catchment_attr` | `bool` | `True` | Include node catchment area attributes. |
| `building_attr` | `bool` | `False` | Add building footprint attributes. |
| `pop_attr` | `bool` | `False` | Add population layer attributes. |
| `poi_attr` | `bool` | `False` | Add points of interest attributes. |
| `svi_attr` | `bool` | `False` | Add Mapillary SVI attributes. |
| `edge_attr` | `bool` | `False` | Compute edge-level indicators. |
| `dual` | `bool` | `False` | If `True`, generate a dual (edge) graph instead. |

**Returns:** `UrbanGraph` object.

---

### `get_point_context`

```python
m.get_point_context(
    location: str,
    points,
    filepath: str = None,
    network_type: str = "drive",
    bandwidth: int = 500,
    graph_attr: bool = True,
    building_attr: bool = False,
    pop_attr: bool = False,
    poi_attr: bool = False,
    svi_attr: bool = False,
    pois_data=None,
    building_data=None,
)
```

Augments an existing `GeoDataFrame` of arbitrary points with urban contextual information within a Euclidean catchment radius.

**Returns:** Enriched `GeoDataFrame`.

---

### `get_aggregate_stats`

```python
m.get_aggregate_stats(
    location: str,
    column: str,
    filepath: str = None,
    bandwidth: int = 0,
    network_type: str = "drive",
    get_graph: bool = True,
    get_building: bool = True,
    get_pop: bool = True,
    get_pois: bool = True,
    get_svi: bool = False,
    pois_data=None,
    building_data=None,
)
```

Returns descriptive statistics for the current bounding polygon without constructing a full network. Useful for rapid area profiling.

**Returns:** `dict` of aggregate statistics.

---

## Graph Methods

### `get_building_nodes`

```python
m.get_building_nodes(
    location: str,
    return_neighbours: bool = True,
    knn: int = 5,
    knn_threshold: float = None,
    distance_threshold: float = 100.0,
    building_data=None,
)
```

Generates a building graph where each building is a node enriched with morphological attributes. Neighbours are computed by K-NN or distance threshold.

**Returns:** `UrbanGraph` object (building-level).

---

### `get_urban_plot_nodes`

```python
m.get_urban_plot_nodes(
    location: str,
    filepath: str = None,
    bandwidth: int = 0,
    network_type: str = "drive",
    pois_data=None,
    building_data=None,
    return_edges: bool = True,
    return_buildings: bool = True,
    minimum_area: float = 0.0,
    out_buffer: float = 0.0,
)
```

Generates an urban plot graph where each city block / urban plot is a node. Useful for morphological analysis at the block level.

---

### `get_urban_graph`

```python
m.get_urban_graph(
    network_filepath: str = None,
    svi_filepath: str = None,
    building_filepath: str = None,
    poi_filepath: str = None,
    canopy_filepath: str = None,
    pop_filepath: str = None,
    bandwidth: int = 0,
    minimum_area: float = 0.0,
    network_type: str = "drive",
    save_as_h5: bool = False,
    save_as_npz: bool = False,
    save_filepath: str = None,
    add_satellite_imagery: bool = False,
    add_gee_layers: bool = False,
    add_lcz: bool = False,
    temporal_years: list = None,
    population_layer: str = "meta",
    satellite_pixel_padding: int = 0,
)
```

Generates a **heterogeneous urban graph** with four node types: urban plots, buildings, streets, and intersections.

| Node type | Description |
|-----------|-------------|
| Urban plots | City block polygons |
| Buildings | Individual building footprints |
| Streets | Street segment midpoints |
| Intersections | Street junctions |

**Returns:** `UrbanGraph` with node GeoDataFrames accessible via `.plot_nodes`, `.building_nodes`, `.street_nodes`, `.intersection_nodes`.

---

## Population & Buildings

### `get_population_layer`

```python
m.get_population_layer(
    population_layer: str = "meta",
    bandwidth: int = 0,
    temporal_years: list = None,
)
```

Fetches and attaches a disaggregated population grid to the network. Supports Meta (30 m) and Global Human Settlement (GHS) layers.

| `population_layer` value | Dataset |
|--------------------------|---------|
| `"meta"` | Meta high-resolution population (30 m) |
| `"ghs"` | Global Human Settlement Layer (100 m) |

---

### `get_building_layer`

```python
m.get_building_layer(
    location: str,
    network_filepath: str = None,
    bandwidth: int = 0,
)
```

Fetches OSM building footprints and attributes (height, levels, use type) for the current boundary.
