# `urbanity.UrbanGraph`

The `UrbanGraph` class is the primary data container returned by `Map.get_urban_graph()`. It stores the four spatial node layers and their inter-layer edges, and provides methods for visualisation, serialisation, and graph-ML export.

```python
from urbanity.data_class import UrbanGraph
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `geo_store` | `dict[str, GeoDataFrame]` | Node GeoDataFrames keyed by type: `'plot'`, `'building'`, `'street'`, `'intersection'`, `'boundary'`. |
| `edge_store` | `dict[str, np.ndarray]` | Edge arrays keyed by connection type (e.g. `'plot_to_plot'`, `'building_to_street'`). |

---

## Edge Initialisation

### `initialize_edges`

```python
urban_graph.initialize_edges(
    building_neighbours: str = 'knn',
    knn: int = 5,
    distance: int = 100,
)
```

Populates `edge_store` with all inter-layer adjacency arrays. Must be called before visualising edges or exporting to graph-ML formats.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `building_neighbours` | `str` | `'knn'` | Strategy for building–building edges: `'knn'` or `'distance'`. |
| `knn` | `int` | `5` | Number of nearest neighbours for `'knn'` mode. |
| `distance` | `int` | `100` | Distance threshold (m) for `'distance'` mode. |

**Example:**

```python
urban_graph = m.get_urban_graph(bandwidth=100)
urban_graph.initialize_edges(building_neighbours='knn', knn=3, distance=100)

# Inspect created edge types
for edge_type, arr in urban_graph.edge_store.items():
    print(f"{edge_type}: {arr.shape}")
```

---

## Visualisation

### `plot_urban_graph`

```python
urban_graph.plot_urban_graph(
    node_type: str = '',
    colname: str = '',
    node_id: str = '',
)
```

Renders an interactive 3-D PyDeck view of the urban graph. Equivalent to calling `plot_graph(urban_graph.geo_store, urban_graph.edge_store, ...)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_type` | `str` | `''` | Highlight this node layer (`'building'`, `'plot'`, `'street'`, `'intersection'`). |
| `colname` | `str` | `''` | Colour nodes by this attribute column. |
| `node_id` | `str` | `''` | If set, highlights this single node and all its neighbours. |

---

## Serialisation

### `save_graph`

```python
urban_graph.save_graph(filename: str)
```

Saves the full graph (all GeoDataFrames in `geo_store` + all arrays in `edge_store`) to a ZIP archive at `filename`. GeoDataFrames are saved as `.parquet`, edge arrays as `.npy`.

!!! tip "Best practice"
    Save *before* calling `initialize_edges()` since edge connectivity is cheap to recompute but the attribute-rich node GeoDataFrames are not.

---

### `load_graph`

```python
urban_graph.load_graph(filename: str)
```

Loads a graph from a `.zip` archive produced by `save_graph()`. Supports both local paths and remote URLs.

**Example:**

```python
from urbanity.data_class import UrbanGraph

urban_graph = UrbanGraph()
urban_graph.load_graph('./data/graph.zip')
# or from a URL:
urban_graph.load_graph('https://example.com/graphs/singapore.zip')
```

---

## Train / Validation / Test Splits

All mask functions return boolean arrays that can be used directly as `train_mask`, `val_mask`, and `test_mask` in PyTorch Geometric or DGL.

### `generate_masks`

```python
generate_masks(
    df_length: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Generates train/val/test boolean masks for `df_length` rows.

---

### `generate_building_masks` / `generate_plot_masks` / `generate_street_masks` / `generate_intersection_masks`

```python
urban_graph.generate_building_masks(
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Convenience wrappers that call `generate_masks` with the length of the corresponding node GeoDataFrame. Identical signature for `building`, `plot`, `street`, and `intersection` variants.

---

## Graph-ML Export

### `to_pyg_graph`

```python
urban_graph.to_pyg_graph(
    target_node: str = 'building',
    categorical: bool = False,
    target_value: list = [],
    train_val_test: list = [0.6, 0.2, 0.2],
    random_seed: int = 0,
)
```

Converts the `UrbanGraph` to a **PyTorch Geometric** `HeteroData` object, ready for use in a Graph Neural Network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_node` | `str` | `'building'` | Node type to treat as the prediction target. |
| `categorical` | `bool` | `False` | If `True`, treats `target_value` as class labels for classification. |
| `target_value` | `list` | `[]` | List of column names to use as node labels/features. |
| `train_val_test` | `list` | `[0.6, 0.2, 0.2]` | Train/val/test split ratios. |
| `random_seed` | `int` | `0` | Random seed for reproducible splits. |

**Example:**

```python
urban_graph.initialize_edges(knn=3)
pyg_data = urban_graph.to_pyg_graph(
    target_node='building',
    target_value=['bid_height'],
    train_val_test=[0.7, 0.15, 0.15],
)
print(pyg_data)
```

---

### `to_dgl`

```python
urban_graph.to_dgl()
```

Converts the `UrbanGraph` to a **DGL** heterogeneous graph object.
