# Quickstart

This guide walks you through generating your first feature-rich urban network in a few lines of code.

---

## 1. Import and create a map

```python
import urbanity

# Create an interactive map centred on a country
m = urbanity.Map(country="Singapore")
m
```

This opens an interactive `ipyleaflet` map. You can:

- **Draw a polygon** — use the polygon tool in the toolbar to trace your area of interest
- **Draw a bounding box** — use the rectangle tool for a quick rectangular selection
```
m.add_bbox()
```
- **Load a shapefile** — pass an existing GeoDataFrame directly (see below) [Shapefile](https://figshare.com/ndownloader/files/42515773)
```
m.add_polygon_boundary('path_to_shapefile')
```

!!! tip "Working in Jupyter"
    The interactive map works in JupyterLab and Jupyter Notebook. For scripts, pass a shapefile directly instead.

To use an existing polygon assign it directly to the `polygon_bounds` properties attribute:

```python
import geopandas as gpd

boundary = gpd.read_file('path_to_shapefile')
G.polygon_bounds = boundary
```

---

## 2. Obtain street network

After drawing your area on the map:

```python
# Build a driving network for the drawn area
m.get_network_layer(add_svi=False)
intersections, streets = m.network[1], m.network[2]
```

Supported `network_type` values: `"drive"`, `"walk"`, `"bike"`, `"all"`.

Visualise street network

```python
fig, ax = plt.subplots(figsize=(10, 10))

streets.plot(ax=ax, color="black", linewidth=0.5)
intersections.plot(ax=ax, color="orange", markersize=2, zorder=2)

plt.show()
```

---

## Next Steps

- **[Urban Street Networks tutorial](tutorials/street-networks.md)** — deeper dive into network construction options
- **[Urban Graphs tutorial](tutorials/urban-graphs.md)** — primal, dual, and spatial graph generation
- **[Graph ML tutorial](tutorials/graph-ml.md)** — node classification and regression with PyG/DGL
- **[API Reference](api/index.md)** — full method signatures and parameters
