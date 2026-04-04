# Getting Started

This tutorial introduces the core concepts of Urbanity: creating a `Map` object, defining a study area, and exploring the interactive basemap.

---

## Creating a Map Object

`Map` inherits from `ipyleaflet.Map` and pre-loads several basemap layers (CartoDB, Google Streets, Google Hybrid, Google Terrain). Passing a country name centres the view automatically.

```python
from urbanity import Map
import geopandas as gpd

# Centre the map on Singapore
m = Map(country='Singapore')
m
```

---

## Defining a Study Area

### Method 1 — Interactive draw

Use the polygon or rectangle draw tool in the map controls to sketch your study area, then capture it:

```python
m.add_bbox()    # commits the drawn shape as the active boundary
```

You can verify the captured geometry:

```python
print("Polygon set:", m.polygon_bounds is not None)
print(m.polygon_bounds.geometry.values[0])
```

### Method 2 — Load a GeoJSON / Shapefile

For reproducible workflows, load a boundary file directly:

```python
# Load the bundled Tanjong Pagar (Singapore) sample boundary
m.add_polygon_boundary('data/tanjong_pagar_singapore.geojson',
                        layer_name='Tanjong Pagar')
```

Remote URLs are also supported:

```python
m.add_polygon_boundary('https://figshare.com/ndownloader/files/42515773')
```

---

## Exploring the Boundary

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
m.polygon_bounds.boundary.plot(ax=ax, color='#e8c27a', linewidth=2)
m.polygon_bounds.plot(ax=ax, color='#457b9d', alpha=0.3)
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

---

## Automated OSM Data Download

Urbanity automatically locates and downloads the smallest available OSM extract (from Geofabrik or BBBike) that fully contains your study area — no manual hunting for `.osm.pbf` links needed.

The download happens transparently the first time you call `get_street_network()` or `get_urban_graph()`. Subsequent calls reuse the cached file located in the `./data` folder.

---

## Environment Variables (Optional)

Some features (street view imagery, satellite tiles) require API keys. Store them in a `.env` file in your project root:

```bash
MAPILLARY_API_TOKEN=<your-token>
MAPBOX_API_TOKEN=<your-token>
```

Core network and graph functions work without any API keys.

---

## Summary

| Step | Method | Notes |
|------|--------|-------|
| Create map | `Map(country=...)` | Centres on country centroid |
| Draw study area | draw tool → `m.add_bbox()` | Interactive |
| Load GeoJSON boundary | `m.add_polygon_boundary(path)` | Reproducible |
| Check boundary | `m.polygon_bounds` | Returns a GeoDataFrame |

→ Next: [Street Network Analysis](street-networks.md)
