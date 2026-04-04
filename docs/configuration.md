# Configuration

Urbanity uses a `.env` file in your working directory to manage API credentials and optional configuration values. No configuration is required for basic use.

---

## Environment Variables

Create a `.env` file at the root of your project:

```bash
# Street view imagery — https://www.mapillary.com/developer/api-documentation
MAPILLARY_API_SECRET=MLY|XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX
MAPILLARY_API_TOKEN=MLY|XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXX

# Satellite tiles — https://www.mapbox.com/developers
MAPBOX_API_TOKEN=pk.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Local Climate Zone tile URL (if self-hosted)
LCZ_URL=
```

Urbanity loads these automatically at import time via `python-dotenv`. You can also set them as shell environment variables instead.

---

## Google Earth Engine

GEE requires a separate authentication step (one-time):

```python
import ee
ee.Authenticate()          # opens browser to sign in
ee.Initialize(project="your-gee-project-id")
```

After authenticating once, credentials are cached locally and reused in future sessions.

---

## Logging & Warnings

By default, Urbanity suppresses Shapely, GeoPandas, and other noisy deprecation warnings. To re-enable them for debugging:

```python
import warnings
warnings.resetwarnings()
import urbanity
```

---

## Data Directories

Urbanity stores cached data under the package's source directory:

| Directory | Contents |
|-----------|---------|
| `building_height_data/` | Cached building height rasters |
| `ghs_data/` | Global Human Settlement population grids |
| `lcz_data/` | Local Climate Zone rasters |
| `svi_data/` | Downloaded street view image tiles |
| `map_data/` | Vector map tile cache |
| `overture_data/` | Overture Maps building footprints |

These directories are created on first use. You can point to custom directories by modifying the relevant function arguments — see the [API Reference](api/index.md).
