---
hide:
  - toc
  - navigation
---

<div class="hero" markdown>

# Urbanity

**Automated modelling and analysis of multidimensional urban networks**

<div class="hero-badges" markdown>

[![PyPI version](https://badge.fury.io/py/urbanity.svg)](https://badge.fury.io/py/urbanity)
[![Downloads](https://pepy.tech/badge/urbanity)](https://pepy.tech/project/urbanity)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/urbanity/badge/?version=latest)](https://urbanity.readthedocs.io/en/latest/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K-6DlBbuQX48WVsxpwAymgLlibPOJHME?usp=sharing)

</div>

[Get Started](installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/winstonyym/urbanity){ .md-button }

</div>

---

## What is Urbanity?

**Urbanity** is a network and graph-based Python package developed at the [NUS Urban Analytics Lab](https://ual.sg/) since 2022. It automates the construction of **feature-rich, contextual, and semantic urban networks and graphs** at any geographical scale — from a single neighbourhood to an entire city.

<p align="center">
  <img src="https://raw.githubusercontent.com/winstonyym/urbanity/main/images/citynetworks.png" width="100%" alt="Urban networks of cities around the world">
  <em>Feature-rich networks of cities around the world</em>
</p>

---

## Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="icon">🏙️</div>
**City-Scale Networks**

Generate complete, analysis-ready street networks for any city in the world using OpenStreetMap data.
</div>

<div class="feature-card" markdown>
<div class="icon">📊</div>
**Rich Indicators**

Automatically compute metric, topological, contextual, and semantic network indicators at every node and edge.
</div>

<div class="feature-card" markdown>
<div class="icon">🗺️</div>
**Multiple Graph Types**

Generate primal planar, dual, and spatial graphs — all convertible to graph-ML-ready formats.
</div>

<div class="feature-card" markdown>
<div class="icon">🏢</div>
**Building Integration**

Integrate building footprints, heights, use types, and energy characteristics into your network.
</div>

<div class="feature-card" markdown>
<div class="icon">👁️</div>
**Street View Imagery**

Process Mapillary street view images for semantic segmentation and visual urban indicators.
</div>

<div class="feature-card" markdown>
<div class="icon">🛰️</div>
**Satellite Imagery**

Pull and process Mapbox satellite tiles and Google Earth Engine raster layers.
</div>

<div class="feature-card" markdown>
<div class="icon">👥</div>
**Population Data**

Overlay disaggregated population grids (GHS, Meta) for demographic context.
</div>

<div class="feature-card" markdown>
<div class="icon">🤖</div>
**Graph ML Ready**

Export directly to PyTorch Geometric or DGL for node, edge, and graph-level prediction tasks.
</div>

</div>

---

## Quickstart

```python
import urbanity

# Create an interactive map
m = urbanity.Map(country="Singapore")
m.show()

# Draw your area of interest on the map, then build the network
G = m.get_network(network_type="drive")
G.get_indicators()
```

→ See the full [Quickstart guide](quickstart.md) for a step-by-step walkthrough.

---

## Global Dataset

Don't want to build from scratch? Download pre-built, feature-rich urban graphs for hundreds of cities:

- 🌍 [Global Feature-Rich Urban Networks](https://figshare.com/articles/dataset/Global_Urban_Network_Dataset/22124219)
- 🌐 [Global Urban Graph Dataset](https://figshare.com/articles/dataset/Global_Graph_Dataset/28852319)

---

## Citation

If you use Urbanity in your research, please cite:

<div class="citation-block">

Yap, W., Stouffs, R. & Biljecki, F. **Urbanity: automated modelling and analysis of multidimensional networks in cities.** *npj Urban Sustainability* 3, 45 (2023). https://doi.org/10.1038/s42949-023-00125-w

</div>

See the full [citation list](changelog.md) for all related publications.

---

<p align="center">
  <a href="https://ual.sg/">
    <img src="https://raw.githubusercontent.com/winstonyym/urbanity/main/images/ualsg.jpeg" width="45%" alt="NUS Urban Analytics Lab">
  </a>
</p>
