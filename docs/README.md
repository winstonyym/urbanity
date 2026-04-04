[![PyPI version](https://badge.fury.io/py/urbanity.svg)](https://badge.fury.io/py/urbanity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/urbanity)](https://badge.fury.io/py/urbanity)
[![Documentation Status](https://img.shields.io/readthedocs/urbanity)](https://urbanity.readthedocs.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K-6DlBbuQX48WVsxpwAymgLlibPOJHME?usp=sharing)

</br>
</br>

![Urbanity Logo](https://raw.githubusercontent.com/winstonyym/urbanity/main/images/urbanity_black_transparent.png#gh-light-mode-only)


--------------------------------------------------------------------------------

</br>

# Urbanity

**Urbanity** is a network-based Python package to automate the construction of feature rich (contextual and semantic) urban networks at any geographical scale. Through an accessible and simple to use interface, users can request heterogeneous urban information such as street view imagery, building morphology, population (including sub-group), and points of interest for target areas of interest. 

</br>

<p align="center">
    <img src="https://raw.githubusercontent.com/winstonyym/urbanity/main/images/citynetworks.png" width = 1000% alt="Logo">
  <h5 align="center">Network of cities around the world</h5>
</p>

</br>
</br>

If you use Urbanity in your work, please cite:
(*Urbanity is currently under review.*)

</br>

## Designed for urban planners
Urbanity is designed in an object-oriented approach that parallels the urban planning process. The urban data science pipeline starts with a base map which users can use to explore their site. Subsequently, there are two ways to specify geographical area of interest: 1) drawing with the polygon and box tools provided; or 2) providing your own polygon shapefiles (all common formats .shp/.geojson are supported). 

Towards exploring complexities underlying urban systems and facilitating comparative study between cities, Urbanity is developed to facilitate downstream descriptive, modelling, and predictive urban analytical tasks.

</br>

## Quickstart

### Installation

Urbanity depends on several geospatial libraries that are best installed through `conda-forge`. We provide an `environment.yml` to create a ready-to-use environment:

```bash
# 1. Clone the repository or download environment.yml
git clone https://github.com/winstonyym/urbanity.git
cd urbanity

# 2. Create the conda environment
conda env create -f environment.yml
conda activate urbanity

# 3. (Optional) Register a Jupyter kernel
python -m ipykernel install --user --name=urbanity
jupyter lab
```

### Three-line Demo

```python
from urbanity import Map

m = Map(country='Singapore')
m.add_polygon_boundary('tanjong_pagar.geojson')

# Build a street network — OSM data is fetched automatically
G, nodes, edges = m.get_street_network()

# Build the full heterogeneous Urban Graph
objects, connections = m.get_urban_graph()
```

> **No manual OSM downloads required.** Urbanity automatically identifies the smallest Geofabrik or BBBike extract that covers your study area and caches it locally.

### Visualisation

```python
from urbanity.visualisation import (
    plot_street_network,
    plot_urban_graph_overview,
    plot_buildings,
    plot_node_attribute,
    plot_network_centrality,
)

# Static dark-mode street network map
fig = plot_street_network(nodes, edges, title='My City Street Network')

# Four-panel overview of all UrbanGraph node types
fig = plot_urban_graph_overview(objects)

# Building footprint choropleth
fig = plot_buildings(objects['building'], colname='bid_area')

# Interactive 3-D PyDeck view
from urbanity.visualisation import plot_graph
plot_graph(objects, connections, node_type='building')
```

</br>

## What can I do with Urbanity?

We demonstrate how you can conduct a diverse range of urban analytical tasks (such as graph machine learning, network assortativity analysis, and benchmarking across cities) with Urbanity. Check out the documentation and tutorials:

- **Tutorial 01** — [Getting Started](notebooks/01_getting_started): Map setup, boundaries, and the automated OSM download workflow.
- **Tutorial 02** — [Street Networks](notebooks/02_street_network): Building and visualising street networks with centrality metrics.
- **Tutorial 03** — [Urban Graph](notebooks/03_urban_graph): Constructing the full heterogeneous UrbanGraph and exploring its layers.
- **Advanced** — [Network Assortativity](notebooks/network_assortativity) and [Graph ML](notebooks/transductive_graph_ml).

Sample datasets and additional notebooks are available at this [repository](https://github.com/winstonyym/urbanity_examples).

</br>

## License

`urbanity` was created by winstonyym. It is licensed under the terms of the MIT license.

</br>

## Credits 

- Logo design: [April Zhu](https://ual.sg/authors/april/)
- Colab notebooks: [Kunihiko Fujiwara](https://ual.sg/authors/kunihiko/)

</br>

--------------------------------------------------------------------------------

</br>
</br>
<p align="center">
  <a href="https://ual.sg/">
    <img src="https://raw.githubusercontent.com/winstonyym/urbanity/main/images/ualsg.jpeg" width = 50% alt="Logo">
  </a>
</p>
