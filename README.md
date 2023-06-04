[![PyPI version](https://badge.fury.io/py/urbanity.svg)](https://badge.fury.io/py/urbanity)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/urbanity)](https://badge.fury.io/py/urbanity)
[![Documentation Status](https://img.shields.io/readthedocs/urbanity)](https://urbanity.readthedocs.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K-6DlBbuQX48WVsxpwAymgLlibPOJHME?usp=sharing)

</br>
</br>

![Urbanity Logo](https://raw.githubusercontent.com/winstonyym/urbanity/main/images/urbanity_black_transparent.png#gh-light-mode-only)
![Urbanity Logo](https://raw.githubusercontent.com/winstonyym/urbanity/main/images/urbanity_white_tranparent.png#gh-dark-mode-only)


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
*How do I install Urbanity?*

Urbanity is built on several geospatial packages (e.g., GeoPandas, ipyleaflet, rasterio, etc.,) that are best compiled through conda-forge. For seamless installation, we provide an environment.yml file for users to create their own conda environment. Please follow the steps below to ensure proper installation:

1) Navigate to a location of choice (e.g., Desktop or project folder).
2) Download environment.yml file and place it at the location of choice (Step 1).
3) Open up a terminal/command prompt and input the following command:

```
$ conda env create -f environment.yml
$ conda activate urbanity
```

4) Installation completed and you should be able to use urbanity without issues.

(Optional) For JupyterLab / JupyterNotebook users, you can additionally add a notebook kernel via:

```
$ python -m ipykernel install --user --name=urbanity
$ jupyter lab
```

</br>

## What can I do with Urbanity?
We demonstrate how you can conduct a diverse range of urban analytical tasks (such as graph machine learning, network assortativity analysis, and benchmarking across cities) with Urbanity. Check out the documentation and examples/tutorials in the documentation site [examples](https://urbanity.readthedocs.io/en/latest/). 

Sample dataset and notebooks to follow examples can be accessed at this [repository](https://github.com/winstonyym/urbanity_examples).

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
