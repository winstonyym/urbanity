# Configuration file for the Sphinx documentation builder.
#
# Full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "urbanity"
copyright = "2024, winstonyym"
author = "winstonyym"
release = "0.2"

# -- General configuration ---------------------------------------------------

# Notebooks are not executed during the build (outputs are pre-rendered).
nb_execution_mode = "off"
nbsphinx_execute = "never"

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# AutoAPI — generate API docs from the src directory
autoapi_dirs = ["../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_python_class_content = "both"   # include __init__ docstring in class docs

# Napoleon settings (Google / NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx — cross-reference external packages
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "geopandas":  ("https://geopandas.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx":   ("https://networkx.org/documentation/stable/", None),
}

# MyST parser options
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when building.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "urbanity_logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}