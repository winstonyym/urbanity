import numpy as np
import pandas as pd
import pydeck as pdk
from ipywidgets import HTML
import numbers

# Static plotting (matplotlib / contextily are optional but greatly improve output)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import warnings

try:
    import contextily as ctx
    _HAS_CONTEXTILY = True
except ImportError:
    _HAS_CONTEXTILY = False

def get_connected_nodes(connected_nodes, connections, connection_type, node_type, node_idx):

    # Get pairs for connection type
    array_pair = connections[connection_type]

    start_node, end_node = connection_type.split('_')[0], connection_type.split('_')[-1]

    # Connections 
    if start_node == node_type:
        connected_nodes[end_node] = list(array_pair[1][np.where(array_pair[0]==node_idx)])

    if end_node == node_type:
        if node_type in connected_nodes:
            connected_nodes[start_node].extend(list(array_pair[0][np.where(array_pair[1]==node_idx)]))
        else:
            connected_nodes[start_node] = list(array_pair[0][np.where(array_pair[1]==node_idx)])

    return connected_nodes

def plot_graph(objects,
               connections,
               node_type="",
               colname="",
               node_id="",
               categorical=False):
    """
    Render a PyDeck view of an UrbanGraph and attach a colour‑bar legend.
    The legend is displayed via the `description` field of the Deck widget.
    """
    if node_type == '':
        node_type = 'plot'
        
    objects_copy = objects.copy()

    objects_copy[node_type][f'{node_type}_id'] = range(len(objects_copy[node_type]))
    
    # Colour palettes -------------------------------------------------
    colour_map_node_types = ['#177e89', '#084c61', '#db3a34', '#ffc857']
    colour_connections   = '#ff6b6b'
    colour_map_pastel    = ['#fec5bb', '#fcd5ce', '#fae1dd', '#f8edeb', '#e8e8e4',
                            '#d8e2dc', '#ece4db', '#ffe5d9', '#ffd7ba', '#fec89a']
    colour_map_vibrant   = ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f',
                            '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1']

    # ----------------------------------------------------------------
    # Default RGBA for each layer type (used unless overridden below)
    # ----------------------------------------------------------------
    if node_type == 'plot':
        building_colour     = "[255,255,255,255]"
        plot_colour         = "[8, 76, 97,20]"
        street_colour       = "[255,255,255,255]"
        intersection_colour = "[255,255,255,255]"
    elif node_type == 'street':
        building_colour     = "[255,255,255,50]"
        plot_colour         = "[255,255,255,255]"
        street_colour       = "[23,126,137,255]"
        intersection_colour = "[255,255,255,255]"
    elif node_type == 'building':
        building_colour     = "[247,235,212,200]"
        plot_colour         = "[0,0,0,10]"
        street_colour       = "[255,255,255,255]"
        intersection_colour = "[255,255,255,255]"
    elif node_type == 'intersection':
        building_colour     = "[255,255,255,50]"
        plot_colour         = "[0,0,0,10]"
        street_colour       = "[255,255,255,255]"
        intersection_colour = "[0,0,0,255]"
    
    # Allow colour override when `colname` points at the layer ----------
    if (colname != '')        & (node_type == 'plot'):        plot_colour         = 'color'
    elif (colname != '')    & (node_type == 'street'):      street_colour       = 'color'
    elif (colname != '')  & (node_type == 'building'):    building_colour     = 'color'
    elif (colname != '') & (node_type == 'intersection'): intersection_colour = 'color'
    
    # ----------------------------------------------------------------
    # Compute fill colour column (`add_gradient_column` must exist)
    # ----------------------------------------------------------------
    if '_id' in colname:
        categorical = True  # treat *_id columns as categories

    if (colname != '') & (categorical != True):
        objects_copy[node_type][colname] = objects_copy[node_type][colname].round(1)

        objects_copy[node_type] = add_gradient_column(
            objects_copy[node_type],
            colname,
            color_stops=colour_map_pastel,
            categorical=categorical
        )
    
    # ----------------------------------------------------------------
    # Selected node & neighbours (optional)
    # ----------------------------------------------------------------
    if node_id != "":
        chosen_row = objects_copy[node_type].iloc[[node_id]].reset_index(drop=True)
        centerx, centery = (chosen_row.geometry[0].centroid.x,
                            chosen_row.geometry[0].centroid.y)
        
        chosen_layer = pdk.Layer(
            "GeoJsonLayer",
            chosen_row,
            opacity=1,
            get_fill_color='[255, 107, 107,200]',
            get_line_color='[0,0,0]',
            line_width_min_pixels=1,
        )
        
        # Collect neighbouring nodes (requires your own helper)
        relevant_edges = [k for k in connections if node_type in k and 'boundary' not in k]
        connected_nodes  = {'plot': [], 'intersection': [], 'street': [], 'building': []}
        for et in relevant_edges:
            connected_nodes = get_connected_nodes(connected_nodes,
                                                  connections,
                                                  et,
                                                  node_type,
                                                  node_id)
        neighbour_layers = []
        for ntype, neigh_idx in connected_nodes.items():
            if not neigh_idx:
                continue
            gdf = objects[ntype].iloc[neigh_idx]
            layer_args = dict(data=gdf,
                              opacity=1,
                              get_fill_color='[255, 107, 107,100]',
                              get_line_color='[0,0,0]',
                              line_width_min_pixels=1)
            if ntype == 'building':
                layer_args.update(extruded=True, get_elevation="bid_height")
            neighbour_layers.append(pdk.Layer("GeoJsonLayer", **layer_args))

    else:
        chosen_layer     = None
        neighbour_layers = []
        centerx, centery = (objects['boundary'].geometry[0].centroid.x,
                            objects['boundary'].geometry[0].centroid.y)
    
    # ----------------------------------------------------------------
    # Deck.gl layers
    # ----------------------------------------------------------------
    # pickable = True if 
    add_pickable = {"pickable": True, "auto_highlight": True}
    
    plot_layer = pdk.Layer(
        "GeoJsonLayer",
        objects_copy['plot'],
        opacity=1,
        get_fill_color=plot_colour,
        get_line_color=plot_colour,
        line_width_min_pixels=2,
        **(add_pickable if node_type == 'plot' else {})
    )
    
    street_layer = pdk.Layer(
        "GeoJsonLayer",
        objects_copy['street'],
        get_line_color=street_colour,
        line_width_min_pixels=3,
        **(add_pickable if node_type == 'street' else {})
    )
    
    intersection_layer = pdk.Layer(
        "ScatterplotLayer",
        objects_copy['intersection'],
        get_position='[x,y]',
        get_fill_color=intersection_colour,
        get_radius=4,
        **(add_pickable if node_type == 'intersection' else {})
    )
    
    building_layer = pdk.Layer(
        "GeoJsonLayer",
        objects_copy['building'],
        extruded=True,
        get_elevation="bid_height",
        get_fill_color=building_colour,
        get_line_color=building_colour,
        **(add_pickable if node_type == 'building' else {})
    )

        
    # Layer draw order
    layers = [building_layer, plot_layer, intersection_layer, street_layer]
    
    # ----------------------------------------------------------------
    # View state
    # ----------------------------------------------------------------
    view_state = pdk.ViewState(
        latitude=centery,
        longitude=centerx,
        zoom=14,
        pitch=45,
        bearing=135,
        height=1000,
        width=500,
    )

    tooltip = {
       "html": f"<b>{node_type} ID:</b> " + "{" +  f'{node_type}_id' + "}",
    }
    
    if colname != '':
        col_title = colname.replace("_", " ").title()
        tooltip = {
           "html": f"<b>{col_title}:</b> " + "{" +  f'{colname}' + "}",
        }
        
        # ----------------------------------------------------------------
        # Build legend HTML
        # ----------------------------------------------------------------
        if categorical:
            cats = (objects_copy[node_type][colname]
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist())
            colours = colour_map_pastel[:len(cats)]
            legend_html = build_html_legend(cats, colours,
                                            title=col_title,
                                            continuous=False)
        else:
            series = objects_copy[node_type][colname]
            minmax = [series.min(), series.max()]
            legend_html = build_html_legend(minmax, colour_map_pastel,
                                            title=col_title,
                                            continuous=True)
    
    # ----------------------------------------------------------------
    # Assemble Deck
    # ----------------------------------------------------------------
    if node_id != '':
        extra_layers = [l for l in ([chosen_layer] + neighbour_layers) if l is not None]
        deck_layers  = layers + extra_layers
    else:
        deck_layers = layers
        
    deck = pdk.Deck(
        layers=deck_layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark_all",
        tooltip = tooltip
    )
    
    if colname != '':
        display(deck,HTML(legend_html))    
    else:
        display(deck)


def hex_to_rgb(hex_color: str):
    """
    Convert a hex color string (e.g. '#edafb8') to an (R, G, B) tuple (0-255 each).
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def interpolate_color(color1, color2, fraction: float):
    """
    Linearly interpolate between two colors (each an (R, G, B) tuple) by the given fraction in [0,1].
    Returns an (R, G, B) tuple.
    """
    r = int(color1[0] + (color2[0] - color1[0]) * fraction)
    g = int(color1[1] + (color2[1] - color1[1]) * fraction)
    b = int(color1[2] + (color2[2] - color1[2]) * fraction)
    return (r, g, b)

def pick_color(value, min_val, max_val, color_stops):
    """
    Given a numeric value within [min_val, max_val], pick a color by piecewise linear interpolation
    among the given color stops (list of hex codes).
    """
    # Handle degenerate case (all values are the same)
    if max_val == min_val:
        return hex_to_rgb(color_stops[0])
    
    # Normalize value to a fraction t in [0, 1]
    t = (value - min_val) / (max_val - min_val)
    t = max(0, min(t, 1))  # clamp to [0, 1]
    
    num_segments = len(color_stops) - 1
    scaled_t = t * num_segments
    
    # Identify segment indices
    idx1 = int(scaled_t)
    idx2 = min(idx1 + 1, num_segments)
    
    # Fraction within this segment
    segment_fraction = scaled_t - idx1
    
    # Convert hex stops to RGB
    c1 = hex_to_rgb(color_stops[idx1])
    c2 = hex_to_rgb(color_stops[idx2])
    
    return interpolate_color(c1, c2, segment_fraction)

def add_gradient_column(
    df: pd.DataFrame,
    target_col: str,
    color_stops: list,
    new_col_name: str = 'color',
    categorical: bool = False
) -> pd.DataFrame:
    """
    Return a copy of df with a new column that contains color values (as (R,G,B) tuples).
    
    If `categorical=False`:
      - Interpolate among the given color_stops over the numeric range of `target_col`.
    
    If `categorical=True`:
      - Assign colors by repeating the color_stops in sequence for each unique category.
    """
    df_copy = df.copy()
    

    if categorical:
        # For categorical data, assign colors in a repeating sequence
        df_copy = df_copy.reset_index()
        unique_cats = df_copy[target_col].unique()
        cat_to_color = {}
        
        for i, cat in enumerate(unique_cats):
            # Cycle through color_stops by taking i % len(color_stops)
            cat_to_color[cat] = hex_to_rgb(color_stops[i % len(color_stops)])
        
        df_copy[new_col_name] = df_copy[target_col].map(cat_to_color)
    
    else:
        # For continuous data, use the numeric color interpolation
        min_val = df_copy[target_col].min()
        max_val = df_copy[target_col].max()
        
        df_copy[new_col_name] = df_copy[target_col].apply(
            lambda val: pick_color(val, min_val, max_val, color_stops)
        )
    
    return df_copy

def build_html_legend(values, colours, *, title="", continuous=False):
    """
    Return an HTML <div> containing a legend.
    See doc‑string in original code for parameters.
    """
    if continuous:                         # gradient bar ---------------------
        gradient_css = ", ".join(colours)
        min_val, max_val = values

        # helper: format only if numeric
        def _fmt(v):
            return f"{v:.2f}" if isinstance(v, numbers.Number) else str(v)

        legend = f"""
        <div style="padding:8px;background:rgba(0,0,0,0.7);border-radius:4px;
                    color:#fff;font-family:Arial;font-size:11px;">
          <div style="font-weight:bold;margin-bottom:4px;">{title}</div>
          <div style="display:flex;align-items:center;gap:4px;">
            <span>{_fmt(min_val)}</span>
            <div style="flex:1;height:12px;background:
                        linear-gradient(to right,{gradient_css});"></div>
            <span>{_fmt(max_val)}</span>
          </div>
        </div>"""
    else:                                  # discrete swatches ---------------
        rows = "".join(
            f"<div style='display:flex;align-items:center;margin-bottom:2px;'>"
            f"  <div style='width:12px;height:12px;background:{c};"
            f"       margin-right:4px;'></div>{v}</div>"
            for v, c in zip(values, colours)
        )
        legend = f"""
        <div style="padding:8px;background:rgba(0,0,0,0.7);border-radius:4px;
                    color:#fff;font-family:Arial;font-size:11px;">
          <div style="font-weight:bold;margin-bottom:4px;">{title}</div>
          {rows}
        </div>"""
    return legend.strip()


# ---------------------------------------------------------------------------
# Static aesthetic plotting helpers
# ---------------------------------------------------------------------------

# Default dark-theme palette used across static plots
_DARK_BG   = "#0d1117"   # near-black background
_STREET_C  = "#e8c27a"   # warm amber for streets
_NODE_C    = "#f05454"   # coral-red for intersections
_BUILD_C   = "#a8dadc"   # sky-blue for buildings
_PLOT_C    = "#457b9d"   # steel-blue for urban plots
_CMAP_CONT = "YlOrRd"    # sequential colour-map for continuous attributes
_CMAP_CAT  = "tab20"     # categorical colour-map


def _ensure_projected(gdf):
    """Return the GeoDataFrame in a local metric CRS (UTM or Web-Mercator)."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    return gdf


def plot_street_network(
    nodes,
    edges,
    colname: str = "",
    cmap: str = _CMAP_CONT,
    node_size: float = 6,
    edge_linewidth: float = 0.8,
    figsize: tuple = (10, 10),
    title: str = "Street Network",
    dark_mode: bool = True,
    basemap: bool = False,
    save_path: str = "",
) -> plt.Figure:
    """Render a publication-quality static map of a street network.

    The function accepts the ``nodes`` and ``edges`` GeoDataFrames returned by
    :py:meth:`~urbanity.urbanity.Map.get_street_network` and produces a
    matplotlib figure styled with a dark background by default.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame
        Point GeoDataFrame of street intersections.
    edges : geopandas.GeoDataFrame
        LineString GeoDataFrame of street segments.
    colname : str, optional
        Column in ``edges`` (or ``nodes``) to colour-code.  When empty the
        network is drawn with a single colour scheme.  Defaults to ``""``.
    cmap : str, optional
        Matplotlib colour-map name used when ``colname`` is set.
        Defaults to ``"YlOrRd"``.
    node_size : float, optional
        Size of intersection markers. Defaults to ``6``.
    edge_linewidth : float, optional
        Width of street lines. Defaults to ``0.8``.
    figsize : tuple, optional
        Figure size in inches. Defaults to ``(10, 10)``.
    title : str, optional
        Plot title. Defaults to ``"Street Network"``.
    dark_mode : bool, optional
        If ``True`` (default), uses a dark background theme.
    basemap : bool, optional
        If ``True`` and ``contextily`` is installed, adds an OpenStreetMap tile
        basemap underneath.  Requires the data to be re-projected to Web Mercator
        (EPSG:3857). Defaults to ``False``.
    save_path : str, optional
        File path to save the figure (e.g. ``"output.png"``).  If empty the
        figure is only returned, not saved. Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    bg    = _DARK_BG if dark_mode else "white"
    fg    = "white"  if dark_mode else "black"
    e_col = _STREET_C if dark_mode else "#1a1a2e"
    n_col = _NODE_C   if dark_mode else "#e63946"

    # Ensure metric CRS
    edges_p = _ensure_projected(edges.copy())
    nodes_p = _ensure_projected(nodes.copy())

    if basemap:
        edges_p = edges_p.to_crs(epsg=3857)
        nodes_p = nodes_p.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    # --- edges ---------------------------------------------------------------
    if colname and colname in edges_p.columns:
        edges_p.plot(
            ax=ax,
            column=colname,
            cmap=cmap,
            linewidth=edge_linewidth,
            legend=True,
            legend_kwds={
                "shrink": 0.5,
                "label": colname.replace("_", " ").title(),
                "orientation": "horizontal",
                "pad": 0.02,
            },
        )
    else:
        edges_p.plot(ax=ax, color=e_col, linewidth=edge_linewidth, alpha=0.85)

    # --- nodes ---------------------------------------------------------------
    if colname and colname in nodes_p.columns:
        nodes_p.plot(
            ax=ax,
            column=colname,
            cmap=cmap,
            markersize=node_size,
            alpha=0.9,
        )
    else:
        nodes_p.plot(ax=ax, color=n_col, markersize=node_size, alpha=0.9)

    # --- basemap -------------------------------------------------------------
    if basemap:
        if _HAS_CONTEXTILY:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatterNoLabels
                            if dark_mode else ctx.providers.CartoDB.Positron,
                            alpha=0.4)
        else:
            warnings.warn("contextily is not installed; basemap skipped. "
                          "Install with: pip install contextily")

    # --- styling -------------------------------------------------------------
    ax.set_axis_off()
    ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=12)

    # legend entries when no colname
    if not colname:
        legend_elements = [
            Line2D([0], [0], color=e_col, lw=1.5, label="Street"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=n_col,
                   markersize=6, label="Intersection"),
        ]
        ax.legend(handles=legend_elements, facecolor="#1e2329" if dark_mode else "white",
                  labelcolor=fg, framealpha=0.8, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    return fig


def plot_buildings(
    buildings,
    colname: str = "",
    cmap: str = _CMAP_CONT,
    boundary=None,
    figsize: tuple = (10, 10),
    title: str = "Building Footprints",
    dark_mode: bool = True,
    save_path: str = "",
) -> plt.Figure:
    """Render a static choropleth of building footprints.

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Polygon GeoDataFrame of building footprints, as returned by
        :py:meth:`~urbanity.urbanity.Map.get_building_layer`.
    colname : str, optional
        Column to use for colour-coding (e.g. ``"bid_area"`` or
        ``"bid_height"``). When empty all buildings share one colour.
        Defaults to ``""``.
    cmap : str, optional
        Matplotlib colour-map name. Defaults to ``"YlOrRd"``.
    boundary : geopandas.GeoDataFrame, optional
        Study area boundary polygon to overlay as a reference frame.
    figsize : tuple, optional
        Figure size in inches. Defaults to ``(10, 10)``.
    title : str, optional
        Plot title. Defaults to ``"Building Footprints"``.
    dark_mode : bool, optional
        Use dark background theme. Defaults to ``True``.
    save_path : str, optional
        If provided, saves the figure to this path. Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bg    = _DARK_BG if dark_mode else "white"
    fg    = "white"  if dark_mode else "black"
    b_col = _BUILD_C if dark_mode else "#457b9d"

    bldg_p = _ensure_projected(buildings.copy())

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    # Boundary outline (optional)
    if boundary is not None:
        bound_p = _ensure_projected(boundary.copy())
        bound_p.boundary.plot(ax=ax, color=fg, linewidth=1.2, linestyle="--", alpha=0.5)

    # Buildings
    if colname and colname in bldg_p.columns:
        bldg_p.plot(
            ax=ax,
            column=colname,
            cmap=cmap,
            linewidth=0,
            legend=True,
            legend_kwds={
                "shrink": 0.4,
                "label": colname.replace("_", " ").title(),
                "orientation": "horizontal",
                "pad": 0.02,
            },
        )
    else:
        bldg_p.plot(ax=ax, color=b_col, edgecolor="none", alpha=0.85)

    ax.set_axis_off()
    ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=12)

    # Summary annotation
    n_bldgs = len(bldg_p)
    ax.annotate(
        f"{n_bldgs:,} buildings",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color=fg, alpha=0.7,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    return fig


def plot_urban_graph_overview(
    objects: dict,
    figsize: tuple = (14, 10),
    title: str = "Urban Graph Overview",
    dark_mode: bool = True,
    save_path: str = "",
) -> plt.Figure:
    """Render a four-panel overview of an UrbanGraph's node layers.

    Each panel shows one node type (urban plots, buildings, streets,
    intersections) coloured distinctively, giving an at-a-glance summary of the
    heterogeneous graph returned by
    :py:meth:`~urbanity.urbanity.Map.get_urban_graph`.

    Parameters
    ----------
    objects : dict
        The ``objects`` dictionary returned by ``get_urban_graph``, with keys
        ``"plot"``, ``"building"``, ``"street"``, and ``"intersection"``.
    figsize : tuple, optional
        Overall figure size. Defaults to ``(14, 10)``.
    title : str, optional
        Super-title for the figure. Defaults to ``"Urban Graph Overview"``.
    dark_mode : bool, optional
        Use dark background theme. Defaults to ``True``.
    save_path : str, optional
        If provided, saves the figure to this path. Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bg = _DARK_BG if dark_mode else "white"
    fg = "white"   if dark_mode else "black"

    layer_cfg = [
        ("plot",         objects.get("plot"),         _PLOT_C,   "Urban Plots"),
        ("building",     objects.get("building"),     _BUILD_C,  "Buildings"),
        ("street",       objects.get("street"),       _STREET_C, "Streets"),
        ("intersection", objects.get("intersection"), _NODE_C,   "Intersections"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=bg)
    fig.suptitle(title, color=fg, fontsize=16, fontweight="bold", y=1.01)

    for ax, (key, gdf, colour, label) in zip(axes.flat, layer_cfg):
        ax.set_facecolor(bg)
        ax.set_axis_off()
        ax.set_title(label, color=fg, fontsize=11, pad=6)

        if gdf is None or len(gdf) == 0:
            ax.annotate("No data", xy=(0.5, 0.5), xycoords="axes fraction",
                        ha="center", color=fg, fontsize=10, alpha=0.5)
            continue

        gdf_p = _ensure_projected(gdf.copy())

        if key == "intersection":
            gdf_p.plot(ax=ax, color=colour, markersize=3, alpha=0.85)
        elif key == "street":
            gdf_p.plot(ax=ax, color=colour, linewidth=0.7, alpha=0.85)
        else:
            gdf_p.plot(ax=ax, color=colour, edgecolor="none", alpha=0.8)

        # Layer count annotation
        ax.annotate(
            f"{len(gdf_p):,} features",
            xy=(0.03, 0.04), xycoords="axes fraction",
            fontsize=8, color=fg, alpha=0.7,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    return fig


def plot_node_attribute(
    gdf,
    colname: str,
    geometry_type: str = "auto",
    cmap: str = _CMAP_CONT,
    categorical: bool = False,
    boundary=None,
    figsize: tuple = (10, 10),
    title: str = "",
    dark_mode: bool = True,
    save_path: str = "",
) -> plt.Figure:
    """Render a single attribute as a choropleth or graduated-symbol map.

    Works with any node-type GeoDataFrame (plots, buildings, streets,
    intersections) and automatically selects an appropriate geometry rendering
    based on ``geometry_type``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the attribute to visualise.
    colname : str
        Name of the column to colour-code.
    geometry_type : str, optional
        One of ``"polygon"``, ``"line"``, ``"point"``, or ``"auto"``.
        When ``"auto"`` the type is inferred from the GeoDataFrame's geometry
        column.  Defaults to ``"auto"``.
    cmap : str, optional
        Matplotlib colour-map name for continuous data. Defaults to ``"YlOrRd"``.
    categorical : bool, optional
        If ``True``, treats ``colname`` as a categorical variable and uses
        ``tab20``. Defaults to ``False``.
    boundary : geopandas.GeoDataFrame, optional
        Study area boundary to draw as a reference frame. Defaults to ``None``.
    figsize : tuple, optional
        Figure size in inches. Defaults to ``(10, 10)``.
    title : str, optional
        Plot title.  Defaults to the column name.
    dark_mode : bool, optional
        Use dark background theme. Defaults to ``True``.
    save_path : str, optional
        If provided, saves the figure to this path. Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bg = _DARK_BG if dark_mode else "white"
    fg = "white"   if dark_mode else "black"

    if title == "":
        title = colname.replace("_", " ").title()

    _cmap = _CMAP_CAT if categorical else cmap

    gdf_p = _ensure_projected(gdf.copy())

    # Infer geometry type
    if geometry_type == "auto":
        sample_geom = gdf_p.geometry.iloc[0]
        gname = type(sample_geom).__name__.lower()
        if "polygon" in gname:
            geometry_type = "polygon"
        elif "linestring" in gname or "line" in gname:
            geometry_type = "line"
        else:
            geometry_type = "point"

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    if boundary is not None:
        bound_p = _ensure_projected(boundary.copy())
        bound_p.boundary.plot(ax=ax, color=fg, linewidth=1.0, linestyle="--", alpha=0.4)

    plot_kwargs = dict(
        ax=ax,
        column=colname,
        cmap=_cmap,
        legend=True,
        legend_kwds={
            "shrink": 0.4,
            "label": title,
            "orientation": "horizontal",
            "pad": 0.03,
        },
    )

    if geometry_type == "line":
        plot_kwargs["linewidth"] = 1.2
        gdf_p.plot(**plot_kwargs)
    elif geometry_type == "point":
        plot_kwargs["markersize"] = 6
        gdf_p.plot(**plot_kwargs)
    else:
        plot_kwargs["edgecolor"] = "none"
        gdf_p.plot(**plot_kwargs)

    ax.set_axis_off()
    ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=12)

    # Colour-bar text colour fix (matplotlib uses rcParams)
    for text in ax.get_children():
        if isinstance(text, matplotlib.text.Text):
            text.set_color(fg)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    return fig


def plot_network_centrality(
    nodes,
    edges,
    centrality_col: str = "betweenness",
    figsize: tuple = (10, 10),
    title: str = "",
    dark_mode: bool = True,
    save_path: str = "",
) -> plt.Figure:
    """Render a street network with edges or nodes sized/coloured by centrality.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame
        Intersection GeoDataFrame; must contain ``centrality_col``.
    edges : geopandas.GeoDataFrame
        Street edge GeoDataFrame (drawn as a faint underlay).
    centrality_col : str, optional
        Column containing centrality scores. Defaults to ``"betweenness"``.
    figsize : tuple, optional
        Figure size in inches. Defaults to ``(10, 10)``.
    title : str, optional
        Plot title.  Defaults to the column name.
    dark_mode : bool, optional
        Use dark background theme. Defaults to ``True``.
    save_path : str, optional
        If provided, saves the figure to this path. Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bg = _DARK_BG if dark_mode else "white"
    fg = "white"   if dark_mode else "black"
    if title == "":
        title = centrality_col.replace("_", " ").title()

    edges_p = _ensure_projected(edges.copy())
    nodes_p = _ensure_projected(nodes.copy())

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    # Faint edge underlay
    edges_p.plot(ax=ax,
                 color="#3a3f4b" if dark_mode else "#cccccc",
                 linewidth=0.5, alpha=0.7)

    if centrality_col in nodes_p.columns:
        # Normalise sizes
        vals = nodes_p[centrality_col].fillna(0)
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        sizes = (norm * 80 + 2).values

        scatter = ax.scatter(
            nodes_p.geometry.x,
            nodes_p.geometry.y,
            c=vals,
            s=sizes,
            cmap=_CMAP_CONT,
            alpha=0.85,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02,
                            orientation="horizontal")
        cbar.set_label(title, color=fg, fontsize=9)
        cbar.ax.xaxis.set_tick_params(color=fg)
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color=fg)
    else:
        nodes_p.plot(ax=ax, color=_NODE_C, markersize=4, alpha=0.8)
        ax.annotate(
            f"Column '{centrality_col}' not found — showing raw intersections",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, color=fg, alpha=0.7,
        )

    ax.set_axis_off()
    ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    return fig

# ---------------------------------------------------------------------------
# Heterogeneous Urban Graph — nodes + edges  (2-D and 3-D)
# ---------------------------------------------------------------------------

# Per-layer colour palette (hex → for matplotlib; RGBA list → for pydeck)
_LAYER_COLOURS = {
    "plot":         {"hex": "#457b9d", "rgba": [69,  123, 157, 180]},
    "building":     {"hex": "#a8dadc", "rgba": [168, 218, 220, 220]},
    "street":       {"hex": "#e8c27a", "rgba": [232, 194, 122, 220]},
    "intersection": {"hex": "#f05454", "rgba": [240,  84,  84, 255]},
}

# Per-edge-type colour (hex → for matplotlib; RGBA list → for pydeck)
_EDGE_COLOURS = {
    "plot_plot":             {"hex": "#74b9ff", "rgba": [116, 185, 255, 200]},
    "plot_building":         {"hex": "#fd79a8", "rgba": [253, 121, 168, 200]},
    "plot_street":           {"hex": "#55efc4", "rgba": [ 85, 239, 196, 200]},
    "building_building":     {"hex": "#fdcb6e", "rgba": [253, 203, 110, 200]},
    "building_street":       {"hex": "#e17055", "rgba": [225, 112,  85, 200]},
    "street_intersection":   {"hex": "#a29bfe", "rgba": [162, 155, 254, 200]},
    "plot_intersection":     {"hex": "#00cec9", "rgba": [  0, 206, 201, 180]},
    "building_intersection": {"hex": "#6c5ce7", "rgba": [108,  92, 231, 180]},
}
_EDGE_DEFAULT = {"hex": "#dfe6e9", "rgba": [223, 230, 233, 160]}


def _connections_to_edge_geodataframe(objects, connections):
    """Convert all connection arrays to a GeoDataFrame of LineStrings.

    Each row carries the edge type so it can be coloured distinctly.
    Returns a GeoDataFrame in EPSG:4326 (or ``None`` if nothing could be built).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = []
    for edge_type, arr in connections.items():
        if arr is None or not hasattr(arr, "shape") or arr.shape[1] == 0:
            continue

        # Determine source and target layer names from the edge type string
        parts = edge_type.split("_")
        if len(parts) < 2:
            continue

        # Try 'a_b' or 'a_boundary_b' etc. — match against known layer keys
        layer_keys = list(objects.keys())
        src_key = dst_key = None
        for lk in layer_keys:
            if edge_type.startswith(lk):
                src_key = lk
                remainder = edge_type[len(lk)+1:]
                if remainder in layer_keys:
                    dst_key = remainder
                break
        if src_key is None or dst_key is None:
            # fall back: first part / last part
            src_key = parts[0]
            dst_key = parts[-1]

        src_gdf = objects.get(src_key)
        dst_gdf = objects.get(dst_key)
        if src_gdf is None or dst_gdf is None or len(src_gdf) == 0 or len(dst_gdf) == 0:
            continue

        # Build centroids for geometry lookup
        try:
            src_centroids = src_gdf.geometry.centroid.reset_index(drop=True)
            dst_centroids = dst_gdf.geometry.centroid.reset_index(drop=True)
        except Exception:
            continue

        src_idx, dst_idx = arr[0], arr[1]

        # Guard against out-of-range indices
        max_src = len(src_centroids) - 1
        max_dst = len(dst_centroids) - 1
        valid = (src_idx <= max_src) & (dst_idx <= max_dst)
        src_idx = src_idx[valid]
        dst_idx = dst_idx[valid]

        if len(src_idx) == 0:
            continue

        sx = src_centroids.iloc[src_idx].x.values
        sy = src_centroids.iloc[src_idx].y.values
        dx = dst_centroids.iloc[dst_idx].x.values
        dy = dst_centroids.iloc[dst_idx].y.values

        for s_x, s_y, d_x, d_y in zip(sx, sy, dx, dy):
            rows.append({"edge_type": edge_type,
                         "geometry": LineString([(s_x, s_y), (d_x, d_y)])})

    if not rows:
        return None

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf

def plot_urban_graph_edges(
    objects: dict,
    connections: dict,
    mode: str = "2d",
    show_nodes: bool = True,
    show_edges: bool = True,
    edge_types: list = None,
    node_types: list = None,
    pitch: float = 45.0,
    bearing: float = 135.0,
    zoom: int = 14,
    map_style: str = "dark_all",
    figsize: tuple = (12, 12),
    title: str = "Urban Graph — Nodes & Edges",
    dark_mode: bool = True,
    node_size_2d: float = 5.0,
    edge_linewidth_2d: float = 0.6,
    save_path: str = "",
):
    """Visualise UrbanGraph node layers **and** their inter-layer edges.

    Supports two rendering modes:

    * ``mode="2d"`` — a flat matplotlib figure with each node layer and edge type
      drawn in a distinct colour and a full legend.
    * ``mode="3d"`` — an interactive PyDeck Deck.gl view rendered inside Jupyter,
      with building polygons extruded by height and edge arcs drawn as thin lines.

    Parameters
    ----------
    objects : dict
        The ``objects`` dictionary returned by
        :py:meth:`~urbanity.urbanity.Map.get_urban_graph`, containing keys such as
        ``"plot"``, ``"building"``, ``"street"``, ``"intersection"``, and
        ``"boundary"``.
    connections : dict
        The ``connections`` dictionary returned by ``get_urban_graph``, mapping
        edge-type strings (e.g. ``"plot_building"``) to NumPy arrays of shape
        ``(2, E)``.
    mode : {"2d", "3d"}, optional
        Rendering backend.  ``"2d"`` uses matplotlib (static); ``"3d"`` uses
        PyDeck (interactive, Jupyter only).  Defaults to ``"2d"``.
    show_nodes : bool, optional
        Whether to render node layers.  Defaults to ``True``.
    show_edges : bool, optional
        Whether to render inter-layer edges.  Defaults to ``True``.
    edge_types : list of str, optional
        Subset of edge types to draw (e.g. ``["plot_building", "street_intersection"]``).
        When ``None`` all available edge types are shown.  Defaults to ``None``.
    node_types : list of str, optional
        Subset of node layer keys to render (e.g. ``["plot", "building"]``).
        When ``None`` all four standard layers are shown.  Defaults to ``None``.
    pitch : float, optional
        Camera pitch in degrees for 3-D mode.  Defaults to ``45.0``.
    bearing : float, optional
        Camera bearing in degrees for 3-D mode.  Defaults to ``135.0``.
    zoom : int, optional
        Initial zoom level for 3-D mode.  Defaults to ``14``.
    map_style : str, optional
        PyDeck / CARTO map style for 3-D mode.  Common values are
        ``"dark_all"``, ``"light_all"``, ``"voyager"``.  Defaults to ``"dark_all"``.
    figsize : tuple, optional
        Figure size for 2-D mode.  Defaults to ``(12, 12)``.
    title : str, optional
        Figure / deck title.  Defaults to ``"Urban Graph — Nodes & Edges"``.
    dark_mode : bool, optional
        Dark background for 2-D mode.  Defaults to ``True``.
    node_size_2d : float, optional
        Marker size for point layers (intersections) in 2-D mode.
        Defaults to ``5.0``.
    edge_linewidth_2d : float, optional
        Line width for edge geometries in 2-D mode.  Defaults to ``0.6``.
    save_path : str, optional
        If non-empty, the 2-D figure is saved to this path (ignored in 3-D mode).
        Defaults to ``""``.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure when ``mode="2d"``.
    pydeck.Deck
        The interactive deck widget when ``mode="3d"``.

    Examples
    --------
    **2-D static map**

    >>> fig = plot_urban_graph_edges(objects, connections, mode="2d")

    **3-D interactive map (Jupyter)**

    >>> deck = plot_urban_graph_edges(objects, connections, mode="3d")

    **Show only plot→building and street→intersection edges**

    >>> fig = plot_urban_graph_edges(
    ...     objects, connections,
    ...     edge_types=["plot_building", "street_intersection"],
    ...     mode="2d",
    ... )
    """
    _node_types_default = ["plot", "building", "street", "intersection"]
    active_node_types = node_types if node_types is not None else _node_types_default
    active_edge_types = edge_types  # None → show all

    # ------------------------------------------------------------------
    # Build edge GeoDataFrame (shared by both modes)
    # ------------------------------------------------------------------
    edge_gdf = None
    if show_edges and connections:
        # Optionally filter to requested edge types
        filtered_connections = {
            k: v for k, v in connections.items()
            if (active_edge_types is None or k in active_edge_types)
        }
        edge_gdf = _connections_to_edge_geodataframe(objects, filtered_connections)

    # =====================================================================
    # 2-D  matplotlib rendering
    # =====================================================================
    if mode == "2d":
        bg = _DARK_BG if dark_mode else "white"
        fg = "white"  if dark_mode else "black"

        fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
        ax.set_facecolor(bg)
        legend_handles = []

        # --- boundary outline (optional) ---------------------------------
        if "boundary" in objects and objects["boundary"] is not None:
            bound = _ensure_projected(objects["boundary"].copy())
            bound.boundary.plot(
                ax=ax, color=fg, linewidth=1.0, linestyle="--", alpha=0.35, zorder=1
            )

        # --- edge lines --------------------------------------------------
        if show_edges and edge_gdf is not None and len(edge_gdf) > 0:
            edge_gdf_p = _ensure_projected(edge_gdf.copy())
            drawn_types = edge_gdf_p["edge_type"].unique()
            for et in drawn_types:
                subset = edge_gdf_p[edge_gdf_p["edge_type"] == et]
                col = _EDGE_COLOURS.get(et, _EDGE_DEFAULT)["hex"]
                label = et.replace("_", " → ")
                subset.plot(
                    ax=ax,
                    color=col,
                    linewidth=edge_linewidth_2d,
                    alpha=0.55,
                    zorder=2,
                )
                legend_handles.append(
                    Line2D([0], [0], color=col, lw=1.5, label=f"Edge: {label}")
                )

        # --- node layers -------------------------------------------------
        if show_nodes:
            for layer_key in active_node_types:
                gdf = objects.get(layer_key)
                if gdf is None or len(gdf) == 0:
                    continue
                col = _LAYER_COLOURS.get(layer_key, {"hex": "#ffffff"})["hex"]
                gdf_p = _ensure_projected(gdf.copy())
                geom_type = gdf_p.geometry.iloc[0].geom_type.lower()

                if "point" in geom_type:
                    gdf_p.plot(
                        ax=ax, color=col, markersize=node_size_2d,
                        alpha=0.9, zorder=4
                    )
                elif "linestring" in geom_type or "line" in geom_type:
                    gdf_p.plot(
                        ax=ax, color=col, linewidth=1.0,
                        alpha=0.85, zorder=3
                    )
                else:
                    gdf_p.plot(
                        ax=ax, color=col, edgecolor="none",
                        alpha=0.65, zorder=3
                    )

                legend_handles.append(
                    mpatches.Patch(
                        color=col,
                        label=f"Nodes: {layer_key.title()}",
                        alpha=0.85,
                    )
                )

        # --- legend & labels ---------------------------------------------
        if legend_handles:
            legend = ax.legend(
                handles=legend_handles,
                loc="lower right",
                facecolor="#1e2329" if dark_mode else "white",
                labelcolor=fg,
                framealpha=0.85,
                fontsize=8,
                title="Layer / Edge Type",
                title_fontsize=8,
            )
            legend.get_title().set_color(fg)

        ax.set_axis_off()
        ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=12)

        # Node / edge count annotation
        n_nodes = sum(
            len(objects[k]) for k in active_node_types
            if k in objects and objects[k] is not None
        )
        n_edges = len(edge_gdf) if edge_gdf is not None else 0
        ax.annotate(
            f"{n_nodes:,} nodes  |  {n_edges:,} edges",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, color=fg, alpha=0.65,
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
        return fig

    # =====================================================================
    # 3-D  PyDeck rendering
    # =====================================================================
    elif mode == "3d":
        import geopandas as gpd

        # Centre on boundary centroid
        if "boundary" in objects and objects["boundary"] is not None:
            cx = objects["boundary"].geometry.iloc[0].centroid.x
            cy = objects["boundary"].geometry.iloc[0].centroid.y
        else:
            # Fall back to mean of all node centroids
            all_cx, all_cy = [], []
            for k in active_node_types:
                gdf = objects.get(k)
                if gdf is not None and len(gdf) > 0:
                    all_cx.extend(gdf.geometry.centroid.x.tolist())
                    all_cy.extend(gdf.geometry.centroid.y.tolist())
            cx = float(np.mean(all_cx)) if all_cx else 0.0
            cy = float(np.mean(all_cy)) if all_cy else 0.0

        view_state = pdk.ViewState(
            latitude=cy,
            longitude=cx,
            zoom=zoom,
            pitch=pitch,
            bearing=bearing,
            height=800,
            width=600,
        )

        layers = []

        # --- edge LineString layer ----------------------------------------
        if show_edges and edge_gdf is not None and len(edge_gdf) > 0:
            # Build one LineLayer per edge type for distinct colours
            for et in edge_gdf["edge_type"].unique():
                if active_edge_types is not None and et not in active_edge_types:
                    continue
                subset = edge_gdf[edge_gdf["edge_type"] == et].copy()
                rgba = _EDGE_COLOURS.get(et, _EDGE_DEFAULT)["rgba"]
                layers.append(
                    pdk.Layer(
                        "GeoJsonLayer",
                        subset,
                        get_line_color=rgba,
                        line_width_min_pixels=1,
                        pickable=False,
                        opacity=0.6,
                    )
                )

        # --- node layers --------------------------------------------------
        if show_nodes:
            for layer_key in active_node_types:
                gdf = objects.get(layer_key)
                if gdf is None or len(gdf) == 0:
                    continue
                rgba = _LAYER_COLOURS.get(layer_key, {"rgba": [200, 200, 200, 200]})["rgba"]
                geom_type = gdf.geometry.iloc[0].geom_type.lower()

                if layer_key == "building":
                    # Extrude buildings when height column exists
                    gdf_copy = gdf.copy()
                    has_height = "bid_height" in gdf_copy.columns
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            gdf_copy,
                            extruded=has_height,
                            get_elevation="bid_height" if has_height else 0,
                            elevation_scale=1,
                            get_fill_color=rgba,
                            get_line_color=rgba,
                            pickable=True,
                            auto_highlight=True,
                            opacity=0.75,
                        )
                    )
                elif "point" in geom_type:
                    # Intersections → ScatterplotLayer
                    gdf_xy = gdf.copy()
                    gdf_xy["x"] = gdf_xy.geometry.x
                    gdf_xy["y"] = gdf_xy.geometry.y
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            gdf_xy,
                            get_position="[x, y]",
                            get_fill_color=rgba,
                            get_radius=6,
                            pickable=True,
                            auto_highlight=True,
                        )
                    )
                elif "linestring" in geom_type or "line" in geom_type:
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            gdf,
                            get_line_color=rgba,
                            line_width_min_pixels=2,
                            pickable=True,
                            auto_highlight=True,
                        )
                    )
                else:
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            gdf,
                            get_fill_color=rgba,
                            get_line_color=rgba,
                            line_width_min_pixels=1,
                            pickable=True,
                            auto_highlight=True,
                            opacity=0.65,
                        )
                    )

        tooltip = {"html": "<b>Layer:</b> {layer_key}"}

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="carto",
            map_style=map_style,
            tooltip=tooltip,
        )

        # Build an HTML legend sidebar and display alongside the deck
        legend_items = ""
        for layer_key in active_node_types:
            col = _LAYER_COLOURS.get(layer_key, {"hex": "#fff"})["hex"]
            legend_items += (
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<div style='width:14px;height:14px;background:{col};"
                f"margin-right:6px;border-radius:2px;'></div>"
                f"<span>{layer_key.title()} nodes</span></div>"
            )
        shown_edge_types = (
            list(connections.keys()) if active_edge_types is None else active_edge_types
        )
        for et in shown_edge_types:
            col = _EDGE_COLOURS.get(et, _EDGE_DEFAULT)["hex"]
            legend_items += (
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<div style='width:14px;height:3px;background:{col};"
                f"margin-right:6px;'></div>"
                f"<span>{et.replace('_',' → ')} edges</span></div>"
            )

        legend_html = f"""
        <div style="padding:10px;background:rgba(0,0,0,0.75);border-radius:6px;
                    color:#fff;font-family:Arial,sans-serif;font-size:11px;
                    max-width:220px;">
          <div style="font-weight:bold;margin-bottom:8px;font-size:12px;">{title}</div>
          {legend_items}
        </div>"""

        try:
            from IPython.display import display
            display(deck, HTML(legend_html))
        except Exception:
            pass

        return deck

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose '2d' or '3d'.")
