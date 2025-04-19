import numpy as np
import pandas as pd
import pydeck as pdk
from ipywidgets import HTML
import numbers   # <- add

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
