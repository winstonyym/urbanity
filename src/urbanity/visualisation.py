import numpy as np
import pandas as pd
import pydeck as pdk

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
               node_id = ''):

    # Preprocess layers
    color_map_pastel = ['#fec5bb', '#fcd5ce', '#fae1dd', '#f8edeb', '#e8e8e4', '#d8e2dc', '#ece4db', '#ffe5d9', '#ffd7ba', '#fec89a']
    color_map_vibrant = ['#f94144', '#f3722c', '#f8961e', '#f9844a', '#f9c74f', '#90be6d', '#43aa8b', '#4d908e', '#577590', '#277da1']

    # Obtain colors
    plot_color = add_gradient_column(objects['plot'], 'plot_id', color_stops=color_map_vibrant, categorical=True)

    buildings_within = objects['building'].overlay(objects['boundary'], how='intersection')
    streets_within = objects['street'].overlay(objects['boundary'], how='intersection')
    intersection_within = objects['intersection'].overlay(objects['boundary'], how='intersection')

    # If chosen node specified
    if node_id != '':
        node_type, idx = node_id.split('_')[0], int(node_id.split('_')[1])
        chosen = objects[node_type].iloc[[idx]]
        chosen = objects[node_type].iloc[[idx]].reset_index(drop=True)
        centerx, centery = chosen.geometry[0].centroid.x, chosen.geometry[0].centroid.y
        chosen = pdk.Layer(
        'GeoJsonLayer',
        chosen,
        opacity=1,
        get_fill_color='[128, 255, 219]',  # RGBA color
        get_line_color='[0,0,0]',  # Border color
        line_width_min_pixels=2,
        )

        # Find connections
        relevant_connections = [k for k in connections.keys() if node_type in k]
        connected_nodes = {'plot':[], 'intersection':[], 'street':[], 'building':[]}
        for edge_type in relevant_connections:
            connected_nodes = get_connected_nodes(connected_nodes, connections, edge_type, node_type, idx)
        
        neighbors = []
        for node_type, neighbor in connected_nodes.items():
            
            if neighbor:
                globals()[f'{node_type}_chosen'] = objects[node_type].iloc[neighbor]

                if node_type == 'building':
                    globals()[f'{node_type}_chosen'] = pdk.Layer(
                        'GeoJsonLayer',
                        globals()[f'{node_type}_chosen'],
                        opacity=1,
                        extruded=True,
                        get_elevation="bid_orientation",
                        get_fill_color='[241, 241, 241]',  # RGBA color
                        get_line_color='[0,0,0]',  # Border color
                        line_width_min_pixels=2,
                    )
                else:
                    globals()[f'{node_type}_chosen'] = pdk.Layer(
                        'GeoJsonLayer',
                        globals()[f'{node_type}_chosen'],
                        opacity=1,
                        get_fill_color='[241, 241, 241]',  # RGBA color
                        get_line_color='[0,0,0]',  # Border color
                        line_width_min_pixels=2,
                    )

                neighbors.append(globals()[f'{node_type}_chosen'])
        
    else:
        chosen = None
        centerx, centery = objects['boundary'].geometry[0].centroid.x, objects['boundary'].geometry[0].centroid.y

    plot = pdk.Layer(
        'GeoJsonLayer',
        plot_color,
        opacity=1,
        get_fill_color='[46, 64, 82, 255]',  # RGBA color
        get_line_color='[46, 64, 82, 255]',  # Border color
        line_width_min_pixels=2,
    )

    # ,properties.feature_importance * 126
    streets = pdk.Layer(
        'GeoJsonLayer',
        streets_within,
        get_line_color='[255, 255, 255, 255]', # Line color (RGB)
        line_width_min_pixels=3,
    )

    intersections = pdk.Layer(
        "ScatterplotLayer",
        intersection_within,
        get_position='[x, y]',
        get_fill_color="[255, 255, 255, 255]",  # RGBA color for points
        get_radius=4,
    )

    buildings = pdk.Layer(
        'GeoJsonLayer',
        buildings_within,
        extruded=True,
        get_elevation="bid_height",
        get_fill_color="[245, 244, 244, 255]",  # RGBA color
        get_line_color='[225, 229, 242, 255]'
    )

    # Define the view
    view_state = pdk.ViewState(
        latitude=centery,  # Center latitude
        longitude=centerx,  # Center longitude
        zoom=14,
        pitch=45,
        bearing=135,
        height=1000, width=500
    )

    # Create the deck
    if node_id != '':
        deck = pdk.Deck(layers=[plot, buildings, intersections, streets, chosen, *neighbors], initial_view_state=view_state, map_provider='carto', map_style='dark_all')
    else:
        deck = pdk.Deck(layers=[plot, buildings, intersections, streets, chosen], initial_view_state=view_state, map_provider='carto', map_style='dark_all')
 
    return deck

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
        unique_cats = df_copy['index'].unique()
        cat_to_color = {}
        
        for i, cat in enumerate(unique_cats):
            # Cycle through color_stops by taking i % len(color_stops)
            cat_to_color[cat] = hex_to_rgb(color_stops[i % len(color_stops)])
        
        df_copy[new_col_name] = df_copy['index'].map(cat_to_color)
    
    else:
        # For continuous data, use the numeric color interpolation
        min_val = df_copy['index'].min()
        max_val = df_copy['index'].max()
        
        df_copy[new_col_name] = df_copy['index'].apply(
            lambda val: pick_color(val, min_val, max_val, color_stops)
        )
    
    return df_copy
