import pydeck as pdk

def plot_graph(bbox, objects):

    centerx, centery = bbox.geometry[0].centroid.x, bbox.geometry[0].centroid.y

    # Define a PyDeck layer for polygons
    boundary = pdk.Layer(
        'GeoJsonLayer',
        bbox,
        opacity=0.05,
        get_fill_color='[255, 255, 255]',  # RGBA color
        line_width_min_pixels=3,
        wireframe=True
    )

    plot = pdk.Layer(
        'GeoJsonLayer',
        objects['plot'],
        get_fill_color='[213, 224, 223, 20]',  # RGBA color
        get_line_color='[213, 224, 223]',  # Border color
        line_width_min_pixels=2,
        wireframe=True
    )

    # ,properties.feature_importance * 126
    streets = pdk.Layer(
        'GeoJsonLayer',
        objects['street'],
        get_line_color='[247, 191, 111, 50]', # Line color (RGB)
        line_width_min_pixels=3,
    )

    intersections = pdk.Layer(
        "ScatterplotLayer",
        objects['intersection'],
        get_position='[x, y]',
        get_fill_color="[255, 255, 255]",  # RGBA color for points
        get_radius=5,
    )

    buildings = pdk.Layer(
        'GeoJsonLayer',
        objects['building'],
        get_fill_color="[224, 220, 213, 100]",  # RGBA color
        get_line_color='[255, 255, 255, 100]'
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
    deck = pdk.Deck(layers=[boundary, buildings, intersections, streets, plot], initial_view_state=view_state, map_provider='carto', map_style='dark_no_labels')
    return deck
