# Map class utility functions
import os
import json
import pkg_resources
import geopandas as gpd
from urbanity.geom import project_gdf
import numpy as np
from IPython.display import display
from ipyleaflet import DrawControl
from urllib.error import HTTPError
from collections import Counter

def get_country_centroids():
    """Utility function to obtain country centroids based on country name.

    Returns:
        dict: Dictionary object with keys as country names and values as centroid locations.
    """    
    data_path = pkg_resources.resource_filename('urbanity', "map_data/country.json")
    with open(data_path) as f:
        country_dict = json.load(f)

    return country_dict

def get_population_data_links(country, use_tif = False):
    """Obtain population data links based on specified country.

    Args:
        country (str): Name of country to obtain population data.
        use_tif (bool, optional): If True, obtains url for .geotiffs instead of csv. Defaults to False.

    Returns:
        dict: Dictionary with keys as data tags and values as links to population data.
    """    
    data_path = pkg_resources.resource_filename('urbanity', "map_data/links_general_tiled.json")
    with open(data_path) as f:
        general_pop_dict = json.load(f)
    return general_pop_dict

def get_available_pop_countries():
    """Prints list of countries where population data is available.
    """    
    general_pop_dict = set(get_population_data_links())
    print(sorted(general_pop_dict))

def get_available_countries():
    """Prints list of countries where centroid information is available. 
    """
    country_dict = set(get_country_centroids())
    print(sorted(country_dict))

def get_available_precomputed_network_data():
    """Prints list of cities available from the Global Urban Network Dataset
    """
    data_path = pkg_resources.resource_filename('urbanity', "map_data/network_data.json")
    with open(data_path) as f:
        city_dict = json.load(f)

    list_of_cities = []
    for entry in city_dict.keys():
        if entry.split('_')[0] not in list_of_cities:
            list_of_cities.append(entry.split('_')[0])
    
    print(f'The following cities are available: {sorted(list_of_cities)}.')

def finetune_poi(df, target, relabel_dict, n=5, pois_data = 'osm'):
    """Relabel and trim poi list to main categories ('Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social')

    Args:
        df (pd.DataFrame): POI dataframe with full list of amenities extracted from OSM/Overture
        target (str): Target column with poi labels
        relabel_dict (dict): Relabelling dictionary to match original poi labels to main categories. Users can provide custom relabelling according to use case by modifying (./src/urbanity/map_data/poi_filter.json)
        n (int, optional): Minimum count of pois to keep. Defaults to 5.
        pois_data (str, optional): Specifies whether osm or Overture poi data should be used. Defaults to 'osm'.

    Returns:
        pd.DataFrame: Dataframe with poi information relabelled according to main categories. 
    """  
    if pois_data == 'osm':
        df2 = df.copy()
        for k,v in relabel_dict.items():
            df2[target] = df2[target].replace(k, v)
        
        # remove categories with less than n instances
        
        cat_list = df2[target].value_counts().index
        cat_mask = (df2[target].value_counts() > n).values
        selected = set(cat_list[cat_mask])
        
        df2 = df2[df2[target].isin(selected)]

    elif pois_data == 'overture':
        df2 = df.copy()
        df2=df2.replace({target: relabel_dict})

    return df2


def get_gadm(country, city, version = '4.1', max_level = 4, level_drop = 0):
    """Function to automate extraction of GADM city boundaries and their subzones. Files are extracted in .geojson format.

    Args:
        city (str): City name to extract from GADM database.
        city_subzone (bool, optional): If True, searches one level down to obtain census subzone for city. Defaults to False.
    """    

    country = country.title()
    city = city.title()
    small_countries = ['Singapore']
    large_scale_countries = ['United States']

    data_path = pkg_resources.resource_filename('urbanity', "map_data/GADM_links.json")

    with open(data_path) as f:
        GADM_dict = json.load(f)
    
    country_code = GADM_dict[country]
    returned = False

    for i in reversed(range(max_level+1)):
        geojson_path = f'https://geodata.ucdavis.edu/gadm/gadm{version}/json/gadm{version.replace(".", "")}_{country_code}_{i}.json'
        try:
            country_df = gpd.read_file(geojson_path)
            print(f'Level {i} downloaded for {country}.')
            if country in small_countries:
                return country_df
            if country in large_scale_countries:
                return country_df
            
            result = []
            for level in range(1,i+1):
                zones = list(country_df[f'NAME_{level}'].unique())
                result = [zone for zone in zones if city in zone]
                if result == []:
                    continue
                elif result != []:
                    print(f'{result[0]} found in level {level}.')
                    if level_drop == 0:
                        print(f'Returning level {level} boundary file.')
                        return country_df[country_df[f'NAME_{level}'] == result[0]]
                    elif level_drop != 0:
                        try:
                            print(f'Retrieving level {level+level_drop} boundary file.')
                            zones = list(country_df[f'NAME_{level+level_drop}'].unique())
                            result = [zone for zone in zones if city in zone]
                            return country_df[country_df[f'NAME_{level+level_drop}'] == result[0]]
                        except KeyError:
                            print('GADM does not provide shapefiles at this level of detail.')
                            return None
                        except IndexError:
                            print(f'No subzone with corresponding name found at this level.')
                            return None
                    
        except HTTPError:
            continue
    
    
def get_building_to_building_edges(building_nodes, adj_column = ''):
    # building_edges = get_building_to_building_edges(building_nodes, adj_column = '3-nn-idx')
    # Prepare edge index. First match with index position then convert to torch tensor. 
    start_list = []
    end_list = []
    for i, neighbours in enumerate(building_nodes[adj_column]):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    building_edges = np.stack([start_list, end_list], axis=1).transpose()
    
    return building_edges

def get_intersection_to_street_edges(intersections, streets):
    # intersection_to_street_edges = get_intersection_to_street_edges(gdfs[1], gdfs[2])
    node_to_id = {}
    for i,node in enumerate(intersections['osmid']):
        node_to_id[node] = i

    start_node = [node_to_id[i] for i in streets['u'].values] + [node_to_id[i] for i in streets['v'].values]
    end_node = list(streets['edge_id'].values) + list(streets['edge_id'].values)

    start_index = np.array(start_node)
    end_index = np.array(end_node)
    intersection_to_street_edges = np.stack([start_index, end_index], axis=1).transpose()
    
    return intersection_to_street_edges

def get_buildings_in_plot_edges(urban_plots, adj_column = ''):
    # building_in_plot_edges = get_buildings_in_plot_edges(urban_plots, adj_column = 'building_ids')
    # Prepare edge index. First match with index position then convert to torch tensor. 
    start_list = []
    end_list = []
    for i, neighbours in enumerate(urban_plots[adj_column]):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    building_in_plot_edges = np.stack([start_list, end_list], axis=1).transpose()
    
    return building_in_plot_edges

def get_edges_along_plot(urban_plots, adj_column = ''):
    # edges_along_plot = get_edges_along_plot(urban_plots, adj_column = 'edge_ids')
    # Prepare edge index. First match with index position then convert to torch tensor. 
    start_list = []
    end_list = []
    for i, neighbours in enumerate(urban_plots[adj_column]):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    edges_along_plot = np.stack([start_list, end_list], axis=1).transpose()
    
    return edges_along_plot

def get_plot_to_plot_edges(urban_plots):
    """Helper function to generate network edges between urban plots and their adjacent plots.

    Args:
        urban_plots (gpd.GeoDataFrame): A geopandas dataframe consisting of Polygons where each row represents an urban plot. 

    Returns:
        np.array: A (2, N) array where the first row corresponds to urban plots and the second row corresponds to adjacent urban plots that are connected by the same street. N is the number of edges between all urban plots in the network. 
    """    
    urban_plots['plot_id'] = urban_plots.index
    urban_plots['nn_plot_ids'] = None

    for index, plot in urban_plots.iterrows():   

        # get 'not disjoint' countries
        neighbors = urban_plots[~urban_plots.geometry.disjoint(plot.geometry)].plot_id.tolist()

        # remove own name of the country from the list
        neighbors = [name for name in neighbors if plot.plot_id != name ]
        
        # add names of neighbors as NEIGHBORS value
        urban_plots.at[index, "nn_plot_ids"] = neighbors

    # Create adjacency matrix for connected plots.
    start_list = []
    end_list = []
    for i, neighbours in enumerate(urban_plots['nn_plot_ids']):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    plot_to_plot_edges = np.stack([start_list, end_list], axis=1).transpose()

    return plot_to_plot_edges


def get_building_to_street_edges(streets, building_nodes):
    """Helper function to generate network edges between buildings and their adjacent (nearest; subject to distance threshold of 50 metres) streets.

    Args:
        streets (gpd.GeoDataFrame): A geopandas dataframe consisting of LineStrings where each row represents a road segment.
        building_nodes (gpd.GeoDataFrame): A geopandas dataframe consisting of Polygons where each row corresponds to a building and its footprint.

    Returns:
        np.array: A (2, N) array where the first row corresponds to street IDs and the second row corresponds building_node IDs. N is the number of edges between all streets and buildings. 
    """    
    # street_to_building = get_building_to_street_edges(gdfs[2], building_nodes)
    building_nodes_copy = building_nodes.copy()
    building_nodes_copy['centroid'] = building_nodes.geometry.centroid
    building_nodes_copy = building_nodes_copy.set_geometry('centroid')
    building_nodes_copy['b_index'] = building_nodes_copy.index

    proj_edge = project_gdf(streets)

    # Find nearest building to street
    edge_intersection = gpd.sjoin_nearest(building_nodes_copy, proj_edge, how='inner', max_distance=50, distance_col = 'building_edges')
    edge_to_building = edge_intersection.groupby(['edge_id'])[['b_index']].aggregate(lambda x: list(x))

    start_list = []
    end_list = []
    for idx, nn in zip(edge_to_building['b_index'].index, edge_to_building['b_index']):
        for k in nn:
            start_list.append(idx)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    street_to_building = np.stack([start_list, end_list], axis=1).transpose()

    return street_to_building

def get_edge_nodes(edges) -> [gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Converts street segments into nodes as part of a multi-nodal graph representation.

    Args:
        edges (gpd.GeoDataFrame): A geopandas GeoDataFrame containing the geometry and attribute features of street segments.

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame consisting of half-edges that retain the original linestring geometry of street segments (for plotting). 
        gpd.GeoDataFrame: A geopandas GeoDataFrame consisting of nodes that were derived from edges. 
    """
    # Project edges to local crs for distance computation
    proj_edges = project_gdf(edges)
    
    # Set edge_id as string and create placeholder lists
    proj_edges['edge_id'] = proj_edges['edge_id'].astype(str)
    x_list = []
    y_list = []
    edge_id_list = []
    u_list = []
    v_list = []
    length_list = []
    geometry_list = []
    
    # Iterate through each edge 
    for i, row in proj_edges.iterrows():
        
        # Get centre from linestring coordinate sequence
        coords_list = list(row['geometry'].coords)
        len_coord_list = len(row['geometry'].coords)
        mid_idx = len_coord_list // 2
        
        # If even number of coords
        if len_coord_list == 2:
            center = ((coords_list[0][0]+ coords_list[1][0])/2, (coords_list[0][1]+ coords_list[1][1])/2) 
            line_segments = [LineString([coords_list[0], center]), LineString([center, coords_list[1]])]
        elif (len_coord_list > 2) & (len_coord_list % 2 == 0):
            center = ((coords_list[mid_idx-1][0]+ coords_list[mid_idx][0])/2, (coords_list[mid_idx-1][1]+ coords_list[mid_idx][1])/2)
            line_segments = [LineString([coords for coords in coords_list[:mid_idx]] + [center]), LineString([center] + [coords for coords in coords_list[mid_idx:]])]
        else: 
            center = coords_list[mid_idx]
            line_segments = [LineString([coords for coords in coords_list[:mid_idx+1]]), LineString([coords for coords in coords_list[mid_idx:]])]
            
        # Add start to midpoint of linestring
        edge_id_list.append(row['edge_id']+'_0')
        u_list.append(row['u'])
        v_list.append(row['edge_id']+'_m')
        length_list.append(line_segments[0].length)
        geometry_list.append(line_segments[0])
        
        # Add midpoint to end of linestring
        edge_id_list.append(row['edge_id']+'_1')
        u_list.append(row['edge_id']+'_m')
        v_list.append(row['v'])
        length_list.append(line_segments[1].length)
        geometry_list.append(line_segments[1])
        x_list.append(center[0])
        y_list.append(center[1])

    # Select attribute columns
    col = list(edges.columns[5:])
    cols = ['edge_id', 'length'] + col
    # Get geodataframes corresponding to edges and edge nodes
    split_edges = gpd.GeoDataFrame({'edge_id': edge_id_list, 'u': u_list, 'v': v_list, 'length': length_list}, crs=proj_edges.crs, geometry = geometry_list)
    edge_nodes = gpd.GeoDataFrame(data=edges[cols], crs = proj_edges.crs, geometry = gpd.points_from_xy(x_list, y_list))
    
    # Reproject to global coordinates
    split_edges = split_edges.to_crs(4326)
    edge_nodes = edge_nodes.to_crs(4326)

    return split_edges, edge_nodes


def most_frequent(List):
    """Helper function which returns the most common element in a list.

    Args:
        List (list): A list of elements with categorical labels.   

    Returns:
        int: The most common integer element. 
    """    
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]