# main module
import os
import json
import time
import math
from unittest import skip
import pkg_resources
from typing import Optional, Union
from webbrowser import get

from .utils import get_country_centroids, finetune_poi
from .geom import *
from .population import get_population_data

import networkx as nx
from networkx import MultiDiGraph
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import ipyleaflet
from ipyleaflet import basemaps, basemap_to_tiles, Icon, Marker, LayersControl, LayerGroup, DrawControl, FullScreenControl, ScaleControl, LocalTileLayer, GeoData
import pyrosm
from pyrosm import get_data
import rasterstats



# Import country coords
country_dict = get_country_centroids()
class Map(ipyleaflet.Map):

    def __init__(self, country: Optional[str] = None, **kwargs):
        self.bbox = None
        self.polygon_bounds = None

        super().__init__(**kwargs)
    
        if country:
            try:
                self.center = country_dict[country]['coords']
            except KeyError as err:
                print(f"KeyError: {err}. Please manually input center coordinates.")
            finally:
                self.country = country
        
        if 'zoom' not in kwargs:
            self.zoom = 11
        
        # Set default attributes
        if 'layout' not in kwargs:
            self.layout.height = "400px"
            self.layout.width = "600px"

        self.attribution_control = False

        # Add controls
        self.add_control(FullScreenControl())
        self.add_control(LayersControl(position='topright'))

        def handle_draw(target, action: str, geo_json: dict):
            print(action)
            print(geo_json)

        dc = (DrawControl(rectangle={'shapeOptions':{'color':"#a52a2a"}},
                                polyline = {"shapeOptions": {"color": "#6bc2e5", "weight": 2, "opacity": 1.0}},
                                polygon = {"shapeOptions": {"fillColor": "#eba134", "color": "#000000", "fillOpacity": 0.5, "weight":2},
                                            "drawError": {"color": "#dd253b", "message": "Delete and redraw"},
                                            "allowIntersection": False},
                                )
                    )
        dc.on_draw(handle_draw)
        self.add(dc)
        self.add_control(ScaleControl(position='bottomleft'))
        

        self.add_layer(basemap_to_tiles(basemaps.CartoDB.VoyagerNoLabels))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", name = 'Google Streets'))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", name = 'Google Hybrid'))
        self.add_layer(LocalTileLayer(path="http://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", name = 'Google Terrain'))

    def add_bbox(self, show: bool = False, remove: bool = True):

        t_index = None
        
        # Find DrawControl layer index
        for i, control in enumerate(self.controls):
            if isinstance(control, DrawControl):
                t_index = i

        if self.controls[t_index].last_action == '':
            print("No bounding box/polygon found. Please draw on map.")
            if show == True:
                display(self)

        elif self.controls[t_index].last_action == 'created': 

            lon_list = []
            lat_list = []
            for lon,lat in [*self.controls[t_index].data[0]['geometry']['coordinates'][0]]:
                lon_list.append(lon)
                lat_list.append(lat)
        
            polygon_geom = Polygon(zip(lon_list, lat_list))
            gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon_geom]) 

            # Assign bounding box as self object attribute
            self.polygon_bounds = gdf
            self.polygon_bounds.crs = 'epsg:4326'

            # Remove drawing on self and close display
            if remove == True: 
                print('Assigned bbox attribute to self object. Removing drawn boundary.')
                self.controls[t_index].clear()
            else: 
                print('Assigned bbox attribute to self object.')
    
    def add_polygon_boundary(
        self, 
        filepath: str,
        layer_name: Optional[str] = 'Site', 
        polygon_style: Optional[dict] = {'style': {'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6},
                                         'hover_style': {'fillColor': 'red' , 'fillOpacity': 0.2}},                   
        show: bool = False, 
        remove: bool = True) -> None:

        if filepath:
            # GeoJSON string file
            gdf = gpd.read_file(filepath)
        
        # Assign polygon boundary attribute to polygon object
        self.polygon_bounds = gdf

        # Add polygon boundary as map layer
        geo_data = GeoData(geo_dataframe = gdf,
                   style=polygon_style['style'],
                   hover_style=polygon_style['hover_style'],
                   name = layer_name)
        self.add_layer(geo_data)
        if show:
            display(self)
        
    def remove_polygon_boundary(self) -> None:
        polygon_exists = False
        for i in self.layers:
            if isinstance(i, ipyleaflet.leaflet.GeoData):
                polygon_exists = True
        if polygon_exists:
            self.remove_layer(self.layers[len(self.layers)-1])
            print('Polygon bounding layer removed.')
        else:
            print('No polygon layer found on map.')

    def get_street_network(
            self, 
            directory: str,
            location: str,
            bandwidth: int = 200,
            network_type: str = 'driving',
            building_attr: bool = True,
            pop_attr: bool = True,
            poi_attr: bool = True,
            svi_attr: bool = False,
            use_tif: bool = False,
            layer_name: str = 'Street Network', 
            polyline_style: dict = {'style': {'color': 'black', 'opacity':0.5, 'weight':1.9},
                                'hover_style': {'color': 'yellow' , 'opacity': 0.2}},  
            show: bool = False, 
            dual: bool = False) -> MultiDiGraph:
        """Function to generate either primal planar or dual (edge) networks. Bandwidth allows to account for
        outside the network that share immediate neighbours with nodes within target area. *_attr arguments
        can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            directory (str): Location to store and download Geofabrik file. Download will be skipped if location file is found.
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            bandwidth (int): Distance to extract information beyond network. Defaults to 200.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            use_tif (bool): Specifies whether csv or tif should be used to construct population attributes. Defaults to False.
            layer_name (str): Layer name to display in ipyleafet. Defaults to 'Street Network'.
            polyline_style (dict): Style dictionary for plotting vector layers in ipyleaflet. Defaults to {'style': {'color': 'black', 'opacity':0.5, 'weight':1.9}, 'hover_style': {'color': 'yellow' , 'opacity': 0.2}}.
            show (bool): Specifies whether a new map view should be created at line location. Defaults to False.
            dual (bool): If true, creates a dual (edge) network graph. Defaults to False.
            
        Raises:
            Exception: No bounding box or polygon found to construct network.

        Returns:
            MultiDiGraph: A networkx/osmnx primal planar or dual (edge) network with specified attribute information.
        """            

        start = time.time()
        try:
            fp = get_data(location, directory = directory)
            print('Creating data folder and downloading osm street data...')
        except KeyError:
            fp = get_data(self.country, directory = directory)
            print(f"KeyError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
        except KeyError:
            raise ValueError('No osm data found for specified location.')

        print('Data extracted successfully. Proceeding to construct street network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)
        nodes, edges = osm.get_network(network_type=network_type, nodes=True)

        if building_attr:
            # Get building spatial data and project 
            building = osm.get_buildings()
            building_proj = project_gdf(building)

            # Make geometry type homogeneous (polygons) to to allow overlay operation
            building_polygon = fill_and_expand(building_proj)

        # Build networkx graph for pre-processing
        G_buff = osm.to_graph(nodes, edges, graph_type="networkx", force_bidirectional=True, retain_all=True)
        
        # Add great circle length to network edges
        G_buff = add_edge_lengths(G_buff)

        # Simplify graph by removing nodes between endpoints and joining linestrings
        G_buff_simple = simplify_graph(G_buff)

        # Identify nodes inside and outside (buffered polygon) of original polygon
        gs_nodes = graph_to_gdf(G_buff_simple, nodes=True)[["geometry"]]
        to_keep = gs_nodes.within(original_bbox)
        to_keep = gs_nodes[to_keep]
        nodes_outside = gs_nodes[~gs_nodes.index.isin(to_keep.index)]
        set_outside = nodes_outside.index

        # Truncate network by edge if all neighbours fall outside original polygon
        nodes_to_remove = set()
        for node in set_outside:
            neighbors = set(G_buff_simple.successors(node)) | set(G_buff_simple.predecessors(node))
            if neighbors.issubset(nodes_outside):
                nodes_to_remove.add(node)
        
        G_buff_trunc = G_buff_simple.copy()
        initial = G_buff_trunc.number_of_nodes()
        G_buff_trunc.remove_nodes_from(nodes_to_remove)

        # Remove unconnected subgraphs
        max_wcc = max(nx.weakly_connected_components(G_buff_trunc), key=len)
        G_buff_trunc = nx.subgraph(G_buff_trunc, max_wcc)
        nodes, edges = graph_to_gdf(G_buff_trunc, nodes=True, edges=True)
        nodes = nodes.fillna('')

        # If not dual representation graph
        if dual == False:
            
            nodes_buffer = project_gdf(nodes)
            nodes_buffer['center_bound'] = nodes_buffer.geometry.buffer(bandwidth)
            nodes_buffer = nodes_buffer.set_geometry('center_bound')

            if building_attr:

                # Compute and add building attributes
                res_intersection = nodes_buffer.overlay(building_polygon, how='intersection')
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['osmid'])['area'].sum()
                
                # Obtain proportion 
                total_area = math.pi*bandwidth**2
                area_series = area_series / total_area
                area_series.name = 'Footprint area'

                
                nodes = nodes.join(area_series)

                # Add perimeter
                res_intersection['perimeter'] = res_intersection.geometry.length
                perimeter_series = res_intersection.groupby(['osmid'])['perimeter'].sum()

                perimeter_series.name = 'Perimeter'
                nodes = nodes.join(perimeter_series)


                 # Add counts
                counts_series = res_intersection['osmid'].value_counts()
                counts_series.name = 'Counts'
                nodes = nodes.join(counts_series)
                
                nodes['Footprint area'] = nodes['Footprint area'].replace(np.nan, 0).astype(float)
                nodes['Perimeter'] = nodes['Perimeter'].replace(np.nan, 0).astype(float)
                nodes['Counts'] = nodes['Counts'].replace(np.nan, 0).astype(int)
            
            if pop_attr:

                groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

                if not use_tif:
                    pop_list, target_cols = get_population_data(self.country, use_tif=use_tif)
                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = project_gdf(data[0])
                        res_intersection = proj_data.overlay(nodes_buffer, how='intersection')

                        # Add population sum
                        pop_sum_series = res_intersection.groupby(['osmid'])[data[1]].sum()
                        pop_sum_series.name = groups[i]
                        nodes = nodes.join(pop_sum_series)

                    for name in groups:
                        nodes[name] = nodes[name].replace(np.nan, 0).astype(float)
                    
                if use_tif: 
                    # Using .tif for computation
                    src_list, rc_list = get_population_data(self.country, use_tif=use_tif)
                    nodes_buffer = nodes_buffer.to_crs('epsg:4326')
                    
                    for group, src, rc in zip(groups, src_list, rc_list):
                        stats = ['sum']

                        result = rasterstats.zonal_stats(
                            nodes_buffer, 
                            rc[0], 
                            nodata = 0, 
                            affine = src.transform, 
                            stats = stats
                        )

                        result = pd.DataFrame(result)

                        # Add population sum
                        nodes[group] = list(result['sum'])
                        nodes[group] = nodes[group].replace(np.nan, 0).astype(int)
                
            if poi_attr:
                poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                with open(poi_path) as poi_filter:
                    poi_filter = json.load(poi_filter)
                
                pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                pois = pois.replace(np.nan, '')

                def poi_col(amenity, shop, tourism, leisure):
                    value = amenity
                    if amenity == '' and shop not in poi_filter['food_set']:
                        value = 'commercial'
                    elif amenity == '' and shop in poi_filter['food_set']:
                        value = shop
                    elif amenity == '' and tourism != '':
                        value = 'entertainment'
                    elif amenity == '' and leisure != '':
                        value = 'recreational'
                    return value
                
                pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
                pois = pois[['id', 'osm_type','lon','lat','name','poi_col','geometry']]

                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=30)
                pois = project_gdf(pois)
                pois['geometry'] = pois.geometry.centroid

                res_intersection = pois.overlay(nodes_buffer, how='intersection')
                poi_series = res_intersection.groupby(['osmid'])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='osmid', columns='poi_col', values=0)
                nodes = nodes.merge(pois_df, how='left', left_index=True, right_index=True).replace(np.nan,0) 
                nodes = nodes.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional'})

            if svi_attr:
                pass
                # svi_path = pkg_resources.resource_filename('urbanity', f"svi_data/{location}.geojson")
                # svi_data = gpd.read_file(svi_path)

                # # returns gdf of image id, tile_id, indicators, and point coords
                # res_intersection = svi_data.overlay(nodes_buffer, how='intersection')
                
                # indicators = ['Green View', 'Sky View', 'Building View', 'Road View']
                # for indicator in indicators:
                    
                #     svi_series = res_intersection.groupby(['osmid'])[indicator].mean()
                #     svi_series.name = indicator
                #     nodes = nodes.join(svi_series)

                #     index_mean = nodes[indicator].mean()
                #     nodes[indicator] = nodes[indicator].replace(np.nan, index_mean)
            
            print("--- %s seconds ---" % round(time.time() - start,3))
            return G_buff_trunc, nodes, edges

        elif dual: 
            # First extract dictionary of osmids and lengths for original nodes associated with each edge
            osmid_view = nx.get_edge_attributes(G_buff_trunc, "osmid")
            osmid_dict = {}
            for u,v in set(osmid_view.items()):
                if u not in osmid_dict:
                    osmid_dict[(u[:2])] = v
                else: 
                    osmid_dict[(u[:2])].append(v)

            length_view = nx.get_edge_attributes(G_buff_trunc, "length")
            length_dict = {}
            for u,v in set(length_view.items()):
                if u not in length_dict:
                    length_dict[(u[:2])] = v
                else: 
                    length_dict[(u[:2])].append(v)

            x_dict = nx.get_node_attributes(G_buff_trunc, "x")
            y_dict = nx.get_node_attributes(G_buff_trunc, "y")

            # Create new placeholder graph and add edges as nodes and adjacency links between edges as new edges
            L = nx.empty_graph(0)
            LG = nx.line_graph(G_buff_trunc)
            L.graph['crs'] = 'EPSG:4326'
            for node in set(G_buff_trunc.edges()):
                L.add_node(node, length = length_dict[node], osmids = osmid_dict[node], x = (x_dict[node[0]]+x_dict[node[1]])/2, y = (y_dict[node[0]]+y_dict[node[1]])/2, geometry=Point((x_dict[node[0]]+x_dict[node[1]])/2, (y_dict[node[0]]+y_dict[node[1]])/2))
            for u,v in set(LG.edges()):
                L.add_edge(u[:2],v[:2])

            # Extract nodes and edges GeoDataFrames from graph
            L_nodes, L_edges = graph_to_gdf(L, nodes=True, edges=True, dual=True)
            L_nodes = L_nodes.fillna('')

            if building_attr:
                building_nodes = project_gdf(L_nodes)
                building_nodes['center_bound'] = building_nodes.geometry.buffer(bandwidth)
                building_nodes = building_nodes.set_geometry('center_bound')
                building_nodes = building_nodes.reset_index()
                building_nodes = building_nodes.rename(columns = {'level_0':'from', 'level_1':'to'})
                building_nodes['unique'] = building_nodes.index

                # Compute and add area
                res_intersection = building_nodes.overlay(building_polygon, how='intersection')
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['unique'])['area'].sum()
                
                # Obtain proportion 
                total_area = math.pi*bandwidth**2
                area_series = area_series / total_area
                area_series = area_series.astype(float)
                area_series.name = 'Footprint area'

                L_nodes['unique'] = list(range(len(L_nodes)))
                L_nodes = L_nodes.join(area_series, on = 'unique')

                # Add perimeter
                res_intersection['perimeter'] = res_intersection.geometry.length
                perimeter_series = res_intersection.groupby(['unique'])['perimeter'].sum()

                perimeter_series.name = 'Perimeter'
                L_nodes = L_nodes.join(perimeter_series, on = 'unique')


                 # Add counts
                counts_series = res_intersection['unique'].value_counts()
                counts_series.name = 'Counts'
                L_nodes = L_nodes.join(counts_series, on = 'unique')
                
                L_nodes['Footprint area'] = L_nodes['Footprint area'].replace(np.nan, 0).astype(float)
                L_nodes['Perimeter'] = L_nodes['Perimeter'].replace(np.nan, 0).astype(float)
                L_nodes['Counts'] = L_nodes['Counts'].replace(np.nan, 0).astype(int)
    

            print("--- %s seconds ---" % round(time.time() - start,3))

            return L, L_nodes, L_edges

    def remove_street_network(self) -> None:
        network_exists = False
        for i in self.layers:
            if isinstance(i, ipyleaflet.leaflet.GeoData):
                network_exists = True
        if network_exists:
            self.remove_layer(self.layers[len(self.layers)-1])
            print('Network layer removed.')
        else:
            print('No network layer found on map.')

    def get_building_network(
            self, 
            directory: str,
            layer_name: Optional[str] = 'Building Network', 
            polyline_style: Optional[dict] = {'style': {'color': 'black', 'opacity':0.5, 'weight':1.9},
                                'hover_style': {'color': 'yellow' , 'opacity': 0.2}},  
            show: Optional[bool] = False,
            network_type: str = 'driving') -> MultiDiGraph:
        start = time.time()
        print('Creating data folder and downloading osm building data...')
        fp = get_data(self.country, directory = directory)
        print('Data extracted successfully. Proceeding to construct building network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)

        # Get buildings
        buildings = osm.get_buildings()
        proj_buildings = project_gdf(buildings)

        # Get adjacency based on spatial intersection
        building_neighbours = {}
        for i,b in zip(proj_buildings.id,proj_buildings.geometry):
            s = proj_buildings.intersects(b.buffer(100))
            building_neighbours[i] = proj_buildings.id[s[s].index].values

        # Set centroid as geometry and create lon and lat columns
        proj_buildings['center'] = proj_buildings.geometry.centroid
        proj_buildings = proj_buildings.set_geometry("center")
        buildings = proj_buildings.to_crs(4326)
        buildings['x'] = buildings.geometry.x
        buildings['y'] = buildings.geometry.y
        buildings = buildings.set_index('id')

        # Create building network graph
        # Get dict of building nodes and their attributes
        id_to_attributes = {}
        for node in set(buildings.index):
            id_to_attributes[node] = buildings.loc[node].to_dict()

        B = nx.empty_graph(0)
        B.graph['crs'] = 'EPSG:4326'

        # Add nodes
        for node in set(buildings.index):
            B.add_node(node)
        nx.set_node_attributes(B, id_to_attributes)

        # Add edges
        for node, neighbours in building_neighbours.items():
            for neighbour in neighbours:
                B.add_edge(node, neighbour)

        # Compute euclidean distance between adjacent building centroids
        id_to_x = dict(zip(buildings.index, buildings.x))
        id_to_y = dict(zip(buildings.index, buildings.y))
        
        distance_between_buildings = {}
        for pair in set(B.edges()):
            distance_between_buildings[pair] = great_circle_vec(id_to_x[pair[0]], id_to_y[pair[0]], id_to_x[pair[1]], id_to_y[pair[1]])
        
        nx.set_edge_attributes(B, distance_between_buildings, 'length')

        # Identify largest weakly connected component
        max_wcc = max(nx.connected_components(B), key=len)
        B_max = nx.subgraph(B, max_wcc)

        B_nodes, B_edges = graph_to_gdf(B_max, nodes=True, edges=True, dual=True)
        B_nodes = B_nodes.fillna('')
        # Return graph and add network layer to Map object

        # building_data = GeoData(geo_dataframe = B_edges,
        #         style=polyline_style['style'],
        #         hover_style=polyline_style['hover_style'],
        #         name = layer_name)

        # self.add_layer(building_data)

        # if show:
        #     display(self)

        print("--- %s seconds ---" % round(time.time() - start,3))

        return B_max, B_nodes, B_edges


    
