# main module
import os
import json
import time
from typing import Optional, Union
from webbrowser import get

from .utils import get_country_centroids
from .geom import *

import networkx as nx
from networkx import MultiDiGraph
import geopandas as gpd
from shapely.geometry import Polygon
import ipyleaflet
from ipyleaflet import basemaps, basemap_to_tiles, Icon, Marker, LayersControl, LayerGroup, DrawControl, FullScreenControl, ScaleControl, LocalTileLayer, GeoData
import pyrosm
from pyrosm import get_data

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
            gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) 

            # Assign bounding box as self object attribute
            self.polygon_bounds = gdf

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
            city: Optional[str] = None,
            layer_name: Optional[str] = 'Street Network', 
            polyline_style: Optional[dict] = {'style': {'color': 'black', 'opacity':0.5, 'weight':1.9},
                                'hover_style': {'color': 'yellow' , 'opacity': 0.2}},  
            show: Optional[bool] = False, 
            dual: Optional[bool] = False,
            network_type: str = 'driving') -> MultiDiGraph:
        start = time.time()
        print('Creating data folder and downloading osm street data...')
        try:
            fp = get_data(self.country, directory = './data')
        except ValueError:
            fp = get_data(city, directory='./data')

        print('Data extracted successfully. Proceeding to construct street network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=50)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)
        nodes, edges = osm.get_network(network_type=network_type, nodes=True)

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

        # If not dual representation graph
        if dual == False:
        #     street_data = GeoData(geo_dataframe = edges,
        #             style=polyline_style['style'],
        #             hover_style=polyline_style['hover_style'],
        #             name = layer_name)

        #     self.add_layer(street_data)

        #     if show:
        #         display(self)
        
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

            # street_edge_data = GeoData(geo_dataframe = L_edges,
            #         style=polyline_style['style'],
            #         hover_style=polyline_style['hover_style'],
            #         name = layer_name + ' (Edges)')

            # self.add_layer(street_edge_data)

            # if show:
            #     display(self)

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
            layer_name: Optional[str] = 'Building Network', 
            polyline_style: Optional[dict] = {'style': {'color': 'black', 'opacity':0.5, 'weight':1.9},
                                'hover_style': {'color': 'yellow' , 'opacity': 0.2}},  
            show: Optional[bool] = False,
            network_type: str = 'driving') -> MultiDiGraph:
        start = time.time()
        print('Creating data folder and downloading osm building data...')
        fp = get_data(self.country, directory = './data')
        print('Data extracted successfully. Proceeding to construct building network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=50)
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

class StreetNetwork(MultiDiGraph):
    def __init__(self, ):
        pass



    
