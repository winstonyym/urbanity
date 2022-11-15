# import base packages
from ipaddress import collapse_addresses
import os
import json
import time
import math
import warnings
from unittest import skip
import pkg_resources
from typing import Optional, Union
from webbrowser import get

# import module functions and classes
from .utils import get_country_centroids, finetune_poi
from .geom import *
from .population import get_population_data, get_tiled_population_data
from .topology import compute_centrality, merge_nx_property, merge_nx_attr

# import external functions and classes
import numpy as np
import networkx as nx
from networkx import MultiDiGraph
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.errors import ShapelyDeprecationWarning
import ipyleaflet
from ipyleaflet import basemaps, basemap_to_tiles, Icon, Marker, LayersControl, LayerGroup, DrawControl, FullScreenControl, ScaleControl, LocalTileLayer, GeoData
import pyrosm
from pyrosm import get_data
import rasterstats
import networkit

# Catch known warnings from shapely and geopandas
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# Import country coords
country_dict = get_country_centroids()
class Map(ipyleaflet.Map):

    def __init__(self, country: str = None, **kwargs):
        """Instantiates a map object that inherits from ipyleaflet.Map. 

        Args:
            country (str, optional): Name of country to position map view. Defaults to None.
        """        
        self.bbox = None
        self.polygon_bounds = None

        if os.path.isdir('./data'):
            self.directory = "./data"
        else:
            os.makedirs('./data')
            self.directory = "./data"
        

        super().__init__(**kwargs)
    
        if country:
            try:
                self.center = country_dict[country]['coords']
            except KeyError as err:
                print(f"KeyError: {err}. Please manually input center coordinates by passing longitude and latitude information to the `center` argument.")
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
        """Specifies drawn bounding box as geographic extent.

        Args:
            show (bool, optional): If True, creates another map view. Defaults to False.
            remove (bool, optional): If True, removes drawn bounding box from map after adding it as an attribute. Defaults to True.
        """        

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
                print('Assigned bbox to map object. Removing drawn boundary.')
                self.controls[t_index].clear()
            else: 
                print('Assigned bbox map object.')
    
    def add_polygon_boundary(
        self, 
        filepath: str,
        layer_name: str = 'Site', 
        polygon_style: dict = {'style': {'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6},
                                         'hover_style': {'fillColor': 'red' , 'fillOpacity': 0.2}},                   
        show: bool = False) -> None:
        """Adds geographical boundary from specified filepath. Accepts .geojson and .shapefile objects.

        Args:
            filepath (str): Filepath to vector file.
            layer_name (str, optional): Layer name to display on map object. Defaults to 'Site'.
            polygon_style (dict, optional): Default visualisation parameters to display geographical layer. Defaults to {'style': {'color': 'black', 'fillColor': '#3366cc', 'opacity':0.05, 'weight':1.9, 'dashArray':'2', 'fillOpacity':0.6}, 'hover_style': {'fillColor': 'red' , 'fillOpacity': 0.2}}.
        """        

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
        
    def remove_polygon_boundary(self) -> None:
        """Removes polygon boundary from map object.
        """        
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
            location: str,
            bandwidth: int = 200,
            network_type: str = 'driving',
            graph_attr: bool = True,
            building_attr: bool = True,
            pop_attr: bool = True,
            poi_attr: bool = True,
            svi_attr: bool = False,
            use_tif: bool = False,
            dual: bool = False) -> MultiDiGraph:
        """Function to generate either primal planar or dual (edge) networks. If multiple geometries are provided, 
        network is constructed for only the first entry. Please merge geometries before use.
        Bandwidth (m) can be specified to buffer network, obtaining neighbouring nodes within buffered area of network.
        *_attr arguments can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            bandwidth (int): Distance to extract information beyond network. Defaults to 200.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            graph_attr (bool): Specifies whether graph metric and topological attributes should be included. Defaults to True.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            use_tif (bool): Specifies whether csv or tif should be used to construct population attributes. Defaults to False.
            dual (bool): If true, creates a dual (edge) network graph. Defaults to False.
            
        Raises:
            Exception: No bounding box or polygon found to construct network.

        Returns:
            MultiDiGraph: A networkx/osmnx primal planar or dual (edge) network with specified attribute information.
        """            

        start = time.time()
        try:
            fp = get_data(location, directory = self.directory)
            print('Creating data folder and downloading osm street data...')
        except KeyError:
            fp = get_data(self.country, directory = self.directory)
            print(f"KeyError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
        except KeyError:
            raise ValueError('No osm data found for specified location.')

        print('Data extracted successfully. Proceeding to construct street network.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            original_bbox = self.polygon_bounds.geometry[0]
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

        # Remove self loops
        G_buff_trunc_loop = G_buff_trunc.copy()
        G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

        nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)
        nodes = nodes.fillna('')

        print(f'Network constructed. Time taken: {round(time.time() - start)}.')

        # If not dual representation graph
        if dual == False:
            
            nodes_buffer = project_gdf(nodes)
            nodes_buffer['geometry'] = nodes_buffer.geometry.buffer(bandwidth)
            nodes_buffer = nodes_buffer.set_geometry('geometry')
            nodes_buffer = nodes_buffer.drop(columns = ['osmid'])
            nodes_buffer = nodes_buffer.reset_index()
            
            if graph_attr:

                # Compute graph attributes

                # Add Node Density
                proj_nodes = project_gdf(nodes)
                res_intersection = proj_nodes.overlay(nodes_buffer, how='intersection')
                res_intersection['Node Density'] = 1
                nodes["Node Density"] = res_intersection.groupby(['osmid_2'])['Node Density'].sum()

                # Add Street Length
                proj_edges = project_gdf(edges)
                proj_edges = proj_edges.drop(columns=['osmid'])
                res_intersection = proj_edges.overlay(nodes_buffer, how='intersection')
                res_intersection['street_len'] = res_intersection.geometry.length
                nodes["Street Length"] = res_intersection.groupby(['osmid'])['street_len'].sum()


                # Add Degree Centrality, Clustering (Weighted and Unweighted)
                nodes = merge_nx_property(nodes, G_buff_trunc_loop.out_degree, 'Degree')
                nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'Clustering')
                nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'Clustering (Weighted)', weight='length')


                #  Add Centrality Measures
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Closeness, 'Closeness Centrality', False, False)
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Betweenness, 'Betweenness Centrality', True)
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.EigenvectorCentrality, 'Eigenvector Centrality')
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.KatzCentrality, 'Katz Centrality')
                nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.PageRank, 'PageRank', 0.85, 1e-8, networkit.centrality.SinkHandling.NoSinkHandling, True)
            
            print(f'Topologic/metric attributes computed. Time taken: {round(time.time() - start)}.')
            
            # If building_attr is True, compute and add building attributes.
            if building_attr:

                # Compute and add building attributes
                res_intersection = nodes_buffer.overlay(building_polygon, how='intersection')
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['osmid'])['area'].sum()
                
                # Obtain proportion 
                total_area = math.pi*bandwidth**2
                nodes['Footprint Proportion'] = area_series / total_area
            
                # Obtain mean area
                mean_series = res_intersection.groupby(['osmid'])['area'].mean()
                nodes['Footprint Mean'] = mean_series

                # Obtain mean area
                std_series = res_intersection.groupby(['osmid'])['area'].std()
                nodes['Footprint Stdev'] = std_series

                # Add perimeter
                res_intersection['perimeter'] = res_intersection.geometry.length
                perimeter_series = res_intersection.groupby(['osmid'])['perimeter'].sum()
                nodes['Perimeter Total'] = perimeter_series

                perimeter_mean_series = res_intersection.groupby(['osmid'])['perimeter'].mean()
                nodes['Perimeter Mean'] = perimeter_mean_series

                perimeter_std_series = res_intersection.groupby(['osmid'])['perimeter'].std()
                nodes['Perimeter Stdev'] = perimeter_std_series

                # Add complexity Mean and Std.dev
                res_intersection['complexity'] = res_intersection['perimeter'] / np.sqrt(np.sqrt(res_intersection['area']))   
                compl_mean_series = res_intersection.groupby(['osmid'])['complexity'].mean()
                nodes['Complexity Mean'] = compl_mean_series
                
                compl_std_series = res_intersection.groupby(['osmid'])['complexity'].std()
                nodes['Complexity Stdev'] = compl_std_series

                 # Add counts
                counts_series = res_intersection['osmid'].value_counts()
                nodes['Counts'] = counts_series
                
                # Add building attributes to node dataframe
                nodes['Footprint Proportion'] = nodes['Footprint Proportion'].replace(np.nan, 0).astype(float)
                nodes['Footprint Mean'] = nodes['Footprint Mean'].replace(np.nan, 0).astype(float)
                nodes['Footprint Stdev'] = nodes['Footprint Stdev'].replace(np.nan, 0).astype(float)
                nodes['Complexity Mean'] = nodes['Complexity Mean'].replace(np.nan, 0).astype(float)
                nodes['Complexity Stdev'] = nodes['Complexity Stdev'].replace(np.nan, 0).astype(float)
                nodes['Perimeter Total'] = nodes['Perimeter Total'].replace(np.nan, 0).astype(float)
                nodes['Perimeter Mean'] = nodes['Perimeter Mean'].replace(np.nan, 0).astype(float)
                nodes['Perimeter Stdev'] = nodes['Perimeter Stdev'].replace(np.nan, 0).astype(float)
                nodes['Counts'] = nodes['Counts'].replace(np.nan, 0).astype(int)
            
            print(f'Building attributes computed. Time taken: {round(time.time() - start)}.')

            # If pop_attr is True, compute and add population attributes.
            if pop_attr:
                large_countries = ['United States']
                groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

                # Use csv for small countries
                if not use_tif and self.country not in large_countries:
                    pop_list, target_cols = get_population_data(self.country, 
                                                                use_tif=use_tif,
                                                                bounding_poly=self.polygon_bounds)
                    

                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = data[0].to_crs(nodes_buffer.crs)
                        res_intersection = proj_data.overlay(nodes_buffer, how='intersection')
                        nodes[groups[i]] = res_intersection.groupby(['osmid'])[data[1]].sum()

                    for name in groups:
                        nodes[name] = nodes[name].replace(np.nan, 0).astype(float)
            
                # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
                elif not use_tif and self.country in large_countries:
                    pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = self.polygon_bounds)
                    
                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = data[0].to_crs(nodes_buffer.crs)
                        res_intersection = proj_data.overlay(nodes_buffer, how='intersection')
                        nodes[groups[i]] = res_intersection.groupby(['osmid'])[data[1]].sum()

                    for name in groups:
                        nodes[name] = nodes[name].replace(np.nan, 0).astype(float)
                   
                # If use_tif is True, use geotiff for population computation instead of csv files  
                if use_tif: 

                    # Using .geotiff for computation
                    src_list, rc_list = get_population_data(self.country, use_tif=use_tif, bounding_poly = self.polygon_bounds)
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

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   

            # If poi_attr is True, compute and add poi attributes.
            if poi_attr:
                # Load poi information 
                poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                with open(poi_path) as poi_filter:
                    poi_filter = json.load(poi_filter)
                
                # Get osm pois based on custom filter
                pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                pois = pois.replace(np.nan, '')

                # Relabel amenities to common typology
                def poi_col(amenity, shop, tourism, leisure):
                    value = amenity
                    if amenity == '' and tourism != '':
                        value = 'entertainment'
                    elif amenity == '' and leisure != '':
                        value = 'recreational'
                    elif amenity == '' and shop in poi_filter['food_set']:
                        value = shop
                    elif amenity == '' and shop not in poi_filter['food_set']:
                        value = 'commercial'
                    
                    return value
                
                pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
                pois = pois[['id', 'osm_type','lon','lat','name','poi_col','geometry']]

                # Remove amenities that have counts of less than n=5
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                pois = project_gdf(pois)
                pois['geometry'] = pois.geometry.centroid

                # Get intersection of amenities with node buffer
                res_intersection = pois.overlay(nodes_buffer, how='intersection')
                poi_series = res_intersection.groupby(['osmid'])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='osmid', columns='poi_col', values=0)

                col_order = list(nodes.columns)
                cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                col_order = col_order + cols

                # Add poi attributes to dataframe of nodes
                nodes = nodes.merge(pois_df, how='left', left_index=True, right_index=True).replace(np.nan,0) 

                for i in cols:
                    if i not in set(nodes.columns):
                        nodes[i] = 0

                nodes = nodes[col_order]
                nodes = nodes.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            print(f'Points of interest computed. Time taken: {round(time.time() - start)}.')

            # If svi_attr is True, compute and add svi attributes.
            if svi_attr:
                svi_path = pkg_resources.resource_filename('urbanity', f"svi_data/{location}.geojson")
                svi_data = gpd.read_file(svi_path)
                svi_data = project_gdf(svi_data)

                # returns gdf of image id, tile_id, indicators, and point coords
                res_intersection = svi_data.overlay(nodes_buffer, how='intersection')
                
                # Compute SVI indices
                indicators = ['Green View', 'Sky View', 'Building View', 'Road View']
                for indicator in indicators:
                    
                    svi_series = res_intersection.groupby(['osmid'])[indicator].mean()
                    nodes[indicator] = svi_series

                    index_mean = nodes[indicator].mean()
                    nodes[indicator] = nodes[indicator].replace(np.nan, index_mean)
            
            # Add computed indices to nodes dataframe
            nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
            edges = edges.reset_index()[['osmid','osm_type','u','v','length','highway','lanes','maxspeed','geometry']]
            print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
            print("Total elapsed time --- %s seconds ---" % round(time.time() - start))
            return G_buff_trunc_loop, nodes, edges

        # If dual is True, construct dual graph with midpoint of original edges as nodes and new edges as adjacency between streets.
        elif dual: 
            # First extract dictionary of osmids and lengths for original nodes associated with each edge
            osmid_view = nx.get_edge_attributes(G_buff_trunc_loop, "osmid")
            osmid_dict = {}
            for u,v in set(osmid_view.items()):
                if u not in osmid_dict:
                    osmid_dict[(u[:2])] = v
                else: 
                    osmid_dict[(u[:2])].append(v)

            length_view = nx.get_edge_attributes(G_buff_trunc_loop, "length")
            length_dict = {}
            for u,v in set(length_view.items()):
                if u not in length_dict:
                    length_dict[(u[:2])] = v
                else: 
                    length_dict[(u[:2])].append(v)

            x_dict = nx.get_node_attributes(G_buff_trunc_loop, "x")
            y_dict = nx.get_node_attributes(G_buff_trunc_loop, "y")

            # Create new placeholder graph and add edges as nodes and adjacency links between edges as new edges
            L = nx.empty_graph(0)
            LG = nx.line_graph(G_buff_trunc_loop)
            L.graph['crs'] = 'EPSG:4326'
            for node in set(G_buff_trunc_loop.edges()):
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
                area_series.name = 'Footprint Proportion'

                L_nodes['unique'] = list(range(len(L_nodes)))
                L_nodes = L_nodes.join(area_series, on = 'unique')

                # Add perimeter
                res_intersection['perimeter'] = res_intersection.geometry.length
                perimeter_series = res_intersection.groupby(['unique'])['perimeter'].sum()

                perimeter_series.name = 'Perimeter (m)'
                L_nodes = L_nodes.join(perimeter_series, on = 'unique')


                 # Add counts
                counts_series = res_intersection['unique'].value_counts()
                counts_series.name = 'Counts'
                L_nodes = L_nodes.join(counts_series, on = 'unique')
                
                L_nodes['Footprint Proportion'] = L_nodes['Footprint Proportion'].replace(np.nan, 0).astype(float)
                L_nodes['Perimeter (m)'] = L_nodes['Perimeter (m)'].replace(np.nan, 0).astype(float)
                L_nodes['Counts'] = L_nodes['Counts'].replace(np.nan, 0).astype(int)
    

            print("--- %s seconds ---" % round(time.time() - start,3))

            return L, L_nodes, L_edges

    def get_aggregate_stats(
        self,
        location: str,
        column: str = None, 
        bandwidth: int = 0,
        use_tif: bool = False,
        network_type: str = 'driving') -> dict:
        """Obtains descriptive statistics for bounding polygon without constructing network. Users can specify bounding polygon either by drawing on the map object, or uploading a geojson/shapefile.
        If geojson/shape file contains multiple geometric objects, descriptive statistics will be returned for all entities. Results are returned in dictionary format. 

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            column (str): Id or name column to identify zones. If None, uses shapefile index column.
            data_path(str): Accepts path to shapefile or geojson object
            bandwidth (int): Distance (m) to buffer site boundary. Defaults to 0.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
        Returns:
            dict: Dictionary of aggregate values for polygon
        """

        start = time.time()
        try:
            fp = get_data(location, directory = self.directory)
            print('Creating data folder and downloading osm street data...')
        except KeyError:
            fp = get_data(self.country, directory = self.directory)
            print(f"KeyError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
        except KeyError:
            raise ValueError('No osm data found for specified location.')

        print('Data extracted successfully. Computing aggregates from shapefile.')

        # Create dictionary keys based on column elements
        attr_stats = {}
        if column == 'name':
            column = 'name_id'
            self.polygon_bounds.rename(columns={'name':column}, inplace=True)
        try:
            for name in self.polygon_bounds[column]:
                attr_stats[name] = {}
        except KeyError:
            for name in self.polygon_bounds.index:
                attr_stats[name] = {}

        # Load Global Data
        local_crs = project_gdf(self.polygon_bounds).crs

        # Points of Interest
        poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
        with open(poi_path) as poi_filter:
            poi_filter = json.load(poi_filter)

        def poi_col(amenity, shop, tourism, leisure):
            value = amenity
            if amenity == '' and tourism != '':
                value = 'entertainment'
            elif amenity == '' and leisure != '':
                value = 'recreational'
            elif amenity == '' and shop in poi_filter['food_set']:
                value = shop
            elif amenity == '' and shop not in poi_filter['food_set']:
                value = 'commercial'
            
            return value
        
        # Population
        large_countries = ['United States']
        groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

        if use_tif: 
            # Using .tif for computation
            src_list, rc_list = get_population_data(self.country, use_tif=use_tif)

        if not use_tif and self.country not in large_countries:
            pop_list, target_cols = get_population_data(self.country, use_tif=use_tif, bounding_poly=self.polygon_bounds)
            for i in range(len(pop_list)):
                pop_list[i] = pop_list[i].to_crs(local_crs)

        if not use_tif and self.country in large_countries:    
            pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly=self.polygon_bounds)
            for i in range(len(pop_list)):
                pop_list[i] = pop_list[i].to_crs(local_crs)

        # Get individual polygon data
        for i, key in enumerate(attr_stats):

            # Project and buffer original polygon
            proj_gdf = project_gdf(self.polygon_bounds.iloc[[i],:])
            proj_gdf_buffered = proj_gdf.buffer(bandwidth)
            proj_gdf_buffered = gpd.GeoDataFrame(data=proj_gdf[column], crs = proj_gdf.crs, geometry = proj_gdf_buffered)
            area = proj_gdf_buffered.geometry.area.values.item() / 1000000
            
            # Obtain pyrosm query object within spatial bounds
            original_bbox = self.polygon_bounds.iloc[[i],:].geometry.values[0]
            buffered_tp = buffer_polygon(self.polygon_bounds.iloc[[i],:], bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
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

            # Remove self loops
            G_buff_trunc_loop = G_buff_trunc.copy()
            G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

            nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)
            nodes = nodes.fillna('')
            
            print(key)
            # Add geometric/metric attributes
            attr_stats[key]["No. of Nodes"] = len(nodes)
            attr_stats[key]["No. of Edges"] = len(edges)
            attr_stats[key]["Area (km2)"] = round(area,2)
            attr_stats[key]["Node density (km2)"] = round(attr_stats[key]["No. of Nodes"] / area, 2)
            attr_stats[key]["Edge density (km2)"]  = round(attr_stats[key]["No. of Edges"] / area, 2)
            attr_stats[key]["Total Length (km)"] = round(G_buff_trunc_loop.size(weight='length')/1000,2)
            attr_stats[key]["Mean Length (m) "] = round(attr_stats[key]["Total Length (km)"] / attr_stats[key]["No. of Edges"] * 1000, 2)
            attr_stats[key]["Length density (km2)"] = round(attr_stats[key]["Total Length (km)"] /area, 2)
            attr_stats[key]["Mean Degree"] = round(2 * attr_stats[key]["No. of Edges"] / attr_stats[key]["No. of Nodes"], 2)
            attr_stats[key]["Mean Neighbourhood Degree"] = round(sum(nx.average_neighbor_degree(G_buff_trunc_loop).values()) / len(nodes), 2)

            # Add Points of Interest
            pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])

            if pois is not None:
                pois = pois.replace(np.nan, '')
                for c in ['shop', 'tourism', 'leisure']:
                    if c not in pois.columns:
                        pois[c] = ''

                pois['poi_col'] = pois.apply(lambda row: poi_col(row['amenity'], row['shop'], row['tourism'], row['leisure']), axis=1)
                pois = pois[['id', 'osm_type','lon','lat','name','poi_col','geometry']]
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=2)

                if len(pois) == 0:
                    cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                    for i in cols:
                        attr_stats[key][i] = 0
                
                else: 
                    pois = project_gdf(pois)
                    pois['geometry'] = pois.geometry.centroid
                    res_intersection = pois.overlay(proj_gdf_buffered, how='intersection')
                    poi_series = res_intersection.groupby([column])['poi_col'].value_counts()

                    cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                    for i in cols:
                        attr_stats[key][i] = 0
                    
                    for poi, counts in poi_series.items():
                        attr_stats[key][str.title(poi[1])] = counts

            else:
                cols = ['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social']
                for i in cols:
                    attr_stats[key][i] = 0
                

            # Add Buildings
            building = osm.get_buildings()
            building_proj = project_gdf(building)
            building_polygon = fill_and_expand(building_proj)

            res_intersection = proj_gdf_buffered.overlay(building_polygon, how='intersection')

            building_area = res_intersection.geometry.area.sum() / 1000000
            attr_stats[key]["Building Footprint (Proportion)"] = round(building_area/attr_stats[key]["Area (km2)"]*100,2)
            attr_stats[key]["Mean Building Footprint (m2)"] = round(res_intersection.geometry.area.mean(),2)
            attr_stats[key]["Building Footprint St.dev (m2)"] = round(res_intersection.geometry.area.std(),2)
            attr_stats[key]["Total Building Perimeter (m)"] = round(res_intersection.geometry.length.sum(), 2)
            attr_stats[key]["Mean Building Perimeter (m)"] = round(res_intersection.geometry.length.mean(), 2)
            attr_stats[key]["Building Perimeter St.dev (m)"] = round(res_intersection.geometry.length.std(),2)
            attr_stats[key]["Mean Building Complexity"] = round(np.mean(res_intersection.geometry.length / np.sqrt(np.sqrt(res_intersection.geometry.area))),2)
            attr_stats[key]["Building Complexity St.dev"] = round(np.std(res_intersection.geometry.length / np.sqrt(np.sqrt(res_intersection.geometry.area))),2)


            # Add Population

            if use_tif:
                for group, src, rc in zip(groups, src_list, rc_list):
                    stats = ['sum']

                    result = rasterstats.zonal_stats(
                        original_bbox, 
                        rc[0], 
                        nodata = 0, 
                        affine = src.transform, 
                        stats = stats
                    )
                    attr_stats[key][group] = round(result[0]['sum'])
                
            if not use_tif:
                for i in range(len(pop_list)):
                    res_intersection = pop_list[i].overlay(proj_gdf_buffered, how='intersection')
                    attr_stats[key][groups[i]] = np.sum(res_intersection.iloc[:,0])


        return attr_stats

    # Not implemented yet
    # def get_building_network(
    #         self, 
    #         network_type: str = 'driving',
    #         bandwidth: int = 200) -> MultiDiGraph:
    #     """Generate a network where nodes correspond to building centroids and edges connect buildings within threshold distance of one another. 

    #     Args:
    #         network_type (str, optional): Specified OpenStreetMap transportation mode. Defaults to 'driving'.

    #     Raises:
    #         Exception: No bounding box found. 

    #     Returns:
    #         nx.MultiDiGraph: Returns a building network object. 
    #     """        
    #     start = time.time()
    #     print('Creating data folder and downloading osm building data...')
    #     fp = get_data(self.country, directory = self.directory)
    #     print('Data extracted successfully. Proceeding to construct building network.')

    #     # Project and buffer original polygon to examine nodes outside boundary
    #     try:
    #         original_bbox = self.polygon_bounds.geometry.values[0]
    #         buffered_tp = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
    #         buffered_bbox = buffered_tp.geometry.values[0]
    #     # catch when it hasn't even been defined 
    #     except (AttributeError, NameError):
    #         raise Exception('Please delimit a bounding box.')

    #     # Obtain nodes and edges within buffered polygon
    #     osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)

    #     # Get buildings
    #     buildings = osm.get_buildings()
    #     proj_buildings = project_gdf(buildings)

    #     # Get adjacency based on spatial intersection
    #     building_neighbours = {}
    #     for i,b in zip(proj_buildings.id,proj_buildings.geometry):
    #         s = proj_buildings.intersects(b.buffer(100))
    #         building_neighbours[i] = proj_buildings.id[s[s].index].values

    #     # Set centroid as geometry and create lon and lat columns
    #     proj_buildings['center'] = proj_buildings.geometry.centroid
    #     proj_buildings = proj_buildings.set_geometry("center")
    #     buildings = proj_buildings.to_crs(4326)
    #     buildings['x'] = buildings.geometry.x
    #     buildings['y'] = buildings.geometry.y
    #     buildings = buildings.set_index('id')

    #     # Create building network graph
    #     # Get dict of building nodes and their attributes
    #     id_to_attributes = {}
    #     for node in set(buildings.index):
    #         id_to_attributes[node] = buildings.loc[node].to_dict()

    #     B = nx.empty_graph(0)
    #     B.graph['crs'] = 'EPSG:4326'

    #     # Add nodes
    #     for node in set(buildings.index):
    #         B.add_node(node)
    #     nx.set_node_attributes(B, id_to_attributes)

    #     # Add edges
    #     for node, neighbours in building_neighbours.items():
    #         for neighbour in neighbours:
    #             B.add_edge(node, neighbour)

    #     # Compute euclidean distance between adjacent building centroids
    #     id_to_x = dict(zip(buildings.index, buildings.x))
    #     id_to_y = dict(zip(buildings.index, buildings.y))
        
    #     distance_between_buildings = {}
    #     for pair in set(B.edges()):
    #         distance_between_buildings[pair] = great_circle_vec(id_to_x[pair[0]], id_to_y[pair[0]], id_to_x[pair[1]], id_to_y[pair[1]])
        
    #     nx.set_edge_attributes(B, distance_between_buildings, 'length')

    #     # Identify largest weakly connected component
    #     max_wcc = max(nx.connected_components(B), key=len)
    #     B_max = nx.subgraph(B, max_wcc)

    #     B_nodes, B_edges = graph_to_gdf(B_max, nodes=True, edges=True, dual=True)
    #     B_nodes = B_nodes.fillna('')

    #     print("--- %s seconds ---" % round(time.time() - start,3))

    #     return B_max, B_nodes, B_edges