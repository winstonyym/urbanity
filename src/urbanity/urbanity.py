# import base packages
import warnings
# Catch known warnings from shapely and geopandas
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
import ee
import json
import time
import math
import glob
import requests
import subprocess
import pkg_resources
from dotenv import load_dotenv

# import external functions and classes
import networkit
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Polygon, box
from shapely.ops import unary_union, polygonize, snap

import ipyleaflet
from ipyleaflet import basemaps, basemap_to_tiles, Icon, Marker, LayersControl, \
                       LayerGroup, DrawControl, FullScreenControl, ScaleControl, LocalTileLayer, GeoData

try:
    import pyrosm  # optional heavy dependency
    from pyrosm import get_data
except ImportError:  # pyrosm not installed
    pyrosm = None
    def get_data(*args, **kwargs):  # type: ignore
        raise ImportError("pyrosm not installed. Install via 'pip install urbanity[osm]' or conda-forge: 'mamba install pyrosm'.")
from scipy.stats import entropy

# import module functions and classes
from .utils import get_country_centroids, finetune_poi, \
                   get_available_precomputed_network_data, most_frequent, gdf_to_poly
from .geom import *
from .building import *
from .svi import parallel_download_image_in_tiles, assign_closest_points_to_linestring_and_sample
from .population import get_meta_population_data, get_tiled_population_data, raster2gdf, extract_tiff_from_shapefile, load_npz_as_raster, mask_raster_with_gdf, \
                        is_bounding_box_within, find_valid_tif_files, mask_raster_with_gdf_large_raster, download_pop_tiff_from_path, merge_raster_list
from .topology import compute_centrality, merge_nx_property, merge_nx_attr
from .satellite import get_and_combine_tiles, get_grid_size, get_tiles_from_bbox, download_satellite_tiles_from_bbox, view_satellite_building_pair, get_max_img_dims, get_building_image_chips, \
                       get_tiles_gdf, download_tiff_from_path, gee_layer_from_boundary, merge_raster_to_gdf

from .data_class import UrbanGraph
from ee.ee_exception import EEException

# Import country coords
country_dict = get_country_centroids()

# Load any environment variables
load_dotenv()
MAPILLARY_API_SECRET = os.environ.get("MAPILLARY_API_SECRET")
MAPILLARY_API_TOKEN = os.environ.get("MAPILLARY_API_TOKEN")
MAPBOX_API_TOKEN = os.environ.get("MAPBOX_API_TOKEN")
LCZ_URL = os.environ.get("LCZ_URL")

class Map(ipyleaflet.Map):

    def __init__(self, country: str = None, **kwargs):
        """Instantiates a map object that inherits from ipyleaflet.Map. 

        Args:
            country (str, optional): Name of country to position map view. Defaults to None.
        """        
        self.location = country
        self.bbox = None
        self.polygon_bounds = None
        self.network = []
        self.target_cols = []
        self.buildings = None
        self.population = []
        self.all_svi = None
        self.svi = None
        self.pois = None
        self.urban_plots = None
        self.plot_geom = None
        self.objects = None
        self.connections = None
        self.segmented_image = None

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

        if filepath[-7:] == 'parquet':
            gdf = gpd.read_parquet(filepath)
        else:
            gdf = gpd.read_file(filepath)
        
        # Check for and drop timestamp column
        timestamp_columns = gdf.select_dtypes(include=['datetime64', 'timedelta64']).columns
        gdf = gdf.drop(columns=timestamp_columns)

        
        if len(gdf) > 1:
            gdf = dissolve_poly(gdf, self.country)

        # Assign polygon boundary attribute to polygon object
        gdf = gdf.set_crs('epsg:4326', allow_override=True)
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
    

    def check_osm_buildings(self,
                            location: str,
                            column: str = "index") -> gpd.GeoDataFrame:
        """Function to check the attribute completeness for OSM buildings as implemented in: https://ual.sg/publication/2020-3-dgeoinfo-3-d-asean/

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            column (str): Accepts name of column with zone ID or name or defaults to use index.

        Returns:
            gpd.DataFrame: A geopandas dataframe with attribute completeness for OSM buildings
        """

        # First step - Check if bounding box is defined
        try:
            original_bbox = self.polygon_bounds.iloc[[0]].geometry[0]
            # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')
        
        # Obtain filepath to OSM data
        try:
            fp = get_data(location, directory = self.directory)
            print(f'Getting osm building data for {location}')
        except ValueError:
            fp = get_data(self.country, directory = self.directory)
            print(f'ValueError: No pre-downloaded osm data available for {location}, will instead try for {self.country}.')
        
        # Create dictionary keys based on column elements
        attr_stats = {}
        if column == 'index':
            print('No column name specified, using index as column name.')
            for name in self.polygon_bounds.index:
                attr_stats[name] = {}
        else:
            for name in self.polygon_bounds[column]:
                attr_stats[name] = {}

        # Get individual polygon data
        for i, key in enumerate(attr_stats):
            print(f"Checking building data for: {key}")
            # Set bounding box
            original_bbox = self.polygon_bounds.iloc[[i]].geometry[i]
            
            # Get OSM parser
            osm = pyrosm.OSM(fp, bounding_box=original_bbox)

            # Retrieve buildings
            buildings = osm.get_buildings()
            num_buildings = len(buildings)
            attr_stats[key]['No. of Buildings'] = num_buildings

            # Compute attributes
            num_height = len(buildings[~buildings['height'].isna()]) if ('height' in buildings.columns) else 0
            perc_num_height = round(num_height/num_buildings*100,2) if ('height' in buildings.columns) else 0
            attr_stats[key]['No. w/ Height'] = num_height
            attr_stats[key]['Perc w/ Height'] = perc_num_height

            num_levels = len(buildings[~buildings['building:levels'].isna()]) if ('building:levels' in buildings.columns) else 0
            perc_num_levels = round(num_levels/num_buildings*100,2) if ('building:levels' in buildings.columns) else 0
            attr_stats[key]['No. w/ Levels'] = num_levels
            attr_stats[key]['Perc w/ Levels'] = perc_num_levels

        df = pd.DataFrame(attr_stats).transpose()
        gdf = gpd.GeoDataFrame(data=df, crs=self.polygon_bounds.crs, geometry = self.polygon_bounds['geometry'])
        return gdf

    def check_population(self,
                        year: int = 2020) -> [dict, gpd.GeoDataFrame]:
        """Function to check the correspondence of Meta's high resolution population counts (30m) with WorldPop (100m) UN-Adjusted dataset:
        WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton; Department of Geography and Geosciences, 
        University of Louisville; Departement de Geographie, Universite de Namur) and Center for International Earth Science Information Network (CIESIN), 
        Columbia University (2018). Global High Resolution Population Denominators Project - Funded by The Bill and Melinda Gates Foundation (OPP1134076).

        Args:
            year (int): Specific year to extract World Population data.

        Returns:
            dict: A dictionary consisting of evaluation metrics 
            gpd.DataFrame: A geopandas dataframe with attribute completeness for OSM buildings
        """

        # Get Worldpop .tiff address from ISO code
        ISO_path = pkg_resources.resource_filename('urbanity', 'map_data/GADM_links.json')
        with open(ISO_path) as file:
          ISO = json.load(file)

        path = f'https://data.worldpop.org/GIS/Population/Global_2000_{year}_Constrained/{year}/BSGM/{ISO[self.country]}/{ISO[self.country].lower()}_ppp_{year}_UNadj_constrained.tif'
        maxar_path = f'https://data.worldpop.org/GIS/Population/Global_2000_{year}_Constrained/{year}/maxar_v1/{ISO[self.country]}/{ISO[self.country].lower()}_ppp_{year}_UNadj_constrained.tif'
        # Check if data folder exists, else create one
        if os.path.isdir('./data'):
            self.directory = "./data"
        else:
            os.makedirs('./data')
            self.directory = "./data"

        # Download Worldpop data for specified `year`
        filename = path.split('/')[-1].replace(" ", "_")  # be careful with file names
        filename_maxar = maxar_path.split('/')[-1].replace(" ", "_")  # be careful with file names
        file_path = os.path.join(self.directory, filename)
        file_path_maxar = os.path.join(self.directory, filename_maxar)

        if not os.path.exists(file_path):
            print('Raster file not found in data folder, proceeding to download.')
            r = requests.get(path, stream=True)
            if r.ok:
                print(f"Saved raster file for {self.country} to: \n", os.path.abspath(file_path))
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
            else:
                r = requests.get(maxar_path, stream=True)
                file_path = file_path_maxar
                print(f"Saved raster file for {self.country} to: \n", os.path.abspath(file_path))
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())

        else:
            print('Data found! Proceeding to load population data. ')

        
        # Load Worldpop data
        print('Loading Worldpop Population Data...')
        from_raster_100m = raster2gdf(file_path, zoom=True, boundary=self.polygon_bounds)
        from_raster_100m['grid_id'] = range(len(from_raster_100m))

        # Load Meta Population Data
        print('Loading Meta Population Data...')
        tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
        with open(tile_countries_path, 'r') as f:
            tile_dict = json.load(f)
                    
        tiled_country = [country[:-13] for country in list(tile_dict.keys())]
        groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']
        
        # Use non-tiled .csv for small countries
        if self.country not in tiled_country:
            print('Using non-tiled population data.')
            pop_gdf, target_col = get_meta_population_data(self.country, 
                                                        bounding_poly=self.polygon_bounds,
                                                        all_only = True)

        # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
        elif self.country in tiled_country:
            print('Using tiled population data.')
            pop_gdf, target_col = get_tiled_population_data(self.country, 
                                                              bounding_poly = self.polygon_bounds, 
                                                              all_only=True)
        # Preprocess both population files; Add aggregate meta to Worldpop grids

    
        res_intersection = gpd.overlay(pop_gdf, from_raster_100m, how='intersection')
        aggregate_series = res_intersection.groupby(['grid_id'])[target_col].sum()
        combined = from_raster_100m.merge(aggregate_series, on ='grid_id')
        # Get non-empty cells
        non_empty = combined[combined['value']!=-99999.0].copy()
        
        # Get percentage of correct cells
        perc_hits = (len(combined[((combined['value'] == -99999.0) | (combined['value'] == 0)) & (combined[target_col]==0)]) + 
                        len(combined[(combined['value'] > 0) & (combined[target_col]>0)])) / len(combined)
        
        # Get total population counts
        meta_total = non_empty[target_col].sum()
        worldpop_total = non_empty['value'].sum()
        
        # Get correlation between non-empty cells
        world_meta_corr = non_empty['value'].corr(non_empty[target_col])
        
        # Get absolute difference
        non_empty.loc[:,'deviance'] = non_empty['value'] - non_empty[target_col]
        non_empty.loc[:,'abs_deviance'] = abs(non_empty['value'] - non_empty[target_col])
        mean_absolute_error = non_empty['abs_deviance'].mean()
        
        values_dict = {}
        values_dict['Grids hit percentage'] = perc_hits
        values_dict['Meta population total'] = meta_total
        values_dict['Worldpop population total'] = worldpop_total
        values_dict['Worldpop/meta correlation'] = world_meta_corr
        values_dict['Mean absolute error'] = mean_absolute_error
        
        return values_dict, non_empty


    def get_network_layer(
            self, 
            location: str = '',
            network_filepath: str = '', 
            bandwidth: int = 100,
            network_type: str = 'driving',
            add_svi: bool = True,
            segment_svi: bool = True,
            fill_svi_gaps: str = 'local',
            bin_distance: int = 20,
            v_threshold: float = 1.0,
            proximity_distance: int = 10):
        """Method to assign urban network layer information (optionally add SVI from Mapillary). 
        Bandwidth (m) can be specified to buffer network, obtaining neighbouring nodes within buffered area of network.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            network_filepath (str): Specify path to osm.pbf file.
            bandwidth (int): Distance to extract information beyond network. Defaults to 100.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            add_svi (bool): If True, collect street view imagery data from Mapillary and add it to network. Defaults to True.
            segment_svi (bool): If True, segments SVI based on street segments. Defaults to True.
            fill_svi_gaps (str): Specify approach to fill gaps in SVI data. Accepts one of ['local', 'spatial_tile', 'global']. If 'local', semantic indicators are propagated along network edges from neighboring streets. 
            If 'spatial_tile', the average within each tile based on Mapillary vector tile ID (zoom 14) is imputed for streets. Else, 'global' uses the global average mean value for the entire planning area. Defaults to 'local'. 
            bin_distance (int): Distance to sample SVI for computational tractibility. Defaults to sampling one image every 20 meters.
            v_threshold (float): Minimum visual complexity threshold to retain segmented images. Defaults to 1.0.
            proximity_distance (int): Distance to include SVI based on adjacent distance to street segment. Defaults to 10 meters and excludes points that fall beyond this distance. 
        """

        # If precomputed available, use precomputed
        self.location = location
        start = time.time()

        if self.network:
            print('Network data found, skipping re-computation')
            G_buff_trunc_loop, nodes, edges = self.network[0], self.network[1], self.network[2]
            original_bbox = self.polygon_bounds.geometry[0]
            buffered_tp = self.polygon_bounds.copy()
            buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
            './data/'
            osm = pyrosm.OSM('./data/temp.osm.pbf', bounding_box=buffered_bbox)
        else:
            if network_filepath == '':
                try:
                    fp = get_data(self.location, directory = self.directory)
                    print('Creating data folder and downloading osm street data...')
                except ValueError:
                    fp = get_data(self.country, directory = self.directory)
                    print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
                except ValueError:
                    raise ValueError('No osm data found for specified location.')

                network_filepath = fp
                print('Data extracted successfully. Proceeding to construct street network.')

            # Project and buffer original polygon to examine nodes outside boundary
            try:
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                buffered_bbox = buffered_tp.geometry.values[0]
            # catch when it hasn't even been defined 
            except (AttributeError, NameError):
                raise Exception('Please delimit a bounding box.')
            
            # Obtain nodes and edges within buffered polygon
            data_root = './data/'
            if not os.path.exists(data_root):
                os.makedirs(data_root)

            poly_path = os.path.join(data_root, 'temp.poly')
            osm_path = os.path.join(data_root, 'temp.osm.pbf')

            if os.path.isfile(poly_path):
                os.remove(poly_path)

            if os.path.isfile(osm_path):
                os.remove(osm_path)
            self.polygon_bounds = self.polygon_bounds[['geometry']]
            self.polygon_bounds = self.polygon_bounds.reset_index()
            self.polygon_bounds.columns = ['boundary_id', 'geometry']

            gdf_to_poly(self.polygon_bounds, poly_path, column='boundary_id')
            cmd = [
                    "osmium", "extract", 
                    "-p", poly_path,
                    network_filepath,
                    "-o", osm_path
                ]

            subprocess.run(cmd, capture_output=False, text=True)

            osm = pyrosm.OSM(osm_path, bounding_box=buffered_bbox)
            
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

            # Fill NA and drop incomplete columns
            nodes = nodes.fillna('')
            edges = edges.fillna('')
            nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
            edges = edges.reset_index()[['u','v','length','geometry']]
    
            # Assign unique IDs
            nodes['intersection_id'] = nodes.index
            nodes = nodes[['intersection_id','osmid', 'x', 'y', 'geometry']]
            
            edges = edges[['u', 'v', 'length','geometry']]

            edges =  drop_duplicate_lines(edges)

            self.network.append(G_buff_trunc_loop)
            self.network.append(nodes)
            self.network.append(edges)

            print(f'Network constructed. Time taken: {round(time.time() - start)}.')

        if self.all_svi is None and add_svi:

            image_gdf = parallel_download_image_in_tiles(self.polygon_bounds, api_key=MAPILLARY_API_TOKEN)
            svi_gdf = assign_closest_points_to_linestring_and_sample(image_gdf, edges, bin_distance = bin_distance)

            # Assign svis to map properties
            self.all_svi = image_gdf
            self.svi = svi_gdf

        if self.segmented_image is None and segment_svi:

            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            from .svi import parallel_segment_images

            processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic", use_fast=True)
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            now = time.time()

            segmented_images = parallel_segment_images(self.svi, 
                                                    processor,
                                                    model,
                                                    MAPILLARY_API_TOKEN, 
                                                    device)
            print(f'Time taken for image segmentation: {time.time()-now} seconds.')

            segmented_images = segmented_images[segmented_images['Visual Complexity']>=v_threshold]
        
            self.segmented_image = segmented_images
        
        if fill_svi_gaps == 'local':
            self.network[2] = self.network[2].iloc[:,:5]
            from .svi import fill_street_network
            segmented_network = fill_street_network(self.segmented_image, self.network[2])
            self.network[2] = segmented_network

        elif fill_svi_gaps == 'global':
            # Columns to aggregate from segmented images
            aggr_cols = ['nearest_street_id'] + list(self.segmented_image.columns[9:])

            # Keep base street attributes
            self.network[2] = self.network[2].iloc[:, :5]

            # Compute mean per street
            image_gdf_grouped = (
                self.segmented_image[aggr_cols]
                .groupby('nearest_street_id', as_index=False)
                .mean()
                .rename(columns={'nearest_street_id': 'street_id'})
            )

            # Merge street-level image statistics
            self.network[2] = self.network[2].merge(
                image_gdf_grouped,
                how='left',
                on='street_id'
            )

            # Fill missing values in using street-level means
            mean_cols = image_gdf_grouped.columns.drop('street_id')

            for col in mean_cols:
                if col in self.network[2].columns:
                    self.network[2][col] = self.network[2][col].fillna(
                        self.network[2][col].mean()
                    )

        elif fill_svi_gaps == 'spatial_tile':
            # Keep base street attributes
            aggr_cols = list(self.segmented_image.columns[9:])
            self.network[2] = self.network[2].iloc[:, :5]

            tile_gdf = get_tile_geometry(buffered_tp)
            tile_gdf = tile_gdf.set_crs(self.polygon_bounds.crs)
            proj_tile_gdf = project_gdf(tile_gdf)
            proj_edges = project_gdf(self.network[2])
            proj_edges['geometry'] = proj_edges.geometry.centroid

            tile_network_gdf = proj_edges.overlay(proj_tile_gdf)[['street_id', 'tile_id']]

            self.network[2] = self.network[2].merge(
                tile_network_gdf[['street_id', 'tile_id']],
                how='left',
                on='street_id'
            )

            # --------------------------------------------------
            # 3. Compute TILE-level means
            # --------------------------------------------------
            tile_mean = (
                self.segmented_image
                .groupby('tile_id', as_index=False)[aggr_cols]
                .mean()
                .rename(columns={c: f"{c}_tile" for c in aggr_cols})
            )

            self.network[2] = self.network[2].merge(
                tile_mean,
                how='left',
                on='tile_id'
            )

            # --------------------------------------------------
            # 4. Compute STREET-level means
            # --------------------------------------------------
            street_mean = (
                self.segmented_image[['nearest_street_id'] + aggr_cols]
                .groupby('nearest_street_id', as_index=False)
                .mean()
                .rename(columns={'nearest_street_id': 'street_id'})
            )

            self.network[2] = self.network[2].merge(
                street_mean,
                how='left',
                on='street_id'
            )

            # --------------------------------------------------
            # 5. Hierarchical filling: street → tile → global
            # --------------------------------------------------
            for col in aggr_cols:
                self.network[2][col] = (
                    self.network[2][col]
                    .fillna(self.network[2][f"{col}_tile"])
                    .fillna(self.network[2][col].mean())
                )

            # --------------------------------------------------
            # 6. Cleanup helper columns
            # --------------------------------------------------
            self.network[2].drop(
                columns=[f"{c}_tile" for c in aggr_cols],
                inplace=True
            )

    def get_population_layer(
            self,
            population_layer = 'meta',
            bandwidth: int = 100,
            temporal_years: list = [2025]

    ):
        """Method to collect population layer
        Args:
            population_layer (str): Specify population layer to add. Accepts one of ['meta', 'ghs']. Defaults to 'meta'.
        """
                    # Project and buffer original polygon to examine nodes outside boundary
        start = time.time()

        try:
            original_bbox = self.polygon_bounds.geometry[0]
            buffered_tp = self.polygon_bounds.copy()
            buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Add population
        if self.population:
            print('Population data found, skipping re-computation.')
        else:
            if population_layer == 'meta': 
                print('Collecting Meta population counts...')
                long_min, lat_min, long_max, lat_max = self.polygon_bounds.geometry.total_bounds

                pop_subgroups = ['population', 'children', 'youth', 'elderly', 'men', 'women']
                # Example usage
                bounding_box = self.polygon_bounds.geometry.total_bounds  # Example bounding box (xmin, ymin, xmax, ymax)
                
                tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                with open(tile_countries_path, 'r') as f:
                    tile_dict = json.load(f)
                    
                tiled_country = [country[:-13] for country in list(tile_dict.keys())]
                
                # Use csv for small countries
                if self.country not in tiled_country:
                    print('Using non-tiled population data.')
                    pop_list, target_cols = get_meta_population_data(self.country, 
                                                                bounding_poly=self.polygon_bounds)
                    
                # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
                elif self.country in tiled_country:
                    print('Using tiled population data.')
                    pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = self.polygon_bounds)
                
                self.population = pop_list
                self.target_cols = target_cols
            
                groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']
                
            if population_layer == 'ghs':
                print('Adding GHS population counts to plots...')
                # Load global tile dataframe and fine overlapping grid
                ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
                ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)

                overlapping_grid = ghs_global_grid.overlay(buffered_tp)
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                # Loop through each year and obtain tif file
                origin_gdf_pop = gpd.GeoDataFrame()

                for k, year in enumerate(temporal_years):
                    # Only compute geometry once
                    if k == 0:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=False)

                        elif len(overlapping_grid) > 1: 
                            raster_list_built = []
                            raster_list_pop = []
                            for i, row in overlapping_grid.iterrows():
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"                           
                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_pop = merge_raster_list(raster_list_pop)
                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=False)
                        
                        
                        raster_gdf_pop.columns = [str(year), 'geometry']
                        origin_gdf_pop = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1))
                        
                    else:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"

                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        elif len(overlapping_grid) > 1: 
                            raster_list_pop = []

                            for i, row in overlapping_grid.iterrows():
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"

                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_pop = merge_raster_list(raster_list_pop)

                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        
                        raster_gdf_pop.columns = [str(year)]
                        origin_gdf_pop  = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1)) 

                temporal_rename_pop = {str(i):f'{i}_pop' for i in temporal_years}
                temporal_pop_columns = [f'{i}_pop' for i in temporal_years]
                origin_gdf_pop = origin_gdf_pop.rename(columns=temporal_rename_pop)
                self.population = origin_gdf_pop

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')  
        
            
                
    def get_building_layer(
            self, 
            location: str = '',
            network_filepath: str = '', 
            bandwidth: int = 100,
        ):
        """Method to collect building layer

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
        """

        # If precomputed available, use precomputed
        self.location = location
        start = time.time()

        if self.network:
            print('Network data found, skipping re-computation')
            G_buff_trunc_loop, nodes, edges = self.network[0], self.network[1], self.network[2]
            original_bbox = self.polygon_bounds.geometry[0]
            buffered_tp = self.polygon_bounds.copy()
            buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
            './data/'
            osm = pyrosm.OSM('./data/temp.osm.pbf', bounding_box=buffered_bbox)
        else:
            if network_filepath == '':
                try:
                    fp = get_data(self.location, directory = self.directory)
                    print('Creating data folder and downloading osm street data...')
                except ValueError:
                    fp = get_data(self.country, directory = self.directory)
                    print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
                except ValueError:
                    raise ValueError('No osm data found for specified location.')

                network_filepath = fp
                print('Data extracted successfully. Proceeding to construct street network.')

            # Project and buffer original polygon to examine nodes outside boundary
            try:
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                buffered_bbox = buffered_tp.geometry.values[0]
            # catch when it hasn't even been defined 
            except (AttributeError, NameError):
                raise Exception('Please delimit a bounding box.')
            
            # Obtain nodes and edges within buffered polygon
            data_root = './data/'
            if not os.path.exists(data_root):
                os.makedirs(data_root)

            poly_path = os.path.join(data_root, 'temp.poly')
            osm_path = os.path.join(data_root, 'temp.osm.pbf')

            if os.path.isfile(poly_path):
                os.remove(poly_path)

            if os.path.isfile(osm_path):
                os.remove(osm_path)
            self.polygon_bounds = self.polygon_bounds[['geometry']]
            self.polygon_bounds = self.polygon_bounds.reset_index()
            self.polygon_bounds.columns = ['boundary_id', 'geometry']

            gdf_to_poly(self.polygon_bounds, poly_path, column='boundary_id')
            cmd = [
                    "osmium", "extract", 
                    "-p", poly_path,
                    network_filepath,
                    "-o", osm_path
                ]

            subprocess.run(cmd, capture_output=False, text=True)

            osm = pyrosm.OSM(osm_path, bounding_box=buffered_bbox)
            
        # Collect buildings from osm 
        buildings = osm.get_buildings()

        # Process geometry and attributes for Overture buildings
        buildings = preprocess_osm_building_geometry(buildings, minimum_area=30)
        # building_polygon = preprocess_osm_building_attributes(building_polygon, return_class_height=False)

        # Obtain unique ids for buildings
        buildings = assign_numerical_id_suffix(buildings, 'osm')

        id_col = 'osm_id'

        buildings['bid'] = buildings[id_col]
        buildings['bid_area'] = buildings.geometry.area
        buildings['bid_perimeter'] = buildings.geometry.length
        building_centroids = buildings.geometry.centroid
        buildings['bid_centroid'] = buildings.geometry.centroid
        buildings = buildings[['bid', 'bid_area', 'bid_perimeter', 'bid_centroid', 'geometry']]

        # Compute building attributes
        buildings = compute_circularcompactness(buildings, element='bid')
        buildings = compute_convexity(buildings, element='bid')
        buildings = compute_corners(buildings, element='bid')
        buildings = compute_elongation(buildings, element='bid')
        buildings = compute_orientation(buildings, element='bid')
        # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
        buildings = compute_longest_axis_length(buildings, element='bid')
        buildings = compute_equivalent_rectangular_index(buildings, element='bid')
        buildings = compute_fractaldim(buildings, element='bid')
        buildings = compute_rectangularity(buildings, element='bid')
        buildings = compute_square_compactness(buildings, element='bid')
        buildings = compute_shape_index(buildings, element='bid')
        buildings = compute_squareness(buildings, element='bid')
        buildings = compute_complexity(buildings, element='bid')

        # Compute building heights
        ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
        ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)
        
        overlapping_grid = ghs_global_grid.overlay(buffered_tp)

        ghs_building_height_path = pkg_resources.resource_filename('urbanity', 'building_height_data/building_grids.json')
        with open(ghs_building_height_path) as f:
            building_height_links = json.load(f)

        # If only one tile
        if len(overlapping_grid) == 1:
            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
            target_key = f"R{row}C{col}.parquet"

            buildings = get_and_assign_building_heights(building_height_links[target_key], target_key, buildings)

        elif len(overlapping_grid) > 1: 
            building_height_list = []
            for i, row in overlapping_grid.iterrows():
                row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                target_key = f"R{row}C{col}.parquet"
                building_height = get_building_heights(target_key, target_key)
                building_height_list.append(building_height)
            combined_df = pd.concat(building_height_list)
            combined_gdf = gpd.GeoDataFrame(combined_df, crs='epsg:4326', geometry=combined_df['geometry'])

            buildings = assign_building_heights(combined_gdf, target_key, buildings)

        # buildings = buildings.drop(columns=['bid_centroid'])
        buildings['bid'] = buildings['bid'].astype(str)
        self.buildings = buildings

    def get_street_network(
            self, 
            location: str,
            filepath: str = '',
            bandwidth: int = 100,
            network_type: str = 'driving',
            graph_attr: bool = True,
            catchment_attr: bool = False, 
            building_attr: bool = False,
            pop_attr: bool = False,
            poi_attr: bool = False,
            svi_attr: bool = False,
            edge_attr: bool = False,
            pois_data: str = 'osm',
            building_data: str = 'osm',
            get_precomputed: bool = False,
            temporal_network: bool = False,
            temporal_years: list = list(range(1975, 2030, 5)),
            built_up_threshold: int = 300,
            dual: bool = False) -> [nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Function to generate either primal planar or dual (edge) networks. If multiple geometries are provided, 
        network is constructed for only the first entry. Please merge geometries before use.
        Bandwidth (m) can be specified to buffer network, obtaining neighbouring nodes within buffered area of network.
        *_attr arguments can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            bandwidth (int): Distance to extract information beyond network. Defaults to 100.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            graph_attr (bool): Specifies whether graph metric and topological attributes should be included. Defaults to True.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            svi_attr (bool): Specifies whether street view attributes should be included. Defaults to False.
            edge_attr (bool): If True, computes edge attributes (available for buildings, pois, population, and svi). Defaults to True.
            pois_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) points of interest. Defaults to 'osm'.
            building_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) building footprints. Defaults to 'osm'.
            get_precomputed (bool): If True, directly downloads network data from the Global Urban Network Repository instead of computing. Defaults to False.
            temporal_network (bool): If True, extracts GHS_BUILT_S_R2023 land cover data for target area and assigns temporal information to street network nodes. 
            temporal_years (list): Specify the yearly interval to obtain GHS_BUILT_S_R2023 land cover data. 
            built_up_theshold (int): Specify threshold of built-up surface area to classify as suburban development.
            dual (bool): If true, creates a dual (edge) network graph. Defaults to False.
            
        Raises:
            Exception: No bounding box or polygon found to construct network.

        Returns:
            nx.MultiDiGraph: Urban network in networkX nx.MultiDiGraph format.
            gpd.GeoDataFrame: A geopandas GeoDataFrame containing network nodes (intersections) and contextual spatial features that are assigned via catchment buffer radius set by `bandwidth`.
            gpd.GeoDataFrame: A geopandas GeoDataFrame containing network edges (streets) and contextual spatial features that are assigned via spatial interpolation.
        """ 
        
        # If precomputed available, use precomputed
        if get_precomputed:   
            try:
                network_dataset = pkg_resources.resource_filename('urbanity', "map_data/network_data.json")
                with open(network_dataset, 'r') as f:
                    network_data = json.load(f)
                nodes = gpd.read_file(network_data[f'{location.title()}_nodes_100m.geojson'])
                edges = gpd.read_file(network_data[f'{location.title()}_edges_100m.geojson'])
                return None,nodes,edges
            except:
                get_available_precomputed_network_data()
                return None, None, None

        start = time.time()

        if self.network:
            print('Network data found, skipping re-computation')
            G_buff_trunc_loop, nodes, edges = self.network[0], self.network[1], self.network[2]
        
        else:
            if filepath == '':
                try:
                    fp = get_data(location, directory = self.directory)
                    print('Creating data folder and downloading osm street data...')
                except ValueError:
                    fp = get_data(self.country, directory = self.directory)
                    print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
                except ValueError:
                    raise ValueError('No osm data found for specified location.')

                print('Data extracted successfully. Proceeding to construct street network.')
            elif filepath != '':
                fp = filepath
                print('Data found! Proceeding to construct street network.')

            # Project and buffer original polygon to examine nodes outside boundary
            try:
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                buffered_bbox = buffered_tp.geometry.values[0]

            # catch when it hasn't even been defined 
            except (AttributeError, NameError):
                raise Exception('Please delimit a bounding box.')
            
            if temporal_network:
                # Load global tile dataframe and fine overlapping grid
                ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
                ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)

                overlapping_grid = ghs_global_grid.overlay(buffered_tp)
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth+500)
                # Loop through each year and obtain tif file
                origin_gdf_built = gpd.GeoDataFrame()
                origin_gdf_pop = gpd.GeoDataFrame()

                for k, year in enumerate(temporal_years):
                    # Only compute geometry once
                    if k == 0:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                            raster_gdf_built = raster2gdf(raster_dataset_built, zoom=True, boundary = buffered_tp, same_geometry=False)
                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=False)

                        elif len(overlapping_grid) > 1: 
                            raster_list_built = []
                            raster_list_pop = []
                            for i, row in overlapping_grid.iterrows():
                                target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"                           
                                raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                                raster_list_built.append(raster_dataset_built)
                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_built = merge_raster_list(raster_list_built)
                            mosaic_pop = merge_raster_list(raster_list_pop)
                            raster_gdf_built = raster2gdf(mosaic_built, zoom=True, boundary = buffered_tp, same_geometry=False)
                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=False)
                        
                        raster_gdf_built.columns = [str(year), 'geometry']
                        origin_gdf_built = gpd.GeoDataFrame(pd.concat([origin_gdf_built, raster_gdf_built], axis=1))
                        
                        raster_gdf_pop.columns = [str(year), 'geometry']
                        origin_gdf_pop = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1))
                        
                    else:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"

                            raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                            raster_gdf_built = raster2gdf(raster_dataset_built, zoom=True, boundary = buffered_tp, same_geometry=True)
                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        elif len(overlapping_grid) > 1: 
                            raster_list_built = []
                            raster_list_pop = []

                            for i, row in overlapping_grid.iterrows():
                                target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"

                                raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                                raster_list_built.append(raster_dataset_built)
                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_built = merge_raster_list(raster_list_built)
                            mosaic_pop = merge_raster_list(raster_list_pop)

                            raster_gdf_built = raster2gdf(mosaic_built, zoom=True, boundary = buffered_tp, same_geometry=True)
                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        raster_gdf_built.columns = [str(year)]
                        origin_gdf_built  = gpd.GeoDataFrame(pd.concat([origin_gdf_built, raster_gdf_built], axis=1)) 
                        
                        raster_gdf_pop.columns = [str(year)]
                        origin_gdf_pop  = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1)) 

                temporal_rename = {str(i):f'{i}_pop' for i in temporal_years}
                origin_gdf_pop = origin_gdf_pop.rename(columns=temporal_rename)

            # Obtain nodes and edges within buffered polygon
            data_root = './data/'
            if not os.path.exists(data_root):
                os.makedirs(data_root)

            poly_path = os.path.join(data_root, 'temp.poly')
            osm_path = os.path.join(data_root, 'temp.osm.pbf')

            if os.path.isfile(poly_path):
                os.remove(poly_path)

            if os.path.isfile(osm_path):
                os.remove(osm_path)
            self.polygon_bounds = self.polygon_bounds[['geometry']]
            self.polygon_bounds = self.polygon_bounds.reset_index()
            self.polygon_bounds.columns = ['boundary_id', 'geometry']

            gdf_to_poly(self.polygon_bounds, poly_path, column='boundary_id')
            cmd = [
                    "osmium", "extract", 
                    "-p", poly_path,
                    fp,
                    "-o", osm_path
                ]

            subprocess.run(cmd, capture_output=False, text=True)

            osm = pyrosm.OSM(osm_path, bounding_box=buffered_bbox)

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

            # Fill NA and drop incomplete columns
            nodes = nodes.fillna('')
            edges = edges.fillna('')
     
            nodes = nodes.drop(columns=['tags','timestamp','version','changeset']).reset_index(drop=True)
            highway_edges = edges.reset_index()[['u', 'v', 'highway']]
            edges = edges.reset_index()[['u','v','length', 'geometry']]

            # Assign unique IDs
            nodes['intersection_id'] = nodes.index
            nodes = nodes[['intersection_id','osmid', 'x', 'y', 'geometry']]
        
            edges['edge_id'] = edges.index
            

            temporal_network_list = []
            temporal_nodes_list = []
            if temporal_network:

                # Get highway nodes
                highway_edges['highway'] = highway_edges['highway'].astype(str).str.contains(r'\btrunk\b', 
                                                                case=False,
                                                                na=False).astype(int)    # don’t let NaNs ruin the mask

                highway_edges = highway_edges[highway_edges['highway']==1]
                highway_nodes = np.unique(np.concatenate([highway_edges['u'].unique(),highway_edges['v'].unique()]))


                nodes = nodes.overlay(origin_gdf_built)
                temporal_years = [str(i) for i in temporal_years]
                nodes["highway"] = nodes["osmid"].isin(highway_nodes).astype(int)
                nodes[temporal_years] = nodes[temporal_years].applymap(lambda x: 1 if x >= built_up_threshold else 0)

                
                for year in temporal_years:
                    nodes[year] = (nodes[year].astype(bool) | nodes["highway"].astype(bool)).astype(int)

                    G_buff_trunc_loop_year = G_buff_trunc_loop.copy()
                    nodes_to_remove = list(nodes[nodes[year] == 0]['osmid'].values)
                    G_buff_trunc_loop_year.remove_nodes_from(nodes_to_remove)
                    max_wcc = max(nx.weakly_connected_components(G_buff_trunc_loop_year), key=len)
                    G_buff_trunc_loop_year = nx.subgraph(G_buff_trunc_loop_year, max_wcc)
                    temporal_network_list.append(G_buff_trunc_loop_year)

                    nodes_year = nodes[nodes['osmid'].isin(set(G_buff_trunc_loop_year.nodes))]
                    temporal_nodes_list.append(nodes_year)
                
                # Add population counts
                nodes = nodes.overlay(origin_gdf_pop)
  
            self.network.append(G_buff_trunc_loop)
            self.network.append(nodes)
            self.network.append(edges)

            print(f'Network constructed. Time taken: {round(time.time() - start)}.')
            
        # If not dual representation graph
        if dual == False:
            # Create buffer zone around nodes for area statistics
            if any([catchment_attr, poi_attr, edge_attr, svi_attr, pop_attr, building_attr]):  # Check if any condition is True
                proj_nodes = project_gdf(nodes)
                proj_edges = project_gdf(edges)

                # Buffer around nodes
                nodes_buffer = proj_nodes.copy()
                nodes_buffer['geometry'] = nodes_buffer.geometry.buffer(bandwidth)

            # Compute areal statistics like intersection density / road length density per km2

            if catchment_attr:
                # Add Node Density
                res_intersection = proj_nodes.overlay(nodes_buffer, how='intersection')
                res_intersection['num_nodes'] = 1
                # Use intersection_id_2 because of node-node_buffer intersection
                nodes["intersection_num_nodes"] = res_intersection.groupby(['intersection_id_2'])['num_nodes'].sum().values
                
                # Add Street Length
                res_intersection = proj_edges.overlay(nodes_buffer, how='intersection')
                res_intersection['street_len'] = res_intersection.geometry.length
                nodes["intersection_total_street_length"] = res_intersection.groupby(['intersection_id'])['street_len'].sum().values
                nodes["intersection_total_street_length"] = nodes["intersection_total_street_length"].round(3)

                print(f'Network catchment-{bandwidth}m attributes computed. Time taken: {round(time.time() - start)}.')

            if graph_attr:
                # Add Degree Centrality, Clustering (Weighted and Unweighted)
                if temporal_network:
                    temporal_years = [str(i) for i in temporal_years]

                    # Loop through year and accumulate 
                    for combination in zip(temporal_years,temporal_network_list, temporal_nodes_list):
                        year, year_network, year_node = combination[0], combination[1], combination[2]
                        nodes = merge_nx_property(nodes, year_network.out_degree, f'intersection_degree_{year}', year_node, temporal=True)
                        nodes = merge_nx_attr(year_network, nodes, nx.clustering, f'intersection_clustering_{year}', year_node,  temporal=True)
                        nodes = merge_nx_attr(year_network, nodes, nx.clustering, f'inter_section_weighted_clustering_{year}', year_node, temporal=True, weight='length')
                        
                        #  Add Centrality Measures
                        nodes = compute_centrality(year_network, nodes, networkit.centrality.Closeness, f'Closeness Centrality_{year}', year_node, False, False, temporal=True)
                        nodes = compute_centrality(year_network, nodes, networkit.centrality.Betweenness, f'Betweenness Centrality_{year}', year_node, True, temporal=True)
                        nodes = compute_centrality(year_network, nodes, networkit.centrality.EigenvectorCentrality, f'Eigenvector Centrality_{year}', year_node, temporal=True)
                        nodes = compute_centrality(year_network, nodes, networkit.centrality.KatzCentrality, f'Katz Centrality_{year}', year_node, temporal=True)
                        nodes = compute_centrality(year_network, nodes, networkit.centrality.PageRank, f'PageRank_{year}', year_node, 0.85, 1e-8, networkit.centrality.SinkHandling.NoSinkHandling, True, temporal=True)
                    
                else:
                    nodes = merge_nx_property(nodes, G_buff_trunc_loop.out_degree, 'intersection_degree', None)
                    nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'intersection_clustering', None)
                    nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'inter_section_weighted_clustering', None, weight='length')

                    #  Add Centrality Measures
                    nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Closeness, 'Closeness Centrality', None, False, False)
                    nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Betweenness, 'Betweenness Centrality', None, True)
                    nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.EigenvectorCentrality, 'Eigenvector Centrality', None)
                    nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.KatzCentrality, 'Katz Centrality', None)
                    nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.PageRank, 'PageRank', None, 0.85, 1e-8, networkit.centrality.SinkHandling.NoSinkHandling, True)
                
                print(f'Topologic/metric attributes computed. Time taken: {round(time.time() - start)}.')
            
            # If building_attr is True, compute and add building attributes.
            if building_attr:
                if self.buildings is None: 
                    if building_data == 'osm':
                        # Get building spatial data and project 
                        buildings = osm.get_buildings()

                        # Process geometry and attributes for Overture buildings
                        building_polygon = preprocess_osm_building_geometry(buildings, minimum_area=30)
                        # building_polygon = preprocess_osm_building_attributes(building_polygon, return_class_height=False)

                        # Obtain unique ids for buildings
                        building_polygon = assign_numerical_id_suffix(building_polygon, 'osm')

                    # elif building_data == 'overture':
                    #     # Get buildings
                    #     buildings = get_overture_buildings(location, boundary=self.polygon_bounds)

                    #     # Filter by boundary
                    #     outside_buildings = buildings.overlay(self.polygon_bounds, how='difference')
                    #     buildings = buildings[~buildings['id'].isin(list(outside_buildings['id'].unique()))]
                    
                    #     # Process geometry and attributes for Overture buildings
                    #     building_polygon = preprocess_overture_building_geometry(buildings, minimum_area=30)
                    #     # building_polygon = preprocess_overture_building_attributes(building_polygon, return_class_height=False)

                    #     # Obtain unique ids for buildings
                    #     building_polygon = assign_numerical_id_suffix(building_polygon, 'overture')

                    # if building_data == 'combined':
                    #     overture_buildings = get_overture_buildings(location, boundary=self.polygon_bounds)
                    #     outside_buildings = overture_buildings.overlay(self.polygon_bounds, how='difference')
                    #     overture_buildings = overture_buildings[~overture_buildings['id'].isin(list(outside_buildings['id'].unique()))]
                    #     osm_buildings = get_osm_buildings(location=location, boundary=self.polygon_bounds)

                    #     # Process geometry and attributes for Overture buildings
                    #     overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
                    #     # overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

                    #     # Process geometry and attributes for Overture buildings
                    #     osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
                    #     # osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

                    #     # Obtain unique ids for buildings
                    #     overture_attr_uids = assign_numerical_id_suffix(overture_geom, 'overture')
                    #     osm_attr_uids = assign_numerical_id_suffix(osm_geom, 'osm')

                    #     # Merged building and augment with additional attributes from OSM
                    #     building_polygon= merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
                    #     building_polygon = extract_attributed_osm_buildings(building_polygon, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)

                    if building_data == 'osm':
                        id_col = 'osm_id'
                    # elif building_data == 'overture':
                    #     id_col = 'overture_id'
                    # else:
                    #     id_col = 'building_id'

                    building_polygon['bid'] = building_polygon[id_col]
                    building_polygon['bid_area'] = building_polygon.geometry.area
                    building_polygon['bid_perimeter'] = building_polygon.geometry.length
                    building_polygon = building_polygon[['bid', 'bid_area', 'bid_perimeter', 'geometry']]

                    # Compute building attributes
                    building_polygon = compute_circularcompactness(building_polygon, element='bid')
                    building_polygon = compute_convexity(building_polygon, element='bid')
                    building_polygon = compute_corners(building_polygon, element='bid')
                    building_polygon = compute_elongation(building_polygon, element='bid')
                    building_polygon = compute_orientation(building_polygon, element='bid')
                    # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
                    building_polygon = compute_longest_axis_length(building_polygon, element='bid')
                    building_polygon = compute_equivalent_rectangular_index(building_polygon, element='bid')
                    building_polygon = compute_fractaldim(building_polygon, element='bid')
                    building_polygon = compute_rectangularity(building_polygon, element='bid')
                    building_polygon = compute_square_compactness(building_polygon, element='bid')
                    building_polygon = compute_shape_index(building_polygon, element='bid')
                    building_polygon = compute_squareness(building_polygon, element='bid')
                    building_polygon = compute_complexity(building_polygon, element='bid')
                    # Set computed building data as map attribute
                    self.buildings = building_polygon

                else:
                    print('Building data found, skipping re-computation.')
                    building_polygon = self.buildings

                # Compute and add building attributes
                res_intersection = building_polygon.overlay(nodes_buffer, how='intersection')
                # building_set = building_polygon.iloc[list(res_intersection['bid'].unique()),:]
                res_intersection['area'] = res_intersection.geometry.area
                area_series = res_intersection.groupby(['intersection_id'])['area'].sum()
                total_area = math.pi*bandwidth**2
                area_series = area_series / total_area
                area_series.name = 'intersection_footprint_proportion'
                
                # Obtain proportion 
                nodes = nodes.merge(area_series, on='intersection_id', how='left')
                
                # Obtain mean area
                mean_series = res_intersection.groupby(['intersection_id'])['bid_area'].mean()
                mean_series.name = 'intersection_mean_building_area'
                nodes = nodes.merge(mean_series, on='intersection_id', how='left')

                # Obtain mean area
                std_series = res_intersection.groupby(['intersection_id'])['bid_area'].std()
                std_series.name = 'intersection_std_building_area'
                nodes = nodes.merge(std_series, on='intersection_id', how='left')

                # Add perimeter
                perimeter_series = res_intersection.groupby(['intersection_id'])['bid_perimeter'].sum()
                perimeter_series.name = 'intersection_total_building_perimeter'
                nodes = nodes.merge(perimeter_series, on='intersection_id', how='left')

                perimeter_mean_series = res_intersection.groupby(['intersection_id'])['bid_perimeter'].mean()
                perimeter_mean_series.name = 'intersection_mean_building_perimeter'
                nodes = nodes.merge(perimeter_mean_series, on='intersection_id', how='left')

                perimeter_std_series = res_intersection.groupby(['intersection_id'])['bid_perimeter'].std()
                perimeter_std_series.name = 'intersection_std_building_perimeter'
                nodes = nodes.merge(perimeter_std_series, on='intersection_id', how='left')

                # Add counts
                counts_series = res_intersection.groupby(['intersection_id'])['bid'].count()
                counts_series.name = 'intersection_num_buildings'
                nodes = nodes.merge(counts_series, on='intersection_id', how='left')

                # Add building attributes to node dataframe
                nodes['intersection_footprint_proportion'] = nodes['intersection_footprint_proportion'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_mean_building_area'] = nodes['intersection_mean_building_area'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_std_building_area'] = nodes['intersection_std_building_area'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_total_building_perimeter'] = nodes['intersection_total_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_mean_building_perimeter'] = nodes['intersection_mean_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_std_building_perimeter'] = nodes['intersection_std_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                nodes['intersection_num_buildings'] = nodes['intersection_num_buildings'].replace(np.nan, 0).astype(int)

                # Additional building attributes
                building_attrs = ['bid_complexity', 'bid_circ_compact', 'bid_convexity', 'bid_corners', 'bid_elongation',
                                'bid_orientation', 'bid_perimeter', 'bid_longest_axis_length', 'bid_eri', 'bid_fractaldim',
                                'bid_rectangularity', 'bid_square_compactness', 'bid_shape_idx', 'bid_squareness']
                
                for attr in building_attrs:
                    mean_series = res_intersection.groupby(['intersection_id'])[attr].mean()
                    mean_series.name = f'intersection_mean_building_{attr}' 
                    nodes = nodes.merge(mean_series, on='intersection_id', how='left')
                    nodes[f'intersection_mean_building_{attr}'] = nodes[f'intersection_mean_building_{attr}'].replace(np.nan, 0).astype(float).round(3)
                    std_series = res_intersection.groupby(['intersection_id'])[attr].std()
                    std_series.name = f'intersection_std_building_{attr}' 
                    nodes = nodes.merge(std_series, on='intersection_id', how='left')
                    nodes[f'intersection_std_building_{attr}'] = nodes[f'intersection_std_building_{attr}'].replace(np.nan, 0).astype(float).round(3)

                if edge_attr:
                    building_polygon_centroids = building_polygon.copy()
                    building_polygon_centroids.loc[:,'geometry'] = building_polygon_centroids.geometry.centroid

                    # Assign buildings to nearest edge
                    edge_intersection = gpd.sjoin_nearest(building_polygon_centroids, proj_edges, how='inner', max_distance=50, distance_col = 'Building Distance')

                    # Add footprint sum
                    edge_building_area_sum_series = edge_intersection.groupby(['edge_id'])['bid_area'].sum()
                    edge_building_area_sum_series.name = 'street_total_building_area'
                    edges = edges.merge(edge_building_area_sum_series, on='edge_id', how='left')

                    # Add footprint mean
                    edge_building_area_mean_series = edge_intersection.groupby(['edge_id'])['bid_area'].mean()
                    edge_building_area_mean_series.name = 'street_mean_building_area'
                    edges = edges.merge(edge_building_area_mean_series, on='edge_id', how='left')

                    # Add footprint std
                    edge_building_area_std_series = edge_intersection.groupby(['edge_id'])['bid_area'].std()
                    edge_building_area_std_series.name = 'street_std_building_area'
                    edges = edges.merge(edge_building_area_std_series, on='edge_id', how='left')

                    # Add length sum
                    edge_building_length_sum_series = edge_intersection.groupby(['edge_id'])['bid_perimeter'].sum()
                    edge_building_length_sum_series.name = 'street_total_building_perimeter'
                    edges = edges.merge(edge_building_length_sum_series, on='edge_id', how='left')

                    # Add length mean
                    edge_building_length_mean_series = edge_intersection.groupby(['edge_id'])['bid_perimeter'].mean()
                    edge_building_length_mean_series.name = 'street_mean_building_perimeter'
                    edges = edges.merge(edge_building_length_mean_series, on='edge_id', how='left')

                    # Add length std
                    edge_building_length_std_series = edge_intersection.groupby(['edge_id'])['bid_perimeter'].std()
                    edge_building_length_std_series.name = 'street_std_building_perimeter'
                    edges = edges.merge(edge_building_length_std_series, on='edge_id', how='left')

                    # Add buildings counts
                    edge_building_count_series = edge_intersection.groupby(['edge_id'])['bid'].count()
                    edge_building_count_series.name = 'street_num_buildings'
                    edges = edges.merge(edge_building_count_series, on='edge_id', how='left')

                    # Add building attributes to node dataframe
                    edges['street_total_building_area'] = edges['street_total_building_area'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_mean_building_area'] = edges['street_mean_building_area'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_std_building_area'] = edges['street_std_building_area'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_total_building_perimeter'] = edges['street_total_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_mean_building_perimeter'] = edges['street_mean_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_std_building_perimeter'] = edges['street_std_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
                    edges['street_num_buildings'] = edges['street_num_buildings'].replace(np.nan, 0).astype(int)

                    # Additional building attributes
                    building_attrs = ['bid_complexity', 'bid_circ_compact', 'bid_convexity', 'bid_corners', 'bid_elongation',
                                    'bid_orientation', 'bid_perimeter', 'bid_longest_axis_length', 'bid_eri', 'bid_fractaldim',
                                    'bid_rectangularity', 'bid_square_compactness', 'bid_shape_idx', 'bid_squareness']
                    
                    for attr in building_attrs:
                        mean_series = edge_intersection.groupby(['edge_id'])[attr].mean()
                        mean_series.name = f'street_mean_building_{attr}' 
                        edges = edges.merge(mean_series, on='edge_id', how='left')
                        edges[f'street_mean_building_{attr}'] = edges[f'street_mean_building_{attr}'].replace(np.nan, 0).astype(float).round(3)
                        std_series = edge_intersection.groupby(['edge_id'])[attr].std()
                        std_series.name = f'street_std_building_{attr}' 
                        edges = edges.merge(std_series, on='edge_id', how='left')
                        edges[f'street_std_building_{attr}'] = edges[f'street_std_building_{attr}'].replace(np.nan, 0).astype(float).round(3)

                print(f'Building attributes computed. Time taken: {round(time.time() - start)}.')

            # If pop_attr is True, compute and add population attributes.
            if pop_attr:
                if self.population:
                    print('Population data found, skipping re-computation.')
                    pop_list = self.population
                    target_cols = self.target_cols

                else:
                    tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                    with open(tile_countries_path, 'r') as f:
                        tile_dict = json.load(f)
                        
                    tiled_country = [country[:-13] for country in list(tile_dict.keys())]
                    
                    # Use csv for small countries
                    if self.country not in tiled_country:
                        print('Using non-tiled population data.')
                        pop_list, target_cols = get_meta_population_data(self.country, 
                                                                    bounding_poly=self.polygon_bounds)
                        
                    # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
                    elif self.country in tiled_country:
                        print('Using tiled population data.')
                        pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = self.polygon_bounds)
                    
                    self.population = pop_list
                    self.target_cols = target_cols
                
                groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

                for i, data in enumerate(zip(pop_list, target_cols)):
                    proj_data = data[0].to_crs(nodes_buffer.crs)
                    res_intersection = proj_data.overlay(nodes_buffer, how='intersection')
                    pop_total_series = res_intersection.groupby(['intersection_id'])[data[1]].sum()
                    pop_total_series.name = groups[i]
                    nodes = nodes.merge(pop_total_series, on='intersection_id', how='left')

                    # Add edge attributes
                    if edge_attr:
                        edge_intersection = gpd.sjoin_nearest(proj_data, proj_edges, how='inner', max_distance=50, distance_col = 'Pop Distance')
                        edge_pop_count_series = edge_intersection.groupby(['edge_id'])[data[1]].sum()
                        edge_pop_count_series.name = groups[i]
                        edges = edges.merge(edge_pop_count_series, on='edge_id', how='left')
                        
                for name in groups:
                    nodes[name] = nodes[name].replace(np.nan, 0).astype(int)
                    if edge_attr:
                            edges[name] = edges[name].replace(np.nan, 0).astype(int)
                    
                print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   

            # If poi_attr is True, compute and add poi attributes.
            if poi_attr:
                if self.pois is None: 
                    if pois_data=='osm':
                        # Load poi information 
                        poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                        with open(poi_path) as poi_filter:
                            poi_filter = json.load(poi_filter)
                        
                        # Get osm pois based on custom filter
                        pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                        pois = pois.replace(np.nan, '')

                        # Fill empty columns
                        cols = ['amenity', 'shop', 'tourism', 'leisure']

                        for i in cols:
                            if i not in set(pois.columns):
                                pois[i] = 0
                            elif i in set(pois.columns):
                                pois[i] = pois[i].replace(np.nan, '')

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
                        
                        pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                        # Remove amenities that have counts of less than n=5
                        pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                        pois = project_gdf(pois)

                        pois['geometry'] = pois.geometry.centroid

                        # Set computed points of interest as map attribute
                        self.pois = pois

                    # if pois_data == 'overture':
                    #     # Load poi information 
                    #     poi_path = pkg_resources.resource_filename('urbanity', "overture_data/overture_places.json")

                    #     with open(poi_path) as poi_filter:
                    #         overture_pois = json.load(poi_filter)

                    #     pois = gpd.read_file(overture_pois[f'{location.replace(" ", "").lower()}_pois.geojson'])

                    #     # Preprocess pois, remove None, add lat and lon, subset columns
                    #     pois = pois[~pois['amenities'].isna()]
                    #     pois['lon'] = pois['geometry'].x
                    #     pois['lat'] = pois['geometry'].y
                    #     pois = pois[['id', 'confidence', 'amenities', 'lon', 'lat', 'geometry']]

                    #     # Load poi relabeller
                    #     poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                    #     with open(poi_path) as poi_filter:
                    #         poi_filter = json.load(poi_filter)

                    #     # Relabel pois
                    #     pois = finetune_poi(pois, 'amenities', poi_filter['overture_replace_dict'], pois_data = pois_data)

                    #     # Project pois
                    #     pois = project_gdf(pois)

                    #     # Set computed points of interest as map attribute
                    #     self.pois = pois
                        
                else: 
                    print('Points of interest found, skipping re-computation.')
                    
                    # Assign precomputed pois to current instance
                    pois = self.pois

                
                if pois_data == 'osm':
                    # Get intersection of amenities with node buffer
                    res_intersection = pois.overlay(nodes_buffer, how='intersection')
                    poi_series = res_intersection.groupby(['intersection_id'])['poi_col'].value_counts()
                    pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                    pois_df = pd.pivot(pois_df, index='intersection_id', columns='poi_col', values=0).fillna(0)

                    col_order = list(nodes.columns)
                    cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                    col_order = col_order + cols

                    # Add poi attributes to dataframe of nodes
                    nodes = nodes.merge(pois_df, on='intersection_id', how='left')

                    for i in cols:
                        if i not in set(nodes.columns):
                            nodes[i] = 0
                        elif i in set(nodes.columns):
                            nodes[i] = nodes[i].replace(np.nan, 0)
                        
                    nodes = nodes[col_order]
                    nodes = nodes.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

                    if edge_attr:
                        
                        # Assign pois to nearest edge
                        edge_intersection = gpd.sjoin_nearest(pois, proj_edges, how='inner', max_distance=50, distance_col = 'POI Distance')
                        edge_poi_series = edge_intersection.groupby(['edge_id'])['poi_col'].value_counts()
                        edge_pois_df = pd.DataFrame(index = edge_poi_series.index, data = edge_poi_series.values).reset_index()
                        edge_pois_df = pd.pivot(edge_pois_df, index='edge_id', columns='poi_col', values=0).fillna(0)
                        
                        col_order = list(edges.columns)
                        cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                        col_order = col_order + cols

                        # Add poi attributes to dataframe of nodes
                        edges = edges.merge(edge_pois_df, on='edge_id', how='left')

                        for i in cols:
                            if i not in set(edges.columns):
                                edges[i] = 0
                            elif i in set(edges.columns):
                                edges[i] = edges[i].replace(np.nan, 0)

                        edges = edges[col_order]
                        edges = edges.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            
                # if pois_data == 'overture':
                #     # Get intersection of amenities with node buffer
                #     res_intersection = pois.overlay(nodes, how='intersection')
                #     poi_series = res_intersection.groupby(['intersection_id'])['amenities'].value_counts()
                #     pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                #     pois_df = pd.pivot(pois_df, index='intersection_id', columns='amenities', values=0).fillna(0)

                #     col_order = list(nodes.columns)
                #     cols = list(['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social'])
                #     col_order = col_order + cols  

                #     # Add poi attributes to dataframe of nodes
                #     nodes = nodes.merge(pois_df, on='intersection_id', how='left')

                #     for i in cols:
                #         if i not in set(nodes.columns):
                #             nodes[i] = 0
                #         elif i in set(nodes.columns):
                #             nodes[i] = nodes[i].replace(np.nan, 0)
                        
                #     nodes = nodes[col_order]

                #     if edge_attr:
                        
                #         # Assign pois to nearest edge
                #         edge_intersection = gpd.sjoin_nearest(pois, proj_edges, how='inner', max_distance=50, distance_col = 'POI Distance')
                #         edge_poi_series = edge_intersection.groupby(['edge_id'])['amenities'].value_counts()
                #         edge_pois_df = pd.DataFrame(index = edge_poi_series.index, data = edge_poi_series.values).reset_index()
                #         edge_pois_df = pd.pivot(edge_pois_df, index='edge_id', columns='amenities', values=0).fillna(0)
                        
                #         col_order = list(edges.columns)
                #         cols = list(['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social'])
                #         col_order = col_order + cols

                #         # Add poi attributes to dataframe of nodes
                #         edges = edges.merge(edge_pois_df, on='edge_id', how='left')

                #         for i in cols:
                #             if i not in set(edges.columns):
                #                 edges[i] = 0
                #             elif i in set(edges.columns):
                #                 edges[i] = edges[i].replace(np.nan, 0)

                #         edges = edges[col_order]
                    
                print(f'Points of interest computed. Time taken: {round(time.time() - start)}.')

            # If svi_attr is True, compute and add svi attributes.
            if svi_attr:
                if self.svi is None:
                    svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
                    with open(svi_path, 'r') as f:
                        svi_dict = json.load(f)
                    svi_location = location.replace(' ', '')
                    svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
                    svi_data = project_gdf(svi_data)

                    # Assign computed svi data as map attribute
                    self.svi = svi_data
                
                else:
                    print('SVI data found, skipping re-computation.')
                    svi_data = self.svi

                # Associate each node with respective tile_id and create mapping dictionary
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)

                tile_gdf = get_tile_geometry(buffered_tp)
                tile_gdf = tile_gdf.set_crs(self.polygon_bounds.crs)
                proj_tile_gdf = project_gdf(tile_gdf)

                tile_id_with_nodes = gpd.sjoin(proj_nodes, proj_tile_gdf)
                node_and_tile = {}
                for k,v in zip(tile_id_with_nodes['tile_id'], tile_id_with_nodes['intersection_id']):
                    node_and_tile[v] = k

                # Spatial intersection of SVI points and node buffers
                res_intersection = svi_data.overlay(nodes, how='intersection')
                
                # Compute SVI indices
                indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
                for indicator in indicators:
                    svi_mean_series = res_intersection.groupby(['intersection_id'])[indicator].mean()
                    svi_mean_series.name = f'intersection_mean_{indicator.replace(" ", "")}'
                    nodes = nodes.merge(svi_mean_series, on='intersection_id', how='left')
                    svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                    nodes[f'intersection_mean_{indicator.replace(" ", "")}'] = nodes.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, node_and_tile, row[f'intersection_mean_{indicator.replace(" ", "")}'],row.intersection_id), axis=1)


                    svi_std_series = res_intersection.groupby(['intersection_id'])[indicator].std()
                    svi_std_series.name = f'intersection_std_{indicator.replace(" ", "")}'
                    nodes = nodes.merge(svi_std_series, on='intersection_id', how='left')
                    svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
                    nodes[f'intersection_std_{indicator.replace(" ", "")}'] = nodes.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, node_and_tile, row[f'intersection_std_{indicator.replace(" ", "")}'],row.intersection_id), axis=1)

                if edge_attr:
                    tile_id_with_edges = gpd.sjoin(proj_edges, proj_tile_gdf)
                    edge_and_tile = {}
                    for k,v in zip(tile_id_with_edges['tile_id'], tile_id_with_edges['edge_id']):
                        edge_and_tile[v] = k
                        
                    # Join SVI points to network edges
                    edge_intersection = gpd.sjoin_nearest(svi_data, proj_edges, how='inner', max_distance=50, distance_col = 'SVI Distance')
                
                    # Add SVI counts
                    edge_svi_count_series = edge_intersection.groupby(['edge_id'])['SVI Distance'].count()
                    edge_svi_count_series.name = 'street_num_images'
                    edges = edges.merge(edge_svi_count_series, on='edge_id', how='left')
                    edges['street_num_images'] = edges['street_num_images'].replace(np.nan, 0)

                    for indicator in indicators:
                        edge_mean_series = edge_intersection.groupby(['edge_id'])[indicator].mean()
                        edge_mean_series.name = f'street_mean_{indicator.replace(" ", "")}'
                        edges = edges.merge(edge_mean_series, on='edge_id', how='left')
                        svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                        edges[f'street_mean_{indicator.replace(" ", "")}'] = edges.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, edge_and_tile, row[f'street_mean_{indicator.replace(" ", "")}'],row.edge_id), axis=1)

                        edge_std_series = edge_intersection.groupby(['edge_id'])[indicator].std()
                        edge_std_series.name = f'street_std_{indicator.replace(" ", "")}'
                        edges = edges.merge(edge_std_series, on='edge_id', how='left')
                        svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
                        edges[f'street_std_{indicator.replace(" ", "")}'] = edges.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, edge_and_tile, row[f'street_std_{indicator.replace(" ", "")}'],row.edge_id), axis=1)


                print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
            # Add computed indices to nodes dataframe

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

            print("--- %s seconds ---" % round(time.time() - start,3))
            return L, L_nodes, L_edges
        
    def get_point_context(
            self, 
            location: str,
            points: gpd.GeoDataFrame,
            filepath: str = '',
            network_type: str = 'driving',
            bandwidth: int = 100,
            graph_attr: bool = True,
            building_attr: bool = True,
            pop_attr: bool = True,
            poi_attr: bool = True,
            svi_attr: bool = False,
            pois_data: str = 'osm',
            building_data: str = 'osm') -> gpd.GeoDataFrame:
        """Function to augment a GeoDataFrame of points with urban contextual information.
        Bandwidth (m) controls Euclidean catchment radius to obtain contextual information.
        *_attr arguments can be toggled on or off to allow computation of additional geographic information into networks.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            points (gpd.GeoDataFrame): A geopandas dataframe with Point geometry.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            bandwidth (int): Distance to extract information beyond network. Defaults to 100.
            graph_attr (bool): Specifies whether graph attributes should be included. Defaults to True.
            building_attr (bool): Specifies whether building morphology attributes should be included. Defaults to True.
            pop_attr (bool): Specifies whether population attributes should be included. Defaults to True.
            poi_attr (bool): Specifies whether points of interest attributes should be included. Defaults to True.
            svi_attr (bool): Specifies whether street view imagery attributes should be included. Defaults to False. 
            pois_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) points of interest. Defaults to 'osm'.
            building_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) building footprints. Defaults to 'osm'.
        Returns:
            gpd.GeoDataFrame: A geopandas dataframe of Point geometry with augmented geospatial contextual indicators.
        """            

        start = time.time()

        if filepath == '':
            try:
                fp = get_data(location, directory = self.directory)
                print('Creating data folder and downloading osm street data...')
            except ValueError:
                fp = get_data(self.country, directory = self.directory)
                print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
            except ValueError:
                raise ValueError('No osm data found for specified location.')

            print('Data extracted successfully. Proceeding to extract contextual attributes for point locations.')
        elif filepath != '':
            fp = filepath
            print('Data found! Proceeding to extract contextual attributes for point locations.')

        # Check if extend is within bounding box of points
        xmin, ymin, xmax, ymax = points.geometry.total_bounds
        geom = box(xmin, ymin, xmax, ymax)
        original_bbox = gpd.GeoDataFrame(data=None, crs = 'epsg:4326', geometry = [geom])

        buffered_tp = original_bbox.copy()
        buffered_tp['geometry'] = buffer_polygon(original_bbox, bandwidth=bandwidth)
        buffered_bbox = buffered_tp.geometry.values[0]

        # Obtain nodes and edges within buffered polygon
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)
        
        points['point_id'] = range(len(points))
        proj_points = project_gdf(points)

        # Buffer around points
        points_buffer = proj_points.copy()
        points_buffer['geometry'] = points_buffer.geometry.buffer(bandwidth)
        
        # If graph_attr is True, compute and add graph attributes
        if graph_attr:
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

            # Fill NA and drop incomplete columns
            nodes = nodes.fillna('')
            edges = edges.fillna('')
            nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
            edges = edges.reset_index()[['u','v','length','geometry']]

            # Assign unique IDs
            nodes['node_id'] = nodes.index
            nodes = nodes[['node_id', 'x', 'y', 'geometry']]
            edges['edge_id'] = edges.index
            edges = edges[['edge_id', 'u', 'v', 'length','geometry']]

            # Project nodes to local crs
            proj_nodes = project_gdf(nodes)
            proj_edges = project_gdf(edges)

            res_intersection = proj_nodes.overlay(points_buffer, how='intersection')

            res_intersection['point_node_count'] = 1
            points["point_num_intersections"] = res_intersection.groupby(['point_id'])['point_node_count'].sum().values
            points["point_intersections_density_km2"] = points['point_num_intersections'] / (math.pi * bandwidth ** 2 / 1000000)
            
            # Add Street Length
            res_intersection = proj_edges.overlay(points_buffer, how='intersection')
            res_intersection['point_street_len'] = res_intersection.geometry.length

            points["point_total_street_length"] = np.round(res_intersection.groupby(['point_id'])['point_street_len'].sum().values, 3)
            points["point_mean_street_length"] = np.round(res_intersection.groupby(['point_id'])['point_street_len'].mean().values, 3)
            points["point_std_street_length"] = np.round(res_intersection.groupby(['point_id'])['point_street_len'].std().values, 3)
            points["point_street_length_density_km2"] = points["point_total_street_length"] / (math.pi * bandwidth ** 2 / 1000000) / 1000
            print(f'Topologic/metric attributes computed. Time taken: {round(time.time() - start)}.')

        # If building_attr is True, compute and add building attributes.
        if building_attr:
            if self.buildings is None:
                if building_data == 'osm':
                    # Get building spatial data and project 
                    buildings = osm.get_buildings()

                    # Process geometry and attributes for Overture buildings
                    building_polygon = preprocess_osm_building_geometry(buildings, minimum_area=30)
                    # building_polygon = preprocess_osm_building_attributes(building_polygon, return_class_height=False)

                    # Obtain unique ids for buildings
                    building_polygon = assign_numerical_id_suffix(building_polygon, 'osm')

                # elif building_data == 'overture':
                #     # Get buildings
                #     buildings = get_overture_buildings(location)
                
                #     # Process geometry and attributes for Overture buildings
                #     building_polygon = preprocess_overture_building_geometry(buildings, minimum_area=30)
                #     # building_polygon = preprocess_overture_building_attributes(building_polygon, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     building_polygon = assign_numerical_id_suffix(building_polygon, 'overture')

                # if building_data == 'combined':
                #     overture_buildings = get_overture_buildings(location)
                #     osm_buildings = osm.get_buildings()

                #     # Process geometry and attributes for Overture buildings
                #     overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
                #     # overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

                #     # Process geometry and attributes for Overture buildings
                #     osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
                #     # osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     overture_attr_uids = assign_numerical_id_suffix(overture_geom, 'overture')
                #     osm_attr_uids = assign_numerical_id_suffix(osm_geom, 'osm')

                #     # Merged building and augment with additional attributes from OSM
                #     building_polygon= merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
                #     building_polygon = extract_attributed_osm_buildings(building_polygon, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)
            
                # Assign unique building id
                if building_data == 'osm':
                    id_col = 'osm_id'
                # elif building_data == 'overture':
                #     id_col = 'overture_id'
                # else:
                #     id_col = 'building_id'

                building_polygon['bid'] = building_polygon[id_col]
                building_polygon['bid_area'] = building_polygon.geometry.area
                building_polygon['bid_perimeter'] = building_polygon.geometry.length
                building_polygon = building_polygon[['bid', 'bid_area', 'bid_perimeter', 'geometry']]

                # Compute building attributes
                building_polygon = compute_circularcompactness(building_polygon, element='bid')
                building_polygon = compute_convexity(building_polygon, element='bid')
                building_polygon = compute_corners(building_polygon, element='bid')
                building_polygon = compute_elongation(building_polygon, element='bid')
                building_polygon = compute_orientation(building_polygon, element='bid')
                # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
                building_polygon = compute_longest_axis_length(building_polygon, element='bid')
                building_polygon = compute_equivalent_rectangular_index(building_polygon, element='bid')
                building_polygon = compute_fractaldim(building_polygon, element='bid')
                building_polygon = compute_rectangularity(building_polygon, element='bid')
                building_polygon = compute_square_compactness(building_polygon, element='bid')
                building_polygon = compute_shape_index(building_polygon, element='bid')
                building_polygon = compute_squareness(building_polygon, element='bid')
                building_polygon = compute_complexity(building_polygon, element='bid')

                # Set computed building data as map attribute
                self.buildings = building_polygon
            else:
                print('Building data found, skipping re-computation.')
                building_polygon = self.buildings

            # Compute and add building attributes
            res_intersection = building_polygon.overlay(points_buffer, how='intersection')
            # building_set = building_polygon.iloc[list(res_intersection['bid'].unique()),:]
            res_intersection['area'] = res_intersection.geometry.area
            area_series = res_intersection.groupby(['point_id'])['area'].sum()
            area_series.name = 'point_total_building_area'
            points = points.merge(area_series, on='point_id', how='left')

            total_area = math.pi*bandwidth**2
            points['point_building_footprint_proportion'] = points['point_total_building_area'] / total_area
            
            # Obtain mean area
            mean_series = res_intersection.groupby(['point_id'])['bid_area'].mean()
            mean_series.name = 'point_mean_building_footprint'
            points = points.merge(mean_series, on='point_id', how='left')

            # Obtain mean area
            std_series = res_intersection.groupby(['point_id'])['bid_area'].std()
            std_series.name = 'point_std_building_footprint'
            points = points.merge(std_series, on='point_id', how='left')

            # Add perimeter
            perimeter_series = res_intersection.groupby(['point_id'])['bid_perimeter'].sum()
            perimeter_series.name = 'point_total_building_perimeter'
            points = points.merge(perimeter_series, on='point_id', how='left')

            perimeter_mean_series = res_intersection.groupby(['point_id'])['bid_perimeter'].mean()
            perimeter_mean_series.name = 'point_mean_building_perimeter'
            points = points.merge(perimeter_mean_series, on='point_id', how='left')

            perimeter_std_series = res_intersection.groupby(['point_id'])['bid_perimeter'].std()
            perimeter_std_series.name = 'point_std_building_perimeter'
            points = points.merge(perimeter_std_series, on='point_id', how='left')

            # Add counts
            counts_series = res_intersection.groupby(['point_id'])['bid'].count()
            counts_series.name = 'point_num_buildings'
            points = points.merge(counts_series, on='point_id', how='left')

            # Add building attributes to point dataframe
            points['point_building_footprint_proportion'] = points['point_building_footprint_proportion'].replace(np.nan, 0).astype(float).round(3)
            points['point_mean_building_footprint'] = points['point_mean_building_footprint'].replace(np.nan, 0).astype(float).round(3)
            points['point_std_building_footprint'] = points['point_std_building_footprint'].replace(np.nan, 0).astype(float).round(3)
            points['point_total_building_perimeter'] = points['point_total_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
            points['point_mean_building_perimeter'] = points['point_mean_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
            points['point_std_building_perimeter'] = points['point_std_building_perimeter'].replace(np.nan, 0).astype(float).round(3)
            points['point_num_buildings'] = points['point_num_buildings'].replace(np.nan, 0).astype(int)

            # Additional building attributes
            building_attrs = ['bid_complexity', 'bid_circ_compact', 'bid_convexity', 'bid_corners', 'bid_elongation',
                              'bid_orientation', 'bid_perimeter', 'bid_longest_axis_length', 'bid_eri', 'bid_fractaldim',
                              'bid_rectangularity', 'bid_square_compactness', 'bid_shape_idx', 'bid_squareness']
            
            for attr in building_attrs:
                mean_series = res_intersection.groupby(['point_id'])[attr].mean()
                mean_series.name = f'point_mean_building_{attr}' 
                points = points.merge(mean_series, on='point_id', how='left')
                points[f'point_mean_building_{attr}'] = points[f'point_mean_building_{attr}'].replace(np.nan, 0).astype(float).round(3)
                std_series = res_intersection.groupby(['point_id'])[attr].std()
                std_series.name = f'point_std_building_{attr}' 
                points = points.merge(std_series, on='point_id', how='left')
                points[f'point_std_building_{attr}'] = points[f'point_std_building_{attr}'].replace(np.nan, 0).astype(float).round(3)

            print(f'Building morphology attributes computed. Time taken: {round(time.time() - start)}.')   
        
            # If pop_attr is True, compute and add population attributes.
        if pop_attr:
            if self.population:
                print('Population data found, skipping re-computation.')
                pop_list = self.population
                target_cols = self.target_cols
            else: 
                tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                with open(tile_countries_path, 'r') as f:
                    tile_dict = json.load(f)
                    
                tiled_country = [country[:-13] for country in list(tile_dict.keys())]
                
                # Use csv for small countries
                if self.country not in tiled_country:
                    print('Using non-tiled population data.')
                    pop_list, target_cols = get_meta_population_data(self.country, 
                                                                bounding_poly=buffered_tp)
                
                elif self.country in tiled_country:
                    print('Using tiled population data.')
                    pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = buffered_tp)
                
                for i, data in enumerate(zip(pop_list, target_cols)):
                    proj_data = data[0].to_crs(points_buffer.crs)
                    self.population.append(proj_data)
                    self.target_cols.append(data[1])

            groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

            for i, data in enumerate(zip(pop_list, target_cols)):
                proj_data = data[0].to_crs(points_buffer.crs)
                res_intersection = proj_data.overlay(points_buffer, how='intersection')
                pop_total_series = res_intersection.groupby(['point_id'])[data[1]].sum()
                pop_total_series.name = groups[i]
                points = points.merge(pop_total_series, on='point_id', how='left')
                    
            for name in groups:
                points[name] = points[name].replace(np.nan, 0).astype(int)

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   

        # If poi_attr is True, compute and add poi attributes.
        if poi_attr:
            if self.pois is None: 
                if pois_data=='osm':
                        # Load poi information 
                    poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                    with open(poi_path) as poi_filter:
                        poi_filter = json.load(poi_filter)
                    
                    # Get osm pois based on custom filter
                    pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                    pois = pois.replace(np.nan, '')

                    cols = ['amenity', 'shop', 'tourism', 'leisure']

                    for i in cols:
                        if i not in set(pois.columns):
                            pois[i] = 0
                        elif i in set(pois.columns):
                            pois[i] = pois[i].replace(np.nan, '')

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
                    
                    pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                    # Remove amenities that have counts of less than n=5
                    pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                    pois = project_gdf(pois)

                    pois['geometry'] = pois.geometry.centroid

                # if pois_data == 'overture':

                #     # Load poi information 
                #     poi_path = pkg_resources.resource_filename('urbanity', "overture_data/overture_places.json")
                #     with open(poi_path) as poi_filter:
                #         overture_pois = json.load(poi_filter)

                #     pois = gpd.read_file(overture_pois[f'{location.replace(" ", "").lower()}_pois.geojson'])

                #     # Preprocess pois, remove None, add lat and lon, subset columns
                #     pois = pois[~pois['amenities'].isna()]
                #     pois['lon'] = pois['geometry'].x
                #     pois['lat'] = pois['geometry'].y
                #     pois = pois[['id', 'confidence', 'amenities', 'lon', 'lat', 'geometry']]

                #     # Load poi relabeller
                #     poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                #     with open(poi_path) as poi_filter:
                #         poi_filter = json.load(poi_filter)

                #     # Relabel pois
                #     pois = finetune_poi(pois, 'amenities', poi_filter['overture_replace_dict'], pois_data = pois_data)

                #     # Project pois
                #     pois = project_gdf(pois)

                # Set computed points of interest as map attribute
                
                self.pois = pois

            else: 
                print('Points of interest data found, skipping re-computation.')
                pois = self.pois


            # if pois_data=='overture':
            #     # Get intersection of amenities with node buffer
            #     res_intersection = pois.overlay(points_buffer, how='intersection')
            #     poi_series = res_intersection.groupby(['point_id'])['amenities'].value_counts()
            #     pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
            #     pois_df = pd.pivot(pois_df, index='point_id', columns='amenities', values=0).fillna(0)

            #     col_order = list(points.columns)
            #     cols = list(['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social'])
            #     col_order = col_order + cols  

            #     # Add poi attributes to dataframe of nodes
            #     points = points.merge(pois_df, on='point_id', how='left')

            #     for i in cols:
            #         if i not in set(points.columns):
            #             points[i] = 0
            #         elif i in set(points.columns):
            #             points[i] = points[i].replace(np.nan, 0)

            #     points = points[col_order]

            if pois_data=='osm':
                # Get intersection of amenities with node buffer
                res_intersection = pois.overlay(points_buffer, how='intersection')
                poi_series = res_intersection.groupby(['point_id'])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='point_id', columns='poi_col', values=0).fillna(0)

                col_order = list(points.columns)
                cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                col_order = col_order + cols

                # Add poi attributes to dataframe of nodes
                points = points.merge(pois_df, on='point_id', how='left')

                for i in cols:
                    if i not in set(points.columns):
                        points[i] = 0
                    elif i in set(points.columns):
                        points[i] = points[i].replace(np.nan, 0)
                    
                points = points[col_order]
                points = points.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            print(f'Points of interest computed. Time taken: {round(time.time() - start)}.')
    
        # If svi_attr is True, compute and add svi attributes.
        if svi_attr:
            if self.svi is None:
                svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
                with open(svi_path, 'r') as f:
                    svi_dict = json.load(f)
                svi_location = location.replace(' ', '')
                svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
                svi_data = project_gdf(svi_data)

                # Assign SVI to map attribute
                self.svi = svi_data
            else:
                print('SVI data found, skipping re-computation.')
                svi_data = self.svi

            # Associate each node with respective tile_id and create mapping dictionary
            tile_gdf = get_tile_geometry(buffered_tp)
            tile_gdf = tile_gdf.set_crs(buffered_tp.crs)
            proj_tile_gdf = project_gdf(tile_gdf)
            
            tile_id_with_nodes = gpd.sjoin(points_buffer, proj_tile_gdf)
            node_and_tile = {}
            for k,v in zip(tile_id_with_nodes['tile_id'], tile_id_with_nodes['point_id']):
                node_and_tile[v] = k

            # Spatial intersection of SVI points and node buffers
            res_intersection = svi_data.overlay(points_buffer, how='intersection')

            # Compute SVI indices
            indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
            for indicator in indicators:
                svi_mean_series = res_intersection.groupby(['point_id'])[indicator].mean()
                svi_mean_series.name = f'point_mean_{indicator.replace(" ", "_")}'
                points = points.merge(svi_mean_series, on='point_id', how='left')
                svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
                points[f'point_mean_{indicator.replace(" ", "_")}'] = points.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, node_and_tile, row[f'point_mean_{indicator.replace(" ", "_")}'],row['point_id']), axis=1)

                svi_std_series = res_intersection.groupby(['point_id'])[indicator].std()
                svi_std_series.name = f'point_std_{indicator.replace(" ", "_")}'
                points = points.merge(svi_std_series, on='point_id', how='left')
                svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
                points[f'point_std_{indicator.replace(" ", "_")}'] = points.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, node_and_tile, row[f'point_std_{indicator.replace(" ", "_")}'],row['point_id']), axis=1)

            print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
        # Add computed indices to nodes dataframe

        print("Total elapsed time --- %s seconds ---" % round(time.time() - start))
        
        return points

    def get_aggregate_stats(
        self,
        location: str,
        column: str = '',
        filepath: str = '',
        bandwidth: int = 0,
        network_type: str = 'driving',
        get_graph: bool = True,
        get_building: bool = True,
        get_pop: bool = True,
        get_pois: bool = True,
        get_svi: bool = False,
        pois_data: str = 'osm',
        building_data: str = 'osm') -> pd.DataFrame:
        """Obtains descriptive statistics for bounding polygon without constructing network. Users can specify bounding polygon either by drawing on the map object, or uploading a geojson/shapefile.
        If geojson/shape file contains multiple geometric objects, descriptive statistics will be returned for all entities. Results are returned in dictionary format. 

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            column (str, optional): Id or name column to identify zones. If None, uses shapefile index column. Defaults to ''.
            filepath (str, optional): If location is not available, user can specify path to osm.pbf file. Defaults to ''.
            bandwidth (int, optional): Distance (m) to buffer site boundary. Defaults to 0.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            get_graph (bool, optional): If True, includes aggregate network indicators. Defaults to True.
            get_building (bool, optional): If True, includes aggregate building morphological indicators. Defaults to True.
            get_pop (bool, optional): SIf True, includes aggregate population indicators. Defaults to True.
            get_pois (bool, optional): If True, includes aggregate POI indicators. Defaults to True.
            get_svi(bool, optional): If True, includes aggregated SVI indicators. Defaults to False.
            pois_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) points of interest. Defaults to 'osm'.
            building_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) building footprints. Defaults to 'osm'.
        Returns:
            pd.DataFrame: Pandas dataframe consisting of aggregate values for each subzone.
        """
        start = time.time()
        if filepath == '':
            try:
                fp = get_data(location, directory = self.directory)
                print('Creating data folder and downloading osm street data...')
            except ValueError:
                fp = get_data(self.country, directory = self.directory)
                print(f"KeyError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
            except ValueError:
                raise ValueError('No osm data found for specified location.')

        elif filepath != '':
            fp = filepath
            print('Data found! Proceeding to construct street network.')

        print('Data extracted successfully. Computing aggregates from shapefile.')

        # Project and buffer original polygon to examine nodes outside boundary
        try:
            dissolved_poly = dissolve_poly(self.polygon_bounds, location)
            original_bbox = dissolved_poly.geometry[0]
            buffered_tp = dissolved_poly.copy()
            buffered_tp['geometry'] = buffer_polygon(buffered_tp, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
        # catch when it hasn't even been defined 
        except (AttributeError, NameError):
            raise Exception('Please delimit a bounding box.')

        # Obtain OSM data from bounding box
        osm = pyrosm.OSM(fp, bounding_box=buffered_bbox)

        # Replace and select column of place names and geometry
        if column == '':
            self.polygon_bounds = self.polygon_bounds.reset_index()
            column = 'index'
            self.polygon_bounds[column] = [f'area_{i}' for i in range(len(self.polygon_bounds))]

        # Replace and select column of place names and geometry
        if column == 'name':
            column = 'name_id'
            self.polygon_bounds.rename(columns={'name':column}, inplace=True)

        aggregate_gdf = self.polygon_bounds[[column, 'geometry']]

        # Project zones to local crs
        aggregate_gdf = project_gdf(aggregate_gdf)

        # Add subzone area and perimeter
        aggregate_gdf['subzone_area'] = aggregate_gdf.geometry.area.values / 1000000
        aggregate_gdf['subzone_perimeter'] = aggregate_gdf.geometry.length.values / 1000

        if get_graph:
            if self.network:
                print('Network data found, skipping re-computation')
                G_buff_trunc_loop, nodes, edges = self.network[0], self.network[1], self.network[2]
            else: 
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
                # max_wcc = max(nx.weakly_connected_components(G_buff_trunc), key=len)
                # G_buff_trunc = nx.subgraph(G_buff_trunc, max_wcc)

                # Remove self loops
                G_buff_trunc_loop = G_buff_trunc.copy()
                G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

                # Obtain network nodes and edges from preprocessed graph
                nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)
                nodes = nodes.fillna('')

                # Add network nodes and edges as map attribute
                self.network.append(G_buff_trunc_loop)
                self.network.append(nodes)
                self.network.append(edges)

            # Project network nodes and edges to local crs
            proj_nodes = project_gdf(nodes)
            proj_nodes['node_id'] = range(len(proj_nodes))
            proj_edges = project_gdf(edges)
            proj_edges['edge_id'] = range(len(proj_edges))

            # Compute spatial intersection
            nodes_intersection = proj_nodes.overlay(aggregate_gdf, how='intersection')
            edges_intersection = proj_edges.overlay(aggregate_gdf, how='intersection')

            # Add number of nodes
            merge_series = nodes_intersection.groupby([column])['node_id'].count()
            merge_series.name = 'subzone_num_nodes'
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            # Add number of edges
            merge_series = edges_intersection.groupby([column])['edge_id'].count()
            merge_series.name = 'subzone_num_edges'
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')


            # Add subzone node and edge density 
            aggregate_gdf['subzone_node_density'] = round(aggregate_gdf['subzone_num_nodes'] / aggregate_gdf['subzone_area'], 3)
            aggregate_gdf['subzone_edge_density'] = round(aggregate_gdf['subzone_num_edges'] / aggregate_gdf['subzone_area'], 3)

            # Add subzone total and mean edge length
            merge_series = edges_intersection.groupby([column])['length'].sum()
            merge_series.name = 'subzone_total_edge_length'
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            merge_series = edges_intersection.groupby([column])['length'].mean()
            merge_series.name = 'subzone_mean_edge_length'
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            merge_series = edges_intersection.groupby([column])['length'].std()
            merge_series.name = 'subzone_std_edge_length'
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            aggregate_gdf['subzone_edge_length_density'] = round(aggregate_gdf['subzone_total_edge_length'] / aggregate_gdf['subzone_area'], 3)

            # Add mean node degree
            aggregate_gdf['subzone_mean_node_degree'] = round(2 * aggregate_gdf['subzone_num_edges'] / aggregate_gdf['subzone_num_nodes'], 3)

        # If building_attr is True, compute and add building attributes.
        if get_building:
            if self.buildings is None:
                if building_data == 'osm':
                    # Get building spatial data and project 
                    buildings = osm.get_buildings()

                    # Process geometry and attributes for Overture buildings
                    building_polygon = preprocess_osm_building_geometry(buildings, minimum_area=30)
                    # building_polygon = preprocess_osm_building_attributes(building_polygon, return_class_height=False)

                    # Obtain unique ids for buildings
                    building_polygon = assign_numerical_id_suffix(building_polygon, 'osm')

                # elif building_data == 'overture':
                #     # Get buildings
                #     buildings = get_overture_buildings(location, boundary=dissolved_poly)

                #     # Filter by boundary
                #     outside_buildings = buildings.overlay(dissolved_poly, how='difference')
                #     buildings = buildings[~buildings['id'].isin(list(outside_buildings['id'].unique()))]
                
                #     # Process geometry and attributes for Overture buildings
                #     building_polygon = preprocess_overture_building_geometry(buildings, minimum_area=30)
                #     # building_polygon = preprocess_overture_building_attributes(building_polygon, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     building_polygon = assign_numerical_id_suffix(building_polygon, 'overture')

                # if building_data == 'combined':
                #     overture_buildings = get_overture_buildings(location, boundary=dissolved_poly)
                #     outside_buildings = overture_buildings.overlay(dissolved_poly, how='difference')
                #     overture_buildings = overture_buildings[~overture_buildings['id'].isin(list(outside_buildings['id'].unique()))]
                #     osm_buildings = osm.get_buildings()

                #     # Process geometry and attributes for Overture buildings
                #     overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
                #     # overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

                #     # Process geometry and attributes for Overture buildings
                #     osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
                #     # osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     overture_attr_uids = assign_numerical_id_suffix(overture_geom, 'overture')
                #     osm_attr_uids = assign_numerical_id_suffix(osm_geom, 'osm')

                #     # Merged building and augment with additional attributes from OSM
                #     building_polygon= merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
                #     building_polygon = extract_attributed_osm_buildings(building_polygon, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)
            
                if building_data == 'osm':
                    id_col = 'osm_id'
                # elif building_data == 'overture':
                #     id_col = 'overture_id'
                # else:
                #     id_col = 'building_id'
                building_polygon['bid'] = building_polygon[id_col]
                building_polygon['bid_area'] = building_polygon.geometry.area
                building_polygon['bid_perimeter'] = building_polygon.geometry.length
                building_polygon = building_polygon[['bid', 'bid_area', 'bid_perimeter', 'geometry']]

                # Compute building attributes
                building_polygon = compute_circularcompactness(building_polygon, element='bid')
                building_polygon = compute_convexity(building_polygon, element='bid')
                building_polygon = compute_corners(building_polygon, element='bid')
                building_polygon = compute_elongation(building_polygon, element='bid')
                building_polygon = compute_orientation(building_polygon, element='bid')
                # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
                building_polygon = compute_longest_axis_length(building_polygon, element='bid')
                building_polygon = compute_equivalent_rectangular_index(building_polygon, element='bid')
                building_polygon = compute_fractaldim(building_polygon, element='bid')
                building_polygon = compute_rectangularity(building_polygon, element='bid')
                building_polygon = compute_square_compactness(building_polygon, element='bid')
                building_polygon = compute_shape_index(building_polygon, element='bid')
                building_polygon = compute_squareness(building_polygon, element='bid')
                building_polygon = compute_complexity(building_polygon, element='bid')

                # Add computed building as map attribute
                self.buildings = building_polygon
            else:
                print('Building data found, skipping re-computation.')
                building_polygon = self.buildings

            # Compute building attributes
            building_intersection = building_polygon.overlay(aggregate_gdf, how='intersection')

            # Add total building footprint
            merge_series = building_intersection.groupby([column])['bid_area'].sum() / 1000000
            merge_series.name = 'subzone_total_building_area' 
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            # Add building footprint proportion
            aggregate_gdf['subzone_building_footprint_proportion'] = round(aggregate_gdf['subzone_total_building_area'] / aggregate_gdf['subzone_area'] * 100, 3)

            # Add mean building footprint
            merge_series = building_intersection.groupby([column])['bid_area'].mean()
            merge_series.name = 'subzone_mean_building_area' 
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            # Add stdev building footprint
            merge_series = building_intersection.groupby([column])['bid_area'].std()
            merge_series.name = 'subzone_std_building_area' 
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            # Add total building perimeter
            merge_series = building_intersection.groupby([column])['bid_perimeter'].sum()
            merge_series.name = 'subzone_total_building_perimeter' 
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            # Add total building perimeter
            merge_series = building_intersection.groupby([column])['bid_perimeter'].mean()
            merge_series.name = 'subzone_mean_building_perimeter' 
            aggregate_gdf = aggregate_gdf.merge(merge_series, on=column, how='left')

            building_attrs = ['bid_complexity', 'bid_circ_compact', 'bid_convexity', 'bid_corners', 'bid_elongation',
                              'bid_orientation', 'bid_perimeter', 'bid_longest_axis_length', 'bid_eri', 'bid_fractaldim',
                              'bid_rectangularity', 'bid_square_compactness', 'bid_shape_idx', 'bid_squareness']
            
            for attr in building_attrs:
                mean_series = building_intersection.groupby([column])[attr].mean()
                mean_series.name = f'subzone_mean_building_{attr}' 
                aggregate_gdf = aggregate_gdf.merge(mean_series, on=column, how='left')
                std_series = building_intersection.groupby([column])[attr].std()
                std_series.name = f'subzone_std_building_{attr}' 
                aggregate_gdf = aggregate_gdf.merge(std_series, on=column, how='left')
    
        # Load OSM urban points of interest data
        if get_pois:
            if self.pois is None: 
                if pois_data=='osm':
                    # Load poi information 
                    poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                    with open(poi_path) as poi_filter:
                        poi_filter = json.load(poi_filter)
                    
                    # Get osm pois based on custom filter
                    pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                    pois = pois.replace(np.nan, '')

                    cols = ['amenity', 'shop', 'tourism', 'leisure']

                    for i in cols:
                        if i not in set(pois.columns):
                            pois[i] = 0
                        elif i in set(pois.columns):
                            pois[i] = pois[i].replace(np.nan, '')

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
                    
                    pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                    # Remove amenities that have counts of less than n=5
                    pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                    pois = project_gdf(pois)

                    pois['geometry'] = pois.geometry.centroid
                    # Set computed points of interest as map attribute

                # if pois_data == 'overture':
                #     # Load poi information 
                #     poi_path = pkg_resources.resource_filename('urbanity', "overture_data/overture_places.json")
                #     with open(poi_path) as poi_filter:
                #         overture_pois = json.load(poi_filter)

                #     pois = gpd.read_file(overture_pois[f'{location.replace(" ", "").lower()}_pois.geojson'])

                #     # Preprocess pois, remove None, add lat and lon, subset columns
                #     pois = pois[~pois['amenities'].isna()]
                #     pois['lon'] = pois['geometry'].x
                #     pois['lat'] = pois['geometry'].y
                #     pois = pois[['id', 'confidence', 'amenities', 'lon', 'lat', 'geometry']]

                #     # Load poi relabeller
                #     poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                #     with open(poi_path) as poi_filter:
                #         poi_filter = json.load(poi_filter)

                #     # Relabel pois
                #     pois = finetune_poi(pois, 'amenities', poi_filter['overture_replace_dict'], pois_data = pois_data)

                #     # Project pois
                #     pois = project_gdf(pois)

                # Set computed points of interest as map attribute
                self.pois = pois
            else:
                print('Points of interest data found, skipping re-computation.')
                pois = self.pois

            if pois_data=='osm':
            # Get intersection of amenities with node buffer
                res_intersection = pois.overlay(aggregate_gdf, how='intersection')
                poi_series = res_intersection.groupby([column])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index=column, columns='poi_col', values=0).fillna(0)

                col_order = list(aggregate_gdf.columns)
                cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                col_order = col_order + cols

                # Add poi attributes to dataframe of nodes
                aggregate_gdf = aggregate_gdf.merge(pois_df, on=column, how='left')

                for i in cols:
                    if i not in set(aggregate_gdf.columns):
                        aggregate_gdf[i] = 0
                    elif i in set(aggregate_gdf.columns):
                        aggregate_gdf[i] = aggregate_gdf[i].replace(np.nan, 0)
                    
                aggregate_gdf = aggregate_gdf[col_order]
                aggregate_gdf = aggregate_gdf.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            # if pois_data=='overture':
            #     # Get intersection of amenities with node buffer
            #     res_intersection = pois.overlay(aggregate_gdf, how='intersection')
            #     poi_series = res_intersection.groupby([column])['amenities'].value_counts()
            #     pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
            #     pois_df = pd.pivot(pois_df, index=column, columns='amenities', values=0).fillna(0)

            #     col_order = list(aggregate_gdf.columns)
            #     cols = list(['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social'])
            #     col_order = col_order + cols  

            #     # Add poi attributes to dataframe of nodes
            #     aggregate_gdf = aggregate_gdf.merge(pois_df, on=column, how='left')

            #     for i in cols:
            #         if i not in set(aggregate_gdf.columns):
            #             aggregate_gdf[i] = 0
            #         elif i in set(aggregate_gdf.columns):
            #             aggregate_gdf[i] = aggregate_gdf[i].replace(np.nan, 0)
                    
            #     aggregate_gdf = aggregate_gdf[col_order]

        # Population
        if get_pop:
            if self.population:
                print('Population data found, skipping recomputation.')
                pop_list = self.population
                target_cols = self.target_cols
            else:
                # Load list of countries with tiled population data
                tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                with open(tile_countries_path, 'r') as f:
                    tile_dict = json.load(f)
                
                tiled_country = [country[:-13] for country in list(tile_dict.keys())]

                # Use csv for small countries
                if self.country not in tiled_country:
                    print('Using non-tiled population data.')
                    pop_list, target_cols = get_meta_population_data(self.country, 
                                                                bounding_poly=buffered_tp)
                elif self.country in tiled_country:
                    print('Using tiled population data.')
                    pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = buffered_tp) 
               
                for i, data in enumerate(zip(pop_list, target_cols)):
                    proj_data = data[0].to_crs(aggregate_gdf.crs)
                    self.population.append(proj_data)
                    self.target_cols.append(data[1])
                
           
            groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

            for i, data in enumerate(zip(pop_list, target_cols)):
                proj_data = data[0].to_crs(aggregate_gdf.crs)
                res_intersection = proj_data.overlay(aggregate_gdf, how='intersection')
                pop_total_series = res_intersection.groupby([column])[data[1]].sum()
                pop_total_series.name = groups[i]
                aggregate_gdf = aggregate_gdf.merge(pop_total_series, on=column, how='left')
                
            for name in groups:
                aggregate_gdf[name] = aggregate_gdf[name].replace(np.nan, 0).astype(int)

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   
            
        
        if get_svi:
            if self.svi is None:
                svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
                with open(svi_path, 'r') as f:
                    svi_dict = json.load(f)
                svi_location = location.replace(' ', '')
                svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
                svi_data = project_gdf(svi_data)
                self.svi = svi_data
            else:
                print('SVI data found, skipping re-computation.')
                svi_data = self.svi

            svi_intersection = svi_data.overlay(aggregate_gdf, how = 'intersection')

            # Compute SVI indices
            indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
            for indicator in indicators:
                mean_series = svi_intersection.groupby([column])[indicator].mean()
                mean_series.name = f'subzone_mean_{indicator.replace(" ","_")}' 
                aggregate_gdf = aggregate_gdf.merge(mean_series, on=column, how='left')
                std_series = svi_intersection.groupby([column])[indicator].std()
                std_series.name = f'subzone_std_{indicator.replace(" ","_")}' 
                aggregate_gdf = aggregate_gdf.merge(std_series, on=column, how='left')

        return aggregate_gdf

    def get_building_nodes(
        self, 
        location: str,
        return_neighbours: str = 'knn',
        knn: list = [3, 5, 8, 10],
        knn_threshold: int = 100,
        distance_threshold: list = [100, 200, 300],
        building_data: str = 'osm') -> gpd.GeoDataFrame:
        """Generate a building network nodes with preprocessed attribute information. For each building, neighbourhood is computed based on either K-nearest neighbour or distance threshold. Neighbourhood information is provided as an attribute 
        that consists of the list of neighbours according to building index. 

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            return_neighbours (str, optional): Specifies what type of neighbour relations should be returned. Allows 'knn', 'distance', or 'none'. Defaults to 'knn'.
            knn (list, optional): Specifies the number of nearest neighbours to form edges between nearby buildings. Defaults to [3, 5, 8, 10].
            knn_threshold (int, optional): Specifies the distance to consider nearby building as neighbour. Defaults to 100.
            distance_threshold (list, optional): Specifies the distance threshold to form edges between nearby buildings. Defaults to [100, 200, 500].
            building_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) building footprints. Defaults to 'osm'.

        Returns:
            gpd.GeoDataFrame: Returns geopandas GeoDataFrames consisting of building polygons, representative points, and neighbours.
        """        
        start = time.time()
        
        if self.buildings is None:
            print(f'No building data passed, computed street network from {building_data}...')

            if building_data == 'osm':
                # Get buildings
                building = get_osm_buildings(location=location, boundary=self.polygon_bounds)

                # Process geometry and attributes for Overture buildings
                building = preprocess_osm_building_geometry(building, minimum_area=30)
                # building = preprocess_osm_building_attributes(building, return_class_height=False)

                # Obtain unique ids for buildings
                building = assign_numerical_id_suffix(building, 'osm')
                building['bid'] = building['osm_id']

            # if building_data == 'overture':
            #     # Get buildings
            #     building = get_overture_buildings(location, boundary=self.polygon_bounds)

            #     # Filter by boundary
            #     outside_buildings = building.overlay(self.polygon_bounds, how='difference')
            #     building = building[~building['id'].isin(list(outside_buildings['id'].unique()))]
            
            #     # Process geometry and attributes for Overture buildings
            #     building = preprocess_overture_building_geometry(building, minimum_area=30)
            #     # building = preprocess_overture_building_attributes(building, return_class_height=False)

            #     # Obtain unique ids for buildings
            #     building = assign_numerical_id_suffix(building, 'overture')
            #     building['bid'] = building['overture_id']

            # elif building_data == 'combined':
            #     # Get buildings
            #     overture_buildings = get_overture_buildings(location, boundary=self.polygon_bounds)
            #     outside_buildings = overture_buildings.overlay(self.polygon_bounds, how='difference')
            #     overture_buildings = overture_buildings[~overture_buildings['id'].isin(list(outside_buildings['id'].unique()))]
            #     osm_buildings = get_osm_buildings(location=location, boundary=self.polygon_bounds)

            #     # Process geometry and attributes for Overture buildings
            #     overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
            #     # overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

            #     # Process geometry and attributes for Overture buildings
            #     osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
            #     # osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

            #     # Obtain unique ids for buildings
            #     overture_attr_uids = assign_numerical_id_suffix(overture_geom, 'overture')
            #     osm_attr_uids = assign_numerical_id_suffix(osm_geom, 'osm')

            #     # Merged building and augment with additional attributes from OSM
            #     building = merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
            #     building = extract_attributed_osm_buildings(building, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)
            #     building['bid'] = building['building_id']

            self.buildings = building
        else: 
            print('Building data found, skipping re-computation...')
            building = self.buildings

        # Obtain building network
        building['building_area'] = building.geometry.area
        building['building_centroid'] = building.geometry.centroid
        building['building_perimeter'] = building.geometry.length

        # Reset index before knn computation to get consistent indexing
        building = building.reset_index(drop=True)

        if return_neighbours == 'knn': 

            def filter_threshold(nn, dist):
                return {k:v for k,v in zip(nn, dist) if v <= knn_threshold}
            
            # Compute attributes
            for i in knn:
                building = building_knn_nearest(building, knn=i)
                building[f'{i}-nn-threshold'] = building.apply(lambda row: filter_threshold(row[f'{i}-nn-idx'], row[f'{i}-dist']), axis=1)

        elif return_neighbours == 'none':
            print('Skipping adjacency computation between buildings.')

        # Compute building attributes
        building = compute_circularcompactness(building)
        building = compute_convexity(building)
        building = compute_corners(building)
        building = compute_elongation(building)
        building = compute_orientation(building)
        # building = compute_shared_wall_ratio(building)
        building = compute_longest_axis_length(building)
        building = compute_equivalent_rectangular_index(building)
        building = compute_fractaldim(building)
        building = compute_rectangularity(building)
        building = compute_square_compactness(building)
        building = compute_shape_index(building)
        building = compute_squareness(building)
        building = compute_complexity(building)

        # Compute knn mean and stdev
        attr_cols = ['building_area', 'building_perimeter', 'building_circ_compact', 'building_convexity',
       'building_corners', 'building_elongation', 'building_orientation', 'building_longest_axis_length',
       'building_eri', 'building_fractaldim', 'building_rectangularity', 'building_squareness',
       'building_square_compactness', 'building_shape_idx', 'building_complexity']

        if return_neighbours == 'knn':
            building = compute_knn_aggregate(building, attr_cols)

        elif return_neighbours == 'distance':
            
            for i in distance_threshold:
                buffer_gdf = gpd.GeoDataFrame(data=None, crs=building.crs, geometry = building['building_centroid'])
                buffer_gdf['geometry'] = buffer_gdf.geometry.buffer(i)
                buffer_gdf['buffer_id'] = range(len(buffer_gdf))

                # Spatial intersection of building
                res_intersection = building.overlay(buffer_gdf, how='intersection')

                for attr in attr_cols:
                    mean_series = res_intersection.groupby(['buffer_id'])[attr].mean()
                    std_series = res_intersection.groupby(['buffer_id'])[attr].std()
                    df = pd.DataFrame({f'{i}m_{attr}_mean':mean_series.values, f'{i}m_{attr}_stdev':std_series.values})
                    building = gpd.GeoDataFrame(pd.concat([building,df],axis=1))
                    
        print(f'Buildings constructed. Time taken: {round(time.time() - start)}.')

        return building

    def get_urban_plot_nodes(
        self, 
        location: str,
        filepath: str = '',
        bandwidth: int = 100,
        network_type: str = 'driving',
        pois_data: str = 'osm',
        building_data: str = 'osm',
        return_edges: bool = True,
        return_buildings: bool = True,
        minimum_area: int = 30,
        out_buffer: int = 10) -> gpd.GeoDataFrame:
        """Generate a urban plot nodes with preprocessed attribute information.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            filepath (str): If location is not available, user can specify path to osm.pbf file.
            bandwidth (int): Distance to extract information beyond network. Defaults to 100.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            pois_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) points of interest. Defaults to 'osm'.
            building_data (str, optional): Available options 'osm'. Specifies whether to use OpenStreetMap or Overture (future feature) building footprints. Defaults to 'osm'.
            return_edges (bool, optional): If True, returns the bounding edges and their ids for each plot. Defaults to True.
            return_buildings (bool, optional): If True, returns the the id of buildings within each plot. Defaults to True.
            minimum_area (int, optional): Minimum area in metres square for each plot. Defaults to 30.
            out_buffer (int, optional): Distance threshold to buffer out polygon to capture edges. Defaults to 10.

        Returns:
            gpd.GeoDataFrame: Returns a geopandas GeoDataFrame consisting of Polygons that correspond to urban plots and their feature attributes. 
        """        
        start = time.time()

        # Check if edges is passed, if not, compute network edges 
        if self.network: 
            print('Network found, skipping re-computation.')
            edges = self.network[2]

        else:
            if filepath == '':
                try:
                    fp = get_data(location, directory = self.directory)
                    print('Creating data folder and downloading osm street data...')
                except ValueError:
                    fp = get_data(self.country, directory = self.directory)
                    print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
                except ValueError:
                    raise ValueError('No osm data found for specified location.')

                print('Data extracted successfully. Proceeding to construct street network.')
            elif filepath != '':
                fp = filepath
                print('Data found! Proceeding to construct street network.')

            # Project and buffer original polygon to examine nodes outside boundary
            try:
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
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

            # Remove self loops
            G_buff_trunc_loop = G_buff_trunc.copy()
            G_buff_trunc_loop.remove_edges_from(nx.selfloop_edges(G_buff_trunc_loop))

            nodes, edges = graph_to_gdf(G_buff_trunc_loop, nodes=True, edges=True)

            # Fill NA and assign unique edge ids
            edges = edges.fillna('')
            edges = edges.reset_index()[['u','v','length','geometry']]
            edges['edge_id'] = edges.index
            edges = edges[['edge_id', 'u', 'v', 'length','geometry']]

            print(f'Network constructed. Time taken: {round(time.time() - start)}.')


        proj_edges = project_gdf(edges)
        proj_boundary = project_gdf(self.polygon_bounds)

        # Create expanded buffer (buffer bandwidth)
        proj_boundary_expanded = proj_boundary.copy()
        proj_boundary_expanded['geometry'] = proj_boundary.buffer(out_buffer)

        # Get inner and out edges
        outside_edges_proj = proj_edges.overlay(proj_boundary_expanded, how='difference')
        inside_edges_proj = proj_edges[~proj_edges['edge_id'].isin(list(outside_edges_proj['edge_id']))]

        # Group linestring with edge_ids
        linestrings_with_attributes = [(linestring, edge_id) for linestring, edge_id in zip(inside_edges_proj.geometry, inside_edges_proj['edge_id'])]

        # Polygonize linestrings
        merged_linestrings = unary_union([line for line, _ in linestrings_with_attributes])
        polygons = list(polygonize(merged_linestrings))

        # Create urban plots
        urban_plots = gpd.GeoDataFrame(data={'plot_id': range(len(polygons))}, crs=proj_edges.crs, geometry = polygons)

        # Compute features
        if return_edges: 

            # Associate polygon geometry with edge ids
            polygon_features = []

            for i, polygon in enumerate(polygons):
                attributes = []
                
                for line, attr in linestrings_with_attributes:
                    if polygon.covers(line):
                        attributes.append(attr)
                
                polygon_features.append(attributes)

            # Create geodataframe associating each polygon with their own edge ids, filter by minimum size
            urban_plots['street_id'] = polygon_features
            urban_plots['plot_area'] = urban_plots.geometry.area
            urban_plots['plot_perimeter'] = urban_plots.geometry.length
            urban_plots = urban_plots[urban_plots['plot_area'] > minimum_area]
            urban_plots = urban_plots.reset_index(drop=True)
            urban_plots['plot_id'] = urban_plots.index
        
    
        if return_buildings:

            if self.buildings is None:
                print(f'No building data passed, computing building from {building_data}...')

                if building_data == 'osm':
                    # Get buildings
                    buildings = get_osm_buildings(location=location, boundary=self.polygon_bounds)

                    # Process geometry and attributes for Overture buildings
                    buildings = preprocess_osm_building_geometry(buildings, minimum_area=30)
                    # buildings = preprocess_osm_building_attributes(buildings, return_class_height=False)

                    # Obtain unique ids for buildings
                    buildings = assign_numerical_id_suffix(buildings, 'osm')
                    buildings['centroid'] = buildings.geometry.centroid
                    buildings['bid'] = buildings['osm_id']

                # if building_data == 'overture':
                #     # Get buildings
                #     buildings = get_overture_buildings(location, boundary=self.polygon_bounds)

                #     # Filter by boundary
                #     outside_buildings = buildings.overlay(self.polygon_bounds, how='difference')
                #     buildings = buildings[~buildings['id'].isin(list(outside_buildings['id'].unique()))]
                
                #     # Process geometry and attributes for Overture buildings
                #     buildings = preprocess_overture_building_geometry(buildings, minimum_area=30)
                #     # buildings = preprocess_overture_building_attributes(buildings, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     buildings = assign_numerical_id_suffix(buildings, 'overture')

                #     buildings['centroid'] = buildings.geometry.centroid
                #     buildings['bid'] = buildings['overture_id']

                # elif building_data == 'combined':
                #     # Get buildings
                #     overture_buildings = get_overture_buildings(location, boundary=self.polygon_bounds)
                #     outside_buildings = overture_buildings.overlay(self.polygon_bounds, how='difference')
                #     overture_buildings = overture_buildings[~overture_buildings['id'].isin(list(outside_buildings['id'].unique()))]
                #     osm_buildings = get_osm_buildings(location=location, boundary=self.polygon_bounds)

                #     # Process geometry and attributes for Overture buildings
                #     overture_geom = preprocess_overture_building_geometry(overture_buildings, minimum_area=30)
                #     # overture_attr = preprocess_overture_building_attributes(overture_geom, return_class_height=False)

                #     # Process geometry and attributes for Overture buildings
                #     osm_geom = preprocess_osm_building_geometry(osm_buildings, minimum_area=30)
                #     # osm_attr = preprocess_osm_building_attributes(osm_geom, return_class_height=False)

                #     # Obtain unique ids for buildings
                #     overture_attr_uids = assign_numerical_id_suffix(overture_geom, 'overture')
                #     osm_attr_uids = assign_numerical_id_suffix(osm_geom, 'osm')

                #     # Merged building and augment with additional attributes from OSM
                #     buildings = merge_osm_to_overture_footprints(overture_attr_uids, osm_attr_uids)
                #     buildings = extract_attributed_osm_buildings(buildings, osm_attr_uids, column = 'osm_combined_heights', threshold = 50)

                #     # Obtain building network
                #     buildings['centroid'] = buildings.geometry.centroid
                #     buildings['bid'] = buildings['building_id']

            else:
                print('Building data found, skipping re-computation...')
                buildings = self.buildings

            buildings['b_index'] = buildings.index
            buildings['building_area'] = buildings.geometry.area
            buildings['building_perimeter'] = buildings.geometry.length

            # Add building attributes
            buildings = compute_circularcompactness(buildings)
            buildings = compute_convexity(buildings)
            buildings = compute_corners(buildings)
            buildings = compute_elongation(buildings)
            buildings = compute_orientation(buildings)
            # buildings = compute_shared_wall_ratio(buildings)
            buildings = compute_longest_axis_length(buildings)
            buildings = compute_equivalent_rectangular_index(buildings)
            buildings = compute_fractaldim(buildings)
            buildings = compute_rectangularity(buildings)
            buildings = compute_square_compactness(buildings)
            buildings = compute_shape_index(buildings)
            buildings = compute_squareness(buildings)
            buildings = compute_complexity(buildings)

            # Compute buildings morphology at plot level
            building_urban_plots_intersection = buildings.overlay(urban_plots)
            urban_plots = urban_plots[urban_plots['plot_id'].isin(list(building_urban_plots_intersection['plot_id'].unique()))]
            urban_plots['building_ids'] = building_urban_plots_intersection.groupby('plot_id')[['b_index']].aggregate(lambda x: list(x))
            urban_plots['plot_building_count'] = urban_plots['building_ids'].apply(len)
            
            attr_cols = ['building_area', 'building_perimeter', 'building_circ_compact', 'building_convexity', 'building_corners', 'building_elongation',
                         'building_orientation', 'building_longest_axis_length', 'building_eri', 'building_fractaldim', 
                         'building_rectangularity', 'building_squareness', 'building_square_compactness', 'building_shape_idx', 'building_complexity']
            
            for attr in attr_cols:
                urban_plots[f'plot_{attr}_mean'] = building_urban_plots_intersection.groupby('plot_id')[attr].mean()
                urban_plots[f'plot_{attr}_std'] = building_urban_plots_intersection.groupby('plot_id')[attr].std()
                urban_plots[f'plot_{attr}_std'] = urban_plots[f'plot_{attr}_std'].replace(np.nan, 0).astype(float)

            # Compute plot level statistics
            urban_plots['plot_building_total_area'] = building_urban_plots_intersection.groupby('plot_id')['building_area'].sum()
            urban_plots['plot_building_built_coverage'] = urban_plots['plot_building_total_area'] / urban_plots['plot_area'] 
            
            # Compute building attributes
            urban_plots = compute_circularcompactness(urban_plots, element='plot')
            urban_plots = compute_convexity(urban_plots, element='plot')
            urban_plots = compute_corners(urban_plots, element='plot')
            urban_plots = compute_elongation(urban_plots, element='plot')
            urban_plots = compute_orientation(urban_plots, element='plot')
            urban_plots = compute_longest_axis_length(urban_plots, element='plot')
            urban_plots = compute_equivalent_rectangular_index(urban_plots, element='plot')
            urban_plots = compute_fractaldim(urban_plots, element='plot')
            urban_plots = compute_rectangularity(urban_plots, element='plot')
            urban_plots = compute_square_compactness(urban_plots, element='plot')
            urban_plots = compute_shape_index(urban_plots, element='plot')
            urban_plots = compute_squareness(urban_plots, element='plot')
            urban_plots = compute_complexity(urban_plots, element = 'plot')
            
        # Compute local climate zone
        if os.path.isdir('./data'):
            self.directory = "./data"
        else:
            os.makedirs('./data')
            self.directory = "./data"

        # Download LCZ file
        lcz_path = pkg_resources.resource_filename('urbanity', 'lcz_data/lcz_data.json')

        with open(lcz_path) as file:
            lcz_data = json.load(file)

        # Download Worldpop data for specified `year`
        try: 
            filename = lcz_data[location].split('/')[-1]  # be careful with file names
            file_path = os.path.join(self.directory, filename)

            # Download Raster Data
            if not os.path.exists(file_path):
                print('Raster file not found in data folder, proceeding to download.')
                r = requests.get(lcz_data[location], stream=True)
                if r.ok:
                    print(f"Saved LCZ raster file for {self.country} to: \n", os.path.abspath(file_path))
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 8):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                                os.fsync(f.fileno())
            else: 
                print(f'Raster file in {os.path.abspath(file_path)}, skipping re-download.')

            lcz_gdf = raster2gdf(file_path)
            lcz_intersection = lcz_gdf.overlay(urban_plots)
            mode_series = lcz_intersection.groupby('plot_id')['value'].aggregate(lambda x: most_frequent(list(x)))
            mode_series.name = 'plot_lcz'
            urban_plots = urban_plots.merge(mode_series, on='plot_id', how='left')

        except KeyError:
            print('Skipping LCZ computation as no LCZ classification map found.')
        
        # Load OSM urban points of interest data
        if self.pois is None: 
            if pois_data=='osm':
                # Load poi information 
                poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                with open(poi_path) as poi_filter:
                    poi_filter = json.load(poi_filter)
                
                # Get osm pois based on custom filter
                pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                pois = pois.replace(np.nan, '')

                cols = ['amenity', 'shop', 'tourism', 'leisure']

                for i in cols:
                    if i not in set(pois.columns):
                        pois[i] = 0
                    elif i in set(pois.columns):
                        pois[i] = pois[i].replace(np.nan, '')

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
                
                pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                # Remove amenities that have counts of less than n=5
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                pois = project_gdf(pois)

                pois['geometry'] = pois.geometry.centroid
                # Set computed points of interest as map attribute

            # if pois_data == 'overture':
            #     # Load poi information 
            #     poi_path = pkg_resources.resource_filename('urbanity', "overture_data/overture_places.json")
            #     with open(poi_path) as poi_filter:
            #         overture_pois = json.load(poi_filter)

            #     pois = gpd.read_file(overture_pois[f'{location.replace(" ", "").lower()}_pois.geojson'])

            #     # Preprocess pois, remove None, add lat and lon, subset columns
            #     pois = pois[~pois['amenities'].isna()]
            #     pois['lon'] = pois['geometry'].x
            #     pois['lat'] = pois['geometry'].y
            #     pois = pois[['id', 'confidence', 'amenities', 'lon', 'lat', 'geometry']]

            #     # Load poi relabeller
            #     poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
            #     with open(poi_path) as poi_filter:
            #         poi_filter = json.load(poi_filter)

            #     # Relabel pois
            #     pois = finetune_poi(pois, 'amenities', poi_filter['overture_replace_dict'], pois_data = pois_data)

            #     # Project pois
            #     pois = project_gdf(pois)

            # Set computed points of interest as map attribute
            self.pois = pois
        else:
            print('Points of interest data found, skipping re-computation.')
            pois = self.pois

        if pois_data=='osm':
        # Get intersection of amenities with node buffer
            res_intersection = pois.overlay(urban_plots, how='intersection')
            poi_series = res_intersection.groupby(['plot_id'])['poi_col'].value_counts()
            pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
            pois_df = pd.pivot(pois_df, index='plot_id', columns='poi_col', values=0).fillna(0)

            col_order = list(urban_plots.columns)
            cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
            col_order = col_order + cols

            # Add poi attributes to dataframe of nodes
            urban_plots = urban_plots.merge(pois_df, on='plot_id', how='left')

            for i in cols:
                if i not in set(urban_plots.columns):
                    urban_plots[i] = 0
                elif i in set(urban_plots.columns):
                    urban_plots[i] = urban_plots[i].replace(np.nan, 0)
                
            urban_plots = urban_plots[col_order]
            urban_plots = urban_plots.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

        # if pois_data=='overture':
        #     # Get intersection of amenities with node buffer
        #     res_intersection = pois.overlay(urban_plots, how='intersection')
        #     poi_series = res_intersection.groupby(['plot_id'])['amenities'].value_counts()
        #     pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
        #     pois_df = pd.pivot(pois_df, index='plot_id', columns='amenities', values=0).fillna(0)

        #     col_order = list(urban_plots.columns)
        #     cols = list(['Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social'])
        #     col_order = col_order + cols  

        #     # Add poi attributes to dataframe of nodes
        #     urban_plots = urban_plots.merge(pois_df, on='plot_id', how='left')

        #     for i in cols:
        #         if i not in set(urban_plots.columns):
        #             urban_plots[i] = 0
        #         elif i in set(urban_plots.columns):
        #             urban_plots[i] = urban_plots[i].replace(np.nan, 0)
                
        #     urban_plots = urban_plots[col_order]

        # Population
        if self.population:
            print('Population data found, skipping recomputation.')
            pop_list = self.population
            target_cols = self.target_cols
        else:
            # Load list of countries with tiled population data
            tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
            with open(tile_countries_path, 'r') as f:
                tile_dict = json.load(f)
            
            tiled_country = [country[:-13] for country in list(tile_dict.keys())]

            # Use csv for small countries
            if self.country not in tiled_country:
                print('Using non-tiled population data.')
                pop_list, target_cols = get_meta_population_data(self.country, 
                                                            bounding_poly=buffered_tp)
            elif self.country in tiled_country:
                print('Using tiled population data.')
                pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = buffered_tp) 
        
            for i, data in enumerate(zip(pop_list, target_cols)):
                proj_data = data[0].to_crs(urban_plots.crs)
                self.population.append(proj_data)
                self.target_cols.append(data[1])
    
        groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

        for i, data in enumerate(zip(pop_list, target_cols)):
            proj_data = data[0].to_crs(urban_plots.crs)
            res_intersection = proj_data.overlay(urban_plots, how='intersection')
            pop_total_series = res_intersection.groupby(['plot_id'])[data[1]].sum()
            pop_total_series.name = groups[i]
            urban_plots = urban_plots.merge(pop_total_series, on='plot_id', how='left')
            
        for name in groups:
            urban_plots[name] = urban_plots[name].replace(np.nan, 0).astype(int)

        print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')   
                
        if self.svi is None:
            svi_path = pkg_resources.resource_filename('urbanity', 'svi_data/svi_data.json')
            with open(svi_path, 'r') as f:
                svi_dict = json.load(f)
            svi_location = location.replace(' ', '')
            svi_data = gpd.read_file(svi_dict[f'{svi_location}.geojson'])
            svi_data = project_gdf(svi_data)
            self.svi = svi_data
        else:
            print('SVI data found, skipping re-computation.')
            svi_data = self.svi

        svi_intersection = svi_data.overlay(urban_plots, how = 'intersection')

        # Associate each plot with respective tile_id and create mapping dictionary
        original_bbox = self.polygon_bounds.geometry[0]
        buffered_tp = self.polygon_bounds.copy()
        buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)

        tile_gdf = get_tile_geometry(buffered_tp)
        tile_gdf = tile_gdf.set_crs(self.polygon_bounds.crs)
        proj_tile_gdf = project_gdf(tile_gdf)

        tile_id_with_plots = gpd.sjoin(urban_plots, proj_tile_gdf)
        plot_and_tile = {}
        for k,v in zip(tile_id_with_plots['tile_id'], tile_id_with_plots['plot_id']):
            plot_and_tile[v] = k

        # Compute SVI indices
        indicators = ['Green View', 'Sky View', 'Building View', 'Road View', 'Visual Complexity']
        for indicator in indicators:
            mean_series = svi_intersection.groupby(['plot_id'])[indicator].mean()
            mean_series.name = f'subzone_mean_{indicator.replace(" ","_")}' 
            urban_plots = urban_plots.merge(mean_series, on='plot_id', how='left')
            svi_tile_mean_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].mean())
            urban_plots[f'subzone_mean_{indicator.replace(" ", "_")}'] = urban_plots.apply(lambda row: replace_nan_with_tile(svi_tile_mean_aggregate, plot_and_tile, row[f'subzone_mean_{indicator.replace(" ", "_")}'],row['plot_id']), axis=1)
            
            std_series = svi_intersection.groupby(['plot_id'])[indicator].std()
            std_series.name = f'subzone_std_{indicator.replace(" ","_")}' 
            urban_plots = urban_plots.merge(std_series, on='plot_id', how='left')
            svi_tile_std_aggregate = dict(svi_data.groupby(['tile_id'])[indicator].std())
            urban_plots[f'subzone_std_{indicator.replace(" ", "_")}'] = urban_plots.apply(lambda row: replace_nan_with_tile(svi_tile_std_aggregate, plot_and_tile, row[f'subzone_mean_{indicator.replace(" ", "_")}'],row['plot_id']), axis=1)
        

        print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
            # pois ['plot_num_commercial']... ['plot_amenity_variety/diversity'] shannon simpson, total num...

        # Reproject to global coordinates
        urban_plots = urban_plots.to_crs(4326)

        print(f'Urban plots constructed. Time taken: {round(time.time() - start)}.')

        return urban_plots
    
    def get_urban_graph(
            self, 
            location: str = '',
            network_filepath: str = '', 
            svi_filepath: str = '',
            building_filepath: str = '',
            poi_filepath: str = '',
            canopy_filepath: str = '',
            pop_filepath: str = '',
            bandwidth: int = 100,
            minimum_area: int = 30,
            network_type: str = 'driving',
            save_as_h5: bool = False,
            save_as_npz: bool = False,
            save_filepath: str = '',
            add_satellite_imagery: bool = False,
            add_gee_layers: dict =  {},
            add_lcz: bool = False,
            temporal_years: list = [2025],
            population_layer = 'meta',
            satellite_pixel_padding: int = 128
            ) -> [gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Function to generate heterogeneous urban graph. Returns node types as individual geodataframes: 1) urban plots; 2) buildings; 3) streets; 4) intersections.
        Bandwidth (m) can be specified to buffer network, obtaining neighbouring nodes within buffered area of network.

        Args:
            location (str): Accepts city name or country name to obtain OpenStreetMap data.
            network_filepath (str): Specify path to osm.pbf file.
            svi_filepath (str): Specify path to street view imagery file.
            building_filepath (str): Specify path to overture building footprint file.
            poi_filepath (str): Specify path to overture poi file.
            canopy_filepath (str): Specify path to canopy heights raster file.
            pop_filepath (str): Specify path to population folder. 
            bandwidth (int): Distance to extract information beyond network. Defaults to 100.
            minimum_area (int): Specifies minimum plot area to filter small urban plots (e.g. small polygons formed by road intersection).
            knn (list): Specifies the number of neighbours for each building to form building to building edges.
            knn_threshold (int): Specifies the distance (metres) for a building to be considered a neighbour. To simplify computation, distance is measured between building to building centroid.
            distance_threshold (list): Specifies the set of buffer distances to search for building neighbours.
            network_type (str): Specified OpenStreetMap transportation mode. Defaults to 'driving'.
            add_self_as_super_node (bool): If True, inserts spatial boundary (as specified in Map object `polygon_bounds` property) as a super node that is connected to other polygon elements (e.g. usually urban plot). Useful for tasks that involve prediction of attributes corresponding to spatial boundary. 
            save_as_h5 (bool): If True, applies standard numerical preprocessing which includes scaling numerical columns and one-hot encoding categorical columns. Saves output as binarized h5py format to optimize space.
            save_as_npz (bool): If True, applies standard numerical preprocessing which includes scaling numerical columns and one-hot encoding categorical columns. Saves output as numpy compressed format to optimize space.
            save_filepath (str): Specifies location to save h5py or npz graph object. 
            add_satellite_imagery (bool): If True, initiates satellite downloading pipeline for buildings.
            add_lcz (bool): If True, obtain global 100m local climate zone data for urban plots.
            add_gee_layers (dict): Dictionary accepting key, value pairs of lists of GEE layers and computation method. Example: {'layer_names': ['Canopy height'], 'layers': ['users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1'], 'methods':['mean']}.
            population_layer (str): Specifies which population layer to use. Accepts ['meta' (30 meters; demographic subgroups), 'ghs' (100 meters; total population counts)]
            temporal_years (list): Specify the yearly intervals to obtain GHS_BUILT_S_R2023 land cover and population data. Accepts any year from 1975 to 2030.
            satellite_pixel_padding (int): Specifies the number of pixels to pad sides of each building to capture urban context (image size is 512 by 512).

        Returns:
            gpd.GeoDataFrame: A geopandas GeoDataFrame of map boundary as super node that is connected to each plot within it.
            gpd.GeoDataFrame: A geopandas GeoDataFrame of urban plots (polygons) and contextual spatial features that are located within each plot.
            gpd.GeoDataFrame: A geopandas GeoDataFrame of building footprints (polygons) and morphological features of each building.
            gpd.GeoDataFrame: A geopandas GeoDataFrame of street segments (linestrings) and street view indices.
            gpd.GeoDataFrame: A geopandas GeoDataFrame of street intersections (points) with metric adn topological node properties.
            np.ndarray: Plot to plot adjacency matrix (P X 2) where P refers to the number of links between each plot. A link is established if plots are adjacent to each other.
            np.ndarray: Plot to building adjacency matrix (P_b X 2) where P_b refers to the number of links between buildings and urban plots. A link is established if a building is within a plot.
            np.ndarray: Plot to street adjacency matrix (P_s X 2) where P_s refers to the number of links between streets and urban plots. A link is established if a street forms the edge of a plot.
            np.ndarray: Building to building adjacency matrix (B X 2) where B refers to the number of links between buildings. A link can be established in two different ways: 1) distance threshold between building centroids or 2) k-nearest neighbours.
            np.ndarray: Building to street adjacency matrix (B_s X 2) where B_s refers to the number of links between buildings and streets. A link is established between each building and its nearest adjacent street. 
            np.ndarray: Street to intersection adjacency matrix (S_i X 2) where S_i refers to the number of links between streets and intersections. A link is established if one end of a street segment is connected to an intersection. 
        """ 
        
        # If precomputed available, use precomputed
        self.location = location
        start = time.time()

        if self.network:
            print('Network data found, skipping re-computation')
            G_buff_trunc_loop, nodes, edges = self.network[0], self.network[1], self.network[2]
            original_bbox = self.polygon_bounds.geometry[0]
            buffered_tp = self.polygon_bounds.copy()
            buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
            buffered_bbox = buffered_tp.geometry.values[0]
            './data/'
            osm = pyrosm.OSM('./data/temp.osm.pbf', bounding_box=buffered_bbox)
        else:
            if network_filepath == '':
                try:
                    fp = get_data(self.location, directory = self.directory)
                    print('Creating data folder and downloading osm street data...')
                except ValueError:
                    fp = get_data(self.country, directory = self.directory)
                    print(f"ValueError: No pre-downloaded osm data available for specified city, will instead try for specified country.")
                except ValueError:
                    raise ValueError('No osm data found for specified location.')

                network_filepath = fp
                print('Data extracted successfully. Proceeding to construct street network.')

            # Project and buffer original polygon to examine nodes outside boundary
            try:
                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                buffered_bbox = buffered_tp.geometry.values[0]
            # catch when it hasn't even been defined 
            except (AttributeError, NameError):
                raise Exception('Please delimit a bounding box.')
            
            # Obtain nodes and edges within buffered polygon
            data_root = './data/'
            if not os.path.exists(data_root):
                os.makedirs(data_root)

            poly_path = os.path.join(data_root, 'temp.poly')
            osm_path = os.path.join(data_root, 'temp.osm.pbf')

            if os.path.isfile(poly_path):
                os.remove(poly_path)

            if os.path.isfile(osm_path):
                os.remove(osm_path)
            self.polygon_bounds = self.polygon_bounds[['geometry']]
            self.polygon_bounds = self.polygon_bounds.reset_index()
            self.polygon_bounds.columns = ['boundary_id', 'geometry']

            gdf_to_poly(self.polygon_bounds, poly_path, column='boundary_id')
            cmd = [
                    "osmium", "extract", 
                    "-p", poly_path,
                    network_filepath,
                    "-o", osm_path
                ]

            subprocess.run(cmd, capture_output=False, text=True)

            osm = pyrosm.OSM(osm_path, bounding_box=buffered_bbox)
            
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

            # Fill NA and drop incomplete columns
            nodes = nodes.fillna('')
            edges = edges.fillna('')
            nodes = nodes.drop(columns=['osmid','tags','timestamp','version','changeset']).reset_index()
            edges = edges.reset_index()[['u','v','length','geometry']]
    
            # Assign unique IDs
            nodes['intersection_id'] = nodes.index
            nodes = nodes[['intersection_id','osmid', 'x', 'y', 'geometry']]
            
            edges = edges[['u', 'v', 'length','geometry']]

            edges =  drop_duplicate_lines(edges)

            # Add network attributes
            proj_nodes = project_gdf(nodes)
            proj_edges = edges.to_crs(proj_nodes.crs)
            # Add Degree Centrality, Clustering (Weighted and Unweighted)

            nodes = merge_nx_property(nodes, G_buff_trunc_loop.out_degree, 'intersection_degree', None)
            nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'intersection_clustering', None)
            nodes = merge_nx_attr(G_buff_trunc_loop, nodes, nx.clustering, 'inter_section_weighted_clustering', None, weight='length')

            #  Add Centrality Measures
            nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Closeness, 'Closeness Centrality', None, False, False)
            nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.Betweenness, 'Betweenness Centrality', None, True)
            nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.EigenvectorCentrality, 'Eigenvector Centrality', None)
            nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.KatzCentrality, 'Katz Centrality', None)
            nodes = compute_centrality(G_buff_trunc_loop, nodes, networkit.centrality.PageRank, 'PageRank', None, 0.85, 1e-8, networkit.centrality.SinkHandling.NoSinkHandling, True)

            self.network.append(G_buff_trunc_loop)
            self.network.append(nodes)
            self.network.append(edges)

            print(f'Network constructed. Time taken: {round(time.time() - start)}.')

            if svi_filepath != '':
                # Add SVI to edges

                svi_data = gpd.read_parquet(svi_filepath)

                # Generate edge uids
                svi_data['uv'] = svi_data['u']+svi_data['v']
                edges['uv'] = edges['u'] + edges['v']

                # Proportion
                count_cols = [col for col in svi_data.columns if 'counts' in col]
                score_cols = [col for col in svi_data.columns if 'score' in col]
                pixel_cols = [col for col in svi_data.columns if 'pixels' in col]
                svi_data[pixel_cols] = svi_data[pixel_cols].div(svi_data['Total'], axis=0)

                # Add visual entropy
                svi_data['visual_complexity'] = svi_data[pixel_cols].apply(lambda row: entropy(row), axis=1)

                # Groupby edge
                aggr_svi_score = svi_data.groupby(['uv'])[score_cols].aggregate('mean')
                aggr_svi_pixels = svi_data.groupby(['uv'])[pixel_cols].aggregate('mean')
                aggr_svi_count = svi_data.groupby(['uv'])[count_cols].aggregate('mean')
                aggr_svi_complexity = svi_data.groupby(['uv'])['visual_complexity'].aggregate('mean')

                # Join pixel mean to network edges
                edges = edges.merge(aggr_svi_score, on='uv', how='left')
                edges[score_cols] = edges[score_cols].fillna(edges[score_cols].mean())

                # Join pixel mean to network edges
                edges = edges.merge(aggr_svi_pixels, on='uv', how='left')
                edges[pixel_cols] = edges[pixel_cols].fillna(edges[pixel_cols].mean())

                # Join count mean to network edges
                edges = edges.merge(aggr_svi_count, on='uv', how='left')
                edges[count_cols] = edges[count_cols].fillna(edges[count_cols].mean())

                # Join count mean to network edges
                edges = edges.merge(aggr_svi_complexity, on='uv', how='left')
                edges['visual_complexity'] = edges['visual_complexity'].fillna(edges['visual_complexity'].mean())

                # Add SVI counts
                edge_svi_count_series = svi_data['uv'].value_counts()
                edge_svi_count_series.name = 'Number_of_SVI'
                edges = edges.merge(edge_svi_count_series, on='uv', how='left')
                edges['Number_of_SVI'] = edges['Number_of_SVI'].replace(np.nan, 0)

                print(f'SVI attributes computed. Time taken: {round(time.time() - start)}.')
                
                self.svi = svi_data
            else:
                # Pull SVI data
                # image_gdf = parallel_download_image_in_tiles(self.polygon_bounds, api_key = mapillary_api_key)
                pass                

        # Compute and add building attributes.
        if self.buildings is not None: 
            print('Building data found, skipping re-computation.')
            buildings = self.buildings

        else:
            if building_filepath != '':
                buildings = get_overture_buildings(building_filepath)
                
                # Process geometry and attributes for Overture buildings; locally projects
                buildings = preprocess_osm_building_geometry(buildings, minimum_area=30, prefix='overture')

                # Obtain uniqu e ids for buildings
                buildings = assign_numerical_id_suffix(buildings, prefix='overture')
                buildings = buildings.reset_index(drop=True)

                buildings['bid'] = buildings.index
                buildings['bid_area'] = buildings.geometry.area
                buildings['bid_perimeter'] = buildings.geometry.length
                building_centroids = buildings.geometry.centroid
                buildings['bid_centroid'] = buildings.geometry.centroid
                buildings = buildings[['bid', 'bid_area', 'bid_perimeter', 'bid_centroid', 'geometry']]

                # Compute building attributes
                buildings = compute_circularcompactness(buildings, element='bid')
                buildings = compute_convexity(buildings, element='bid')
                buildings = compute_corners(buildings, element='bid')
                buildings = compute_elongation(buildings, element='bid')
                buildings = compute_orientation(buildings, element='bid')
                # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
                buildings = compute_longest_axis_length(buildings, element='bid')
                buildings = compute_equivalent_rectangular_index(buildings, element='bid')
                buildings = compute_fractaldim(buildings, element='bid')
                buildings = compute_rectangularity(buildings, element='bid')
                buildings = compute_square_compactness(buildings, element='bid')
                buildings = compute_shape_index(buildings, element='bid')
                buildings = compute_squareness(buildings, element='bid')
                buildings = compute_complexity(buildings, element='bid')

                attr_cols = ['bid_area', 'bid_perimeter', 'bid_circ_compact', 'bid_convexity',
                'bid_corners', 'bid_elongation', 'bid_orientation', 'bid_longest_axis_length',
                'bid_eri', 'bid_fractaldim', 'bid_rectangularity', 'bid_squareness',
                'bid_square_compactness', 'bid_shape_idx', 'bid_complexity']

                buildings = buildings.to_crs('EPSG:4326')
                self.buildings = buildings

            else: 
                buildings = osm.get_buildings()

                # Process geometry and attributes for Overture buildings
                buildings = preprocess_osm_building_geometry(buildings, minimum_area=30)
                # building_polygon = preprocess_osm_building_attributes(building_polygon, return_class_height=False)

                # Obtain unique ids for buildings
                buildings = assign_numerical_id_suffix(buildings, 'osm')

                id_col = 'osm_id'

                buildings['bid'] = buildings[id_col]
                buildings['bid_area'] = buildings.geometry.area
                buildings['bid_perimeter'] = buildings.geometry.length
                building_centroids = buildings.geometry.centroid
                buildings['bid_centroid'] = buildings.geometry.centroid
                buildings = buildings[['bid', 'bid_area', 'bid_perimeter', 'bid_centroid', 'geometry']]

                # Compute building attributes
                buildings = compute_circularcompactness(buildings, element='bid')
                buildings = compute_convexity(buildings, element='bid')
                buildings = compute_corners(buildings, element='bid')
                buildings = compute_elongation(buildings, element='bid')
                buildings = compute_orientation(buildings, element='bid')
                # building_polygon = compute_shared_wall_ratio(building_polygon, element='bid')
                buildings = compute_longest_axis_length(buildings, element='bid')
                buildings = compute_equivalent_rectangular_index(buildings, element='bid')
                buildings = compute_fractaldim(buildings, element='bid')
                buildings = compute_rectangularity(buildings, element='bid')
                buildings = compute_square_compactness(buildings, element='bid')
                buildings = compute_shape_index(buildings, element='bid')
                buildings = compute_squareness(buildings, element='bid')
                buildings = compute_complexity(buildings, element='bid')

                # Compute building heights
                ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
                ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)
                
                overlapping_grid = ghs_global_grid.overlay(buffered_tp)

                ghs_building_height_path = pkg_resources.resource_filename('urbanity', 'building_height_data/building_grids.json')
                with open(ghs_building_height_path) as f:
                    building_height_links = json.load(f)

                # If only one tile
                if len(overlapping_grid) == 1:
                    row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                    target_key = f"R{row}C{col}.parquet"

                    buildings = get_and_assign_building_heights(building_height_links[target_key], target_key, buildings)

                elif len(overlapping_grid) > 1: 
                    building_height_list = []
                    for i, row in overlapping_grid.iterrows():
                        row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                        target_key = f"R{row}C{col}.parquet"
                        building_height = get_building_heights(target_key, target_key)
                        building_height_list.append(building_height)
                    combined_df = pd.concat(building_height_list)
                    combined_gdf = gpd.GeoDataFrame(combined_df, crs='epsg:4326', geometry=combined_df['geometry'])

                    buildings = assign_building_heights(combined_gdf, target_key, buildings)
        
                # buildings = buildings.drop(columns=['bid_centroid'])
                buildings['bid'] = buildings['bid'].astype(str)
                self.buildings = buildings

            print(f'Buildings constructed. Time taken: {round(time.time() - start)}.')

            if add_satellite_imagery:

                # Check if satellite data folder exists, else create one
                if os.path.isdir('./satellite_data'):
                    self.satellite_directory = './satellite_data'
                else:
                    os.makedirs('./satellite_data')
                    self.satellite_directory = './satellite_data'

                # Slightly buffer boundary by 200m to capture edge tile cases
                boundary_proj = project_gdf(self.polygon_bounds)
                boundary_proj['geometry'] = boundary_proj.buffer(200)
                boundary_proj = boundary_proj.to_crs('epsg:4326')
                
                bbox = list(boundary_proj.geometry.total_bounds)

                # Obtain tiles gdf
                tiles = get_tiles_from_bbox(bbox)
                tiles_gdf = get_tiles_gdf(tiles, bounds = boundary_proj)
                tiles_gdf_proj = project_gdf(tiles_gdf)

                def get_tile_id(row):
                    return str(row['Z']) + '/' + str(row['X']) + '/' + str(row['Y'])

                # Add tile_id to geodataframe
                tiles_gdf_proj['tile_id'] = tiles_gdf_proj.apply(lambda row: get_tile_id(row), axis=1)

                # Download mapbox from 
                download_satellite_tiles_from_bbox(bbox, MAPBOX_API_TOKEN)

                # Download for individual buildings
                proj_buildings = project_gdf(buildings)
                total_max, half_max = get_max_img_dims(proj_buildings)
                building_chips = get_building_image_chips(proj_buildings, tiles_gdf_proj, add_context=True, pad_npixels=satellite_pixel_padding)

                get_and_combine_tiles(tiles_gdf_proj, building_chips, './satellite_data/', './building_satellite/')

        if self.urban_plots is not None:
            print('Urban plots data found, skipping re-computation.')
            urban_plots = self.urban_plots

        else:
            proj_nodes = project_gdf(self.network[1])
            proj_edges = self.network[2].to_crs(proj_nodes.crs)

            if self.plot_geom is not None:
                print('Urban plots data found, skipping re-computation.')
                urban_plots = self.plot_geom
            else:
                # Obtain urban plots

                original_bbox = self.polygon_bounds.geometry[0]
                buffered_tp = self.polygon_bounds.copy()
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=0)
                proj_boundary = project_gdf(buffered_tp)
                proj_boundary = proj_boundary.to_crs(proj_edges.crs)

                # Create expanded buffer (buffer bandwidth)
                proj_boundary_expanded = proj_boundary.copy()
                proj_boundary_expanded['geometry'] = proj_boundary.buffer(10)
        
                # Get inner and out edges
                outside_edges_proj = proj_edges.overlay(proj_boundary_expanded, how='difference')
                inside_edges_proj = proj_edges[~proj_edges['street_id'].isin(list(outside_edges_proj['street_id']))]

                # Group linestring with street_id
                linestrings_with_attributes = [(linestring, street_id) for linestring, street_id in zip(inside_edges_proj.geometry, inside_edges_proj['street_id'])]
                clipped_lines = gpd.clip(inside_edges_proj, proj_boundary)

                tolerance = 1
                
                inside_edges_proj = inside_edges_proj.copy()
                inside_edges_proj['geometry'] = clipped_lines['geometry'].apply(
                    lambda line: snap(line, proj_boundary.geometry, tolerance)
                )[0]
                
                # Polygonize linestrings
                merged_linestrings = unary_union(pd.concat([inside_edges_proj.geometry, proj_boundary.boundary.geometry]))
                polygons = list(polygonize(merged_linestrings))

                # Create urban plots
                urban_plots = gpd.GeoDataFrame(data={'plot_id': range(len(polygons))}, crs=proj_edges.crs, geometry = polygons)

                # Generate geodataframe of lines and street_id
                urban_lines = gpd.GeoDataFrame(data=[k[1] for k in linestrings_with_attributes], crs = proj_edges.crs, geometry=[k[0] for k in linestrings_with_attributes])
                urban_lines.columns = ['street_id', 'geometry']
                urban_lines['street_id'] = urban_lines['street_id'].astype(int)

                # 1) Spatial join on 'intersects' (or nearest if you must handle near matches)
                joined = gpd.sjoin(urban_lines, urban_plots, how='right', predicate='intersects')

                # 2) Compute intersection for each line‐polygon pair
                urban_plots = urban_plots.rename_geometry("plot_geom")   # so we do not lose it in sjoin
                joined = gpd.sjoin(urban_lines, urban_plots, how='inner', predicate='intersects')

                # If the default geometry in 'joined' is from urban_lines, we can pull the polygon geometry from matched rows:
                joined = joined.merge(urban_plots[['plot_id','plot_geom']], on='plot_id', how='left')

                # Compute the intersection as a new column
                joined['intersection'] = joined.apply(
                    lambda row: row.geometry.intersection(row.plot_geom), 
                    axis=1
                )

                # 3) Keep only intersections whose geometry is a (Multi)LineString with > 0 length
                def is_nonpoint_line(geom):
                    return (
                        geom.geom_type in ['LineString', 'MultiLineString'] 
                        and geom.length > 0
                    )

                joined = joined[joined['intersection'].apply(is_nonpoint_line)]
                joined = joined.groupby('plot_id')['street_id'].unique()

                urban_plots = urban_plots.merge(joined, how='left', on='plot_id')
                urban_plots = urban_plots.rename_geometry("geometry")
                self.plot_geom = urban_plots
            # If you need them grouped by polygon:

            print(f'Urban plots constructed. Time taken: {round(time.time() - start)}.')

            # Create geodataframe associating each polygon with their own edge ids, filter by minimum size
            urban_plots['plot_area'] = urban_plots.geometry.area
            urban_plots['plot_perimeter'] = urban_plots.geometry.length
            urban_plots = urban_plots[urban_plots['plot_area'] > minimum_area]
            urban_plots = urban_plots.loc[urban_plots.geometry.is_valid]
            urban_plots = urban_plots.reset_index(drop=True)
            urban_plots['plot_id'] = urban_plots.index

            bounding_polygon = self.polygon_bounds.to_crs(proj_edges.crs)
            urban_plots = urban_plots.overlay(bounding_polygon)

            urban_plots = urban_plots.to_crs('EPSG:4326')

            if canopy_filepath != '':
                # Add canopy height
                print('Loading canopy height map...')
                canopy_vars = ['canopy_mean', 'canopy_stdev', 'canopy_skewness', 'canopy_kurtosis']

                mosaic, meta = load_npz_as_raster(canopy_filepath)
                self.canopy = mosaic
                self.canopy_meta = meta

                if (mosaic[0].shape[0] >100000) or (mosaic[0].shape[1] >100000):
                    canopy_df = mask_raster_with_gdf_large_raster(urban_plots, mosaic)
                else:
                    canopy_df = mask_raster_with_gdf(urban_plots, mosaic)
                canopy_df = canopy_df[['canopy_mean', 'canopy_stdev', 'canopy_skewness', 'canopy_kurtosis', 'plot_id']]
                urban_plots = urban_plots.merge(canopy_df, on='plot_id', how='left')
                urban_plots[canopy_vars] = urban_plots[canopy_vars].fillna(0)

            print('Adding morphology to plots...')
            # Add building id to plot
            buildings = buildings.reset_index()
            building_urban_plots_intersection = buildings.overlay(urban_plots)
            urban_plots['bid'] = building_urban_plots_intersection.groupby('plot_id')[['index']].aggregate(lambda x: list(x))
            urban_plots['bid'] = urban_plots['bid'].fillna(0)

            # urban_plots['plot_building_count'] = urban_plots['index'].apply(lambda x: len(x) if x !=0 else 0)
            
            # attr_cols = ['bid_area', 'bid_perimeter', 'bid_circ_compact', 'bid_convexity', 'bid_corners', 'bid_elongation',
            #              'bid_orientation', 'bid_longest_axis_length', 'bid_eri', 'bid_fractaldim', 
            #              'bid_rectangularity', 'bid_squareness', 'bid_square_compactness', 'bid_shape_idx', 'bid_complexity']
            
            # for attr in attr_cols:
            #     urban_plots[f'plot_{attr}_mean'] = building_urban_plots_intersection.groupby('plot_id')[attr].mean()
            #     urban_plots[f'plot_{attr}_std'] = building_urban_plots_intersection.groupby('plot_id')[attr].std()
            #     urban_plots[f'plot_{attr}_std'] = urban_plots[f'plot_{attr}_std'].replace(np.nan, 0).astype(float)

            # # Compute plot level statistics
            # urban_plots['plot_bid_total_area'] = building_urban_plots_intersection.groupby('plot_id')['bid_area'].sum()
            # urban_plots['plot_bid_built_coverage'] = urban_plots['plot_bid_total_area'] / urban_plots['plot_area'] 
            
            print('Adding plot morphology to plots...')
            # Compute building attributes
            urban_plots = compute_circularcompactness(urban_plots, element='plot')
            urban_plots = compute_convexity(urban_plots, element='plot')
            urban_plots = compute_corners(urban_plots, element='plot')
            urban_plots = compute_elongation(urban_plots, element='plot')
            urban_plots = compute_orientation(urban_plots, element='plot')
            urban_plots = compute_longest_axis_length(urban_plots, element='plot')
            urban_plots = compute_equivalent_rectangular_index(urban_plots, element='plot')
            urban_plots = compute_fractaldim(urban_plots, element='plot')
            urban_plots = compute_rectangularity(urban_plots, element='plot')
            urban_plots = compute_square_compactness(urban_plots, element='plot')
            urban_plots = compute_shape_index(urban_plots, element='plot')
            urban_plots = compute_squareness(urban_plots, element='plot')
            urban_plots = compute_complexity(urban_plots, element = 'plot')
            urban_plots = urban_plots.fillna(0)

            if poi_filepath != '':
                # Add pois to urban plot
                print('Adding poi categories to plots...')
                poi_columns = ['Cultural Institutions', 'Groceries', 'Parks', 'Religious Organizations', 'Restaurants', 'Schools', 'Services', 'Drugstores', 'Healthcare']
                pois = gpd.read_parquet(poi_filepath)
                self.pois = pois

                poi_intersection = pois.overlay(urban_plots)
                poi_series = poi_intersection.groupby(['plot_id'])['Category'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='plot_id', columns='Category', values=0).fillna(0)

                for i in poi_columns:
                    if i not in set(pois_df.columns):
                        pois_df[i] = 0
                    elif i in set(pois_df.columns):
                        pois_df[i] = pois_df[i].replace(np.nan, 0)
                pois_df = pois_df[poi_columns]

                urban_plots = urban_plots.merge(pois_df, on='plot_id', how='left')
                urban_plots[poi_columns] = urban_plots[poi_columns].fillna(0)
            else:
                print('Adding poi categories to plots...')
                # Load poi information 
                poi_path = pkg_resources.resource_filename('urbanity', "map_data/poi_filter.json")
                with open(poi_path) as poi_filter:
                    poi_filter = json.load(poi_filter)
                
                # Get osm pois based on custom filter
                pois = osm.get_pois(custom_filter = poi_filter['custom_filter'])
                pois = pois.replace(np.nan, '')

                # Fill empty columns
                cols = ['amenity', 'shop', 'tourism', 'leisure']

                for i in cols:
                    if i not in set(pois.columns):
                        pois[i] = 0
                    elif i in set(pois.columns):
                        pois[i] = pois[i].replace(np.nan, '')

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
                
                pois = pois[['id', 'osm_type','lon','lat','poi_col','geometry']]

                # Remove amenities that have counts of less than n=5
                pois = finetune_poi(pois, 'poi_col', poi_filter['replace_dict'], n=5)

                pois['geometry'] = pois.geometry.centroid
                self.pois = pois

                # Get intersection of amenities with node buffer
                res_intersection = pois.overlay(urban_plots, how='intersection')
                poi_series = res_intersection.groupby(['plot_id'])['poi_col'].value_counts()
                pois_df = pd.DataFrame(index = poi_series.index, data = poi_series.values).reset_index()
                pois_df = pd.pivot(pois_df, index='plot_id', columns='poi_col', values=0).fillna(0)

                col_order = list(nodes.columns)
                cols = list(['civic', 'commercial', 'entertainment', 'food', 'healthcare', 'institutional', 'recreational', 'social'])
                col_order = col_order + cols

                # Add poi attributes to dataframe of nodes
                urban_plots = urban_plots.merge(pois_df, on='plot_id', how='left')

                for i in cols:
                    if i not in set(urban_plots.columns):
                        urban_plots[i] = 0
                    elif i in set(urban_plots.columns):
                        urban_plots[i] = urban_plots[i].replace(np.nan, 0)
                    
                urban_plots[cols] = urban_plots[cols].fillna(0)
                urban_plots = urban_plots.rename(columns = {'commercial':'Commercial', 'entertainment':'Entertainment','food':'Food','healthcare':'Healthcare','civic':'Civic', 'institutional':'Institutional', 'recreational':'Recreational', 'social':'Social'})

            if add_gee_layers:
                print('Adding earth engine layers to plots...')
                for i in range(len(add_gee_layers['layers'])):
                    layer = add_gee_layers['layers'][i]
                    method = add_gee_layers['methods'][i]
                    layer_name = add_gee_layers['layer_names'][i]
                    
                    try:
                        raster_gdf = gee_layer_from_boundary(self.polygon_bounds, layer, band='', index=0)
                    except EEException:
                        ee.Authenticate()
                        ee.Initialize()
                        raster_gdf = gee_layer_from_boundary(self.polygon_bounds, layer, band='', index=0)

                    if isinstance(raster_gdf, np.ndarray):
                        raster_df = mask_raster_with_gdf(urban_plots, raster_gdf)
                        raster_df = raster_df[['canopy_mean', 'canopy_stdev', 'canopy_skewness', 'canopy_kurtosis', 'plot_id']]
                        urban_plots = urban_plots.merge(raster_df, on='plot_id', how='left')
                        urban_plots[['canopy_mean', 'canopy_stdev', 'canopy_skewness', 'canopy_kurtosis']] = urban_plots[['canopy_mean', 'canopy_stdev', 'canopy_skewness', 'canopy_kurtosis']].fillna(0)

                    else:
                        raster_gdf.columns = [layer_name, 'geometry']
                        urban_plots = merge_raster_to_gdf(raster_gdf, urban_plots, id_col = 'plot_id', raster_col=layer_name, method=method)

            if add_lcz:
                print('Adding local climate zone categories to plots...')
                try:
                    lcz_array_gdf = gee_layer_from_boundary(self.polygon_bounds, 'RUB/RUBCLIM/LCZ/global_lcz_map/latest', band='', index=0, large=False)

                except EEException:
                    ee.Authenticate()
                    ee.Initialize()
                    lcz_array_gdf = gee_layer_from_boundary(self.polygon_bounds, 'RUB/RUBCLIM/LCZ/global_lcz_map/latest', band='', index=0, large=False)
                
                lcz_array_gdf.columns = ['lcz', 'geometry']
                urban_plots = merge_raster_to_gdf(lcz_array_gdf, urban_plots, id_col = 'plot_id', raster_col='lcz', num_classes = 18, method='proportion')

            # Add population
            if population_layer == 'meta': 
                print('Adding Meta population counts to plots...')
                long_min, lat_min, long_max, lat_max = self.polygon_bounds.geometry.total_bounds

                pop_subgroups = ['population', 'children', 'youth', 'elderly', 'men', 'women']

                # Example usage
                bounding_box = self.polygon_bounds.geometry.total_bounds  # Example bounding box (xmin, ymin, xmax, ymax)

                if pop_filepath != '':
                    tif_files = glob.glob(os.path.join(pop_filepath, '*.tif'))

                    valid_tif_files = find_valid_tif_files(tif_files, bounding_box)
                    for subgroup in pop_subgroups:
                        current_paths = []

                        for path in valid_tif_files:
                            if subgroup == 'men':
                                if all(x in path for x in subgroup) and ('women' not in path):
                                    current_paths.append(path)
                            else:
                                if subgroup in path:
                                    current_paths.append(path)

                        if subgroup == 'population':
                            subgroup_pop_gdf = gpd.GeoDataFrame()
                            for subgroup_tif_fp in current_paths:
                                temp_subgroup_pop_gdf = raster2gdf(subgroup_tif_fp, zoom=True, boundary=self.polygon_bounds, same_geometry=False)
                                temp_subgroup_pop_gdf = temp_subgroup_pop_gdf.fillna(0)
                                temp_subgroup_pop_gdf = temp_subgroup_pop_gdf.rename(columns={'value':subgroup})
                                subgroup_pop_gdf = pd.concat([subgroup_pop_gdf, temp_subgroup_pop_gdf], ignore_index=True, axis=0)

                        else: 
                            new_pop_gdf = pd.DataFrame()
                            for subgroup_tif_fp in current_paths:
                                temp_new_pop_gdf = raster2gdf(subgroup_tif_fp, zoom=True, boundary=self.polygon_bounds, same_geometry=True)
                                temp_new_pop_gdf = temp_new_pop_gdf.fillna(0)
                                temp_new_pop_gdf = temp_new_pop_gdf.rename(columns={'value':subgroup})   
                                new_pop_gdf = pd.concat([new_pop_gdf, temp_new_pop_gdf], ignore_index=True, axis=0)

                            subgroup_pop_gdf = pd.concat([subgroup_pop_gdf, new_pop_gdf], axis=1)

                    self.pop = subgroup_pop_gdf
                    res_intersection = gpd.sjoin(subgroup_pop_gdf, urban_plots, how='inner')

                    subgroup_pop_series = res_intersection.groupby('plot_id')[['population','men','women','elderly','youth','children']].aggregate('sum')

                    # Join pixel mean to network edges
                    urban_plots = urban_plots.merge(subgroup_pop_series, on='plot_id', how='left')
                    urban_plots[['population','men','women','elderly','youth','children']] = urban_plots[['population','men','women','elderly','youth','children']].fillna(0)
                else: 
                    if self.population:
                        print('Population data found, skipping re-computation.')
                        pop_list = self.population
                        target_cols = self.target_cols

                    else:
                        tile_countries_path = pkg_resources.resource_filename('urbanity', "map_data/tiled_data.json")
                        with open(tile_countries_path, 'r') as f:
                            tile_dict = json.load(f)
                            
                        tiled_country = [country[:-13] for country in list(tile_dict.keys())]
                        
                        # Use csv for small countries
                        if self.country not in tiled_country:
                            print('Using non-tiled population data.')
                            pop_list, target_cols = get_meta_population_data(self.country, 
                                                                        bounding_poly=self.polygon_bounds)
                            
                        # If big country, use csv and custom tiled population data: (e.g. USA: https://figshare.com/articles/dataset/USA_TILE_POPULATION/21502296)
                        elif self.country in tiled_country:
                            print('Using tiled population data.')
                            pop_list, target_cols = get_tiled_population_data(self.country, bounding_poly = self.polygon_bounds)
                        
                        self.population = pop_list
                        self.target_cols = target_cols
                    
                    groups = ['PopSum', 'Men', 'Women', 'Elderly','Youth','Children']

                    for i, data in enumerate(zip(pop_list, target_cols)):
                        proj_data = data[0].to_crs(urban_plots.crs)
                        res_intersection = proj_data.overlay(urban_plots, how='intersection')
                        pop_total_series = res_intersection.groupby(['plot_id'])[data[1]].sum()
                        pop_total_series.name = groups[i]
                        urban_plots = urban_plots.merge(pop_total_series, on='plot_id', how='left')
                            
                    for name in groups:
                        urban_plots[name] = urban_plots[name].replace(np.nan, 0).astype(int)
            
            elif population_layer == 'ghs':
                print('Adding GHS population counts to plots...')
                # Load global tile dataframe and fine overlapping grid
                ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
                ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)

                overlapping_grid = ghs_global_grid.overlay(buffered_tp)
                buffered_tp['geometry'] = buffer_polygon(self.polygon_bounds, bandwidth=bandwidth)
                # Loop through each year and obtain tif file
                origin_gdf_built = gpd.GeoDataFrame()
                origin_gdf_pop = gpd.GeoDataFrame()

                for k, year in enumerate(temporal_years):
                    # Only compute geometry once
                    if k == 0:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                            raster_gdf_built = raster2gdf(raster_dataset_built, zoom=True, boundary = buffered_tp, same_geometry=False)
                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=False)

                        elif len(overlapping_grid) > 1: 
                            raster_list_built = []
                            raster_list_pop = []
                            for i, row in overlapping_grid.iterrows():
                                target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"                           
                                raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)
                                raster_list_built.append(raster_dataset_built)
                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_built = merge_raster_list(raster_list_built)
                            mosaic_pop = merge_raster_list(raster_list_pop)
                            raster_gdf_built = raster2gdf(mosaic_built, zoom=True, boundary = buffered_tp, same_geometry=False)
                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=False)
                        
                        raster_gdf_built.columns = [str(year), 'geometry']
                        origin_gdf_built = gpd.GeoDataFrame(pd.concat([origin_gdf_built, raster_gdf_built], axis=1))
                        
                        raster_gdf_pop.columns = [str(year), 'geometry']
                        origin_gdf_pop = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1))
                        
                    else:
                        # If only one tile
                        if len(overlapping_grid) == 1:
                            row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                            target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                            target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"

                            raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                            raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                            raster_gdf_built = raster2gdf(raster_dataset_built, zoom=True, boundary = buffered_tp, same_geometry=True)
                            raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        elif len(overlapping_grid) > 1: 
                            raster_list_built = []
                            raster_list_pop = []

                            for i, row in overlapping_grid.iterrows():
                                target_tif_built = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"
                                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"

                                raster_dataset_built = download_pop_tiff_from_path(target_tif_built)
                                raster_dataset_pop = download_pop_tiff_from_path(target_tif_pop)

                                raster_list_built.append(raster_dataset_built)
                                raster_list_pop.append(raster_dataset_pop)

                            # Merge rasters
                            mosaic_built = merge_raster_list(raster_list_built)
                            mosaic_pop = merge_raster_list(raster_list_pop)

                            raster_gdf_built = raster2gdf(mosaic_built, zoom=True, boundary = buffered_tp, same_geometry=True)
                            raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

                        raster_gdf_built.columns = [str(year)]
                        origin_gdf_built  = gpd.GeoDataFrame(pd.concat([origin_gdf_built, raster_gdf_built], axis=1)) 
                        
                        raster_gdf_pop.columns = [str(year)]
                        origin_gdf_pop  = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1)) 

                temporal_rename_pop = {str(i):f'{i}_pop' for i in temporal_years}
                temporal_pop_columns = [f'{i}_pop' for i in temporal_years]
                origin_gdf_pop = origin_gdf_pop.rename(columns=temporal_rename_pop)

                temporal_rename_built = {str(i):f'{i}_built' for i in temporal_years}
                temporal_built_columns = [f'{i}_built' for i in temporal_years]
                origin_gdf_built = origin_gdf_built.rename(columns=temporal_rename_built)

                # Merge pop counts to plots
                origin_gdf_pop = origin_gdf_pop.to_crs(urban_plots.crs)
                res_intersection = origin_gdf_pop.overlay(urban_plots, how='intersection')
                pop_total_df = res_intersection.groupby(['plot_id'])[temporal_pop_columns].sum()
                urban_plots = urban_plots.merge(pop_total_df, on='plot_id', how='left')
                
                # Merge built up percentage to plots
                origin_gdf_built = origin_gdf_built.to_crs(urban_plots.crs)
                res_intersection = origin_gdf_built.overlay(urban_plots, how='intersection')
                built_total_df = res_intersection.groupby(['plot_id'])[temporal_built_columns].mean()
                urban_plots = urban_plots.merge(built_total_df, on='plot_id', how='left')

            print(f'Population attributes computed. Time taken: {round(time.time() - start)}.')  

            urban_plots['geometry'] = urban_plots.make_valid()

            urban_plots['plot_id'] = urban_plots.index

            self.urban_plots = urban_plots

        # Get edge connections
        return UrbanGraph(self.polygon_bounds, self.buildings, self.urban_plots, self.network[2], self.network[1])       


        print("Total elapsed time --- %s seconds ---" % round(time.time() - start))
        
        if save_as_h5:
            # Preprocess graph
    
            process_objects = fill_na_in_objects(objects)
            process_objects = remove_non_numeric_columns_objects(process_objects)
            # process_objects = standardise_and_scale(process_objects)
            save_to_h5(save_filepath, process_objects, connections)
            return objects, connections
        
        if save_as_npz:

            process_objects = fill_na_in_objects(objects)
            process_objects = remove_non_numeric_columns_objects(process_objects)
            # process_objects = standardise_and_scale(process_objects)
            save_to_npz(save_filepath, process_objects, connections)
            return process_objects, connections

        
        else:
            process_objects = fill_na_in_objects(objects)
            process_objects = remove_non_numeric_columns_objects(process_objects, keep_geometry=True)
            self.objects = process_objects
            self.connections = connections
            return process_objects, connections
            