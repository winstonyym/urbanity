import os
import cv2
import time
import math
import torch
import torch.nn as nn
import requests
import warnings
from collections import deque
warnings.filterwarnings("ignore", category=UserWarning)

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import mercantile
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
import threading
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib import cm
from shapely.strtree import STRtree
from scipy.stats import entropy

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from shapely.geometry import Point, Polygon, box
from mapbox_vector_tile import decode
from vt2geojson.tools import vt_bytes_to_geojson
from vt2geojson.features import Layer, Feature

from ipyleaflet import Marker, Popup
from ipywidgets import HTML
from shapely.geometry import LineString, Point
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
NUM_CLASSES = 65

CLS_DICT = {'0': 'Bird',
 '1': 'Ground-Animal',
 '2': 'Curb',
 '3': 'Fence',
 '4': 'Guard-Rail',
 '5': 'Barrier',
 '6': 'Wall',
 '7': 'Bike-Lane',
 '8': 'Crosswalk---Plain',
 '9': 'Curb-Cut',
 '10': 'Parking',
 '11': 'Pedestrian-Area',
 '12': 'Rail-Track',
 '13': 'Road',
 '14': 'Service-Lane',
 '15': 'Sidewalk',
 '16': 'Bridge',
 '17': 'Building',
 '18': 'Tunnel',
 '19': 'Person',
 '20': 'Bicyclist',
 '21': 'Motorcyclist',
 '22': 'Other-Rider',
 '23': 'Lane-Marking---Crosswalk',
 '24': 'Lane-Marking---General',
 '25': 'Mountain',
 '26': 'Sand',
 '27': 'Sky',
 '28': 'Snow',
 '29': 'Terrain',
 '30': 'Vegetation',
 '31': 'Water',
 '32': 'Banner',
 '33': 'Bench',
 '34': 'Bike-Rack',
 '35': 'Billboard',
 '36': 'Catch-Basin',
 '37': 'CCTV-Camera',
 '38': 'Fire-Hydrant',
 '39': 'Junction-Box',
 '40': 'Mailbox',
 '41': 'Manhole',
 '42': 'Phone-Booth',
 '43': 'Pothole',
 '44': 'Street-Light',
 '45': 'Pole',
 '46': 'Traffic-Sign-Frame',
 '47': 'Utility-Pole',
 '48': 'Traffic-Light',
 '49': 'Traffic-Sign-(Back)',
 '50': 'Traffic-Sign-(Front)',
 '51': 'Trash-Can',
 '52': 'Bicycle',
 '53': 'Boat',
 '54': 'Bus',
 '55': 'Car',
 '56': 'Caravan',
 '57': 'Motorcycle',
 '58': 'On-Rails',
 '59': 'Other-Vehicle',
 '60': 'Trailer',
 '61': 'Truck',
 '62': 'Wheeled-Slow',
 '63': 'Car-Mount',
 '64': 'Ego-Vehicle',
 '-1': 'Others'}

# For Visualisation
CLS_DICT_INT = {int(k): v for k, v in CLS_DICT.items()}
LABEL_TO_ID = {v: k for k, v in CLS_DICT.items()}          # string id
LABEL_TO_ID_INT = {v: int(k) for k, v in CLS_DICT.items()} # int id

perception_model_path = '../models'

perception = ['safety', 'lively', 'wealthy',
              'beautiful', 'boring', 'depressing']

perception_model_dict = {
    'safety': 'safety.pth',
    'lively': 'lively.pth',
    'wealthy': 'wealthy.pth',
    'beautiful': 'beautiful.pth',
    'boring': 'boring.pth',
    'depressing': 'depressing.pth',
}

# Client secret
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

transform_img = T.Compose([
    T.ToTensor(),
    T.Resize((384, 384)),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def predict(model, img, device, transform=None):

    if transform:
        img = transform(img)

    img = img.view(1, 3, 384, 384)
    # inference
    if device == 'cuda':
        pred = model(img.cuda())
    else:
        pred = model(img)

    softmax = nn.Softmax(dim=1)
    pred = softmax(pred)[0][1].item()
    pred = round(pred*10, 2)

    return pred

def batch_predict(model, img_batch):

    pred = model(img_batch)

    softmax = nn.Softmax(dim=1)
    pred = [round(i.item()*10,2) for i in softmax(pred)[:,1]]
    
    return pred

def load_perception_models(perception_model_path, device):
    perception_models = []
    for p in perception:
        model_path = perception_model_path + "/" + perception_model_dict[p]
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
        model = model.to(device)
        model.eval()
        perception_models.append(model)
    return perception_models

def addInstance(output_max):
    if isinstance(output_max, list):
       output_max = output_max[0]

    set_dictionary = {}
    for i in CLS_DICT.values():
        set_dictionary[i] = 0

    list_unique, list_counts = torch.unique(output_max['segmentation'].int(), return_counts=True)

    total = torch.sum(list_counts).item() 

    if -1 in list_unique:
        set_dictionary['Others'] = round(list_counts[0].item() / total, 5)
        list_unique = list_unique[1:]
        list_counts = list_counts[1:]

    matching_dict = {}

    for i, k in zip(range(len(output_max['segments_info'])), output_max['segments_info']):
        matching_dict[i] = int(k['label_id'])

    for i, k in zip(list_unique, list_counts):
        set_dictionary[CLS_DICT[str(matching_dict[i.item()])]] += round(k.item() / total, 5)
    
    set_dictionary['Visual Complexity'] = entropy(np.array(list(set_dictionary.values())), base=2)

    return set_dictionary

def addInstanceCounts(output_max):
    if isinstance(output_max, list):
       output_max = output_max[0]

    instance_dictionary = {}
    
    instance_dictionary = {}
    for i in CLS_DICT.values():
                instance_dictionary[str(i)] = 0
    
    # for each segment, draw its legend
    for segment in output_max['segments_info']:
        segment_id = segment['id']
        segment_label_id = str(segment['label_id'])
        instance_dictionary[CLS_DICT[segment_label_id]] += 1

    return instance_dictionary


def merge_zones_by_group(gdf, column):
    """Function to merge spatial dataframe according to a higher level of administrative grouping. 
    For example, this function can merge neighbourhoods into districts. 

    Args:
        gdf (gpd.GeoDataFrame): A spatial dataframe object with administrative zoning at a higher granularity.
        column (str): Specify column name to spatially merge administrative subzones groups.

    Returns:
        gpd.GeoDataFrame: Returns a spatial dataframe object with merged subzones. 
    """    
    unique_zones = list(gdf[column].unique())
    
    # Create empty city dataframe
    city_subzone = pd.DataFrame({'GID':[], 'geometry':[]})
    
    # Define internal merging function
    def merge_subzones(gdf, column, zone):
        polygons = gdf[gdf[column] == zone]
        geom = polygons['geometry'].unary_union.wkt
        df = pd.DataFrame({'GID':[zone], 'geometry':geom})
        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x))
        gdf = gpd.GeoDataFrame(data=df, crs = gdf.crs, geometry=df['geometry'])
        return gdf
    
    # Iteratively add merged subzones
    for zone in unique_zones:
        merged = merge_subzones(gdf, column, zone)
        city_subzone = pd.concat([city_subzone, merged], axis=0)
        
    # Convert to GeoDataFrame
    city_subzone = gpd.GeoDataFrame(city_subzone, crs='epsg:4326', geometry = city_subzone['geometry'])
    
    return city_subzone

def sequence_download_image_in_tiles(gdf, api_key = ''):

    # Get bounds of shapefile
    bbox = list(gdf.geometry.bounds.values[0])

    # Obtain mercantile tiles for mapillary query
    tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], 14))

    # Initialise empty lists
    lat_list = []
    long_list = []
    image_list = []
    time_list = []
    angle_list = []
    tile_list = []
    creator_list = []
    sequence_list = []

    # Loop through tiles and download image meta info
    for tile in tqdm(tiles):
        tile_url = 'https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{}/{}/{}'.format(tile.z,tile.x,tile.y)
        response = requests.get(tile_url, params={'access_token': api_key})
        
        geojson = request_to_geo_json(response, tile.x, tile.y, tile.z)
        
        # push to output geojson object if yes
        if geojson: 
            for feature in geojson['features']:
                # Filter images within bbox since tiles may extend beyond bounding box
                if not feature['properties']['is_pano']:
                    long_list.append(feature['geometry']['coordinates'][0])
                    lat_list.append(feature['geometry']['coordinates'][1])
                    image_list.append(feature['properties']['id'])
                    time_list.append(feature['properties']['captured_at'])
                    angle_list.append(feature['properties']['compass_angle'])
                    creator_list.append(feature['properties']['creator_id'])
                    sequence_list.append(feature['properties']['sequence_id'])
                    tile_list.append(f'{tile.z}/{tile.x}/{tile.y}')
                
    df = pd.DataFrame({'tile_id': tile_list, 
                       'image_id': image_list, 
                       'time':time_list, 
                       'angle': angle_list,
                       'creator': creator_list,
                       'sequence': sequence_list})
    image_gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(long_list, lat_list), crs = 'EPSG:4326')

    image_gdf = image_gdf[image_gdf.within(gdf.geometry.values[0])]

    return image_gdf

# Process a tile with a session
def process_tile(tile, session, api_key, max_retries=3, retry_delay=1):
    tile_url = f'https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{tile.z}/{tile.x}/{tile.y}'
    attempt = 0
    while attempt < max_retries:
        try:
            response = session.get(tile_url, params={'access_token': api_key})
            response.raise_for_status()
            break  # Exit loop if request is successful
        except requests.RequestException as e:
            attempt += 1
            print(f'Connection error for tile {tile.z}/{tile.x}/{tile.y} (Attempt {attempt}/{max_retries}): {e}')
            if attempt < max_retries:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f'Failed to fetch tile {tile.z}/{tile.x}/{tile.y} after {max_retries} attempts.')
                return []  # Return an empty list if all attempts fail

    geojson = request_to_geo_json(response, tile.x, tile.y, tile.z)
    
    data = []
    if geojson:
        for feature in geojson['features']:
            if not feature['properties']['is_pano']:
                data.append({
                    'long': feature['geometry']['coordinates'][0],
                    'lat': feature['geometry']['coordinates'][1],
                    'image_id': feature['properties']['id'],
                    'time': feature['properties']['captured_at'],
                    'angle': feature['properties']['compass_angle'],
                    'creator_id': feature['properties']['creator_id'],
                    'sequence_id': feature['properties']['sequence_id'],
                    'tile_id': f'{tile.z}/{tile.x}/{tile.y}'
                })
    return data

def parallel_download_image_in_tiles(gdf, api_key = ''):
    bbox = list(gdf.geometry.bounds.values[0])
    total_tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], 14))

    geometries = [box(*mercantile.bounds(tile)) for tile in total_tiles]

    # Create a GeoDataFrame
    tile_gdf = gpd.GeoDataFrame(data = {'tile_id':list(range(len(geometries)))},geometry=geometries, crs="EPSG:4326")
    tile_intersection = tile_gdf.overlay(gdf)
    tiles = [total_tiles[idx] for idx in tile_intersection['tile_id'].values]

    # Initialize lists
    lat_list = []
    long_list = []
    image_list = []
    time_list = []
    angle_list = []
    tile_list = []
    creator_list = []
    sequence_list = []

    results = []
    results_lock = threading.Lock()

    # Create a session object
    session = requests.Session()
    
    # Create and start threads
    # with tqdm(total=len(tiles)) as pbar:
    with ThreadPoolExecutor() as executor:
        future_to_tile = {executor.submit(process_tile, tile, session, api_key): tile for tile in tiles}
        for future in as_completed(future_to_tile):
            try:
                data = future.result()
                with results_lock:
                    results.extend(data)
            except Exception as e:
                print(f'Error occurred while processing tile: {e}')
                # pbar.update(1)

    # Close the session
    session.close()

    # Collect results from all threads
    for entry in results:
        long_list.append(entry['long'])
        lat_list.append(entry['lat'])
        image_list.append(entry['image_id'])
        time_list.append(entry['time'])
        angle_list.append(entry['angle'])
        creator_list.append(entry['creator_id'])
        sequence_list.append(entry['sequence_id'])
        tile_list.append(entry['tile_id'])

    df = pd.DataFrame({
        'tile_id': tile_list, 
        'image_id': image_list, 
        'time': time_list, 
        'angle': angle_list,
        'creator': creator_list,
        'sequence': sequence_list
    })

    image_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(long_list, lat_list), crs='EPSG:4326')
    image_gdf = image_gdf[image_gdf.within(gdf.geometry.values[0])]

    return image_gdf


def process_image(image_id, session, api_key, max_retries=3, retry_delay=1):

    url = f'https://graph.mapillary.com/{image_id}?fields=thumb_256_url'

    attempt = 0
    while attempt < max_retries:
        try:
            response = session.get(url, params={'access_token': api_key})
            response.raise_for_status()
            image_data = response.json()
            image_url = image_data['thumb_256_url']
    
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            response = response.raw
        
            break  # Exit loop if request is successful
        except requests.RequestException as e:
            attempt += 1
            print(f'Connection error for image: {image_id} (Attempt {attempt}/{max_retries}): {e}')
            if attempt < max_retries:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f'Failed to fetch image: {image_id} after {max_retries} attempts.')
                return []  # Return an empty list if all attempts fail

    image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)   
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_id, image_rgb

def segment_image(img_id, image_rgb, processor, segment_model, device):
    # Preprocessing
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        pixel_values = inputs['pixel_values'].to(device)
        pixel_mask = inputs['pixel_mask'].to(device)
        outputs = segment_model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    # Post-process
    out = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image_rgb.shape[:-1]], threshold=0.5
    )
    return img_id, addInstance(out), addInstanceCounts(out)

def parallel_segment_images(image_gdf, processor, segment_model, api_key, device):
    # Initialise placeholders
    image_instances_dict = {}
    image_indicators_dict = {}

    results = {}
    results_lock = threading.Lock()

    # Create a session object
    session = requests.Session()
    
    failures = []
    # Create and start threads
    with ThreadPoolExecutor(max_workers=None) as executor:
        future_to_id = {executor.submit(process_image, image_id, session, api_key): image_id for image_id in image_gdf['image_id']}
        with tqdm(total=len(future_to_id), desc="Downloading images", unit="img") as pbar:
            for future in as_completed(future_to_id):
                image_id = future_to_id[future]
                try:
                    img_id, image_rgb = future.result()
                    with results_lock:
                        results[img_id] = image_rgb
                except Exception as e:
                    failures.append((image_id, e))
                finally:
                    pbar.update(1)

    # Close the session
    session.close()
    print('Downloaded images...')
    print('Starting segmentation')
    # Load model, etc. as before
    segment_model = segment_model.to(device)

    # Prepare results containers
    image_instances_dict = {}
    image_indicators_dict = {}

    futures = []
    with ThreadPoolExecutor(max_workers=None) as executor:  # adjust workers as needed
        for img_id, image_rgb in results.items():
            futures.append(executor.submit(
                segment_image, img_id, image_rgb, processor, segment_model, device
            ))
        for future in tqdm(as_completed(futures), total=len(futures)):
            img_id, indicators, instances = future.result()
            image_indicators_dict[img_id] = indicators
            image_instances_dict[img_id] = instances

    # Convert to dataframes
    indicators_df = pd.DataFrame.from_dict(image_indicators_dict, orient='index')
    instances_df = pd.DataFrame.from_dict(image_instances_dict, orient='index')

    # Merge dataframes
    merged = image_gdf.merge(indicators_df, how='left', left_on='image_id', right_on = indicators_df.index)
    merged = merged.merge(instances_df, how='left', left_on='image_id', right_on = indicators_df.index, suffixes=('_pixels', '_counts'))

    return merged

def parallel_batch_images(api_key, batch_size_ids):

    results = {}
    results_lock = threading.Lock()

    # Create a session object
    session = requests.Session()

    
    # Create and start threads
    # with tqdm(total=len(tiles)) as pbar:
    with ThreadPoolExecutor() as executor:
        future_to_tile = {executor.submit(process_image, image_id, session, api_key): image_id for image_id in batch_size_ids}
        for future in as_completed(future_to_tile):
            try:
                image_id, image_rgb = future.result()
                with results_lock:
                    results[image_id] = image_rgb
            except Exception as e:
                print(f'Error occurred while processing tile: {e}')
                # pbar.update(1)

    # Close the session
    session.close()
    
    return results.items()

def parallel_segment_images_in_batches(image_gdf, 
                                       processor,
                                       model, 
                                       api_key, 
                                       device, 
                                       batch_size=1000,
                                       gpu_batch=4,
                                       add_perception=False,
                                       perception_models=None):

    # Load images
    image_set = list(image_gdf['image_id'].unique())

    model = model.to(device)
    
    # Load Perception Models
    if add_perception:
        if perception_models is None:
            raise Exception('Please specify perception models')
        
        columns = ['image_id'] + [str(p)+"_score" for p in perception]
        perception_df = pd.DataFrame()

    # Initialise placeholders
    image_instances_dict = {}
    image_indicators_dict = {}

    for i in range(0, len(image_set), batch_size):
        batch_image_ids = image_set[i:i + batch_size]
        results = parallel_batch_images(api_key, batch_image_ids)
        print(f'Segmenting images: {i+batch_size}/{len(image_set)}')

        results_img_id = [img_id for img_id, img_rgb in results]
        results_img_rgb = [img_rgb for img_id, img_rgb in results]

        for i in range(0, len(results), gpu_batch):
            img_id_list = results_img_id[i:i + gpu_batch]
            img_rgb_list = results_img_rgb[i:i + gpu_batch]
            processed = [processor(images=img_rgb, return_tensors="pt") for img_rgb in img_rgb_list]
            value_tensor_list = torch.concat([value['pixel_values'] for value in processed]).to(device)
            mask_tensor_list = torch.concat([value['pixel_mask'] for value in processed]).to(device)
            # Inference
            outputs = model(pixel_values=value_tensor_list, pixel_mask=mask_tensor_list)
            out = processor.post_process_instance_segmentation(outputs, target_sizes=[img.shape[:-1] for img in img_rgb_list], threshold=0.5)

            instance_list = map(addInstance, out)
            instance_counts_list = map(addInstanceCounts, out)

            for img_id, instance_dict in zip(img_id_list, instance_list):
                image_instances_dict[img_id] = instance_dict

            for img_id, instance_counts_dict in zip(img_id_list, instance_counts_list):
                image_indicators_dict[img_id] = instance_counts_dict

            if add_perception:
                perception_preds = [batch_predict(model, value_tensor_list) for model in perception_models]
                new_df = pd.DataFrame({'image_id': img_id_list,
                                    'safety_score': perception_preds[0],
                                    'lively_score': perception_preds[1],
                                    'wealthy_score': perception_preds[2],
                                    'beautiful_score': perception_preds[3],
                                    'boring_score': perception_preds[4],
                                    'depressing_score': perception_preds[5]})

                perception_df = pd.concat([perception_df, pd.DataFrame(new_df)], ignore_index=True)

    # Convert to dataframes
    indicators_df = pd.DataFrame.from_dict(image_indicators_dict, orient='index')
    instances_df = pd.DataFrame.from_dict(image_instances_dict, orient='index')

    # Merge dataframes
    merged = image_gdf.merge(indicators_df, how='left', left_on='image_id', right_on = indicators_df.index)
    merged = merged.merge(instances_df, how='left', left_on='image_id', right_on = indicators_df.index, suffixes=('_counts', '_pixels'))

    if add_perception:
        merged = merged.merge(perception_df, how='left', on='image_id')
    
    return merged
    
class MyFeature(Feature):
    def __init__(self, x, y, z, obj, extent=4096):
        super().__init__(x, y, z, obj, extent=4096)
    
    @property
    def tiles_coordinates(self):     
        return self.obj['geometry']['coordinates'] 
    
    @property
    def geometry_type(self):
        return 'Point'
    
    @property
    def properties(self):
        return self.obj['properties']   
    
    def toGeoJSON(self):
        size = self.extent * 2 ** self.z
        x0 = self.extent * self.x
        y0 = self.extent * self.y

        def project_one(p_x, p_y):
            y2 = 180 - (p_y + y0) * 360. / size
            long_res = (p_x + x0) * 360. / size - 180
            lat_res = 360. / math.pi * math.atan(math.exp(y2 * math.pi / 180)) - 90
            return [long_res, lat_res]

        def project(coords):
            if all(isinstance(x, int) or isinstance(x, float) for x in coords):
                assert len(coords) == 2
                return project_one(coords[0], coords[1])
            return [project(l) for l in coords]

        coords = project(self.tiles_coordinates)
        geometry_type = self.geometry_type

        result = {
            "type": "Feature",
            "geometry": {
                "type": geometry_type,
                "coordinates": coords
            },
            "properties": self.properties
        }
        return result

class MyLayer(Layer):
    def __init__(self, x, y, z, name, obj):
        super().__init__(x, y, z, name, obj)
    
    def toGeoJSON(self):
        return {
            "type": "FeatureCollection",
            "features": [MyFeature(x=self.x, y=self.y, z=self.z, obj=image, extent=4096).toGeoJSON()
                         for image in self.obj['features']]
        }
    
def request_to_geo_json(response: bytes, x: int, y: int, z: int, layer='image') -> dict:
    """
    Make a GeoJSON from vector tile query
    :param content: request information from Mapillary vector tile query
    :param x: tile x coordinate.
    :param y: tile y coordinate.
    :param z: tile z coordinate.
    :param layer: image or sequence layer. Default: 'image'.
    :return: GeoJSON dictionary.
    """
    
    try:
        data = decode(response.content, y_coord_down=True)
        geojson = MyLayer(x=x, y=y, z=z, name=layer, obj=data[layer]).toGeoJSON()
        return geojson
    except:
        # No images found in tile --skipping--
        pass
    
def filter_by_alignment(G, gdf, limit=10):
    
    # Add bearings to graph edges
    G = add_edge_bearings(G, precision=1)
    
    # Project graph to enable distance computation
    gdf2 = project_gdf(gdf) # image_gdf
    near_edge = nearest_edges(G, gdf2.geometry.x, gdf2.geometry.y)
    
    # Create list of origin and destination tuples
    nearest_edge_list = []
    for edge in near_edge:
        nearest_edge_list.append(edge[0:2])
    
    # Create dataframe consisting of nearest edge, bearing, and back_bearing
    edges = graph_to_gdf(G, edges=True).reset_index()
    edges['nearest_edge'] = list(zip(edges.u, edges.v))
    edges = edges[['nearest_edge', 'bearing', 'back_bearing']]
    
    # Assign nearest edge joining column to projected graph
    gdf2['nearest_edge'] = nearest_edge_list
    
    # Merge dataframes on nearest edge to combine image angle and bearing information
    merged_df = gdf2.merge(edges, how = 'left', on = 'nearest_edge')
    
    # If angle difference less than 20, return image, else discard
    merged_df['aligned'] = merged_df.apply(lambda row: check_align(row['bearing'], row['back_bearing'], row['angle'], limit), axis=1)
    merged_df = merged_df[merged_df['aligned'] == True]
    merged_df = merged_df.to_crs(4326)
    
    return merged_df

def explode_linestrings(gdf):
    """Explode linestring into single pairs of coordinate sequences"""
    exploded_lines = []
    
    # Iterate over each row in the GeoDataFrame
    for _, row in gdf.iterrows():
        line = row.geometry
        coords = list(line.coords)
        
        # Create segments from coordinates
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            exploded_lines.append({'geometry': segment, **row.drop('geometry')})
    
    # Create a new GeoDataFrame from exploded lines
    exploded_gdf = gpd.GeoDataFrame(exploded_lines, geometry='geometry', crs=gdf.crs)
    exploded_gdf['exploded_edge_id'] = exploded_gdf.index
    return exploded_gdf


def filter_by_alignment_finegrained(image_gdf_proj, tract_edges, limit=10):
    exploded_gdf = explode_linestrings(tract_edges)
    exploded_gdf['bearing'] = exploded_gdf['geometry'].apply(lambda x: calculate_bearing(x.coords[0][0], x.coords[0][1], x.coords[1][0], x.coords[1][1]))
    exploded_gdf['back_bearing'] = exploded_gdf['bearing'].apply(lambda x: (x - 180) if x >= 180 else (x + 180))

    exploded_gdf_proj = project_gdf(exploded_gdf)
    nearest_edge = gpd.sjoin_nearest(image_gdf_proj, exploded_gdf_proj, how='inner')
    nearest_edge = nearest_edge.drop_duplicates(subset='image_id', keep='first')

    # If angle difference less than limit, return image, else discard
    nearest_edge['aligned'] = nearest_edge.apply(lambda row: check_align(row['bearing'], row['back_bearing'], row['angle'], limit), axis=1)
    nearest_edge = nearest_edge[nearest_edge['aligned'] == True]
    nearest_edge.crs = image_gdf_proj.crs
    nearest_edge = nearest_edge.to_crs(4326)



# Attach images to network and assign image bearing
def calculate_bearing(lat1, lng1, lat2, lng2):
    """
    Calculate the compass bearing(s) between pairs of lat-lng points.

    Vectorized function to calculate (initial) bearings between two points'
    coordinates or between arrays of points' coordinates. Expects coordinates
    in decimal degrees. Bearing represents angle in degrees (clockwise)
    between north and the geodesic line from point 1 to point 2.

    Parameters
    ----------
    lat1 : float or numpy.array of float
        first point's latitude coordinate
    lng1 : float or numpy.array of float
        first point's longitude coordinate
    lat2 : float or numpy.array of float
        second point's latitude coordinate
    lng2 : float or numpy.array of float
        second point's longitude coordinate

    Returns
    -------
    bearing : float or numpy.array of float
        the bearing(s) in decimal degrees
    """
    # get the latitudes and the difference in longitudes, in radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    d_lng = np.radians(lng2 - lng1)

    # calculate initial bearing from -180 degrees to +180 degrees
    y = np.sin(d_lng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lng)
    initial_bearing = np.degrees(np.arctan2(y, x))

    # normalize to 0-360 degrees to get compass bearing
    return initial_bearing % 360


def add_edge_bearings(G, precision=1):
    """
    Add compass `bearing` attributes to all graph edges.

    Vectorized function to calculate (initial) bearing from origin node to
    destination node for each edge in a directed, unprojected graph then add
    these bearings as new edge attributes. Bearing represents angle in degrees
    (clockwise) between north and the geodesic line from the origin node to
    the destination node. Ignores self-loop edges as their bearings are
    undefined.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        unprojected graph
    precision : int
        decimal precision to round bearing

    Returns
    -------
    G : networkx.MultiDiGraph
        graph with edge bearing attributes
    """

    # extract edge IDs and corresponding coordinates from their nodes
    uvk = [(u, v, k) for u, v, k in G.edges if u != v]
    x = G.nodes(data="x")
    y = G.nodes(data="y")
    coords = np.array([(y[u], x[u], y[v], x[v]) for u, v, k in uvk])

    # calculate bearings then set as edge attributes
    bearings = calculate_bearing(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
    back_bearing = calculate_bearing(coords[:, 2], coords[:, 3], coords[:, 0], coords[:, 1])
    val = zip(uvk, bearings.round(precision))
    back_val = zip(uvk, back_bearing.round(precision))
    nx.set_edge_attributes(G, dict(val), name="bearing")
    nx.set_edge_attributes(G, dict(back_val), name="back_bearing")

    return G

def nearest_edges(G, X, Y, interpolate=None, return_dist=False):
    """
    Find the nearest edge to a point or to each of several points.

    If `X` and `Y` are single coordinate values, this will return the nearest
    edge to that point. If `X` and `Y` are lists of coordinate values, this
    will return the nearest edge to each point.

    If `interpolate` is None, search for the nearest edge to each point, one
    at a time, using an r-tree and minimizing the euclidean distances from the
    point to the possible matches. For accuracy, use a projected graph and
    points. This method is precise and also fastest if searching for few
    points relative to the graph's size.

    For a faster method if searching for many points relative to the graph's
    size, use the `interpolate` argument to interpolate points along the edges
    and index them. If the graph is projected, this uses a k-d tree for
    euclidean nearest neighbor search, which requires that scipy is installed
    as an optional dependency. If graph is unprojected, this uses a ball tree
    for haversine nearest neighbor search, which requires that scikit-learn is
    installed as an optional dependency.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        graph in which to find nearest edges
    X : float or list
        points' x (longitude) coordinates, in same CRS/units as graph and
        containing no nulls
    Y : float or list
        points' y (latitude) coordinates, in same CRS/units as graph and
        containing no nulls
    interpolate : float
        spacing distance between interpolated points, in same units as graph.
        smaller values generate more points.
    return_dist : bool
        optionally also return distance between points and nearest edges

    Returns
    -------
    ne or (ne, dist) : tuple or list
        nearest edges as (u, v, key) or optionally a tuple where `dist`
        contains distances between the points and their nearest edges
    """
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        # make coordinates arrays if user passed non-iterable values
        is_scalar = True
        X = np.array([X])
        Y = np.array([Y])

    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")
    geoms = project_gdf(graph_to_gdf(G, edges=True))["geometry"]

    # if no interpolation distance was provided
    if interpolate is None:

        # build the r-tree spatial index by position for subsequent iloc
        rtree = RTreeIndex()
        for pos, bounds in enumerate(geoms.bounds.values):
            rtree.insert(pos, bounds)

        # use r-tree to find possible nearest neighbors, one point at a time,
        # then minimize euclidean distance from point to the possible matches
        ne_dist = list()
        for xy in zip(X, Y):
            dists = geoms.iloc[list(rtree.nearest(xy, num_results=10))].distance(Point(xy))
            ne_dist.append((dists.idxmin(), dists.min()))
        ne, dist = zip(*ne_dist)

    # otherwise, if interpolation distance was provided
    else:

        # interpolate points along edges to index with k-d tree or ball tree
        uvk_xy = list()
        for uvk, geom in zip(geoms.index, geoms.values):
            uvk_xy.extend((uvk, xy) for xy in utils_geo.interpolate_points(geom, interpolate))
        labels, xy = zip(*uvk_xy)
        vertices = pd.DataFrame(xy, index=labels, columns=["x", "y"])

        if projection.is_projected(G.graph["crs"]):
            # if projected, use k-d tree for euclidean nearest-neighbor search
            if cKDTree is None:  # pragma: no cover
                raise ImportError("scipy must be installed to search a projected graph")
            dist, pos = cKDTree(vertices).query(np.array([X, Y]).T, k=1)
            ne = vertices.index[pos]

        else:
            # if unprojected, use ball tree for haversine nearest-neighbor search
            if BallTree is None:  # pragma: no cover
                raise ImportError("scikit-learn must be installed to search an unprojected graph")
            # haversine requires lat, lng coords in radians
            vertices_rad = np.deg2rad(vertices[["y", "x"]])
            points_rad = np.deg2rad(np.array([Y, X]).T)
            dist, pos = BallTree(vertices_rad, metric="haversine").query(points_rad, k=1)
            dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters
            ne = vertices.index[pos[:, 0]]

    # convert results to correct types for return
    ne = list(ne)
    dist = list(dist)
    if is_scalar:
        ne = ne[0]
        dist = dist[0]

    if return_dist:
        return ne, dist
    else:
        return ne
    
def graph_to_gdf(G, nodes=False, edges=False, dual=False):

    crs = G.graph['crs']
    if nodes:
        if not G.nodes:  # pragma: no cover
            raise ValueError("graph contains no nodes")
            
        nodes, data = zip(*G.nodes(data=True))
        
        # convert node x/y attributes to Points for geometry column
        geom = (Point(d["x"], d["y"]) for d in data)
        gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
         
        if not dual:
            gdf_nodes.index.rename("osmid", inplace=True)
        
    if edges:
        if not G.edges:  # pragma: no cover
            raise ValueError("graph contains no edges")
        
        if not dual: 
            u, v, k, data = zip(*G.edges(keys=True, data=True))
        else: 
            u, v, data = zip(*G.edges(data=True))
            
        x_lookup = nx.get_node_attributes(G, "x")
        y_lookup = nx.get_node_attributes(G, "y")
        def make_geom(u, v, data, x= x_lookup, y= y_lookup):
            if "geometry" in data:
                return data["geometry"]
            else:
                return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))
            
        geom = map(make_geom, u, v, data)
        gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))


        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        if not dual: 
            gdf_edges["key"] = k
            gdf_edges.set_index(["u", "v", "key"], inplace=True)
        else: 
            gdf_edges.set_index(["u", "v"], inplace=True)
    if nodes and edges:
        return gdf_nodes, gdf_edges
    elif nodes:
        return gdf_nodes
    elif edges:
        return gdf_edges
    
def project_gdf(gdf):
    mean_longitude = gdf["geometry"].representative_point().x.mean()

    # Compute UTM crs
    utm_zone = int(np.floor((mean_longitude + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # project the GeoDataFrame to the UTM CRS
    gdf_proj = gdf.to_crs(utm_crs)
    # print(f"Projected to {gdf_proj.crs}")
    
    return gdf_proj

def check_align(bearing, back_bearing, angle, limit = 20):
    if (abs(angle - bearing) < limit) or (abs(angle - back_bearing) < limit):
        return True
    else:
        return False
    
def dissolve_poly(filepath, name):
    df = gpd.read_file(filepath)
    df['geometry'] = df.buffer(0.001)
    df[name] = name
    df = df.dissolve(name).reset_index()
    return df

def filter_by_datetime(df, timezone, start='9:00', end='18:00'):
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize("UTC").tz_convert(timezone)
    df = df.between_time(start, end)
    df = df.reset_index()
    df = df.drop(columns= ['time'])
    return df, len(df)

def filter_by_edge_length(proj_gdf, interval = 5):
    proj_gdf['expected_images'] = proj_gdf['length'].apply(lambda x: int(x/interval))
    proj_gdf = proj_gdf.drop_duplicates(subset='image_id', keep='first')
    proj_gdf = proj_gdf.groupby('edge_id').apply(sample_group, include_groups=False).reset_index(drop=True)
    return proj_gdf

def sample_group(group):
    # Check if the group size is less than expected_images
    if len(group) < group['expected_images'].iloc[0]:
        # If so, sample all rows
        return group.sample(len(group))
    else:
        # Otherwise, sample 'expected_images' rows
        return group.sample(group['expected_images'].iloc[0])

def download_csv_to_dataframe(url, chunk_size=1024):
    """
    Downloads a large CSV file from the given URL and reads it into a pandas DataFrame.
    
    Parameters:
    - url (str): URL of the CSV file to download.
    - chunk_size (int): Size of chunks to read from the response (default is 1024 bytes).
    
    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.
    """
    # Stream the response
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Create a StringIO object to hold the CSV data in memory
    csv_data = StringIO()
    
    # Stream the data and write it to the StringIO object
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:  # filter out keep-alive new chunks
            csv_data.write(chunk.decode('utf-8'))
    
    # Move to the start of the StringIO object
    csv_data.seek(0)
    
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_data)
    
    return df


def find_optimal_chunk_size(url, buffer_sizes=[8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024]):
    """
    Finds the optimal chunk size for downloading a file from the given URL based on download time.
    
    Parameters:
    - url (str): URL of the file to download.
    - buffer_sizes (list of int): List of chunk sizes (in bytes) to test.
    
    Returns:
    - optimal_chunk_size (int): The buffer size that resulted in the shortest download time.
    - results (dict): Dictionary with buffer sizes as keys and times taken as values.
    """
    results = {}
    
    for buffer_size in buffer_sizes:
        print(f"Testing buffer size: {buffer_size} bytes")
        
        start_time = time.time()
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Temporarily store the file to measure download time
            with open('temp_download_file', 'wb') as file:
                for chunk in response.iter_content(chunk_size=buffer_size):
                    if chunk:
                        file.write(chunk)
        
        except Exception as e:
            print(f"Error occurred with buffer size {buffer_size}: {e}")
            results[buffer_size] = {'error': str(e)}
            continue
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results[buffer_size] = {'time': elapsed_time}
        print(f"Buffer size: {buffer_size} bytes, Time taken: {elapsed_time:.2f} seconds")
    
    # Determine the optimal chunk size
    optimal_chunk_size = min(results, key=lambda k: results[k]['time'] if 'time' in results[k] else float('inf'))
    
    return optimal_chunk_size, results

def get_image_url(image_id, session, api_key=''):
    """
    Query Mapillary Graph API for a single image's 256-thumb URL.
    """
    url = f'https://graph.mapillary.com/{image_id}?fields=thumb_256_url'
    resp = session.get(url, params={'access_token': api_key})
    resp.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
    data = resp.json()
    return data.get('thumb_256_url', None)


def view_image(image_id, api_key = ''):

    url = f'https://graph.mapillary.com/{image_id}?fields=thumb_256_url'

    response = session.get(url, params={'access_token': api_key})
    image_data = response.json()
    image_url = image_data['thumb_256_url']

    response = session.get(image_url, stream=True)
    response = response.raw

    image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)   
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def draw_line(lon, lat, bearing):
    r = 0.0001  # or whatever fits you
    plt.arrow(lon, lat, r*math.cos(bearing), r*math.sin(bearing), width=0.0001, color='red') 

def save_image_to_folder(image_gdf, output_dir, session, api_key, max_retries=3, retry_delay=1):
    image_set = list(image_gdf['image_id'].unique())
    for image_id in image_set:
        image_path = f'{image_id}.jpg'
        url = f'https://graph.mapillary.com/{image_id}?fields=thumb_256_url'

        attempt = 0
        while attempt < max_retries:
            try:
                response = session.get(url, params={'access_token': api_key})
                response.raise_for_status()
                image_data = response.json()
                image_url = image_data['thumb_256_url']
        
                response = session.get(image_url, stream=True)
                response.raise_for_status()
                response = response.raw
            
                break  # Exit loop if request is successful
            except requests.RequestException as e:
                attempt += 1
                print(f'Connection error for image: {image_id} (Attempt {attempt}/{max_retries}): {e}')
                if attempt < max_retries:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    print(f'Failed to fetch image: {image_id} after {max_retries} attempts.')
                    return []  # Return an empty list if all attempts fail

        image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)   
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, image_path), image_rgb)

def assign_closest_points_to_linestring(image_gdf, street_gdf):
    svi_gdf = image_gdf.copy()
    # 0) Project to meters
    if svi_gdf.crs is None or svi_gdf.crs.is_geographic:
        points_gdf = svi_gdf.to_crs(3857)
        lines_gdf  = street_gdf.to_crs(3857)

    # 1) Build an STRtree over line geometries (cheap to query many times)
    line_geoms   = lines_gdf.geometry.values
    tree         = STRtree(line_geoms)
    line_ids_arr = lines_gdf.index.to_numpy()  # or use a dedicated id column

    # 2) Query all nearest lines at once
    pt_geoms = points_gdf.geometry.values
    # idx_pt: indices into pt_geoms; idx_ln: indices into line_geoms
    idx, dists = tree.query_nearest(pt_geoms, return_distance=True)
    idx_pt = idx[0,:]
    idx_ln = idx[1,:]
    # 3) Place results back into arrays aligned to points_gdf
    nearest_line_id = np.empty(len(points_gdf), dtype=line_ids_arr.dtype)
    nearest_dist_m  = np.empty(len(points_gdf), dtype="float64")
    nearest_line_id[idx_pt] = line_ids_arr[idx_ln]
    nearest_dist_m[idx_pt]  = dists

    # 4) Attach to a copy or the original GeoDataFrame
    out = points_gdf.copy()
    out["nearest_street_id"] = nearest_line_id
    out["dist_to_line"] = nearest_dist_m

    svi_gdf = svi_gdf.merge(out[['image_id', 'nearest_street_id', 'dist_to_line']], on='image_id')

    return svi_gdf

def assign_closest_points_to_linestring_and_sample(image_gdf, street_gdf, bin_distance = 20, proximity_distance = 10):
    svi_gdf = image_gdf.copy()
    if svi_gdf.crs is None or svi_gdf.crs.is_geographic:
        points_gdf = svi_gdf.to_crs(3857)
        lines_gdf  = street_gdf.to_crs(3857)

        # 1) Build STRtree on lines once
        line_geoms = lines_gdf.geometry.values
        tree = STRtree(line_geoms)

        # Use a stable line identifier
        line_ids_arr = lines_gdf.index.to_numpy()

        # 2) Nearest line for every point
        pt_geoms = points_gdf.geometry.values
        idx, dists = tree.query_nearest(pt_geoms, return_distance=True)
        idx_pt = idx[0,:]
        idx_ln = idx[1,:]

        # Align results to points_gdf’s index order
        n = len(points_gdf)
        nearest_line_idx = np.empty(n, dtype=np.int64)
        nearest_line_id = np.empty(n, dtype=line_ids_arr.dtype)
        nearest_dist_m = np.empty(n, dtype="float64")
        nearest_line_idx[idx_pt] = idx_ln
        nearest_line_id[idx_pt]  = line_ids_arr[idx_ln]
        nearest_dist_m[idx_pt] = dists

        # 3) Project each point onto its matched line to get arclength along the line
        #    This is O(N) in Python but uses shapely C-core per operation and is fast in practice
        proj_m = np.fromiter(
            (line_geoms[i].project(pt) for pt, i in zip(pt_geoms, nearest_line_idx)),
            dtype="float64", count=n
        )

        # 4) Bin by arclength along the line
        bins = (proj_m // bin_distance).astype(np.int64)

        # 5) Within each (line_id, bin) keep the point closest to the line (smallest distance)
        df = pd.DataFrame({
            "nearest_street_id": nearest_line_id,
            "bin": bins,
            "dist_m": nearest_dist_m,
        }, index=points_gdf.index)

        df = df[df['dist_m']<=proximity_distance]

        keep_idx = df.groupby(["nearest_street_id", "bin"])["dist_m"].idxmin()

        # 6) Return thinned points with metadata
        out = points_gdf.loc[keep_idx].copy()
        out["nearest_street_id"] = df.loc[keep_idx, "nearest_street_id"].values
        out["dist_to_line"]  = df.loc[keep_idx, "dist_m"].values
        out.attrs["bin_size_m"] = bin_distance
        svi_gdf = svi_gdf.merge(out[['image_id','nearest_street_id', 'dist_to_line']], on='image_id')

    return svi_gdf

def _default_class_colors():
    """Create a full color map for all classes, with sensible defaults for common labels."""
    num_classes = len(CLS_DICT_INT)
    cmap = cm.get_cmap('tab20', num_classes)
    class_colors = {cid: (np.array(cmap(i)[:3]) * 255).astype(int).tolist()
                    for i, cid in enumerate(sorted(CLS_DICT_INT.keys()))}
    # Preferred overrides
    class_colors[27] = [142, 202, 230]  # Sky
    class_colors[17] = [74, 87, 89]     # Building
    class_colors[30] = [96, 108, 56]    # Vegetation / Greenery
    class_colors[55] = [252, 163, 17]   # Car
    return class_colors

def _colorize_mask(mask: np.ndarray, class_colors: dict) -> np.ndarray:
    """Convert a 2D segmentation mask to an RGB image using class_colors {class_id: [r,g,b]}."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Vectorized painting per class id
    for cid, color in class_colors.items():
        rgb[mask == cid] = color
    return rgb

def _compute_percentages(seg_mask: np.ndarray, ids_of_interest=None, min_pct=0.0, top_k=None):
    """
    Returns list of tuples (class_id, pct_float) sorted by descending percentage.
    ids_of_interest can be None for all present classes.
    min_pct filters out tiny classes.
    top_k limits the number of rows returned.
    """
    total = seg_mask.size
    unique, counts = np.unique(seg_mask, return_counts=True)
    pct_map = {int(k): 100.0 * c / total for k, c in zip(unique, counts)}

    # restrict to interest ids if provided
    items = pct_map.items()
    if ids_of_interest is not None:
        ids_set = set(int(i) for i in ids_of_interest)
        items = [(cid, pct) for cid, pct in items if cid in ids_set]

    # filter by min pct
    rows = [(cid, pct) for cid, pct in items if pct >= min_pct]

    # sort and cap
    rows.sort(key=lambda x: -x[1])
    if top_k is not None:
        rows = rows[:top_k]
    return rows

def _rgb255_to_hex(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))


def display_segmentation_grid(
    image_gdf,
    processor,
    model,
    device,
    rows=3,
    cols=4,
    id_col='image_id',
    mapillary_token_env='MAPILLARY_API_TOKEN',
    classes_to_color=['Sky', 'Building', 'Vegetation', 'Car'],       # list of labels for figure legend and also default ids_of_interest
    extra_color_overrides={'Sky': '#8ecae6',
                            'Building': '#4a5759',
                            'Vegetation': '#606c38',   # maps to Vegetation
                            'Car': '#ffc8dd',
                            },  # dict label -> hex or [r,g,b]
    title_fn=None,
    save_path=None,
    # New controls for per subplot percentage box
    per_subplot_min_pct=1.0,     # only show classes with percentage at least this value
    per_subplot_top_k=None,      # limit number of lines shown per subplot
    per_subplot_loc='lower left',# legend location within each segmentation axis
    per_subplot_fontsize=9,      # text size
    per_subplot_box=True         # add a small frame so text is readable
):
    assert cols % 2 == 0, "cols must be even since each image uses two columns"
    n_pairs = rows * (cols // 2)
    image_ids = list(image_gdf[id_col].values)[:n_pairs]

    # colors
    class_colors = _default_class_colors()

    def _to_rgb255(c):
        if isinstance(c, str) and c.startswith('#'):
            c = c.lstrip('#')
            return [int(c[i:i+2], 16) for i in (0, 2, 4)]
        if isinstance(c, (list, tuple)) and len(c) == 3:
            return list(map(int, c))
        raise ValueError("Color must be hex '#RRGGBB' or [r,g,b]")

    selected_labels = list(classes_to_color) if classes_to_color else []
    if extra_color_overrides:
        for label, color in extra_color_overrides.items():
            if label in LABEL_TO_ID_INT:
                class_colors[LABEL_TO_ID_INT[label]] = _to_rgb255(color)

    # figure level legend for selected labels
    legend_patches = []
    for label in selected_labels:
        if label in LABEL_TO_ID_INT:
            cid = LABEL_TO_ID_INT[label]
            legend_patches.append(
                mpatches.Patch(color=_rgb255_to_hex(class_colors[cid]), label=label)
            )

    fig_h = rows * 3.2
    fig_w = 16
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.02, wspace=0.05)

    token = os.environ.get(mapillary_token_env, None)
    if token is None:
        raise RuntimeError(f"Set {mapillary_token_env} in your environment")

    def _load_image(image_id):
        url = f'https://graph.mapillary.com/{image_id}?fields=thumb_2048_url'
        data = requests.get(url, params={'access_token': token}).json()
        im_url = data['thumb_2048_url']
        return Image.open(requests.get(im_url, stream=True).raw).convert('RGB')

    # ids of interest for percentage box
    ids_of_interest = None
    if selected_labels:
        ids_of_interest = [LABEL_TO_ID_INT[lbl] for lbl in selected_labels if lbl in LABEL_TO_ID_INT]

    for i, image_id in enumerate(image_ids):
        row = i // (cols // 2)
        col = (i % (cols // 2)) * 2

        im = _load_image(image_id)

        inputs = processor(images=im, return_tensors="pt")
        with torch.no_grad():
            pixel_values = inputs['pixel_values'].to(device)
            pixel_mask = inputs.get('pixel_mask')
            if pixel_mask is not None:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask.to(device))
            else:
                outputs = model(pixel_values=pixel_values)

        target_size = [im.size[::-1]]
        seg = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_size
        )[0].to('cpu').numpy()
        seg_rgb = _colorize_mask(seg, class_colors)

        # original subplot
        ax0 = fig.add_subplot(gs[row, col])
        ax0.imshow(im)
        ax0.axis('off')
        if title_fn:
            ax0.set_title(str(title_fn(i, image_id, image_gdf.iloc[i] if i < len(image_gdf) else None)), fontsize=10)
        else:
            ax0.set_title(f'Original {i+1}', fontsize=10)

        # segmentation subplot
        ax1 = fig.add_subplot(gs[row, col + 1])
        ax1.imshow(seg_rgb, interpolation='nearest')
        ax1.axis('off')
        ax1.set_title('Segmented', fontsize=10)

        # compute and draw per subplot percentages
        rows_pct = _compute_percentages(
            seg_mask=seg,
            ids_of_interest=ids_of_interest,      # None means all classes present
            min_pct=per_subplot_min_pct,
            top_k=per_subplot_top_k
        )

        if rows_pct:
            handles = []
            for cid, pct in rows_pct:
                label = CLS_DICT_INT.get(cid, str(cid))
                color_hex = _rgb255_to_hex(class_colors.get(cid, [200, 200, 200]))
                handles.append(mpatches.Patch(color=color_hex, label=f"{label} {pct:.1f}%"))

            leg = ax1.legend(handles=handles,
                             loc=per_subplot_loc,
                             fontsize=per_subplot_fontsize,
                             frameon=per_subplot_box,
                             handlelength=0.8,
                             borderpad=0.3,
                             labelspacing=0.3)
            if per_subplot_box and leg is not None:
                leg.get_frame().set_alpha(0.75)

    # figure level legend
    if legend_patches:
        leg = fig.legend(handles=legend_patches, loc='upper center', handlelength=1,
                         bbox_to_anchor=(0.5, 0.93), ncol=min(len(legend_patches), 6),
                         frameon=False, prop={'size': 11})
        for patch in leg.get_patches():
            patch.set_height(10)
            patch.set_y(-0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=180)
    plt.show()


def fill_attributes_by_graph(
    gdf,
    aggr_cols,
    restrict_within_street=False,
    weight_col=None,
    inplace=False,
):

    if not inplace:
        gdf = gdf.copy()

    # Build MultiGraph
    G = nx.MultiGraph()
    for idx, row in gdf.iterrows():
        u, v = row["u"], row["v"]
        data = {"idx": idx}
        if weight_col is not None and (weight_col in gdf.columns) and pd.notna(row.get(weight_col)):
            data[weight_col] = row[weight_col]
        G.add_edge(u, v, key=idx, **data)

    def _incident_edge_indices(node):
        if node not in G:
            return []
        idxs = []
        for _, keyed in G[node].items():
            for _, d in keyed.items():
                idxs.append(d["idx"])
        return idxs

    def _nearest_seed_values(seed_values_per_node):
        import heapq
        dist = {n: np.inf for n in G.nodes}
        val = {n: np.nan for n in G.nodes}
        h = []
        for n, sv in seed_values_per_node.items():
            if not np.isnan(sv):
                dist[n] = 0.0
                val[n] = sv
                heapq.heappush(h, (0.0, n, sv))

        def _edge_weight(n, nbr):
            if weight_col is None:
                return 1.0
            weights = [d.get(weight_col, 1.0) for _, d in G[n][nbr].items()]
            return min(weights) if weights else 1.0

        visited = set()
        while h:
            d, n, sv = heapq.heappop(h)
            if n in visited:
                continue
            visited.add(n)
            dist[n] = d
            val[n] = sv
            for nbr in G.neighbors(n):
                if nbr in visited:
                    continue
                nd = d + _edge_weight(n, nbr)
                heapq.heappush(h, (nd, nbr, sv))
        return val, dist

    summary = {}
    street_groups = [("ALL", gdf.index)]
    if restrict_within_street:
        street_groups = list(gdf.groupby("street_id").groups.items())

    for col in aggr_cols:

        filled_count_total = 0
        for _, idxs in street_groups:
            rows_idx = pd.Index(idxs)

            nodes_in_group = set(gdf.loc[rows_idx, "u"]).union(set(gdf.loc[rows_idx, "v"]))

            node_seeds = {}
            for n in nodes_in_group:
                vals = []
                for eidx in _incident_edge_indices(n):
                    if eidx in rows_idx:
                        v = gdf.at[eidx, col]
                        if pd.notna(v):
                            vals.append(v)
                node_seeds[n] = np.mean(vals) if len(vals) else np.nan

            if not any(pd.notna(v) for v in node_seeds.values()):
                continue

            node_val, _ = _nearest_seed_values(node_seeds)

            # ✅ Correct row boolean indexing
            missing_mask = gdf.loc[rows_idx, col].isna()
            if not missing_mask.any():
                continue
            eidxs_to_fill = missing_mask.index[missing_mask.values]

            for eidx in eidxs_to_fill:
                u = gdf.at[eidx, "u"]
                v = gdf.at[eidx, "v"]
                candidates = [node_val.get(u, np.nan), node_val.get(v, np.nan)]
                candidates = [c for c in candidates if pd.notna(c)]
                if candidates:
                    gdf.at[eidx, col] = float(np.mean(candidates))
                    filled_count_total += 1


    return gdf

def fill_street_network(image_gdf, street_gdf):

    image_gdf_mod = image_gdf.copy()
    street_gdf_mod = street_gdf.copy()


    # Get target columns
    aggr_cols = ['nearest_street_id'] + list(image_gdf_mod.columns[9:])

    # Get mean of each street
    image_gdf_grouped = image_gdf_mod[aggr_cols].groupby(['nearest_street_id']).mean().reset_index()
    image_gdf_grouped = image_gdf_grouped.rename(columns={'nearest_street_id':'street_id'})

    # Merge to street
    street_gdf_mod = street_gdf_mod.merge(image_gdf_grouped, how='left', on = 'street_id')

    filled_gdf = fill_attributes_by_graph(street_gdf_mod,
                                        aggr_cols=aggr_cols[1:],
                                        restrict_within_street=False,  # set True to keep propagation within each street_id
                                        weight_col=None,               # or e.g. "length_m" if available
                                        inplace=False
                                        )
    return filled_gdf