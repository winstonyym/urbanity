# import base packages
import os
import json
import glob
import zipfile
import requests
import pkg_resources
from io import BytesIO

# import external packages
import rasterio
from rasterio import features
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.io import MemoryFile

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from scipy.stats import skew, kurtosis


# import package functions and classes
from .utils import get_population_data_links, get_available_pop_countries


def get_population_data(country: str, bounding_poly = None, all_only=False):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    Due to raster band sparsity for recent time periods, prioritises construction of indicators 
    from .csv file over .tif. Values are rounded to integers for ease of intepretation and for 
    convenient raster plotting.

    Args:
        country (str): Country name
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile overlap for large countries
        all_only (bool): If True, only computes total population counts. Defaults to False. 

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    
    general_pop_dict = get_population_data_links(country)

    try:
        general_pop_dict[country]
    except KeyError:
        print(f"Population data is not available for {country}. Please use utils.get_available_pop_countries function to view which countries are available.")
    minx, miny, maxx, maxy = bounding_poly.bounds['minx'].min(), bounding_poly.bounds['miny'].min(), bounding_poly.bounds['maxx'].max(), bounding_poly.bounds['maxy'].max()
    
    if all_only: 
        data_link = general_pop_dict[country][f'pop_csv_url']
        pop_df = pd.read_csv(data_link)
        lat_name = [col for col in list(pop_df.columns) if 'at' in col][0]
        lon_name = [col for col in list(pop_df.columns) if 'on' in col][0]
        pop_name  = list(pop_df.columns)[-1]
        pop_df = pop_df[(pop_df[lat_name] >= miny) & (pop_df[lat_name] <= maxy) & (pop_df[lon_name] >= minx) & (pop_df[lon_name] <= maxx)]
        
        pop_gdf = gpd.GeoDataFrame(
            data={pop_name:pop_df[pop_name]}, 
            crs = 'epsg:4326',
            geometry = gpd.points_from_xy(pop_df[lon_name], pop_df[lat_name]))

        return pop_gdf, pop_name
    
    pop_groups = ['pop', 'men', 'women','elderly','youth','children'] 

    groups_df = []
    target_cols = []

    for group in pop_groups:
        data_link = general_pop_dict[country][f'{group}_csv_url']
        pop_df = pd.read_csv(data_link)
        lat_name = [col for col in list(pop_df.columns) if 'at' in col][0]
        lon_name = [col for col in list(pop_df.columns) if 'on' in col][0]
        pop_name  = list(pop_df.columns)[-1]
        pop_df = pop_df[(pop_df[lat_name] >= miny) & (pop_df[lat_name] <= maxy) & (pop_df[lon_name] >= minx) & (pop_df[lon_name] <= maxx)]
        
        pop_gdf = gpd.GeoDataFrame(
            data={pop_name:pop_df[pop_name]}, 
            crs = 'epsg:4326',
            geometry = gpd.points_from_xy(pop_df[lon_name], pop_df[lat_name]))
        groups_df.append(pop_gdf)
        target_cols.append(pop_name)

    return groups_df, target_cols
        
    
def get_tiled_population_data(country: str, bounding_poly = None, all_only = False):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    For large areas, population data are pre-processed into tiles to allow quick extraction across 
    multiple geographic scales. This operation is not implemented for .tif format. 

    Args:
        country (str): Country name
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile overlap for large countries
        all_only (bool): If True, only computes total population counts. Defaults to False. 

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    groups_df = []
    target_cols = []

    general_pop_dict = get_population_data_links(country)
    minx, miny, maxx, maxy = bounding_poly.bounds['minx'].min(), bounding_poly.bounds['miny'].min(), bounding_poly.bounds['maxx'].max(), bounding_poly.bounds['maxy'].max()

    tile_polygon_path = pkg_resources.resource_filename('urbanity', 'map_data/tiled_data.json')
    with open(tile_polygon_path, 'r') as f:
        tile_dict = json.load(f)

    tile_polygons = gpd.read_file(tile_dict[f'{country}_tile.geojson'])
    res_intersection = bounding_poly.overlay(tile_polygons, how='intersection')
    target_tiles = list(np.unique(res_intersection['TILEID']))

    # Extract point data for target tiles
    point_df = pd.DataFrame()
    for tile in target_tiles:
        data_link = general_pop_dict[country][f'tile_all_{tile}.parquet']
        tile_df = pd.read_parquet(data_link)
        tile_df = tile_df[(tile_df['latitude'] >= miny) & (tile_df['latitude'] <= maxy) & (tile_df['longitude'] >= minx) & (tile_df['longitude'] <= maxx)]
        point_df = pd.concat([point_df, tile_df])
    point_df['group'] = point_df['group'].replace({'men_1':'men', 'men_2':'men', 'women_1':'women', 'women_2': 'women',
                                    'pop_1': 'pop', 'pop_2':'pop', 'pop_3':'pop', 'pop_4': 'pop', 'pop_5':'pop', 'pop_6':'pop',
                                    'Total':'pop', 'Youth':'youth','Men':'men', 'Women':'women', 'Elderly':'elderly','Children':'children'})
    point_df = point_df.rename(columns={"value": "population"})
    points_gdf = gpd.GeoDataFrame(data=point_df[['population','group']], crs='epsg:4326', geometry=gpd.points_from_xy(point_df['longitude'],point_df['latitude']))
    if all_only:
        pop_gdf = points_gdf[points_gdf['group'] == 'pop']
        target_col = 'population'
        return pop_gdf, target_col

    for group in ['pop','men','women', 'elderly','youth','children']:
        groups_df.append(points_gdf[points_gdf['group'] == group])
        target_cols.append('population')
    return groups_df, target_cols

def raster2gdf(raster_path, chosen_band = 1, get_grid=True, zoom=False, boundary=None, same_geometry=False):
    """Helper function to convert a raster dataset (.tiff) to GeoDataFrame object.

    Args:
        raster_path (str): Filepath to a raster dataset (.tiff)
        chosen_band (int, optional): Selection of raster channel. Defaults to 1.
        get_grid (bool, optional): If True, returns gridded GeoDataFrame, otherwise returns center points. Defaults to True.
        zoom (bool, optional): If True, reads only a windowed portion of large .tiff file. Requires `boundary` to be provided. Defaults to False. 
        boundary (gpd.GeoDataFrame, optional): Geographic boundary to specify window extent for raster filtering. Defaults to None. 
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of points coordinates that correspond to original raster tile (centroid) positions.
    """    
    
    try:
        raster_dataset = rasterio.open(raster_path)
    except:
        with rasterio.open(raster_path) as src:
            print(src.profile)

    if zoom and boundary is not None:
        xarray = np.linspace(raster_dataset.bounds[0], raster_dataset.bounds[2],raster_dataset.width+1)
        yarray = np.linspace(raster_dataset.bounds[3], raster_dataset.bounds[1],raster_dataset.height+1)
        half_x_grid_size = (raster_dataset.bounds[2] - raster_dataset.bounds[0])/(raster_dataset.width)/2
        half_y_grid_size = (raster_dataset.bounds[3] - raster_dataset.bounds[1])/(raster_dataset.height)/2

        # Get boundary coordinates
        xmin, ymin, xmax, ymax = boundary.total_bounds

        # Get raster offsets, width, and height
        list_x_range = [i for i in list(xarray) if i >= xmin-half_x_grid_size if i <=xmax+half_x_grid_size]
        list_y_range = [i for i in list(yarray) if i >= ymin-half_y_grid_size if i <=ymax+half_y_grid_size]
        x_width = len(list_x_range)
        y_width = len(list_y_range)
        x_off = np.where(xarray == list_x_range[0])[0].item()
        y_off = np.where(yarray == list_y_range[0])[0].item()
        
        if same_geometry: 
            # Read windowed raster view
            with rasterio.open(raster_path) as src:
                value_array = src.read(1, window=Window(x_off, y_off, x_width-1, y_width-1))
            return pd.DataFrame({'value':np.ravel(value_array, order='C')})

        geom_list = []
        for y_i in range(len(list_y_range)-1):
            for x_i in range(len(list_x_range)-1): 
                geom_list.append(Polygon([(list_x_range[x_i], list_y_range[y_i]), 
                                        (list_x_range[x_i], list_y_range[y_i+1]), 
                                        (list_x_range[x_i+1], list_y_range[y_i+1]), 
                                        (list_x_range[x_i+1], list_y_range[y_i])]))

        # Read windowed raster view
        with rasterio.open(raster_path) as src:
            value_array = src.read(1, window=Window(x_off, y_off, x_width-1, y_width-1))
        
        mydf = pd.DataFrame({'value':np.ravel(value_array, order='C')})

        return gpd.GeoDataFrame(data = mydf, crs = raster_dataset.crs, geometry = geom_list)

    # Raster value array
    value_array = raster_dataset.read(chosen_band)
    
    if get_grid:
        xarray = np.linspace(raster_dataset.bounds[0], raster_dataset.bounds[2],raster_dataset.width+1)
        yarray = np.linspace(raster_dataset.bounds[3], raster_dataset.bounds[1],raster_dataset.height+1)
        
        mydf = pd.DataFrame({'value':np.ravel(value_array, order='C')})
        geom_list = []
        for y_i in range(len(yarray)-1):
            for x_i in range(len(xarray)-1): 
                geom_list.append(Polygon([(xarray[x_i], yarray[y_i]), (xarray[x_i], yarray[y_i+1]), (xarray[x_i+1], yarray[y_i+1]), (xarray[x_i+1], yarray[y_i])]))
        gdf = gpd.GeoDataFrame(data = mydf, crs = raster_dataset.crs, geometry = geom_list)
    else:
        # Get longitudinal half-grid size
        half_x_grid_size = (raster_dataset.bounds[2] - raster_dataset.bounds[0])/(raster_dataset.width)/2
        half_y_grid_size = (raster_dataset.bounds[3] - raster_dataset.bounds[1])/(raster_dataset.height)/2

        # Get longitudinal intervals
        xarray = np.linspace(raster_dataset.bounds[0]+half_x_grid_size, raster_dataset.bounds[2]-half_x_grid_size,raster_dataset.width)
        yarray = np.linspace(raster_dataset.bounds[3]-half_y_grid_size, raster_dataset.bounds[1]+half_y_grid_size,raster_dataset.height)

        # Obtain full list of coordinates
        lat_coords = np.repeat(yarray,raster_dataset.width)
        long_coords = np.tile(xarray, raster_dataset.height)

        # Option C - Changing from last index dimension first 
        gdf = gpd.GeoDataFrame(data = {'value':np.ravel(value_array, order='C')}, crs = raster_dataset.crs, geometry = gpd.points_from_xy(x=long_coords, y=lat_coords))
        
    return gdf

def csv2grid(filepath, lat_col = 'Y', lon_col = 'X'):
    """Helper function to convert a dataset with latitude and longitude points (.csv) to GeoDataFrame object.

    Args:
        filepath (str): Filepath to a local csv file.
        lat_col (str, optional): Column that corresponds to Latitude. Defaults to 'Y'.
        lon_col (str, optional): Column that corresponds to Longitude. Defaults to 'X'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of gridded polygons that correspond to original latitude and longitude locations. 
    """    
    
    df = pd.read_csv(filepath)
    x_list = np.sort(df[lon_col].unique())
    y_list = np.sort(df[lat_col].unique())
    half_x_grid_size = (x_list[1] - x_list[0])/2
    half_y_grid_size = (y_list[1] - y_list[0])/2
    
    geom_list = []
    for x_i, y_i in zip(df[lon_col], df[lat_col]):
        geom_list.append(Polygon([(x_i-half_x_grid_size, y_i-half_y_grid_size), 
                                  (x_i-half_x_grid_size, y_i+half_y_grid_size), 
                                  (x_i+half_x_grid_size, y_i+half_y_grid_size), 
                                  (x_i+half_x_grid_size, y_i-half_y_grid_size)]))
        
    gdf = gpd.GeoDataFrame(data = df, crs = 'epsg:4326', geometry = geom_list)
    
    return gdf


def extract_tiff_from_shapefile(geotiff_path, shapefile, output_filepath=None, zipped=True):

    if zipped:
        with zipfile.ZipFile(geotiff_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(geotiff_path))

        os.remove(geotiff_path)
        geotiff_path = geotiff_path[:-4]

    # Get geotiff info and mask with shapefile
    with rasterio.open(geotiff_path, 'r') as data:
        out_image, out_transform = rasterio.mask.mask(data, shapes=[shapefile.geometry[0]], crop=True)
        out_meta = data.meta
  

    # Set transformation of geotiff
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    # If output_filepath, write to disk
    if output_filepath:
        with rasterio.open(output_filepath, "w", **out_meta) as dest:
            dest.write(out_image)
    else:
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_image)
                
                # Read the data from the in-memory dataset
                data_array = dataset.read(1)
                data_array = np.nan_to_num(data_array, nan=0)
                
        return data_array
    

def get_population_tif_from_coords(pop_folder, subgroup, lat, long):
    tif_files = glob.glob(os.path.join(pop_folder, '*'))
    
    # Get intervals
    if subgroup=='population':
        lat_intervals = [int(os.path.basename(path).split('_')[2]) for path in tif_files if subgroup in path]
        long_intervals = [int(os.path.basename(path).split('_')[3]) for path in tif_files if subgroup in path]
    else:   
        lat_intervals = [int(os.path.basename(path).split('_')[2]) for path in tif_files if subgroup in path]
        long_intervals = [int(os.path.basename(path).split('_')[4]) for path in tif_files if subgroup in path]

    snapped_lat = max((x for x in lat_intervals if x <= lat), default=None)
    snapped_long = max((x for x in long_intervals if x <= long), default=None)

    targets = [subgroup, str(snapped_lat), str(snapped_long)]

    if subgroup == 'men':
        for path in tif_files:
            if all(x in path for x in targets) and ('women' not in path):
                return path
    else:
        for path in tif_files:
            if all(x in path for x in targets):
                return path
            

def load_npz_as_raster(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    mosaic = data['mosaic']
    meta = data['meta'].item()  # Convert from structured array to dict if needed
    
    return mosaic, meta

def mask_raster_with_gdf(gdf, raster, meta):
    gdf_proj = gdf.copy()

    if gdf.crs != meta['crs']:
        gdf_proj = gdf.copy()
        gdf_proj = gdf_proj.to_crs(meta['crs'])

    # Raster stats
    geom = gdf_proj[['geometry','plot_id']].values.tolist()

    fields_rasterized = features.rasterize(geom, 
                                       out_shape=raster[0].shape, 
                                       transform=meta['transform'])
    
    categories = np.unique(fields_rasterized)

    # Dictionary to store the average values for each category
    statistics = {
        'canopy_mean':{},
        'canopy_stdev':{},
        'canopy_skewness':{},
        'canopy_kurtosis':{}
    }

    # Calculate the average for each category
    for category in categories:
        if category != 0:  # Assuming 0 is the background/no data
            mask = (fields_rasterized == category)

            average = raster[0][mask].mean() 
            statistics['canopy_mean'][category] = average

            stdev = raster[0][mask].std() 
            statistics['canopy_stdev'][category] = stdev

            skewness = skew(raster[0][mask])
            statistics['canopy_skewness'][category] = skewness

            kurtosisness = kurtosis(raster[0][mask])
            statistics['canopy_kurtosis'][category] = kurtosisness

    canopy_df = pd.DataFrame.from_dict({k:v for k,v in statistics.items()})
    canopy_df['plot_id'] = canopy_df.index

    return canopy_df
