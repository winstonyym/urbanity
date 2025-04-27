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
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio import features
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.io import MemoryFile

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Polygon
from scipy.stats import skew, kurtosis

from urbanity.geom import buffer_polygon

# import package functions and classes
from .utils import get_population_data_links, get_available_pop_countries
from .satellite import get_affine_transform

def get_ghs_population_data(bounding_poly, 
                            bandwidth=100, 
                            temporal_years = list(range(1975, 2030, 5))):
    """Extract GHS Population Density at 3 arcsecond resolution from spatial boundary file.  
    Args:
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile extent.
        bandwidth (int): Distance to extract information beyond network. Defaults to 100.
        temporal_years (list): Specify the yearly interval to obtain GHS_POP_GLOBE land cover data. 

    Returns:
        gpd.DataFrame: GeoDataFrame in gridded format with population information
    """    

    # Get buffered polygon boudns
    original_bbox = bounding_poly.geometry[0]
    buffered_tp = bounding_poly.copy()
    buffered_tp['geometry'] = buffer_polygon(bounding_poly, bandwidth=bandwidth)
    buffered_bbox = buffered_tp.geometry.values[0]

    # Extract GHS population data
    ghs_global_grid_path = pkg_resources.resource_filename('urbanity', 'ghs_data/global_ghs_grid.parquet')
    ghs_global_grid = gpd.read_parquet(ghs_global_grid_path)
    overlapping_grid = ghs_global_grid.overlay(buffered_tp)
    buffered_tp['geometry'] = buffer_polygon(bounding_poly, bandwidth=bandwidth+500)

    # Loop through each year and obtain tif file
    origin_gdf_pop = gpd.GeoDataFrame()

    for k, year in enumerate(temporal_years):
        # Only compute geometry once
        if k == 0:
            # If only one tile
            if len(overlapping_grid) == 1:
                row, col = overlapping_grid['row'].values.item(), overlapping_grid['col'].values.item()
                target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row}_C{col}.zip"
                raster_dataset_pop = download_tiff_from_path(target_tif_pop)
                raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=False)

            elif len(overlapping_grid) > 1: 
                raster_list_pop = []
                for i, row in overlapping_grid.iterrows():
                    
                    target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"                           
                    raster_dataset_pop = download_tiff_from_path(target_tif_pop)
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

                raster_dataset_pop = download_tiff_from_path(target_tif_pop)
                raster_gdf_pop = raster2gdf(raster_dataset_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

            elif len(overlapping_grid) > 1: 
                raster_list_pop = []

                for i, row in overlapping_grid.iterrows():
                    target_tif_pop = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E{year}_GLOBE_R2023A_4326_3ss_V1_0_R{row['row']}_C{row['col']}.zip"
                    raster_dataset_pop = download_tiff_from_path(target_tif_pop)
                    raster_list_pop.append(raster_dataset_pop)

                # Merge rasters
                mosaic_pop = merge_raster_list(raster_list_pop)
                raster_gdf_pop = raster2gdf(mosaic_pop, zoom=True, boundary = buffered_tp, same_geometry=True)

            raster_gdf_pop.columns = [str(year)]
            origin_gdf_pop  = gpd.GeoDataFrame(pd.concat([origin_gdf_pop, raster_gdf_pop], axis=1)) 

    temporal_rename = {str(i):f'{i}_pop' for i in temporal_years}
    origin_gdf_pop = origin_gdf_pop.rename(columns=temporal_rename)
    return origin_gdf_pop



def get_meta_population_data(country: str, bounding_poly = None, all_only=False):
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
        meta_file (dict): Contains meta information of .tiff file.
        chosen_band (int, optional): Selection of raster channel. Defaults to 1.
        get_grid (bool, optional): If True, returns gridded GeoDataFrame, otherwise returns center points. Defaults to True.
        zoom (bool, optional): If True, reads only a windowed portion of large .tiff file. Requires `boundary` to be provided. Defaults to False. 
        boundary (gpd.GeoDataFrame, optional): Geographic boundary to specify window extent for raster filtering. Defaults to None. 
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of points coordinates that correspond to original raster tile (centroid) positions.
    """    
    
    # Check if raster file is already open
    if isinstance(raster_path, rasterio.io.DatasetReader):
        raster_dataset = raster_path
    elif isinstance(raster_path, tuple):
        value_array, bounds, raster_dataset = raster_path
    else: 
        try:
            raster_dataset = rasterio.open(raster_path)
        except:
            with rasterio.open(raster_path) as src:
                print(src.profile)

    if zoom and boundary is not None:
        if isinstance(raster_path, tuple):
            xarray = np.linspace(bounds[0], bounds[2],value_array.shape[2]+1)
            yarray = np.linspace(bounds[3],bounds[1],value_array.shape[1]+1)
            half_x_grid_size = (bounds[2] - bounds[0])/(value_array.shape[2])/2
            half_y_grid_size = (bounds[3] - bounds[1])/(value_array.shape[1])/2
        else:
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
            if isinstance(raster_path, rasterio.io.DatasetReader):
                value_array = raster_path.read(1, window=Window(x_off, y_off, x_width-1, y_width-1))
            elif isinstance(raster_path, tuple):
                value_array = raster_path[0][0][y_off:(y_off + y_width-1), x_off:(x_off + x_width-1)]
            else:
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
        if isinstance(raster_path, rasterio.io.DatasetReader):
            value_array = raster_path.read(1, window=Window(x_off, y_off, x_width-1, y_width-1))
        elif isinstance(raster_path, tuple):
            value_array = raster_path[0][0][y_off:(y_off + y_width-1), x_off:(x_off + x_width-1)]
        else: 
            with rasterio.open(raster_path) as src:
                value_array = src.read(1, window=Window(x_off, y_off, x_width-1, y_width-1))
        
        mydf = pd.DataFrame({'value':np.ravel(value_array, order='C')})
        if isinstance(raster_path, tuple):
            mygdf = gpd.GeoDataFrame(data = mydf, crs = raster_path[2], geometry = geom_list)
        else:
            mygdf = gpd.GeoDataFrame(data = mydf, crs = raster_dataset.crs, geometry = geom_list)

        return mygdf
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
    
def is_bounding_box_within(bounds, box):
    """
    Check if the bounding box is within the raster bounds.

    Parameters:
    bounds: Tuple[float, float, float, float] - (min_x, min_y, max_x, max_y)
    box: Tuple[float, float, float, float] - (xmin, ymin, xmax, ymax)

    Returns:
    bool - True if box is within bounds, False otherwise
    """
    min_x, min_y, max_x, max_y = bounds
    xmin, ymin, xmax, ymax = box
    
    return (xmin >= min_x and 
            ymin >= min_y and 
            xmax <= max_x and 
            ymax <= max_y)

def is_overlapped_with_bounding_box(bounds, box):
    """
    Check if there is any spatial overlap between the bounding box and the raster bounds.

    Parameters:
    bounds: Tuple[float, float, float, float] - (min_x, min_y, max_x, max_y)
    box: Tuple[float, float, float, float] - (xmin, ymin, xmax, ymax)

    Returns:
    bool - True if there is any overlap between box and bounds, False otherwise
    """
    min_x, min_y, max_x, max_y = bounds
    xmin, ymin, xmax, ymax = box
    
    return not (xmax < min_x or 
                xmin > max_x or 
                ymax < min_y or 
                ymin > max_y)

def find_valid_tif_files(filepaths, bounding_box):
    """
    Open raster TIFF files and check if bounding box falls within the raster bounds.

    Parameters:
    filepaths: List[str] - List of TIFF file paths
    bounding_box: Tuple[float, float, float, float] - (xmin, ymin, xmax, ymax)

    Returns:
    List[str] - List of valid file paths where the bounding box is within the raster
    """
    valid_files = []

    for filepath in filepaths:
        with rasterio.open(filepath) as src:
            # Get the bounds of the raster
            bounds = src.bounds
            
            # Check if the bounding box is within the raster bounds
            if is_overlapped_with_bounding_box(bounds, bounding_box):
                valid_files.append(filepath)

    return valid_files

def get_population_tif_from_coords(pop_folder, subgroup, long_min, lat_min, long_max, lat_max):
    tif_files = glob.glob(os.path.join(pop_folder, '*'))
    
    # Get intervals
    if subgroup=='population':
        lat_intervals = np.unique([int(os.path.basename(path).split('_')[2]) for path in tif_files if subgroup in path])
        long_intervals = np.unique([int(os.path.basename(path).split('_')[3]) for path in tif_files if subgroup in path])
    else:   
        lat_intervals = np.unique([int(os.path.basename(path).split('_')[2]) for path in tif_files if subgroup in path])
        long_intervals = np.unique([int(os.path.basename(path).split('_')[4]) for path in tif_files if subgroup in path])

    # Check which rasters are needed
    snapped_lat_min = max((x for x in lat_intervals if x <= lat_min), default=None)
    snapped_long_min = max((x for x in long_intervals if x <= long_min), default=None)
    snapped_lat_max = max((x for x in lat_intervals if x <= lat_max), default=None)
    snapped_long_max = max((x for x in long_intervals if x <= long_max), default=None)

    targets = [subgroup, str(snapped_lat_min), str(snapped_long_min)]
    
    if (snapped_lat_min != snapped_lat_max) or (snapped_lat_max != snapped_long_max):
        targets2 = [subgroup, str(snapped_lat_max), str(snapped_long_max)]

        path_list = []
        if subgroup == 'men':

            for path in tif_files:
                if all(x in path for x in targets) and ('women' not in path):
                    path_list.append(path)
                if all(x in path for x in targets2) and ('women' not in path):
                    path_list.append(path)
            else:
                for path in tif_files:
                    if all(x in path for x in targets):
                        path_list.append(path)
                    if all(x in path for x in targets2):
                        path_list.append(path)
                    
    path_list = []
    if subgroup == 'men':
        for path in tif_files:
            if all(x in path for x in targets) and ('women' not in path):
                path_list.append(path)
    else:
        for path in tif_files:
            if all(x in path for x in targets):
                path_list.append(path)
                
    return path_list
            

def load_npz_as_raster(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    mosaic = data['mosaic']
    meta = data['meta'].item()  # Convert from structured array to dict if needed
    
    return mosaic, meta

def split_raster_into_grids(raster):
    # Get the dimensions of the original raster
    rows, cols = raster.shape
    
    # Calculate the dimensions of the smaller grids
    mid_row = rows // 4
    mid_col = cols // 4
    
    # Create a list to hold the 16 smaller grids
    grids = []

    # Slice the original raster into 16 smaller grids
    for i in range(4):  # 4 rows
        for j in range(4):  # 4 columns
            start_row = i * mid_row
            start_col = j * mid_col
            end_row = start_row + mid_row if i < 3 else rows  # Handle last slice
            end_col = start_col + mid_col if j < 3 else cols  # Handle last slice
            grid = raster[start_row:end_row, start_col:end_col]
            grids.append(grid)
    
    return grids

def subset_raster_from_coords(gdf, xarray, yarray, safety_value, raster_category, raster_values, category):
    xmin, ymin, xmax, ymax = gdf.iloc[[category]].total_bounds

    transform = get_affine_transform(gdf, raster_values)

    # ------------------------------------------------------------------
    # 3. Work out target shape and transform
    # ------------------------------------------------------------------
    if raster_values.ndim == 3:            # (bands, rows, cols)
        raster_values = raster_values[:,:,0]

    # Get raster offsets, width, and height
    list_x_range = [i for i in list(xarray) if i >= xmin-safety_value if i <=xmax+safety_value]
    list_y_range = [i for i in list(yarray) if i >= ymin-safety_value if i <=ymax+safety_value]
    x_width = len(list_x_range)
    y_width = len(list_y_range)
    x_off = np.where(xarray == list_x_range[0])[0].item()
    y_off = np.where(yarray == list_y_range[0])[0].item()

    raster_values_subset = raster_values[y_off:(y_off+y_width), x_off:(x_off+x_width)]
    raster_category_subset = raster_category[y_off:(y_off+y_width), x_off:(x_off+x_width)]

    if (raster_category_subset.shape[0] > 10000) or (raster_category_subset.shape[1] > 10000):
        print('Splitting large subset into sections.')

        raster_category_splits = split_raster_into_grids(raster_category_subset)
        raster_value_splits = split_raster_into_grids(raster_values_subset)
        
        total_value = 0
        total_len = 0
        for value_array, category_array in zip(raster_value_splits,raster_category_splits):
            mask = (category_array==category)
            values = value_array[mask]
            total_len += len(values)
            total_value += (values.sum())

        average = total_value / total_len

        total_deviation = 0
        total_skewness = 0
        for value_array, category_array in zip(raster_value_splits,raster_category_splits):
            mask = (category_array==category)
            values = value_array[mask]
            total_deviation += np.sum(np.square(values - average))
            total_skewness += np.sum(np.power(values - average, 3))

        stdev = np.sqrt(total_deviation / total_len)
        skewness = total_skewness / ((total_len - 1) * stdev)
        kurtosisness = average / stdev

    else: 
        mask = (raster_category_subset==category)
        average = raster_values_subset[mask].mean() 
        stdev = raster_values_subset[mask].std() 
        skewness = skew(raster_values_subset[mask])
        kurtosisness = kurtosis(raster_values_subset[mask])

    return average, stdev, skewness, kurtosisness


def mask_raster_with_gdf_large_raster(gdf, raster, tile_size=512):
    gdf_proj = gdf.copy()

    gdf_proj = gdf_proj.to_crs('epsg:4326')

    if raster.ndim == 3:
        raster = raster[:,:,0]

    # Prepare output raster
    output_shape = raster.shape
    output_raster = np.full(output_shape, fill_value=-1, dtype=np.int32)

    transform = get_affine_transform(gdf, raster)
    # Iterate over the raster in tiles
    for row in range(0, output_shape[0], tile_size):
        for col in range(0, output_shape[1], tile_size):
            # Define the tile bounds
            row_min = row
            row_max = min(row + tile_size, output_shape[0])
            col_min = col
            col_max = min(col + tile_size, output_shape[1])

            # Create a bounding box polygon for the current tile
            tile_bounds = box(
                transform[2] + col_min * transform[0],
                transform[5] + row_min * transform[4],
                transform[2] + col_max * transform[0],
                transform[5] + row_max * transform[4]
            )

            # Select geometries that intersect with the current tile
            geometries = gdf_proj[gdf_proj.geometry.intersects(tile_bounds)]

            # Rasterize the geometries for the current tile
            if not geometries.empty:
                geom = geometries[['geometry', 'plot_id']].values.tolist()
                tile_rasterized = features.rasterize(
                    geom,
                    out_shape=(row_max - row_min, col_max - col_min),
                    transform=transform,
                    fill=-1
                )
                # Combine the tile back into the output raster
                output_raster[row_min:row_max, col_min:col_max] = np.where(
                    tile_rasterized != -1, tile_rasterized, output_raster[row_min:row_max, col_min:col_max]
                )

    # Dictionary to store the average values for each category
    statistics = {
        'canopy_mean':{},
        'canopy_stdev':{},
        'canopy_skewness':{},
        'canopy_kurtosis':{}
    }

    if (raster[0].shape[0] > 10000) or (raster[0].shape[1] > 10000):
        print('Large raster: using raster subset')
        xarray = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2],raster[0].shape[1]+1)
        yarray = np.linspace(gdf.total_bounds[3], gdf.total_bounds[1],raster[0].shape[0]+1)
        safety_value = 0.001

        # Calculate the average for each category
        for category in gdf_proj.index:
            average, stdev, skewness, kurtosisness = subset_raster_from_coords(gdf, xarray, yarray, safety_value, output_raster, raster, category)

            statistics['canopy_mean'][category] = average
            statistics['canopy_stdev'][category] = stdev
            statistics['canopy_skewness'][category] = skewness
            statistics['canopy_kurtosis'][category] = kurtosisness
    else:
        for category in gdf_proj.index:
            mask = (output_raster == category)
            average = raster[mask].mean() 
            statistics['canopy_mean'][category] = average
            stdev = raster[mask].std() 
            statistics['canopy_stdev'][category] = stdev
            skewness = skew(raster[mask])
            statistics['canopy_skewness'][category] = skewness
            kurtosisness = kurtosis(raster[mask])
            statistics['canopy_kurtosis'][category] = kurtosisness

    canopy_df = pd.DataFrame.from_dict({k:v for k,v in statistics.items()})
    canopy_df['plot_id'] = canopy_df.index

    return canopy_df


def mask_raster_with_gdf(gdf, raster):
    gdf_proj = gdf.copy()
    gdf_proj = gdf_proj.to_crs('epsg:4326')

    transform = get_affine_transform(gdf, raster)

    if raster.ndim == 3:            # (bands, rows, cols)
        raster = raster[:,:,0]
    
    # Raster stats
    geom = gdf_proj[['geometry','plot_id']].values.tolist()

    fields_rasterized = features.rasterize(geom, 
                                       out_shape=raster.shape, 
                                       transform=transform,
                                       fill=-1,
                                       dtype=np.int32)

    # Dictionary to store the average values for each category
    statistics = {
        'canopy_mean':{},
        'canopy_stdev':{},
        'canopy_skewness':{},
        'canopy_kurtosis':{}
    }

    if (raster.shape[0] > 10000) or (raster.shape[1] > 10000):
        print('Large raster: using raster subset')
        xarray = np.linspace(gdf.total_bounds[0], gdf.total_bounds[2],raster.shape[1]+1)
        yarray = np.linspace(gdf.total_bounds[3], gdf.total_bounds[1],raster.shape[0]+1)
        safety_value = 0.001

        # Calculate the average for each category
        for category in gdf_proj.index:
            average, stdev, skewness, kurtosisness = subset_raster_from_coords(gdf, xarray, yarray, safety_value, fields_rasterized, raster, category)

            statistics['canopy_mean'][category] = average
            statistics['canopy_stdev'][category] = stdev
            statistics['canopy_skewness'][category] = skewness
            statistics['canopy_kurtosis'][category] = kurtosisness
    else:
        for category in gdf_proj.index:
            mask = (fields_rasterized == category)
            average = raster[mask].mean() 
            statistics['canopy_mean'][category] = average
            stdev = raster[mask].std() 
            statistics['canopy_stdev'][category] = stdev
            skewness = skew(raster[mask])
            statistics['canopy_skewness'][category] = skewness
            kurtosisness = kurtosis(raster[mask])
            statistics['canopy_kurtosis'][category] = kurtosisness

    canopy_df = pd.DataFrame.from_dict({k:v for k,v in statistics.items()})
    canopy_df['plot_id'] = canopy_df.index

    return canopy_df


def download_pop_tiff_from_path(lcz_path):
    """
    Download and read a TIFF file from a given URL into a rasterio object.
    Handles cases where the TIFF is inside a ZIP archive.

    Parameters:
    lcz_path (str): URL of the raster file or ZIP archive.

    Returns:
    rasterio.io.DatasetReader: The raster dataset loaded into memory.
    """
    # Download the raster data
    response = requests.get(lcz_path, stream=True)
    if not response.ok:
        raise Exception(f"Failed to download file from {lcz_path}. Status code: {response.status_code}")

    # Read the content into a BytesIO buffer
    file_buffer = BytesIO(response.content)
    
    # Check if the file is a ZIP archive
    if zipfile.is_zipfile(file_buffer):
        with zipfile.ZipFile(file_buffer) as z:
            # Find the first TIFF file in the ZIP archive
            tiff_files = [f for f in z.namelist() if f.lower().endswith('.tif')]
            if not tiff_files:
                raise Exception("No TIFF files found in the ZIP archive.")
            
            # Open the first TIFF file
            with z.open(tiff_files[0]) as tiff_file:
                tiff_buffer = BytesIO(tiff_file.read())
                raster = rasterio.open(tiff_buffer)
                return raster
    else:
        # Assume the file is directly a TIFF file
        raster = rasterio.open(file_buffer)
        return raster
    
def merge_raster_list(raster_list):
    "Utility function to merge list of rasters"
    mosaic, out_trans = merge(raster_list)
    bounds = array_bounds(mosaic.shape[1], mosaic.shape[2], out_trans)
    return mosaic, bounds, raster_list[0].crs