# import base packages
import zipfile
import requests
from io import BytesIO
import rasterio
from rasterio.io import MemoryFile
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
import numpy as np
import pkg_resources

# import package functions and classes
from .utils import get_population_data_links, get_available_pop_countries


def get_population_data(country: str, use_tif = False, bounding_poly = None):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    Due to raster band sparsity for recent time periods, prioritises construction of indicators 
    from .csv file over .tif. Values are rounded to integers for ease of intepretation and for 
    convenient raster plotting.

    Args:
        country (str): Country name
        use_tif (bool): Choose whether to use csv or tif source
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile overlap for large countries

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    
    general_pop_dict = get_population_data_links(country)

    try:
        general_pop_dict[country]
    except KeyError:
        print(f"Population data is not available for {country}. Please use utils.get_available_pop_countries function to view which countries are available.")
    minx, miny, maxx, maxy = bounding_poly.bounds['minx'][0], bounding_poly.bounds['miny'][0], bounding_poly.bounds['maxx'][0], bounding_poly.bounds['maxy'][0]
    
    large_countries = ['United States']
    pop_groups = ['pop', 'men', 'women','elderly','youth','children'] 

    if not use_tif and country not in large_countries:
        # use csv
        groups_df = []
        target_cols = []
        for group in pop_groups:
            data_link = general_pop_dict[country][f'{group}_csv_url']
            pop_df = dd.read_csv(data_link, blocksize=None)
            
            lat_name, lon_name = list(pop_df.columns)[:2]
            pop_name  = list(pop_df.columns)[-1]

            pop_df = pop_df.persist()
            pop_df = pop_df[(pop_df[lat_name] >= miny) & (pop_df[lat_name] <= maxy) & (pop_df[lon_name] >= minx) & (pop_df[lon_name] <= maxx)]

            pop_gdf = gpd.GeoDataFrame(
                data={pop_name:pop_df[pop_name]}, 
                crs = 'epsg:4326',
                geometry = gpd.points_from_xy(pop_df[lon_name], pop_df[lat_name]))
            groups_df.append(pop_gdf)
            target_cols.append(pop_name)

        return groups_df, target_cols

    elif country in large_countries:
        country_mod = country.replace(' ', '')
        tile_polygon_path = pkg_resources.resource_filename('urbanity', f"map_data/{country_mod}.geojson")
        tile_polygons = gpd.read_file(tile_polygon_path)
        res_intersection = bounding_poly.overlay(tile_polygons, how='intersection')
        target_tiles = list(np.unique(res_intersection['TILEID']))

        # Extract point data for target tiles
        point_gdf = pd.GeoDataFrame()
        for tile in target_tiles:
            data_link = general_pop_dict[country][f'tile_all_{tile}.parquet']
            tile_df = pd.read_parquet(data_link)
            point_gdf = pd.concat([point_gdf, tile_df])
        
    else:
        # use tif files
        src_list = []
        rc_list = []
        for group in pop_groups:
            data_link = general_pop_dict[country][f'{group}_tif_url']
            filename = data_link.split('/')[-1][:-12]
            req = requests.get(data_link)

            try:
            # Extract and read tif into memory from zip file
                with zipfile.ZipFile(BytesIO(req.content)) as myzip:
                    with myzip.open(f'{filename}.tif', 'r') as myfile:
                        pop_map = myfile.read()
            except KeyError:
                filename = filename.replace(filename[:3], filename[:3].upper())
                with zipfile.ZipFile(BytesIO(req.content)) as myzip:
                    with myzip.open(f'{filename}.tif', 'r') as myfile:
                        pop_map = myfile.read()

            # Read TIF in-memory
            with rasterio.Env(CHECK_DISK_FREE_SPACE=False):
                with MemoryFile(pop_map) as memfile:
                    src = memfile.open()
                    rc = src.read(out_dtype='uint16')   

                # Update TIF in-memory
                meta = src.meta
                meta.update({
                "dtype": 'uint16',
                    "nodata": 0
                })

                with MemoryFile() as myfile:
                    src = myfile.open(** meta)
                    src.write(rc)
                    rc = src.read()
            
            src_list.append(src)
            rc_list.append(rc)
    
        return src_list, rc_list


def get_tiled_population_data(country: str, bounding_poly = None):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    For large areas, population data are pre-processed into tiles to allow quick extraction across 
    multiple geographic scales. This operation is not implemented for .tif format. 

    Args:
        country (str): Country name
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile overlap for large countries

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    groups_df = []
    target_cols = []

    general_pop_dict = get_population_data_links(country)
    minx, miny, maxx, maxy = bounding_poly.bounds['minx'][0], bounding_poly.bounds['miny'][0], bounding_poly.bounds['maxx'][0], bounding_poly.bounds['maxy'][0]
    country_mod = country.replace(' ', '')
    tile_polygon_path = pkg_resources.resource_filename('urbanity', f"map_data/{country_mod}.geojson")
    tile_polygons = gpd.read_file(tile_polygon_path)
    res_intersection = bounding_poly.overlay(tile_polygons, how='intersection')
    target_tiles = list(np.unique(res_intersection['TILEID']))

    # Extract point data for target tiles
    point_df = pd.DataFrame()
    for tile in target_tiles:
        data_link = general_pop_dict[country][f'tile_all_{tile}.parquet']
        tile_df = dd.read_parquet(data_link)
        tile_df = tile_df[(tile_df['latitude'] >= miny) & (tile_df['latitude'] <= maxy) & (tile_df['longitude'] >= minx) & (tile_df['longitude'] <= maxx)].compute()
        point_df = pd.concat([point_df, tile_df])
    point_df['group'] = point_df['group'].replace({'men_1':'men', 'men_2':'men', 'women_1':'women', 'women_2': 'women',
                                    'pop_1': 'pop', 'pop_2':'pop', 'pop_3':'pop', 'pop_4': 'pop', 'pop_5':'pop', 'pop_6':'pop'})
    points_gdf = gpd.GeoDataFrame(data=point_df[['population','group']], crs='epsg:4326', geometry=gpd.points_from_xy(point_df['longitude'],point_df['latitude']))

    for group in ['pop','men','women', 'elderly','youth','children']:
        groups_df.append(points_gdf[points_gdf['group'] == group])
        target_cols.append('population')
    return groups_df, target_cols