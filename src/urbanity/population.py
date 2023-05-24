# import base packages
import json
import zipfile
import requests
from io import BytesIO
import pandas as pd
import geopandas as gpd
import numpy as np
import pkg_resources

# import package functions and classes
from .utils import get_population_data_links, get_available_pop_countries


def get_population_data(country: str, bounding_poly = None):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    Due to raster band sparsity for recent time periods, prioritises construction of indicators 
    from .csv file over .tif. Values are rounded to integers for ease of intepretation and for 
    convenient raster plotting.

    Args:
        country (str): Country name
        bounding_poly (gpd.GeoDataFrame): Bounding box to compute raster tile overlap for large countries

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    
    general_pop_dict = get_population_data_links(country)

    try:
        general_pop_dict[country]
    except KeyError:
        print(f"Population data is not available for {country}. Please use utils.get_available_pop_countries function to view which countries are available.")
    minx, miny, maxx, maxy = bounding_poly.bounds['minx'].min(), bounding_poly.bounds['miny'].min(), bounding_poly.bounds['maxx'].max(), bounding_poly.bounds['maxy'].max()
    
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

    for group in ['pop','men','women', 'elderly','youth','children']:
        groups_df.append(points_gdf[points_gdf['group'] == group])
        target_cols.append('population')
    return groups_df, target_cols