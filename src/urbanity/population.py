import zipfile
import requests
from io import BytesIO
import rasterio
from rasterio.io import MemoryFile
import pandas as pd
import geopandas as gpd
import numpy as np

from .utils import get_population_data_links, get_available_pop_countries


def get_population_data(country: str, use_tif = False):
    """Extract Meta High Resolution Population Density dataset from HDX based on country name. 
    Due to raster band sparsity for recent time periods, prioritises construction of indicators 
    from .csv file over .tif. Values are rounded to integers for ease of intepretation and for 
    convenient raster plotting.

    Args:
        country (str): Country name

    Returns:
        pd.DataFrame/np.ndarray: Dataframe (csv) or array (tif) with population information
    """    
    
    general_pop_dict = get_population_data_links()

    try:
        general_pop_dict[country]
    except KeyError:
        print(f"Population data is not available for {country}. Please use utils.get_available_pop_countries function to view which countries are available.")

    groups = ['pop', 'men', 'women', 'elderly','youth','children']

    if not use_tif:
        groups_df = []
        target_cols = []
        for group in groups:
            data_link = general_pop_dict[country][f'{group}_csv_url']
            df = pd.read_csv(data_link)
            for colname in list(df.columns):
                if 'population' in colname:
                    target_col = colname
            df[target_col] = df[target_col].astype(int)
            pop_gdf = gpd.GeoDataFrame(
                data=df[target_col], 
                crs = 'epsg:4326',
                geometry = gpd.points_from_xy(df.longitude, df.latitude))
            groups_df.append(pop_gdf)
            target_cols.append(target_col)

        return groups_df, target_cols

    src_list = []
    rc_list = []
    for group in groups:
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
