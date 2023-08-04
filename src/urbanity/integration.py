# Functions to integrate external data into Urbanity
import pandas as pd
import geopandas as gpd

def process_time_series(filepath, lat_col = 'latitude', lon_col = 'longitude', minx=103.605, miny=1.211, maxx=104.084, maxy=1.471):
    """Function to filter, clean, and process raw time series .parquet file. Returns a GeoDataFrame with lat/lon geometry column.

    Args:
        filepath (str): Path to external data file
        lat_col (str): Column name for latitude column. Defaults to latitude.
        lon_col (str): Column name for longitude column. Defaults to longitude.
        minx (float, optional): Minimum longitude for point locations. Defaults to 103.605.
        miny (float, optional): Minimum latitude for point locations. Defaults to 1.211.
        maxx (float, optional): Maximum latitude for point locations. Defaults to 104.084.
        maxy (float, optional): Maximum longitude for point locations. Defaults to 1.471.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame that consists of filtered points (by boundary) with included geometry column of point locations.
    """    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.gizp'):
        df = pd.read_parquet(filepath)
    
    notnan_df = df[~df[lon_col].isna()]
    notnan_df = notnan_df[~notnan_df[lat_col].isna()]
    
    # Filter by Sg latitude bounds
    notnan_df_lat = notnan_df[(notnan_df[lat_col] > miny) & (notnan_df[lat_col] < maxy)]
    
    # Filter by Sg longitude bounds
    notnan_df_lat_lon = notnan_df_lat[(notnan_df_lat[lon_col] > minx) & (notnan_df_lat[lon_col] < maxx)]
    
    # Create GeoDataFrame
    # x, y = np.unique(notnan_df_lat_lon[['latitude','longitude']],axis=0)[:,[0]].squeeze(1), np.unique(out[['latitude','longitude']],axis=0)[:,[1]].squeeze(1)
    gdf = gpd.GeoDataFrame(data=notnan_df_lat_lon, crs='epsg:4326', geometry=gpd.points_from_xy(notnan_df_lat_lon[lon_col],notnan_df_lat_lon[lat_col]))
    
    gdf.insert(loc=0, column='oid', value = list(range(len(gdf))))

    return gdf