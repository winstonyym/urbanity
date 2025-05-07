import os
import io
import ee
import math
import json
import random
import glob
import geemap
import requests
import rasterio
import zipfile
import mercantile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Polygon, Point

from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import rasterio.transform as rtransform
from shapely.geometry import box
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling



def get_tiles_from_bbox(bbox: list = None,
                        zoom = 18):

    tiles = list(mercantile.tiles(bbox[0], bbox[1], bbox[2], bbox[3], zoom))
    
    return tiles

def download_satellite_tiles_from_bbox(bbox, api_key: str):
    # Get tiles from bounding box
    tiles = get_tiles_from_bbox(bbox) 
    
    # Create directory if not exists
    output_dir = "./satellite_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize progress bar
    with tqdm(total=len(tiles), desc="Downloading Tiles") as pbar:
        for tile in tiles:
            img_path = os.path.join(output_dir, f"{tile.z}_{tile.x}_{tile.y}.jpg")
            
            # Check if file already exists
            if os.path.exists(img_path):
                pbar.update(1)
                continue
            
            tile_url = f'https://api.mapbox.com/v4/mapbox.satellite/{tile.z}/{tile.x}/{tile.y}@2x.jpg90?'
            response = requests.get(tile_url, params={'access_token': api_key})
            
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            
            pbar.update(1)

def get_grid_size(tile_gdf_proj):
    x_grid_size = (tile_gdf_proj.iloc[[0]].geometry.total_bounds[2] - tile_gdf_proj.iloc[[0]].geometry.total_bounds[0])/512
    y_grid_size = (tile_gdf_proj.iloc[[0]].geometry.total_bounds[3] - tile_gdf_proj.iloc[[0]].geometry.total_bounds[1])/512
    return x_grid_size, y_grid_size

def get_max_img_dims(buildings_proj):
    ''' Helper function to return the maximum number of pixels in length and width to determine image chipset size'''
    
    bounds_df = buildings_proj.geometry.bounds
    x_max = max(bounds_df['maxx'] - bounds_df['minx'])
    y_max = max(bounds_df['maxy'] - bounds_df['miny'])
    total_max = max(x_max, y_max)
    half_max = total_max / 2
    return total_max, half_max


def get_and_combine_tiles(tiles_gdf_proj, building_chips, data_folder, output_folder):
    ''' Helper function to crop satellite image to desired building chip sizes'''
    
    # Placeholder list
    problems = []

    # Create directory to store building satellite image chips
    os.makedirs(os.path.join(os.getcwd(), output_folder), exist_ok=True)
        
    # Obtain chosen satellite images
    chosen_tiles = tiles_gdf_proj.overlay(building_chips)
    
    x_grid_size, y_grid_size = get_grid_size(tiles_gdf_proj)
    
    # Use tqdm to track progress
    for bid in tqdm(chosen_tiles['bid'].unique(), desc="Processing buildings"):
        temp_gdf = chosen_tiles[chosen_tiles['bid'] == bid]
        columns = []
        
        # First loop over x values
        for x_val in temp_gdf['X'].unique():
            x_gdf = temp_gdf[temp_gdf['X'] == x_val]
            
            rasters = []
            for _, row in x_gdf.iterrows():
                raster_path = os.path.join(os.getcwd(), data_folder, f"{row['Z']}_{row['X']}_{row['Y']}.jpg")
                with rasterio.open(raster_path) as f:
                    raster = f.read()
                    rasters.append(raster)
            
            columns.append(np.concatenate((rasters), axis=1))
            
        try:  
            # Combine satellite chips into single image
            img = np.concatenate((columns), axis=2)    
            # Extract building from image
            b_minx, b_miny, b_maxx, b_maxy = building_chips[building_chips['bid'] == bid]['geometry'].bounds.values[0]
            tile_minx, tile_miny, tile_maxx, tile_maxy = tiles_gdf_proj[tiles_gdf_proj['tile_id'].isin(list(chosen_tiles[chosen_tiles['bid'] == bid]['tile_id']))].total_bounds
            
            img_x_start = int((b_minx - tile_minx) / x_grid_size)
            img_y_start = int((tile_maxy - b_maxy) / y_grid_size)
            img_x_end = int((b_maxx - b_minx) / x_grid_size) + img_x_start
            img_y_end = img_y_start + int((b_maxy - b_miny) / y_grid_size)
            cropped_img = img[:, img_y_start:img_y_end, img_x_start:img_x_end]

            # Save image chip for building
            try:
                im = Image.fromarray(np.transpose(cropped_img, (1, 2, 0)))
                im.save(os.path.join(os.getcwd(), output_folder, f"{bid}.png"))
            except ValueError:
                problems.append(bid)
                
        except ValueError:
            problems.append(bid)
    
    # Try again for problems
    for bid in tqdm(problems, desc="Retrying failed cases"):
        temp_gdf = chosen_tiles[chosen_tiles['bid'] == bid]
        columns = []
        
        # First loop over x values
        for x_val in temp_gdf['X'].unique():
            rasters = []
            for y_val in temp_gdf['Y'].unique():
                raster_path = os.path.join(os.getcwd(), 'satellite_data/', f"18_{x_val}_{y_val}.jpg")
                with rasterio.open(raster_path) as f:
                    raster = f.read()
                    rasters.append(raster)
        
            columns.append(np.concatenate((rasters), axis=1))

        # Combine satellite chips into single image
        img = np.concatenate((columns), axis=2)    
        
        # Extract building from image
        b_minx, b_miny, b_maxx, b_maxy = building_chips[building_chips['bid'] == bid]['geometry'].bounds.values[0]
        tile_minx, tile_miny, tile_maxx, tile_maxy = tiles_gdf_proj[tiles_gdf_proj['tile_id'].isin(list(chosen_tiles[chosen_tiles['bid'] == bid]['tile_id']))].total_bounds

        img_x_start = int((b_minx - tile_minx) / x_grid_size)
        img_y_start = int((tile_maxy - b_maxy) / y_grid_size)
        img_x_end = int((b_maxx - b_minx) / x_grid_size) + img_x_start
        img_y_end = img_y_start + int((b_maxy - b_miny) / y_grid_size)
        cropped_img = img[:, img_y_start:img_y_end, img_x_start:img_x_end]

        # Save image chip for building
        im = Image.fromarray(np.transpose(cropped_img, (1, 2, 0)))
        im.save(os.path.join(os.getcwd(), output_folder, f"{bid}.png"))



def view_satellite_building_pair(gdf, bids):    
    # Initialize figure and a 4x4 GridSpec
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4,4, wspace=0.3, hspace=0.3)
    
    chosen_bids = random.sample(sorted(bids), 8)
    for i, bid in enumerate(chosen_bids):
        # Determine row and column for this pair
        #
        # We have 4 rows total (0..3).
        # Each row has 4 columns (0..3).
        # Each row is split into 2 pairs: (col 0, col 1) and (col 2, col 3).
        # i // 2 is the row index, (i % 2)*2 is the column index for the polygon.
        row = i // 2
        col = (i % 2) * 2

        # 1) Plot the building polygon on the left subplot of the pair
        ax_poly = fig.add_subplot(gs[row, col])
        gdf[gdf['bid'] == bid].plot(ax=ax_poly, color='lightblue', edgecolor='black')
        ax_poly.set_title(f"BID: {bid} – Polygon")
        ax_poly.set_aspect('auto')
        ax_poly.set_xticks([])
        ax_poly.set_yticks([])

        # 2) Plot the corresponding satellite image on the right subplot
        ax_img = fig.add_subplot(gs[row, col+1])
        img_path = f'./building_satellite/{bid}.png'
        img = mpimg.imread(img_path)
        ax_img.imshow(img)
        ax_img.set_title(f"BID: {bid} – Satellite Image")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_aspect('auto')

    plt.show()


def get_building_image_chips(building_proj, tiles_gdf_proj, add_context = False, pad_npixels = 0):    
    # Project building to local coordinates
    building_chips = building_proj.copy()
    
    if add_context:
        geom_list = []
        x_grid_size, y_grid_size = get_grid_size(tiles_gdf_proj)
        building_chips['geometry'] = building_chips['geometry'].bounds.apply(lambda row: geometry.box(row['minx']-x_grid_size*pad_npixels, 
                                                                                                      row['miny']-y_grid_size*pad_npixels, 
                                                                                                      row['maxx']+x_grid_size*pad_npixels, 
                                                                                                      row['maxy']+y_grid_size*pad_npixels), axis=1)
    else: 
        building_chips['geometry'] = building_chips['geometry'].bounds.apply(lambda row: geometry.box(row['minx'], row['miny'], row['maxx'], row['maxy']), axis=1)
    
    return building_chips


def get_tiles_gdf(tiles, bounds = None):
    x_list, y_list, z_list = [], [], []
    
    geom = []
    for tile in tiles:
        x_list.append(tile.x)
        y_list.append(tile.y)
        z_list.append(tile.z)
        geom.append(geometry.box(*mercantile.bounds(tile)))
    df = pd.DataFrame({'X': x_list, 'Y': y_list, 'Z': z_list})
    gdf = gpd.GeoDataFrame(data=df, crs = 'epsg:4326', geometry=geom)
    gdf['GID'] = range(len(gdf))
    
    if bounds is not None:
        intersect = bounds.overlay(gdf)
        selected_grids = gdf[gdf['GID'].isin(list(intersect['GID']))]
        selected_grids = selected_grids[['X','Y','Z','geometry']]

        return selected_grids
    
    return gdf


def download_tiff_from_path(lcz_path):
    # Send a request to get the file size (for progress calculation)
    response = requests.get(lcz_path, stream=True)
    
    if response.ok:
        total_size = int(response.headers.get('content-length', 0))  # Get total file size in bytes
        chunk_size = 1024 * 8  # Define chunk size
        save_path = './lcz_data/lcz_data.tif'

        # Open file and start downloading with progress bar
        with open(save_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    bar.update(len(chunk))  # Update progress bar

        print("Downloaded global LCZ data to:", save_path)
    else:
        print(f"Failed to download file: {response.status_code}")

def reproject_raster(input_path, output_path):
    # Paths to input and output files


    # The target CRS is EPSG:4326
    dst_crs = "EPSG:4326"

    # Open the source dataset
    with rasterio.open(input_path) as src:
        # Calculate the transform and dimensions of the new raster
        transform, width, height = calculate_default_transform(
            src.crs,       # The current coordinate reference system
            dst_crs,       # The desired coordinate reference system
            src.width,
            src.height,
            *src.bounds     # The bounding coordinates of the original file
        )

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Create and write to the new reprojected raster
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

        print(f"Reprojected raster saved to {output_path}")
        
def raster2gdf(raster_path, chosen_band = 1, get_grid=True, zoom=False, boundary=None):
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
        boundary = boundary.to_crs(raster_dataset.crs)
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
        my_gdf = gpd.GeoDataFrame(data = mydf, crs = raster_dataset.crs, geometry = geom_list)
        my_gdf = my_gdf.to_crs('epsg:4326')
        return my_gdf

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
        gdf = gdf.to_crs('epsg:4326')
    return gdf

def raster2gdf_with_reprojection(raster_path, boundary_gdf, chosen_band=1):
    """
    Example function that:
      1) Reads a raster,
      2) Reprojects entire data to EPSG:4326 in memory,
      3) Zooms to a user boundary, also in EPSG:4326,
      4) Returns a polygonal GeoDataFrame of the zoomed portion.
    """
    dst_crs = "EPSG:4326"
    # ---------------------------
    # 1. Read entire source raster
    # ---------------------------
    with rasterio.open(raster_path) as src:
        src_data = src.read(chosen_band)
        src_transform = src.transform
        src_crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

    # ---------------------------
    # 2. Reproject to EPSG:4326
    # ---------------------------

    
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,        # from
        dst_crs,        # to
        width, 
        height, 
        *bounds
    )

    # Prepare empty array for reprojected data
    dst_data = np.empty((dst_height, dst_width), dtype=src_data.dtype)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    # Reproject the boundary to EPSG:4326 as well
    boundary_gdf = boundary_gdf.to_crs(dst_crs)
    # ---------------------------
    # 3. Build the full coordinate mesh (EPSG:4326)
    #    for the *entire* reprojected raster
    # ---------------------------
    # array_bounds returns (bottom, top) in final elements for y, so be mindful
    full_left, full_bottom, full_right, full_top = array_bounds(
        dst_height, dst_width, dst_transform
    )

    xarray_full = np.linspace(full_left,   full_right,  dst_width + 1)
    yarray_full = np.linspace(full_top,    full_bottom, dst_height + 1)

    # Pixel half-sizes in the reprojected space
    half_x = (full_right  - full_left)   / dst_width  / 2
    half_y = (full_top    - full_bottom) / dst_height / 2

    # ---------------------------
    # 4. Determine the zoom window in EPSG:4326
    #    from boundary_gdf's total_bounds
    # ---------------------------
    xmin, ymin, xmax, ymax = boundary_gdf.total_bounds

    # We want to include any pixel whose boundary overlaps
    # the boundary_gdf bounding box
    list_x_range = [
        x for x in xarray_full
        if (x >= xmin - half_x and x <= xmax + half_x)
    ]
    list_y_range = [
        y for y in yarray_full
        if (y <= ymax + half_y and y >= ymin - half_y)
    ]

    # Handle edge cases (e.g. boundary outside the raster)
    if not list_x_range or not list_y_range:
        print("Boundary is entirely outside the reprojected raster.")
        return None

    # Offsets in the reprojected array
    x_off = np.where(xarray_full == list_x_range[0])[0].item()
    y_off = np.where(yarray_full == list_y_range[0])[0].item()

    # The width/height in pixels of the zoomed region
    x_width = len(list_x_range) - 1
    y_height = len(list_y_range) - 1

    # ---------------------------
    # 5. Slice the reprojected data to get only the zoom
    #    region
    # ---------------------------
    # The data array has shape = (dst_height, dst_width)
    zoomed_data = dst_data[y_off:(y_off + y_height), x_off:(x_off + x_width)]

    # Build a new transform for this subset
    from rasterio.windows import Window, transform
    window = Window(x_off, y_off, x_width, y_height)
    zoom_transform = transform(window, dst_transform)

    # ---------------------------
    # 6. Create polygons for each pixel in the zoom region
    # ---------------------------
    zoom_left, zoom_bottom, zoom_right, zoom_top = rasterio.transform.array_bounds(
        y_height, x_width, zoom_transform
    )
    # Note: array_bounds -> (left, bottom, right, top)

    zoom_xarray = np.linspace(zoom_left,  zoom_right,  x_width  + 1)
    zoom_yarray = np.linspace(zoom_top,   zoom_bottom, y_height + 1)

    # Flatten the pixel values
    vals = zoomed_data.ravel(order='C')
    df = pd.DataFrame({'value': vals})

    polygons = []
    for row_i in range(y_height):
        for col_i in range(x_width):
            p = Polygon([
                (zoom_xarray[col_i],   zoom_yarray[row_i]),
                (zoom_xarray[col_i],   zoom_yarray[row_i+1]),
                (zoom_xarray[col_i+1], zoom_yarray[row_i+1]),
                (zoom_xarray[col_i+1], zoom_yarray[row_i])
            ])
            polygons.append(p)

    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=dst_crs)
    return gdf

def mosaic_and_save_tiff(source, dest, name, fmt='tiff'):
    tifs = glob.glob(os.path.join(source,  '*'))

    src_files_to_mosaic = []
    for fp in tifs:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    mosaic, out_trans = merge(src_files_to_mosaic)

    if fmt == 'tiff':
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans})
        
        with rasterio.open(os.path.join(dest,f'{name}.tif'), "w", **out_meta) as dest:
            dest.write(mosaic)
            
    elif fmt == 'npz':
        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        np.savez_compressed(os.path.join(dest, f'{name}.npz'), 
                            mosaic=mosaic, 
                            meta=out_meta)
        

def extract_tiff_from_shapefile(geotiff_path, shapefile, output_filepath=None, zipped=True):

    if zipped:
        with zipfile.ZipFile(geotiff_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(geotiff_path))
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
    

def merge_raster_to_gdf(raster, gdf, id_col = 'plot_id', raster_col='value', num_classes = 1, method='mean'):
    res_intersection = raster.overlay(gdf)

    if method == 'proportion':
        val_series = res_intersection.groupby(id_col)[raster_col].value_counts()

        raster_df = pd.DataFrame(index = val_series.index, data = val_series.values).reset_index()

        # return lcz_df
        raster_df = pd.pivot(raster_df, index=id_col, columns=raster_col, values=0).fillna(0)
    
        # Sum across rows to get total for each row
        row_totals = raster_df.sum(axis=1)
        raster_df = raster_df.div(row_totals, axis=0) * 100

        # Rename columns
        raster_df.columns = [raster_col + '_' + str(col) for col in raster_df.columns]
        raster_column_names = [raster_col + '_' +str(i) for i in range(0,num_classes)]

        for i in raster_column_names:
            if i not in set(raster_df.columns):
                raster_df[i] = 0
            elif i in set(raster_df.columns):
                raster_df[i] = raster_df[i].replace(np.nan, 0)
        
        raster_df = raster_df[raster_column_names]
        
        # Join pixel mean to network edges
        gdf = gdf.merge(raster_df, on=id_col, how='left')

    elif method == 'mean': 
        mean_series = res_intersection.groupby(id_col)[raster_col].mean()
        mean_series.name = f'{raster_col}_mean'

        std_series = res_intersection.groupby(id_col)[raster_col].std()
        std_series.name = f'{raster_col}_std'
        
        # Join pixel mean to network edges
        gdf = gdf.merge(mean_series, on=id_col, how='left')
        gdf[f'{raster_col}_mean'] = gdf[f'{raster_col}_mean'].fillna(0)

        gdf = gdf.merge(std_series, on=id_col, how='left')
        gdf[f'{raster_col}_std'] = gdf[f'{raster_col}_std'].fillna(0)

    return gdf

def tiled_large_array(boundary, layer_name, delta=0.1, band='', index=0):
    """Utility function to return mosaic array for large area by first tiling it. 

    Args:
        boundary (GeoDataFrame): The boundary polygon(s) for the area of interest.
        layer_name (str): Earth Engine ImageCollection ID or Image ID.
        delta (float): Optional parameter (not strictly needed if using ee_to_numpy).
        band (str): The band name to select. If provided, we select by band name.
        index (int): The index of the image in the collection to fetch (if multiple images).

    Returns:
        Full numpy array with mosaic-ed raster tiles
    """
    array_list = []

    tiles = split_bbox_into_grid(boundary, n=64)

    for i in range(len(tiles)):
        current_row = tiles.iloc[[i]]

        # 2. Convert boundary to Earth Engine geometry
        tile_geom = current_row[['geometry']]  # Keep only geometry column
        gdf_json = json.loads(tile_geom.to_json())
        ee_fc = ee.FeatureCollection(gdf_json["features"])
        roi = ee_fc.geometry()

            # 3. Attempt to load layer_name as an ImageCollection and pick one by index.
        #    If that fails or yields an empty collection, treat as single Image.
        try:
            # Try as an ImageCollection
            collection = ee.ImageCollection(layer_name).filterBounds(roi)
            count = collection.size().getInfo()
            if count > 0:
                # Sort by time and select the desired image
                sorted_collection = collection.sort("system:time_start", False)
                if index >= count:
                    raise ValueError(
                        f"Requested index {index} is out of range. Only {count} images found in collection."
                    )
                image_ee = ee.Image(sorted_collection.toList(count).get(index))
            else:
                # Fallback to treating layer_name as single Image
                image_ee = ee.Image(layer_name).clip(roi)
        except Exception:
            # If it fails as a collection, interpret as a single Image
            image_ee = ee.Image(layer_name).clip(roi)

        # Check available bands
        info = image_ee.getInfo()
        available_bands = [b['id'] for b in info['bands']]

        # 4. If the user specified a band name, select it; otherwise select the first band
        if band != '':
            if band not in available_bands:
                raise ValueError(
                    f"Error: '{band}' is not a valid band. Available bands are: {available_bands}"
                )
            image_ee = image_ee.select(band).clip(roi).unmask()
        else:
            first_band_name = available_bands[0]
            image_ee = image_ee.select(first_band_name).clip(roi).unmask()
        
        arr = geemap.ee_to_numpy(image_ee, region=roi)
        array_list.append(arr)

    return concat_grid(array_list)
        
def gee_layer_from_boundary(boundary, layer_name, delta=0.1, band='', index=0, large=True):
    """
    Fetch a raster from the given Earth Engine ImageCollection or single Image 
    over the input boundary, and convert it to a GeoDataFrame with polygon 
    geometries for each pixel.

    Args:
        boundary (GeoDataFrame): The boundary polygon(s) for the area of interest.
        layer_name (str): Earth Engine ImageCollection ID or Image ID.
        delta (float): Optional parameter (not strictly needed if using ee_to_numpy).
        band (str): The band name to select. If provided, we select by band name.
        index (int): The index of the image in the collection to fetch (if multiple images).

    Returns:
        GeoDataFrame with columns ['value'] and the polygon geometry of each pixel.
    """
    # Toggle tiling mechanism if area is large and want to use large
    if (boundary.geometry.area.values > 0.1) and large:
        large=True
    else:
        large=False
    
    if large:
        # 2. Convert boundary to Earth Engine geometry
        boundary = boundary[['geometry']]  # Keep only geometry column
        gdf_json = json.loads(boundary.to_json())
        ee_fc = ee.FeatureCollection(gdf_json["features"])
        roi = ee_fc.geometry()

        # 3. Attempt to load layer_name as an ImageCollection and pick one by index.
        #    If that fails or yields an empty collection, treat as single Image.
        try:
            # Try as an ImageCollection
            collection = ee.ImageCollection(layer_name).filterBounds(roi)
            count = collection.size().getInfo()
            if count > 0:
                # Sort by time and select the desired image
                sorted_collection = collection.sort("system:time_start", False)
                if index >= count:
                    raise ValueError(
                        f"Requested index {index} is out of range. Only {count} images found in collection."
                    )
                image_ee = ee.Image(sorted_collection.toList(count).get(index))
            else:
                # Fallback to treating layer_name as single Image
                image_ee = ee.Image(layer_name).clip(roi)
        except Exception:
            # If it fails as a collection, interpret as a single Image
            image_ee = ee.Image(layer_name).clip(roi)

        # Check available bands
        info = image_ee.getInfo()
        available_bands = [b['id'] for b in info['bands']]

        # 4. If the user specified a band name, select it; otherwise select the first band
        if band != '':
            if band not in available_bands:
                raise ValueError(
                    f"Error: '{band}' is not a valid band. Available bands are: {available_bands}"
                )
            image_ee = image_ee.select(band).clip(roi).unmask()
        else:
            first_band_name = available_bands[0]
            image_ee = image_ee.select(first_band_name).clip(roi).unmask()

        arr = tiled_large_array(boundary, layer_name, delta=0.1, band='', index=0)
        return arr

    else:

        # 2. Convert boundary to Earth Engine geometry
        boundary = boundary[['geometry']]  # Keep only geometry column
        gdf_json = json.loads(boundary.to_json())
        ee_fc = ee.FeatureCollection(gdf_json["features"])
        roi = ee_fc.geometry()

        # 3. Attempt to load layer_name as an ImageCollection and pick one by index.
        #    If that fails or yields an empty collection, treat as single Image.
        try:
            # Try as an ImageCollection
            collection = ee.ImageCollection(layer_name).filterBounds(roi)
            count = collection.size().getInfo()
            if count > 0:
                # Sort by time and select the desired image
                sorted_collection = collection.sort("system:time_start", False)
                if index >= count:
                    raise ValueError(
                        f"Requested index {index} is out of range. Only {count} images found in collection."
                    )
                image_ee = ee.Image(sorted_collection.toList(count).get(index))
            else:
                # Fallback to treating layer_name as single Image
                image_ee = ee.Image(layer_name).clip(roi)
        except Exception:
            # If it fails as a collection, interpret as a single Image
            image_ee = ee.Image(layer_name).clip(roi)

        # Check available bands
        info = image_ee.getInfo()
        available_bands = [b['id'] for b in info['bands']]

        # 4. If the user specified a band name, select it; otherwise select the first band
        if band != '':
            if band not in available_bands:
                raise ValueError(
                    f"Error: '{band}' is not a valid band. Available bands are: {available_bands}"
                )
            image_ee = image_ee.select(band).clip(roi).unmask()
        else:
            first_band_name = available_bands[0]
            image_ee = image_ee.select(first_band_name).clip(roi).unmask()

        # 5. Convert to NumPy array
        # arr = fetch_tiled_array(image_ee, roi, max_tile_px=2048)
        arr = geemap.ee_to_numpy(image_ee, region=roi)

    if arr is None:
        raise ValueError(
            "ee_to_numpy returned None. Possibly no valid pixels in the region or an error retrieving data."
        )

    # Handle single- or multi-band arrays
    if arr.ndim == 3:
        arr_2d = arr[:, :, 0]
    else:
        arr_2d = arr

    nrows, ncols = arr_2d.shape

    # 6. Get the bounding box from the ROI
    bounds_info = roi.bounds().getInfo()['coordinates'][0]
    minx = min(coord[0] for coord in bounds_info)
    maxx = max(coord[0] for coord in bounds_info)
    miny = min(coord[1] for coord in bounds_info)
    maxy = max(coord[1] for coord in bounds_info)

    # 7. Get the projection info (CRS) from the Image
    proj = image_ee.projection()
    proj_info = proj.getInfo()

    if 'crs' in proj_info:
        crs = proj_info['crs']
    else:
        # Fallback or guess. Often EPSG:4326 is a reasonable default
        crs = "EPSG:4326"

    # 8. Create an affine transform from bounding box and array shape
    transform = rtransform.from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    # 9. Build polygons for each pixel
    polygons = []
    values = []

    for row in range(nrows):
        for col in range(ncols):
            pixel_value = arr_2d[row, col]

            # Top-left corner of pixel
            xleft, ytop = transform * (col, row)
            # Bottom-right corner of pixel
            xright, ybottom = transform * (col + 1, row + 1)

            poly = Polygon([
                (xleft, ytop),
                (xright, ytop),
                (xright, ybottom),
                (xleft, ybottom),
                (xleft, ytop)  # close polygon
            ])

            polygons.append(poly)
            values.append(pixel_value)
            
    # points = []
    # values = []

    # for row in range(nrows):
    #     for col in range(ncols):
    #         pixel_value = arr_2d[row, col]

    #         # (x_center, y_center) is the center of the pixel
    #         x_center, y_center = transform * (col + 0.5, row + 0.5)

    #         point = Point(x_center, y_center)
    #         points.append(point)
    #         values.append(pixel_value)


    # 10. Create a GeoDataFrame
    df = pd.DataFrame({"value": values})
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=crs)

    return gdf

def concat_grid(tile_list, *, nrows=None, ncols=None, fill_value=np.nan):
    """
    Stitch a flat list of NumPy arrays—supplied row-major—into one mosaic.

    Parameters
    ----------
    tile_list : list[np.ndarray]
        Tiles in row-major order:  [row-0-col-0, row-0-col-1, …, row-1-col-0, …]
    nrows, ncols : int, optional
        Grid dimensions.  • If both are None, they are inferred by
          *sqrt*—works for 4, 16, 64, … (perfect squares).
        • If only one is given, the other is derived from `len(tile_list)`.
    fill_value : scalar, default np.nan
        Value used when padding tiles whose shapes differ within a row/column.

    Returns
    -------
    np.ndarray
        The stitched raster.

    Raises
    ------
    ValueError
        If the list length is incompatible with the requested grid.
    """
    n_tiles = len(tile_list)

    # ---- 1. derive grid shape ----------------------------------------
    if nrows is None and ncols is None:
        root = int(math.isqrt(n_tiles))
        if root * root != n_tiles:
            raise ValueError("tile_list length is not a perfect square; "
                             "specify nrows and/or ncols.")
        nrows = ncols = root
    elif nrows is None:
        if n_tiles % ncols:
            raise ValueError("ncols does not divide tile count.")
        nrows = n_tiles // ncols
    elif ncols is None:
        if n_tiles % nrows:
            raise ValueError("nrows does not divide tile count.")
        ncols = n_tiles // nrows
    else:
        if nrows * ncols != n_tiles:
            raise ValueError("nrows × ncols != tile count.")

    # ---- 2. find max height per row & max width per col --------------
    max_h_per_row = [
        max(tile_list[r * ncols + c].shape[0] for c in range(ncols))
        for r in range(nrows)
    ]
    max_w_per_col = [
        max(tile_list[r * ncols + c].shape[1] for r in range(nrows))
        for c in range(ncols)
    ]

    # ---- 3. pad tiles so shapes align within grid --------------------
    padded = []
    for r in range(nrows):
        for c in range(ncols):
            t = tile_list[r * ncols + c]
            h_pad = max_h_per_row[r] - t.shape[0]
            w_pad = max_w_per_col[c] - t.shape[1]
            if h_pad or w_pad:
                t = np.pad(
                    t,
                    ((0, h_pad), (0, w_pad)),
                    mode="constant",
                    constant_values=fill_value,
                )
            padded.append(t)

    # ---- 4. horizontal then vertical concat --------------------------
    rows = [
        np.hstack(padded[r * ncols : (r + 1) * ncols]) for r in range(nrows)
    ]
    mosaic = np.vstack(rows)
    return mosaic

def split_bbox_into_grid(bbox_gdf, *, n=None, nrows=None, ncols=None,
                         id_col="tile_id", row_major=True):
    """
    Split the bounding box of `bbox_gdf` into `nrows × ncols` equal-sized
    rectangles, or into *≈ n* rectangles if only `n` is given.

    Parameters
    ----------
    bbox_gdf : GeoDataFrame
        Any GeoDataFrame whose `.total_bounds` defines the area to split.
    n : int, optional
        Desired number of tiles (e.g. 4, 8, 16).  
        If `nrows`/`ncols` are **not** supplied, the function chooses a
        rectangular grid whose tile count ≥ `n` and is as square as possible.
    nrows, ncols : int, optional
        Explicit grid dimensions.  If one is `None`, it is derived from
        the other and `n` (e.g. `nrows=2, n=None, ncols=None` → `ncols`
        chosen so `2×ncols ≥ n`).
    id_col : str, default "tile_id"
        Name of the identifier column in the returned GeoDataFrame.
    row_major : bool, default True
        If `True`, IDs increase left-to-right, top-to-bottom (matrix style).
        If `False`, they increase column-wise (top-to-bottom, then left-right).

    Returns
    -------
    GeoDataFrame
        Grid cells in the same CRS as the input.  Extra cells (if the chosen
        grid has more than `n` tiles) are trimmed from the **bottom-right**
        corner so the row/column structure stays intact.
    """
    # ---- derive grid shape -------------------------------------------
    if nrows is None and ncols is None:
        if n is None:
            raise ValueError("Specify `n` or (`nrows`, `ncols`).")
        # choose the most square grid with rows*cols >= n
        nrows = int(math.floor(math.sqrt(n)))
        ncols = int(math.ceil(n / nrows))
    elif nrows is None:                    # ncols is given
        if n is None:
            raise ValueError("Need `n` to derive nrows.")
        nrows = int(math.ceil(n / ncols))
    elif ncols is None:                    # nrows is given
        if n is None:
            raise ValueError("Need `n` to derive ncols.")
        ncols = int(math.ceil(n / nrows))

    # ---- bounds -------------------------------------------------------
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    dx = (maxx - minx) / ncols
    dy = (maxy - miny) / nrows

    # ---- build tiles --------------------------------------------------
    tiles, ids = [], []
    current_id = 0
    for r in range(nrows):
        y_top    = maxy - r * dy
        y_bottom = maxy - (r + 1) * dy
        for c in range(ncols):
            if n is not None and current_id >= n:          # trim extras
                break
            x_left  = minx + c * dx
            x_right = minx + (c + 1) * dx
            tiles.append(box(x_left, y_bottom, x_right, y_top))
            ids.append(current_id)
            current_id += 1
        if n is not None and current_id >= n:
            break

    # ---- reorder IDs (row-major vs column-major) ----------------------
    if not row_major:
        # transpose indexing: id = c * nrows + r  instead of r * ncols + c
        ids = sorted(range(len(tiles)),
                     key=lambda k: (k % nrows) * ncols + (k // nrows))

    return gpd.GeoDataFrame({id_col: ids},
                            geometry=tiles,
                            crs=bbox_gdf.crs)


def get_affine_transform(gdf, raster):

    # ❶ Bounding box (in the same CRS you want for the raster)
    minx, miny, maxx, maxy = gdf.total_bounds   # (west, south, east, north)

    # ❷ Raster dimensions
    if raster.ndim == 3:          # (bands, rows, cols)
        height, width = raster[:,:,0].shape
    else:                      # (rows, cols)
        height, width = raster.shape

    transform = from_bounds(
        minx, miny, maxx, maxy,   # geographic extent
        width, height             # raster grid size
    )

    return transform