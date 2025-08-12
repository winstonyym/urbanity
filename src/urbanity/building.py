# Import libraries
import os
import json
import math
import random
try:
    import pyrosm  # optional heavy dependency
    from pyrosm import get_data
except ImportError:  # pyrosm not installed
    pyrosm = None
    def get_data(*args, **kwargs):  # type: ignore
        raise ImportError("pyrosm not installed. Install via 'pip install urbanity[osm]' or conda-forge 'mamba install pyrosm'.")
import pandas as pd
import shapely
from shapely.geometry import Polygon, LineString
from shapely import wkb
import pkg_resources
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

from urbanity.geom import fill_and_expand, project_gdf, buffer_polygon

def get_osm_buildings(location = '', fp = '', boundary=None):
    """Wrapper around pyrosm API to retrieve OpenStreetMap building footprints from Geofabrik. Optionally accepts a GeoDataFrame as bounding spatial extent.

    Args:
        location (str): Specfic country or city name to obtain OpenStreetMap data.
        boundary (gpd.GeoDataFrame, optional): A GeoDataFrame corresponding to bounding spatial extent. Defaults to None.

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame containing OSM building footprints for specified spatial extent.
    """    
    if os.path.exists('./data'):
        pass
    else:
        os.makedirs('./data')

    if fp == '' and location != '':
        fp = get_data(location, directory = './data')
        osm = pyrosm.OSM(fp, bounding_box=boundary.geometry.values[0])

    elif fp == '' and location == '':
        raise ValueError("Please specify a valid city or country name.")
    else:
        buffered_boundary = buffer_polygon(boundary)
        osm = pyrosm.OSM(fp, bounding_box=buffered_boundary.geometry.values[0])

    osm_buildings = osm.get_buildings()

    
    return osm_buildings

def get_overture_buildings(building_data):
    """Temporary loader to directly load Overture building footprints.

    Args:
        building_data (str): Path to building footprint data.

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame containing OSM building footprints for specified spatial extent.
    """    
    building_gdf = gpd.read_parquet(building_data)
    if 'geometry_polygon' in building_gdf.columns:
        building_gdf['geometry'] = building_gdf['geometry_polygon']
        building_gdf = building_gdf.drop(columns = 'geometry_polygon')
    return building_gdf


def remove_overlapping_polygons(building):
    """Function to remove instances of buildings that overlap with one another. In cases of overlap, the biggest polygon is retained.

    Args:
        building (gpd.GeoDataFrame): A geopandas dataframe consisting of original building footprints.

    Returns:
        gpd.GeoDataFrame: Modified geopandas dataframe with each building footprint isolated from one another.
    """    
    # Intersect polygon with itself
    build_intersect = building.overlay(building)
    
    # Find ids which have more than one intersection with nearby polygons 
    duplicate_ids = build_intersect['bid_1'].value_counts().index[(build_intersect['bid_1'].value_counts().values > 1)]
    
    # For duplicate ids, remove them from the original polygon, and only keep the biggest one
    for bid in duplicate_ids:
        
        bids_to_remove = list(build_intersect[build_intersect['bid_1']==bid]['bid_2'])
        max_area = build_intersect[build_intersect['bid_1']==bid]['bid_area_2'].max()
        biggest_bid = build_intersect[(build_intersect['bid_area_2'] == max_area) & (build_intersect['bid_1']==bid)]['bid_2'].values[0]
        bids_to_remove.remove(biggest_bid)
        building = building[~building['bid'].isin(bids_to_remove)]
    
    building = building.reset_index(drop=True) 
    return building

def building_knn_nearest(gdf, knn=3, non_nan_col=None):
    """
    Helper function to generate nearest neighbours and distances for each building.
    Optionally, it excludes buildings that have NaN in `non_nan_col` from being 
    considered as neighbors.

    Args:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing building footprints 
                                and centroids in a column named 'bid_centroid'.
        knn (int, optional): Number of nearest neighbours to return. Defaults to 3.
        non_nan_col (str, optional): Name of a column. Buildings with NaN in this
                                     column are excluded from neighbors. If None,
                                     use all buildings. Defaults to None.

    Returns:
        gpd.GeoDataFrame: The original GeoDataFrame with two new columns:
                          1) `<knn>-nn-idx` with lists of nearest-neighbor indices
                          2) `<knn>-dist`   with the corresponding distances
    """    
    # 1. Extract coordinates of ALL buildings
    coords_all = np.array([
        (centroid.x, centroid.y) for centroid in gdf['bid_centroid']
    ])

    # 2. Determine which buildings are *valid* neighbors
    #    i.e. they do NOT have NaN in `non_nan_col`.
    if non_nan_col is not None:
        valid_mask = ~gdf[non_nan_col].isna()
    else:
        # If no column is specified, all rows are valid.
        valid_mask = np.ones(len(gdf), dtype=bool)

    # Get the subset of coordinates (and their indices) that are valid neighbors
    valid_coords = coords_all[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # 3. Build a cKDTree using only the valid neighbors
    tree = cKDTree(valid_coords)

    # 4. For every building (including those possibly invalid themselves):
    #    - Query the tree to get the K nearest valid buildings
    #    - Store distances and the original row indices of these valid neighbors
    all_distances = []
    all_indices = []
    for i, point in enumerate(coords_all):
        # Query among valid coords only
        dist, idx = tree.query(point, k=knn)
        
        # If knn==1, dist/idx is not an array but a scalar. Convert to array for consistency.
        if knn == 1:
            dist = np.array([dist])
            idx  = np.array([idx])
        
        # Convert "valid space" indices back to "global" gdf row indices
        global_indices = valid_indices[idx]

        all_distances.append(dist)
        all_indices.append(global_indices)

    # 5. Attach results back to the GeoDataFrame
    gdf[f'{knn}-nn-idx'] = all_indices
    gdf[f'{knn}-dist'] = all_distances

    return gdf

def compute_knn_aggregate(building_nodes, attr_cols):
    """Helper function to compute aggregate mean and standard deviation for k-nearest neighbours.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame that consists of buildings footprints.
        attr_cols (list): List of columns that contain target attributes to generate knn aggregate statistics. 

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame consisting of computed knn aggregate statistics.
    """    

    # Get knn index column
    target_cols = []
    for i in building_nodes.columns:
        if 'nn-idx' in i:
            target_cols.append(i)
            
    # For each attribute, create mapping dictionary {index: value}
    for attr in attr_cols:
        mapping_dict_attr = {k:v for k,v in zip(building_nodes.index, building_nodes[attr])}
        
        for target_col in target_cols: 
            mean_list = []
            std_list = []
            for list_idx in building_nodes[target_col]:
                total = []
                for i in list_idx:
                    total.append(mapping_dict_attr[i])
                mean_list.append(np.mean(total))
                std_list.append(np.std(total, ddof=1))
            df = pd.DataFrame({f'{target_col}_{attr}_mean':mean_list, f'{target_col}_{attr}_stdev':std_list})
            building_nodes = gpd.GeoDataFrame(pd.concat([building_nodes,df],axis=1))
            # building_nodes[f'{target_col}_{attr}_mean'] = mean_list
            # building_nodes_attr[f'{target_col}_{attr}_stdev'] = std_list
    
    return building_nodes

def preprocess_osm_building_geometry(buildings, minimum_area=30, prefix='osm'):
    """Experimental function. Helper function to preprocess OSM or Overture building footprint data. The function first converts all geometry time to Polygons, 
    checks the validity of each Polygon object, applies local projection, and removes buildings with area less than a specified minimum area.

    Args:
        overture_buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame consisting of raw Overture building footprints.
        minimum_area (int, optional): Area theshold for filtering. Buildings with area below minimum value will be filtered out. Defaults to 30.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame consisting of geometrically processed Overture building footprints.
    """    

    # Remove linestrings, explode multipolygons, and remove invalid polygons
    building_polygons = fill_and_expand(buildings)
    building_polygons = building_polygons[building_polygons.geometry.is_valid]
    
    # Get total number of buildings
    total_buildings = len(building_polygons)
    # print(f'Total number of buildings in osm-building-dataset is: {total_buildings}.')
    
    # Add prefix to all columns to facilitate spatial overlay operations (no duplicate key)
    building_polygons.columns = [f'{prefix}_'+i if i!='geometry' else i for i in building_polygons.columns]
    
    # Locally project building polygons
    building_proj = project_gdf(building_polygons)
    
    # Compute building footprint area
    building_proj[f'{prefix}_'+'original_area'] = building_proj.geometry.area
    
    # Filter out buildings with footprint area less than 30 sqm
    building_geom_gdf = building_proj[building_proj[f'{prefix}_'+'original_area'] >= minimum_area]
    # print(f'Removed {total_buildings - len(building_geom_gdf)} buildings with area less than {minimum_area} sqm.')
    # print(f'Resulting number of buildings in osm-building-dataset is: {len(building_geom_gdf)}.')
    
    # Reset index 
    building_geom_gdf.index = range(len(building_geom_gdf))
    
    return building_geom_gdf



def assign_numerical_id_suffix(gdf, prefix='osm'):
    """Experimental function. Helper function to assign unique building ids to building footprints. Items with duplicate ids are assigned suffixes corresponding to their count "_(count)". 
    For example, if two building polygons have the id: 12093210, the first will be renamed to 12093210_1 and second to 12093210_2.

    Args:
        gdf (gpd.GeoDataFrame): A geopandas GeoDataFrame with duplicate column id due to conversion between MultiPolygon to Polygons.
        prefix (str): Specifies whether the GeoDataFrame assigned corresponds to Overture (overture) or OSM (osm) building footprints. 

    Returns:
        gpd.GeoDataFrame: Returns a modified GeoDataFrame with unique building footprint ids.
    """    
    modified_gdf = gdf.copy()

    if prefix == 'osm':
        modified_gdf[f'{prefix}_id'] = modified_gdf[f'{prefix}_id'].astype(str)
    elif prefix == 'overture':
        modified_gdf[f'{prefix}_id'] = modified_gdf[f'{prefix}_id'].astype(str)

    original_ids = list(modified_gdf[f'{prefix}_id'].value_counts().index[modified_gdf[f'{prefix}_id'].value_counts() > 1])
    
    for target in original_ids:
        indices = modified_gdf.loc[modified_gdf[f'{prefix}_id'] == target, f'{prefix}_id'].index
        
        for i, idx in enumerate(indices):
            modified_gdf.loc[idx, f'{prefix}_id'] = f'{target}_{i}'

    return modified_gdf


# The following code sections for building morphology attribute computation have been adapted from the Momepy Python Package at: https://github.com/pysal/momepy. It has been wrapped into Urbanity in such a manner for convenience, since directly importing momepy created to geospatial dependency issues. 
# Copyright (c) 2018-2021, Martin Fleischmann and PySAL Developers. All rights reserved. 
_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def compute_complexity(building_nodes, element = 'building'):
    """Wrapper function to compute complexity for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint complexity attribute added. 
    """  
    building_nodes[f'{element}_complexity'] = building_nodes[f'{element}_perimeter'] / np.sqrt(np.sqrt(building_nodes[f'{element}_area']))

    return building_nodes


def compute_squareness(building_nodes, element = 'building'):
    """Wrapper function to compute squareness for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint squareness attribute added. 
    """  
    results_list = []
    # fill new column with the value of area, iterating over rows one by one
    for geom in building_nodes.geometry:
        if geom.geom_type == "Polygon" or (
            geom.geom_type == "MultiPolygon" and len(geom.geoms) == 1
        ):
            # unpack multis with single geoms
            if geom.geom_type == "MultiPolygon":
                geom = geom.geoms[0]
            results_list.append(_calc(geom))
        else:
            results_list.append(np.nan)

    building_nodes[f'{element}_squareness'] = results_list
    return building_nodes

def compute_shape_index(building_nodes, element = 'building'):
    """Wrapper function to compute shape index for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint shape index attribute added. 
    """  
    building_nodes[f'{element}_shape_idx'] = np.sqrt((building_nodes[f'{element}_area']/math.pi))/(0.5 * building_nodes[f'{element}_longest_axis_length'])

    return building_nodes

def compute_square_compactness(building_nodes, element = 'building'):
    """Wrapper function to compute square compactness for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint square compactness attribute added. 
    """  

    building_nodes[f'{element}_square_compactness'] = ((4 * np.sqrt(building_nodes[f'{element}_area'])) / building_nodes[f'{element}_perimeter'])**2


    return building_nodes


def compute_rectangularity(building_nodes, element = 'building'):
    """Wrapper function to compute rectangularity for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint rectangularity attribute added. 
    """  

    mbr_list = get_minimum_bounding_rectangle(building_nodes)
    area_list = [i.area for i in mbr_list]
    building_nodes[f'{element}_rectangularity'] = building_nodes[f'{element}_area'] / area_list

    return building_nodes


def compute_fractaldim(building_nodes, element = 'building'):
    """Wrapper function to compute fractal dimension for each building polygons from momepy: https://github.com/pysal/momepy. Based on Kevin McGarigal and Barbara J Marks. FRAGSTATS: Spatial Pattern Analysis Program for Quantifying Landscape Structure. Volume 351. US Department of Agriculture, Forest Service, Pacific Northwest Research Station, Portland, OR, 1995. doi:10.2737/PNW-GTR-351.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint fractal dimension attribute added. 
    """  

    building_nodes[f'{element}_fractaldim'] = (2 * np.log(building_nodes[f'{element}_perimeter'] / 4)) / np.log(building_nodes[f'{element}_area'])

    return building_nodes


def compute_equivalent_rectangular_index(building_nodes, element = 'building'):
    """Wrapper function to compute equivalent rectangular index for each building polygons from momepy: https://github.com/pysal/momepy. Based on Melih Basaraner and Sinan Cetinkaya. Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS. International Journal of Geographical Information Science, 31(10):1952–1977, July 2017. doi:10.1080/13658816.2017.1346257.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint equivalent rectangular index attribute added. 
    """  

    mbr_list = get_minimum_bounding_rectangle(building_nodes)
    building_nodes[f'{element}_eri'] = np.sqrt(building_nodes[f'{element}_area'] / [i.area for i in mbr_list]) * ([i.length for i in mbr_list] / building_nodes[f'{element}_perimeter'])
    return building_nodes


def compute_longest_axis_length(building_nodes, element = 'building'):
    """Wrapper function to compute longest axis length for each building polygons from momepy: https://github.com/pysal/momepy. Axis is defined as a diameter of minimal circumscribed circle around the convex hull. It does not have to be fully inside an object.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attribute. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint longest axis length attribute added. 
    """  
    hulls = building_nodes.convex_hull.exterior
    diameter = hulls.apply(lambda g: _circle_radius(list(g.coords))) * 2
    building_nodes[f'{element}_longest_axis_length'] = diameter

    return building_nodes
    
def compute_shared_wall_ratio(building_nodes, element = 'building'):
    """Wrapper function to compute shared wall length and ratio for each building polygons from momepy: https://github.com/pysal/momepy. From: Rachid Hamaina, Thomas Leduc, and Guillaume Moreau. Towards Urban Fabrics Characterization Based on Buildings Footprints. In Bridging the Geographic Information Sciences, volume 2, pages 327–346. Springer, Berlin, Heidelberg, January 2012. doi:10.1007/978-3-642-29063-3_18.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint shared wall length and ratio attribute added. 
    """  
    inp, res = building_nodes.sindex.query_bulk(building_nodes.geometry, predicate="intersects")
    left = building_nodes.geometry.take(inp).reset_index(drop=True)
    right = building_nodes.geometry.take(res).reset_index(drop=True)
    intersections = left.intersection(right).length
    results = intersections.groupby(inp).sum().reset_index(
        drop=True
    ) - building_nodes.geometry.length.reset_index(drop=True)

    building_nodes[f'{element}_swl'] = results
    building_nodes[f'{element}_swl_ratio'] = building_nodes[f'{element}_swl'] / building_nodes[f'{element}_perimeter']

    return building_nodes


def compute_orientation(building_nodes, element = 'building'):
    """Wrapper function to compute orientation for each building polygons from momepy: https://github.com/pysal/momepy. Here 'orientation' is defined as an orientation of the
    longest axis of bounding rectangle in range 0 - 45. The orientation of LineStrings is represented by the orientation of the line connecting the first and the last point of the segment.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint orientation attribute added. 
    """  
    results_list = []
    mbr_list = get_minimum_bounding_rectangle(building_nodes)

    for geom, bbox in zip(building_nodes.geometry, mbr_list):
        if geom.geom_type in ["Polygon", "MultiPolygon", "LinearRing"]:
            bbox = bbox.exterior.coords
            axis1 = _dist(bbox[0], bbox[3])
            axis2 = _dist(bbox[0], bbox[1])

            if axis1 <= axis2:
                az = _azimuth(bbox[0], bbox[1])
            else:
                az = _azimuth(bbox[0], bbox[3])
        elif geom.geom_type in ["LineString", "MultiLineString"]:
            coords = geom.exterior.coords
            az = _azimuth(coords[0], coords[-1])
        else:
            results_list.append(np.nan)
            continue

        results_list.append(az)

    # get a deviation from cardinal directions
    results = np.abs((np.array(results_list, dtype=float) + 45) % 90 - 45)
    building_nodes[f'{element}_orientation'] = results

    return building_nodes

def compute_elongation(building_nodes, element = 'building'):
    """Wrapper function to compute elongation for each building polygons from momepy: https://github.com/pysal/momepy

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for footprint elongation attribute added. 
    """  
    # Get mbr from building footprints
    mbr_list = get_minimum_bounding_rectangle(building_nodes)

    # Formula based on Jorge Gil, Nuno Montenegro, J N Beirão, and J P Duarte. On the Discovery of Urban Typologies: Data Mining the Multi-dimensional Character of Neighbourhoods. Urban Morphology, 16(1):27–40, January 2012.
    a_list = [i.area for i in mbr_list]
    p_list = [k.length for k in mbr_list]
    a_array = np.array(a_list)
    p_array = np.array(p_list)
    cond1 = p_array ** 2
    cond2 = a_array * 16
    bigger = cond1 >= cond2
    sqrt = np.empty(len(a_array))
    sqrt[bigger] = cond1[bigger] - cond2[bigger]
    sqrt[~bigger] = 0

    # calculate both width/length and length/width
    elo1 = ((p_array - np.sqrt(sqrt)) / 4) / ((p_array / 2) - ((p_array - np.sqrt(sqrt)) / 4))
    elo2 = ((p_array + np.sqrt(sqrt)) / 4) / ((p_array / 2) - ((p_array + np.sqrt(sqrt)) / 4))

    # use the smaller one (e.g. shorter/longer)
    res = np.empty(len(a_array))
    res[elo1 <= elo2] = elo1[elo1 <= elo2]
    res[~(elo1 <= elo2)] = elo2[~(elo1 <= elo2)]

    building_nodes[f'{element}_elongation'] = res

    return building_nodes

def compute_corners(building_nodes, element = 'building'):
    """Wrapper function to compute number of corners for each building polygons from momepy: https://github.com/pysal/momepy

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for number of corners attribute added. 
    """  
    results_list = [] 
    # fill new column with the value of area, iterating over rows one by one
    for geom in building_nodes.geometry:
        if geom.geom_type == "Polygon":
            corners = 0  # define empty variables
            points = list(geom.exterior.coords)  # get points of a shape
            stop = len(points) - 1  # define where to stop
            for i in np.arange(
                len(points)
            ):  # for every point, calculate angle and add 1 if True angle
                if i == 0:
                    continue
                elif i == stop:
                    a = np.asarray(points[i - 1])
                    b = np.asarray(points[i])
                    c = np.asarray(points[1])

                    if _true_angle(a, b, c) is True:
                        corners = corners + 1
                    else:
                        continue

                else:
                    a = np.asarray(points[i - 1])
                    b = np.asarray(points[i])
                    c = np.asarray(points[i + 1])

                    if _true_angle(a, b, c) is True:
                        corners = corners + 1
                    else:
                        continue
        elif geom.geom_type == "MultiPolygon":
            corners = 0  # define empty variables
            for g in geom.geoms:
                points = list(g.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue
        else:
            corners = np.nan

        results_list.append(corners)
    
    building_nodes[f'{element}_corners'] = results_list

    return building_nodes

def compute_convexity(building_nodes, element = 'building'):
    """Wrapper function to compute convexity metric building polygons from momepy: https://github.com/pysal/momepy

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for convexity attribute added. 
    """  
    building_nodes[f'{element}_convexity'] = building_nodes[f'{element}_area'] / building_nodes.convex_hull.area
    return building_nodes

def compute_circularcompactness(building_nodes, element = 'building'):
    """Wrapper function to compute circular compactness of building polygons from momepy: https://github.com/pysal/momepy

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame which consists of building polygons and their computed attributes. 

    Returns:
        gpd.GeoDataFrame: Modified geopandas GeoDataFrame with column for Circular Compactness attribute added. 
    """   
    hull = building_nodes.convex_hull.exterior
    radius = hull.apply(
            lambda g: _circle_radius(list(g.coords)) if g is not None else None
        )
    building_nodes[f'{element}_circ_compact'] = building_nodes[f'{element}_area'] / (np.pi * radius**2)
    return building_nodes


def _circle_radius(points): 
    """Helper function to generate geometric circles from points. 

    Args:
        points (list): A list of shapely Point objects.

    Returns:
        list: List of shapely Polygons corresponding to circles around points. 
    """    
    if len(points[0]) == 3:
        points = [x[:2] for x in points]
    circ = _make_circle(points)
    return circ[2]

def _make_circle(points):
    """Helper function to generate circles from list of shapely points. 

    Args:
        points (list): List of shapely Points.

    Returns:
        list: List of circle Polygons. 
    """    
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for i, p in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c

def _make_circle_one_point(points, p):
    """Helper function to generate circles from list of shapely points when one boundary point is known.

    Args:
        points (list): List of shapely Points.
        p (shapely.geometry.Point): Boundary point

    Returns:
        list: List of circle Polygons. 
    """    
    c = (p[0], p[1], 0)
    for i, q in enumerate(points):
        if not _is_in_circle(c, q):
            if c[2] == 0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c

def _make_circle_two_points(points, p, q):
    """Helper function to generate circles from list of shapely points when two boundary points are known.

    Args:
        points (list): List of shapely Points.
        p (shapely.geometry.Point): First boundary point.
        q (shapely.geometry.Point): Second boundary point

    Returns:
        list: List of circle Polygons. 
    """    
    circ = _make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if _is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = _make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0 and (
            left is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            > _cross_product(px, py, qx, qy, left[0], left[1])
        ):
            left = c
        elif cross < 0 and (
            right is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            < _cross_product(px, py, qx, qy, right[0], right[1])
        ):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    if left is None:
        return right
    if right is None:
        return left
    if left[2] <= right[2]:
        return left
    return right


def _make_circumcircle(p0, p1, p2):
    """Helper function to generate a circumscribed circle bounded by three points. Mathematical algorithm from Wikipedia: Circumscribed circle.

    Args:
        p0 (shapely.geometry.Point): First boundary point.
        p1 (shapely.geometry.Point): Second boundary point.
        p2 (shapely.geometry.Point): Third boundary point.

    Returns:
        tuple: Returns a set of values corresponding to circle center and its radius.
    """    
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2
    ax -= ox
    ay -= oy
    bx -= ox
    by -= oy
    cx -= ox
    cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2
    if d == 0:
        return None
    x = (
        ox
        + (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        )
        / d
    )
    y = (
        oy
        + (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        )
        / d
    )
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))

def _is_in_circle(c, p):
    """Helper function to check if a point is within a circle

    Args:
        c (tuple): Tuple consisting of circle's x, y, and radius. 
        p (shapely.geometry.Point): Shapely point. 

    Returns:
        bool: Returns True if point is within circle. 
    """    
    return (
        c is not None
        and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON
    )

def _cross_product(x0, y0, x1, y1, x2, y2):
    """Returns twice the signed area of the
    triangle defined by (x0, y0), (x1, y1), (x2, y2).

    Args:
        x0 (_type_): x coordinate of first point.
        y0 (_type_): y coordinate of first point.
        x1 (_type_): x coordinate of second point.
        y1 (_type_): y coordinate of second point.
        x2 (_type_): x coordinate of third point.
        y2 (_type_): y coordinate of third point.

    Returns:
        float: Twice signed area of triangle.
    """    
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def _make_diameter(p0, p1):
    """Helper function to return a circle between two points.

    Args:
        p0 (shapely.geometry.Point): First Shapely point. 
        p1 (shapely.geometry.Point): Second Shapely point. 

    Returns:
        tuple: Returns a set of values corresponding to the mid point of two points and its radius. 
    """    
    cx = (p0[0] + p1[0]) / 2
    cy = (p0[1] + p1[1]) / 2
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))

def _true_angle(a, b, c):
    """Returns the true angle between three points. 

    Args:
        a (np.array): Point one.
        b (np.array): Point two.
        c (np.array): Point three.

    Returns:
        bool: Returns true angle. If degree <= 170 or >= 190, return True, else False. 
    """    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # TODO: add arg to specify these values
    if np.degrees(angle) <= 170:
        return True
    if np.degrees(angle) >= 190:
        return True
    return False


def get_minimum_bounding_rectangle(building_nodes):
    """Helper function to generate minimum bounding rectangle around each Polygon.

    Args:
        building_nodes (gpd.GeoDataFrame): A geopandas GeoDataFrame corresponding to building footprints. 

    Returns:
        list: A list consisting of minimum bounding rectangles (shapely.geometry.Polygon) for each building footprint. 
    """    
    # Adapted and modified to work with gpd.GeoDataFrame input (Original algorithm:  https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput answered by user JesseBuesking)
    
    from scipy.ndimage import rotate
    pi2 = np.pi/2
    
    # Get convex hull coordinates
    convex_hull_points = [np.array(i.coords) for i in building_nodes.convex_hull.exterior] 
    
    # Get vector between each point by subtraction
    edges = [point[1:] - point[:-1] for point in convex_hull_points]
    
    # Compute angle between vectors
    angles = [np.arctan2(edge[:, 1], edge[:, 0]) for edge in edges]
    angles = [np.abs(np.mod(angle, pi2)) for angle in angles]
    angles = [np.unique(angle) for angle in angles]
    
    mbr_list = []
    
    for angle, hull_coords in zip(angles, convex_hull_points):
        rotations = np.vstack([
                np.cos(angle),
                np.cos(angle-pi2),
                np.cos(angle+pi2),
                np.cos(angle)]).T

        rotations = rotations.reshape((-1, 2, 2))
        
        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_coords.T)
        
        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)
        
        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        
        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)
        
        mbr_list.append(Polygon(rval))
        
    return mbr_list

def _dist(a, b):
    """Returns the Euclidean distance between two points.

    Args:
        a (shapely.geometry.Point): Shapely Point.
        b (shapely.geometry.Point): Shapely Point.

    Returns:
        float: Distance between two points.
    """    
    return math.hypot(b[0] - a[0], b[1] - a[1])

def _azimuth(point1, point2):
    """Return the azimuth between 2 shapely points (interval 0 - 180).

    Args:
        point1 (shapely.geometry.Point): Shapely Point.
        point2 (shapely.geometry.Point): Shapely Point.

    Returns:
        float: Azimuth between two points.
    """
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) % 180

def _make_linestring(centroid, angle, length):
    """Helper function to generate linestring with given length and angle. 

    Args:
        centroid (tuple): Tuple containing x and y coordinates.
        angle (float): Angle of vector.
        length (float): Length of vector.

    Returns:
        shapely.geometry.LineString: Linestring object.
    """    
    x, y = centroid[0], centroid[1]
    endy = y + length * math.sin(angle)
    endx = x + length * math.cos(angle)
    fromy = y - length * math.sin(angle)
    fromx = x - length * math.cos(angle)

    return LineString([(fromx,fromy),(x,y), (endx, endy)])

def _angle(a, b, c):
    """Helper function to compute angle between vectors.

    Args:
        a (np.array): Numerical vector a.
        b (np.array): Numerical vector b.
        c (np.array): Numerical vector c.

    Returns:
        float: Angle between vectors.
    """    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def _calc(geom):
    """Helper function to calculate the angle of deviation between points.

    Args:
        geom (shapely.geometry.Polygon): Shapely polygon input.

    Returns:
        float: Mean angle deviation.
    """
    
    
    angles = []
    points = list(geom.exterior.coords)  # get points of a shape
    n_points = len(points)

    if n_points < 3:
        return np.nan

    stop = n_points - 1

    i = 1
    while i < n_points:
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        c = np.asarray(points[i + 1]) if i != stop else np.asarray(points[1])

        ang = _angle(a, b, c)

        if np.isnan(ang):
            # If angle is NaN, remove duplicate points and restart
            points = remove_duplicate_points(points)
            n_points = len(points)
            if n_points < 3:
                return np.nan
            stop = n_points - 1
            i = 0  # restart iteration
            angles = []  # reset angles
            continue

        if ang <= 175 or ang >= 185:
            angles.append(ang)

        i += 1  # increment index normally

    if not angles:
        return np.nan  # if no valid angles were found

    deviations = [abs(90 - i) for i in angles]
    return np.mean(deviations)

def remove_duplicate_points(points):
        """Remove globally duplicated points but keep one occurrence."""
        seen = set()
        cleaned = []
        for pt in points:
            key = tuple(np.round(pt, decimals=8))  # rounding to avoid floating point issues
            if key not in seen:
                seen.add(key)
                cleaned.append(pt)
        return cleaned

def _calc(geom):
    """Helper function to calculate the angle of deviation between points.

    Args:
        geom (shapely.geometry.Polygon): Shapely polygon input.

    Returns:
        float: Mean angle deviation.
    """    
    angles = []
    points = list(geom.exterior.coords)  # get points of a shape
    n_points = len(points)
    if n_points < 3:
        return np.nan
    stop = n_points - 1
    for i in range(
        1, n_points
    ):  # for every point, calculate angle and add 1 if True angle
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        # in last case, needs to wrap around start to find finishing angle
        c = np.asarray(points[i + 1]) if i != stop else np.asarray(points[1])
        ang = _angle(a, b, c)

        if ang <= 175 or ang >= 185:
            angles.append(ang)
        else:
            continue
    deviations = [abs(90 - i) for i in angles]
    mean_deviations = np.mean(deviations)

    if np.isnan(mean_deviations):
        angles = []
        points = remove_duplicate_points(points)
        n_points = len(points)
        if n_points < 3:
            return np.nan
        stop = n_points - 1
        for i in range(
            1, n_points
        ):  # for every point, calculate angle and add 1 if True angle
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            # in last case, needs to wrap around start to find finishing angle
            c = np.asarray(points[i + 1]) if i != stop else np.asarray(points[1])
            ang = _angle(a, b, c)

            if ang <= 175 or ang >= 185:
                angles.append(ang)
            else:
                continue
        deviations = [abs(90 - i) for i in angles]
        mean_deviations = np.mean(deviations)
        return mean_deviations
    return mean_deviations

def get_building_heights(filepath, target_key):
    dest_crs = 'epsg:3857'

    folder_path = f'./data/'
    data_path = os.path.join(folder_path, target_key)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(data_path):
        heights = gpd.read_parquet(data_path)
    else:
        heights = pd.read_parquet(filepath)
        # Example: df has a column 'geom_wkb' with WKB in bytes
        heights['geometry'] = heights['geometry'].apply(wkb.loads)
        heights = gpd.GeoDataFrame(heights, crs='epsg:4326', geometry='geometry')
        heights.to_parquet(data_path)

    return heights

def assign_building_heights(heights, building_gdf):
    dest_crs = 'epsg:3857'
    
    proj_building_gdf = building_gdf.to_crs(dest_crs)
    proj_heights = heights.to_crs(dest_crs)
    
    # Add area
    res_intersection = proj_heights.overlay(proj_building_gdf)

    # Calculate weighted average height per id
    weighted_heights = res_intersection.groupby('bid')[['Height']].mean().reset_index()

    proj_building_gdf = proj_building_gdf.merge(weighted_heights, on='bid', how='left')

    # # Add building centroid for knn computation
    proj_building_gdf['bid_centroid'] = proj_building_gdf.geometry.centroid
    proj_building_gdf = building_knn_nearest(proj_building_gdf, knn=1, non_nan_col='Height')
    proj_building_gdf = compute_knn_aggregate(proj_building_gdf, ['Height'])
    proj_building_gdf = proj_building_gdf.drop(columns = ['Height', 'bid_centroid', '1-nn-idx', '1-dist', '1-nn-idx_Height_stdev'])
    proj_building_gdf = proj_building_gdf.rename(columns = {'1-nn-idx_Height_mean':'bid_height'})
    return proj_building_gdf.to_crs('epsg:4326')

def get_and_assign_building_heights(filepath, target_key, building_gdf):

    dest_crs = 'epsg:3857'

    folder_path = f'./data/'
    data_path = os.path.join(folder_path, target_key)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(data_path):
        heights = gpd.read_parquet(data_path)
    else:
        heights = pd.read_parquet(filepath)
        # Example: df has a column 'geom_wkb' with WKB in bytes
        heights['geometry'] = heights['geometry'].apply(wkb.loads)
        heights = gpd.GeoDataFrame(heights, crs='epsg:4326', geometry='geometry')
        heights.to_parquet(data_path)

    proj_building_gdf = building_gdf.to_crs(dest_crs)
    proj_heights = heights.to_crs(dest_crs)
    
    # Add area
    res_intersection = proj_heights.overlay(proj_building_gdf)

    # Calculate weighted average height per id
    weighted_heights = res_intersection.groupby('bid')[['Height']].mean().reset_index()

    proj_building_gdf = proj_building_gdf.merge(weighted_heights, on='bid', how='left')

    # # Add building centroid for knn computation
    proj_building_gdf['bid_centroid'] = proj_building_gdf.geometry.centroid
    proj_building_gdf = building_knn_nearest(proj_building_gdf, knn=1, non_nan_col='Height')
    proj_building_gdf = compute_knn_aggregate(proj_building_gdf, ['Height'])
    proj_building_gdf = proj_building_gdf.drop(columns = ['Height', 'bid_centroid','1-nn-idx', '1-dist', '1-nn-idx_Height_stdev'])
    proj_building_gdf = proj_building_gdf.rename(columns = {'1-nn-idx_Height_mean':'bid_height'})
    return proj_building_gdf.to_crs('epsg:4326')