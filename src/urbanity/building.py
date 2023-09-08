import os
import json
import pyrosm
import pandas as pd
from pyrosm import get_data
import pkg_resources
import numpy as np
import geopandas as gpd
from urbanity.geom import fill_and_expand, project_gdf

def get_overture_buildings(location):
    """Helper function to load Overture building footprints. Returns a GeoDataFrame with raw Overture building footprint data.

    Args:
        location (str): Specific city name to retrieve Overture building footprint data.

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame containing building footprints for the specified city. 
    """    
    # Get Filepath to Building Data
    overture_buildings_dl_path = pkg_resources.resource_filename('urbanity', 'overture_data/overture_building.json')
    
    # Load Overture Building Footprints
    with open('../src/urbanity/overture_data/overture_building.json') as f:
        overture_buildings = json.load(f)
        
    overture_buildings_gdf = gpd.read_file(overture_buildings[f'{location.lower()}_buildings.geojson'])
    
    return overture_buildings_gdf


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

    if isinstance(boundary, str):
        bounding_box = gpd.read_file(boundary)

    if fp == '' and location != '':
        fp = get_data(location, directory = './data')
        osm = pyrosm.OSM(fp, bounding_box=bounding_box.geometry.values[0])
    elif fp == '' and location == '':
        raise ValueError("Please specify a valid city or country name.")
    else:
        osm = pyrosm.OSM(fp, bounding_box=bounding_box.geometry.values[0])

    osm_buildings = osm.get_buildings()
    
    return osm_buildings
    

def preprocess_overture_building_geometry(overture_buildings_gdf, minimum_area=30):
    """Helper function to preprocess Overture building footprint data. The function first converts all geometry time to Polygons, 
    checks the validity of each Polygon object, applies local projection, and removes buildings with area less than a specified minimum area.

    Args:
        overture_buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame consisting of raw Overture building footprints.
        minimum_area (int, optional): Area theshold for filtering. Buildings with area below minimum value will be filtered out. Defaults to 30.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame consisting of geometrically processed Overture building footprints.
    """    
    
    # Remove linestrings, explode multipolygons, and remove invalid polygons
    building_polygons = fill_and_expand(overture_buildings_gdf)
    building_polygons = building_polygons[building_polygons.geometry.is_valid]

    # Get total number of buildings
    total_buildings = len(building_polygons)
    print(f'Total number of buildings in overture-building-dataset is: {total_buildings}.')
    
    # Add overture prefix to all columns to facilitate spatial overlay operations (no duplicate key)
    building_polygons.columns = ['overture_'+i if i!='geometry' else i for i in building_polygons.columns]
    
    # Locally project building polygons
    building_proj = project_gdf(building_polygons)
    
    # Compute building footprint area
    building_proj['overture_original_area'] = building_proj.geometry.area
    
    # Filter out buildings with footprint area less than 30 sqm
    building_geom_gdf = building_proj[building_proj['overture_original_area'] >= minimum_area]
    print(f'Removed {total_buildings - len(building_geom_gdf)} buildings with area less than {minimum_area} sqm.')
    print(f'Resulting number of buildings in overture-building-dataset is: {len(building_geom_gdf)}.')
    
    # Reset index 
    building_geom_gdf.index = range(len(building_geom_gdf))

    return building_geom_gdf


def preprocess_osm_building_geometry(osm_buildings, minimum_area=30):
    """Helper function to preprocess OSM building footprint data. The function first converts all geometry time to Polygons, 
    checks the validity of each Polygon object, applies local projection, and removes buildings with area less than a specified minimum area.

    Args:
        overture_buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame consisting of raw Overture building footprints.
        minimum_area (int, optional): Area theshold for filtering. Buildings with area below minimum value will be filtered out. Defaults to 30.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame consisting of geometrically processed Overture building footprints.
    """    

    # Remove linestrings, explode multipolygons, and remove invalid polygons
    building_polygons = fill_and_expand(osm_buildings)
    building_polygons = building_polygons[building_polygons.geometry.is_valid]
    
    # Get total number of buildings
    total_buildings = len(building_polygons)
    print(f'Total number of buildings in osm-building-dataset is: {total_buildings}.')
    
    # Add prefix to all columns to facilitate spatial overlay operations (no duplicate key)
    building_polygons.columns = ['osm_'+i if i!='geometry' else i for i in building_polygons.columns]
    
    # Locally project building polygons
    building_proj = project_gdf(building_polygons)
    
    # Compute building footprint area
    building_proj['osm_original_area'] = building_proj.geometry.area
    
    # Filter out buildings with footprint area less than 30 sqm
    building_geom_gdf = building_proj[building_proj['osm_original_area'] >= minimum_area]
    print(f'Removed {total_buildings - len(building_geom_gdf)} buildings with area less than {minimum_area} sqm.')
    print(f'Resulting number of buildings in osm-building-dataset is: {len(building_geom_gdf)}.')
    
    # Reset index 
    building_geom_gdf.index = range(len(building_geom_gdf))
    
    return building_geom_gdf


def preprocess_overture_building_attributes(building_geom_gdf, return_class_height=False):
    """Helper function to preprocess Overture building attribute data. Replaces missing values with zero, and combines elevation data across columns.

    Args:
        building_geom_gdf (gpd.GeoDataFrame): A geopandas GeoDataframe of Overture building footprints.
        return_class_height (bool, optional): If True, returns the mapping between building type and average floor height. 
    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataframe with pre-processed building attributes information.
    """    

    # Create a building geodataframe copy
    building_geom_gdf_nan = building_geom_gdf.copy()
    
    # Assign "0" value to height column with "None" and cast as float
    building_geom_gdf_nan.loc[building_geom_gdf_nan["overture_height"].isna(), "overture_height"] = "0"
    building_geom_gdf_nan['overture_height'] = building_geom_gdf_nan['overture_height'].astype(float)
    
    def get_level(row):
        if '+' in row:
            level = eval(row)
        else:
            level = row.split(',')[0]
        return level

    # Assign "0" value to level column with NaN and cast as float
    building_geom_gdf_nan.loc[building_geom_gdf_nan["overture_level"].isna(), "overture_level"] = "0"
    building_geom_gdf_nan['overture_level'] = building_geom_gdf_nan['overture_level'].astype(float)
    
    # Assign "0" value to level column with NaN and cast as float
    building_geom_gdf_nan.loc[building_geom_gdf_nan["overture_numfloors"].isna(), "overture_numfloors"] = "0"
    building_geom_gdf_nan['overture_numfloors'] = building_geom_gdf_nan['overture_numfloors'].astype(float)
    
    # Harmonise columns with level and numfloors information
    def combine_level(level, numfloors):
        if level == 0:
            return numfloors
        else:
            return level
        
    building_geom_gdf_nan['overture_combined_level'] = building_geom_gdf_nan.apply(lambda row: combine_level(row['overture_level'], row['overture_numfloors']), axis=1)

    # Remove underground buildings
    building_geom_gdf_above_ground = building_geom_gdf_nan.copy()
    building_geom_gdf_above_ground = building_geom_gdf_above_ground[building_geom_gdf_above_ground['overture_combined_level'] >= 0]
    
    # Get average height across building classes
    class_list = [i for i in building_geom_gdf_above_ground['overture_class'].unique() if i!=None]
    
    average_class_height = {}
    for building_class in class_list:
        temp = building_geom_gdf_above_ground[(building_geom_gdf_above_ground['overture_class'] == building_class) & 
                                              (building_geom_gdf_above_ground['overture_height']>0) & 
                                              (building_geom_gdf_above_ground['overture_combined_level']>0)]
        
        average_class_height[building_class] = round((temp['overture_height'] / temp['overture_combined_level']).mean(),3)
    
    temp = building_geom_gdf_above_ground[(building_geom_gdf_above_ground['overture_height']>0) & 
                                          (building_geom_gdf_above_ground['overture_combined_level']>0)]
    
    average_class_height['total'] = round((temp['overture_height'].sum() / temp['overture_combined_level'].sum()), 3)

    # Harmonise columns with building height and combined_level information
    def combine_height(height, combined_level, building_class):
        if height!=0:
            return height
        
        elif (combined_level!=0) and (building_class in class_list):
            if np.isnan(average_class_height[building_class]):
                multiplier = average_class_height['total']
            else: 
                multiplier = average_class_height[building_class]
            return combined_level * multiplier
    
        elif (combined_level!=0) and (building_class not in class_list):
            multiplier = average_class_height['total']
            return combined_level * multiplier
        
        else:
            return 0
    
    building_geom_gdf_above_ground['overture_combined_heights'] = building_geom_gdf_above_ground.apply(lambda row: combine_height(row['overture_height'], 
                                                                                                                                   row['overture_combined_level'],
                                                                                                                                   row['overture_class']), 
                                                                                                                                   axis=1)
    
    
    # Remove outbuildings class
    building_geom_gdf_above_ground = building_geom_gdf_above_ground[building_geom_gdf_above_ground['overture_class'] != 'outbuilding']
    
    if return_class_height:
        return building_geom_gdf_above_ground, average_class_height

    return building_geom_gdf_above_ground


def preprocess_osm_building_attributes(building_geom_gdf, return_class_height=False):
    """Helper function to preprocess OSM building attribute data. Replaces missing values with zero, and combines elevation data across columns.

    Args:
        building_geom_gdf (gpd.GeoDataFrame): A geopandas GeoDataframe of OSM building footprints.
        return_class_height (bool, optional): If True, returns the mapping between building type and average floor height. 

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataframe with pre-processed building attributes information.
    """    
    
    # Create a building geodataframe copy
    building_geom_gdf_nan = building_geom_gdf.copy()
    
    # Assign "0" value to height column with "None" and cast as float
    building_geom_gdf_nan.loc[building_geom_gdf_nan["osm_height"].isna(), "osm_height"] = "0"
    building_geom_gdf_nan['osm_height'] = building_geom_gdf_nan['osm_height'].astype(float)
    
    # If '+', mathematically evaluate. If multiple building levels, report highest. 
    def get_level(row):
        if '+' in row:
            level = eval(row)
        elif len(row) > 3:
            level = max([float(i) for i in row.split(',')])
        else:
            level = row
        return level

    # Assign "0" value to level column with NaN and cast as float
    building_geom_gdf_nan.loc[building_geom_gdf_nan["osm_building:levels"].isna(), "osm_building:levels"] = "0"
    building_geom_gdf_nan['osm_building:levels'] = building_geom_gdf_nan['osm_building:levels'].apply(get_level)
    building_geom_gdf_nan['osm_building:levels'] = building_geom_gdf_nan['osm_building:levels'].astype(float)

    # Remove underground buildings
    building_geom_gdf_above_ground = building_geom_gdf_nan.copy()
    building_geom_gdf_above_ground = building_geom_gdf_above_ground[building_geom_gdf_above_ground['osm_building:levels'] >= 0]
    
    
    # Get average height across building classes
    cols = ['osm_building', 'osm_amenity', 'osm_building:use', 'osm_craft', 'osm_landuse', 'osm_office', 'osm_shop']
    building_geom_gdf_above_ground[cols] = building_geom_gdf_above_ground[cols].astype('string')
    building_geom_gdf_above_ground['osm_class'] = building_geom_gdf_above_ground[cols].apply(lambda row: '|'.join([i for i in row.values.astype(str) if i not in ['<NA>','yes']]), axis=1)
    
    # Generated with ChatGPT3.5. Text prompt: Please help to assign every element of the following list: <list> to one and only one categories in the following list: <list>. Each item should only be mapped on one category. Please return your response as a python dictionary. 
    element_to_category = {
    'train_station': 'transportation',
    'civic': 'civic',
    'mall': 'commercial',
    'retail|mall': 'commercial',
    'public|library': 'civic',
    'public': 'civic',
    'commercial': 'commercial',
    'police': 'civic',
    'office': 'commercial',
    'place_of_worship': 'religious',
    'fire_station': 'civic',
    'industrial': 'industrial',
    'electronics': 'industrial',
    'supermarket': 'commercial',
    'hospital': 'medical',
    'apartments': 'residential',
    'residential': 'residential',
    'parking': 'transportation',
    'retail': 'commercial',
    'transportation': 'transportation',
    'retail|fast_food': 'commercial',
    'commercial|food_court': 'commercial',
    'commercial|mall': 'commercial',
    'hotel': 'commercial',
    'office|diplomatic': 'commercial',
    'house|government': 'civic',
    'food_court': 'commercial',
    'government': 'civic',
    'garage|parking': 'transportation',
    'company': 'commercial',
    'office|company': 'commercial',
    'library': 'civic',
    'toilets': 'service',
    'retail|food_court': 'commercial',
    'fuel': 'service',
    'university': 'education',
    'bank': 'commercial',
    'warehouse': 'industrial',
    'social_facility': 'civic',
    'hospital|hospital': 'medical',
    'temple': 'religious',
    'office|insurance': 'commercial',
    'insurance': 'commercial',
    'financial': 'commercial',
    'conference_centre': 'civic',
    'service': 'service',
    'school|college': 'education',
    'religion|place_of_worship': 'religious',
    'car': 'transportation',
    'warehouse|company': 'industrial',
    'community_centre': 'civic',
    'grandstand': 'entertainment',
    'clinic': 'medical',
    'greenhouse': 'agricultural',
    'restaurant': 'entertainment',
    'retail|restaurant': 'commercial',
    'church|place_of_worship': 'religious',
    'shelter': 'civic',
    'it': 'service',
    'retail|retail': 'commercial',
    'retail|doctors|retail': 'commercial',
    'retail|cafe': 'commercial',
    'retail|ice_cream': 'commercial',
    'retail|retail|furniture': 'commercial',
    'retail|school': 'commercial',
    'retail|pharmacy': 'commercial',
    'retail|restaurant|retail': 'commercial',
    'retail|optician': 'commercial',
    'retail|school|retail': 'commercial',
    'retail|pub|retail': 'commercial',
    'retail|restaurant|retail|supermarket': 'commercial',
    'retail|supermarket': 'commercial',
    'retail|bank|retail': 'commercial',
    'retail|fast_food|retail': 'commercial',
    'retail|retail|hairdresser': 'commercial',
    'retail|bar|retail': 'commercial',
    'apartments|residential': 'residential',
    'parking|carpark': 'transportation',
    'parking|carpark;shops|mall': 'commercial',
    'residential|residential': 'residential',
    'roof': 'civic',
    'parking|parking': 'transportation',
    'residential;commercial': 'commercial',
    'apartments|residential;commercial': 'commercial',
    'residential;commercial|residential;commercial': 'commercial',
    'school': 'education',
    'roof|fuel': 'civic',
    'mosque|place_of_worship': 'religious',
    'parking|parking|carpark': 'transportation',
    'fuel|convenience': 'service',
    'reservoir': 'agricultural',
    'college': 'education',
    'hangar': 'industrial',
    'commercial|residential': 'commercial',
    'retail|residential': 'commercial',
    'commercial|office': 'commercial',
    'MRT': 'transportation',
    'commercial|shophouses': 'commercial',
    'marketplace': 'commercial',
    'commercial|marketplace': 'commercial',
    'commercial|general': 'commercial',
    'public|community_centre': 'civic',
    'sports_centre': 'entertainment',
    'theatre': 'entertainment',
    'church': 'religious',
    'camera': 'entertainment',
    'printing': 'service',
    'fast_food': 'entertainment',
    'public|cafe': 'civic',
    'garage|parking|carpark': 'transportation',
    'cafe': 'entertainment',
    'nursing_home': 'medical',
    'waste_transfer_station': 'service',
    'office|bank': 'commercial',
    'garage': 'industrial',
    'public|place_of_worship': 'religious',
    'public|shelter': 'civic',
    'temple|place_of_worship': 'religious',
    'childcare': 'civic',
    'garage|waste_disposal': 'industrial',
    'car_repair': 'industrial',
    'estate_agent': 'commercial',
    'music_school': 'education',
    'public_building': 'civic',
    'public|townhall': 'civic',
    'public|food_court': 'civic',
    'construction': 'industrial',
    'industrial|company': 'industrial',
    'retail|company': 'commercial',
    'retail|furniture;electronics': 'commercial',
    'kindergarten': 'education',
    'motorcycle': 'transportation',
    'paint': 'industrial',
    'tyres': 'industrial',
    'manufacture': 'industrial',
    'commercial|company': 'commercial',
    'boathouse': 'entertainment',
    'dormitory': 'residential',
    'apartments|parking': 'residential',
    'commercial|car': 'commercial',
    'storage_rental': 'commercial',
    'parking_space': 'transportation',
    'office|telecommunication': 'commercial',
    'arts_centre': 'entertainment',
    'education': 'education',
    'roof|shelter': 'civic',
    'public|marketplace': 'civic',
    'pub': 'entertainment',
    'house': 'residential',
    'waste_disposal': 'service',
    'sports_centre|public_building': 'entertainment',
    'office|mall': 'commercial',
    'public|ferry_terminal': 'civic',
    'retail|parking|mall': 'commercial',
    'roof|place_of_worship': 'civic',
    'terrace': 'residential',
    'convenience': 'service',
    'semidetached_house': 'residential',
    'residential|atm': 'residential',
    'residential|construction': 'residential',
    'chapel|place_of_worship': 'religious',
    'religious': 'religious',
    'roof|theatre': 'civic',
    'ruins': 'civic',
    'commercial|furniture': 'commercial',
    'roof|fuel|convenience': 'civic',
    'hospital|social_facility': 'medical',
    'residential|parking': 'residential',
    'diplomatic': 'civic',
    'metal_construction': 'industrial',
    'bakery': 'industrial',
    'retail|bicycle': 'commercial',
    'furniture': 'industrial',
    'educational_institution': 'education',
    'detached': 'residential',
    'commercial|restaurant': 'commercial',
    'religious|place_of_worship': 'religious',
    'swimming_pool_changing_room|dressing_room': 'entertainment',
    'clubhouse': 'entertainment',
    'club house': 'entertainment',
    'car_wash': 'industrial',
    'civic|community_centre': 'civic',
    'driving_school': 'education',
    'bus_station': 'transportation',
    'ferry_terminal': 'transportation',
    'cinema': 'entertainment',
    'winter_sports': 'entertainment',
    'mobile_phone': 'commercial',
    'jewelry': 'commercial',
    'pawnbroker': 'commercial',
    'retail|restaurant|caterer': 'commercial',
    'beauty': 'entertainment',
    'clothes': 'commercial',
    'toys': 'commercial',
    'frame': 'industrial',
    'fabric': 'industrial',
    'money_lender': 'commercial',
    'doctors': 'medical',
    'retail|pub': 'commercial',
    'pottery': 'industrial',
    'bar': 'entertainment',
    'gift': 'commercial',
    'travel_agency': 'commercial',
    'car_parts': 'industrial',
    'craft': 'industrial',
    'hairdresser': 'entertainment',
    'residential|social_facility': 'residential',
    'condominium': 'residential',
    'residential|variety_store': 'residential',
    'monastery': 'religious',
    'parlour': 'commercial',
    'variety_store': 'commercial',
    'public|waste_disposal': 'service',
    'commercial|car_repair': 'commercial',
    'mix_used|parking': 'transportation',
    'books': 'education',
    'warehouse|storage_rental': 'industrial',
    'retail|florist': 'commercial',
    'hvac': 'industrial',
    'kindergarten|kindergarten': 'education',
    'shop': 'commercial',
    'service|telecommunication': 'service',
    'electrical': 'industrial',
    'energy_supplier': 'industrial',
    'mosque': 'religious',
    'commercial|fuel': 'commercial',
    'Lucasfilm': 'entertainment',
    'commercial|commercial|mall': 'commercial',
    'industrial|waste_transfer_station': 'industrial',
    'retail|convenience': 'commercial',
    'detached|diplomatic': 'residential',
    'interior_decoration': 'commercial',
    'military': 'civic',
    'retail|marketplace': 'commercial',
    'residential|waste_transfer_station': 'residential',
    'office|hospital': 'medical',
    'shed': 'industrial',
    'stable': 'agricultural',
    'crematorium': 'religious',
    'residential|college': 'education',
    'gazebo|shelter': 'civic',
    'residential|place_of_worship': 'religious',
    'meadow': 'agricultural',
    'post_office': 'civic',
    'CET_Campus_East': 'education',
    'roof|community_centre': 'civic',
    'pet': 'agricultural',
    'restaurant|wine': 'entertainment',
    'foundation': 'civic',
    'veterinary': 'medical',
    'winery': 'agricultural',
    'commercial|bank': 'commercial',
    'industrial|furniture': 'industrial',
    'apartments|mall': 'residential',
    'retail|pub|brewery': 'commercial',
    'wayside_shrine': 'religious',
    'retail|dentist': 'commercial',
    'EiS_Residences': 'residential',
    'studio': 'entertainment',
    'school|school': 'education',
    'residential|nursing_home': 'residential',
    'alcohol': 'entertainment',
    'herbalist': 'commercial',
    'research': 'education',
    'roof|mall': 'civic',
    'shop|fuel': 'commercial',
    'telecommunication': 'service',
    'medical': 'medical',
    'car_rental': 'transportation',
    'stadium': 'entertainment',
    'ice_cream': 'entertainment',
    'garage|parking_space': 'transportation',
    'industrial|vehicle_inspection': 'industrial',
    'train_station|mall': 'transportation',
    'IMM|parking': 'transportation',
    'government|community_centre': 'civic',
    'farm_auxiliary': 'agricultural',
    'greengrocer': 'agricultural',
    'shrine': 'religious',
    'engineering': 'industrial',
    'civic|second_hand': 'civic',
    'Temple_Chinese': 'religious',
    'school|educational_institution': 'education',
    'school|place_of_worship': 'religious',
    'semidetached_house|residential': 'residential',
    'hut': 'agricultural',
    'security': 'civic',
    'administration': 'civic',
    'bungalow': 'residential',
    'residential|childcare': 'residential',
    'sports_hall': 'entertainment',
    'charity': 'civic',
    'retail|fuel': 'commercial',
    'hall': 'civic',
    'social_centre': 'civic',
    'ngo': 'civic',
    'carport': 'residential',
    'retail|plant_nursery|garden_centre': 'commercial',
    'retail|garden_centre': 'commercial',
    'house|diplomatic': 'residential',
    'terrace|kindergarten': 'residential',
    'retail|gardener': 'commercial',
    'house|bar': 'residential',
    'fishing': 'agricultural',
    'language_school|educational_institution': 'education',
    'service|waste_transfer_station': 'service',
    'commercial|builder': 'commercial',
    'grocery': 'commercial',
    'industrial|engineering': 'industrial',
    'funeral_directors': 'religious',
    'toilets|toilets': 'service',
    'commercial|place_of_worship': 'religious',
    'commercial|car_parts': 'commercial',
    'commercial|studio': 'commercial',
    'commercial|pub': 'commercial',
    'garden_centre': 'agricultural',
    'university|library': 'education',
    'retail|educational_institution': 'commercial',
    'retail|car_repair': 'commercial',
    'watches': 'commercial',
    'commercial|lighting': 'commercial',
    'bathroom_furnishing': 'commercial',
    'hardware': 'industrial',
    'retail|beauty': 'commercial',
    'retail|parking': 'commercial',
    'public|toilets': 'civic',
    'multi-purpose_stage': 'entertainment',
    'no|community_centre': 'civic',
    'government|government': 'civic',
    'industrial|logistics': 'industrial',
    'commercial|community_centre': 'commercial',
    'pavilion|shelter': 'civic',
    'gateway': 'civic',
    'seasonal|theatre': 'entertainment',
    'hut|shelter': 'civic',
    'residential|shelter': 'residential',
    'pavilion': 'civic',
    'detached|kindergarten': 'residential',
    'office|consulting': 'commercial',
    'construction|parking': 'industrial',
    'lighting': 'industrial',
    'events_venue': 'entertainment',
    'garages': 'industrial',
    'bag': 'commercial',
    'yes;retail|mall': 'commercial',
    'animal_boarding': 'agricultural',
    'public|events_venue': 'civic',
    'bicycle_parking': 'transportation',
    'taxi': 'transportation',
    'commercial|bar': 'commercial',
    'retail|commercial': 'commercial',
    'transportation|bus_station': 'transportation',
    'bridge': 'transportation',
    'chapel': 'religious',
    'no|construction': 'industrial',
    'jtc_nanospace': 'industrial',
    'shed|security': 'industrial',
    'multi-purpose_hall|events_venue': 'entertainment',
    'tent': 'civic',
    'industrial|mall': 'industrial',
    'yes;industrial': 'industrial',
    'garage|mall': 'industrial',
    'recreation_ground': 'entertainment',
    'office|police': 'civic',
    'public|government': 'civic',
    'public|clinic': 'civic',
    '':None
    }
    
    building_geom_gdf_above_ground=building_geom_gdf_above_ground.replace({"osm_class": element_to_category})
    
    # Get average height across building classes
    class_list = [i for i in building_geom_gdf_above_ground['osm_class'].unique() if i!=None]
    
    average_class_height = {}
    for building_class in class_list:
        temp = building_geom_gdf_above_ground[(building_geom_gdf_above_ground['osm_class'] == building_class) & 
                                              (building_geom_gdf_above_ground['osm_height']>0) & 
                                              (building_geom_gdf_above_ground['osm_building:levels']>0)]
        
        average_class_height[building_class] = round((temp['osm_height'] / temp['osm_building:levels']).mean(),3)
    
    temp = building_geom_gdf_above_ground[(building_geom_gdf_above_ground['osm_height']>0) & 
                                          (building_geom_gdf_above_ground['osm_building:levels']>0)]
    
    average_class_height['total'] = round((temp['osm_height'].sum() / temp['osm_building:levels'].sum()), 3)

    # Harmonise columns with building height and combined_level information
    def combine_height(height, combined_level, building_class):
        if height!=0:
            return height
        
        elif (combined_level!=0) and (building_class in class_list):
            if np.isnan(average_class_height[building_class]):
                multiplier = average_class_height['total']
            else: 
                multiplier = average_class_height[building_class]
            return combined_level * multiplier
    
        elif (combined_level!=0) and (building_class not in class_list):
            multiplier = average_class_height['total']
            return combined_level * multiplier
        
        else:
            return 0
    
    building_geom_gdf_above_ground['osm_combined_heights'] = building_geom_gdf_above_ground.apply(lambda row: combine_height(row['osm_height'], 
                                                                                                                                   row['osm_building:levels'],
                                                                                                                                   row['osm_class']), 
                                                                                                                                   axis=1)
    
    if return_class_height:
        return building_geom_gdf_above_ground, average_class_height

    return building_geom_gdf_above_ground


def assign_numerical_id_suffix(gdf, prefix):
    """Helper function to assign unique building ids to building footprints. Items with duplicate ids are assigned suffixes corresponding to their count "_(count)". 
    For example, if two building polygons have the id: 12093210, the first will be renamed to 12093210_1 and second to 12093210_2.

    Args:
        gdf (gpd.GeoDataFrame): A geopandas GeoDataFrame with duplicate column id due to conversion between MultiPolygon to Polygons.
        prefix (str): Specifies whether the GeoDataFrame assigned corresponds to Overture (overture_) or OSM (osm_) building footprints. 

    Returns:
        gpd.GeoDataFrame: Returns a modified GeoDataFrame with unique building footprint ids.
    """    
    modified_gdf = gdf.copy()
    modified_gdf[f'{prefix}_id'] = modified_gdf[f'{prefix}_id'].astype(str)
    
    original_ids = list(modified_gdf[f'{prefix}_id'].value_counts().index[modified_gdf[f'{prefix}_id'].value_counts() > 1])
    
    for target in original_ids:
        indices = modified_gdf.loc[modified_gdf[f'{prefix}_id'] == target, f'{prefix}_id'].index
        
        for i, idx in enumerate(indices):
            modified_gdf.loc[idx, f'{prefix}_id'] = f'{target}_{i}'

    return modified_gdf


def merge_osm_to_overture_footprints(overture_buildings_attr_uids, osm_attr_uids):
    """Helper function to harmonise OSM and Overture building footprint data. First, the geometric difference between both sets of building footprints are computed. Using Overture as the base, missing buildings from OSM are then systematically added to the Overture set.
    The function automatically harmonises column names and computes the unprojected global bounding box coordinates for OSM buildings. 

    Args:
        overture_buildings_attr_uids (gpd.GeoDataFrame): GeoDataFrame consisting of Overture building footprints with unique ids and pre-processed geometry and attributes. 
        osm_attr_uids (gpd.GeoDataFrame): GeoDataFrame consisting of OSM building footprints with unique ids and pre-processed geometry and attributes.

    Returns:
        gpd.GeoDataFrame: Returns a harmonised building footprint dataset with missing building footprints from OSM dataset added to the Overture dataset.
    """    
    
    # Extract footprints that are part of osm but not part of overture
    buildings_diff_osm = osm_attr_uids.overlay(overture_buildings_attr_uids, how='difference')
    buildings_diff_osm['osm_difference_area'] = buildings_diff_osm.geometry.area
    buildings_diff_osm['osm_proportion_not_covered'] = round(buildings_diff_osm['osm_difference_area'] / buildings_diff_osm['osm_original_area'] * 100, 3)
    
    # From proportion area, extract OSM buildings which are totally missing from Overture 
    osm_only_buildings = buildings_diff_osm.copy()
    osm_only_buildings = osm_only_buildings[osm_only_buildings['osm_proportion_not_covered'] == 100]
    
    # Project to global coordinate to get footprint bounding box
    osm_only_buildings_epsg4326 = osm_only_buildings.copy()
    osm_only_buildings_epsg4326 = osm_only_buildings_epsg4326.to_crs('epsg:4326')
    
    # Apply function to rows to extract bounding box
    def get_bbox_dict(minx, maxx, miny, maxy):
        return {'minx':minx, 'maxx':maxx, 'miny':miny, 'maxy':maxy}
    osm_only_buildings['osm_bbox'] = osm_only_buildings_epsg4326.geometry.bounds.apply(lambda row: get_bbox_dict(row['minx'], 
                                                                                                                 row['maxx'], 
                                                                                                                 row['miny'], 
                                                                                                                 row['maxy']), 
                                                                                       axis=1)
    
    # Select common columns between both datasets
    selected_overture = overture_buildings_attr_uids[['overture_id', 'overture_names', 'overture_class', 'overture_combined_level', 
                                                      'overture_combined_heights', 'overture_original_area', 'overture_bbox', 'geometry']]
    
    selected_osm = osm_only_buildings[['osm_id', 'osm_name', 'osm_class', 'osm_building:levels', 'osm_combined_heights', 'osm_original_area', 
                                       'osm_bbox', 'geometry']]
    
    col_names = ['building_id', 'building_names', 'building_class', 'building_level', 'building_height', 'building_area', 'building_bbox', 'geometry']
    selected_osm.columns = col_names
    selected_overture.columns = col_names
    
    merged_gdf = gpd.GeoDataFrame(pd.concat([selected_overture, selected_osm], ignore_index=True, axis=0))
    
    return merged_gdf


def extract_attributed_osm_buildings(merged_gdf, osm_attr_uids, column = 'osm_combined_heights', threshold = 50):
    """Checks OSM building dataset for available building height data and adds it to the combined (Overture+OSM) dataset where building height data is unavailable.

    Args:
        merged_gdf (gpd.GeoDataFrame): Harmonised building footprint dataset with both OSM and Overture building footprints.
        osm_attr_uids (gpd.GeoDataFrame): GeoDataFrame consisting of OSM building footprints with unique ids and pre-processed geometry and attributes.
        column (str, optional): Specifies the column to check for building attribute data. Defaults to 'osm_combined_heights'.
        threshold (int, optional): Determines the proportion of geometric overlap in building footprint to consider correspondance. Defaults to 50.

    Returns:
        gpd.GeoDataFrame: Returns building footprint dataset augmented with additional building semantic attribute.
    """    
    
    # Get all OSM buildings with target information
    osm_buildings_with_info = osm_attr_uids.copy()
    osm_buildings_with_info = osm_buildings_with_info[osm_buildings_with_info[column] > 0]
    
    # Determine proportion of geometric intersection with buildings dataset
    building_intersection = merged_gdf.overlay(osm_buildings_with_info)
    building_intersection['overlapping_area'] = building_intersection.geometry.area
    building_intersection['proportion_overlapping_area'] = round(building_intersection['overlapping_area'] / building_intersection['building_area'] * 100, 3)
    
    # Filter based on threshold
    thresholded_buildings = building_intersection[(building_intersection['proportion_overlapping_area'] > threshold) & (building_intersection['building_height'] == 0)]

    # Add attribute information to merged_gdf
    columns_to_merge = thresholded_buildings.copy()
    columns_to_merge = thresholded_buildings[['building_id', column]]
    
    # Make copy
    combined_with_height = merged_gdf.copy()

    # Add height attribute
    for building_id, osm_height in zip(columns_to_merge['building_id'], columns_to_merge[column]):
        combined_with_height.loc[combined_with_height['building_id'] == building_id, 'building_height'] = osm_height
        
    return combined_with_height