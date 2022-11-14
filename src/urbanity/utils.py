# Map class utility functions
from __future__ import annotations
import os
import json
import pkg_resources
from typing import Optional
from IPython.display import display
from ipyleaflet import DrawControl


def get_country_centroids():
    """Utility function to obtain country centroids based on country name.

    Returns:
        dict: Dictionary object with keys as country names and values as centroid locations.
    """    
    data_path = pkg_resources.resource_filename('urbanity', "map_data/country.json")
    with open(data_path) as f:
        country_dict = json.load(f)

    return country_dict

def get_population_data_links(country, use_tif = False):
    """Obtain population data links based on specified country.

    Args:
        country (str): Name of country to obtain population data.
        use_tif (bool, optional): If True, obtains url for .geotiffs instead of csv. Defaults to False.

    Returns:
        dict: Dictionary with keys as data tags and values as links to population data.
    """    
    if country == 'United States' and use_tif:
        data_path = pkg_resources.resource_filename('urbanity', "map_data/usa_general_tif.json")
    else:
        data_path = pkg_resources.resource_filename('urbanity', "map_data/links_general.json")
    with open(data_path) as f:
        general_pop_dict = json.load(f)
    return general_pop_dict

def get_available_pop_countries():
    """Prints list of countries where population data is available.
    """    
    general_pop_dict = set(get_population_data_links())
    print(sorted(general_pop_dict))

def get_available_countries():
    """Prints list of countries where centroid information is available. 
    """
    country_dict = set(get_country_centroids())
    print(sorted(country_dict))

def finetune_poi(df, target, relabel_dict, n=5):
    """Relabel and trim poi list to main categories ('Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social')

    Args:
        df (pd.DataFrame): POI dataframe with full list of amenities extracted from OSM
        target (str): Target column with poi labels
        relabel_dict (dict): Relabelling dictionary to match original poi labels to main categories. Users can provide custom relabelling according to use case by modifying (./src/urbanity/map_data/poi_filter.json)
        n (int, optional): Minimum count of pois to keep. Defaults to 5.

    Returns:
        pd.DataFrame: Dataframe with poi information relabelled according to main categories. 
    """    
    df2 = df.copy()
    for k,v in relabel_dict.items():
        df2[target] = df2[target].replace(k, v)
    
    # remove categories with less than n instances
    
    cat_list = df2[target].value_counts().index
    cat_mask = (df2[target].value_counts() > n).values
    selected = set(cat_list[cat_mask])
    
    df2 = df2[df2[target].isin(selected)]
    
    return df2
