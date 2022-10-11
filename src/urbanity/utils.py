# Map class utility functions
from __future__ import annotations
import os
import json
import pkg_resources
from typing import Optional
from IPython.display import display
from ipyleaflet import DrawControl


def get_country_centroids():
    data_path = pkg_resources.resource_filename('urbanity', "map_data/country.json")
    with open(data_path) as f:
        country_dict = json.load(f)

    return country_dict

def get_population_data_links():
    data_path = pkg_resources.resource_filename('urbanity', "map_data/links_general.json")
    with open(data_path) as f:
        general_pop_dict = json.load(f)
    return general_pop_dict

def get_available_pop_countries():
    general_pop_dict = set(get_population_data_links())
    print(sorted(general_pop_dict))

def get_available_countries():
    country_dict = set(get_country_centroids())
    print(sorted(country_dict))

def gdf_to_tensor():
    pass

def finetune_poi(df, target, relabel_dict, n=30):
    
    df2 = df.copy()
    for k,v in relabel_dict.items():
        df2[target] = df2[target].replace(k, v)
    
    # remove categories with less than n instances
    
    cat_list = df2[target].value_counts().index
    cat_mask = (df2[target].value_counts() > n).values
    selected = set(cat_list[cat_mask])
    
    df2 = df2[df2[target].isin(selected)]
    
    return df2
