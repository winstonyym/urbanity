# Map class utility functions
from __future__ import annotations
import os
import json
from typing import Optional
from IPython.display import display
from ipyleaflet import DrawControl


def get_country_centroids(filepath: Optional[str] = None):
    if filepath is None:
        data_path = os.path.join(os.getcwd(), "..", "data/country.json")
    with open(data_path) as f:
        country_dict = json.load(f)

    return country_dict

def get_available_countries():
    country_dict = set(get_country_centroids())
    print(sorted(country_dict))


