# Map class utility functions
import os
import io
import re
import math
import json
import time
import requests
import pandas as pd
import pkg_resources
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString, shape, Polygon
from shapely.ops import unary_union
from urbanity.geom import project_gdf
import numpy as np
from IPython.display import display
from ipyleaflet import DrawControl
from urllib.error import HTTPError
from collections import Counter

from urbanity.building import building_knn_nearest, compute_knn_aggregate


GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
BBBIKE_BASE_URL = "https://download.bbbike.org/osm/bbbike"


def fetch_geofabrik_index(cache_path="./data/geofabrik_index.json"):
    """Fetch and cache the Geofabrik extract index (with geometries)."""
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < 7 * 86400:
            with open(cache_path) as f:
                return json.load(f)

    resp = requests.get(GEOFABRIK_INDEX_URL)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(resp.json(), f)
    return resp.json()


def _parse_geofabrik_candidates(index, aoi_geometry):
    """Return candidate extracts from Geofabrik that fully contain the AOI."""
    candidates = []
    for feature in index["features"]:
        props = feature["properties"]
        if feature["geometry"] is None:
            continue
        extract_geom = shape(feature["geometry"])
        if extract_geom.contains(aoi_geometry):
            urls = props.get("urls", {})
            pbf_url = urls.get("pbf")
            if pbf_url:
                size = props.get("pbf_file_size", float("inf"))
                candidates.append({
                    "source": "geofabrik",
                    "name": props["name"],
                    "url": pbf_url,
                    "size": size,
                })
    return candidates


# ---------------------------------------------------------------------------
# BBBike helpers
# ---------------------------------------------------------------------------

def _parse_poly_file(text):
    """Parse an Osmosis .poly file into a Shapely geometry."""
    polygons = []
    current_ring = []
    is_hole = False

    for line in text.strip().splitlines():
        line = line.strip()
        if line == "END":
            if current_ring and len(current_ring) >= 3:
                poly = Polygon(current_ring)
                if poly.is_valid:
                    polygons.append((poly, is_hole))
            current_ring = []
            is_hole = False
            continue

        # Section header (polygon name); "!" prefix means hole
        parts = line.split()
        if len(parts) == 1:
            is_hole = line.startswith("!")
            continue

        # Coordinate line: longitude  latitude
        try:
            lon, lat = float(parts[0]), float(parts[1])
            current_ring.append((lon, lat))
        except (ValueError, IndexError):
            continue

    # Combine shells, subtract holes
    shells = [p for p, h in polygons if not h]
    holes = [p for p, h in polygons if h]
    if not shells:
        return None
    result = unary_union(shells)
    for hole in holes:
        result = result.difference(hole)
    return result


def fetch_bbbike_index(cache_dir="./data/bbbike"):
    """Build a BBBike index: list of cities with their boundary geometries.

    Downloads the city list once, then lazily fetches .poly files.
    Returns a list of dicts with keys: name, geometry, pbf_url.
    The whole index is cached as a single JSON file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "bbbike_index.json")

    # Return cached index if fresh
    if os.path.exists(index_path):
        mtime = os.path.getmtime(index_path)
        if (time.time() - mtime) < 7 * 86400:
            with open(index_path) as f:
                return json.load(f)

    # Step 1: get city list from the directory listing
    resp = requests.get(f"{BBBIKE_BASE_URL}/")
    resp.raise_for_status()
    # Parse city names from href links like <a href="CityName/">
    cities = re.findall(r'href="([A-Z][A-Za-z0-9_-]+)/"', resp.text)
    # Filter out non-city entries
    cities = [c for c in cities if c not in (".", "..")]

    # Step 2: for each city, download the .poly file and parse it
    index = []
    for city in cities:
        poly_url = f"{BBBIKE_BASE_URL}/{city}/{city}.poly"
        try:
            poly_resp = requests.get(poly_url, timeout=15)
            poly_resp.raise_for_status()
            geom = _parse_poly_file(poly_resp.text)
            if geom is None or geom.is_empty:
                continue
            index.append({
                "name": city,
                "geometry": geom.__geo_interface__,
                "pbf_url": f"{BBBIKE_BASE_URL}/{city}/{city}.osm.pbf",
            })
        except requests.RequestException:
            continue

    # Cache the index
    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"BBBike index built: {len(index)} cities")
    return index


def _parse_bbbike_candidates(bbbike_index, aoi_geometry):
    """Return candidate extracts from BBBike that fully contain the AOI."""
    candidates = []
    for entry in bbbike_index:
        extract_geom = shape(entry["geometry"])
        if extract_geom.contains(aoi_geometry):
            candidates.append({
                "source": "bbbike",
                "name": entry["name"],
                "url": entry["pbf_url"],
                "size": None,  # unknown until we do a HEAD request
            })
    return candidates


def _resolve_bbbike_sizes(candidates):
    """Fill in file sizes for BBBike candidates via HEAD requests."""
    for c in candidates:
        if c["size"] is None:
            try:
                resp = requests.head(c["url"], allow_redirects=True, timeout=10)
                c["size"] = int(resp.headers.get("Content-Length", float("inf")))
            except (requests.RequestException, ValueError):
                c["size"] = float("inf")
    return candidates


# ---------------------------------------------------------------------------
# Unified lookup
# ---------------------------------------------------------------------------

def find_smallest_extract(aoi_geometry, data_dir="./data"):
    """Find the smallest .osm.pbf extract containing the AOI.

    Searches both Geofabrik and BBBike, returns the smallest candidate.
    """
    # Gather candidates from both sources
    geofabrik_index = fetch_geofabrik_index(
        cache_path=os.path.join(data_dir, "geofabrik_index.json")
    )
    bbbike_index = fetch_bbbike_index(
        cache_dir=os.path.join(data_dir, "bbbike")
    )

    candidates = []
    candidates.extend(_parse_geofabrik_candidates(geofabrik_index, aoi_geometry))
    candidates.extend(_parse_bbbike_candidates(bbbike_index, aoi_geometry))

    if not candidates:
        raise ValueError(
            "No extract from Geofabrik or BBBike fully contains the area of interest."
        )

    # Resolve sizes for BBBike candidates (Geofabrik sizes come from the index)
    bbbike_candidates = [c for c in candidates if c["source"] == "bbbike"]
    if bbbike_candidates:
        _resolve_bbbike_sizes(bbbike_candidates)

    candidates.sort(key=lambda x: x.get("size", float("inf")))

    best = candidates[0]
    print(candidates)
    print(f"Best extract: {best['name']} ({best['source']}) "
          f"— {best['size'] / 1e6:.1f} MB")
    return best


# ---------------------------------------------------------------------------
# Download with caching
# ---------------------------------------------------------------------------

def _get_expected_size(url):
    """Get the file size from the server via a HEAD request."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length", 0))
    except (requests.RequestException, ValueError):
        return 0


def is_extract_cached(dest_path, expected_size=None, url=None):
    """Check if a valid .osm.pbf file already exists at dest_path."""
    if not os.path.exists(dest_path):
        return False

    local_size = os.path.getsize(dest_path)
    if local_size == 0:
        return False

    if expected_size is None and url:
        expected_size = _get_expected_size(url)

    if expected_size and expected_size > 0:
        return local_size == expected_size

    return True


def download_extract(url, dest_path, expected_size=None):
    """Download a .osm.pbf file, skipping if already cached."""
    if is_extract_cached(dest_path, expected_size=expected_size, url=url):
        print(f"Using cached extract: {dest_path} "
              f"({os.path.getsize(dest_path) / 1e6:.1f} MB)")
        return dest_path

    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Progress: {pct:.1f}% "
                      f"({downloaded / 1e6:.1f}/{total / 1e6:.1f} MB)",
                      end="", flush=True)

    print()
    print(f"Download complete: {dest_path}")
    return dest_path


def get_country_centroids():
    """Utility function to obtain country centroids based on country name.

    Returns:
        dict: Dictionary object with keys as country names and values as centroid locations.
    """    
    data_path = pkg_resources.resource_filename('urbanity', "map_data/country.json")
    with open(data_path) as f:
        country_dict = json.load(f)

    return country_dict

def get_population_data_links():
    """Obtain population data links based on specified country.

    Args:
        country (str): Name of country to obtain population data.
        use_tif (bool, optional): If True, obtains url for .geotiffs instead of csv. Defaults to False.

    Returns:
        dict: Dictionary with keys as data tags and values as links to population data.
    """    
    data_path = pkg_resources.resource_filename('urbanity', "map_data/links_general_tiled.json")
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

def get_available_precomputed_network_data():
    """Prints list of cities available from the Global Urban Network Dataset
    """
    data_path = pkg_resources.resource_filename('urbanity', "map_data/network_data.json")
    with open(data_path) as f:
        city_dict = json.load(f)

    list_of_cities = []
    for entry in city_dict.keys():
        if entry.split('_')[0] not in list_of_cities:
            list_of_cities.append(entry.split('_')[0])
    
    print(f'The following cities are available: {sorted(list_of_cities)}.')

def finetune_poi(df, target, relabel_dict, n=5, pois_data = 'osm'):
    """Relabel and trim poi list to main categories ('Civic', 'Commercial', 'Entertainment', 'Food', 'Healthcare', 'Institutional', 'Recreational', 'Social')

    Args:
        df (pd.DataFrame): POI dataframe with full list of amenities extracted from OSM/Overture
        target (str): Target column with poi labels
        relabel_dict (dict): Relabelling dictionary to match original poi labels to main categories. Users can provide custom relabelling according to use case by modifying (./src/urbanity/map_data/poi_filter.json)
        n (int, optional): Minimum count of pois to keep. Defaults to 5.
        pois_data (str, optional): Specifies whether osm or Overture poi data should be used. Defaults to 'osm'.

    Returns:
        pd.DataFrame: Dataframe with poi information relabelled according to main categories. 
    """  
    if pois_data == 'osm':
        df2 = df.copy()
        for k,v in relabel_dict.items():
            df2[target] = df2[target].replace(k, v)
        
        # remove categories with less than n instances
        
        cat_list = df2[target].value_counts().index
        cat_mask = (df2[target].value_counts() > n).values
        selected = set(cat_list[cat_mask])
        
        df2 = df2[df2[target].isin(selected)]

    elif pois_data == 'overture':
        df2 = df.copy()
        df2=df2.replace({target: relabel_dict})

    return df2


def get_gadm(country, city, version = '4.1', max_level = 4, level_drop = 0):
    """Function to automate extraction of GADM city boundaries and their subzones. Files are extracted in .geojson format.

    Args:
        city (str): City name to extract from GADM database.
        city_subzone (bool, optional): If True, searches one level down to obtain census subzone for city. Defaults to False.
    """    

    country = country.title()
    city = city.title()
    small_countries = ['Singapore']
    large_scale_countries = ['United States']

    data_path = pkg_resources.resource_filename('urbanity', "map_data/GADM_links.json")

    with open(data_path) as f:
        GADM_dict = json.load(f)
    
    country_code = GADM_dict[country]
    returned = False

    for i in reversed(range(max_level+1)):
        geojson_path = f'https://geodata.ucdavis.edu/gadm/gadm{version}/json/gadm{version.replace(".", "")}_{country_code}_{i}.json'
        try:
            country_df = gpd.read_file(geojson_path)
            print(f'Level {i} downloaded for {country}.')
            if country in small_countries:
                return country_df
            if country in large_scale_countries:
                return country_df
            
            result = []
            for level in range(1,i+1):
                zones = list(country_df[f'NAME_{level}'].unique())
                result = [zone for zone in zones if city in zone]
                if result == []:
                    continue
                elif result != []:
                    print(f'{result[0]} found in level {level}.')
                    if level_drop == 0:
                        print(f'Returning level {level} boundary file.')
                        return country_df[country_df[f'NAME_{level}'] == result[0]]
                    elif level_drop != 0:
                        try:
                            print(f'Retrieving level {level+level_drop} boundary file.')
                            zones = list(country_df[f'NAME_{level+level_drop}'].unique())
                            result = [zone for zone in zones if city in zone]
                            return country_df[country_df[f'NAME_{level+level_drop}'] == result[0]]
                        except KeyError:
                            print('GADM does not provide shapefiles at this level of detail.')
                            return None
                        except IndexError:
                            print(f'No subzone with corresponding name found at this level.')
                            return None
                    
        except HTTPError:
            continue
    
    
def get_building_to_building_edges(buildings, 
                                   return_neighbours = 'knn', 
                                   knn: int = 3,
                                   distance_threshold: int = 100,
                                   knn_threshold = 100, 
                                   add_reverse=True):
    
    buildings_copy = buildings.copy()
    buildings_copy = buildings_copy.to_crs('epsg:3857')
    buildings_copy['bid_centroid'] = buildings_copy.geometry.centroid
    if return_neighbours == 'knn':
        def filter_threshold(nn, dist):
            return {k:v for k,v in zip(nn, dist) if v <= knn_threshold}

        # Compute attributes
        buildings_copy = building_knn_nearest(buildings_copy, knn=knn)
        buildings_copy[f'{knn}-nn-threshold'] = buildings_copy.apply(lambda row: filter_threshold(row[f'{knn}-nn-idx'], row[f'{knn}-dist']), axis=1)
        adj_column = f'{knn}-nn-idx'

    elif return_neighbours == 'distance':
        def remove_self(neighbours, bid):
            try:
                neighbours.remove(bid)
                return neighbours
            except ValueError:
                return neighbours

        buffer_gdf = gpd.GeoDataFrame(data={'buffer_id':buildings_copy.index}, crs=buildings_copy.crs, geometry = buildings_copy.geometry.centroid)
        buffer_gdf['geometry'] = buffer_gdf.geometry.buffer(distance_threshold)

        # Spatial intersection of building
        res_intersection = buildings_copy.overlay(buffer_gdf, how='intersection')
        buildings_copy[f'{distance_threshold}_dist_idx'] = res_intersection.groupby(['buffer_id'])['bid'].agg(list)
        buildings_copy[f'{distance_threshold}_dist_idx'] = buildings_copy.apply(lambda row: remove_self(row[f'{distance_threshold}_dist_idx'], row['bid']), axis=1)
        adj_column = f'{distance_threshold}_dist_idx'

    # building_edges = get_building_to_building_edges(building_nodes, adj_column = '3-nn-idx')
    # Prepare edge index. First match with index position then convert to torch tensor. 
    start_list = []
    end_list = []
    for i, neighbours in enumerate(buildings_copy[adj_column]):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    
    # Edge from main building to neighbouring buildings
    building_to_building = np.stack([start_list, end_list], axis=1).transpose()

    # Add reverse edges
    if add_reverse:

        # Edge from neighbouring buildings to main building
        building_rev_building = np.flip(building_to_building, axis=0)
        return building_to_building, building_rev_building
    
    return building_to_building

def get_intersection_to_street_edges(intersections, streets, add_reverse=True):
    # intersection_to_street_edges = get_intersection_to_street_edges(gdfs[1], gdfs[2])
    node_to_id = {}
    for i,node in enumerate(intersections['osmid']):
        node_to_id[node] = i

    start_node = [node_to_id[i] for i in streets['u'].values] + [node_to_id[i] for i in streets['v'].values]
    end_node = list(streets['street_id'].values) + list(streets['street_id'].values)

    start_index = np.array(start_node)
    end_index = np.array(end_node)
    intersection_to_street_edges = np.stack([start_index, end_index], axis=1).transpose()

    # Add reverse edges
    if add_reverse:

        # Edge from neighbouring buildings to main building
        street_to_intersection_edges = np.flip(intersection_to_street_edges, axis=0)
        return intersection_to_street_edges, street_to_intersection_edges
    
    return intersection_to_street_edges

def get_buildings_in_plot_edges(urban_plots, add_reverse=True):
    # building_in_plot_edges = get_buildings_in_plot_edges(urban_plots, adj_column = 'building_ids')
    # Prepare edge index. First match with index position then convert to torch tensor. 
    start_list = []
    end_list = []
    for i, neighbours in enumerate(urban_plots['bid']):
        if neighbours != 0:
            for k in neighbours:
                start_list.append(i)
                end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    building_to_plot_edges = np.stack([end_index, start_index], axis=1).transpose()
    building_to_plot_edges = building_to_plot_edges.astype(int)
    if add_reverse:

        # Edge from neighbouring buildings to main building
        plot_to_building_edges = np.flip(building_to_plot_edges, axis=0)
        return building_to_plot_edges, plot_to_building_edges
    
    return building_to_plot_edges

def gdf_to_poly(gdf, poly_path, column: str = "boundary_id"):
    """
    Write a GeoDataFrame of Polygon / MultiPolygon geometries to a .poly file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain only Polygon or MultiPolygon geometries.
    poly_path : str | PathLike
        Output file path.
    column : str, default "boundary_id"
        Attribute whose value will be written as the header for each geometry.
    """
    with open(poly_path, "w") as poly_file:

        for _, row in gdf.iterrows():
            poly_file.write(f"{row[column]}\n")        # header
            geom = row.geometry

            # --- collect all exterior/interior rings in one list -----------
            rings = []

            if geom.geom_type == "Polygon":
                rings.append(geom.exterior)
                rings.extend(geom.interiors)

            elif geom.geom_type == "MultiPolygon":
                # Shapely ≥2.0: iterate via `.geoms`
                for poly in geom.geoms:                 # each poly is a Polygon
                    rings.append(poly.exterior)
                    rings.extend(poly.interiors)

            else:
                raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

            # --- write coordinates for every ring --------------------------
            for ring in rings:
                for x, y in ring.coords:
                    poly_file.write(f"  {x} {y}\n")
                poly_file.write("END\n")                # end of ring/part

        poly_file.write("END\n")         

def get_edges_along_plot(urban_plots, add_reverse=True):
    # edges_along_plot = get_edges_along_plot(urban_plots, adj_column = 'edge_ids')
    # Prepare edge index. First match with index position then convert to torch tensor. 

    start_list = []
    end_list = []
    for i, neighbours in enumerate(urban_plots['street_id']):
        if isinstance(neighbours, np.ndarray):
            for k in neighbours:
                start_list.append(i)
                end_list.append(int(k))
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    edges_to_plot = np.stack([end_index, start_index], axis=1).transpose()

    if add_reverse:

    # Edge from neighbouring buildings to main building
        plot_to_edges = np.flip(edges_to_plot, axis=0)
        return edges_to_plot, plot_to_edges
    
    return edges_to_plot

def get_plot_to_plot_edges(urban_plots, add_reverse=True):
    """Generate plot-to-plot adjacency edges based on shared street boundaries.

    Two plots are considered neighbours if they share at least one street
    segment (identified by ``street_id``).

    The function guards against the outer encircling polygon that
    ``shapely.ops.polygonize`` occasionally produces: that polygon has an
    anomalously large number of bounding streets (it touches every outer-ring
    segment) and would otherwise create spurious edges to every adjacent plot.
    When a ``plot_area`` column is present, plots whose area exceeds the 99th
    percentile *and* whose street count exceeds three standard deviations above
    the mean are treated as artefacts and excluded before building the edge
    list.

    Parameters
    ----------
    urban_plots : geopandas.GeoDataFrame
        Polygons where each row is an urban plot.  Must contain a
        ``plot_id`` column and a list-valued ``street_id`` column.
    add_reverse : bool, optional
        If ``True`` (default), also return the reversed edge array so the
        graph is undirected.

    Returns
    -------
    np.ndarray
        Plot-to-plot edge array of shape ``(2, N)``.
    np.ndarray, optional
        Reversed edge array of the same shape (only when
        ``add_reverse=True``).
    """
    # ------------------------------------------------------------------
    # Defensive filter: drop the outer encircling polygon artefact.
    # It is characterised by having far more bounding streets than any
    # real city block.  We use a two-condition check so legitimate large
    # plots (parks, campuses) are not removed.
    # ------------------------------------------------------------------
    plots_work = urban_plots.copy()

    # Count how many street segments bound each plot
    street_counts = plots_work['street_id'].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )
    mean_sc  = street_counts.mean()
    std_sc   = street_counts.std()
    outlier_street_mask = street_counts > (mean_sc + 3 * std_sc)

    if outlier_street_mask.any() and 'plot_area' in plots_work.columns:
        # Only drop if the same plot is also in the top 1 % by area
        area_threshold = plots_work['plot_area'].quantile(0.99)
        outlier_area_mask = plots_work['plot_area'] >= area_threshold
        drop_mask = outlier_street_mask & outlier_area_mask
        if drop_mask.any():
            n_dropped = drop_mask.sum()
            print(f"[get_plot_to_plot_edges] Dropping {n_dropped} outer-envelope "
                  f"polygon(s) (area >= {area_threshold:.0f} m² and "
                  f"street count > {mean_sc + 3*std_sc:.1f}).")
            plots_work = plots_work[~drop_mask].reset_index(drop=True)
            # Re-index plot_id to keep it contiguous
            plots_work['plot_id'] = plots_work.index

    urban_plots_edges = plots_work.explode('street_id')

    neighbors_df = urban_plots_edges.merge(
                    urban_plots_edges,
                    on='street_id',
                    suffixes=('', '_right')
                    )

    neighbors_df = neighbors_df[neighbors_df['plot_id'] != neighbors_df['plot_id_right']]
    neighbors_df = neighbors_df[['plot_id', 'plot_id_right']].drop_duplicates()

    # Aggregate neighboring plot_ids
    neighbors_dict = neighbors_df.groupby('plot_id')['plot_id_right'].apply(list)
    neighbors_df = pd.DataFrame(data=neighbors_dict)
    neighbors_df.columns = ['nn_plot_ids']

    plots_work = plots_work.merge(neighbors_df, on='plot_id')

    # Create adjacency matrix for connected plots.
    start_list = []
    end_list = []
    for i, neighbours in enumerate(plots_work['nn_plot_ids']):
        for k in neighbours:
            start_list.append(i)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)

    # Edge from main plot to neighbouring plots
    plot_to_plot = np.stack([start_list, end_list], axis=1).transpose()

    # Add reverse edges
    if add_reverse:

        # Edge from neighbouring plot to main plot
        plot_rev_plot = np.flip(plot_to_plot, axis=0)
        return plot_to_plot, plot_rev_plot

    return plot_to_plot

def select_columns(objects):
    """Helper function to drop identifier ids

    Args:
        objects (dict): Set of object and their geodataframes

    Returns:
        _type_: Return set of object and their geodataframes with id columns removed
    """    
    
    objects['intersection'] = objects['intersection'][:,4:]
    objects['building'] = objects['building'][:,1:]
    objects['street'] = objects['street'][:,[3,24,28,38,41,141]]
    objects['plot'] = objects['plot'][:,1:]
    return objects

def get_building_to_street_edges(streets, building_nodes, add_reverse=True):
    """Helper function to generate network edges between buildings and their adjacent (nearest; subject to distance threshold of 50 metres) streets.

    Args:
        streets (gpd.GeoDataFrame): A geopandas dataframe consisting of LineStrings where each row represents a road segment.
        building_nodes (gpd.GeoDataFrame): A geopandas dataframe consisting of Polygons where each row corresponds to a building and its footprint.

    Returns:
        np.array: A (2, N) array where the first row corresponds to street IDs and the second row corresponds building_node IDs. N is the number of edges between all streets and buildings. 
    """    
    # street_to_building = get_building_to_street_edges(gdfs[2], building_nodes)
    building_nodes = project_gdf(building_nodes)
    building_nodes_copy = building_nodes.copy()
    building_nodes_copy['centroid'] = building_nodes.geometry.centroid
    building_nodes_copy = building_nodes_copy.set_geometry('centroid')
    building_nodes_copy['b_index'] = building_nodes_copy.index

    proj_edge = streets.to_crs(building_nodes_copy.crs)
    
    # Find nearest building to street
    edge_intersection = gpd.sjoin_nearest(building_nodes_copy, proj_edge, how='inner', max_distance=50, distance_col = 'building_edges')
    edge_to_building = edge_intersection.groupby(['street_id'])[['b_index']].aggregate(lambda x: list(x))

    start_list = []
    end_list = []
    for idx, nn in zip(edge_to_building['b_index'].index, edge_to_building['b_index']):
        for k in nn:
            start_list.append(idx)
            end_list.append(k)
            
    start_index = np.array(start_list)
    end_index = np.array(end_list)
    building_to_street = np.stack([end_index, start_index], axis=1).transpose()

    # Add reverse edges
    if add_reverse:
        street_to_building = np.flip(building_to_street, axis=0)
        return building_to_street, street_to_building
    
    return building_to_street

def get_edge_nodes(edges) -> [gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Converts street segments into nodes as part of a multi-nodal graph representation.

    Args:
        edges (gpd.GeoDataFrame): A geopandas GeoDataFrame containing the geometry and attribute features of street segments.

    Returns:
        gpd.GeoDataFrame: A geopandas GeoDataFrame consisting of half-edges that retain the original linestring geometry of street segments (for plotting). 
        gpd.GeoDataFrame: A geopandas GeoDataFrame consisting of nodes that were derived from edges. 
    """
    # Project edges to local crs for distance computation
    proj_edges = project_gdf(edges)
    
    # Set edge_id as string and create placeholder lists
    proj_edges['street_id'] = proj_edges['street_id'].astype(str)
    x_list = []
    y_list = []
    edge_id_list = []
    u_list = []
    v_list = []
    length_list = []
    geometry_list = []
    
    # Iterate through each edge 
    for i, row in proj_edges.iterrows():
        
        # Get centre from linestring coordinate sequence
        coords_list = list(row['geometry'].coords)
        len_coord_list = len(row['geometry'].coords)
        mid_idx = len_coord_list // 2
        
        # If even number of coords
        if len_coord_list == 2:
            center = ((coords_list[0][0]+ coords_list[1][0])/2, (coords_list[0][1]+ coords_list[1][1])/2) 
            line_segments = [LineString([coords_list[0], center]), LineString([center, coords_list[1]])]
        elif (len_coord_list > 2) & (len_coord_list % 2 == 0):
            center = ((coords_list[mid_idx-1][0]+ coords_list[mid_idx][0])/2, (coords_list[mid_idx-1][1]+ coords_list[mid_idx][1])/2)
            line_segments = [LineString([coords for coords in coords_list[:mid_idx]] + [center]), LineString([center] + [coords for coords in coords_list[mid_idx:]])]
        else: 
            center = coords_list[mid_idx]
            line_segments = [LineString([coords for coords in coords_list[:mid_idx+1]]), LineString([coords for coords in coords_list[mid_idx:]])]
            
        # Add start to midpoint of linestring
        edge_id_list.append(row['street_id']+'_0')
        u_list.append(row['u'])
        v_list.append(row['street_id']+'_m')
        length_list.append(line_segments[0].length)
        geometry_list.append(line_segments[0])
        
        # Add midpoint to end of linestring
        edge_id_list.append(row['street_id']+'_1')
        u_list.append(row['street_id']+'_m')
        v_list.append(row['v'])
        length_list.append(line_segments[1].length)
        geometry_list.append(line_segments[1])
        x_list.append(center[0])
        y_list.append(center[1])

    # Select attribute columns
    col = list(edges.columns[5:])
    cols = ['street_id', 'length'] + col
    # Get geodataframes corresponding to edges and edge nodes
    split_edges = gpd.GeoDataFrame({'street_id': edge_id_list, 'u': u_list, 'v': v_list, 'length': length_list}, crs=proj_edges.crs, geometry = geometry_list)
    edge_nodes = gpd.GeoDataFrame(data=edges[cols], crs = proj_edges.crs, geometry = gpd.points_from_xy(x_list, y_list))
    
    # Reproject to global coordinates
    split_edges = split_edges.to_crs(4326)
    edge_nodes = edge_nodes.to_crs(4326)

    return split_edges, edge_nodes


def most_frequent(List):
    """Helper function which returns the most common element in a list.

    Args:
        List (list): A list of elements with categorical labels.   

    Returns:
        int: The most common integer element. 
    """    
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


# def load_npz(filepath):
#     out = np.load(filepath, allow_pickle=True)
#     objects = {}
#     connections = {}

#     for k,v in out.items():
#         if '_' in k:
#             connections[k] = v
#         else:
#             objects[k] = v
#     return objects, connections

# def save_to_npz(save_filepath, objects, connections):
#     objects.update(connections)
#     np.savez_compressed(save_filepath, **objects)
        
        
def fill_na_in_objects(objects):

    for key, object in objects.items():
        na_cols = []
        for col in object.columns:
            if sum(object[col].isna()) != 0:
                na_cols.append(col)
        
        for missing_col in na_cols:
            temp_mean = object[missing_col].mean()

            # Fill NaN values and assign back to the DataFrame
            object[missing_col] = object[missing_col].fillna(value=temp_mean)
        
        objects[key] = object
        
    return objects

def one_hot_encode_categorical(df, target_col = '', prefix = ''):
    '''Helper function to convert categorical column into numerical binary columns. 
    Prefix is added to distinguish between categories.'''
    df_dummies = pd.get_dummies(df[target_col], prefix=prefix)
    df = df.drop(columns=[target_col], axis=1)
    df = df.join(df_dummies)
    return df


def remove_non_numeric_columns_objects(objects, keep_geometry=False):
    objects_new = objects.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    if keep_geometry:
        numerics += ['geometry']

    for key, object in objects_new.items():
        only_numerics = object.select_dtypes(include=numerics)
 
        if key == 'intersection':
            only_numerics = only_numerics.drop(columns = ['intersection_id', 'osmid', 'x', 'y'], axis=1)
        elif key == 'plot':
            only_numerics = only_numerics.drop(columns = ['plot_id'], axis=1)
        elif key == 'building':
            only_numerics = only_numerics
        elif key == 'street':
            only_numerics = only_numerics.drop(columns = ['u', 'v', 'street_id'], axis=1)
            
        objects_new[key] = only_numerics

    return objects_new


def standardise_and_scale(objects):
    '''Helper function to scale dataframes. '''

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    
    scale = StandardScaler()

    for key, df in objects.items():
        all_columns = list(df.columns)
        boolean_mask = (df.dtypes == 'bool').values
        numeric_columns = [i for idx, i in enumerate(all_columns) if ~boolean_mask[idx]]

        ct = ColumnTransformer([
            ('somename', StandardScaler(), numeric_columns)
        ], remainder='passthrough')

        objects[key] = ct.fit_transform(df)

    return objects
    

def boundary_to_plot(plot, add_reverse=True):
    '''Helper function to add super node to graph. Specify target to create links to specific layer'''
    boundary_to_plot = np.zeros((2, len(plot)))
    boundary_to_plot[1, :] = np.arange(len(plot))
    boundary_to_plot = boundary_to_plot.astype(int)
    
    if add_reverse:
        plot_to_boundary = np.flip(boundary_to_plot, axis=0)
    
    return boundary_to_plot, plot_to_boundary


import difflib
from shapely.geometry import mapping

# Mapping from common alternative names to your target list names
COUNTRY_NAME_ALIASES = {
    # reverse geocoding quirks → your list names
    "United States of America": "United States",
    "USA": "United States",
    "US": "United States",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Eswatini": "Kingdom of Eswatini",
    "Swaziland": "Kingdom of Eswatini",
    "Tanzania": "United Republic of Tanzania",
    "Luxembourg": "Luxemburg",
    "India": "Pakistan & India",
    "Pakistan": "Pakistan & India",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Nauru": "Naura",
    "Republic of the Congo": "Congo",
    "DR Congo": "Democratic Republic of the Congo",
    "Micronesia": "Federated States of Micronesia",
    "South Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
}

# Your target extract list
EXTRACT_NAMES = [
    'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla',
    'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia',
    'Austria', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
    'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia & Herzegovina', 'Botswana', 'Brazil', 'British Virgin Islands',
    'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde',
    'Cambodia', 'Cameroon', 'Cayman Islands', 'Central African Republic', 'China',
    'Chad', 'Chile', 'Colombia', 'Comoros', 'Congo',
    'Continent of Africa', 'Cook Islands', 'Costa Rica', 'Croatia',
    'Czechia', "Côte d'Ivoire", 'Democratic Republic of the Congo',
    'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
    'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
    'Ethiopia', 'Faroe Islands', 'Federated States of Micronesia', 'Fiji',
    'Finland', 'France', 'French Guiana', 'French Polynesia', 'Gabon',
    'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece',
    'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea',
    'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong',
    'Hungary', 'Iceland', 'Indonesia', 'Iraq', 'Ireland', 'Isle of Man',
    'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
    'Kingdom of Eswatini', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos',
    'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',
    'Lithuania', 'Luxemburg', 'Macau', 'Madagascar', 'Malawi', 'Malaysia',
    'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania',
    'Mauritius', 'Mayotte', 'Mexico', 'Moldova', 'Monaco', 'Mongolia',
    'Montserrat', 'Mozambique', 'Namibia', 'Naura', 'Nepal',
    'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger',
    'Nigeria', 'North Macedonia', 'Northern Mariana Islands', 'Norway',
    'Oman', 'Pakistan & India', 'Palau', 'Panama', 'Papua New Guinea',
    'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico',
    'Qatar', 'Romania', 'Rwanda', 'Réunion', 'Saint Kitts and Nevis',
    'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa',
    'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal',
    'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia',
    'Slovenia', 'Solomon Islands', 'South Africa', 'South Korea', 'Spain',
    'Sri Lanka', 'Suriname', 'Sweden', 'Switzerland', 'Taiwan',
    'Tajikistan', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga',
    'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan',
    'Turks and Caicos', 'Tuvalu', 'US Virgin Islands', 'Uganda',
    'Ukraine', 'United Arab Emirates', 'United Kingdom',
    'United Republic of Tanzania', 'United States', 'Uruguay',
    'Uzbekistan', 'Vanuatu', 'Vietnam', 'Wallis and Futuna Islands',
    'Zambia', 'Zimbabwe',
]


def get_country_from_polygon(polygon_geometry):
    """Determine which country a polygon is in using reverse geocoding,
    then match to the extract name list.

    Uses Nominatim (free, no key required) to reverse geocode the centroid.
    Falls back to a representative point if the centroid is outside the polygon.

    Parameters
    ----------
    polygon_geometry : shapely.geometry.Polygon or MultiPolygon

    Returns
    -------
    str or None
        Matched country name from EXTRACT_NAMES, or None if no match.
    """
    import requests

    # Use representative_point (guaranteed inside the polygon) rather than
    # centroid (which can fall outside concave shapes)
    point = polygon_geometry.representative_point()
    lat, lon = point.y, point.x

    # Nominatim reverse geocoding
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "zoom": 3,  # country-level
        "accept-language": "en",
    }
    headers = {"User-Agent": "osm-extract-downloader/1.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Extract country name from response
    country_raw = data.get("address", {}).get("country")
    if not country_raw:
        # Fallback: try the display_name (last component is usually the country)
        display = data.get("display_name", "")
        country_raw = display.split(",")[-1].strip() if display else None

    if not country_raw:
        return None

    return _match_country_name(country_raw)


def _match_country_name(country_raw):
    """Match a raw country name to the extract name list.

    Tries: exact match → alias lookup → fuzzy match.

    Parameters
    ----------
    country_raw : str
        Country name from reverse geocoding.

    Returns
    -------
    str or None
        Best matching name from EXTRACT_NAMES.
    """
    name = country_raw.strip()

    # 1. Exact match
    if name in EXTRACT_NAMES:
        return name

    # 2. Alias lookup
    if name in COUNTRY_NAME_ALIASES:
        alias = COUNTRY_NAME_ALIASES[name]
        if alias in EXTRACT_NAMES:
            return alias

    # 3. Case-insensitive exact match
    lower_map = {n.lower(): n for n in EXTRACT_NAMES}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    # 4. Fuzzy match
    matches = difflib.get_close_matches(name, EXTRACT_NAMES, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    return None