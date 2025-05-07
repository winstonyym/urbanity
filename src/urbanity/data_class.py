import os
from dataclasses import dataclass, field
import geopandas as gpd
import pandas as pd
import numpy as np
import h5py
import pickle
import torch
from .utils import get_plot_to_plot_edges, get_building_to_street_edges, get_edge_nodes, get_building_to_building_edges, \
                   get_intersection_to_street_edges, get_buildings_in_plot_edges, get_edges_along_plot, boundary_to_plot, \
                   remove_non_numeric_columns_objects, standardise_and_scale, fill_na_in_objects, one_hot_encode_categorical
from typing import Dict
from .visualisation import plot_graph
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

import zipfile
import io

@dataclass
class UrbanGraph:
    boundary: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    building: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    plot: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    street: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    intersection: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())

    boundary_to_plot: np.ndarray = field(default_factory=lambda: np.array('None'))
    plot_to_boundary: np.ndarray = None
    plot_to_plot: np.ndarray = None
    building_to_building: np.ndarray = None
    building_to_street: np.ndarray = None
    street_to_building: np.ndarray = None
    intersection_to_street: np.ndarray = None
    street_to_intersection: np.ndarray = None
    building_to_plot: np.ndarray = None
    plot_to_building: np.ndarray = None
    street_to_plot: np.ndarray = None
    plot_to_street: np.ndarray = None

    # We'll defer populating these until after the instance is created:
    geo_store: Dict = field(default_factory=dict, init=False)
    edge_store: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Called automatically after dataclass __init__ to populate geo_store and edge_store."""
        self.geo_store = {
            'boundary': self.boundary,
            'building': self.building,
            'plot': self.plot,
            'street': self.street,
            'intersection': self.intersection
        }

        self.edge_store = {
            'boundary_to_plot': self.boundary_to_plot,
            'plot_to_boundary': self.plot_to_boundary,
            'plot_to_plot': self.plot_to_plot,
            'building_to_building': self.building_to_building,
            'building_to_street': self.building_to_street,
            'street_to_building': self.street_to_building,
            'intersection_to_street': self.intersection_to_street,
            'street_to_intersection': self.street_to_intersection,
            'building_to_plot': self.building_to_plot,
            'plot_to_building': self.plot_to_building,
            'street_to_plot': self.street_to_plot,
            'plot_to_street': self.plot_to_street
        }

    def __repr__(self) -> str:
        """
        Return a string representation of the UrbanGraph, summarizing each property.
        """

        # Build strings for each
        info1 = [self.size_repr(k, v, 2) for k, v in self.geo_store.items()]
        info2 = [self.size_repr(k, v, 2) for k, v in self.edge_store.items()]

        info = ',\n'.join(info1 + info2)
        # Optionally add line breaks if not empty
        info = f'\n{info}\n' if info else ''

        return f'{self.__class__.__name__}({info})'
    
    def size_repr(self, name, data, indent=2):
        """
        Return a string that describes the size or shape of data.
        Indent controls how many spaces prefix the line.
        """
        prefix = ' ' * indent
        if data is None:
            return f"{prefix}{name}: None"
        if isinstance(data, gpd.GeoDataFrame):
            return f"{prefix}{name}: GeoDataFrame with {len(data)} rows"
        elif isinstance(data, pd.DataFrame):
            return f"{prefix}{name}: DataFrame with {len(data)} rows, {data.shape[1]} columns"
        elif isinstance(data, np.ndarray):
            return f"{prefix}{name}: ndarray shape {data.shape}"
        else:
            # Fallback for other types
            return f"{prefix}{name}: {type(data).__name__}"
        
    def initialize_edges(self, building_neighbours = 'knn', knn=5, distance=100):

        if self.boundary_to_plot.size == 1: 
            _, self.plot_to_plot = get_plot_to_plot_edges(self.plot)
            _, self.building_to_building = get_building_to_building_edges(self.building, 
                                                                        return_neighbours = building_neighbours,
                                                                        knn = knn,
                                                                        distance_threshold = distance,
                                                                        knn_threshold = distance, 
                                                                        add_reverse=True)
            
            self.boundary_to_plot, self.plot_to_boundary = boundary_to_plot(self.plot)
            self.street_to_plot, self.plot_to_street = get_edges_along_plot(self.plot)
            self.building_to_street, self.street_to_building = get_building_to_street_edges(self.street, self.building)
            self.intersection_to_street, self.street_to_intersection = get_intersection_to_street_edges(self.intersection, self.street)
            self.building_to_plot, self.plot_to_building = get_buildings_in_plot_edges(self.plot)
            self.street_to_plot, self.plot_to_street = get_edges_along_plot(self.plot)

            self.geo_store = {
                'boundary': self.boundary,
                'building': self.building,
                'plot': self.plot,
                'street': self.street,
                'intersection': self.intersection
            }

            self.edge_store = {
                'boundary_to_plot': self.boundary_to_plot,
                'plot_to_boundary': self.plot_to_boundary,
                'plot_to_plot': self.plot_to_plot,
                'building_to_building': self.building_to_building,
                'building_to_street': self.building_to_street,
                'street_to_building': self.street_to_building,
                'intersection_to_street': self.intersection_to_street,
                'street_to_intersection': self.street_to_intersection,
                'building_to_plot': self.building_to_plot,
                'plot_to_building': self.plot_to_building,
                'street_to_plot': self.street_to_plot,
                'plot_to_street': self.plot_to_street
            }

            # Process for downstream tasks
            self.geo_store = remove_non_numeric_columns_objects(self.geo_store, keep_geometry=True)
        else:
            print('Graph edges already initialized.')

    # Any additional methods (including __repr__) go here
    def plot_urban_graph(self, node_type="", colname="", node_id=""):
        graph_viz = plot_graph(self.geo_store, 
                               self.edge_store, 
                               node_type=node_type,
                               colname=colname,
                               node_id=node_id,
                               categorical=False)
        
    def save_graph(self, filename):
        with zipfile.ZipFile(filename, 'w') as z:
            # Save each GeoDataFrame as a GeoParquet file inside the zip
            for name, gdf in self.geo_store.items():
                if name == 'plot':
                    target_cols = ['street_id', 'bid']
                    for col in target_cols:
                        npy_name = f"plot_{col}.npy"
                        arr = np.array(self.geo_store[name][col].to_list(), dtype=object)
                        np.save(npy_name, arr, allow_pickle=True)
                        z.write(npy_name, arcname=npy_name)
                        os.remove(npy_name)  # clean up
                        gdf = gdf.drop(columns=col)
                parquet_name = f"{name}.parquet"
                gdf.to_parquet(parquet_name)
                z.write(parquet_name, arcname=parquet_name)
                os.remove(parquet_name)  # clean up

            # Save each numpy array
            for name, arr in self.edge_store.items():
                npy_name = f"{name}.npy"
                np.save(npy_name, arr)
                z.write(npy_name, arcname=npy_name)
                os.remove(npy_name)  # clean up

    def load_graph(self, filename):
        """
        Reads each .parquet and .npy file in the zip, storing them in two dictionaries.
        Returns:
            geodf_dict: dict of {filename_without_ext: GeoDataFrame}
            array_dict: dict of {filename_without_ext: numpy.ndarray}
        """
        geodf_dict = {}
        array_dict = {}
        
        with zipfile.ZipFile(filename, 'r') as z:
            # List all files in the ZIP archive
            for file in z.namelist():
                # If it's a GeoDataFrame in Parquet format
                if file.endswith('.parquet'):
                    with z.open(file) as f:
                        gdf = gpd.read_parquet(f)
                    # Use the filename (minus extension) as the dictionary key
                    dict_key = file.rsplit('.', 1)[0]
                    geodf_dict[dict_key] = gdf

            for file in z.namelist():
                # If it's a NumPy array
                if file.endswith('.npy'):
                    if 'id' in file:
                        with z.open(file) as f:
                            data = io.BytesIO(f.read())
                            arr = np.load(data, allow_pickle=True)
                            dict_key = file.rsplit('.', 1)[0][5:]
                            geodf_dict['plot'][dict_key] = arr
                    else:
                        with z.open(file) as f:
                            # Read the data into memory as bytes, then np.load from that
                            data = io.BytesIO(f.read())
                            arr = np.load(data, allow_pickle=True)
                        dict_key = file.rsplit('.', 1)[0]
                        array_dict[dict_key] = arr

        self.boundary = geodf_dict.get('boundary', None)
        self.building = geodf_dict.get('building', None)
        self.plot = geodf_dict.get('plot', None)
        self.street = geodf_dict.get('street', None)
        self.intersection = geodf_dict.get('intersection', None)

        self.boundary_to_plot = array_dict.get('boundary_to_plot', None)
        self.plot_to_boundary = array_dict.get('plot_to_boundary', None)
        self.plot_to_plot = array_dict.get('plot_to_plot', None)
        self.building_to_building = array_dict.get('building_to_building', None)
        self.building_to_street = array_dict.get('building_to_street', None)
        self.street_to_building = array_dict.get('street_to_building', None)
        self.intersection_to_street = array_dict.get('intersection_to_street', None)
        self.street_to_intersection = array_dict.get('street_to_intersection', None)
        self.building_to_plot = array_dict.get('building_to_plot', None)
        self.plot_to_building = array_dict.get('plot_to_building', None)
        self.street_to_plot = array_dict.get('street_to_plot', None)
        self.plot_to_street = array_dict.get('plot_to_street', None)

        # Optionally re-store them:
        self.geo_store = geodf_dict if geodf_dict else None
        self.edge_store = array_dict if array_dict else None

    def generate_masks(self, df_length: int,
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.2,
                       random_state: int = 42):
        """
        Generate boolean masks for train, validation, and test splits given 
        the number of rows in a GeoDataFrame and the desired split ratios.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.")

        indices = np.arange(df_length)
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

        train_end = int(train_ratio * df_length)
        val_end = train_end + int(val_ratio * df_length)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = np.zeros(df_length, dtype=bool)
        val_mask = np.zeros(df_length, dtype=bool)
        test_mask = np.zeros(df_length, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    def generate_building_masks(self,
                                train_ratio: float = 0.6,
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.2,
                                random_state: int = 42):
        """
        Generate train, validation, and test masks for the building GeoDataFrame.
        """
        return self.generate_masks(len(self.building),
                                   train_ratio,
                                   val_ratio,
                                   test_ratio,
                                   random_state)

    def generate_plot_masks(self,
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2,
                            test_ratio: float = 0.2,
                            random_state: int = 42):
        """
        Generate train, validation, and test masks for the plot GeoDataFrame.
        """
        return self.generate_masks(len(self.plot),
                                   train_ratio,
                                   val_ratio,
                                   test_ratio,
                                   random_state)

    def generate_street_masks(self,
                              train_ratio: float = 0.6,
                              val_ratio: float = 0.2,
                              test_ratio: float = 0.2,
                              random_state: int = 42):
        """
        Generate train, validation, and test masks for the street GeoDataFrame.
        """
        return self.generate_masks(len(self.street),
                                   train_ratio,
                                   val_ratio,
                                   test_ratio,
                                   random_state)

    def generate_intersection_masks(self,
                                    train_ratio: float = 0.6,
                                    val_ratio: float = 0.2,
                                    test_ratio: float = 0.2,
                                    random_state: int = 42):
        """
        Generate train, validation, and test masks for the intersection GeoDataFrame.
        """
        return self.generate_masks(len(self.intersection),
                                   train_ratio,
                                   val_ratio,
                                   test_ratio,
                                   random_state)

    def to_pyg_graph(self, target_col = 'building', target_value = []):

        data = HeteroData()
        objects_copy = self.geo_store.copy()

        node_types = ['boundary', 'plot','building','street','intersection']

        for node in node_types:
            objects_copy[node][f'{node}_id'] = range(len(objects_copy[node]))
            objects_copy[node] = objects_copy[node].drop(columns = ['geometry'], axis=1)
            data[node].x = torch.from_numpy(objects_copy[node].to_numpy().astype(np.float32))
                
        # Insert edges
        for key, arr in self.edge_store.items():
            splitted = key.split('_')
            if 'rev' in key:
                data[splitted[0], 'rev_to', splitted[-1]].edge_index = torch.from_numpy(arr).to(torch.int64)
            else:
                data[splitted[0], 'to', splitted[2]].edge_index = torch.from_numpy(arr).to(torch.int64)

        if target_value:
            data[target_col].y = torch.from_numpy(np.array(target_value))
        else:
            data[target_col].y = torch.from_numpy(np.array(np.random.randint(0,5,len(objects_copy[target_col]))))

        data = T.AddSelfLoops()(data)
        data = T.NormalizeFeatures()(data)

        return data

    def to_dgl():
        pass
    