from dataclasses import dataclass, field
import geopandas as gpd
import pandas as pd
import numpy as np
import h5py
import pickle
from .utils import get_plot_to_plot_edges, get_building_to_street_edges, get_edge_nodes, get_building_to_building_edges, \
                   get_intersection_to_street_edges, get_buildings_in_plot_edges, get_edges_along_plot, boundary_to_plot, \
                   remove_non_numeric_columns_objects, standardise_and_scale, fill_na_in_objects, one_hot_encode_categorical, \
                   save_to_h5, save_to_npz

from .visualisation import plot_graph


@dataclass
class UrbanGraph:
    boundary: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    building: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    plot: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    street: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())
    intersection: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())

    # Edges
    boundary_to_plot: np.ndarray = None
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

    def __repr__(self) -> str:
        """
        Return a string representation of the UrbanGraph, summarizing each property.
        """
        # You could store them in separate dictionaries if you like:
        # For demonstration, we'll create them on the fly:
        geo_store = {
            'boundary': self.boundary,
            'building': self.building,
            'plot': self.plot,
            'street': self.street,
            'intersection': self.intersection
        }

        edge_store = {
            'boundary_to_plot': self.boundary_to_plot,
            'plot_to_boundary': self.plot_to_boundary,
            'plot_to_plot': self.plot_to_plot,
            'building_to_building': self.building_to_building,
            'plot_to_street': self.plot_to_street,
            'street_to_plot': self.street_to_plot,
            'building_to_street': self.building_to_street,
            'street_to_building': self.street_to_building,
            'intersection_to_street': self.intersection_to_street,
            'street_to_intersection': self.street_to_intersection,
            'building_to_plot': self.building_to_plot,
            'plot_to_building': self.plot_to_building,
            'street_to_plot': self.street_to_plot,
            'plot_to_street': self.plot_to_street
        }

        # Build strings for each
        info1 = [self.size_repr(k, v, 2) for k, v in geo_store.items()]
        info2 = [self.size_repr(k, v, 2) for k, v in edge_store.items()]

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
        
    def initialize_edges(self, return_neighbours = 'knn', knn=5, threshold=100):
        _, self.plot_to_plot = get_plot_to_plot_edges(self.plot)
        _, self.building_to_building = get_building_to_building_edges(self.building, 
                                                                      return_neighbours = return_neighbours,
                                                                      knn = knn,
                                                                      distance_threshold = threshold,
                                                                      knn_threshold = threshold, 
                                                                      add_reverse=True)
        
        self.boundary_to_plot, self.plot_to_boundary = boundary_to_plot(self.plot)
        self.street_to_plot, self.plot_to_street = get_edges_along_plot(self.plot)
        self.building_to_street, self.street_to_building = get_building_to_street_edges(self.street, self.building)
        self.intersection_to_street, self.street_to_intersection = get_intersection_to_street_edges(self.intersection, self.street)
        self.building_to_plot, self.plot_to_building = get_buildings_in_plot_edges(self.plot)
        self.street_to_plot, self.plot_to_street = get_edges_along_plot(self.plot)

    # Any additional methods (including __repr__) go here
    def plot_urban_graph(self, node_id=''):
        deck = plot_graph(node_id = node_id)
        return deck
    
    def save_to_h5(self, filepath: str):
        """
        Use the provided `save_to_h5` function to store the UrbanGraph data.
        """
        # We'll split the data into two dicts: one for GDFs, one for arrays
        gdf_dict = {
            'building': self.building,
            'plot': self.plot,
            'street': self.street,
            'intersection': self.intersection
        }
        array_dict = {
            'building_plot_edges': self.building_plot_edges,
            'building_street_edges': self.building_street_edges,
            'building_intersection_edges': self.building_intersection_edges,
            'plot_street_edges': self.plot_street_edges,
            'plot_intersection_edges': self.plot_intersection_edges,
            'street_intersection_edges': self.street_intersection_edges
        }

        save_to_h5(filepath, gdf_dict, array_dict)

    @classmethod
    def load_from_h5(cls, filepath: str):
        """
        Use the companion `load_from_h5` function to retrieve the data 
        and instantiate an UrbanGraph.
        """
        gdf_dict, array_dict = load_from_h5(filepath)
        return cls(
            building=gdf_dict.get('building', None),
            plot=gdf_dict.get('plot', None),
            street=gdf_dict.get('street', None),
            intersection=gdf_dict.get('intersection', None),
            building_plot_edges=array_dict.get('building_plot_edges', None),
            building_street_edges=array_dict.get('building_street_edges', None),
            building_intersection_edges=array_dict.get('building_intersection_edges', None),
            plot_street_edges=array_dict.get('plot_street_edges', None),
            plot_intersection_edges=array_dict.get('plot_intersection_edges', None),
            street_intersection_edges=array_dict.get('street_intersection_edges', None)
        )

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

    def to_pyg():
        pass

    def to_dgl():
        pass
    
    def save_to_h5(self, filename: str):
        """
        Save the UrbanGraph to an HDF5 file. This method uses 'pickle' to store
        each GeoDataFrame (including geometry) and each edges array. If your data
        are large, you might want a more specialized approach.
        """
        with h5py.File(filename, 'w') as hf:
            # Store each GeoDataFrame as a pickled byte string
            building_pickled = pickle.dumps(self.building)
            plot_pickled = pickle.dumps(self.plot)
            street_pickled = pickle.dumps(self.street)
            intersection_pickled = pickle.dumps(self.intersection)

            hf.create_dataset('building', data=np.void(building_pickled))
            hf.create_dataset('plot', data=np.void(plot_pickled))
            hf.create_dataset('street', data=np.void(street_pickled))
            hf.create_dataset('intersection', data=np.void(intersection_pickled))

            # Store each edge array if present
            if self.building_plot_edges is not None:
                hf.create_dataset('building_plot_edges', data=self.building_plot_edges)
            if self.building_street_edges is not None:
                hf.create_dataset('building_street_edges', data=self.building_street_edges)
            if self.building_intersection_edges is not None:
                hf.create_dataset('building_intersection_edges', data=self.building_intersection_edges)
            if self.plot_street_edges is not None:
                hf.create_dataset('plot_street_edges', data=self.plot_street_edges)
            if self.plot_intersection_edges is not None:
                hf.create_dataset('plot_intersection_edges', data=self.plot_intersection_edges)
            if self.street_intersection_edges is not None:
                hf.create_dataset('street_intersection_edges', data=self.street_intersection_edges)

    @classmethod
    def load_from_h5(cls, filename: str):
        """
        Load an UrbanGraph instance from an HDF5 file previously saved by 'save_to_h5'.
        """
        with h5py.File(filename, 'r') as hf:
            # Unpickle each GeoDataFrame
            building = pickle.loads(hf['building'][()].tobytes())
            plot = pickle.loads(hf['plot'][()].tobytes())
            street = pickle.loads(hf['street'][()].tobytes())
            intersection = pickle.loads(hf['intersection'][()].tobytes())

            # Retrieve edge arrays if present in the file
            def _get_if_exists(h5_file, dataset_name):
                return h5_file[dataset_name][()] if dataset_name in h5_file else None

            building_plot_edges = _get_if_exists(hf, 'building_plot_edges')
            building_street_edges = _get_if_exists(hf, 'building_street_edges')
            building_intersection_edges = _get_if_exists(hf, 'building_intersection_edges')
            plot_street_edges = _get_if_exists(hf, 'plot_street_edges')
            plot_intersection_edges = _get_if_exists(hf, 'plot_intersection_edges')
            street_intersection_edges = _get_if_exists(hf, 'street_intersection_edges')

        return cls(building=building,
                   plot=plot,
                   street=street,
                   intersection=intersection,
                   building_plot_edges=building_plot_edges,
                   building_street_edges=building_street_edges,
                   building_intersection_edges=building_intersection_edges,
                   plot_street_edges=plot_street_edges,
                   plot_intersection_edges=plot_intersection_edges,
                   street_intersection_edges=street_intersection_edges)