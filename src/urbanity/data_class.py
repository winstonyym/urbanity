# import geopandas as gpd
# import numpy as np
# import pandas as pd
# import h5py
# import pickle

# class UrbanGraph:
#     """
#     A class to represent an 'UrbanGraph' object containing four GeoDataFrames
#     (building, plot, street, intersection) and optional edge connections 
#     between them in 2 x N integer arrays.
#     """

#     def __init__(self,
#                  building: gpd.GeoDataFrame = None,
#                  plot: gpd.GeoDataFrame = None,
#                  street: gpd.GeoDataFrame = None,
#                  intersection: gpd.GeoDataFrame = None,
#                  building_plot_edges: np.ndarray = None,
#                  building_street_edges: np.ndarray = None,
#                  building_intersection_edges: np.ndarray = None,
#                  plot_street_edges: np.ndarray = None,
#                  plot_intersection_edges: np.ndarray = None,
#                  street_intersection_edges: np.ndarray = None):
#         """
#         Initialize the UrbanGraph with the provided GeoDataFrames and edge arrays.
#         The edges are assumed to be 2 x N arrays where each column is a pair of 
#         indices referencing rows in the corresponding GeoDataFrames.
#         """
#         self.building = building if building is not None else gpd.GeoDataFrame()
#         self.plot = plot if plot is not None else gpd.GeoDataFrame()
#         self.street = street if street is not None else gpd.GeoDataFrame()
#         self.intersection = intersection if intersection is not None else gpd.GeoDataFrame()

#         # Pairs of edges for different combinations
#         self.building_plot_edges = building_plot_edges
#         self.building_street_edges = building_street_edges
#         self.building_intersection_edges = building_intersection_edges
#         self.plot_street_edges = plot_street_edges
#         self.plot_intersection_edges = plot_intersection_edges
#         self.street_intersection_edges = street_intersection_edges

#     def describe(self):
#         """
#         Print a summary of the UrbanGraph, including the number of records in 
#         each GeoDataFrame and the number of edges in each 2xN array.
#         """
#         # Print counts for each GeoDataFrame
#         print("UrbanGraph Description:")
#         print(f"  Buildings:        {len(self.building)} entries")
#         print(f"  Plots:            {len(self.plot)} entries")
#         print(f"  Streets:          {len(self.street)} entries")
#         print(f"  Intersections:    {len(self.intersection)} entries")
#         print()

#         # Print shapes for each edge array (if present)
#         def _edge_shape(edge_array):
#             return edge_array.shape if edge_array is not None else (0, 0)

#         print("Edge arrays (2 x N):")
#         print(f"  building_plot_edges:         {_edge_shape(self.building_plot_edges)}")
#         print(f"  building_street_edges:       {_edge_shape(self.building_street_edges)}")
#         print(f"  building_intersection_edges: {_edge_shape(self.building_intersection_edges)}")
#         print(f"  plot_street_edges:           {_edge_shape(self.plot_street_edges)}")
#         print(f"  plot_intersection_edges:     {_edge_shape(self.plot_intersection_edges)}")
#         print(f"  street_intersection_edges:   {_edge_shape(self.street_intersection_edges)}")

#     def generate_masks(self, df_length: int,
#                        train_ratio: float = 0.6,
#                        val_ratio: float = 0.2,
#                        test_ratio: float = 0.2,
#                        random_state: int = 42):
#         """
#         Generate boolean masks for train, validation, and test splits given 
#         the number of rows in a GeoDataFrame and the desired split ratios.
#         """
#         if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
#             raise ValueError("Train, validation, and test ratios must sum to 1.")

#         indices = np.arange(df_length)
#         rng = np.random.default_rng(random_state)
#         rng.shuffle(indices)

#         train_end = int(train_ratio * df_length)
#         val_end = train_end + int(val_ratio * df_length)

#         train_idx = indices[:train_end]
#         val_idx = indices[train_end:val_end]
#         test_idx = indices[val_end:]

#         train_mask = np.zeros(df_length, dtype=bool)
#         val_mask = np.zeros(df_length, dtype=bool)
#         test_mask = np.zeros(df_length, dtype=bool)

#         train_mask[train_idx] = True
#         val_mask[val_idx] = True
#         test_mask[test_idx] = True

#         return train_mask, val_mask, test_mask

#     def generate_building_masks(self,
#                                 train_ratio: float = 0.6,
#                                 val_ratio: float = 0.2,
#                                 test_ratio: float = 0.2,
#                                 random_state: int = 42):
#         """
#         Generate train, validation, and test masks for the building GeoDataFrame.
#         """
#         return self.generate_masks(len(self.building),
#                                    train_ratio,
#                                    val_ratio,
#                                    test_ratio,
#                                    random_state)

#     def generate_plot_masks(self,
#                             train_ratio: float = 0.6,
#                             val_ratio: float = 0.2,
#                             test_ratio: float = 0.2,
#                             random_state: int = 42):
#         """
#         Generate train, validation, and test masks for the plot GeoDataFrame.
#         """
#         return self.generate_masks(len(self.plot),
#                                    train_ratio,
#                                    val_ratio,
#                                    test_ratio,
#                                    random_state)

#     def generate_street_masks(self,
#                               train_ratio: float = 0.6,
#                               val_ratio: float = 0.2,
#                               test_ratio: float = 0.2,
#                               random_state: int = 42):
#         """
#         Generate train, validation, and test masks for the street GeoDataFrame.
#         """
#         return self.generate_masks(len(self.street),
#                                    train_ratio,
#                                    val_ratio,
#                                    test_ratio,
#                                    random_state)

#     def generate_intersection_masks(self,
#                                     train_ratio: float = 0.6,
#                                     val_ratio: float = 0.2,
#                                     test_ratio: float = 0.2,
#                                     random_state: int = 42):
#         """
#         Generate train, validation, and test masks for the intersection GeoDataFrame.
#         """
#         return self.generate_masks(len(self.intersection),
#                                    train_ratio,
#                                    val_ratio,
#                                    test_ratio,
#                                    random_state)

#     def to_pyg():
#         pass

#     def to_dgl():
#         pass
    
#     def save_to_h5(self, filename: str):
#         """
#         Save the UrbanGraph to an HDF5 file. This method uses 'pickle' to store
#         each GeoDataFrame (including geometry) and each edges array. If your data
#         are large, you might want a more specialized approach.
#         """
#         with h5py.File(filename, 'w') as hf:
#             # Store each GeoDataFrame as a pickled byte string
#             building_pickled = pickle.dumps(self.building)
#             plot_pickled = pickle.dumps(self.plot)
#             street_pickled = pickle.dumps(self.street)
#             intersection_pickled = pickle.dumps(self.intersection)

#             hf.create_dataset('building', data=np.void(building_pickled))
#             hf.create_dataset('plot', data=np.void(plot_pickled))
#             hf.create_dataset('street', data=np.void(street_pickled))
#             hf.create_dataset('intersection', data=np.void(intersection_pickled))

#             # Store each edge array if present
#             if self.building_plot_edges is not None:
#                 hf.create_dataset('building_plot_edges', data=self.building_plot_edges)
#             if self.building_street_edges is not None:
#                 hf.create_dataset('building_street_edges', data=self.building_street_edges)
#             if self.building_intersection_edges is not None:
#                 hf.create_dataset('building_intersection_edges', data=self.building_intersection_edges)
#             if self.plot_street_edges is not None:
#                 hf.create_dataset('plot_street_edges', data=self.plot_street_edges)
#             if self.plot_intersection_edges is not None:
#                 hf.create_dataset('plot_intersection_edges', data=self.plot_intersection_edges)
#             if self.street_intersection_edges is not None:
#                 hf.create_dataset('street_intersection_edges', data=self.street_intersection_edges)

#     @classmethod
#     def load_from_h5(cls, filename: str):
#         """
#         Load an UrbanGraph instance from an HDF5 file previously saved by 'save_to_h5'.
#         """
#         with h5py.File(filename, 'r') as hf:
#             # Unpickle each GeoDataFrame
#             building = pickle.loads(hf['building'][()].tobytes())
#             plot = pickle.loads(hf['plot'][()].tobytes())
#             street = pickle.loads(hf['street'][()].tobytes())
#             intersection = pickle.loads(hf['intersection'][()].tobytes())

#             # Retrieve edge arrays if present in the file
#             def _get_if_exists(h5_file, dataset_name):
#                 return h5_file[dataset_name][()] if dataset_name in h5_file else None

#             building_plot_edges = _get_if_exists(hf, 'building_plot_edges')
#             building_street_edges = _get_if_exists(hf, 'building_street_edges')
#             building_intersection_edges = _get_if_exists(hf, 'building_intersection_edges')
#             plot_street_edges = _get_if_exists(hf, 'plot_street_edges')
#             plot_intersection_edges = _get_if_exists(hf, 'plot_intersection_edges')
#             street_intersection_edges = _get_if_exists(hf, 'street_intersection_edges')

#         return cls(building=building,
#                    plot=plot,
#                    street=street,
#                    intersection=intersection,
#                    building_plot_edges=building_plot_edges,
#                    building_street_edges=building_street_edges,
#                    building_intersection_edges=building_intersection_edges,
#                    plot_street_edges=plot_street_edges,
#                    plot_intersection_edges=plot_intersection_edges,
#                    street_intersection_edges=street_intersection_edges)