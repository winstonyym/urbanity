# import external packages
import networkx as nx
import numpy as np
import pandas as pd
import networkit

def compute_centrality(G, G_nodes, function, colname, *args):
    """Employs networkit fast centrality computation to obtain various network centrality measures.

    Args:
        G (nx.MultiDiGraph): Urban network graph object with nodes and edges information. 
        G_nodes (gpd.GeoDataFrame): Geopandas dataframe object with node attributes. 
        function (networkit.centrality.Centrality): Centrality function from networkit.centrality module. Accepts (e.g., EigenvectorCentrality, KatzCentrality, BetweennessCentrality).
        colname (str): New column to add computed centrality indicators to node attribute dataframe.

    Returns:
        gpd.GeoDataFrame: Node attribute dataframe with network centrality measures included. 
    """    

    # Convert to networkx to networkit graph object 
    nk_G = networkit.nxadapter.nx2nk(G, weightAttr=None)
    nk_centrality = function(nk_G, *args)

    # Compute centrality 
    nk_centrality.run()
    value = [v for k,v in nk_centrality.ranking()[:]]
    G_nodes[colname] = value
    
    return G_nodes


def merge_nx_attr(G, G_nodes, nxfunction, colname, **kwargs):
    """Add graph attributes from networkx graph into node attribute dataframe. 

    Args:
        G (nx.MultiDiGraph): Urban network graph with nodes and edges information. 
        G_nodes (gpd.GeoDataFrame): Geopandas dataframe object with node attributes. 
        nxfunction (nx.algorithms.cluster.clustering): Clustering function from networkx
        colname (str): New column to add computed centrality indicators to node attribute dataframe.

    Returns:
        gpd.GeoDataFrame: Node attribute dataframe with network clustering measures included. 
    """  
    # If Graph is nx.MultiDiGraph  
    try:
        attr_dict = nxfunction(G, **kwargs)
    
    # else, convert to nx.MultiDiGraph
    except nx.exception.NetworkXNotImplemented:
        G2 = nx.DiGraph(G)
        attr_dict = nxfunction(G2, **kwargs)
    
    G_nodes[colname] = list(attr_dict.values())

    return G_nodes


def merge_nx_property(G_nodes, nxproperty, colname, *args):
    """Add graph property from networkx graph into node attribute dataframe.

    Args:
        G_nodes (gpd.GeoDataFrame): Geopandas dataframe object with node attributes.
        nxproperty (nx.MultiDiGraph.out_degree): Out degree method of networkx graph.
        colname (str): New column to add computed centrality indicators to node attribute dataframe.

    Returns:
        gpd.GeoDataFrame: Node attribute dataframe with network property included.
    """    
    attr_dict = dict(nxproperty)
    G_nodes[colname] = list(attr_dict.values())

    return G_nodes