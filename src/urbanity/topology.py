# import external packages
import networkx as nx
import numpy as np
import pandas as pd
import networkit

def compute_centrality(G, G_nodes, function, colname, T_nodes, *args, temporal=False):
    """Employs networkit fast centrality computation to obtain various network centrality measures.

    Args:
        G (nx.MultiDiGraph): Urban network graph object with nodes and edges information. 
        G_nodes (gpd.GeoDataFrame): Geopandas dataframe object with node attributes. 
        function (networkit.centrality.Centrality): Centrality function from networkit.centrality module. Accepts (e.g., EigenvectorCentrality, KatzCentrality, BetweennessCentrality).
        colname (str): New column to add computed centrality indicators to node attribute dataframe.

    Returns:
        gpd.GeoDataFrame: Node attribute dataframe with network centrality measures included. 
    """    
    G_nodes_mod = G_nodes.copy()
    
    nk_G = networkit.nxadapter.nx2nk(G, weightAttr=None)
    nk_centrality = function(nk_G, *args)
    nk_centrality.run()

    value = [v for k,v in nk_centrality.ranking()[:]]
    if temporal:
        T_nodes_mod= T_nodes.copy()
        T_nodes_mod[colname] = value
        T_nodes_mod = T_nodes_mod[['osmid', colname]]
        G_nodes_mod = pd.merge(G_nodes_mod, T_nodes_mod, on='osmid', how='left')
        G_nodes_mod[colname] = G_nodes_mod[colname].fillna(0)
        G_nodes_mod[colname] = G_nodes_mod[colname].round(3)
    else:
        
        G_nodes_mod[colname] = value
        G_nodes_mod[colname] = G_nodes_mod[colname].round(3)
    
    return G_nodes_mod

def merge_nx_attr(G, G_nodes, nxfunction, colname, T_nodes, temporal=False ,**kwargs):
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
    G_nodes_mod = G_nodes.copy()
    

    try:
        attr_dict = nxfunction(G, **kwargs)
    
    # else, convert to nx.MultiDiGraph
    except nx.exception.NetworkXNotImplemented:
        G2 = nx.DiGraph(G)
        attr_dict = nxfunction(G2, **kwargs)
    
    if temporal:
        T_nodes_mod = T_nodes.copy()
        T_nodes_mod[colname] = list(attr_dict.values())
        T_nodes_mod = T_nodes_mod[['osmid', colname]]
        G_nodes_mod = pd.merge(G_nodes_mod, T_nodes_mod, on='osmid', how='left')
        G_nodes_mod[colname] = G_nodes_mod[colname].fillna(0)
        G_nodes_mod[colname] = G_nodes_mod[colname].round(3)
    else:
        G_nodes_mod[colname] = list(attr_dict.values())
        G_nodes_mod[colname] = G_nodes_mod[colname].round(3)
    return G_nodes_mod


def merge_nx_property(G_nodes, nxproperty, colname, T_nodes, temporal=False, *args):
    """Add graph property from networkx graph into node attribute dataframe.

    Args:
        G_nodes (gpd.GeoDataFrame): Geopandas dataframe object with node attributes.
        nxproperty (nx.MultiDiGraph.out_degree): Out degree method of networkx graph.
        colname (str): New column to add computed centrality indicators to node attribute dataframe.

    Returns:
        gpd.GeoDataFrame: Node attribute dataframe with network property included.
    """    
    G_nodes_mod = G_nodes.copy()

    attr_dict = dict(nxproperty)
    if temporal:
        T_nodes_mod = T_nodes.copy()
        T_nodes_mod[colname] = list(attr_dict.values())
        T_nodes_mod = T_nodes_mod[['osmid', colname]]
        G_nodes_mod = pd.merge(G_nodes_mod, T_nodes_mod, on='osmid', how='left')
        G_nodes_mod[colname] = G_nodes_mod[colname].fillna(0)
    else:
        G_nodes_mod[colname] = list(attr_dict.values())
    return G_nodes_mod