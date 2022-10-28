import networkx as nx
import numpy as np
import pandas as pd
import networkit

def compute_centrality(G, G_nodes, function, colname, *args):
    nk_G = networkit.nxadapter.nx2nk(G, weightAttr=None)
    nk_centrality = function(nk_G, *args)
    nk_centrality.run()
    value = [v for k,v in nk_centrality.ranking()[:]]
    G_nodes[colname] = value
    
    return G_nodes


def merge_nx_attr(G, G_nodes, nxfunction, colname, **kwargs):
    try:
        attr_dict = nxfunction(G, **kwargs)
        
    except nx.exception.NetworkXNotImplemented:
        G2 = nx.DiGraph(G)
        attr_dict = nxfunction(G2, **kwargs)
    
    G_nodes[colname] = list(attr_dict.values())

    return G_nodes


def merge_nx_property(G_nodes, nxproperty, colname, *args):
    
    attr_dict = dict(nxproperty)
    G_nodes[colname] = list(attr_dict.values())

    return G_nodes