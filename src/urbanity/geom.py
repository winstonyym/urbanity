# Geometric helper functions
from shapely.ops import split
from functools import partial
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

def project_gdf(gdf):
    """Utility function to project GeoDataFrames into local coordinates.

    Args:
        gdf (gpd.GeoDataFrame): A geopandas dataframe object.

    Returns:
        gpd.GeoDataFrame: A geopandas dataframe object projected to local coordinate.
    """    
    # Get representative point
    mean_longitude = gdf["geometry"].representative_point().x.mean()

    # Compute UTM crs
    utm_zone = int(np.floor((mean_longitude + 180) / 6) + 1)
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    # project the GeoDataFrame to the UTM CRS
    gdf_proj = gdf.to_crs(utm_crs)
    # print(f"Projected to {gdf_proj.crs}")
    
    return gdf_proj

def buffer_polygon(gdf, bandwidth = 200):
    """Utility function to buffer geometry by bandwidth (m). 
    Geodataframes are first projected to local coordinates to allow for metric computation before being
    re-projected to global coordinates. 

    Args:
        gdf (gpd.GeoDataFrame): A geopandas dataframe as input to be buffered.
        bandwidth (int, optional): Distance to buffer geometry. Defaults to 200.

    Returns:
        gpd.GeoDataFrame: A buffered geopandas dataframe with global projection.
    """    
    buffer_zone = project_gdf(gdf).buffer(bandwidth)
    buffered_gdf = buffer_zone.to_crs(4326)
    return buffered_gdf

def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """Computes the great-circle distance (straightline) between point pairs.

    Args:
        lat1 (float): Latitude of first point.
        lng1 (float): Longitude of first point.
        lat2 (float): Latitude of second point. 
        lng2 (_type_): Longitude of second point. 
        earth_radius (int, optional): Radius of earth. Defaults to 6_371_009.

    Returns:
        float: Great circle distance between point 1 and point 2.
    """    

    y1 = np.deg2rad(lat1)
    y2 = np.deg2rad(lat2)
    dy = y2 - y1

    x1 = np.deg2rad(lng1)
    x2 = np.deg2rad(lng2)
    dx = x2 - x1

    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    return arc * earth_radius


def add_edge_lengths(G, precision=3):
    """Function to incorporate edge length information into a network graph (G).

    Args:
        G (nx.MultiDiGraph): Urban street network graph with nodes and edges information.
        precision (int, optional): Number of decimal places to round edge lengths. Defaults to 3.

    Raises:
        KeyError: Missing edge or node information.

    Returns:
        nx.MultiDiGraph: Urban street network graph with added edge length information.
    """    
    uvk = tuple(G.edges)
    x = G.nodes(data="x")
    y = G.nodes(data="y")
    
    try:
        # two-dimensional array of coordinates: y0, x0, y1, x1
        c = np.array([(y[u], x[u], y[v], x[v]) for u, v, k in uvk])
        
    except KeyError:  
        raise KeyError("some edges missing nodes, possibly due to input data clipping issue")
        
    dists = great_circle_vec(c[:, 0], c[:, 1], c[:, 2], c[:, 3]).round(precision)
    dists[np.isnan(dists)] = 0
    nx.set_edge_attributes(G, values=dict(zip(uvk, dists)), name="length")
    
    return G

def _is_endpoint(G, node, strict=True):
    """Wrapper function to check if nodes are endpoints from OSMnx: https://github.com/gboeing/osmnx
    Return nodes that correspond to real intersections or deadends in a street network.

    Args:
        G (nx.MultiDiGraph): Urban street network with nodes and edge information.
        node (int): osmid of target node
        strict (bool, optional): If False, allow nodes with different osmid edges to be end points 
        even if they fail specified endpoint rules. Defaults to True.

    Returns:
        bool: True/False to whether node is an endpoint.
    """

    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    elif not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]["osmid"])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]["osmid"])

        # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    # if none of the preceding rules returned true, then it is not an endpoint
    else:
        return False
    
def _build_path(G, endpoint, endpoint_successor, endpoints):
    """Wrapper function to build path and connect endpoint nodes from OSMnx: https://github.com/gboeing/osmnx
    Returns a osmid sequence where first and last item correspond to endpoints. 

    Args:
        G (_type_): Urban network graph with nodes and edges information.
        endpoint (int): osmid for start node
        endpoint_successor (int): osmid for subsequent interspatially connected nodes
        endpoints (set): Set of all osmids corresponding to endpoint nodes.

    Raises:
        Exception: Unexpected simplify pattern handling.

    Returns:
        list: A sequence of osmids where first and last item are endpoints.
    """    
    
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:  # pragma: no cover
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        utils.log(
                            f"Unexpected simplify pattern handled near {successor}", level=lg.WARN
                        )
                        return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    raise Exception(f"Unexpected simplify pattern failed near {successor}")

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path

def _get_paths_to_simplify(G, strict=True):
    """Wrapper function to obtain list of simplified paths OSMnx: https://github.com/gboeing/osmnx
    Returns a list of paths to be simplfiied. 

    Args:
        G (nx.MultiDiGraph): Urban network graph with nodes and edges information.
        strict (bool, optional): If False, allow nodes with different osmid edges to be end points 
        even if they fail specified endpoint rules. Defaults to True.

    Yields:
        list: List of simplified paths.
    """    

    # first identify all the nodes that are endpoints
    endpoints = set([n for n in G.nodes if _is_endpoint(G, n, strict=strict)])

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)
                
def simplify_graph(G, strict=True, remove_rings=True):
    """Wrapper function to simplified network graph from OSMnx: https://github.com/gboeing/osmnx
    Returns a simplified graph where interstitial nodes are removed and geometry of original edges are preserved. 

    Args:
        G (nx.MultiDiGraph): Urban network graph with nodes and edges information.
        strict (bool, optional): If False, allow nodes with different osmid edges to be end points 
        even if they fail specified endpoint rules. Defaults to True.
        remove_rings (bool, optional): Remove isolated rings with self-loops. Defaults to True.

    Raises:
        Exception: Error if graph has already been simplified.

    Returns:
        nx.MultiDiGraph: Returns a simplified graph that have interstitial nodes removed. 
    """
    
    if "simplified" in G.graph and G.graph["simplified"]:  # pragma: no cover
        raise Exception("This graph has already been simplified, cannot simplify it again.")


    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, strict=strict):

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        path_attributes = dict()
        for u, v in zip(path[:-1], path[1:]):

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                utils.log(f"Found {edge_count} edges between {u} and {v} when simplifying")

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            edge_data = G.edges[u, v, 0]
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            if attr in attrs_to_sum:
                # if this attribute must be summed, sum it now
                path_attributes[attr] = sum(path_attributes[attr])
            elif len(path_attributes[attr]) == 1:
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th):
                path_attributes[attr] = path_attributes[attr][0]
            else:
                # otherwise, if there are multiple values, keep one of each
                path_attributes[attr] = tuple(path_attributes[attr])

        # construct the new consolidated edge's geometry for this path
        path_attributes["geometry"] = LineString(
            [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
        )

        # add the nodes and edge to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = nx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(_is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

    # mark graph as having been simplified
    G.graph["simplified"] = True

    return G

def graph_to_gdf(G, nodes=False, edges=False, dual=False):
    """Utility function to obtain both nodes and edges dataframes from a networkx graph object. 

    Args:
        G (nx.MultiDiGraph): Urban network graph with nodes and edges information.
        nodes (bool, optional): If True, returns nodes dataframe. Defaults to False.
        edges (bool, optional): If True, returns edges dataframe. Defaults to False.
        dual (bool, optional): If True, returns both nodes and edges dataframes. Defaults to False.

    Raises:
        ValueError: Graph does not contain any nodes.
        ValueError: Graph does not contain any edges.

    Returns:
        gpd.GeoDataFrame: A geopandas dataframe object with attribute table for nodes and edges. 
    """    

    crs = G.graph['crs']
    if nodes:
        if not G.nodes:  # pragma: no cover
            raise ValueError("graph contains no nodes")
            
        nodes, data = zip(*G.nodes(data=True))
        
        # convert node x/y attributes to Points for geometry column
        geom = (Point(d["x"], d["y"]) for d in data)
        gdf_nodes = gpd.GeoDataFrame(data, index=nodes, crs=crs, geometry=list(geom))
         
        if not dual:
            gdf_nodes.index.rename("osmid", inplace=True)
        
    if edges:
        if not G.edges:  # pragma: no cover
            raise ValueError("graph contains no edges")
        
        if not dual: 
            u, v, k, data = zip(*G.edges(keys=True, data=True))
        else: 
            u, v, data = zip(*G.edges(data=True))
            
        x_lookup = nx.get_node_attributes(G, "x")
        y_lookup = nx.get_node_attributes(G, "y")
        def make_geom(u, v, data, x= x_lookup, y= y_lookup):
            if "geometry" in data:
                return data["geometry"]
            else:
                return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))
            
        geom = map(make_geom, u, v, data)
        gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))


        # add u, v, key attributes as index
        gdf_edges["u"] = u
        gdf_edges["v"] = v
        if not dual: 
            gdf_edges["key"] = k
            gdf_edges.set_index(["u", "v", "key"], inplace=True)
        else: 
            gdf_edges.set_index(["u", "v"], inplace=True)
    if nodes and edges:
        return gdf_nodes, gdf_edges
    elif nodes:
        return gdf_nodes
    elif edges:
        return gdf_edges

def fill_and_expand(gdf):
    """Function to expand multilinestring and multipolygons into Polygon,
     to allow for overlay operation when computing building attributes.

    Args:
        gdf (geopandas.geodataframe.GeoDataFrame): Dataframe consisting of network nodes.

    Returns:
        geopandas.geodataframe.GeoDataFrame: Modified dataframe with homogenous geometry type. 
    """    
    gdf2 = gdf.copy()
    for i, geom in enumerate(gdf.geometry):

        if geom.geom_type == 'LineString':
            gdf2 = gdf2.drop(i)

        elif geom.geom_type=='MultiLineString':
            linestring = gdf.loc[[i],:]
            linestring_exploded = linestring.explode(index_parts=True)

            # Find bounding polygon
            poly_len = len(linestring_exploded)
            list_of_minx = list(linestring_exploded.bounds.minx)
            min_ind = list_of_minx.index(min(list_of_minx))
            
            # Create list of hole polygons
            holes = []
            for k, geom in enumerate(linestring_exploded.geometry):
                if (k != min_ind) and (geom.geom_type!='LineString'):
                    holes.append(list(geom.coords))

            try:
                polygon_geom = Polygon(list(list(linestring_exploded.geometry.iloc[min_ind].coords)), 
                                  holes = holes)
                polygon = gpd.GeoDataFrame(index=[0], crs=linestring.crs, geometry=[polygon_geom]) 
                gdf2 = gpd.GeoDataFrame(pd.concat([gdf2, polygon], ignore_index=True), crs=gdf2.crs) 

            except ValueError:
                gdf2 = gdf2.drop(i)    
                
    for i, geom in enumerate(gdf2.geometry):
        if geom.geom_type == 'MultiLineString':
            gdf2 = gdf2.drop(i)
            
    # Expand MultiPolygons
    gdf2 = gdf2.explode(index_parts=True)
    
    return gdf2