import json
from datetime import datetime
import networkx as nx
from networkx.readwrite import json_graph
from copy import deepcopy
from typing import Dict, Tuple, List, Any
from functools import partial
from networkx import MultiDiGraph


def load_json(json_filename: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Given, a json filename, loads that file and returns the node and edge dictionaries
    :param json_filename: string representing the file name of a json file
    :return:
    nodes: list of nodes and their attributes
    edges: list of edges and their attributes
    """
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
        nodes = data['nodes']
        edges = data['edges']
    return nodes, edges


def make_graph(nodes: List[str], edges: Tuple[str, str, Dict]) -> MultiDiGraph:
    """
    Given a list of nodes represented by their pub_keys, and a list of edges represented by tuples like so:
    [(u,v,dict of edge attributes)...], make a multi-directed graph and return it.
    Note: some edges may include nodes that are not in the list of nodes. By default, NetworkX will implicitly
    create these nodes in the graph. In order to prevent this behavior, a subgraph is created using the nodes provided.
    :param nodes:
    :param edges:
    :return:
    g: a NetworkX MultiDirectedGraph
    """
    g = nx.MultiDiGraph()
    g.add_edges_from(edges)
    g = g.subgraph(nodes)
    return g


def get_timestamp_from_filename(json_filename: str) -> float:
    """
    Given a json filename, extracts datetime from filename and converts it to a timestamp.
    :param json_filename: string representing the file name of a json file
    :return:
    timestamp: float representing POSIX time of snapshot
    """
    date = json_filename.rstrip('.json')
    date_time_obj = datetime.strptime(date, '%Y-%m-%d_%H:%M:%S')
    timestamp = datetime.timestamp(date_time_obj)
    return timestamp


def channel_fee_function(channel: Dict[str, Any]) -> Tuple[float, float]:
    """
    Given an channel dictionary, calculate the cost to FORWARD a fixed sized payment in either direction,
    (from node 1 to node 2 and from node 2 to node 1), according to the node's policies.
    Fee base and fee rate are in different units. msats = 1/1000 of a sat, milli_msats = 1/1,000,000 of a sat
    Fees are returned in sats.
    Note: there is no fee to SEND payment between adjacent nodes.
    :param channel: dictionary representing a channel, including fee information
    :return:
    node1_fee_sats: float representing fee in sats to forward payment from node 1 to node 2
    node2_fee_sats: float representing fee in sats to forward payment from node 2 to node 1
    """
    payment_amount = 1_000_000
    mmsats_per_msats = 1_000
    sats_per_mmsats = 1_000_000

    n1_scaling_fee_rate = int(channel['node1_policy']['fee_rate_milli_msat'])
    n1_base_fee_rate = int(channel['node1_policy']['fee_base_msat']) * mmsats_per_msats
    node1_fee_mmsats = (payment_amount * n1_scaling_fee_rate) + n1_base_fee_rate

    n2_scaling_fee_rate = int(channel['node2_policy']['fee_rate_milli_msat'])
    n2_base_fee_rate = int(channel['node2_policy']['fee_base_msat']) * mmsats_per_msats
    node2_fee_mmsats = (payment_amount * n2_scaling_fee_rate) + n2_base_fee_rate

    node1_fee_sats = node1_fee_mmsats / sats_per_mmsats
    node2_fee_sats = node2_fee_mmsats / sats_per_mmsats

    return node1_fee_sats, node2_fee_sats


def updated_recently(snapshot_time: float, last_update_time: float) -> bool:
    """
    Given the timestamp that a snapshot was taken and the timestamp when a particular node was last updated,
    determine whether the node had updated within some time limit relative to the time the snapshot was taken.
    Time limit is in days.
    :param snapshot_time: float representing the POSIX timestamp the snapshot was taken
    :param last_update_time: float representing the POSIX timestamp a particular node had updated
    :return: bool representing whether the node had updated within some time limit
    """
    time_limit_days = 180  # ~1/2 year
    seconds_per_day = 86400
    seconds_since_last_update = snapshot_time - last_update_time
    days_since_last_update = seconds_since_last_update / seconds_per_day
    return days_since_last_update <= time_limit_days


def has_minimum_degree(g, node: str) -> bool:
    """
    Given a NetworkX Graph and a particular node in that graph, determine whether the node has a minimum degree.
    :param g: nx Graph
    :param node: node in g
    :return:
    bool representing whether node has some minimum degree
    """
    minimum_degree = 2
    return g.degree(node) >= minimum_degree


def has_minimum_capacity(channel: Dict[str, Any], minimum_capacity: int) -> bool:
    """
    Given a channel, determine if that channel has a minimum capacity.
    Capacity represents the maximum amount of sats a channel can SEND/RECEIVE/FORWARD, assuming correct liquidity.
    :param channel: dictionary representing a channel, including capacity information
    :param minimum_capacity: int defining minimum capacity of a given channel
    :return:
    bool representing whether channel has some minimum capacity
    """
    channel_capacity = int(channel['capacity'])
    return channel_capacity >= minimum_capacity


def has_both_node_policies(channel: Dict[str, Any]) -> bool:
    """
    Given a channel, determine if both nodes involved in that channel have well-defined policies.
    :param channel: dictionary representing a channel, including policy information
    :return:
    bool representing whether channel has both policies
    """
    return channel['node1_policy'] is not None and channel['node2_policy'] is not None


def has_both_active_policies(channel: Dict[str, Any]):
    """
    Given a channel, determine if both node policies are active (not disabled).
    :param channel: dictionary representing a channel, including policy information
    :return:
    bool representing whether both policies in channel are active (not disabled)
    """
    n1_policy = channel["node1_policy"]
    n2_policy = channel["node2_policy"]
    return not n1_policy["disabled"] and not n2_policy["disabled"]


def clean_nodes(nodes: List[Dict], json_filename: str) -> List[Dict]:
    """
    Given a list of nodes and a json filename, filter for nodes that have not been updated recently.
    :param nodes: list of nodes and their attributes
    :param json_filename: string representing the file name of a json file
    :return:
    list of nodes that have all been updated recently
    """
    timestamp = get_timestamp_from_filename(json_filename)
    node_filter = partial(updated_recently, snapshot_time=timestamp)
    active_nodes = list(filter(lambda x: node_filter(last_update_time=x["last_update"]), nodes))
    return active_nodes


def get_pubkeys(nodes: List[Dict]) -> List[str]:
    """
    Given a list of nodes and their attributes, return a list of the nodes pub_keys.
    :param nodes: list of nodes and their attributes
    :return:
    list of strings representing node pub_keys
    """
    return [node["pub_key"] for node in nodes]


def clean_edges(edges: List[Dict], config) -> List[Dict]:
    """
    Given a list of edges and their attributes, filter for channels that do not meet requirements:
    - channel has some minimum capacity
    - both channel policies are defined
    - both channel policies are active
    :param edges: list of edges and their attributes
    :param config: settings in regard to how to filter edges
    :return:
    list of edges that meet certain requirements
    """
    capacity = config.getint("minimum_capacity")

    def channel_filter(channel):
        return has_minimum_capacity(channel, capacity) and \
               has_both_node_policies(channel) and \
               has_both_active_policies(channel)

    return list(filter(channel_filter, edges))


def get_channels_with_fees(edges: List[Dict]) -> List[Tuple[str, str, Dict]]:
    """
    Given a list of edges and their attributes, calculate the cost to forward payments in either direction,
    and create a list of tuples as such: [(u pub_key, v_pub_key, cost),...].
    :param edges: list of edges and their attributes
    :return:
    list of tuples (str, str, float) representing edges and their costs
    """
    channels = []
    for edge in edges:
        n1_weight, n2_weight = channel_fee_function(edge)
        edge1 = (edge['node1_pub'], edge['node2_pub'], {"weight": n1_weight})
        edge2 = (edge['node2_pub'], edge['node1_pub'], {"weight": n2_weight})
        channels.extend([edge1, edge2])
    return channels


def reduce_to_mainnet(nx_graph):
    """
    Given a NetworkX Graph, make an undirected copy, remove all bridges from the copy,
    take the largest connected component (mainnet) created by the removal of these edges,
    and return a subgraph containing all nodes from that component.
    This has the effect of removing not only pre-existing smaller components, but also nodes of degree 1,
    and nodes who are dependent on the liquidity of a single channel (the bridge) to send payments in the network.
    :param nx_graph:
    :return:
    """
    graph_copy = deepcopy(nx_graph.to_undirected())
    bridges = list(nx.bridges(graph_copy))
    graph_copy.remove_edges_from(bridges)
    all_subgraphs = nx.connected_components(graph_copy)
    largest_subgraph = max(all_subgraphs, key=len)
    return nx_graph.subgraph(largest_subgraph)


def save_graph(g, json_filename: str) -> None:
    """
    Given a NetworkX graph and a json filename, save that graph to a file by that filename.
    :param g: NetworkX graph
    :param json_filename: string representing filename to save graph into
    :return: None
    """
    graph_json = json_graph.node_link_data(g)
    keys = ["directed", "multigraph", "graph"]
    [graph_json.pop(key) for key in keys]
    with open(json_filename, "w") as clean_json_file:
        json.dump(graph_json, clean_json_file)


def simplify_graph(multidi_g):
    """
    Given a MultiDiGraph, create a DiGraph such that if a pair of nodes has more than one edge between them,
    the edge with the minimum weight is retained.
    :param multidi_g: NetworkX MultiDiGraph
    :return: NetworkX DiGraph
    """
    di_g = nx.DiGraph()
    for u, v, data in multidi_g.edges(data=True):
        w = data['weight']
        if di_g.has_edge(u, v):
            di_g[u][v]['weight'] = min(di_g[u][v]['weight'], w)
        else:
            di_g.add_edge(u, v, weight=w)
    return di_g
