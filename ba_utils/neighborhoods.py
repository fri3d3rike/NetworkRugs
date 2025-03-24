import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt


def get_influence(node, neighbor, graph):
    if graph.has_edge(node, neighbor):
        return graph[node][neighbor].get("influence", 0)
    return 0 

def get_weight(node1, node2, graph):
    return graph[node1][node2].get('weight', 1)

def get_connected_components(graph):
    connected_components = []
    for component in nx.connected_components(graph):
        connected_components.append(list(component))
    return connected_components

def get_nearest_neighbors(graph, node, k=1):
    nearest_neighbors = []
    visited = set()
    queue = [(0, node)]
    while queue:
        distance, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if distance > k:
            break
        if current != node:
            nearest_neighbors.append(current)
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                heapq.heappush(queue, (distance + get_influence(current, neighbor, graph), neighbor))
    return nearest_neighbors

# Get all nearest neighbors of a node across all timestamps
def get_all_nearest_neighbors(graphs_dict, node, k=1):
    nearest_neighbors_set = set()
    for timestamp, graph in graphs_dict.items():
        nearest_neighbors = get_nearest_neighbors(graph, node, k)
        nearest_neighbors_set.update(set(nearest_neighbors))
    return nearest_neighbors_set

def get_common_neighbors(graph, node1, node2):
    return list(nx.common_neighbors(graph, node1, node2))

# Get all common neighbors of two nodes across all timestamps
def get_all_common_neighbors(graphs_dict, node1, node2):
    common_neighbors_set = set()
    for timestamp, graph in graphs_dict.items():
        common_neighors = get_common_neighbors(graph, node1, node2)
        common_neighbors_set.update(set(common_neighors))
    return common_neighbors_set