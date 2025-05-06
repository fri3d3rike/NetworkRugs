import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain


# nodes with higher weights: come first
# nodes with equal weights:  sorted by their ID in ascending order
def sort_by_weight(graph, node):
            return sorted(graph.neighbors(node), 
                  key=lambda neighbor: (-graph[node][neighbor].get('weight', 1), neighbor))

# nodes with equal weights:  sorted by their ID in descending order
def sort_by_weight_desID(graph, node):
    return sorted(graph.neighbors(node), 
                  key=lambda neighbor: (-graph[node][neighbor].get('weight', 1), -neighbor))

def sort_by_id(graph, node):
    return sorted(graph[node], reverse=False)

# TODO: check this function
def sort_by_common_neighbors(graph, node):
    neighbors = list(graph.neighbors(node))
    #neighbors.sort(key=lambda neighbor: len(list(nx.common_neighbors(graph, node, neighbor))), reverse=True)
    neighbors.sort(key=lambda neighbor: (-len(list(nx.common_neighbors(graph, node, neighbor))), neighbor))
    #print(timestamp, node, neighbors)
    return neighbors

# Priority: combine edge weight, degree, and common neighbors
def calculate_priority(graph, current_node, neighbor):
    """
    - calculates a priority score for a neighbor node based on a combination of edge weight,
    node degree, and number of common neighbors
    - intended for use in BFS traversal strategies where a priority queue is
    used to determine the order of node expansion
    - the computed priority ensures that nodes with stronger and more structurally significant connections are visited earlier.

    Priority is defined as:
        -edge_weight + 1 / (degree + 1) - number_of_common_neighbors

    Note:
        The priority is inverted (i.e., lower values are higher priority in a min-heap).

    float
        The computed priority score (lower values indicate higher priority).
    """
    edge_weight = graph[current_node][neighbor].get('weight', 1)  # Default weight is 1
    degree = graph.degree(neighbor)  # Higher degree = more connections
    common_neighbors = len(list(nx.common_neighbors(graph, current_node, neighbor)))  # Number of common neighbors

    # Negative because min-heap (lower priority = higher value)
    return -edge_weight + 1 / (degree + 1) - common_neighbors

def sort_by_priority(graph, current_node, neighbors):
    return sorted(neighbors, key=lambda neighbor: calculate_priority(graph, current_node, neighbor))

def get_start_node(graphs, metric='degree', mode='highest'):
    """
    Select a single consistent start node based on the first graph (timestamp 0 by default).
    """
    first_timestamp = min(graphs.keys())
    G0 = graphs[first_timestamp]
    
    if metric == 'degree':
        values = dict(G0.degree())
    elif metric == 'closeness_centrality':
        values = nx.closeness_centrality(G0)
    elif metric == 'betweenness_centrality':
        values = nx.betweenness_centrality(G0)
    elif metric == 'eigenvector_centrality':
        values = nx.eigenvector_centrality(G0, max_iter=10000, tol=1e-6, weight='weight')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if mode == 'highest':
        chosen_node = max(values, key=values.get)
    elif mode == 'lowest':
        chosen_node = min(values, key=values.get)
    else:
        raise ValueError(f"Mode must be 'highest' or 'lowest'.")

    # Now use same node for all timestamps
    start_nodes = {timestamp: chosen_node for timestamp in graphs.keys()}
    
    return start_nodes


def get_DFS_ordering(graphs, start_nodes=None):
    """
    Perform DFS on each graph, prioritizing edges with the highest influence.

    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        start_nodes (dict, optional): A dictionary specifying the starting node for each timestamp.

    Returns:
        dict: A dictionary containing DFS ordering of nodes for each graph.
    """
    dfs_ordering = {}
    #print("Graphs received:", graphs)

    # loop through each graph/timestamp
    for timestamp, graph in graphs.items():
        sorted_nodes = sorted(graph.nodes)

        visited = set()
        ordering = []

        # DFS traversal function
        def dfs(node):
            if node not in visited:
                visited.add(node)
                ordering.append(node)
                neighbors = sort_by_weight(graph, node)
                #print(f"Node {node} has neighbors {neighbors}")
                #print(f"Visited nodes: {visited}")
                #print(f"Current ordering: {ordering}")
                #print("next to visit:", neighbors[0])
                # recursively visit neighbors
                for neighbor in neighbors:
                    dfs(neighbor)

        # Determine the starting node
        start_node = start_nodes.get(timestamp) if start_nodes and timestamp in start_nodes else None

        if start_node and start_node in graph.nodes:
            #print(f"Timestamp {timestamp}: Starting DFS with specified start node {start_node}")
            dfs(start_node)

        # Perform DFS from any remaining unvisited nodes
        for node in sorted_nodes:
            if node not in visited:
                #print(f"Timestamp {timestamp}: Starting DFS with node {node} (sorted order)")
                dfs(node)

        dfs_ordering[timestamp] = ordering
        #print(f"DFS ordering for {sorted_nodes}:", ordering)

    return dfs_ordering


def get_BFS_ordering(graphs, start_nodes=None, sorting_key='weight'):
    """
    Perform BFS on each graph, prioritizing edges with the highest weight (or influence).
    
    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        start_nodes (dict, optional): A dictionary specifying the starting node for each timestamp.

    Returns:
        dict: A dictionary containing BFS ordering of nodes for each graph.
    """
    bfs_ordering = {}
    #print("Graphs received:", graphs)

    for timestamp, graph in graphs.items():
        sorted_nodes = sorted(graph.nodes)

        visited = set()
        ordering = []

        # BFS traversal function
        def bfs(node):
            queue = [node]
            visited.add(node)

            while queue:
                current_node = queue.pop(0)
                ordering.append(current_node)

                if sorting_key == 'weight':
                    neighbors = sort_by_weight(graph, current_node)
                elif sorting_key == 'weight_desID':
                    neighbors = sort_by_weight_desID(graph, current_node)
                elif sorting_key == 'id':
                    neighbors = sort_by_id(graph, current_node)
                elif sorting_key == 'common_neighbors':
                    neighbors = sort_by_common_neighbors(graph, current_node)
                else:
                    raise ValueError("Invalid sorting key specified.")
                    
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Determine the starting node
        start_node = start_nodes.get(timestamp) if start_nodes and timestamp in start_nodes else None

        if start_node and start_node in graph.nodes:
            # Start BFS with the specified start node if provided
            bfs(start_node)

        # Perform BFS from any remaining unvisited nodes
        for node in sorted_nodes:
            if node not in visited:
                bfs(node)

        bfs_ordering[timestamp] = ordering

    return bfs_ordering


def get_degree_ordering(graphs, reverse=True):
    """
    Get the degree-based ordering of nodes for each graph.
    
    If degrees are the same, nodes are sorted by their IDs.

    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        reverse (bool, optional): Whether to reverse the ordering based on degree.

    Returns:
        dict: A dictionary containing degree-based ordering of nodes for each graph.
    """
    degree_ordering = {}

    for timestamp, graph in graphs.items():
        # Sort by degree (primary) and node ID (secondary)
        sorted_nodes = sorted(graph.nodes, key=lambda node: (-graph.degree(node), node) if reverse else (graph.degree(node), node))
        degree_ordering[timestamp] = sorted_nodes

    return degree_ordering


def get_centrality_ordering(graphs, centrality_measure='degree', reverse=False):
    """
    Get the node ordering based on centrality measures for each graph.

    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        centrality_measure (str, optional): The centrality measure to use ('degree', 'closeness', 'betweenness', 'eigenvector').
        reverse (bool, optional): Whether to reverse the ordering based on centrality. Defaults to descending (most central first).

    Returns:
        dict: A dictionary containing centrality-based ordering of nodes for each graph.
    """
    centrality_ordering = {}

    for timestamp, graph in graphs.items():
        # Compute centrality based on the selected measure
        if centrality_measure == 'degree':
            centrality = nx.degree_centrality(graph)
        elif centrality_measure == 'closeness':
            centrality = nx.closeness_centrality(graph)
        elif centrality_measure == 'betweenness':
            centrality = nx.betweenness_centrality(graph)
        elif centrality_measure == 'eigenvector':
            centrality = nx.eigenvector_centrality(graph, max_iter=10000, tol=1e-6, weight='weight')
        else:
            raise ValueError("Invalid centrality measure specified.")

        # Sort by centrality (descending) and by ID (ascending)
        sorted_nodes = sorted(
            graph.nodes,
            key=lambda node: (-centrality[node], node) if not reverse else (centrality[node], node)
        )
        centrality_ordering[timestamp] = sorted_nodes

    return centrality_ordering


def get_community_ordering(graphs, sorting_key='id'):
    """
    Get the node ordering based on community detection for each graph.

    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        sorting_key (str, optional): The key to sort nodes within each community ('id' or 'degree').

    Returns:
        dict: A dictionary containing community-based ordering of nodes for each graph.
    """
    neighborhoods_ordering = {}

    for timestamp, graph in graphs.items():
        # Detect communities using the Louvain method
        partition = community_louvain.best_partition(graph)

        # Create a dictionary to hold nodes for each community
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        # Sort nodes within each community based on the sorting_key
        for community in communities:
            if sorting_key == 'id':
                communities[community] = sorted(communities[community])
            elif sorting_key == 'degree':
                communities[community] = sorted(communities[community], key=lambda node: (-graph.degree(node), node))

        # Combine the sorted nodes from all communities
        sorted_nodes = []
        for community in sorted(communities.keys()):
            sorted_nodes.extend(communities[community])

        neighborhoods_ordering[timestamp] = sorted_nodes

    return neighborhoods_ordering


def get_priority_bfs_ordering(graphs, start_nodes=None):
    """
    Perform BFS with priority on each graph, prioritizing edges with the highest influence.

    Args:
        graphs (dict): A dictionary of NetworkX graphs for each timestamp.
        start_nodes (dict, optional): A dictionary specifying the starting node for each timestamp.

    Returns:
        dict: A dictionary containing BFS ordering of nodes for each graph.
    """
    bfs_ordering = {}

    # Loop through each graph/timestamp
    for timestamp, graph in graphs.items():
        sorted_nodes = sorted(graph.nodes)

        visited = set()
        ordering = []
        pq = []  # Priority queue (min-heap)

        start_node = start_nodes.get(timestamp) if start_nodes and timestamp in start_nodes else None
        if start_node and start_node in graph.nodes:
            heapq.heappush(pq, (0, start_node))  # Push (priority, node)
        else:
            heapq.heappush(pq, (0, sorted_nodes[0]))

        # Priority BFS traversal
        while pq:
            _, current_node = heapq.heappop(pq)  # Pop the node with the highest priority
            if current_node not in visited:
                visited.add(current_node)
                ordering.append(current_node)

                # Get neighbors and calculate their priorities
                neighbors = graph.neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        priority = calculate_priority(graph, current_node, neighbor)
                        heapq.heappush(pq, (priority, neighbor))

        bfs_ordering[timestamp] = ordering

    return bfs_ordering
