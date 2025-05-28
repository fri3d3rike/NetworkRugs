import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import itertools


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


def compute_global_stats(graphs):
    """
    Given a dict of timestamp→Graph, returns
    (min_w, max_w), (min_d, max_d), (min_c, max_c)
    for edge-weights, node-degrees, and common-neighbor counts.
    """
    all_ws, all_ds, all_cs = [], [], []
    for G in graphs.values():
        # collect edge weights
        for u,v,data in G.edges(data=True):
            all_ws.append(data.get('weight', 1))
        # collect node degrees
        for n in G.nodes():
            all_ds.append(G.degree(n))
        # collect counts of common neighbors for every pair
        for u,v in itertools.combinations(G.nodes(), 2):
            c = len(list(nx.common_neighbors(G, u, v)))
            all_cs.append(c)

    min_w, max_w = min(all_ws), max(all_ws)
    min_d, max_d = min(all_ds), max(all_ds)
    min_c, max_c = min(all_cs), max(all_cs)
    return (min_w, max_w), (min_d, max_d), (min_c, max_c)

def norm(x, xmin, xmax):
    """Normalize a value between xmin and xmax to [0,1] range."""
    return (x - xmin) / (xmax - xmin + 1e-9)


# Priority: combine edge weight, degree, and common neighbors
def calculate_priority(graph, current_node, neighbor, stats):
    """
    Calculates a priority score for a neighbor node based on a combination of three normalized structural metrics:

    avoids hub dominance by favoring low-degree neighbors and penalizes redundant paths by considering common neighbors

    1. Edge Weight (strength of the connection)
    2. Node Degree (connectivity of the neighbor)
    3. Common Neighbors (shared connections with the current node)

    Intended for use in BFS traversal strategies where a priority queue determines the order of node expansion. 
    The computed priority ensures that nodes with stronger and more structurally significant connections are visited earlier and less informative or overly connected nodes later.

    Priority formula:
    old:
        score = - (edge_weight_normalized + 1 / (degree_normalized + 1) + common_neighbors_normalized)
    new:
        score = -edge_weight_normalized - 1/(degree_normalized + 1) - common_neighbors_normalized
    where:
    - **edge_weight_normalized**: normalized edge weight (in [0,1])
    - **degree_normalized**: normalized degree (in [0,1])
    - **common_neighbors_normalized**: normalized common neighbors (in [0,1])

    Explanation of terms:
    - **-edge_weight_normalized**  
    Stronger edges have higher normalized weights (in [0,1])

    - **+ 1/(degree_normalized + 1)**  
    extra emphasis to low-degree neighbors, counteracts hub dominance by favoring exploration of less-connected neighbors

    - **-common_neighbors_normalized**  
    Neighbors that share many neighbors with the current node have higher normalized common-neighbor counts. Subtracting this term penalizes redundant paths, promoting discovery of novel graph regions.

    Note:
    - The priority is inverted: lower `score` indicates higher priority in a min-heap.
    - All input metrics are first normalized to the [0,1] range using global min/max statistics across timestamps.
    """

    (min_w, max_w), (min_d, max_d), (min_c, max_c) = stats

    edge_weight = graph[current_node][neighbor].get('weight', 1)  # Default weight is 1
    degree = graph.degree(neighbor)  # Higher degree = more connections
    common_neighbors = len(list(nx.common_neighbors(graph, current_node, neighbor)))  # Number of common neighbors

    edge_weight_normalized = norm(edge_weight, min_w, max_w)
    degree_normalized = norm(degree, min_d, max_d)
    common_neighbors_normalized = norm(common_neighbors, min_c, max_c)

    # prioritizes strong, low‐degree, high‐common‐neighbor nodes
    score = (
    - edge_weight_normalized        # strong edge → large negative contribution
    - 1.0 / (degree_normalized + 1) # low degree → large reciprocal → large negative contribution
    - common_neighbors_normalized   # many common neighbors → large negative contribution
    )

    # prioritizes strong, high‐degree, high‐common‐neighbor nodes
    score_old =  -edge_weight_normalized + 1 / (degree_normalized + 1) - common_neighbors_normalized
    
    #print("score:", score)
    # Negative because min-heap (lower priority = higher value)
    return score_old

def calculate_priority_normalized(graph, current, neigh, w, d, c, stats):
    """
    Tunable priority based on normalized edge weight, degree, and common neighbors.

    Args:
        graph: NetworkX graph
        current: Current node
        neigh: Neighbor node to evaluate
        w: weight factor for edge strength
        d: weight factor for high-degree preference
        c: weight factor for prioritizing shared neighborhood
        stats: tuple of three tuples ((min_w, max_w), (min_d, max_d), (min_c, max_c))

    Returns:
        float: Normalized priority score
    """
    (min_w, max_w), (min_d, max_d), (min_c, max_c) = stats

    # Raw metrics
    edge_weight = graph[current][neigh].get('weight', 1)
    degree = graph.degree(neigh)
    common_neighbors = len(list(nx.common_neighbors(graph, current, neigh)))

    # Normalize metrics
    w_n = norm(edge_weight, min_w, max_w)
    d_n = norm(degree, min_d, max_d)
    c_n = norm(common_neighbors, min_c, max_c)

    #print(w_n, d_n, c_n)

    # negative priority score for min-heap
    return -w * w_n - d * d_n - c * c_n


def sort_by_priority(graph, current_node, neighbors):
    return sorted(neighbors, key=lambda neighbor: calculate_priority(graph, current_node, neighbor))

def get_start_node(graphs, metric='degree', mode='highest'):
    """
    Select a single consistent start node, either:
      - 'highest' / 'lowest':   extreme on the first graph
      - 'highest_global' / 'lowest_global': extreme aggregated across all graphs
    Returns a dict mapping each timestamp to the chosen node.
    
    Raises:
        ValueError: If graphs is empty or if no valid nodes are found.
    """
    if not graphs:
        raise ValueError("Empty graph dictionary provided")
    def compute_metric(G):
        if metric == 'degree':
            return dict(G.degree())
        elif metric == 'closeness_centrality':
            return nx.closeness_centrality(G)
        elif metric == 'betweenness_centrality':
            return nx.betweenness_centrality(G)
        elif metric == 'eigenvector_centrality':
            return nx.eigenvector_centrality(G, max_iter=10000, tol=1e-6, weight='weight')
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    timestamps = sorted(graphs.keys())
    first_ts = timestamps[0]

    if mode in ('highest', 'lowest'):
        # compute only on the first graph
        values = compute_metric(graphs[first_ts])

    elif mode in ('highest_global', 'lowest_global'):
        # aggregate across all graphs
        agg = {}        # node -> sum of metric over all graphs
        for G in graphs.values():
            vals = compute_metric(G)
            for node, score in vals.items():
                agg[node] = agg.get(node, 0.0) + score
        values = agg

    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    # pick the node with highest or lowest value
    if 'highest' in mode:
        chosen = max(values, key=values.get)
    else:
        chosen = min(values, key=values.get)

    # same start node for every timestamp
    return {ts: chosen for ts in timestamps}



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


def get_priority_bfs_ordering(graphs, start_nodes=None, stats=None):
    """
    Perform a priority-based Breadth-First Search (BFS) traversal on a series of graphs.

    This function computes a BFS ordering for each graph in the input dictionary, where nodes are visited
    based on a priority score. The priority score is calculated using the `calculate_priority` function,
    which combines edge weight, node degree, and common neighbors.

    Args:
        graphs (dict): A dictionary where keys are timestamps and values are NetworkX graphs.
        start_nodes (dict, optional): A dictionary specifying the starting node for each timestamp. 
                                      If not provided, the first node (sorted by ID) is used as the start node.
        stats (tuple, optional): Global statistics for normalization. If not provided, they are computed from the graphs.
    
    Returns:
        dict: A dictionary where keys are timestamps and values are lists of nodes in BFS order.

    Example:
        graphs = {
            0: nx.Graph([(1, 2), (2, 3), (3, 4)]),
            1: nx.Graph([(1, 3), (3, 4), (4, 5)])
        }
        start_nodes = {0: 1, 1: 3}
        bfs_ordering = get_priority_bfs_ordering(graphs, start_nodes)
        print(bfs_ordering)
        # Output: {0: [1, 2, 3, 4], 1: [3, 1, 4, 5]}
    """
    bfs_ordering = {}

    if stats is None:
        stats = compute_global_stats(graphs)
        print("Global stats:", stats)

    # Loop through each graph/timestamp
    for timestamp, graph in graphs.items():
        sorted_nodes = sorted(graph.nodes)

        visited = set()
        ordering = []
        pq = []  # Priority queue (min-heap)

        # Determine the starting node
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
                        priority = calculate_priority(graph, current_node, neighbor, stats)
                        heapq.heappush(pq, (priority, neighbor))
        bfs_ordering[timestamp] = ordering

    return bfs_ordering


def get_tunable_priority_bfs_ordering(graphs, start_nodes=None, stats=None, w=1.0, d=1.0, c=1.0):
    """
    Perform BFS with tunable priority on each graph timestamp.

    Calculates a normalized priority score for a neighbor node based on a combination of edge weight,
    node degree, and number of common neighbors.

    Args:
        graphs (dict): timestamp -> networkx.Graph
        start_nodes (dict): timestamp -> preferred start node
        stats (tuple): Global statistics for normalization
        w (float): Coefficient for edge weight
        d (float): Coefficient for degree preference
        c (float): Coefficient for common neighbors

    Returns:
        dict: timestamp -> list of nodes in priority BFS order
    """
    if stats is None:
        stats = compute_global_stats(graphs)
        print("Global stats:", stats)
        
    bfs_ordering = {}
    for timestamp, graph in graphs.items():
        visited = set()
        ordering = []
        pq = []
        # Determine start node
        start = start_nodes.get(timestamp) if start_nodes else None
        if start and start in graph:
            heapq.heappush(pq, (0.0, start))
        else:
            heapq.heappush(pq, (0.0, sorted(graph.nodes())[0]))
        # Traverse
        while pq:
            priority, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            ordering.append(node)
            for nbr in graph.neighbors(node):
                if nbr not in visited:
                    pr = calculate_priority_normalized(graph, node, nbr, w, d, c, stats)
                    heapq.heappush(pq, (pr, nbr))
        bfs_ordering[timestamp] = ordering
    return bfs_ordering
