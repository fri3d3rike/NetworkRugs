import networkx as nx
import random

def generate_dynamic_graphs(
    num_nodes=30,
    num_steps=10,
    num_groups=1,
    change_rate=0,
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    seed=42
):
    """
    Generate dynamic graphs with evolving group structures and guaranteed connectedness.

    Parameters:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of time steps (snapshots).
        num_groups (int): Number of initial communities.
        change_rate (float): Fraction of nodes changing group per step.
        intra_community_strength (float): Probability of edge within a community (0 to 1).
        inter_community_strength (float): Probability of edge between communities (0 to 1).
        seed (int): Random seed for reproducibility.

    Returns:
        graphs (dict): timestep → nx.Graph
        ground_truth (dict): timestep → {node: group}
        change_log (dict): timestep → list of changed node IDs
    """
    random.seed(seed)
    nodes = list(range(num_nodes))
    community_assignment = {node: node % num_groups for node in nodes}

    graphs = {}
    ground_truth = {}
    change_log = {}

    for t in range(num_steps):
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Add intra- and inter-community edges
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                same_group = community_assignment[i] == community_assignment[j]
                p = intra_community_strength if same_group else inter_community_strength
                if random.random() < p:
                    G.add_edge(i, j, weight=1.0 if same_group else 0.3)

        # Force minimal connectivity between communities
        community_nodes = {g: [] for g in range(num_groups)}
        for node, group in community_assignment.items():
            community_nodes[group].append(node)

        sorted_communities = sorted(community_nodes.keys())
        for i in range(len(sorted_communities) - 1):
            node_a = random.choice(community_nodes[sorted_communities[i]])
            node_b = random.choice(community_nodes[sorted_communities[i + 1]])
            G.add_edge(node_a, node_b, weight=0.01)  # Very weak connecting edge

        # Store current graph and group structure
        graphs[t] = G
        ground_truth[t] = community_assignment.copy()

        # Track community changes (from previous timestep)
        if t > 0:
            prev_assignment = ground_truth[t-1]
            changed_nodes = [
                node for node in nodes
                if community_assignment[node] != prev_assignment[node]
            ]
            change_log[t] = changed_nodes

        # Evolve group membership
        num_changes = int(change_rate * num_nodes)
        for _ in range(num_changes):
            node = random.choice(nodes)
            current_group = community_assignment[node]
            new_group = random.choice([g for g in range(num_groups) if g != current_group])
            community_assignment[node] = new_group

    return graphs, ground_truth, change_log


''' Split triggered at time T: We identify the group to split, say group g.
We gather that group’s nodes, shuffle them, and schedule them for “phased reassignment” over the next d steps.
Phased Reassignment from T to T+d−1:
Each step, we move a chunk of that group’s nodes to the new group ID.
By the end of d steps, the entire subset is reassigned.
This looks more organic in your network, rather than an abrupt jump. '''

# working on split events
def generate_split_data(
     num_nodes=30,
    num_steps=10,
    num_groups=3,
    change_rate=0.1,
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    # split_events is defined as { timestep: [(group_to_split, duration), ...] }
    
    split_events=None,
    split_fraction=0.5,
    seed=42
):
    """
    Generate dynamic graphs with evolving group structures and scheduled partial split events.

    Parameters:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of time steps.
        num_groups (int): Initial number of groups.
        change_rate (float): Fraction of nodes that change community each step.
        intra_community_strength (float): Edge probability within communities.
        inter_community_strength (float): Edge probability between communities.
        split_events (dict): { timestep: [(group_to_split, duration), ...] }
                             At a given timestep, only a fraction (here 50%) of nodes in the group will be split
                             over the specified duration.
        split_fraction (float): Fraction of nodes to move during a split event.
        seed (int): Random seed for reproducibility.

    Returns:
        graphs (dict): timestep → NetworkX graph
        ground_truth (dict): timestep → {node: group_id}
        change_log (dict): timestep → list of node IDs that changed group since the previous step
    """
    random.seed(seed)
    nodes = list(range(num_nodes))
    if split_events is None:
        default_duration = max(1, num_steps // 5)
        split_events = {num_steps // 2: [(0, default_duration)]}    # Default split at midpoint for group 0

    
    # Round-robin assignment
    community_assignment = {node: node % num_groups for node in nodes}

    # Track active groups (prevents nodes from switching into groups that aren't yet active)
    active_groups = set(range(num_groups))

    graphs = {}
    ground_truth = {}
    change_log = {}
    
    # Track partial splits in progress:
    # ongoing_splits is a list of dicts like:
    # { "end_step": t_end, "remaining_nodes": [...], "old_group": g_old, "new_group": g_new }
    ongoing_splits = []

    for t in range(num_steps):
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Build the graph based on current community_assignment
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                same_group = community_assignment[i] == community_assignment[j]
                p = intra_community_strength if same_group else inter_community_strength
                if random.random() < p:
                    G.add_edge(i, j, weight=1.0 if same_group else 0.3)

        # Force connectivity with weak links between communities (Needed for visualization!)
        community_nodes = {g: [] for g in set(community_assignment.values())}
        for node, grp in community_assignment.items():
            community_nodes[grp].append(node)

        sorted_communities = sorted(community_nodes.keys())
        for idx in range(len(sorted_communities) - 1):
            node_a = random.choice(community_nodes[sorted_communities[idx]])
            node_b = random.choice(community_nodes[sorted_communities[idx + 1]])
            G.add_edge(node_a, node_b, weight=0.01)

        # Record graph and ground truth at timestep t
        graphs[t] = G
        ground_truth[t] = community_assignment.copy()

        # Record changes from previous timestep
        if t > 0:
            prev_assignment = ground_truth[t - 1]
            changed_nodes = [node for node in nodes if community_assignment[node] != prev_assignment[node]]
            change_log[t] = changed_nodes

        # Process ongoing partial splits (phased split)
        still_ongoing = []
        for split_info in ongoing_splits:
            if t < split_info["end_step"]:
                steps_left = split_info["end_step"] - t
                # Determine how many nodes to move this step (using ceiling or floor, here we ensure at least one)
                num_remaining = len(split_info["remaining_nodes"])
                num_to_move = max(1, num_remaining // steps_left)
                # Get the list of nodes to move this timestep
                to_move = split_info["remaining_nodes"][:num_to_move]
                split_info["remaining_nodes"] = split_info["remaining_nodes"][num_to_move:]
                for node in to_move:
                    community_assignment[node] = split_info["new_group"]
                if split_info["remaining_nodes"]:
                    still_ongoing.append(split_info)
            # Else, the partial split is complete; do nothing.
        ongoing_splits = still_ongoing

        # Check for new split events at this time step
        if split_events and t in split_events:
            for (group_to_split, duration) in split_events[t]:
                # Identify nodes currently in the target group
                nodes_in_group = [n for n, g in community_assignment.items() if g == group_to_split]
                if len(nodes_in_group) >= 2:
                    # Select only half of the nodes to be moved for the split
                    split_size = int(len(nodes_in_group) * split_fraction)
                    nodes_to_move = random.sample(nodes_in_group, split_size)
                    new_group_id = max(active_groups) + 1
                    active_groups.add(new_group_id)
                    end_step = t + duration
                    ongoing_splits.append({
                        "end_step": end_step,
                        "remaining_nodes": nodes_to_move,
                        "old_group": group_to_split,
                        "new_group": new_group_id
                    })

        # Normal random evolution (node switches group among active groups)
        num_changes = int(change_rate * num_nodes)
        for _ in range(num_changes):
            node = random.choice(nodes)
            current_group = community_assignment[node]
            other_groups = [g for g in active_groups if g != current_group]
            if other_groups:
                new_group = random.choice(other_groups)
                community_assignment[node] = new_group

    return graphs, ground_truth, change_log