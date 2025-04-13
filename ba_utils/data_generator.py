import networkx as nx
import random
import math

def generate_dynamic_graphs_old(
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

def generate_dynamic_graphs(
    num_nodes=30,
    num_steps=10,
    initial_groups=1,
    change_rate=0,
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    split_events=None,
    merge_events=None,
    split_fraction=0.5,
    merge_fraction=1.0,
    init_mode="block",
    seed=42
):
    """
    Generate dynamic graphs with evolving group structures and scheduled partial split and merge events.
    
    Parameters:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of timesteps (snapshots).
        initial_groups (int): Initial number of communities.
        change_rate (float): Fraction of nodes that change groups randomly each step.
        intra_community_strength (float): Probability of an edge within a community.
        inter_community_strength (float): Probability of an edge between communities.
        split_events (dict): { timestep: [(group_to_split, duration), ...] }
                             At a given timestep, a partial split is scheduled for the specified group.
        merge_events (dict): { timestep: [(src, dst, duration), ...] }
                             At a given timestep, a merge is scheduled to gradually move nodes from group src to group dst.
        split_fraction (float): Fraction of nodes to move during a split event.
        merge_fraction (float): Fraction of nodes to move during a merge event (default=1.0 means complete merge).
        seed (int): Random seed for reproducibility.
    
    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since the previous timestep.
    """
    random.seed(seed)
    nodes = list(range(num_nodes))

    community_assignment = {}

    if init_mode == "round_robin":
        # Initial round-robin assignment (if num_groups==1, all nodes start in group 0)
        community_assignment = {node: node % initial_groups for node in nodes}

    elif init_mode == "block":
        nodes_per_group = num_nodes // initial_groups
        for g in range(initial_groups):
            group_nodes = nodes[g * nodes_per_group : (g + 1) * nodes_per_group]
            for node in group_nodes:
                community_assignment[node] = g
        #leftover nodes
        for node in nodes[initial_groups * nodes_per_group:]:
            community_assignment[node] = initial_groups - 1


    
    
    
    # Track active groups (these are the group IDs available so far)
    active_groups = set(range(initial_groups))
    
    graphs = {}
    ground_truth = {}
    change_log = {}
    
    # Ongoing partial split events:
    # Each entry is a dict: { "end_step": t_end, "remaining_nodes": [...], "old_group": g_old, "new_group": g_new }
    ongoing_splits = []
    # Ongoing partial merge events:
    # Each entry is a dict: { "end_step": t_end, "remaining_nodes": [...], "src": src, "dst": dst }
    ongoing_merges = []
    
    for t in range(num_steps):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        # Build the graph edges based on current community_assignment.
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                same_group = community_assignment[i] == community_assignment[j]
                p = intra_community_strength if same_group else inter_community_strength
                if random.random() < p:
                    # Use weight 1.0 for intra-group, 0.3 for inter-group
                    G.add_edge(i, j, weight=1.0 if same_group else 0.3)
        
        # Force minimal connectivity between communities (to help with visualization)
        # Here, we add a very weak edge between a randomly chosen node from each adjacent group.
        community_nodes = {}
        for group in set(community_assignment.values()):
            community_nodes[group] = [node for node, grp in community_assignment.items() if grp == group]
        sorted_groups = sorted(community_nodes.keys())
        for idx in range(len(sorted_groups)-1):
            node_a = random.choice(community_nodes[sorted_groups[idx]])
            node_b = random.choice(community_nodes[sorted_groups[idx+1]])
            G.add_edge(node_a, node_b, weight=0.01)
        
        # Record current graph and ground truth.
        graphs[t] = G
        ground_truth[t] = community_assignment.copy()
        if t > 0:
            prev_assignment = ground_truth[t-1]
            changed_nodes = [node for node in nodes if community_assignment[node] != prev_assignment[node]]
            change_log[t] = changed_nodes
        
        still_ongoing_merges = []
        for merge_info in ongoing_merges:
            if t < merge_info["start"]:
                still_ongoing_merges.append(merge_info)
                continue
            # calculated how many swapps
            elapsed = t - merge_info["start"]
            expected_moves = int(elapsed // merge_info["interval"]) + 1
            moves_to_do = expected_moves - merge_info["nodes_moved"]
            if moves_to_do > 0:
                num_available = len(merge_info["remaining_nodes"])
                num_to_move = min(moves_to_do, num_available)
                to_move = merge_info["remaining_nodes"][:num_to_move]
                merge_info["remaining_nodes"] = merge_info["remaining_nodes"][num_to_move:]
                for node in to_move:
                    community_assignment[node] = merge_info["dst"]
                merge_info["nodes_moved"] += num_to_move
            if t < merge_info["end_step"] and merge_info["remaining_nodes"]:
                still_ongoing_merges.append(merge_info)
            else:
                # Merge-Event ist zu Ende; falls keine Knoten mehr in der Quellgruppe sind, entferne sie.
                remaining_in_src = [n for n, g in community_assignment.items() if g == merge_info["src"]]
                if not remaining_in_src:
                    active_groups.discard(merge_info["src"])
        ongoing_merges = still_ongoing_merges
        
        # Process ongoing split events with fixed interval.
        '''
        still_ongoing_splits = []
        for split_info in ongoing_splits:
            if t < split_info["end_step"]:
                elapsed = t - split_info["start"]
                expected_moves = int(elapsed // split_info["interval"]) + 1
                moves_to_do = expected_moves - split_info["nodes_moved"]
                if moves_to_do > 0:
                    num_remaining = len(split_info["remaining_nodes"])
                    num_to_move = min(moves_to_do, num_remaining)
                    #Move the next num_to_move nodes evenly.
                    to_move = split_info["remaining_nodes"][:num_to_move]  # slice: Get the first num_to_move elements
                    split_info["remaining_nodes"] = split_info["remaining_nodes"][num_to_move:]
                    for node in to_move:
                        community_assignment[node] = split_info["new_group"]
                    split_info["nodes_moved"] += num_to_move
                    
                    
                if t < split_info["end_step"] and split_info["remaining_nodes"]:
                    still_ongoing_splits.append(split_info)
        ongoing_splits = still_ongoing_splits
        '''
        
        still_ongoing_splits = []
        for split_info in ongoing_splits:
            if t < split_info["end_step"]:
                elapsed = t - split_info["start"]
                expected_moves = int(elapsed // split_info["interval"]) + 1
                already_moved = split_info["nodes_moved"]
                moves_to_do = expected_moves - already_moved
                if moves_to_do > 0:
                    # Dynamically select live candidates still in old group
                    candidates = [n for n, g in community_assignment.items() if g == split_info["group_to_split"]]
                    remaining_to_move = split_info["split_size"] - already_moved
                    num_to_move = min(moves_to_do, remaining_to_move, len(candidates))
                    
                    # Use sorted order for reproducibility (optional)
                    to_move = sorted(candidates)[:num_to_move]
                    
                    for node in to_move:
                        community_assignment[node] = split_info["new_group"]
                    
                    split_info["nodes_moved"] += num_to_move

                if t < split_info["end_step"] and split_info["nodes_moved"] < split_info["split_size"]:
                    still_ongoing_splits.append(split_info)
        ongoing_splits = still_ongoing_splits

        
        # Check for new split events at this timestep.
        if split_events and t in split_events:
            for (group_to_split, duration) in split_events[t]:
                nodes_in_group = [n for n, g in community_assignment.items() if g == group_to_split]
                if len(nodes_in_group) >= 2:
                    split_size = int(len(nodes_in_group) * split_fraction)
                    split_interval = duration / split_size if split_size > 0 else duration
                    new_group_id = max(active_groups) + 1
                    active_groups.add(new_group_id)
                    end_step = t + duration
                    #nodes_to_move = random.sample(nodes_in_group, split_size)
                    ongoing_splits.append({
                        "start": t,
                        "end_step": end_step,
                        #"remaining_nodes": nodes_to_move,
                        #"old_group": group_to_split,
                        "group_to_split": group_to_split,
                        "new_group": new_group_id,
                        "split_size": split_size,
                        "interval": split_interval,
                        "nodes_moved": 0
                    })
                    
        # Check for new merge events at this timestep.
        if merge_events and t in merge_events:
            for (src, dst, duration) in merge_events[t]:
                nodes_in_src = [n for n, g in community_assignment.items() if g == src]
                if len(nodes_in_src) >= 1:
                    merge_size = int(len(nodes_in_src) * merge_fraction)
                    #for evenly spaced merge, calculate the interval between moves
                    merge_interval = duration / merge_size if merge_size > 0 else duration
                    effective_start = t  
                    effective_end = t + duration
                    nodes_to_move = random.sample(nodes_in_src, merge_size)
                    ongoing_merges.append({
                        "start": effective_start,
                        "end_step": effective_end,
                        "remaining_nodes": nodes_to_move,
                        "src": src,
                        "dst": dst,
                        "interval": merge_interval,
                        "nodes_moved": 0
                    })
        
        # Process normal random evolution (if any).
        num_changes = int(change_rate * num_nodes)
        for _ in range(num_changes):
            node = random.choice(nodes)
            current_group = community_assignment[node]
            other_groups = [g for g in active_groups if g != current_group]
            if other_groups:
                new_group = random.choice(other_groups)
                community_assignment[node] = new_group
    
    return graphs, ground_truth, change_log

# wrapper function for split-merge
def generate_split_merge_data(
    num_nodes=30,
    num_steps=15,
    initial_groups=1,
    split_time=None,
    split_duration=10,
    merge_time=None,
    merge_duration=10,
    split_fraction=0.5,
    merge_fraction=1.0,
    intra_community_strength=0.8,
    inter_community_strength=0.05,
    seed=42
):
    """
    Generate dynamic graphs demonstrating a split event followed by a merge event.
    
    Parameters:
        num_nodes (int): Total number of nodes.
        num_steps (int): Total number of timesteps.
        initial_groups (int): Initial number of groups (for a split demo, set to 1).
        split_time (int): Timestep when the split is triggered
        split_duration (int): Duration over which to perform the split gradually.
        merge_time (int): Timestep when the merge is triggered
        merge_duration (int): Duration over which to perform the merge gradually.
        split_fraction (float): Fraction of nodes in the splitting group to move.
        merge_fraction (float): Fraction of nodes in the source group to merge (default 1.0 means complete merge).
        intra_community_strength (float): Edge probability within communities.
        inter_community_strength (float): Edge probability between communities.
        seed (int): Random seed.
        
    Returns:
        graphs, ground_truth, change_log from generate_dynamic_graphs.
    """
    # Disable additional random evolution for demonstration.
    change_rate = 0
    if split_time is None:
        split_events = {}
    if merge_time is None:
        merge_events = {}
    
    # Setup split events: trigger a split on group 0 at split_time over split_duration timesteps.
    split_events = { split_time: [(0, split_duration)] }
    # Setup merge events: trigger a merge that gradually moves nodes from the new group (assumed to be group 1)
    # into group 0 at merge_time over merge_duration timesteps.
    merge_events = { merge_time: [(1, 0, merge_duration)] }
    
    return generate_dynamic_graphs(
        num_nodes=num_nodes,
        num_steps=num_steps,
        initial_groups=initial_groups,
        change_rate=change_rate,
        intra_community_strength=intra_community_strength,
        inter_community_strength=inter_community_strength,
        split_events=split_events,
        merge_events=merge_events,
        split_fraction=split_fraction,
        merge_fraction=merge_fraction,
        seed=seed
    )



def generate_proportional_transition(
    num_nodes=30,
    num_steps=10,
    initial_state=None,   # z.B. {0: 25, 1: 25, 2: 25, 3: 25} (Prozentwerte, Summe=100)
    final_state=None,     # z.B. {0: 10, 1: 50, 2: 20, 3: 20} 
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    seed=42
):
    """
    Generate dynamic graphs in which the group proportions (as percentages) change
    from an initial state to a final state over a specified number of timesteps (snapshots).

    Input:
        num_nodes (int): Total number of nodes in the network.
        num_steps (int): Number of timesteps (snapshots).
        initial_state (dict): The initial state of the group distribution, e.g., {0: 25, 1: 25, 2: 25, 3: 25}
                            (percentage values, sum = 100).
        final_state (dict): The final state of the group distribution, e.g., {0: 10, 1: 50, 2: 20, 3: 20}.
        intra_community_strength (float): The probability for an edge within the same group.
        inter_community_strength (float): The probability for an edge between different groups.
        seed (int): Seed for the random number generator.

    Returns:
        graphs (dict): A mapping from timestep to a networkx Graph.
        ground_truth (dict): A mapping from timestep to {node: group_id}.
        change_log (dict): A mapping from timestep to a list of nodes that have changed group compared 
                        to the previous timestep.
    """
    random.seed(seed)
    nodes = list(range(num_nodes))
    
    # if not provided, set initial_state and final_state to default values
    if initial_state is None:
        initial_state = {0: 100}
    if final_state is None:
        final_state = initial_state.copy()
    
    # check percentages are valid (sum to 100)
    if abs(sum(initial_state.values()) - 100) > 1e-6 or abs(sum(final_state.values()) - 100) > 1e-6:
        raise ValueError("Initial and final state percentages must sum to 100.")
    
    groups = list(initial_state.keys())
    num_groups = len(groups)
    
    # calculate the number of nodes in each group at the start and end
    # convert percentages to counts
    def percentages_to_counts(state):
        counts = {g: (state[g] / 100) * num_nodes for g in state}
        rounded = {g: int(math.floor(counts[g])) for g in counts}
        remaining = num_nodes - sum(rounded.values())
        # distribution of remaining nodes to the groups with the largest difference between counts and rounded counts
        remainder_groups = sorted(state.keys(), key=lambda g: counts[g] - rounded[g], reverse=True)
        for g in remainder_groups:
            if remaining <= 0:
                break
            rounded[g] += 1
            remaining -= 1
        return rounded
    
    init_counts = percentages_to_counts(initial_state)
    final_counts = percentages_to_counts(final_state)
    
    # init first state: random assignment of nodes to groups
    all_nodes = nodes.copy()
    #random.shuffle(all_nodes) # shuffle the nodes for random assignment
    community_assignment = {}
    index = 0
    for g in sorted(init_counts.keys()):
        count = init_counts[g]   # number of nodes in group g
        for _ in range(count):
            community_assignment[all_nodes[index]] = g
            index += 1
    
    graphs = {}
    ground_truth = {}
    change_log = {}
    
    # gradual change from init_counts to final_counts for each timestep
    # desired node volume for group g at time t (linearly interpolated):
    def desired_count(g, t):
        return round(init_counts[g] + (final_counts[g] - init_counts[g]) * t / (num_steps - 1))
    
    for t in range(num_steps):
        desired = {g: desired_count(g, t) for g in init_counts}
        current = {}
        for g in init_counts:
            current[g] = sum(1 for n in community_assignment.values() if n == g)  # count nodes in group g
        
        # Calculate surplus (current > desired) and deficit (current < desired)
        # Surplus: groups from which nodes should be removed
        excess = {}  # {g: count of excess nodes in group g}, ..}
        #  Deficit: groups to which nodes should be added
        deficit = {}
        for g in init_counts:
            if current[g] > desired[g]:
                excess[g] = current[g] - desired[g]
            elif current[g] < desired[g]:
                deficit[g] = desired[g] - current[g]
        
        # If there are both deficits and surpluses, transfer nodes accordingly
        # randomly transfer nodes to groups with a deficit
        for g_excess in list(excess.keys()):  # g_excess: groups with excess nodes
            if excess[g_excess] <= 0:
                continue
            # list of nodes currently assigned to the surplus groups
            nodes_in_excess = [n for n, grp in community_assignment.items() if grp == g_excess]
            #random.shuffle(nodes_in_excess)
            for g_deficit in list(deficit.keys()):
                if deficit[g_deficit] <= 0:
                    continue
                # how many nodes can be transferred from g_excess to g_deficit
                # excess[g_excess]: number of extra nodes in surplus group g_excess
                # deficit[g_deficit]: number of missing nodes group g_deficit
                transfer = min(excess[g_excess], deficit[g_deficit])
                # move nodes
                for node in nodes_in_excess[:transfer]:
                    community_assignment[node] = g_deficit
                excess[g_excess] -= transfer
                deficit[g_deficit] -= transfer
                # update the list of nodes in the surplus group
                nodes_in_excess = nodes_in_excess[transfer:]
                if excess[g_excess] <= 0:
                    break

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                same_group = community_assignment[i] == community_assignment[j]
                p = intra_community_strength if same_group else inter_community_strength
                if random.random() < p:
                    G.add_edge(i, j, weight=1.0 if same_group else 0.3)
        
        # weak edges between communities for visualization
        community_nodes = {}
        for g in set(community_assignment.values()):
            community_nodes[g] = [node for node, grp in community_assignment.items() if grp == g]
        sorted_groups = sorted(community_nodes.keys())
        for idx in range(len(sorted_groups)-1):
            node_a = random.choice(community_nodes[sorted_groups[idx]])
            node_b = random.choice(community_nodes[sorted_groups[idx+1]])
            G.add_edge(node_a, node_b, weight=0.01)
        
        graphs[t] = G
        ground_truth[t] = community_assignment.copy()
        if t > 0:
            prev_state = ground_truth[t-1]
            changes = [n for n in nodes if community_assignment[n] != prev_state[n]]
            change_log[t] = changes
    
    return graphs, ground_truth, change_log
