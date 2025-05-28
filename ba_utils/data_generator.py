import networkx as nx
import random
import math
import ast # literal evaluation


# --- Helper Functions ---

def _percentages_to_counts(state_percentages, all_groups, num_nodes):
    """
    Converts state percentages to node counts, distributing remainders.
    
    Args:
        state_percentages (dict): Dictionary mapping group IDs to percentage values (0-100).
        all_groups (list): List of all group IDs to consider.
        num_nodes (int): Total number of nodes to distribute.
        
    Returns:
        dict: Mapping from group ID to node count, with total sum equal to num_nodes.
        
    Note:
        Handles rounding by allocating remaining nodes to groups with the largest fractional parts.
    """
    counts = {g: (state_percentages.get(g, 0) / 100) * num_nodes for g in all_groups}
    rounded = {g: int(math.floor(counts[g])) for g in counts}
    remaining = num_nodes - sum(rounded.values())
    # Distribute remaining nodes to the groups with the largest fractional parts
    remainder_groups = sorted(all_groups, key=lambda g: counts[g] - rounded[g], reverse=True)
    for g in remainder_groups:
        if remaining <= 0:
            break
        rounded[g] += 1
        remaining -= 1
    return rounded

def _create_graph(nodes, community_assignment, intra_p, inter_p, connect_communities=True, weight_intra=1, weight_inter=1):
    """
    Creates a graph based on community assignments and connection probabilities.
    
    Args:
        nodes (list): List of node identifiers.
        community_assignment (dict): Mapping of node to community.
        intra_p (float): Probability of edge within a community.
        inter_p (float): Probability of edge between communities.
        connect_communities (bool): Whether to ensure connectivity between communities.
        weight_intra (int): Weight of intra-community edges.
        weight_inter (int): Weight of inter-community edges.
        
    Returns:
        nx.Graph: Network with community structure.
        
    Note:
        If connect_communities is True, ensures minimal connectivity between adjacent communities
        by adding at least one edge between each pair of consecutive community IDs.
    """
    G = nx.Graph()
    num_nodes = len(nodes)
    G.add_nodes_from(nodes)

    # Add intra- and inter-community edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Check if nodes exist in assignment (might not if graph is changing)
            if i not in community_assignment or j not in community_assignment:
                continue
            same_group = community_assignment[i] == community_assignment[j]
            p = intra_p if same_group else inter_p
            if random.random() < p:
                G.add_edge(i, j, weight=weight_intra if same_group else weight_inter)

    # Optionally force minimal connectivity between communities
    if connect_communities:
        community_nodes = {}
        # Ensure all groups present in the assignment are considered
        present_groups = set(community_assignment.values())
        for group in present_groups:
             community_nodes[group] = [node for node, grp in community_assignment.items() if grp == group]

        sorted_groups = sorted(present_groups)
        for idx in range(len(sorted_groups) - 1):
            group_a = sorted_groups[idx]
            group_b = sorted_groups[idx+1]
            # Ensure both communities have nodes before attempting to add an edge
            if community_nodes.get(group_a) and community_nodes.get(group_b):
                 node_a = random.choice(community_nodes[group_a])
                 node_b = random.choice(community_nodes[group_b])
                 # Add edge only if it doesn't exist, using the inter-community weight
                 if not G.has_edge(node_a, node_b):
                    G.add_edge(node_a, node_b, weight=weight_inter) # Use inter_weight for consistency
    return G


# --- Main Generator Functions ---

def generate_dynamic_graphs(
    num_nodes=30,
    num_steps=30,
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
    
    Args:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of timesteps (snapshots).
        initial_groups (int): Initial number of communities.
        init_mode (str): Initial assignment mode ('random' or 'block').
        change_rate (float): Fraction of nodes that change groups randomly each step (0.0-1.0).
        intra_community_strength (float): Probability of an edge within a community (0.0-1.0).
        inter_community_strength (float): Probability of an edge between communities (0.0-1.0).
        split_events (dict): { timestep: [(group_to_split, duration), ...] }
                             At a given timestep, a partial split is scheduled for the specified group.
        merge_events (dict): { timestep: [(src, dst, duration), ...] }
                             At a given timestep, a merge is scheduled to gradually move nodes 
                             from group src to group dst.
        split_fraction (float): Fraction of nodes to move during a split event (0.0-1.0).
        merge_fraction (float): Fraction of nodes to move during a merge event 
                                (default=1.0 means complete merge).
        seed (int): Random seed for reproducibility.
    
    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since 
                           the previous timestep.
                           
    Note:
        When init_mode='block', nodes are assigned to communities in sequential blocks.
        When init_mode='random', nodes are assigned randomly to communities.
        Split events create a new community by moving a fraction of nodes from an existing community.
        Merge events move nodes from one community to another, potentially eliminating the source community.
    """
    random.seed(seed)
    nodes = list(range(num_nodes))

    community_assignment = {}

    if init_mode == "random":
        # Randomly assign nodes to groups
        for node in nodes:
            community_assignment[node] = random.choice(range(initial_groups))

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
        G = _create_graph(
            nodes, 
            community_assignment, 
            intra_community_strength, 
            inter_community_strength
        )
        
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
                # end of merge event
                remaining_in_src = [n for n, g in community_assignment.items() if g == merge_info["src"]]
                if not remaining_in_src:
                    active_groups.discard(merge_info["src"])
        ongoing_merges = still_ongoing_merges
        
        # Process ongoing split events with fixed interval.
        
        if init_mode == "random":
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
        
        else:
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
                    print(f"[t={t}] New group created from split: {new_group_id}")
                    end_step = t + duration

                    # for random split event: 
                    #nodes_to_move = random.sample(nodes_in_group, split_size)

                    if init_mode == "random":
                        nodes_to_move = random.sample(nodes_in_group, split_size)
                        ongoing_splits.append({
                        "start": t,
                        "end_step": end_step,
                        "remaining_nodes": nodes_to_move,
                        "old_group": group_to_split,
                        "group_to_split": group_to_split,
                        "new_group": new_group_id,
                        "split_size": split_size,
                        "interval": split_interval,
                        "nodes_moved": 0
                    })
                        
                    else:
                        ongoing_splits.append({
                            "start": t,
                            "end_step": end_step,
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
    
    Args:
        num_nodes (int): Total number of nodes.
        num_steps (int): Total number of timesteps.
        initial_groups (int): Initial number of groups (for a split demo, set to 1).
        split_time (int): Timestep when the split is triggered.
        split_duration (int): Duration over which to perform the split gradually.
        merge_time (int): Timestep when the merge is triggered.
        merge_duration (int): Duration over which to perform the merge gradually.
        split_fraction (float): Fraction of nodes in the splitting group to move (0.0-1.0).
        merge_fraction (float): Fraction of nodes in the source group to merge 
                                (default=1.0 means complete merge).
        intra_community_strength (float): Edge probability within communities (0.0-1.0).
        inter_community_strength (float): Edge probability between communities (0.0-1.0).
        seed (int): Random seed.
        
    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since 
                          the previous timestep.
                          
    Note:
        This is a convenience wrapper around generate_dynamic_graphs that creates a scenario
        where one group splits at split_time and then merges back at merge_time.
        If split_time or merge_time is None, the corresponding event is disabled.
    """
    # Disable additional random evolution for demonstration
    change_rate = 0
    
    # Setup split and merge events
    split_events = {} if split_time is None else {split_time: [(0, split_duration)]}
    merge_events = {} if merge_time is None else {merge_time: [(1, 0, merge_duration)]}
    
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
    initial_state=None,
    final_state=None,
    states=None,  # New: list of (timestep, state) pairs
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    seed=42
):
    """
    Generate dynamic graphs where group proportions change over time,
    supporting intermediate states defined either by a list of tuples or a custom string format.
    
    Args:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of timesteps (snapshots).
        initial_state (dict): Group distribution at t=0, e.g., {0: 50, 1: 50} (percentages).
        final_state (dict): Group distribution at t=num_steps-1, e.g., {0: 30, 1: 70}.
        states (list | str | None): Intermediate states. Can be:
            - None: Linear transition from initial_state to final_state.
            - list: List of (timestep, state_dict) tuples.
            - str: Custom format 't1={{...}}; t2={{...}}'
        intra_community_strength (float): Intra-group edge probability (0.0-1.0).
        inter_community_strength (float): Inter-group edge probability (0.0-1.0).
        seed (int): Random seed.

    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since 
                          the previous timestep.
                          
    Note:
        All state dictionaries must sum to 100 (percentages). 
        Specifying intermediate states allows creating complex community evolution patterns
        with multiple phases of transitions.
    """
    random.seed(seed)
    nodes = list(range(num_nodes))

    # --- Handle state inputs and parse custom string format if provided ---
    parsed_states = []
    if states is None:
        if initial_state is None:
            initial_state = {0: 100}
        if final_state is None:
            final_state = initial_state.copy()
        parsed_states = [(0, initial_state), (num_steps - 1, final_state)]
    elif isinstance(states, str):
        if initial_state is None or final_state is None:
            raise ValueError("initial_state and final_state must be provided when using string format for intermediate states.")
        parsed_states.append((0, initial_state))
        try:
            state_definitions = states.strip().split(';')
            for definition in state_definitions:
                if not definition.strip():
                    continue
                time_str, dict_str = definition.split('=', 1)
                timestep = int(time_str.strip())
                state_dict = ast.literal_eval(dict_str.strip())
                if not isinstance(state_dict, dict):
                    raise TypeError("Parsed state is not a dictionary.")
                parsed_states.append((timestep, state_dict))
        except Exception as e:
            raise ValueError(f"Error parsing states string: {e}. Format should be 't1={{...}}; t2={{...}}'.") from e
        parsed_states.append((num_steps - 1, final_state))
        parsed_states = sorted(parsed_states, key=lambda x: x[0]) # Ensure sorted by time
    elif isinstance(states, list):
        # Assume it's already in the correct list-of-tuples format
        parsed_states = sorted(states, key=lambda x: x[0])
    else:
        raise TypeError("states argument must be a list, string, or None.")

    # Validation
    for _, state in parsed_states:
        if abs(sum(state.values()) - 100) > 1e-6:
            raise ValueError("Each state's percentages must sum to 100.")

    all_groups = set()
    for _, state in parsed_states:
        all_groups.update(state.keys())
    all_groups = sorted(list(all_groups)) # Ensure it's a list for sorting in helper

    # Helper: interpolate between two states
    def interpolate_states(states, t):
        # Use the parsed_states list internally
        for idx in range(len(parsed_states)-1):
            t_start, state_start = parsed_states[idx]
            t_end, state_end = parsed_states[idx+1]
            if t_start <= t <= t_end:
                break
        else:
            # If t is beyond the last defined state, return the last state
            return parsed_states[-1][1]

        fraction = (t - t_start) / (t_end - t_start) if t_end > t_start else 0
        interpolated = {}
        for g in all_groups:
            val_start = state_start.get(g, 0)
            val_end = state_end.get(g, 0)
            interpolated[g] = val_start + fraction * (val_end - val_start)
        return interpolated

    # Initialize first assignment
    # Use the parsed_states list internally
    init_state_interpolated = interpolate_states(parsed_states, 0)
    init_counts = _percentages_to_counts(init_state_interpolated, all_groups, num_nodes)

    all_nodes = nodes.copy()
    #random.shuffle(all_nodes) # shuffle the nodes for random assignment
    community_assignment = {}
    index = 0
    for g in all_groups:
        count = init_counts.get(g, 0)
        for _ in range(count):
            community_assignment[all_nodes[index]] = g
            index += 1

    graphs = {}
    ground_truth = {}
    change_log = {}

    for t in range(num_steps):
        # Use the parsed_states list internally
        interpolated_state = interpolate_states(parsed_states, t)
        desired_counts = _percentages_to_counts(interpolated_state, all_groups, num_nodes)

        current_counts = {g: sum(1 for n in community_assignment.values() if n == g) for g in all_groups}
        
        excess = {g: current_counts[g] - desired_counts[g] for g in all_groups if current_counts[g] > desired_counts[g]}
        deficit = {g: desired_counts[g] - current_counts[g] for g in all_groups if current_counts[g] < desired_counts[g]}

        # Transfer nodes
        for g_excess in list(excess.keys()):
            if excess[g_excess] <= 0:
                continue
            nodes_in_excess = [n for n, grp in community_assignment.items() if grp == g_excess]
            for g_deficit in list(deficit.keys()):
                if deficit[g_deficit] <= 0:
                    continue
                transfer = min(excess[g_excess], deficit[g_deficit])
                for node in nodes_in_excess[:transfer]:
                    community_assignment[node] = g_deficit
                nodes_in_excess = nodes_in_excess[transfer:]
                excess[g_excess] -= transfer
                deficit[g_deficit] -= transfer
                if excess[g_excess] <= 0:
                    break

        # Use the helper function to create the graph
        G = _create_graph(
            nodes,
            community_assignment,
            intra_community_strength,
            inter_community_strength,
            connect_communities=True,
            weight_intra=3,
            weight_inter=1
        )

        graphs[t] = G
        ground_truth[t] = community_assignment.copy()
        if t > 0:
            prev_state = ground_truth[t-1]
            changes = [n for n in nodes if community_assignment[n] != prev_state[n]]
            change_log[t] = changes

    return graphs, ground_truth, change_log


# --- Old Versions ---

def generate_proportional_transitionOLD(
    num_nodes=30,
    num_steps=10,
    initial_state=None,   # z.B. {0: 25, 1: 25} (Prozentwerte, Summe=100)
    final_state=None,     
    intra_community_strength=1.0,
    inter_community_strength=0.1,
    seed=42
):
    """
    Generate dynamic graphs in which the group proportions (as percentages) change
    from an initial state to a final state over a specified number of timesteps (snapshots).

    Args:
        num_nodes (int): Total number of nodes in the network.
        num_steps (int): Number of timesteps (snapshots).
        initial_state (dict): The initial state of the group distribution, e.g., {0: 25, 1: 25, 2: 25}
                            (percentage values, sum = 100).
        final_state (dict): The final state of the group distribution, e.g., {0: 10, 1: 50, 2: 20}.
        intra_community_strength (float): The probability for an edge within the same group (0.0-1.0).
        inter_community_strength (float): The probability for an edge between different groups (0.0-1.0).
        seed (int): Seed for the random number generator.

    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since 
                           the previous timestep.
                           
    Note:
        This is an older version of generate_proportional_transition.
        It linearly interpolates between initial_state and final_state.
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

        # Use the helper function to create the graph
        G = _create_graph(
            nodes,
            community_assignment,
            intra_community_strength,
            inter_community_strength,
            connect_communities=True,
            weight_intra=3,
            weight_inter=1
        )
        
        graphs[t] = G
        ground_truth[t] = community_assignment.copy()
        if t > 0:
            prev_state = ground_truth[t-1]
            changes = [n for n in nodes if community_assignment[n] != prev_state[n]]
            change_log[t] = changes
    
    return graphs, ground_truth, change_log

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

    Args:
        num_nodes (int): Total number of nodes.
        num_steps (int): Number of time steps (snapshots).
        num_groups (int): Number of initial communities.
        change_rate (float): Fraction of nodes changing group per step (0.0-1.0).
        intra_community_strength (float): Probability of edge within a community (0.0-1.0).
        inter_community_strength (float): Probability of edge between communities (0.0-1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        graphs (dict): Mapping from timestep to NetworkX graph.
        ground_truth (dict): Mapping from timestep to {node: group_id}.
        change_log (dict): Mapping from timestep to list of node IDs that changed groups since 
                          the previous timestep.
                          
    Note:
        This is an older version of generate_dynamic_graphs with simpler functionality.
        Nodes initially assigned to groups using modulo (node % num_groups).
    """
    random.seed(seed)
    nodes = list(range(num_nodes))
    community_assignment = {node: node % num_groups for node in nodes}

    graphs = {}
    ground_truth = {}
    change_log = {}

    for t in range(num_steps):
        # Use the helper function to create the graph
        G = _create_graph(
            nodes,
            community_assignment,
            intra_community_strength,
            inter_community_strength,
            connect_communities=True,
            weight_intra=3,
            weight_inter=0.01  # Keep the original very weak connecting edge
        )

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
