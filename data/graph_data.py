import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import json
from pathlib import Path
import os


def create_graphs(data):
    """
    Create NetworkX graphs from the provided data.
    
    Args:
        data (dict): The input data containing nodes, links, and metadata for each timestamp.

    Returns:
        dict: A dictionary of graphs for each timestamp.
    """
    graphs = {}
    
    for timestamp, content in data.items():
        #G = nx.DiGraph()  # Directed graph to account for influence as a factor
        G = nx.Graph()  # Undirected graph

        for node in content["nodes"]:
            if "name" not in node:
                node["name"] = str(node["id"])
            G.add_node(node["id"], name=node["name"], num_exhibitions=node["num_exhibitions"])
        
        # Add edges with weights and influence
        for link in content["links"]:
            G.add_edge(link["source"], link["target"], weight=link["weight"], influence=link["influence"])
        
        graphs[timestamp] = G
    
    return graphs

split_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\split_combined_network_data.json"
with open(split_path, 'r') as f:
    split_data = json.load(f)
    split_graphs = create_graphs(split_data)

split2_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\split2_combined_network_data.json"
with open(split2_path, 'r') as f:
    split2_data = json.load(f)
    split2_graphs = create_graphs(split2_data)
    
merge_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\merge_combined_network_data.json"
with open(merge_path, 'r') as f:
    merge_data = json.load(f)
    merge_graphs = create_graphs(merge_data)
    
join_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\join_combined_network_data.json"
with open(join_path, 'r') as f:
    join_data = json.load(f)
    join_graphs = create_graphs(join_data)
    
join_stable_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\join_stable_combined_network_data.json"
with open(join_stable_path, 'r') as f:
    join_stable_data = json.load(f)
    join_stable_graphs = create_graphs(join_stable_data)

stagnation_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\stagnation_combined_network_data.json"
with open(stagnation_path, 'r') as f:
    stagnation_data = json.load(f)
    stagnation_graphs = create_graphs(stagnation_data)

trend_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\trend_combined_network_data.json"
with open(trend_path, 'r') as f:
    trend_data = json.load(f)
    trend_graphs = create_graphs(trend_data)
    
two_groups_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\two_groups_combined_network_data.json"
with open(two_groups_path, 'r') as f:
    two_groups_data = json.load(f)
    two_groups_graphs = create_graphs(two_groups_data)

three_groups_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\three_groups_combined_network_data.json"
with open(three_groups_path, 'r') as f:
    three_groups_data = json.load(f)
    three_groups_graphs = create_graphs(three_groups_data)

three_groups_new_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\three_groups_new_combined_network_data.json"
with open(three_groups_new_path, 'r') as f:
    three_groups_new_data = json.load(f)
    three_groups_new_graphs = create_graphs(three_groups_new_data)

interpolated_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\interpolated_network_data.json"
with open(interpolated_path, 'r') as f:
    interpolated_data = json.load(f)
    interpolated_graphs = create_graphs(interpolated_data)

extended_split_path = r"C:\Users\fried\Desktop\Uni\Projekt\networkRug\data\coding\data\extended_network_data.json"
with open(extended_split_path, 'r') as f:
    extended_split_data = json.load(f)
    extended_split_graphs = create_graphs(extended_split_data)

print("Graphs created")