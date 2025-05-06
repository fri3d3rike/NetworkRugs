import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ba_utils.orderings as orderings
import ba_utils.color as colorMapper
import os
import networkx as nx

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import ba_utils.interface as interface

# run this command to run
# python -m ba_utils.visualization

#linear_mapper = colormap.LinearHSVColorMapper(colormap="seismic")
#binned_mapper = colorMapper.BinnedPercentileColorMapper( colormap="seismic", bins=10)

def normalize(values_dict, min_val=None, max_val=None):
    if max_val == min_val:
        # case where all values are the same
        return {k: 0.5 for k in values_dict}
    return {k: (v - min_val) / (max_val - min_val) for k, v in values_dict.items()}

def draw_rug_from_graphs(graphs_data, ordering, color_encoding='id', colormap='turbo',
                         labels=False, pixel_size=None, ax=None, 
                         target_width_px=1600, target_height_px=900, dpi=100,
                         mapper_type='linear'):
    """
    Draws a rug plot from a series of graphs data, with dynamic layout that can fit the figure
    into a specified pixel window (e.g., 1600x900).

    Parameters:
    graphs_data (dict): A dictionary where keys are timestamps and values are networkx graphs.
    ordering (dict): A dictionary where keys are timestamps and values are lists of node IDs in the desired order.
    color_encoding (str, optional): Method to encode colors. Options are 'id', 'id2', 'id3', 
        'betweenness_centrality', 'degree_centrality', 'closeness_centrality', 
        'eigenvector_centrality', and 'degree'. Default is 'id'.
    colormap (str, optional): Name of the matplotlib colormap to use for continuous color encodings. Default is 'turbo'.
    labels (bool, optional): Whether to display labels on the plot. Default is False.
    pixel_size (int, optional): If provided, sets the fixed size of each pixel in the plot. 
        If None, the pixel size is automatically calculated to fit within the given target dimensions.
    ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to draw the plot on. 
        If None, a new figure and axes are created.
    target_width_px (int, optional): Desired total width of the figure in pixels. Used only if pixel_size is None.
    target_height_px (int, optional): Desired total height of the figure in pixels. Used only if pixel_size is None.
    dpi (int, optional): Dots per inch for the figure. Used for converting pixel dimensions to inches. Default is 100.
    mapper_type (str, optional): Type of color mapper to use. Options are 'linear' or 'binned'. Default is 'linear'.
    """
    fig = None
    centrality_encodings = ['betweenness_centrality', 'degree_centrality', 'closeness_centrality', 'eigenvector_centrality']
    timestamps = sorted(graphs_data.keys())
    num_artists = len(ordering[next(iter(ordering))])
    print("num_artists", num_artists)
    
    #max_size = 20
    #max_size = max(10, min(30, len(timestamps) * pixel_size / 100))

    #fig_width = min(len(timestamps) * pixel_size / 100, max_size)
    #fig_height = min(num_artists * pixel_size / 100, max_size)
    
    if pixel_size is None: 
        # dynamic pixel size to fit figure
        pixel_size_x = target_width_px // len(timestamps)
        pixel_size_y = target_height_px // num_artists
        pixel_size = min(pixel_size_x, pixel_size_y)  # to keep squares
        
        print("pixelsize updated to:", pixel_size)

    # figure size in inches
    fig_width = (len(timestamps) * pixel_size) / dpi
    fig_height = (num_artists * pixel_size) / dpi

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    if color_encoding in centrality_encodings or color_encoding == 'degree':
        # Get minimum and maximum across all timestamps for global color_encoding
        min_degree_all = float('inf')
        max_degree_all = float('-inf')
        max_centrality_all = float('-inf')
        min_centrality_all = float('inf')
        centralities = {}
        
        for timestamp in timestamps:
            G = graphs_data[timestamp]
            if color_encoding == 'betweenness_centrality':
                centralities[timestamp] = nx.betweenness_centrality(G, normalized=True, weight='weight')
            elif color_encoding == 'degree_centrality':
                centralities[timestamp] = nx.degree_centrality(G)
            elif color_encoding == 'closeness_centrality':
                centralities[timestamp] = nx.closeness_centrality(G, distance='weight')
            elif color_encoding == 'eigenvector_centrality':
                centralities[timestamp] = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-6, weight='weight')
            if color_encoding == 'degree':
                degrees = dict(G.degree())
                centralities[timestamp] = degrees
                max_degree_all = max(max_degree_all, max(degrees.values()))
                min_degree_all = min(min_degree_all, min(degrees.values()))

            if color_encoding != 'degree':
                vals = list(centralities[timestamp].values())
                if vals:
                    min_centrality_all = min(min_centrality_all, min(vals))
                    max_centrality_all = max(max_centrality_all, max(vals))

    all_ids = {id for ts in ordering for id in ordering[ts]}
    min_id, max_id = min(all_ids), max(all_ids)    
 

    # Create the appropriate mapper based on mapper_type
    if mapper_type == 'binned':
        if color_encoding in centrality_encodings:
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap, 
                bins=10,
                min_val=min_centrality_all,
                max_val=max_centrality_all
            )
        elif color_encoding == 'degree':
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap,
                bins=10,
                min_val=min_degree_all,
                max_val=max_degree_all
            )
        else:  # id encoding
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap,
                bins=10,
                min_val=min_id,
                max_val=max_id
            )
    else:  # linear mapper
        if color_encoding in centrality_encodings:
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_centrality_all,
                max_val=max_centrality_all
            )
        elif color_encoding == 'degree':
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_degree_all,
                max_val=max_degree_all
            )
        else:  # id encoding
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_id,
                max_val=max_id
            )

    for t_idx, timestamp in enumerate(timestamps):
        G = graphs_data[timestamp]
        node_order = ordering[timestamp]
        
        for y_idx, artist_id in enumerate(node_order):
            if color_encoding == 'id':
                color = mapper.get_color_by_value(artist_id)
            elif color_encoding == 'id2':
                color = colorMapper.get_color_by_id2(artist_id, num_artists)
            elif color_encoding == 'id3':
                color = colorMapper.get_color_by_id3(artist_id, num_artists)
            elif color_encoding in centrality_encodings:
                color = mapper.get_color_by_value(centralities[timestamp][artist_id])
            elif color_encoding == 'degree':
                color = mapper.get_color_by_value(G.degree(artist_id))
                
            # matplotlib rectangle
            x_start = t_idx * pixel_size
            y_start = y_idx * pixel_size
            rect = plt.Rectangle(
                (x_start, y_start), pixel_size, pixel_size, color=color
            )
            ax.add_patch(rect)
            if labels:
                ax.text(
                    x_start + pixel_size / 2, 
                    y_start + pixel_size / 2, 
                    str(artist_id), 
                    color='white', 
                    ha='center', 
                    va='center',
                    #fontsize=12
                    fontsize=pixel_size * 0.3
                )
    ax.set_xlim(0, len(timestamps) * pixel_size)
    ax.set_ylim(0, num_artists * pixel_size)
    ax.set_xticks([pixel_size * i + pixel_size / 2 for i in range(len(timestamps))])
    ax.set_xticklabels(timestamps, rotation=45)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # invert y-axis to have the first artist at the top
    ax.axis('off')

    # Only return the figure if a new one was created
    if ax is None:
        plt.show()
        
    if fig is None and ax is not None:
        fig = ax.get_figure() 
    
    interface.enable_simple_hover(ax, fig, timestamps, ordering, pixel_size)
    
    return fig if ax is None else None

def draw_rug_with_color_mapping(graphs_data, ordering, color_encoding='id', colormap='turbo',
                                    labels=False, pixel_size=None, ax=None, 
                                    target_width_px=1600, target_height_px=900, dpi=100,
                                    mapper_type='linear'):
    """
    Draws a rug plot like `draw_rug_from_graphs` but ALSO returns a dict with node colors.
    
    Returns:
        fig: matplotlib Figure (if ax was None)
        color_mapping: dict mapping (timestamp, node_id) -> color
    """
    fig = None
    centrality_encodings = ['betweenness_centrality', 'degree_centrality', 'closeness_centrality', 'eigenvector_centrality']
    timestamps = sorted(graphs_data.keys())
    num_artists = len(ordering[next(iter(ordering))])

    color_mapping = {}

    if pixel_size is None: 
        pixel_size_x = target_width_px // len(timestamps)
        pixel_size_y = target_height_px // num_artists
        pixel_size = min(pixel_size_x, pixel_size_y)

    fig_width = (len(timestamps) * pixel_size) / dpi
    fig_height = (num_artists * pixel_size) / dpi

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if color_encoding in centrality_encodings or color_encoding == 'degree':
        min_degree_all = float('inf')
        max_degree_all = float('-inf')
        max_centrality_all = float('-inf')
        min_centrality_all = float('inf')
        centralities = {}

        for timestamp in timestamps:
            G = graphs_data[timestamp]
            if color_encoding == 'betweenness_centrality':
                centralities[timestamp] = nx.betweenness_centrality(G, normalized=True, weight='weight')
            elif color_encoding == 'degree_centrality':
                centralities[timestamp] = nx.degree_centrality(G)
            elif color_encoding == 'closeness_centrality':
                centralities[timestamp] = nx.closeness_centrality(G, distance='weight')
            elif color_encoding == 'eigenvector_centrality':
                centralities[timestamp] = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-6, weight='weight')
            elif color_encoding == 'degree':
                degrees = dict(G.degree())
                centralities[timestamp] = degrees
                max_degree_all = max(max_degree_all, max(degrees.values()))
                min_degree_all = min(min_degree_all, min(degrees.values()))

            if color_encoding != 'degree':
                vals = list(centralities[timestamp].values())
                if vals:
                    min_centrality_all = min(min_centrality_all, min(vals))
                    max_centrality_all = max(max_centrality_all, max(vals))

    all_ids = {id for ts in ordering for id in ordering[ts]}
    min_id, max_id = min(all_ids), max(all_ids)

    if mapper_type == 'binned':
        if color_encoding in centrality_encodings:
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap, bins=10,
                min_val=min_centrality_all, max_val=max_centrality_all
            )
        elif color_encoding == 'degree':
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap, bins=10,
                min_val=min_degree_all, max_val=max_degree_all
            )
        else:  # id encoding
            mapper = colorMapper.BinnedPercentileColorMapper(
                colormap=colormap, bins=10,
                min_val=min_id, max_val=max_id
            )
    else:  # linear mapper
        if color_encoding in centrality_encodings:
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_centrality_all, max_val=max_centrality_all
            )
        elif color_encoding == 'degree':
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_degree_all, max_val=max_degree_all
            )
        else:  # id encoding
            mapper = colorMapper.LinearHSVColorMapper(
                colormap=colormap,
                min_val=min_id, max_val=max_id
            )

    for t_idx, timestamp in enumerate(timestamps):
        G = graphs_data[timestamp]
        node_order = ordering[timestamp]

        for y_idx, artist_id in enumerate(node_order):
            if color_encoding == 'id':
                color = mapper.get_color_by_value(artist_id)
            elif color_encoding == 'id2':
                color = colorMapper.get_color_by_id2(artist_id, num_artists)
            elif color_encoding == 'id3':
                color = colorMapper.get_color_by_id3(artist_id, num_artists)
            elif color_encoding in centrality_encodings:
                color = mapper.get_color_by_value(centralities[timestamp][artist_id])
            elif color_encoding == 'degree':
                color = mapper.get_color_by_value(G.degree(artist_id))

            color_mapping[(timestamp, artist_id)] = color  # <-- Save the color assignment!

            # Draw the rectangle
            x_start = t_idx * pixel_size
            y_start = y_idx * pixel_size
            rect = plt.Rectangle((x_start, y_start), pixel_size, pixel_size, color=color)
            ax.add_patch(rect)

            if labels:
                ax.text(
                    x_start + pixel_size/2, y_start + pixel_size/2,
                    str(artist_id),
                    color='white', ha='center', va='center',
                    fontsize=pixel_size * 0.3
                )

    ax.set_xlim(0, len(timestamps) * pixel_size)
    ax.set_ylim(0, num_artists * pixel_size)
    ax.set_xticks([pixel_size * i + pixel_size / 2 for i in range(len(timestamps))])
    ax.set_xticklabels(timestamps, rotation=45)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.axis('off')

    if ax is None:
        plt.show()

    if fig is None and ax is not None:
        fig = ax.get_figure()

    interface.enable_simple_hover(ax, fig, timestamps, ordering, pixel_size)

    return (fig if fig is not None else ax.get_figure()), color_mapping


# Older versions
def draw_rug_plot_with_ids(data, ordering, pixel_size=40, ax=None):
    """
    Creates a rug plot visualization with artist IDs in each rectangle.
    
    Parameters:
        data: Loaded JSON data.
        ordering: Dictionary with timestamps and artist orderings.
        pixel_size: Size of each pixel.
        ax: Matplotlib axis object. If None, creates a new figure.
    """
    timestamps = sorted(data.keys())
    num_artists = len(ordering[next(iter(ordering))])
    max_size = 20
    fig_width = min(len(timestamps) * pixel_size / 100, max_size)
    fig_height = min(num_artists * pixel_size / 100, max_size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    max_exhibitions = max(
        node['num_exhibitions']
        for t in timestamps
        for node in data[t]['nodes']
    )

    for t_idx, timestamp in enumerate(timestamps):
        time_data = data[timestamp]
        node_order = ordering[timestamp]

        # dict with artist id as key and number of exhibitions as value
        node_exhibitions = {node['id']: node['num_exhibitions'] for node in time_data['nodes']}

        for y_idx, artist_id in enumerate(node_order):
            num_exhibitions = node_exhibitions.get(artist_id, 0)
            color = color.get_color_by_id(artist_id, num_artists)
            
            # matplotlib rectangle
            x_start = t_idx * pixel_size
            y_start = y_idx * pixel_size
            rect = plt.Rectangle(
                (x_start, y_start), pixel_size, pixel_size, color=color
            )
            ax.add_patch(rect)

            # adding id to the rectangle
            ax.text(
                x_start + pixel_size / 2, 
                y_start + pixel_size / 2, 
                str(artist_id), 
                color='white', 
                ha='center', 
                va='center',
                fontsize=15
            )

    ax.set_xlim(0, len(timestamps) * pixel_size)
    ax.set_ylim(0, num_artists * pixel_size)
    ax.set_xticks([pixel_size * i + pixel_size / 2 for i in range(len(timestamps))])
    ax.set_xticklabels(timestamps, rotation=45)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # invert y-axis to have the first artist at the top
    ax.axis('off')

    # Only return the figure if a new one was created
    if ax is None:
        plt.show()
        #return fig

def draw_rug_plot_without_ids(data, ordering, pixel_size=40):
    """
    Creates a rug plot visualization. With artist ids in each rectangle.
    
    Parameters:
        data: Loaded JSON data.
        ordering: Dictionary with timestamps and artist orderings.
        pixel_size: Size of each pixel.
    """
    timestamps = sorted(data.keys())
    num_artists = len(ordering[next(iter(ordering))])
    max_size = 20
    fig_width = min(len(timestamps) * pixel_size / 100, max_size)
    fig_height = min(num_artists * pixel_size / 100, max_size)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    max_exhibitions = max(
        node['num_exhibitions']
        for t in timestamps
        for node in data[t]['nodes']
    )

    #fig, ax = plt.subplots(figsize=(len(timestamps) * 2, num_artists * 2))

    for t_idx, timestamp in enumerate(timestamps):
        time_data = data[timestamp]
        node_order = ordering[timestamp]  

        # dict with artist id as key and number of exhibitions as value
        node_exhibitions  = {node['id']: node['num_exhibitions'] for node in time_data['nodes']}

        for y_idx, artist_id in enumerate(node_order):
            num_exhibitions = node_exhibitions.get(artist_id, 0)
            #print("id", artist_id, "num", num_exhibitions), 
            #color = get_color(num_exhibitions, max_exhibitions)
            color = color.get_color_by_id(artist_id, num_artists)
            
            # mathplotlib rectangle
            x_start = t_idx * pixel_size
            y_start = y_idx * pixel_size
            rect = plt.Rectangle(
                (x_start, y_start), pixel_size, pixel_size, color=color
            )
            ax.add_patch(rect)

    ax.set_xlim(0, len(timestamps) * pixel_size)
    ax.set_ylim(0, num_artists * pixel_size)
    ax.set_xticks([pixel_size * i + pixel_size / 2 for i in range(len(timestamps))])
    ax.set_xticklabels(timestamps, rotation=45)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # invert y-axis to have the first artist at the top
    ax.axis('off')

    plt.show()

    return fig

# generate plots for multiple orderings, given the desired color encoding color_encodings = 
def draw_all_colored(graphs, title="", save=False, color_encoding='degree_centrality', labels=False):
    """
    Visualizes multiple graph orderings using subplots, with optional color encoding and labeling.
    Parameters:
        graphs (list): A list of graph objects to be visualized.
        title (str, optional): A title for the visualization. Defaults to an empty string.
        save (bool, optional): If True, saves the figure to a file. Defaults to False.
        color_encoding (str, optional): The attribute used for node color encoding. 
            Defaults to 'degree_centrality'. 
            Options ["degree_centrality", "degree", "eigenvector_centrality","closeness_centrality", "betweenness_centrality","id2", "id3"]
        labels (bool, optional): If True, displays labels on the nodes. Defaults to False.
    Returns:
        matplotlib.figure.Figure: The generated figure containing the visualizations.
    Notes:
        - The function generates subplots for different graph orderings, including DFS, BFS 
          (with various sorting keys), priority-based BFS, community-based ordering, degree-based 
          ordering, and centrality-based orderings (eigenvector and closeness).
        - If `save` is True, the figure is saved in the `plt_out` directory relative to the script's 
          location, with a filename based on the `title` and `color_encoding`.
        - The function uses helper functions such as `draw_rug_from_graphs` and ordering functions 
          from the `orderings` module to generate the visualizations.
    """
    fig, axes = plt.subplots(1, 8, figsize=(20, 6))
    colormap = 'bwr'
    
    dfs_ordering = orderings.get_DFS_ordering(graphs)
    draw_rug_from_graphs(graphs, dfs_ordering, ax=axes[0], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[0].set_title("DFS")

    bfs_ordering_weight = orderings.get_BFS_ordering(graphs, sorting_key='weight')
    draw_rug_from_graphs(graphs, bfs_ordering_weight, ax=axes[1], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[1].set_title("BFS: Weight")

    bfs_ordering_cn = orderings.get_BFS_ordering(graphs, sorting_key='common_neighbors')
    draw_rug_from_graphs(graphs, bfs_ordering_cn, ax=axes[2], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[2].set_title("BFS: Common Neighbors")
    
    #bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='id')
    #draw_rug_from_graphs(graphs, bfs_ordering, ax=axes[2], color_encoding=color_encoding, colormap=colormap, labels=labels)
    #axes[2].set_title("BFS: ID")
    
    #bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='weight_desID')
    #draw_plot_from_graphs(graphs, bfs_ordering, ax=axes[2])
    #axes[2].set_title("BFS:Weight_descending ID")
    
    priority_ordering = orderings.get_priority_bfs_ordering(graphs)
    draw_rug_from_graphs(graphs, priority_ordering, ax=axes[3], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[3].set_title("Priority")

    neighborhoods_ordering = orderings.get_community_ordering(graphs, "closeness")
    draw_rug_from_graphs(graphs, neighborhoods_ordering, ax=axes[4], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[4].set_title("Community")
    
    degree_ordering = orderings.get_degree_ordering(graphs)
    draw_rug_from_graphs(graphs, degree_ordering, ax=axes[5], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[5].set_title("Degree")
    
    centrality_ordering = orderings.get_centrality_ordering(graphs, centrality_measure='eigenvector')
    draw_rug_from_graphs(graphs, centrality_ordering, ax=axes[6], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[6].set_title("Centrality: Eigenvector")
    
    centrality_ordering2 = orderings.get_centrality_ordering(graphs, centrality_measure='closeness')
    draw_rug_from_graphs(graphs, centrality_ordering2, ax=axes[7], color_encoding=color_encoding, colormap=colormap, labels=labels)
    axes[7].set_title("Centrality: Closeness")

    if title != "":
        modified_title = f"Ordering Comparison for {title}, Color Encoding: {color_encoding}"
        plt.figtext(0.5, -0.05, modified_title, ha="center", fontsize=12)
    plt.tight_layout()
    
    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'plt_out')
        os.makedirs(output_dir, exist_ok=True)
        modified_title = f"{title}_{color_encoding}"
        output_path = os.path.join(output_dir, f"{modified_title.replace(' ', '_').lower()}_summary.png")
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved as {os.path.abspath(output_path)}")
    
    return fig

def draw_all_orderings(graphs, data, title="", save=False):
    fig, axes = plt.subplots(1, 7, figsize=(20, 6))

    bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='weight')
    draw_rug_plot_with_ids(data, bfs_ordering, ax=axes[0])
    axes[0].set_title("BFS: Weight")

    bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='common_neighbors')
    draw_rug_plot_with_ids(data, bfs_ordering, ax=axes[1])
    axes[1].set_title("BFS: Common Neighbors")
    
    bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='id')
    draw_rug_plot_with_ids(data, bfs_ordering, ax=axes[2])
    axes[2].set_title("BFS: ID")
    
    #bfs_ordering = orderings.get_BFS_ordering(graphs, sorting_key='weight_desID')
    #draw_rug_plot_with_ids(data, bfs_ordering, ax=axes[2])
    #axes[2].set_title("BFS:Weight_descending ID")
    
    priority_ordering = orderings.get_priority_bfs_ordering(graphs)
    draw_rug_plot_with_ids(data, priority_ordering, ax=axes[3])
    axes[3].set_title("Priority")

    neighborhoods_ordering = orderings.get_community_ordering(graphs, "closeness")
    draw_rug_plot_with_ids(data, neighborhoods_ordering, ax=axes[4])
    axes[4].set_title("Community")
    
    #dfs_ordering = orderings.get_DFS_ordering(graphs)
    #visualization.draw_rug_plot_with_ids(data, dfs_ordering, ax=axes[4])
    #axes[4].set_title("DFS")
    
    degree_ordering = orderings.get_degree_ordering(graphs)
    draw_rug_plot_with_ids(data, degree_ordering, ax=axes[5])
    axes[5].set_title("Degree")
    
    centrality_ordering = orderings.get_centrality_ordering(graphs, centrality_measure='eigenvector')
    draw_rug_plot_with_ids(data, centrality_ordering, ax=axes[6])
    axes[6].set_title("Centrality: Eigenvector")

    if title != "":
        plt.figtext(0.5, -0.05, title, ha="center", fontsize=12)
    plt.tight_layout()
    
    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'plt_out')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_summary.png")
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved as {os.path.abspath(output_path)}")
    
    return fig

def get_results(all_graphs, names, save=False, labels=False):
    color_encodings = [
        "degree_centrality", "degree", "eigenvector_centrality",
        "closeness_centrality", "betweenness_centrality",
         "id2", "id3"
    ]
    graph_figures = {}
    
    for color_encoding in color_encodings:
        for graphs, name in zip(all_graphs, names):
            fig = draw_all_colored(graphs, name, save=False, color_encoding=color_encoding, labels=labels)
            
            if name not in graph_figures:
                graph_figures[name] = []
            graph_figures[name].append(fig)
            print(f"Generated figure for {name} with color encoding: {color_encoding}")
    
    for graph_name, figures in graph_figures.items():
        print(f"Figures for {graph_name}: {len(figures)} total")
        fig, axes = plt.subplots(nrows=len(color_encodings), ncols=1, figsize=(20, 6 * len(color_encodings)))
        
        for ax, sub_fig, encoding in zip(axes, figures, color_encodings):
            buf = BytesIO()
            sub_fig.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(encoding, fontsize=16)
            buf.close()
            plt.close(sub_fig)  # Close the sub-figure to free up memory
        
        plt.figtext(0.5, 0.02, f"Comparison for {graph_name}", ha='center', fontsize=20)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        if save:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'plt_out')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{graph_name.replace(' ', '_').lower()}_results.png")
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved as {os.path.abspath(output_path)}")
        
        plt.show()
        plt.close(fig)  # Close the main figure to free up memory

def visualize_graphs(graphs, title=None, save=False):
    """
    Visualize NetworkX graphs as node-link diagrams for each timestamp.
    
    Args:
        graphs (dict): A dictionary of NetworkX graphs.
        title (str, optional): Title for the entire figure.
    """
    num_graphs = len(graphs)
    cols = 5
    rows = -(-num_graphs // cols)  # Ceiling division
    
    plt.figure(figsize=(15, rows * 3))
    for i, (timestamp, graph) in enumerate(graphs.items(), 1):
        plt.subplot(rows, cols, i)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=300, font_size=8, font_color='white')
        plt.title(timestamp, fontsize=10)
    
    plt.tight_layout()
    if title:
        plt.figtext(0.5, 0.01, title, ha='center', fontsize=16)
        
    if save:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'plt_out')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_node_link.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved as {os.path.abspath(output_path)}")
    return plt

def visualize_adjacency_matrix(adjacency_matrix, node_order_in_matrix, title="Reordered Adjacency Matrix"):
    """
    Visualize a reordered adjacency matrix with custom node labels.

    Args:
        adjacency_matrix (np.ndarray): Reordered adjacency matrix of the graph.
        ordered_nodes (list): List of node labels representing the new node order.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(adjacency_matrix, cmap='Blues', interpolation='none')
    plt.title(title)
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.xticks(range(len(node_order_in_matrix)), node_order_in_matrix)
    plt.yticks(range(len(node_order_in_matrix)), node_order_in_matrix)
    
    plt.colorbar(label="Connection Strength")
    plt.grid(False)
    plt.show()
