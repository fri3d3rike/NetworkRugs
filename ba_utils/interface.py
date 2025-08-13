import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os

import ba_utils.orderings as orderings
import ba_utils.color as colorMapper
import ba_utils.data_generator as dg
import ba_utils.visualization as visualization


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Global hover state ---
hover_timer = None
last_hover_coords = (None, None)
hover_preview_canvas = None
hover_delay_ms = 500  # 1 second to trigger preview
click_popups = []

def enable_simple_hover(ax, fig, timestamps, ordering, pixel_size):
    """
    Enables simple hover annotations in the rug plot.
    
    Parameters:
        ax: Matplotlib axes object
        fig: Matplotlib figure object
        timestamps: List of timestamps in the visualization
        ordering: Dictionary with node ordering for each timestamp
        pixel_size: Size of each pixel in the visualization
    """
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    def motion(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        x_idx = int(event.xdata // pixel_size)
        y_idx = int(event.ydata // pixel_size)

        if x_idx >= len(timestamps):
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        timestamp = timestamps[x_idx]
        node_order = ordering.get(timestamp, [])
        if y_idx >= len(node_order):
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        node_id = node_order[y_idx]

        annot.xy = (event.xdata, event.ydata)
        annot.set_text(f"Time: {timestamp}\nNode ID: {node_id}")
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion)

def enable_hover(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40):
    """
    Enables hover previews over a matplotlib Axes inside a Tkinter window.
    When hovering over a pixel, after a delay, a small preview of the graph appears.
    """

    global hover_timer, last_hover_coords, hover_preview_canvas

    canvas = fig.canvas
    parent_widget = canvas.get_tk_widget().master

    def open_hover_preview(timestamp, node_id, x_canvas, y_canvas):
        global hover_preview_canvas

        # Remove existing preview
        if hover_preview_canvas:
            hover_preview_canvas.get_tk_widget().destroy()
            hover_preview_canvas = None

        # Create mini figure
        fig_preview = plt.Figure(figsize=(2, 2))
        ax_preview = fig_preview.add_subplot(111)
        G = graphs_data[timestamp]
        pos = nx.spring_layout(G)

        node_colors = [color_mapping.get((timestamp, n), 'C0') for n in G.nodes()]
        nx.draw(G, pos, with_labels=False, node_size=50, font_size=4, node_color=node_colors, ax=ax_preview)
        ax_preview.axis('off')
        ax_preview.set_title(f"t={timestamp}", fontsize=8)

        hover_preview_canvas = FigureCanvasTkAgg(fig_preview, master=parent_widget)
        hover_preview_canvas.draw()
        widget = hover_preview_canvas.get_tk_widget()
        widget.place(x=x_canvas, y=y_canvas)

    def on_hover(event):
        global hover_timer, last_hover_coords, hover_preview_canvas

        if event.inaxes is None or not hasattr(event, 'xdata') or not hasattr(event, 'ydata'):
            return

        x, y = int(event.xdata), int(event.ydata)
        canvas_x = event.guiEvent.x
        canvas_y = event.guiEvent.y

        def trigger_hover_preview():
            try:
                t_idx = x // pixel_size
                y_idx = y // pixel_size

                timestamps = sorted(graphs_data.keys())
                if t_idx >= len(timestamps):
                    return
                timestamp = timestamps[t_idx]
                node_ordering = ordering[timestamp]
                if y_idx >= len(node_ordering):
                    return
                node_id = node_ordering[y_idx]

                open_hover_preview(timestamp, node_id, canvas_x, canvas_y)

            except Exception as e:
                print("Error during hover preview:", e)

        # Movement cancels preview
        last_x, last_y = last_hover_coords
        if last_x is not None and last_y is not None:
            move_distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
            if move_distance > 5:
                if hover_timer:
                    parent_widget.after_cancel(hover_timer)
                hover_timer = None
                if hover_preview_canvas:
                    widget = hover_preview_canvas.get_tk_widget()
                    if widget.winfo_exists():
                        try:
                            widget.destroy()
                        except tk.TclError:
                            pass
                    hover_preview_canvas = None

        last_hover_coords = (x, y)

        if hover_timer is None:
            hover_timer = parent_widget.after(hover_delay_ms, trigger_hover_preview)

    def on_leave(event):
        global hover_preview_canvas, hover_timer
        # Cancel hover timer if running
        if hover_timer:
            parent_widget.after_cancel(hover_timer)
            hover_timer = None
        # Destroy the preview if it exists
        if hover_preview_canvas:
            widget = hover_preview_canvas.get_tk_widget()
            if widget.winfo_exists():
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
            hover_preview_canvas = None

    # --- Connect ---
    canvas.mpl_connect('motion_notify_event', on_hover)
    canvas.mpl_connect('figure_leave_event', on_leave)

def open_popup(timestamp, node_id, graphs_data, color_mapping):
    fig_graph = plt.Figure(figsize=(4, 4))
    ax_graph = fig_graph.add_subplot(111)
    G = graphs_data[timestamp]
    pos = nx.spring_layout(G)

    node_colors = [color_mapping.get((timestamp, n), 'C0') for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, font_color='white', node_color=node_colors, ax=ax_graph)
    ax_graph.set_title(f"Graph at t={timestamp}")

    popup = tk.Toplevel()
    popup.title(f"Graph at t={timestamp}")
    canvas_graph = FigureCanvasTkAgg(fig_graph, master=popup)
    canvas_graph.draw()
    canvas_graph.get_tk_widget().pack(fill='both', expand=True)
    click_popups.append(popup)

def close_all_popups():
    global click_popups
    for popup in click_popups:
        try:
            popup.destroy()
        except Exception:
            pass
    click_popups = []  

def enable_click(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40):
    canvas = fig.canvas

    def on_click(event):
        if event.inaxes is None or not hasattr(event, 'xdata') or not hasattr(event, 'ydata'):
            return
        x, y = int(event.xdata), int(event.ydata)

        try:
            t_idx = x // pixel_size
            y_idx = y // pixel_size

            timestamps = sorted(graphs_data.keys())
            if t_idx >= len(timestamps):
                return
            timestamp = timestamps[t_idx]
            node_ordering = ordering[timestamp]
            if y_idx >= len(node_ordering):
                return
            node_id = node_ordering[y_idx]

            open_popup(timestamp, node_id, graphs_data, color_mapping)

        except Exception as e:
            print("Error during click:", e)

    canvas.mpl_connect('button_press_event', on_click)

def create_tooltip(widget, text):
    tooltip = None
    def show_tooltip(event):
        nonlocal tooltip
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tooltip, text=text, justify='left',
                            background="#ffffe0", relief='solid', borderwidth=1,
                            padding=(5, 2, 5, 2))
        label.pack()
    def hide_tooltip(event):
        nonlocal tooltip
        if tooltip:
            tooltip.destroy()
            tooltip = None
    widget.bind('<Enter>', show_tooltip)
    widget.bind('<Leave>', hide_tooltip)

# Global graph data
graphs_data = {}

def open_rug_window():
    if not graphs_data:
        print("No graphs generated yet.")
        return

    rug_window = tk.Toplevel()
    rug_window.title("NetworkRug Visualizer")
    rug_window.geometry("1600x900")

    def on_rug_window_close():
        close_all_popups()
        rug_window.destroy()
    rug_window.protocol("WM_DELETE_WINDOW", on_rug_window_close)

    color_options = ['id','degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality']
    ordering_options = ['priority', 'priority_tunable', 'dfs', 'bfs', 'metric_ordering']
    start_node_options = ['degree', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'id']
    start_node_mode_options = ['highest', 'highest_global','lowest', 'lowest_global']
    colormap_options = ['turbo', 'gist_rainbow', 'ocean', 'rainbow', 'bwr', 'viridis', 'plasma', 'cividis', 'cool' ]
    colormap_type_options = ['linear', 'binned']
    metric_options = ['degree', 'closeness', 'betweenness', 'eigenvector']

    color_var = tk.StringVar(value='id')
    order_var = tk.StringVar(value='priority')
    start_node_var = tk.StringVar(value='id')
    start_node_mode_var = tk.StringVar(value='lowest')
    label_var = tk.BooleanVar()
    mapper_type_var = tk.StringVar(value='linear')
    colormap_var = tk.StringVar(value='turbo')
    parameters_var = tk.StringVar(value="w=1,d=1,c=1")
    metric_var = tk.StringVar(value='degree')

    control_frame = ttk.Frame(rug_window)
    control_frame.pack(side='top', fill='x', pady=10)

    color_label = ttk.Label(control_frame, text="Color Encoding")
    color_label.grid(row=0, column=0, padx=5)
    create_tooltip(color_label, "Select the attribute to use for color encoding.")
    ttk.Combobox(control_frame, textvariable=color_var, values=color_options).grid(row=0, column=1, padx=5)

    ttk.Label(control_frame, text="Colormap").grid(row=1, column=0, padx=5)
    ttk.Combobox(control_frame, textvariable=colormap_var, values=colormap_options).grid(row=1, column=1, padx=5)

    # --- Mapper Type Dropdown ---
    color_mapping_label = ttk.Label(control_frame, text="Color Mapping Type:")
    color_mapping_label.grid(row=0, column=2, sticky='e', padx=5)
    create_tooltip(color_mapping_label,
        "Select the type of color mapping to use.\n"
        "Linear: Maps values linearly across the colormap.\n"
        "Binned: Maps values to discrete bins in the colormap."
    )
    ttk.Combobox(control_frame, textvariable=mapper_type_var, values=colormap_type_options).grid(row=0, column=3, padx=5)

    ordering_label = ttk.Label(control_frame, text="Ordering")
    ordering_label.grid(row=0, column=4, padx=5)
    create_tooltip(ordering_label,
        "Select the ordering method for the nodes.\n"
        "Priority: Uses a priority-based BFS ordering.\n"
        "Priority Tunable: Uses a tunable priority-based BFS ordering.\n"
        "DFS: Uses a depth-first search ordering.\n"
        "BFS: Uses a breadth-first search ordering.\n"
        "Metric Ordering: Orders nodes based on a centrality metric."
    )
    ttk.Combobox(control_frame, textvariable=order_var, values=ordering_options).grid(row=0, column=5, padx=5)

    # --- Start Node Dropdown ---
    start_node_label = ttk.Label(control_frame, text="Start Node")
    start_node_label.grid(row=0, column=6, padx=5)
    create_tooltip(start_node_label, "Select the attribute to use for determining the start node.")
    ttk.Combobox(control_frame, textvariable=start_node_var, values=start_node_options).grid(row=0, column=7, padx=5)

    start_node_mode_label = ttk.Label(control_frame, text="Start Node Mode")
    start_node_mode_label.grid(row=1, column=6, padx=5)
    create_tooltip(start_node_mode_label,
        "Select the mode for determining the start node.\n"
        "Highest: Selects the node with the highest value of the selected attribute.\n"
        "Lowest: Selects the node with the lowest value of the selected attribute.\n"
        "Highest Global: Selects the node with the highest value across all timestamps.\n"
        "Lowest Global: Selects the node with the lowest value across all timestamps.")
    ttk.Combobox(control_frame, textvariable=start_node_mode_var, values=start_node_mode_options).grid(row=1, column=7, padx=5)

    labels_button = ttk.Checkbutton(control_frame, text="Show Labels", variable=label_var)
    labels_button.grid(row=1, column=3, padx=5)
    create_tooltip(labels_button, "Check to show node labels in the visualization.")

        # Placeholder for the "Parameters" input field
    parameters_label = None
    parameters_entry = None

    def update_parameters_input(*args):
        nonlocal parameters_label, parameters_entry
        if order_var.get() == "priority_tunable":
            if not parameters_label and not parameters_entry:
                parameters_label = ttk.Label(control_frame, text="Parameters  (?)")
                parameters_label.grid(row=1, column=4, padx=5)
                create_tooltip(
                    parameters_label,
                    "Format: w=<float>,d=<float>,c=<float>\n"
                    "Example: w=0.5,d=1,c=0\n"
                    "Effect:\n"
                    "- w: Higher values prioritize stronger edge weights.\n"
                    "- d: Higher values prioritize high-degree nodes.\n"
                    "- c: Higher values prioritize nodes with more common neighbors. \n"
                    " \n"
                    "d > 0: high-degree nodes → higher priority \n"
                    "d < 0: low-degree nodes → higher priority \n"
                    "|d|: magnitude tunes strength of that effect" 
                )
                parameters_entry = ttk.Entry(control_frame, textvariable=parameters_var)
                parameters_entry.grid(row=1, column=5, padx=5)
        if order_var.get() == "metric_ordering":
            if not parameters_label and not parameters_entry:
                parameters_label = ttk.Label(control_frame, text="Centrality Metric  (?)")
                parameters_label.grid(row=1, column=4, padx=5)
                create_tooltip(
                    parameters_label,
                    "Select the metric to use for ordering the nodes."
                )
                parameters_entry = ttk.Combobox(control_frame, textvariable=metric_var, values=metric_options)
                parameters_entry.grid(row=1, column=5, padx=5)
        else:
            if parameters_label:
                parameters_label.destroy()
                parameters_label = None
            if parameters_entry:
                parameters_entry.destroy()
                parameters_entry = None

    # Bind the update function to the order_var changes
    order_var.trace_add("write", update_parameters_input)


    rug_canvas = ttk.Frame(rug_window)
    rug_canvas.pack(fill='both', expand=True)

    def render_rug():
        fig = plt.Figure()
        ax = fig.add_subplot(111)

        if start_node_var.get() == 'id' and start_node_mode_var.get() == 'lowest':
            start_nodes = None
        elif start_node_var.get() == 'id' and start_node_mode_var.get() == 'highest':
            start_nodes = {timestamp: max(graphs_data[timestamp].nodes()) for timestamp in graphs_data.keys()}
        else:
            start_nodes = orderings.get_start_node(graphs_data, start_node_var.get(), start_node_mode_var.get())

        # --- Select ordering ---
        if order_var.get() == "priority":
            ordering = orderings.get_priority_bfs_ordering(graphs_data, start_nodes)
            print("Using priority BFS ordering")
        elif order_var.get() == "priority_tunable":
            parameters = parameters_var.get()
            if parameters == "":
                ordering = orderings.get_tunable_priority_bfs_ordering(graphs_data, start_nodes)
                print("Using normalized priority BFS with default parameters")
            else:
                try:
                    # Parse the parameters string (e.g., "w=0.5,d=1,c=0")
                    param_dict = dict(item.split('=') for item in parameters.split(','))
                    w = float(param_dict.get('w', 1.0))  # Default to 1.0 if not provided
                    d = float(param_dict.get('d', 1.0))
                    c = float(param_dict.get('c', 1.0))
                    print(f"Using priority tunable BFS ordering with parameters: w={w}, d={d}, c={c}")
                    ordering = orderings.get_tunable_priority_bfs_ordering(graphs_data, start_nodes, w=w, d=d, c=c)
                except Exception as e:
                    print(f"Error parsing parameters: {e}")
                    return
        elif order_var.get() == "bfs":
            ordering = orderings.get_BFS_ordering(graphs_data, start_nodes, sorting_key='weight')
            print("Using BFS ordering")
        elif order_var.get() == "dfs":
            ordering = orderings.get_DFS_ordering(graphs_data, start_nodes)
            print("Using DFS ordering")
        elif order_var.get() == "metric_ordering":
            ordering = orderings.get_centrality_ordering(graphs_data, parameters_entry.get())
            print("Using metric ordering wiht centrality metric:", parameters_entry.get())
        else:
            ordering = orderings.get_priority_bfs_ordering(graphs_data, start_nodes)
            print("Using default priority BFS ordering")

        # --- Draw NetworkRug ---
        fig, color_mapping = visualization.draw_rug_with_color_mapping(
            graphs_data,
            ordering=ordering,
            color_encoding=color_var.get(),
            colormap=colormap_var.get(),
            mapper_type=mapper_type_var.get(),
            labels=label_var.get(),
            pixel_size=40,  # fixed for click mapping
            ax=ax
        )

        # --- Setup Matplotlib Canvas ---
        for widget in rug_canvas.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=rug_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        enable_hover(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40)
        enable_click(ax, fig, graphs_data, ordering, color_mapping, pixel_size=40)
        return fig

    render_rug()
    ttk.Button(control_frame, text="Render Rug", command=render_rug).grid(row=2, column=0, columnspan=4, pady=10)




        # Mapping dictionaries for file naming
    COLOR_MAPPING = {
        'id': 'id_coloring',
        'degree_centrality': 'DC_coloring',
        'closeness_centrality': 'CC_coloring',
        'betweenness_centrality': 'BC_coloring',
        'eigenvector_centrality': 'EC_coloring'
    }

    START_NODE_MAPPING = {
        'degree': 'degree',
        'closeness_centrality': 'CC',
        'betweenness_centrality': 'BC',
        'eigenvector_centrality': 'EC',
        'id': 'id'
    }

    MODE_MAPPING = {
        'highest': 'high',
        'highest_global': 'highest_global',
        'lowest': 'low',
        'lowest_global': 'lowest_global'
    }

    # Assume color_var, colormap_var, order_var, start_node_var, start_node_mode_var are defined elsewhere

    def save_figure(fig):
        if fig:
            metric = color_var.get()
            metric_name = COLOR_MAPPING.get(metric, metric)
            colormap = colormap_var.get()
            order = order_var.get()
            if order == "priority_tunable":
                order = f"priority_tunable_{parameters_var.get()}"
            elif order == "priority":
                order = "priority"
            else:
                order = f"{order}_ordering"
            start_node = start_node_var.get()
            start_mode = start_node_mode_var.get()
            mode_name = MODE_MAPPING.get(start_mode, start_mode)
            node_name = START_NODE_MAPPING.get(start_node, start_node)
            start_str = f",start={mode_name}_{node_name}"

            event_string = "t=50,growth,t=250"

            config_str = f"{metric_name}_{colormap},{order}_{event_string}{start_str}"
            filename = f"{config_str}.png"
            os.makedirs("outputs", exist_ok=True)
            full_path = os.path.join("outputs", filename)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {full_path}")

    def save_figure_old(fig):
        if fig:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_str = (
                f"color_{color_var.get()}__"
                f"order_{order_var.get()}__"
                f"cmap_{colormap_var.get()}__"
                f"num_nodes_{len(next(iter(graphs_data.values())))}__"
                f"time_steps_{len(graphs_data)}__"
            )
            filename = f"networkrug__{config_str}__{timestamp_str}.png"
            os.makedirs("outputs", exist_ok=True)
            full_path = os.path.join("outputs", filename)

            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {full_path}")

    ttk.Button(control_frame, text="Save as PNG", command=lambda: save_figure(render_rug())).grid(row=2, column=4, columnspan=2, pady=10)


def start_gui():
    global graphs_data
    root = tk.Tk()
    root.title("NetworkRug Data Generator")
    root.geometry("820x1600")

    left_frame = ttk.Frame(root)
    for i in range(4):  # assuming you use up to column 3
        left_frame.columnconfigure(i, weight=1)
    left_frame.pack(fill='both', expand=True)

    num_nodes_var = tk.IntVar(value=100)
    num_steps_var = tk.IntVar(value=300)
    intra_strength_var = tk.DoubleVar(value=0.8)
    inter_strength_var = tk.DoubleVar(value=0.1)
    mode_var = tk.StringVar(value='timed')

    ttk.Label(left_frame, text="Num Nodes").grid(row=0, column=0, sticky='e')
    ttk.Entry(left_frame, textvariable=num_nodes_var).grid(row=0, column=1)
    ttk.Label(left_frame, text="Timesteps").grid(row=1, column=0, sticky='e')
    ttk.Entry(left_frame, textvariable=num_steps_var).grid(row=1, column=1)
    ttk.Label(left_frame, text="Intra Community Strength").grid(row=2, column=0, sticky='e')
    ttk.Entry(left_frame, textvariable=intra_strength_var).grid(row=2, column=1)
    ttk.Label(left_frame, text="Inter Community Strength").grid(row=3, column=0, sticky='e')
    ttk.Entry(left_frame, textvariable=inter_strength_var).grid(row=3, column=1)

    ttk.Label(left_frame, text="Generation Mode").grid(row=4, column=0)
    mode_dropdown = ttk.Combobox(left_frame, textvariable=mode_var, values=["timed", "proportioned"])
    mode_dropdown.grid(row=4, column=1)

    dynamic_widgets_frame = ttk.Frame(left_frame)
    dynamic_widgets_frame.grid(row=0, column=2, rowspan=5, sticky='nw', padx=10, pady=5)
    input_state_var = tk.StringVar(value="{0:25,1:25,2:50}")
    final_state_var = tk.StringVar(value="{0:10,1:50,2:40}")
    initial_groups_var = tk.StringVar(value="1")
    init_mode_var = tk.StringVar(value="block")
    split_events_var = tk.StringVar(value="{50: [(0, 100)]}")
    merge_events_var = tk.StringVar(value="{200: [(0, 1, 50)]} ")
    state_input_var = tk.StringVar(value="20={0:100,1:0,2:0}; 40={0:70,1:20,2:10}")

    def update_dynamic_inputs(*args):
        for widget in dynamic_widgets_frame.winfo_children():
            widget.destroy()

        if mode_var.get() == "proportioned":
            ttk.Label(dynamic_widgets_frame, text="Initial State").grid(row=0, column=0, sticky='e')
            initial_entry = ttk.Entry(dynamic_widgets_frame, textvariable=input_state_var, width=30)
            initial_entry.grid(row=0, column=1)
            initial_help = ttk.Label(dynamic_widgets_frame, text="(?)")
            initial_help.grid(row=0, column=2)
            create_tooltip(initial_help, "Format: {group_id: percentage, ...}\nExample: {0:25, 1:25, 2:50} means 25% in group 0, 25% in group 1, and 50% in group 2")

            ttk.Label(dynamic_widgets_frame, text="Intermediate States").grid(row=1, column=0, sticky='e')
            intermediate_entry = ttk.Entry(dynamic_widgets_frame, textvariable=state_input_var, width=30)
            intermediate_entry.grid(row=1, column=1)
            intermediate_help = ttk.Label(dynamic_widgets_frame, text="(?)")
            intermediate_help.grid(row=1, column=2)
            create_tooltip(intermediate_help, "Format: timestep={group_id: percentage, ...}; timestep={...}\nExample: 20={0:100,1:0,2:0}; 40={0:70,1:20,2:10} defines states at t=20 and t=40")

            ttk.Label(dynamic_widgets_frame, text="Final State").grid(row=2, column=0, sticky='e')
            final_entry = ttk.Entry(dynamic_widgets_frame, textvariable=final_state_var, width=30)
            final_entry.grid(row=2, column=1)
            final_help = ttk.Label(dynamic_widgets_frame, text="(?)")
            final_help.grid(row=2, column=2)
            create_tooltip(final_help, "Format: {group_id: percentage, ...}\nExample: {0:10, 1:50, 2:40} means 10% in group 0, 50% in group 1, and 40% in group 2")


        else:
            ttk.Label(dynamic_widgets_frame, text="Initial Groups").grid(row=0, column=0, sticky='e')
            ttk.Entry(dynamic_widgets_frame, textvariable=initial_groups_var).grid(row=0, column=1)
            ttk.Label(dynamic_widgets_frame, text="Init Mode").grid(row=1, column=0, sticky='e')
            ttk.Combobox(dynamic_widgets_frame, textvariable=init_mode_var, values=["block", "random"]).grid(row=1, column=1)
            
            # Split Events
            split_label = ttk.Label(dynamic_widgets_frame, text="Split Events")
            split_label.grid(row=2, column=0, sticky='e')
            split_entry = ttk.Entry(dynamic_widgets_frame, textvariable=split_events_var)
            split_entry.grid(row=2, column=1)
            split_help = ttk.Label(dynamic_widgets_frame, text="(?)")
            split_help.grid(row=2, column=2)
            create_tooltip(split_help, "Format: {timestamp: [(group to split, duration).]..}\nExample: {50: [(0, 100)]} means at t=50, group 0 splits for 100 timesteps")
            
            # Merge Events
            merge_label = ttk.Label(dynamic_widgets_frame, text="Merge Events")
            merge_label.grid(row=3, column=0, sticky='e')
            merge_entry = ttk.Entry(dynamic_widgets_frame, textvariable=merge_events_var)
            merge_entry.grid(row=3, column=1)
            merge_help = ttk.Label(dynamic_widgets_frame, text="(?)")
            merge_help.grid(row=3, column=2)
            create_tooltip(merge_help, "Format: {timestep: [(source, destination, duration).]..}\nExample: {200: [(0, 1, 50)]} means at t=200, group 1 merges into group 2 for 50 timesteps")

    mode_dropdown.bind("<<ComboboxSelected>>", update_dynamic_inputs)
    update_dynamic_inputs()

    left_canvas = ttk.Frame(left_frame)
    left_canvas.grid(row=9, column=0, columnspan=4, sticky='nsew', pady=10)

    def visualize_graph_grid():
        for widget in left_canvas.winfo_children():
            widget.destroy()
        left_canvas.update_idletasks()  # ensure geometry is calculated
        width = left_canvas.winfo_width()
        height = left_canvas.winfo_height()

        # Fallback size if dimensions are 0 (e.g., on first call)
        if width < 100 or height < 100:
            width, height = 800, 600

        fig = plt.Figure(figsize=(width / 100, height / 100))
        selected_keys = sorted(graphs_data.keys())
        step = max(1, len(selected_keys) // 20)
        show_keys = selected_keys[::step]

        cols = 5
        rows = (len(show_keys) + cols - 1) // cols
        fig.clf()

        for i, t in enumerate(show_keys):
            sub_ax = fig.add_subplot(rows, cols, i + 1)
            pos = nx.spring_layout(graphs_data[t])
            nx.draw(graphs_data[t], pos, with_labels=True, node_size=150, font_size=6, font_color='white', ax=sub_ax)
            sub_ax.set_title(f"t={t}", fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=left_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        print("Mini graphs rendered with canvas size:", width, "x", height)

    def generate_and_visualize():
        global graphs_data
        try:
            if mode_var.get() == "proportioned":
                '''graphs_data, _, _ = dg.generate_proportional_transitionOLD(
                    num_nodes=num_nodes_var.get(),
                    num_steps=num_steps_var.get(),
                    initial_state=init_state_var.get(),
                    final_state=final_state_var.get(),
                    intra_community_strength=intra_strength_var.get(),
                    inter_community_strength=inter_strength_var.get()
                )
                '''

                init_state = eval(input_state_var.get())
                final_state = eval(final_state_var.get())
                states_str = state_input_var.get().strip()
                states_input = states_str if states_str else None

                graphs_data, _, _ = dg.generate_proportional_transition(
                    num_nodes=num_nodes_var.get(),
                    num_steps=num_steps_var.get(),
                    initial_state=init_state,
                    final_state=final_state,
                    states=states_input,
                    intra_community_strength=intra_strength_var.get(),
                    inter_community_strength=inter_strength_var.get()
                )


            else:
                split_ev = eval(split_events_var.get() or "{}")
                merge_ev = eval(merge_events_var.get() or "{}")
                graphs_data, _, _ = dg.generate_dynamic_graphs(
                    num_nodes=num_nodes_var.get(),
                    num_steps=num_steps_var.get(),
                    initial_groups=int(initial_groups_var.get()),
                    intra_community_strength=intra_strength_var.get(),
                    inter_community_strength=inter_strength_var.get(),
                    split_events=split_ev,
                    merge_events=merge_ev,
                    init_mode=init_mode_var.get()
                )
            visualize_graph_grid()
            #open_rug_window()
        except Exception as e:
            print("Error generating graphs:", e)

    ttk.Button(left_frame, text="Generate & Show Graphs", command=generate_and_visualize).grid(row=8, column=0, columnspan=2, pady=10)
    ttk.Button(left_frame, text="Open NetworkRug Viewer", command=open_rug_window).grid(row=8, column=2, columnspan=2, pady=1)

    root.mainloop()
