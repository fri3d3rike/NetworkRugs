import matplotlib.pyplot as plt
from ipywidgets import Dropdown, Checkbox, IntSlider, VBox, HBox
from IPython.display import display
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ba_utils.orderings import *
from ba_utils.visualization import draw_rug_from_graphs

#python -m ba_utils.network_rugs


# --- Drawing Function ---
def draw_networkrug(graphs, color_encoding='closeness_centrality', colormap='turbo', labels=False, pixel_size=6, order="", ax=None, start_nodes=None):
    """
    Draws a single NetworkRug with a specific color encoding and pixel size.
    Adjusts figure size dynamically to support scalability in node/time dimensions.
    calls draw_rug_from_graphs() from visualization.py
    
    Use this for start_nodes
    start_nodes = {timestamp: 15 for timestamp in test.keys()}
    """
    if order == "prio":
        ordering = get_priority_bfs_ordering(graphs, start_nodes)
    if order == "bfs":
        ordering = get_BFS_ordering(graphs, start_nodes, sorting_key='weight')
    if order == "dfs":
        ordering = get_DFS_ordering(graphs, start_nodes)
    if order == "degree":
        ordering = get_centrality_ordering(graphs, centrality_measure="degree")
    else:
        ordering = get_priority_bfs_ordering(graphs, start_nodes)

    fig = draw_rug_from_graphs(
        graphs_data=graphs,
        ordering=ordering,
        color_encoding=color_encoding,
        colormap= colormap,
        labels=labels,
        pixel_size=pixel_size,
        ax=ax
    )

    return fig

# --- Jupyter Notebook Interface ---
def interactive_rug(graphs):
    color_options = ['custom', 'id', 'id2', 'id3', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality']
    color_dropdown = Dropdown(options=color_options, value='custom', description='Color:')
    
    colormap_option = ['turbo', 'gist_rainbow', 'ocean', 'rainbow', 'bwr', 'viridis', 'plasma', 'cividis']
    colormap_dropdown = Dropdown(options=colormap_option, value='turbo', description='Colormap:')
    
    label_toggle = Checkbox(value=False, description='Show Labels')
    pixel_slider = IntSlider(value=6, min=2, max=20, step=1, description='Pixel Size:', continuous_update=False)
    
    ui = VBox([HBox([color_dropdown, colormap_dropdown, label_toggle, pixel_slider])])

    def update_plot(change=None):
        draw_networkrug(graphs, color_encoding=color_dropdown.value, colormap=colormap_dropdown.value, labels=label_toggle.value, pixel_size=pixel_slider.value)

    color_dropdown.observe(update_plot, names='value')
    colormap_dropdown.observe(update_plot, names='value')
    label_toggle.observe(update_plot, names='value')
    pixel_slider.observe(update_plot, names='value')

    display(ui)
    update_plot()

# --- Tkinter GUI Interface ---
def launch_tkinter_ui(graphs):
    def render():
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        draw_networkrug(
            graphs,
            color_encoding=color_var.get(),
            labels=label_var.get(),
            pixel_size=pixel_var.get(),
            ax=ax
        )
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_pixel_label(value):
        pixel_value_label.config(text=f"{int(float(value))} px")

    root = tk.Tk()
    root.title("NetworkRug Visualizer")
    #window.geometry("widthxheight+XPOS+YPOS")
    root.geometry("1600x800")  # Default window size

    # --- Variables ---
    color_var = tk.StringVar(value='closeness_centrality')
    label_var = tk.BooleanVar()
    pixel_var = tk.IntVar(value=6)

    # --- Control Frame ---
    control_frame = ttk.Frame(root)
    control_frame.pack(side='top', pady=10)

    # --- Color Dropdown ---
    ttk.Label(control_frame, text="Color Encoding:").grid(row=0, column=0, sticky='e', padx=5)
    color_box = ttk.Combobox(control_frame, textvariable=color_var, values=[
        'id', 'id2', 'id3', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'eigenvector_centrality'])
    color_box.grid(row=0, column=1, padx=5)

    # --- Label Checkbox ---
    ttk.Checkbutton(control_frame, text="Show Labels", variable=label_var).grid(row=0, column=2, padx=5)

    # --- Pixel Size Slider + Label ---
    ttk.Label(control_frame, text="Pixel Size:").grid(row=1, column=0, sticky='e', padx=5)
    pixel_slider = ttk.Scale(control_frame, variable=pixel_var, from_=2, to=20, orient='horizontal',
                             command=update_pixel_label)
    pixel_slider.grid(row=1, column=1, sticky='we', padx=5)
    pixel_value_label = ttk.Label(control_frame, text=f"{pixel_var.get()} px")
    pixel_value_label.grid(row=1, column=2, padx=5)

    # --- Render Button ---
    ttk.Button(control_frame, text="Render", command=render).grid(row=2, column=0, columnspan=3, pady=10)

    # --- Canvas for Plot ---
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(side='bottom', fill='both', expand=True)

    root.mainloop()
