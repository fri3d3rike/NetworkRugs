
### Ordering Strategies for 1D Pixel Visualization for Dynamic Networks

This project focuses on visualizing dynamic networks using various graph-based metrics and orderings. The visualizations are created using Python and NetworkX, and the results are displayed using Matplotlib. Different ordering strategies are compared and analyzed, and various color encodings are explored.

You can view the project presentation slides [here](assets/Final_Presentation_BachelorProject_Friederike_Koerte.pdf)

![PDF Preview](assets/slides_preview.jpg)



#### Key Components

##### Utilities
- color.py: Functionality for color mapping.
- neighborhoods.py: Functions related to graph neighborhoods.
- orderings.py: Functions for generating different node orderings.
- visualization.py: Functions for creating the NetworkRug visualizations.

##### Notebooks
- 02_testing.ipynb: Initial testing of graph data and visualizations.
- 03_testing_BFS.ipynb: Testing BFS orderings with different sorting strategies.
- 04_testing_community.ipynb: Testing community detection and common neighbors orderings.
- 05_BFS_adjustment.ipynb: Adjustments to BFS orderings using a priority queue.
- 06_color_encoding.ipynb: Implementing and testing color encoding for visualizations.
- 07_latest_work.ipynb: Latest developments and experiments.

##### Data
- graph_data.py: Functions for loading and creating graph data.
- Various JSON files containing network data, exported from Cytoscape. (Session file also included to have a look at the node-link diagrams)

##### Usage
1. Setup: Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

2. Run Notebooks: Open and run the Jupyter notebooks to see the visualizations and experiments.

3. Visualizations: Use the functions in visualization.py to create custom visualizations of the network data.

##### Example
```python
from data.graph_data import create_graphs
import ba_utils.visualization as visualization
import ba_utils.orderings as orderings

# Load your graph data
graphs_data = create_graphs('path/to/your/data.json')

# Get an ordering
ordering = orderings.get_BFS_ordering(graphs_data)

# Draw the rug plot
fig = visualization.draw_rug_from_graphs(graphs_data, ordering, color_encoding='degree_centrality')
plt.show()
```
