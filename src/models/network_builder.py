# models/network_builder.py
import networkx as nx
import matplotlib.pyplot as plt

def build_graph(edges, nodes):
    """
    Build an undirected graph using NetworkX from edge and node data.
    """
    G = nx.Graph()
    
    # Create mapping dictionaries
    code_to_name = {}
    name_to_code = {}
    
    # First pass - add nodes and build mappings
    for node_id, node in nodes.items():
        G.add_node(
            node_id,
            name=node.name,
            country=node.country,
            type=node.node_type,
            operating_hours=node.operating_hours
        )
        code_to_name[node_id] = node.name
        name_to_code[node.name] = node_id
    
    # Store mappings in graph attributes
    G.graph['code_to_name'] = code_to_name
    G.graph['name_to_code'] = name_to_code
    
    # Add edges using node codes
    for edge_id, edge in edges.items():
        G.add_edge(
            edge.nodes[0],
            edge.nodes[1],
            edge_id=edge_id,
            distance=edge.distance,
            base_time=edge.base_time,
            road_type=edge.road_type,
            fuzzy_travel_time=edge.fuzzy_travel_time,
            country_time_windows=edge.country_time_windows
        )
    return G

def get_shortest_path(G, origin, destination, weight='distance'):
    """
    Find the shortest path between two nodes.
    """
    try:
        path = nx.shortest_path(G, origin, destination, weight=weight)
        path_weight = nx.shortest_path_length(G, origin, destination, weight=weight)
        return path, path_weight
    except nx.NetworkXNoPath:
        print(f"No path found between {origin} and {destination}")
        return None, None
    except nx.NodeNotFound as e:
        print(f"Node not found error: {e}")
        return None, None

def get_fuzzy_path_time(G, path):
    """
    Calculate the total fuzzy travel time for a given path.

    Parameters:
    - G (networkx.Graph): The network graph
    - path (list): List of node IDs representing the path

    Returns:
    - TriangularFuzzyNumber: Total fuzzy travel time for the path
    """
    if not path or len(path) < 2:
        return None
    
    total_time = None
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        if edge_data and 'fuzzy_travel_time' in edge_data:
            if total_time is None:
                total_time = edge_data['fuzzy_travel_time']
            else:
                total_time = total_time + edge_data['fuzzy_travel_time']
    
    return total_time

def find_nearest_neighbors(G, node_id, n=5):
    """
    Find the n nearest neighbors to a given node based on distance.

    Parameters:
    - G (networkx.Graph): The network graph
    - node_id (str): The ID of the node to find neighbors for
    - n (int): Number of nearest neighbors to find

    Returns:
    - list: List of tuples (neighbor_id, distance) sorted by distance
    """
    if node_id not in G:
        return []
    
    distances = []
    for neighbor in G.nodes():
        if neighbor != node_id:
            try:
                dist = nx.shortest_path_length(G, node_id, neighbor, weight='distance')
                distances.append((neighbor, dist))
            except nx.NetworkXNoPath:
                continue
    
    # Sort by distance and return top n
    return sorted(distances, key=lambda x: x[1])[:n]

# def visualize_network(G, with_labels=True, node_size=1500, figsize=(12, 8)):
#     """
#     Visualize the network graph.

#     Parameters:
#     - G (networkx.Graph): The network graph
#     - with_labels (bool): Whether to show node labels
#     - node_size (int): Size of nodes in visualization
#     - figsize (tuple): Figure size (width, height)
#     """
#     plt.figure(figsize=figsize)
    
#     # Set up the layout
#     pos = nx.spring_layout(G, k=1, iterations=50)
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, 
#                           node_color='lightblue',
#                           node_size=node_size)
    
#     # Draw edges with distances as labels
#     nx.draw_networkx_edges(G, pos)
#     edge_labels = nx.get_edge_attributes(G, 'distance')
#     edge_labels = {k: f'{v:.0f}km' for k, v in edge_labels.items()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
#     if with_labels:
#         # Get node labels (using 'name' attribute if available)
#         labels = {}
#         for node in G.nodes():
#             labels[node] = G.nodes[node].get('name', node)
#         nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
#     plt.title("Transportation Network")
#     plt.axis('off')
#     plt.tight_layout()
    
#     # Save the plot
#     plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
#     plt.show()