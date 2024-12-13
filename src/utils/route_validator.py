from typing import List, Dict
import networkx as nx
def validate_route(route: List[str], graph: nx.Graph, node_converter: Dict) -> bool:
    """
    Validate if a route is feasible using consistent node naming.
    
    Parameters:
    - route: List of node IDs/codes in the route
    - graph: NetworkX graph of the network
    - node_converter: Dictionary with 'get_node_name' and 'get_node_code' functions
    
    Returns:
    - bool: True if route is valid, False otherwise
    """
    if not route:
        return True

    current = "NNG"  # Start from Nanning using code
    
    print(f"\nValidating route: {current} -> {' -> '.join(route)}")
    
    for next_stop in route:
        try:
            # Convert both current and next stop to full names for path finding
            current_name = node_converter['get_node_name'](current)
            next_name = node_converter['get_node_name'](next_stop)
            
            print(f"Checking path: {current_name} -> {next_name}")
            
            path = nx.shortest_path(graph, current_name, next_name)
            if not path:
                print(f"No path found between {current_name} and {next_name}")
                return False
                
            current = next_stop
            
        except nx.NetworkXNoPath:
            print(f"No valid path between {current_name} and {next_name}")
            return False
        except KeyError as e:
            print(f"Node conversion error: {e}")
            return False
            
    print("Route is valid!")
    return True