import networkx as nx
import itertools
from typing import List, Dict, Callable
from src.models.shipment import Shipment
from src.models.vehicle import Vehicle

def nearest_neighbor_with_2opt(origin: str, 
                             points: List[str], 
                             graph: nx.Graph) -> List[str]:
    """
    Construct a route using nearest neighbor algorithm and improve it with 2-opt.
    
    Args:
        origin: Starting point
        points: List of points to visit
        graph: NetworkX graph with distances
        
    Returns:
        Optimized route order
    """
    if not points:
        return []
        
    # Nearest neighbor construction
    unvisited = points.copy()
    current = origin
    route = [current]
    
    while unvisited:
        try:
            # Find nearest unvisited point
            next_point = min(unvisited,
                           key=lambda x: nx.shortest_path_length(graph, current, x, weight='distance'))
            route.append(next_point)
            unvisited.remove(next_point)
            current = next_point
        except nx.NetworkXNoPath:
            # If no path found, add remaining points in original order
            route.extend(unvisited)
            break
    
    # 2-opt improvement
    improved = True
    while improved:
        improved = False
        for i in range(0, len(route) - 2):
            for j in range(i + 2, len(route)):
                try:
                    if j - i == 1:
                        continue
                    # Calculate current distance
                    current_distance = (
                        nx.shortest_path_length(graph, route[i-1], route[i], weight='distance') +
                        nx.shortest_path_length(graph, route[j-1], route[j], weight='distance')
                    )
                    # Calculate new distance if we reverse the segment
                    new_distance = (
                        nx.shortest_path_length(graph, route[i-1], route[j-1], weight='distance') +
                        nx.shortest_path_length(graph, route[i], route[j], weight='distance')
                    )
                    if new_distance < current_distance:
                        # Reverse the segment
                        route[i:j] = reversed(route[i:j])
                        improved = True
                except nx.NetworkXNoPath:
                    continue
    
    # Remove origin from result as it's added separately in the calling function
    return route[1:]

def consolidate_destinations(solution: Dict[Vehicle, List[Shipment]], 
                           graph: nx.Graph,
                           node_converter: Callable[[str], str]) -> Dict[Vehicle, List[Shipment]]:
    """
    Pre-process step to consolidate shipments going to the same destination.
    """
    # Group all shipments by destination
    dest_groups = {}
    all_shipments = []
    for vehicle, shipments in solution.items():
        all_shipments.extend(shipments)
    
    for shipment in all_shipments:
        dest = node_converter(shipment.delivery_location_id)
        if dest not in dest_groups:
            dest_groups[dest] = []
        dest_groups[dest].append(shipment)
    
    # Sort destinations by total volume and distance from origin
    origin = node_converter("Nanning")
    sorted_dests = []
    for dest, shipments in dest_groups.items():
        try:
            distance = nx.shortest_path_length(graph, origin, dest, weight='distance')
            volume = sum(s.total_cbm for s in shipments)
            sorted_dests.append((dest, shipments, distance, volume))
        except nx.NetworkXNoPath:
            continue
    
    # Sort by volume first, then by distance
    sorted_dests.sort(key=lambda x: (-x[3], x[2]))
    
    # Reset all vehicles
    new_solution = {vehicle: [] for vehicle in solution.keys()}
    
    # Assign shipments to vehicles
    for dest, shipments, _, _ in sorted_dests:
        # Try to keep shipments together
        best_vehicle = None
        min_waste = float('inf')
        
        for vehicle in new_solution:
            current_volume = sum(s.total_cbm for s in new_solution[vehicle])
            current_weight = sum(s.weight for s in new_solution[vehicle])
            group_volume = sum(s.total_cbm for s in shipments)
            group_weight = sum(s.weight for s in shipments)
            
            if (current_volume + group_volume <= vehicle.max_cbm * 1.001 and
                current_weight + group_weight <= vehicle.max_weight * 1.001):
                waste = vehicle.max_cbm - (current_volume + group_volume)
                if waste < min_waste:
                    min_waste = waste
                    best_vehicle = vehicle
        
        if best_vehicle:
            new_solution[best_vehicle].extend(shipments)
        else:
            # If can't keep together, try individual assignments
            for shipment in sorted(shipments, key=lambda s: s.total_cbm, reverse=True):
                assigned = False
                for vehicle in new_solution:
                    if vehicle.can_add_shipment(shipment):
                        new_solution[vehicle].append(shipment)
                        assigned = True
                        break
                if not assigned:
                    print(f"Warning: Could not assign shipment {shipment.shipment_id}")
    
    return new_solution

def optimize_vehicle_route(route: List[str], 
                         graph: nx.Graph,
                         node_converter: Callable[[str], str]) -> List[str]:
    """
    Optimize route using TSP-like approach with strict consolidation.
    """
    if len(route) <= 2:
        return route
    
    # Convert and consolidate destinations
    origin = node_converter(route[0])
    destinations = {}
    
    for dest in route[1:]:
        conv_dest = node_converter(dest)
        if conv_dest not in destinations:
            destinations[conv_dest] = []
        destinations[conv_dest].append(dest)
    
    # Create unique route points
    unique_points = list(destinations.keys())
    
    # Try all possible permutations for small routes
    if len(unique_points) <= 6:
        best_distance = float('inf')
        best_order = unique_points
        
        for perm in itertools.permutations(unique_points):
            full_route = [origin] + list(perm)
            try:
                distance = sum(
                    nx.shortest_path_length(graph, a, b, weight='distance')
                    for a, b in zip(full_route[:-1], full_route[1:])
                )
                if distance < best_distance:
                    best_distance = distance
                    best_order = perm
            except nx.NetworkXNoPath:
                continue
    else:
        # Use nearest neighbor with 2-opt for larger routes
        best_order = nearest_neighbor_with_2opt(origin, unique_points, graph)
    
    # Reconstruct full route
    final_route = [route[0]]
    for dest in best_order:
        final_route.extend(destinations[dest])
    
    return final_route

def group_shipments_by_region(shipments: List[Shipment], 
                            graph: nx.Graph,
                            node_converter: Callable[[str], str]) -> Dict:
    """
    Group shipments by geographical regions based on distance from origin.
    Returns dictionary of region-grouped shipments.
    """
    origin = node_converter("Nanning")
    regions = {}
    
    # Calculate distances from origin to all destinations
    distances = {}
    for shipment in shipments:
        try:
            dest = node_converter(shipment.delivery_location_id)
            distances[shipment] = nx.shortest_path_length(graph, origin, dest, weight='distance')
        except nx.NetworkXNoPath:
            continue
    
    # Define regions based on distance quartiles
    sorted_distances = sorted(distances.values())
    if not sorted_distances:
        return {'default': shipments}
        
    q1 = sorted_distances[len(sorted_distances)//4]
    q2 = sorted_distances[len(sorted_distances)//2]
    q3 = sorted_distances[3*len(sorted_distances)//4]
    
    # Assign shipments to regions
    for shipment, distance in distances.items():
        if distance <= q1:
            region = "near"
        elif distance <= q2:
            region = "mid"
        elif distance <= q3:
            region = "far"
        else:
            region = "very_far"
            
        if region not in regions:
            regions[region] = []
        regions[region].append(shipment)
    
    return regions

def evaluate_route_efficiency(route: List[str], 
                            graph: nx.Graph,
                            node_converter: Callable[[str], str]) -> float:
    """
    Calculate route efficiency score with stronger penalties.
    """
    if len(route) <= 2:
        return 0
    
    # Convert nodes
    converted_route = [node_converter(node) for node in route]
    
    penalty = 0
    
    # Stronger penalty for repeated destinations
    seen_destinations = set()
    for dest in converted_route:
        if dest in seen_destinations and dest != node_converter("Nanning"):
            penalty += 2000  # Doubled the penalty
        seen_destinations.add(dest)
    
    try:
        # Calculate total distance
        total_distance = sum(
            nx.shortest_path_length(graph, a, b, weight='distance')
            for a, b in zip(converted_route[:-1], converted_route[1:])
        )
        
        # Stronger penalty for backtracking
        for i in range(len(converted_route)-2):
            a, b, c = converted_route[i:i+3]
            direct = nx.shortest_path_length(graph, a, c, weight='distance')
            through_b = (nx.shortest_path_length(graph, a, b, weight='distance') +
                        nx.shortest_path_length(graph, b, c, weight='distance'))
            if through_b > direct * 1.2:  # Reduced threshold to 20% longer
                penalty += (through_b - direct) * 5  # Fifth the penalty
    except nx.NetworkXNoPath:
        return float('inf')
    
    return total_distance + penalty

def optimize_solution_routes(solution: Dict[Vehicle, List[Shipment]], 
                           graph: nx.Graph,
                           node_converter: Callable[[str], str]) -> Dict[Vehicle, List[Shipment]]:
    """
    Optimize routes with improved consolidation and balancing.
    """
    # First consolidate destinations
    consolidated = consolidate_destinations(solution, graph, node_converter)
    
    # Then optimize each vehicle's route
    final_solution = {}
    for vehicle, shipments in consolidated.items():
        if not shipments:
            final_solution[vehicle] = shipments
            continue
        
        route = ['Nanning'] + [s.delivery_location_id for s in shipments]
        optimized_route = optimize_vehicle_route(route, graph, node_converter)
        
        # Reorder shipments
        route_order = {loc: idx for idx, loc in enumerate(optimized_route)}
        ordered_shipments = sorted(
            shipments,
            key=lambda s: route_order.get(s.delivery_location_id, float('inf'))
        )
        
        final_solution[vehicle] = ordered_shipments
    
    return final_solution