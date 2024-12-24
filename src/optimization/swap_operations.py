from typing import Dict, List
import random
import copy
import networkx as nx
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment
from src.optimization.route_optimizer import nearest_neighbor_with_2opt


def swap_shipments_between_vehicles(solution: Dict[Vehicle, List[Shipment]], 
                                  network_graph: nx.Graph,
                                  node_converter) -> Dict[Vehicle, List[Shipment]]:
    """Main swap function that chooses and executes a swap strategy."""
    print("\nExecuting swap operation...")
    
    # Create deep copy of solution
    new_solution = copy.deepcopy(solution)
    vehicles = list(new_solution.keys())
    
    if len(vehicles) < 2:
        return new_solution

    # Choose random vehicles regardless of their load
    v1, v2 = random.sample(vehicles, 2)
    
    # Try different strategies based on vehicle states
    strategies = []
    
    # If one vehicle is empty, try transfer
    if not new_solution[v1] or not new_solution[v2]:
        strategies.append(('transfer', transfer_shipments))
    else:
        # Both vehicles have shipments
        strategies.extend([
            ('single', single_shipment_swap),
            ('pair', pair_shipment_swap),
            ('destination', destination_based_swap),
            ('proximity', proximity_based_swap),  # New strategy
            ('transfer', transfer_shipments)      # Allow transfers even between active vehicles
        ])
    
    random.shuffle(strategies)
    
    for strategy_name, strategy_func in strategies:
        print(f"Attempting {strategy_name} swap")
        result = strategy_func(new_solution, v1, v2, network_graph)
        if result != new_solution:
            print(f"Successful {strategy_name} swap")
            return result
    
    return new_solution

def transfer_shipments(solution: Dict[Vehicle, List[Shipment]], 
                      source: Vehicle, 
                      target: Vehicle,
                      network_graph: nx.Graph) -> Dict[Vehicle, List[Shipment]]:
    """Transfer shipments between vehicles (can make a vehicle empty)."""
    if not solution[source]:  # If source is empty, swap source and target
        source, target = target, source
        
    if not solution[source]:  # If still no shipments, return unchanged
        return solution
        
    # Try to transfer random number of shipments
    num_shipments = len(solution[source])
    num_to_transfer = random.randint(1, num_shipments)
    shipments_to_move = random.sample(solution[source], num_to_transfer)
    
    # Check capacity constraints
    total_volume = sum(s.total_cbm for s in shipments_to_move)
    total_weight = sum(s.weight for s in shipments_to_move)
    target_current_volume = sum(s.total_cbm for s in solution[target])
    target_current_weight = sum(s.weight for s in solution[target])
    
    if (total_volume + target_current_volume <= target.max_cbm and 
        total_weight + target_current_weight <= target.max_weight):
        # Perform transfer
        for shipment in shipments_to_move:
            solution[source].remove(shipment)
            solution[target].append(shipment)
        return solution
    
    return solution

def single_shipment_swap(solution: Dict[Vehicle, List[Shipment]], 
                        v1: Vehicle, v2: Vehicle,
                        network_graph: nx.Graph) -> Dict[Vehicle, List[Shipment]]:
    """Swap single shipments between vehicles."""
    if not solution[v1] or not solution[v2]:  # Check if either vehicle is empty
        return solution
        
    s1 = random.choice(solution[v1])
    s2 = random.choice(solution[v2])
    
    # Check capacity constraints
    v1_new_load = sum(s.total_cbm for s in solution[v1] if s != s1) + s2.total_cbm
    v2_new_load = sum(s.total_cbm for s in solution[v2] if s != s2) + s1.total_cbm
    
    if (v1_new_load <= v1.max_cbm and v2_new_load <= v2.max_cbm):
        solution[v1].remove(s1)
        solution[v2].remove(s2)
        solution[v1].append(s2)
        solution[v2].append(s1)
        print(f"Swapped {s1.shipment_id} with {s2.shipment_id}")
        return solution
    
    return solution

def pair_shipment_swap(solution: Dict[Vehicle, List[Shipment]], 
                      v1: Vehicle, v2: Vehicle,
                      network_graph: nx.Graph) -> Dict[Vehicle, List[Shipment]]:
    """Swap pairs of shipments between vehicles."""
    if len(solution[v1]) >= 2 and len(solution[v2]) >= 2:
        pair1 = random.sample(solution[v1], 2)
        pair2 = random.sample(solution[v2], 2)
        
        v1_new_load = sum(s.total_cbm for s in solution[v1] if s not in pair1) + sum(s.total_cbm for s in pair2)
        v2_new_load = sum(s.total_cbm for s in solution[v2] if s not in pair2) + sum(s.total_cbm for s in pair1)
        
        if (v1_new_load <= v1.max_cbm and v2_new_load <= v2.max_cbm):
            for s in pair1:
                solution[v1].remove(s)
            for s in pair2:
                solution[v2].remove(s)
            
            solution[v1].extend(pair2)
            solution[v2].extend(pair1)
            return solution
    
    return solution

def destination_based_swap(solution: Dict[Vehicle, List[Shipment]], 
                           v1: Vehicle, 
                           v2: Vehicle,
                           network_graph: nx.Graph) -> Dict[Vehicle, List[Shipment]]:
    """Swap shipments going to the same or nearby destinations, with capacity checks."""
    v1_dests = {s.delivery_location_id for s in solution[v1]}
    v2_dests = {s.delivery_location_id for s in solution[v2]}
    
    # Find any common destinations between v1 and v2
    common_dests = v1_dests.intersection(v2_dests)
    if not common_dests:
        return solution  # No common destinations => no swap
    
    # Pick one random destination from the intersection
    dest = random.choice(list(common_dests))
    
    # Randomly pick one shipment from each vehicle that goes to this destination
    s1 = random.choice([s for s in solution[v1] if s.delivery_location_id == dest])
    s2 = random.choice([s for s in solution[v2] if s.delivery_location_id == dest])
    
    # Calculate what the loads would be AFTER swapping
    # For v1, remove s1 and add s2
    v1_new_cbm = sum(s.total_cbm for s in solution[v1] if s != s1) + s2.total_cbm
    v1_new_weight = sum(s.weight for s in solution[v1] if s != s1) + s2.weight
    
    # For v2, remove s2 and add s1
    v2_new_cbm = sum(s.total_cbm for s in solution[v2] if s != s2) + s1.total_cbm
    v2_new_weight = sum(s.weight for s in solution[v2] if s != s2) + s1.weight
    
    # Check if both vehicles remain within capacity
    if (v1_new_cbm <= v1.max_cbm and 
        v1_new_weight <= v1.max_weight and
        v2_new_cbm <= v2.max_cbm and
        v2_new_weight <= v2.max_weight):
        
        # All good: perform the swap
        solution[v1].remove(s1)
        solution[v2].remove(s2)
        solution[v1].append(s2)
        solution[v2].append(s1)
        
    # Return the (possibly) updated solution
    return solution


def proximity_based_swap(solution: Dict[Vehicle, List[Shipment]], 
                        v1: Vehicle, 
                        v2: Vehicle,
                        network_graph: nx.Graph) -> Dict[Vehicle, List[Shipment]]:
    """Swap shipments going to nearby destinations."""
    if not solution[v1] or not solution[v2]:
        return solution
        
    # Get all pairs of shipments
    for s1 in solution[v1]:
        for s2 in solution[v2]:
            # Check if destinations are close (using network distance)
            try:
                distance = nx.shortest_path_length(
                    network_graph, 
                    s1.delivery_location_id, 
                    s2.delivery_location_id, 
                    weight='distance'
                )
                
                # If destinations are close (threshold can be adjusted)
                if distance < 1000:  # Example threshold of 1000km
                    # Check capacity constraints
                    v1_new_load = sum(s.total_cbm for s in solution[v1] if s != s1) + s2.total_cbm
                    v2_new_load = sum(s.total_cbm for s in solution[v2] if s != s2) + s1.total_cbm
                    
                    if (v1_new_load <= v1.max_cbm and v2_new_load <= v2.max_cbm):
                        # Perform swap
                        solution[v1].remove(s1)
                        solution[v2].remove(s2)
                        solution[v1].append(s2)
                        solution[v2].append(s1)
                        return solution
            except nx.NetworkXNoPath:
                continue
    
    return solution

def get_route_from_solution(shipments: List[Shipment]) -> List[str]:
    """Extract ordered route from shipments."""
    route = []
    for shipment in shipments:
        if shipment.delivery_location_id not in route:
            route.append(shipment.delivery_location_id)
    return route

def optimize_solution_routes(solution: Dict[Vehicle, List[Shipment]], 
                           graph: nx.Graph,
                           node_converter) -> Dict[Vehicle, List[Shipment]]:
    """Optimize routes while preserving vehicle assignments."""
    optimized = {}
    
    for vehicle, shipments in solution.items():
        if not shipments:
            optimized[vehicle] = []
            continue
            
        # Create a copy of the shipments list
        vehicle_shipments = shipments.copy()
        
        # Get route
        route = ['Nanning'] + [s.delivery_location_id for s in vehicle_shipments]
        optimized_route = nearest_neighbor_with_2opt(route[0], route[1:], graph)
        
        # Reorder shipments according to optimized route
        route_order = {loc: idx for idx, loc in enumerate(['Nanning'] + optimized_route)}
        ordered_shipments = sorted(
            vehicle_shipments,
            key=lambda s: route_order.get(s.delivery_location_id, float('inf'))
        )
        
        optimized[vehicle] = ordered_shipments
    
    return optimized