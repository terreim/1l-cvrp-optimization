from typing import List, Dict, Tuple
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment
import networkx as nx
import copy


def first_fit_decreasing_packing(vehicles: List[Vehicle], 
                               shipments: List[Shipment],
                               network_graph: nx.Graph = None) -> Dict[Vehicle, List[Shipment]]:
    """
    Pack shipments into vehicles using First-Fit Decreasing algorithm with destination grouping.
    """
    # Create deep copies of vehicles to avoid modifying originals
    
    code_to_name = network_graph.graph.get('code_to_name', {})
    name_to_code = network_graph.graph.get('name_to_code', {})

    vehicles = [copy.deepcopy(v) for v in vehicles]
    for vehicle in vehicles:
        vehicle.reset_state()  # Reset vehicle state before packing
    
    solution = {v: [] for v in vehicles}
    
    # Group shipments by destination
    dest_groups = {}
    for shipment in shipments:
        if shipment.delivery_location_id not in dest_groups:
            dest_groups[shipment.delivery_location_id] = []
        dest_groups[shipment.delivery_location_id].append(shipment)
    
    print(f"\nGrouped shipments by {len(dest_groups)} destinations")
    
    # Sort destinations by total volume and distance
    sorted_destinations = []
    origin = "Nanning"  # Use full name instead of code
    
    for dest, group in dest_groups.items():
        total_volume = sum(s.total_cbm for s in group)
        distance = float('inf')
        if network_graph:
            try:
                distance = nx.shortest_path_length(network_graph, origin, code_to_name.get(dest, dest), weight='distance')
            except nx.NetworkXNoPath:
                print(f"Warning: No path found from {origin} to {dest}")
                continue
        sorted_destinations.append((dest, group, total_volume, distance))
    
    # Sort by distance (ascending) and volume (descending)
    sorted_destinations.sort(key=lambda x: (x[3], -x[2]))
    
    print("\nDestination groups sorted by distance and volume:")
    for dest, group, volume, dist in sorted_destinations:
        print(f"{dest}: {volume:.2f} CBM, {dist:.0f} km, {len(group)} shipments")
    
    # Pack shipments by destination groups
    for dest, group, total_volume, _ in sorted_destinations:
        print(f"\nPacking destination {dest} ({total_volume:.2f} CBM):")
        
        # Sort shipments within group by size (descending)
        sorted_shipments = sorted(group, key=lambda x: -x.total_cbm)
        
        # Try to keep destination groups together
        best_vehicle = None
        best_fit = float('inf')
        
        # Check if any vehicle can fit the entire group
        for vehicle in vehicles:
            # Use vehicle's methods to check capacity
            can_fit_group = True
            temp_vehicle = copy.deepcopy(vehicle)
            
            for shipment in sorted_shipments:
                if not temp_vehicle.can_add_shipment(shipment):
                    can_fit_group = False
                    break
                temp_vehicle.add_shipment(shipment)
            
            if can_fit_group:
                remaining = temp_vehicle.get_remaining_capacity()['cbm']
                if remaining < best_fit:
                    best_vehicle = vehicle
                    best_fit = remaining
        
        if best_vehicle:
            # Pack all shipments for this destination
            print(f"Assigned all {dest} shipments to {best_vehicle.vehicle_id}")
            for shipment in sorted_shipments:
                best_vehicle.add_shipment(shipment)
                solution[best_vehicle].append(shipment)
        else:
            # If can't keep together, pack individually
            print(f"Cannot keep {dest} shipments together, trying individual packing")
            for shipment in sorted_shipments:
                packed = False
                for vehicle in vehicles:
                    if vehicle.can_add_shipment(shipment):
                        vehicle.add_shipment(shipment)
                        solution[vehicle].append(shipment)
                        print(f"Assigned {shipment.shipment_id} to {vehicle.vehicle_id}")
                        packed = True
                        break
                if not packed:
                    print(f"WARNING: Could not pack shipment {shipment.shipment_id}")
    
    # Print final allocation
    print("\nFinal allocation:")
    for vehicle, assigned_shipments in solution.items():
        print(f"\n{vehicle.vehicle_id}:")
        weight_util, volume_util = vehicle.get_capacity_utilization()
        print(f"Volume utilization: {volume_util:.1f}%")
        print(f"Weight utilization: {weight_util:.1f}%")
        for s in assigned_shipments:
            print(f"  - {s.shipment_id} to {s.delivery_location_id}")
    
    return solution