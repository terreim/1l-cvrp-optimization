from typing import List, Dict, Tuple
import random
import networkx as nx
import json
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment
from src.models.cost import Cost
from src.models.network_builder import get_shortest_path, get_fuzzy_path_time

class SolutionGenerator:
    def __init__(self, vehicles: List[Vehicle], shipments: List[Shipment], 
                 network_graph, origin_node: str, cost_calculator: Cost):
        self.vehicles = vehicles
        self.shipments = shipments
        self.graph = network_graph
        self.origin = origin_node
        self.cost_calculator = cost_calculator

        # Create mappings between node names and IDs
        self.name_to_id = {}
        self.id_to_name = {}
        
        try:
            for node_id in self.graph.nodes():
                self.name_to_id[self.graph.nodes[node_id]['name']] = node_id
                self.id_to_name[node_id] = self.graph.nodes[node_id]['name']
        except KeyError:
            pass
        
        # Calculate required vehicles based on total volume
        self.total_cbm = sum(s.total_cbm for s in shipments)
        self.required_vehicles = max(1, int(self.total_cbm / 66.76) + 1)
        
        # Debug info
        print(f"Total CBM: {self.total_cbm}")
        print(f"Required vehicles: {self.required_vehicles}")

    def _get_node_id(self, node_name: str) -> str:
        """Convert node name to node ID."""
        print(self.graph.nodes)
        return self.name_to_id.get(node_name)

    def _get_node_name(self, node_id: str) -> str:
        """Convert node ID to node name."""
        return self.id_to_name.get(node_id)

    def _calculate_route_cost(self, vehicle: Vehicle, shipments: List[Shipment]) -> float:
        """Calculate total cost for a vehicle's route."""
        if not shipments:
            return 0.0

        total_cost = 0.0
        current_node = self.origin
        
        try:
            current_country = self.graph.nodes[current_node]['country']
        except KeyError:
            print(f"Error: Node {current_node} not found in graph")
            return float('inf')
        
        # Create route including origin
        route = [self.origin]
        for shipment in shipments:
            route.append(shipment.delivery_location_id)

        # Calculate costs for each leg
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Verify nodes exist in graph
            if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
                print(f"Error: Node {from_node} or {to_node} not found in graph")
                return float('inf')
            
            try:
                origin_from_node_name = self._get_node_name(from_node)
                origin_to_node_name = self._get_node_name(to_node)
                # Get shortest path and distance
                path = nx.shortest_path(self.graph, origin_from_node_name, origin_to_node_name, weight='distance')
                distance = nx.shortest_path_length(self.graph, origin_from_node_name, origin_to_node_name, weight='distance')
                
                to_country = self.graph.nodes[to_node]['country']
                
                # Calculate daily costs
                is_border_crossing = current_country != to_country
                is_first_day = i == 0
                
                # Get value of goods being delivered at this stop
                delivered_goods_value = sum(
                    s.price for s in shipments 
                    if s.delivery_location_id == to_node
                )
                
                daily_costs = self.cost_calculator.calculate_daily_costs(
                    distance=distance,
                    fuel_efficiency=vehicle.fuel_efficiency,
                    goods_value=delivered_goods_value,
                    from_country=current_country,
                    to_country=to_country,
                    is_border_crossing=is_border_crossing,
                    is_first_day=is_first_day
                )
                
                total_cost += daily_costs['total_cost']
                current_country = to_country
                
            except (nx.NetworkXNoPath, KeyError) as e:
                print(f"Error calculating route: {str(e)}")
                return float('inf')

        return total_cost

    def generate_random_solution(self) -> Tuple[Dict[Vehicle, List[Shipment]], float]:
        """Generate initial solution using random assignment."""
        # Group shipments by destination
        dest_shipments = {}
        for shipment in self.shipments:
            if shipment.delivery_location_id not in dest_shipments:
                dest_shipments[shipment.delivery_location_id] = []
            dest_shipments[shipment.delivery_location_id].append(shipment)
        
        # Get list of destinations and shuffle
        destinations = list(dest_shipments.keys())
        random.shuffle(destinations)
        
        # Initialize solution
        solution = {vehicle: [] for vehicle in self.vehicles[:self.required_vehicles]}
        total_cost = 0.0
        unassigned_shipments = []
        
        # Assign shipments by destination
        for dest in destinations:
            shipments = dest_shipments[dest]
            total_dest_cbm = sum(s.total_cbm for s in shipments)
            total_dest_weight = sum(s.weight for s in shipments)
            
            # Try to find a vehicle that can accommodate all shipments for this destination
            assigned = False
            for vehicle in solution.keys():
                current_cbm = sum(s.total_cbm for s in solution[vehicle])
                current_weight = sum(s.weight for s in solution[vehicle])
                
                if (current_cbm + total_dest_cbm <= vehicle.max_cbm and 
                    current_weight + total_dest_weight <= vehicle.max_weight):
                    for shipment in shipments:
                        vehicle.add_shipment(shipment)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_shipments.extend(shipments)
                print(f"Warning: Could not assign shipments to {dest}")
        
        # Calculate total cost and print debug info
        for vehicle, assigned_shipments in solution.items():
            route_cost = self._calculate_route_cost(vehicle, assigned_shipments)
            total_cost += route_cost
            
            # Debug info
            total_cbm = sum(s.total_cbm for s in assigned_shipments)
            total_weight = sum(s.weight for s in assigned_shipments)
            print(f"\nVehicle {vehicle.vehicle_id} utilization:")
            print(f"CBM: {total_cbm:.2f}/{vehicle.max_cbm:.2f}")
            print(f"Weight: {total_weight:.2f}/{vehicle.max_weight:.2f}")
        
        if unassigned_shipments:
            print(f"\nWarning: {len(unassigned_shipments)} shipments could not be assigned")
        
        # Save solution to file
        solution_data = {
            'solution': {
                vehicle.vehicle_id: {
                    'shipments': [s.shipment_id for s in assigned_shipments],
                    'total_cbm': sum(s.total_cbm for s in assigned_shipments),
                    'total_weight': sum(s.weight for s in assigned_shipments),
                    'cost': self._calculate_route_cost(vehicle, assigned_shipments)
                }
                for vehicle, assigned_shipments in solution.items()
            },
            'total_cost': total_cost,
            'unassigned_shipments': [s.shipment_id for s in unassigned_shipments]
        }
        
        with open('initial_solution.json', 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        return solution, total_cost