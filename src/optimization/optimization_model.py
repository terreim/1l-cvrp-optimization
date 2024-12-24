from typing import Dict, List, Tuple
import copy
import random
import networkx as nx
import json
import csv
from src.fuzzy.fuzzy_number import TriangularFuzzyNumber
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment
from src.models.cost import Cost
from src.optimization.acceptance_probability import acceptance_probability, fuzzy_dominance
from src.optimization.swap_operations import swap_shipments_between_vehicles
from src.optimization.packing import first_fit_decreasing_packing
from src.utils.solution_validator import SolutionValidator
from src.utils.route_validator import validate_route
from src.optimization.route_optimizer import (
    nearest_neighbor_with_2opt,
    optimize_solution_routes,
    evaluate_route_efficiency,
    group_shipments_by_region,
    consolidate_destinations
)

class SimulatedAnnealingOptimizer:
    def __init__(self, vehicles: List[Vehicle], shipments: List[Shipment],
             graph: nx.Graph, cost_calculator: Cost, validator: SolutionValidator,
             initial_temperature: float = 100.0, cooling_rate: float = 0.95,
             termination_temperature: float = 1.0, max_iterations: int = 1000):
        """Initialize the Simulated Annealing Optimizer."""
        self.vehicles = vehicles
        self.shipments = shipments
        self.cost_calculator = cost_calculator
        self.validator = validator
        self.graph = graph
        
        # SA parameters
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.termination_temperature = termination_temperature
        self.max_iterations = max_iterations
        
        # Store mappings at initialization
        self.code_to_name = graph.graph.get('code_to_name', {})
        self.name_to_code = graph.graph.get('name_to_code', {})
        self.node_converter = {
            'get_node_id': self.get_node_code,
            'get_node_name': self.get_node_name
        }

        self.current_solution = None
        self.best_solution = None
        self.best_cost = None
        self.best_validation = None
        
        # Generate initial solution
        print("\nGenerating initial solution...")
        self.generate_initial_solution()

    def get_node_name(self, code: str) -> str:
        """Convert node code to full name."""
        return self.code_to_name.get(code, code)

    def get_node_code(self, name: str) -> str:
        """Convert full name to node code."""
        return self.name_to_code.get(name, name)

    def print_solution_routes(self, solution: Dict[Vehicle, List[Shipment]], title: str = "Current Solution"):
        """Print detailed route information with consistent node naming."""
        print(f"\n=== {title} ===")
        total_cbm = 0
        total_shipments = 0
        
        for vehicle, shipments in solution.items():
            if not shipments:
                continue
                
            total_cbm += sum(s.total_cbm for s in shipments)
            total_shipments += len(shipments)
            
            print(f"\nVehicle {vehicle.vehicle_id}:")
            print(f"Capacity: {vehicle.current_load_length:.2f}/{vehicle.max_cbm:.2f} CBM")
            print(f"Weight: {vehicle.current_load_weight:.2f}/{vehicle.max_weight:.2f} kg")
            print("Route:")
            print(f"START: {self.get_node_name('NNG')}")
            
            # Group shipments by destination
            dest_shipments = {}
            for s in shipments:
                dest_name = self.get_node_name(s.delivery_location_id)
                if dest_name not in dest_shipments:
                    dest_shipments[dest_name] = []
                dest_shipments[dest_name].append(s)
            
            # Print each stop with its shipments
            for dest_name, dest_shipments_list in dest_shipments.items():
                print(f"  └─> {dest_name}:")
                total_dest_cbm = sum(s.total_cbm for s in dest_shipments_list)
                for s in dest_shipments_list:
                    print(f"      - {s.shipment_id}: {s.total_cbm:.2f} CBM")
                print(f"      Total: {total_dest_cbm:.2f} CBM")
        
        print(f"\nTotal Solution Statistics:")
        print(f"Total Shipments: {total_shipments}")
        print(f"Total Volume: {total_cbm:.2f} CBM")

    def is_solution_valid(self, solution: Dict[Vehicle, List[Shipment]]) -> bool:
        """Validate solution with consistent node naming."""
        print("\nValidating complete solution:")
        
        for vehicle, shipments in solution.items():
            print(f"\nValidating vehicle {vehicle.vehicle_id}:")
            
            # Capacity checks remain the same...
            
            # Route validation with consistent naming
            route = []
            current = self.get_node_name('NNG')  # Starting point
            
            # Build route using full names
            for shipment in shipments:
                dest_name = self.get_node_name(shipment.delivery_location_id)
                if dest_name not in route:
                    route.append(dest_name)
                    try:
                        path = nx.shortest_path(self.graph, current, dest_name)
                        print(f"Valid path: {' -> '.join(path)}")
                        current = dest_name
                    except nx.NetworkXNoPath:
                        print(f"Invalid: No path from {current} to {dest_name}")
                        return False
        
        print("All routes are valid!")
        return True
    
    def is_route_valid(route: List[str], network_graph: nx.Graph, node_converter) -> bool:
        """Check if a route is valid in terms of connectivity."""
        return validate_route(route, network_graph, node_converter)

    def generate_initial_solution(self) -> Dict[Vehicle, List[Shipment]]:
        """Generate initial solution using First-Fit Decreasing with region grouping."""
        # Reset all vehicles
        for vehicle in self.vehicles:
            vehicle.reset_state()
        
        # Group shipments by region
        region_groups = group_shipments_by_region(
            self.shipments, 
            self.graph,
            self.node_converter['get_node_name']  # Pass the converter function
        )

        # Pack shipments region by region
        initial_solution = first_fit_decreasing_packing(
            self.vehicles, 
            self.shipments,
            self.graph
        )

        # Consolidate destinations and optimize routes
        consolidated = consolidate_destinations(
            initial_solution, 
            self.graph,
            self.node_converter['get_node_name']
        )
        
        # Optimize routes
        optimized_solution = optimize_solution_routes(
            initial_solution, 
            self.graph,
            self.node_converter['get_node_name']  # Pass the converter function
        )

        solution_data = {
        'solution': {
            vehicle.vehicle_id: {
                'shipments': [s.shipment_id for s in shipments],
                'total_cbm': sum(s.total_cbm for s in shipments),
                'total_weight': sum(s.weight for s in shipments),
                'route': ['Nanning'] + [s.delivery_location_id for s in shipments]
            }
            for vehicle, shipments in optimized_solution.items()
        }
        }
    
        with open('initial_solution.json', 'w') as f:
            json.dump(solution_data, f, indent=2)

        initial_costs = {}
        for vehicle, shipments in optimized_solution.items():
            if shipments:
                route = ['Nanning'] + [s.delivery_location_id for s in shipments]
                route_cost = self.cost_calculator.calculate_route_cost(route, self.graph)
                initial_costs[vehicle.vehicle_id] = {
                    'total_cost': route_cost.defuzzify(),
                    'route': route,
                    'volume_utilization': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
                    'weight_utilization': sum(s.weight for s in shipments) / vehicle.max_weight * 100
                }
        
        with open('initial_costs.json', 'w') as f:
            json.dump(initial_costs, f, indent=2)

        metrics_data = {
        vehicle.vehicle_id: {
            'Total_Distance_km': sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance')
                                for i in range(len(route)-1)),
            'Travel_Time_min': sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='time')
                                for i in range(len(route)-1)),
            'Stops': len(set(s.delivery_location_id for s in shipments)),
            'Fuel_Cost': self.cost_calculator.calculate_fuel_cost(
                sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance') 
                    for i in range(len(route)-1)),
                vehicle.fuel_efficiency
            ),
            'Total_Cost': sum(cost_components),
            'Real_CBM': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
            'Optimized_CBM': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
            'Improvement': 0  # Will be calculated after optimization
        }
        for vehicle, shipments in optimized_solution.items()
        if shipments  # Only include vehicles with assignments
        for route in [['Nanning'] + [s.delivery_location_id for s in shipments]]
        for cost_components in [[
            self.cost_calculator.calculate_fuel_cost(
                sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance') 
                    for i in range(len(route)-1)),
                vehicle.fuel_efficiency
            ),
            ]]
        }
        
        # Save to CSV
        with open('initial_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle_ID', 'Total_Distance_km', 'Travel_Time_min', 'Stops', 
                            'Fuel_Cost', 'Penalties', 'Driver_Costs', 'Customs_Fees', 
                            'Total_Cost', 'Real_CBM', 'Optimized_CBM', 'Improvement'])
            for vehicle_id, metrics in metrics_data.items():
                writer.writerow([vehicle_id] + list(metrics.values()))

        self.current_solution = optimized_solution
        return optimized_solution

    def generate_neighbor(self, current_solution: Dict[Vehicle, List[Shipment]]) -> Dict[Vehicle, List[Shipment]]:
        """Generate a neighbor solution using multiple possible moves"""
        # Create a deep copy of the current solution
        neighbor = copy.deepcopy(current_solution)
        
        # Select a random move type with weights
        move_type = random.choices(
            ['swap_operation', 'transfer_shipment', 'reverse_subroute'],
            weights=[0.5, 0.3, 0.2]
        )[0]
        
        print(f"\nAttempting move type: {move_type}")
        
        if move_type == 'swap_operation':
            neighbor = swap_shipments_between_vehicles(
                neighbor,
                self.graph,
                self.node_converter['get_node_name']
            )
            
        elif move_type == 'transfer_shipment':
            active_vehicles = [v for v in neighbor.keys() if neighbor[v]]
            possible_targets = [v for v in neighbor.keys()]
            
            if active_vehicles:
                source = random.choice(active_vehicles)
                targets = [v for v in possible_targets if v != source]
                
                if targets and neighbor[source]:  # Check if source has shipments
                    target = random.choice(targets)
                    shipment = random.choice(neighbor[source])
                    
                    # Check if target vehicle can accommodate the shipment
                    target_load = sum(s.total_cbm for s in neighbor[target]) + shipment.total_cbm
                    if target_load <= target.max_cbm:
                        print(f"Transferring shipment {shipment.shipment_id} from {source.vehicle_id} to {target.vehicle_id}")
                        neighbor[source].remove(shipment)
                        neighbor[target].append(shipment)
        
        elif move_type == 'reverse_subroute':
            active_vehicles = [v for v in neighbor.keys() if len(neighbor[v]) >= 3]
            if active_vehicles:
                vehicle = random.choice(active_vehicles)
                shipments = neighbor[vehicle]
                
                if len(shipments) >= 3:
                    start = random.randint(0, len(shipments) - 3)
                    end = random.randint(start + 2, len(shipments))
                    print(f"Reversing subroute in {vehicle.vehicle_id} from position {start} to {end}")
                    neighbor[vehicle][start:end] = list(reversed(neighbor[vehicle][start:end]))
        
        # Print solution state before optimization
        print("\nSolution state after move (before optimization):")
        for vehicle in neighbor:
            print(f"\n{vehicle.vehicle_id}:")
            print(f"Shipments: {[s.shipment_id for s in neighbor[vehicle]]}")
            print(f"Total volume: {sum(s.total_cbm for s in neighbor[vehicle]):.2f}/{vehicle.max_cbm:.2f}")
        
        # Only optimize routes within each vehicle, don't change assignments
        for vehicle in neighbor:
            if neighbor[vehicle]:
                route = ['Nanning'] + [s.delivery_location_id for s in neighbor[vehicle]]
                optimized_route = nearest_neighbor_with_2opt(route[0], route[1:], self.graph)
                
                # Reorder shipments according to optimized route
                route_order = {loc: idx for idx, loc in enumerate(['Nanning'] + optimized_route)}
                neighbor[vehicle].sort(
                    key=lambda s: route_order.get(s.delivery_location_id, float('inf'))
                )
        
        # Print final solution state
        print("\nFinal solution state after optimization:")
        for vehicle in neighbor:
            print(f"\n{vehicle.vehicle_id}:")
            print(f"Shipments: {[s.shipment_id for s in neighbor[vehicle]]}")
            print(f"Total volume: {sum(s.total_cbm for s in neighbor[vehicle]):.2f}/{vehicle.max_cbm:.2f}")
        
        return neighbor

    def optimize(self):
        """Run the simulated annealing optimization."""
        iteration = 0
        current_solution = self.current_solution
        current_cost, current_validation = self.evaluate_solution(current_solution)
        
        # Initialize best solution tracking
        self.best_solution = copy.deepcopy(current_solution)
        self.best_cost = current_cost
        self.best_validation = current_validation
        
        # Statistics tracking
        accepted_solutions = 0
        rejected_solutions = 0
        improvements = 0
        
        while self.temperature > self.termination_temperature and iteration < self.max_iterations:
            if iteration % 10 == 0:
                print(f"\n{'='*50}")
                print(f"Progress: {(iteration/self.max_iterations)*100:.1f}% (Iteration {iteration}/{self.max_iterations})")
                print(f"Temperature: {self.temperature:.2f}")
                print("\nCosts:")
                print(f"Current: {current_cost.defuzzify():.2f}")
                print(f"Best: {self.best_cost.defuzzify():.2f}")
                
                if accepted_solutions + rejected_solutions > 0:
                    acceptance_rate = (accepted_solutions / (accepted_solutions + rejected_solutions)) * 100
                    print("\nStatistics:")
                    print(f"Acceptance rate: {acceptance_rate:.1f}%")
                    print(f"Accepted solutions: {accepted_solutions}")
                    print(f"Rejected solutions: {rejected_solutions}")
                    print(f"Improvements found: {improvements}")
            
            # Generate neighbor solution
            new_solution = swap_shipments_between_vehicles(
                current_solution, 
                self.graph,
                self.node_converter
            )
            
            # Evaluate new solution
            new_cost, new_validation = self.evaluate_solution(new_solution)
            
            # Check if new solution is valid
            if not new_validation.get('is_valid', False):
                rejected_solutions += 1
                iteration += 1
                continue
            
            # Calculate acceptance probability
            cost_diff = new_cost.defuzzify() - current_cost.defuzzify()
            ap = self.acceptance_probability(cost_diff)
            rand_val = random.random()

            if iteration % 10 == 0:
                print(f"\nNeighbor solution {'ACCEPTED' if rand_val < ap else 'REJECTED'}")
                print(f"Cost difference: {'+' if cost_diff >= 0 else ''}{(cost_diff/current_cost.defuzzify())*100:.2f}%")
            
            # Accept or reject new solution
            if rand_val < ap:
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                current_validation = new_validation
                accepted_solutions += 1
                
                # Update best solution if better
                if new_cost.defuzzify() < self.best_cost.defuzzify():
                    print(f"\nNew best solution found: {new_cost.defuzzify():.2f}")
                    self.best_solution = copy.deepcopy(new_solution)
                    self.best_cost = new_cost
                    self.best_validation = new_validation
                    improvements += 1
            else:
                rejected_solutions += 1
            
            # Cool down
            self.temperature *= self.cooling_rate
            iteration += 1
        
        print(f"\nOptimization completed after {iteration} iterations")
        print(f"Best solution cost: {self.best_cost.defuzzify():.2f}")
        
        metrics = self.calculate_solution_metrics(self.best_solution)
    
        # Combine results
        final_results = {
            'is_valid': self.best_validation.get('is_valid', False),
            'cost_comparisons': self.best_validation.get('cost_comparisons', {}),
            'metrics': metrics,
            'statistics': {
                'iterations': iteration,
                'accepted_solutions': accepted_solutions,
                'rejected_solutions': rejected_solutions,
                'improvements': improvements
            }
        }
        
        final_costs = {}
        for vehicle, shipments in self.best_solution.items():
            if shipments:
                route = ['Nanning'] + [s.delivery_location_id for s in shipments]
                route_cost = self.cost_calculator.calculate_route_cost(route, self.graph)
                final_costs[vehicle.vehicle_id] = {
                    'total_cost': route_cost.defuzzify(),
                    'route': route,
                    'volume_utilization': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
                    'weight_utilization': sum(s.weight for s in shipments) / vehicle.max_weight * 100,
                    }
        
        with open('final_costs.json', 'w') as f:
            json.dump(final_costs, f, indent=2)

        final_metrics = {
                vehicle.vehicle_id: {
                    'Total_Distance_km': sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance')
                                        for i in range(len(route)-1)),
                    'Travel_Time_min': sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='time')
                                        for i in range(len(route)-1)),
                    'Stops': len(set(s.delivery_location_id for s in shipments)),
                    'Fuel_Cost': self.cost_calculator.calculate_fuel_cost(
                        sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance') 
                            for i in range(len(route)-1)),
                        vehicle.fuel_efficiency
                    ),
                    'Total_Cost': sum(cost_components),
                    'Real_CBM': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
                    'Optimized_CBM': sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100,
                    }
                for vehicle, shipments in self.best_solution.items()
                if shipments  # Only include vehicles with assignments
                for route in [['Nanning'] + [s.delivery_location_id for s in shipments]]
                for cost_components in [[
                self.cost_calculator.calculate_fuel_cost(
                    sum(nx.shortest_path_length(self.graph, self.get_node_name(route[i]), self.get_node_name(route[i+1]), weight='distance') 
                        for i in range(len(route)-1)),
                    vehicle.fuel_efficiency
                ),]]
            }
            
            # Save to CSV
        with open('final_metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Vehicle_ID', 'Total_Distance_km', 'Travel_Time_min', 'Stops', 
                            'Fuel_Cost', 'Penalties', 'Driver_Costs', 'Customs_Fees', 
                            'Total_Cost', 'Real_CBM', 'Optimized_CBM', 'Improvement'])
            for vehicle_id, metrics in final_metrics.items():
                writer.writerow([vehicle_id] + list(metrics.values()))
        
        return self.best_solution, self.best_cost, final_results

    def evaluate_solution(self, solution: Dict[Vehicle, List[Shipment]]) -> Tuple[TriangularFuzzyNumber, Dict]:
        """Evaluate solution with improved cost calculation."""
        total_cost = TriangularFuzzyNumber(0, 0, 0)
        solution_costs = {}
        
        # Calculate utilization metrics
        total_volume_utilization = 0
        total_weight_utilization = 0
        vehicles_used = 0
        
        # Track historical costs for unused vehicles
        unused_vehicle_costs = 0
        
        for vehicle, shipments in solution.items():
            if not shipments:
                # Add historical cost for unused vehicle
                if hasattr(vehicle, 'total_costs'):
                    historical_cost = vehicle.total_costs.get('total_cost', 0)
                    vehicle_costs = TriangularFuzzyNumber(
                        historical_cost * 0.95,
                        historical_cost,
                        historical_cost * 1.05
                    )
                    solution_costs[vehicle] = vehicle_costs
                    total_cost = total_cost + vehicle_costs
                    unused_vehicle_costs += historical_cost
                continue
                
            vehicles_used += 1
            
            # Calculate utilization
            volume_util = sum(s.total_cbm for s in shipments) / vehicle.max_cbm * 100
            weight_util = sum(s.weight for s in shipments) / vehicle.max_weight * 100
            total_volume_utilization += volume_util
            total_weight_utilization += weight_util
            
            # Build and optimize route
            route = ['Nanning'] + [s.delivery_location_id for s in shipments]
            
            # Add route efficiency penalty
            efficiency_penalty = evaluate_route_efficiency(
                route, 
                self.graph,
                self.node_converter['get_node_name']
            )
            
            # Calculate route cost (this already includes tax and customs fees)
            route_cost = self.cost_calculator.calculate_route_cost(route, self.graph)
            
            # Add penalties
            # Penalty for poor utilization
            if volume_util < 60 or weight_util < 30:
                utilization_penalty = TriangularFuzzyNumber(500, 750, 1000)
                route_cost = route_cost + utilization_penalty
            
            # Penalty for route inefficiency
            efficiency_factor = TriangularFuzzyNumber(
                efficiency_penalty * 0.1,
                efficiency_penalty * 0.15,
                efficiency_penalty * 0.2
            )
            route_cost = route_cost + efficiency_factor
            
            solution_costs[vehicle] = route_cost
            total_cost = total_cost + route_cost
        
        # Add global solution penalties
        if vehicles_used > 0:
            avg_volume_util = total_volume_utilization / vehicles_used
            avg_weight_util = total_weight_utilization / vehicles_used
            
            # Penalty for unbalanced utilization
            if max(abs(avg_volume_util - volume_util) for _, shipments in solution.items() if shipments) > 20:
                balance_penalty = TriangularFuzzyNumber(1000, 1500, 2000)
                total_cost = total_cost + balance_penalty
            
            # Penalty for excessive unused vehicles (if more than 1 vehicle is unused)
            unused_vehicles = len(solution) - vehicles_used
            if unused_vehicles > 1:
                unused_penalty = TriangularFuzzyNumber(
                    1000 * (unused_vehicles - 1),
                    1500 * (unused_vehicles - 1),
                    2000 * (unused_vehicles - 1)
                )
                total_cost = total_cost + unused_penalty
        
        # Create detailed validation results
        validation_results = {
            'is_valid': True,
            'vehicles_used': vehicles_used,
            'unused_vehicles': len(solution) - vehicles_used,
            'unused_vehicle_costs': unused_vehicle_costs,
            'total_volume_utilization': total_volume_utilization / vehicles_used if vehicles_used > 0 else 0,
            'total_weight_utilization': total_weight_utilization / vehicles_used if vehicles_used > 0 else 0,
            'cost_breakdown': {
                vehicle.vehicle_id: {
                    'total_cost': cost.defuzzify(),
                    'utilization': {
                        'volume': sum(s.total_cbm for s in solution[vehicle]) / vehicle.max_cbm * 100 if solution[vehicle] else 0,
                        'weight': sum(s.weight for s in solution[vehicle]) / vehicle.max_weight * 100 if solution[vehicle] else 0
                    },
                    'shipments': len(solution[vehicle]) if solution[vehicle] else 0,
                    'is_active': bool(solution[vehicle])
                }
                for vehicle, cost in solution_costs.items()
            }
        }
        
        # Add validator results if available
        if self.validator:
            validator_results = self.validator.validate_solution(solution, solution_costs)
            validation_results.update(validator_results)
        
        return total_cost, validation_results

    def acceptance_probability(self, cost_diff: float) -> float:
        """Calculate acceptance probability using current temperature."""
        if cost_diff <= 0:
            return 1.0
        return min(1.0, max(0.0001, pow(2.718, -cost_diff / self.temperature)))
    
    def plan_route(self, vehicle: Vehicle, shipments: List[Shipment]) -> List[str]:
        """
        Plan optimal route for a vehicle considering FILO constraints.
        
        Parameters:
        - vehicle: Vehicle to plan route for
        - shipments: List of shipments assigned to vehicle
        
        Returns:
        - List[str]: Ordered list of node IDs representing the route
        """
        if not shipments:
            return []

        # Group shipments by destination
        dest_shipments = {}
        for shipment in shipments:
            if shipment.delivery_location_id not in dest_shipments:
                dest_shipments[shipment.delivery_location_id] = []
            dest_shipments[shipment.delivery_location_id].append(shipment)

        # Order destinations by total volume (FILO: larger volumes delivered last)
        destinations = [(dest, sum(s.total_cbm for s in group))
                    for dest, group in dest_shipments.items()]
        destinations.sort(key=lambda x: x[1], reverse=True)

        # Build route
        route = ["NNG"]  # Starting point
        for dest, _ in destinations:
            route.append(dest)
        
        return route

    def check_route_validity(self, route: List[str], graph: nx.Graph) -> bool:
        """Check if a route is valid using consistent node codes."""
        current = "NNG"
        
        for next_loc in route[1:]:
            next_name = self.code_to_name.get(next_loc, next_loc)
            try:
                path = nx.shortest_path(graph, current, next_name, weight='distance')
                current = next_name
            except nx.NetworkXNoPath:
                print(f"No valid path between {current} and {next_name}")
                return False
        return True
    
    def calculate_solution_metrics(self, solution: Dict[Vehicle, List[Shipment]]) -> Dict:
        """Calculate metrics for the given solution."""
        total_distance = 0
        total_border_crossings = 0
        vehicle_metrics = {}
        
        for vehicle, shipments in solution.items():
            # Initialize vehicle metrics
            vehicle_metrics[vehicle.vehicle_id] = {
                'distance': 0,
                'border_crossings': 0,
                'volume_utilization': 0,
                'weight_utilization': 0,
                'num_shipments': len(shipments),
                'route': []
        }
            if not shipments:
                continue
                
            # Calculate route distance
            route = ['Nanning'] + [self.node_converter['get_node_name'](s.delivery_location_id) for s in shipments]
            vehicle_metrics[vehicle.vehicle_id]['route'] = route
            
            route_distance = sum(
                nx.shortest_path_length(self.graph, route[i], route[i+1], weight='distance')
                for i in range(len(route)-1)
            )
            vehicle_metrics[vehicle.vehicle_id]['distance'] = route_distance
            total_distance += route_distance
            
            # Count border crossings
            border_crossings = 0
            current_country = self.graph.nodes['NNG']['country']
            for node in route[1:]:
                next_country = self.graph.nodes[self.node_converter['get_node_id'](node)]['country']
                if current_country != next_country:
                    border_crossings += 1
                    current_country = next_country
            
            vehicle_metrics[vehicle.vehicle_id]['border_crossings'] = border_crossings
            total_border_crossings += border_crossings

            total_volume = sum(s.total_cbm for s in shipments)
            total_weight = sum(s.weight for s in shipments)
            vehicle_metrics[vehicle.vehicle_id]['volume_utilization'] = (total_volume / vehicle.max_cbm) * 100
            vehicle_metrics[vehicle.vehicle_id]['weight_utilization'] = (total_weight / vehicle.max_weight) * 100
        
        return {
        'total_distance': total_distance,
        'border_crossings': total_border_crossings,
        'total_vehicles_used': sum(1 for v, s in solution.items() if s),
        'total_shipments': sum(len(s) for s in solution.values()),
        'vehicle_metrics': vehicle_metrics
    }