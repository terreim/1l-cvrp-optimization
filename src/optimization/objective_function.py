from typing import Tuple, Dict
from src.fuzzy.fuzzy_number import TriangularFuzzyNumber
from src.models.cost import Cost
import networkx as nx

def evaluate_solution(solution: dict, 
                     network_graph: nx.Graph, 
                     cost_calculator: Cost,
                     validator = None) -> Tuple[TriangularFuzzyNumber, Dict]:
    """
    Evaluate the total operational cost of the solution using fuzzy numbers.
    
    Parameters:
    - solution: Dict[Vehicle, List[Shipment]] - The current solution
    - network_graph: NetworkX graph containing the network
    - cost_calculator: Cost calculator instance
    - validator: Optional validator for historical comparison
    
    Returns:
    - Tuple[TriangularFuzzyNumber, Dict]: Total fuzzy cost and validation results
    """
    vehicle_costs = {}
    total_cost = TriangularFuzzyNumber(0, 0, 0)
    code_to_name = network_graph.graph.get('code_to_name', {})
    name_to_code = network_graph.graph.get('name_to_code', {})
    
    
    for vehicle, shipments in solution.items():
        if not shipments:
            continue
            
        current_node = "Nanning"  # Origin node
        current_country = network_graph.nodes[name_to_code.get("Nanning", "Nanning")]['country']
        vehicle_total_cost = TriangularFuzzyNumber(0, 0, 0)
        
        # Create route including origin
        route = [current_node]
        for shipment in shipments:
            delivery_node = code_to_name.get(shipment.delivery_location_id, shipment.delivery_location_id)
            route.append(delivery_node)
            
        # Calculate costs for each leg
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            try:
                # Get path information using node names
                path = nx.shortest_path(network_graph, from_node, to_node, weight='distance')
                distance = nx.shortest_path_length(network_graph, from_node, to_node, weight='distance')
                
                to_country = network_graph.nodes[name_to_code.get(to_node, to_node)]['country']
                is_border_crossing = current_country != to_country
                is_first_day = i == 0
                
                # Calculate delivered goods value
                delivered_goods_value = sum(
                    s.price for s in shipments 
                    if code_to_name.get(s.delivery_location_id, s.delivery_location_id) == to_node
                )
                
                # Get daily costs
                daily_costs = cost_calculator.calculate_daily_costs(
                    distance=distance,
                    fuel_efficiency=vehicle.fuel_efficiency,
                    goods_value=delivered_goods_value,
                    from_country=current_country,
                    to_country=to_country,
                    is_border_crossing=is_border_crossing,
                    is_first_day=is_first_day
                )
                
                # Convert to fuzzy number and add to total
                leg_cost = TriangularFuzzyNumber(
                    daily_costs['total_cost'] * 0.95,  # left
                    daily_costs['total_cost'],         # peak
                    daily_costs['total_cost'] * 1.05   # right
                )
                vehicle_total_cost += leg_cost
                current_country = to_country
                vehicle_costs[vehicle] = vehicle_total_cost

                # Add to total cost
                total_cost += vehicle_total_cost

            except nx.NetworkXNoPath:
                print(f"No path found between {from_node} and {to_node}")
                return TriangularFuzzyNumber(float('inf'), float('inf'), float('inf')), {'is_valid': False}
            except Exception as e:
                print(f"Error calculating route: {str(e)}")
                return TriangularFuzzyNumber(float('inf'), float('inf'), float('inf')), {'is_valid': False}
                
    validation_results = {'is_valid': True} if validator is None else validator.validate_solution(solution, vehicle_costs)
    return total_cost, validation_results