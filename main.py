from src.utils.data_loader import load_json, parse_nodes, parse_vehicles, parse_edges
from src.models.network_builder import build_graph
from src.models.cost import Cost
from src.optimization.optimization_model import SimulatedAnnealingOptimizer
from src.utils.solution_validator import SolutionValidator
import json

def main():
    # 1. Load all necessary data
    nodes_data = load_json('data/constants/nodes.json')
    vehicles_data = load_json('data/constants/vehicles.json')
    edges_data = load_json('data/constants/edges.json')
    packing_data = load_json('data/historical_data/packing_plan/packing_plan3.json')
    historical_costs = load_json('data/historical_data/cost_breakdown/cost_breakdown3.json')

    # 2. Parse data into objects
    nodes = parse_nodes(nodes_data) 
    vehicles = parse_vehicles(vehicles_data, packing_data)
    edges = parse_edges(edges_data)
    
    # 3. Build and visualize network
    G = build_graph(edges, nodes)
    
    # 4. Initialize components
    cost_calculator = Cost()
    validator = SolutionValidator(historical_costs)
    
    # 5. Create and run optimizer
    optimizer = SimulatedAnnealingOptimizer(
        vehicles=vehicles,
        shipments=[s for v in vehicles for s in v.shipments],  # Flatten shipments
        graph=G,
        cost_calculator=cost_calculator,
        validator=validator,
        initial_temperature=2000.0,
        cooling_rate=0.995,
        termination_temperature=0.1,
        max_iterations=1000
    )
    
    # 6. Run optimization
    best_solution, best_cost, results = optimizer.optimize()
    
    # 7. Print detailed results
    print("\n=== Overall Results ===")
    print(f"Best solution cost: ${best_cost.defuzzify():,.2f}")
    print(f"Total distance: {results['metrics']['total_distance']:,.2f} km")
    print(f"Total border crossings: {results['metrics']['border_crossings']}")
    print(f"Solution validity: {results['is_valid']}")
    
    print("\n=== Optimization Statistics ===")
    print(f"Total iterations: {results['statistics']['iterations']}")
    print(f"Accepted solutions: {results['statistics']['accepted_solutions']}")
    print(f"Improvements found: {results['statistics']['improvements']}")
    
    # Print vehicle-specific results
    print("\n=== Vehicle Details ===")
    for vehicle in best_solution:
        metrics = results['metrics']['vehicle_metrics'][vehicle.vehicle_id]
        print(f"\nVehicle {vehicle.vehicle_id}:")
        if metrics['num_shipments'] > 0:
            print(f"Route: {' -> '.join(metrics['route'])}")
            print(f"Distance: {metrics['distance']:,.2f} km")
            print(f"Border crossings: {metrics['border_crossings']}")
            print(f"Volume utilization: {metrics['volume_utilization']:.1f}%")
            print(f"Weight utilization: {metrics['weight_utilization']:.1f}%")
            print("Shipments:")
            for shipment in best_solution[vehicle]:
                print(f"  - {shipment.shipment_id} to {shipment.delivery_location_id}")
                print(f"    Volume: {shipment.total_cbm:.2f} CBM")
                print(f"    Weight: {shipment.weight:,.2f} kg")
        else:
            print("No shipments assigned")
    
    # Cost Comparisons
    print("\n=== Cost Comparison ===")
    total_historical = 0
    total_optimized = 0
    
    for vehicle_id, comparison in results['cost_comparisons'].items():
        print(f"\nVehicle {vehicle_id}:")
        print(f"  Historical cost: ${comparison['historical_cost']:,.2f}")
        print(f"  Optimized cost: ${comparison['current_cost']:,.2f}")
        print(f"  Improvement: {comparison['improvement_percentage']:,.2f}%")
        print(f"  Savings: ${comparison['difference']:,.2f}")
        
        total_historical += comparison['historical_cost']
        total_optimized += comparison['current_cost']
    
    # Overall cost improvement
    print("\n=== Overall Cost Improvement ===")
    total_savings = total_historical - total_optimized
    improvement_percentage = (total_savings / total_historical * 100) if total_historical > 0 else 0
    print(f"Total Historical Cost: ${total_historical:,.2f}")
    print(f"Total Optimized Cost: ${total_optimized:,.2f}")
    print(f"Total Savings: ${total_savings:,.2f}")
    print(f"Overall Improvement: {improvement_percentage:,.2f}%")
    
    # Violations and Improvements (if any)
    if results.get('violations'):
        print("\n=== Violations ===")
        for violation in results['violations']:
            print(f"- {violation}")
    
    if results.get('improvements'):
        print("\n=== Specific Improvements ===")
        for improvement in results['improvements']:
            print(f"- {improvement}")
    
    # 8. Save results
    save_results(best_solution, results, 'optimization_results2.json')
    print("\nResults have been saved to 'optimization_results2.json'")

def save_results(solution, results, filename):
    """Save optimization results to a JSON file."""
    output = {
        'solution': {
            vehicle.vehicle_id: {
                'shipments': [s.shipment_id for s in shipments],
                'route': results['metrics']['vehicle_metrics'][vehicle.vehicle_id]['route'],
                'metrics': results['metrics']['vehicle_metrics'][vehicle.vehicle_id]
            }
            for vehicle, shipments in solution.items()
        },
        'overall_metrics': {
            'total_distance': results['metrics']['total_distance'],
            'total_border_crossings': results['metrics']['border_crossings'],
            'is_valid': results['is_valid']
        },
        'cost_comparisons': results['cost_comparisons'],
        'violations': results.get('violations', []),
        'improvements': results.get('improvements', [])
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
