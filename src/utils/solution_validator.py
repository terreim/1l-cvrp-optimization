from typing import Dict, List, Tuple
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment
from src.fuzzy.fuzzy_number import TriangularFuzzyNumber

class SolutionValidator:
    def __init__(self, historical_costs: Dict):
        """
        Initialize validator with historical cost data.
        
        Parameters:
        - historical_costs: Dictionary containing historical cost data
        """
        self.historical_costs = historical_costs
        # Extract just the total_cost values from the 'total' dictionary
        self.historical_totals = {
            vehicle_id: data['total']['total_cost']  # Access nested total_cost
            for vehicle_id, data in historical_costs.items()
        }

    def validate_solution(self, 
                     solution: Dict[Vehicle, List[Shipment]], 
                     solution_costs: Dict[Vehicle, TriangularFuzzyNumber]) -> Dict:
        """
        Validate a solution against historical data.
        
        Returns:
        - Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'cost_comparisons': {},
            'violations': [],
            'improvements': []
        }

        # Check capacity constraints
        if not self.is_solution_feasible(solution):
            validation_results['is_valid'] = False
            validation_results['violations'].append("Capacity constraints violated")
            return validation_results

        # Compare with historical costs
        for vehicle, shipments in solution.items():
            if not shipments:  # Skip empty vehicles
                continue
                
            vehicle_id = vehicle.vehicle_id
            if vehicle_id not in self.historical_totals:
                continue

            # Only process vehicles that have costs calculated
            if vehicle not in solution_costs:
                continue

            historical_total = float(self.historical_totals[vehicle_id])  # Convert to float
            current_cost = solution_costs[vehicle].defuzzify()

            comparison = {
                'historical_cost': historical_total,
                'current_cost': current_cost,
                'difference': historical_total - current_cost,
                'improvement_percentage': 
                    ((historical_total - current_cost) / historical_total) * 100
                    if historical_total > 0 else 0
            }
            
            validation_results['cost_comparisons'][vehicle_id] = comparison

            # Track improvements
            if comparison['difference'] > 0:
                validation_results['improvements'].append(
                    f"Vehicle {vehicle_id}: {comparison['improvement_percentage']:.2f}% improvement"
                )

        return validation_results

    @staticmethod
    def is_solution_feasible(solution: Dict[Vehicle, List[Shipment]]) -> bool:
        """Check if the solution respects vehicle capacity constraints."""
        for vehicle, shipments in solution.items():
            total_volume = sum(s.total_cbm for s in shipments)
            total_weight = sum(s.weight for s in shipments)
            
            if total_volume > vehicle.max_cbm or total_weight > vehicle.max_weight:
                return False
        return True