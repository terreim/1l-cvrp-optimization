from typing import Dict, List
import networkx as nx
from src.fuzzy.fuzzy_number import TriangularFuzzyNumber
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment

class Cost:
    def __init__(self, fuel_price_per_liter: float = 0.8):
        self.fuel_price = fuel_price_per_liter
        self.per_diem_rate = 30.0
        
        # Tax rates by country name
        self.tax_rates = {
            "China": 0.10,      # From CV001-003 customs data
            "Vietnam": 0.10,    # From VC001 customs data
            "Laos": 0.10,      # From LC001 customs data
            "Cambodia": 0.10,   # From CT001-003 customs data
            "Thailand": 0.07,   # From LT001-002, CT001-003 customs data
            "Myanmar": 0.12,    # From MT001-003 customs data
            "Malaysia": 0.10,   # From TM001-002 customs data
            "Singapore": 0.08   # From MS001 customs data
        }

        # Base customs fees by country pairs (can be extended based on border crossing data)
        self.customs_base_fees = {
            ("China", "Vietnam"): 160,
            ("China", "Laos"): 162,
            ("Vietnam", "Laos"): 162,
            ("Vietnam", "Cambodia"): 160,
            ("Laos", "Cambodia"): 160,
            ("Laos", "Thailand"): 161,
            ("Laos", "Myanmar"): 160,
            ("Cambodia", "Thailand"): 160,
            ("Myanmar", "Thailand"): 160,
            ("Thailand", "Malaysia"): 158,
            ("Malaysia", "Singapore"): 158
        }

        self.max_driving_hours_per_day = 10  # Maximum driving hours per day
        self.average_speed = 60  # Average speed in km/h
        self.fuel_tank_capacity = 400  # Liters
        self.refuel_time = 0.5  # Hours
        self.rest_time_per_day = 10  # Hours (including sleep and breaks)
        self.border_crossing_time = 4  # Hours
        
        # Refueling costs by country (base cost + service fee)
        self.refuel_costs = {
            "China": 5,
            "Vietnam": 4,
            "Laos": 4,
            "Cambodia": 4,
            "Thailand": 3,
            "Myanmar": 4,
            "Malaysia": 3,
            "Singapore": 6
        }

    def calculate_cost(self, solution: Dict[Vehicle, List[Shipment]]) -> float:
        """Calculate total cost of solution including route efficiency and load balance."""
        route_cost = self.calculate_route_cost(solution)
        balance_penalty = self.calculate_balance_penalty(solution)
        proximity_penalty = self.calculate_proximity_penalty(solution)
        unused_vehicle_penalty = self.calculate_unused_vehicle_penalty(solution)
        
        return route_cost + balance_penalty + proximity_penalty + unused_vehicle_penalty

    def calculate_proximity_penalty(self, solution: Dict[Vehicle, List[Shipment]]) -> float:
        """Calculate penalty for not consolidating nearby destinations."""
        penalty = 0
        
        # Define nearby city pairs (distances in km)
        nearby_cities = {
            ('KualaLumpur', 'PortOfSingapore'): 350,  # Approximate distance
            ('Bangkok', 'PhnomPenh'): 650,
            ('HoChiMinh', 'PhnomPenh'): 230,
            ('Hanoi', 'HaiPhong'): 105,
        }
        
        for vehicle, shipments in solution.items():
            if not shipments:
                continue
                
            # Get all destinations for this vehicle
            destinations = [s.delivery_location_id for s in shipments]
            
            # Check each destination against nearby cities
            for i, dest1 in enumerate(destinations):
                for j, dest2 in enumerate(destinations[i+1:], i+1):
                    # Skip if same destination
                    if dest1 == dest2:
                        continue
                        
                    # Check if these cities are in our nearby pairs
                    for (city1, city2), threshold_dist in nearby_cities.items():
                        if ((dest1 == city1 and dest2 == city2) or 
                            (dest1 == city2 and dest2 == city1)):
                            # Add penalty if nearby cities are in different vehicles
                            penalty += 500
        
        return penalty

    def calculate_balance_penalty(self, solution: Dict[Vehicle, List[Shipment]]) -> float:
        """Calculate penalty for unbalanced load distribution."""
        # Get volume and weight utilization for each vehicle
        volume_utils = []
        weight_utils = []
        
        for vehicle, shipments in solution.items():
            if vehicle.max_cbm > 0 and vehicle.max_weight > 0:
                volume_util = sum(s.total_cbm for s in shipments) / vehicle.max_cbm
                weight_util = sum(s.weight for s in shipments) / vehicle.max_weight
                volume_utils.append(volume_util)
                weight_utils.append(weight_util)
        
        if not volume_utils:
            return 0
        
        # Calculate standard deviation for both volume and weight
        def calc_std_dev(utils):
            avg = sum(utils) / len(utils)
            variance = sum((u - avg) ** 2 for u in utils) / len(utils)
            return variance ** 0.5
        
        volume_std_dev = calc_std_dev(volume_utils)
        weight_std_dev = calc_std_dev(weight_utils)
        
        # Penalize based on both volume and weight imbalance
        return (volume_std_dev * 1000) + (weight_std_dev * 800)

    def calculate_unused_vehicle_penalty(self, solution: Dict[Vehicle, List[Shipment]]) -> float:
        """Calculate penalty for unused vehicles."""
        unused_count = sum(1 for shipments in solution.values() if not shipments)
        # Lower penalty for unused vehicles to allow for better consolidation
        return unused_count * 300

    def calculate_balance_penalty(self, solution: Dict[Vehicle, List[Shipment]]) -> float:
        """Calculate penalty for unbalanced load distribution."""
        # Get volume utilization for each vehicle
        utilizations = [
            sum(s.total_cbm for s in shipments) / vehicle.max_cbm
            for vehicle, shipments in solution.items()
            if vehicle.max_cbm > 0  # Avoid division by zero
        ]
        
        if not utilizations:
            return 0
        
        # Calculate standard deviation of utilizations
        avg_util = sum(utilizations) / len(utilizations)
        variance = sum((u - avg_util) ** 2 for u in utilizations) / len(utilizations)
        std_dev = variance ** 0.5
        
        # Penalize based on deviation from balanced load
        return std_dev * 1000  # Adjust multiplier as needed

    def calculate_travel_time(self, distance: float, is_border_crossing: bool) -> Dict[str, float]:
        """
        Calculate travel time including breaks and border crossings.
        Returns dictionary with days and hours.
        """
        # Calculate pure driving time
        driving_hours = distance / self.average_speed
        
        # Add border crossing time if applicable
        if is_border_crossing:
            driving_hours += self.border_crossing_time
        
        # Calculate number of refuel stops needed
        fuel_range = self.fuel_tank_capacity / 0.3  # km per tank (using standard fuel efficiency)
        refuel_stops = max(0, distance // fuel_range)
        refuel_time = refuel_stops * self.refuel_time
        
        # Add refuel time to total hours
        total_hours = driving_hours + refuel_time
        
        # Calculate full days needed
        working_hours_per_day = self.max_driving_hours_per_day
        days = total_hours / working_hours_per_day
        days = max(1, round(days))  # Minimum 1 day
        
        return {
            "days": days,
            "hours": total_hours,
            "refuel_stops": refuel_stops
        }

    def get_fuzzy_driver_salary(self, distance: float) -> TriangularFuzzyNumber:
        """
        Calculate fuzzy driver salary based on route distance.
        Longer routes require more experienced drivers with higher rates.
        
        Parameters:
        - distance: Total route distance in km
        
        Returns:
        - TriangularFuzzyNumber representing daily salary rate
        """
        if distance <= 500:  # Short haul
            return TriangularFuzzyNumber(28.0, 30.5, 33.0)
        elif distance <= 1000:  # Medium haul
            return TriangularFuzzyNumber(30.5, 33.0, 35.5)
        else:  # Long haul
            return TriangularFuzzyNumber(33.0, 35.5, 38.0)

    def calculate_fuel_cost(self, distance: float, fuel_efficiency: float) -> float:
        """Calculate fuel cost for a given distance."""
        return distance * fuel_efficiency * self.fuel_price

    def get_customs_fee(self, from_country: str, to_country: str) -> float:
        """Get base customs fee for crossing between two countries."""
        countries = tuple(sorted([from_country, to_country]))  # Sort to ensure consistent lookup
        return self.customs_base_fees.get(countries, 160.0)  # Default to 160 if not found

    def calculate_daily_costs(self, 
                            distance: float,
                            fuel_efficiency: float,
                            goods_value: float,
                            from_country: str,
                            to_country: str,
                            is_border_crossing: bool,
                            is_first_day: bool) -> Dict[str, float]:
        """Calculate all costs for a day including travel time and refueling."""
        if distance == float('inf'):
            return {
                'total_cost': float('inf'),
                'details': 'Invalid route - no path exists'
            }
        
        # Calculate travel time and refueling needs
        travel_info = self.calculate_travel_time(distance, is_border_crossing)
        days = travel_info["days"]
        refuel_stops = travel_info["refuel_stops"]
        
        # Get fuzzy driver salary for the route
        fuzzy_salary = self.get_fuzzy_driver_salary(distance)
        
        # Calculate refueling costs
        refuel_cost = sum([
            self.refuel_costs.get(from_country, 4),  # Base country refuel cost
            self.refuel_costs.get(to_country, 4)     # Destination country refuel cost
        ]) * refuel_stops / 2  # Average between countries
        
        costs = {
            "per_diem": self.per_diem_rate * days,  # Per diem for actual days traveled
            "driver_salary": fuzzy_salary.defuzzify() * days,  # Salary for actual days
            "fuel_cost": self.calculate_fuel_cost(distance, fuel_efficiency),
            "refuel_service_cost": refuel_cost,
            "custom_fee": self.get_customs_fee(from_country, to_country) if is_border_crossing else 0.0,
            "tax_on_goods": self.calculate_tax(goods_value, to_country) if is_border_crossing else 0.0,
            "overhead": 100.0 if is_first_day else 0.0,
            "emergency": 200.0 if is_first_day else 0.0
        }
        
        costs["total_cost"] = sum(costs.values())
        costs["travel_days"] = days
        costs["refuel_stops"] = refuel_stops
        return costs

    def calculate_tax(self, goods_value: float, country: str) -> float:
        """Calculate tax based on goods value and destination country."""
        return goods_value * self.tax_rates.get(country, 0.0)

    def get_fuzzy_processing_time(self, from_country: str, to_country: str, is_inbound: bool) -> TriangularFuzzyNumber:
        """
        Get fuzzy processing time for border crossing.
        This could be expanded based on the border crossing data.
        """
        if is_inbound:
            return TriangularFuzzyNumber(120, 240, 360)  # Default inbound times
        else:
            return TriangularFuzzyNumber(30, 60, 120)    # Default outbound times
        
    def calculate_route_cost(self, route: List[str], graph: nx.Graph) -> TriangularFuzzyNumber:
        """Calculate total cost for a route using fuzzy numbers, including travel time."""
        if not route or len(route) < 2:
            return TriangularFuzzyNumber(float('inf'), float('inf'), float('inf'))
        
        total_cost = TriangularFuzzyNumber(0, 0, 0)
        total_days = 0
        total_refuel_stops = 0
        
        code_to_name = graph.graph.get('code_to_name', {})
        name_to_code = graph.graph.get('name_to_code', {})
            
        # Get vehicle attributes from graph
        fuel_efficiency = graph.graph.get('fuel_efficiency', 0.3)
        goods_value = graph.graph.get('goods_value', 10000)
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            try:
                # Calculate leg costs and time
                distance = nx.shortest_path_length(graph, 
                    code_to_name.get(from_node, from_node), 
                    code_to_name.get(to_node, to_node), 
                    weight='distance')
                
                from_country = graph.nodes[from_node].get('country', 'Unknown')
                to_country = graph.nodes[to_node].get('country', 'Unknown')
                is_border_crossing = from_country != to_country
                
                daily_costs = self.calculate_daily_costs(
                    distance=distance,
                    fuel_efficiency=fuel_efficiency,
                    goods_value=goods_value,
                    from_country=from_country,
                    to_country=to_country,
                    is_border_crossing=is_border_crossing,
                    is_first_day=(i == 0)
                )
                
                total_days += daily_costs["travel_days"]
                total_refuel_stops += daily_costs["refuel_stops"]
                
                # Convert to fuzzy number
                leg_cost = TriangularFuzzyNumber(
                    daily_costs['total_cost'] * 0.95,
                    daily_costs['total_cost'],
                    daily_costs['total_cost'] * 1.05
                )
                total_cost = total_cost + leg_cost
                
            except nx.NetworkXNoPath:
                print(f"No path found between {from_node} and {to_node}")
                return TriangularFuzzyNumber(float('inf'), float('inf'), float('inf'))
        
        # Add route summary to graph attributes
        graph.graph['route_summary'] = {
            'total_days': total_days,
            'total_refuel_stops': total_refuel_stops
        }
        
        return total_cost