from src.models.shipment import Shipment

class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, dimensions, max_weight, fuel_capacity, fuel_efficiency):
        """
        Initialize a Vehicle.

        Parameters:
        - vehicle_id (str): Unique identifier for the vehicle.
        - vehicle_type (str): Type of the vehicle (e.g., 'standard_truck').
        - dimensions (dict): Dimensions with keys 'length', 'width', 'height' in meters.
                              Example: {'length': 12.192, 'width': 2.438, 'height': 2.591}
        - max_weight (float): Maximum weight capacity in kilograms.
        - fuel_capacity (float): Fuel capacity in liters.
        - fuel_efficiency (float): Fuel consumption rate (liters per km).
        """
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.dimensions = dimensions  # {'length': float, 'width': float, 'height': float}
        self.max_weight = max_weight
        self.fuel_capacity = fuel_capacity
        self.fuel_efficiency = fuel_efficiency
        self.current_load_weight = 0.0  # in kg
        self.current_load_length = 0.0  # in meters (1D packing)
        self.shipments = []  # List of Shipment objects
        self.route = []  # Ordered list of Node IDs
        self.costs = {}  # Dictionary to store daily costs and totals
        self.max_cbm = dimensions['length'] * dimensions['width'] * dimensions['height']  # Total available volume

    def reset_state(self):
        """Reset vehicle's state to initial conditions."""
        self.current_load_weight = 0.0
        self.current_load_length = 0.0
        self.shipments = []
        self.route = []

    def can_add_shipment(self, shipment: Shipment) -> bool:
        """Check if shipment can be added considering both volume and weight."""
        new_volume = self.current_load_length + shipment.total_cbm
        new_weight = self.current_load_weight + shipment.weight
        
        # Add some tolerance (e.g., 0.1%) to handle floating point precision
        volume_ok = new_volume <= (self.max_cbm * 1.001)
        weight_ok = new_weight <= (self.max_weight * 1.001)
        
        print(f"\nCapacity check for vehicle {self.vehicle_id}:")
        print(f"Volume: {new_volume:.2f}/{self.max_cbm:.2f} ({volume_ok})")
        print(f"Weight: {new_weight:.2f}/{self.max_weight:.2f} ({weight_ok})")
        
        return volume_ok and weight_ok

    def get_remaining_capacity(self):
        """Get remaining capacity in both weight and volume."""
        return {
            'weight': self.max_weight - self.current_load_weight,
            'cbm': self.max_cbm - self.current_load_length
        }
       
    def add_shipment(self, shipment):
        """
        Add a shipment to the vehicle.

        Parameters:
        - shipment (Shipment): The shipment to add.
        """
        self.shipments.append(shipment)
        self.current_load_weight += shipment.weight
        self.current_load_length += shipment.total_cbm  # Assuming total_cbm represents length in 1D

    def remove_shipment(self, shipment):
        """
        Remove a shipment from the vehicle.

        Parameters:
        - shipment (Shipment): The shipment to remove.
        """
        if shipment in self.shipments:
            self.shipments.remove(shipment)
            self.current_load_weight -= shipment.weight
            self.current_load_length -= shipment.total_cbm

    def get_capacity_utilization(self):
        """Calculate capacity utilization percentages."""
        weight_utilization = (self.current_load_weight / self.max_weight) * 100 if self.max_weight > 0 else 0
        volume_utilization = (self.current_load_length / self.max_cbm) * 100 if self.max_cbm > 0 else 0
        return weight_utilization, volume_utilization

    def add_route_leg(self, route_leg):
        """
        Add a RouteLeg to the vehicle's route.

        Parameters:
        - route_leg (RouteLeg): The RouteLeg object to add.
        """
        self.route.append(route_leg)

    def add_daily_cost(self, date, per_diem, driver_salary, fuel_cost, custom_fee, tax_on_goods, overhead, emergency):
        """
        Add daily cost breakdown.

        Parameters:
        - date (str): Date in 'YYYY/M/D' format.
        - per_diem (float)
        - driver_salary (float)
        - fuel_cost (float)
        - custom_fee (float)
        - tax_on_goods (float)
        - overhead (float)
        - emergency (float)
        """
        total_cost = per_diem + driver_salary + fuel_cost + custom_fee + tax_on_goods + overhead + emergency
        self.costs[date] = {
            "per_diem": per_diem,
            "driver_salary": driver_salary,
            "fuel_cost": fuel_cost,
            "custom_fee": custom_fee,
            "tax_on_goods": tax_on_goods,
            "overhead": overhead,
            "emergency": emergency,
            "total_cost": total_cost
        }

    def calculate_total_costs(self):
        """
        Calculate aggregated total costs.

        Returns:
        - dict: Aggregated costs.
        """
        aggregated = {
            "per_diem": 0.0,
            "driver_salary": 0.0,
            "fuel_cost": 0.0,
            "custom_fee": 0.0,
            "tax_on_goods": 0.0,
            "overhead": 0.0,
            "emergency": 0.0,
            "total_cost": 0.0
        }
        for day, costs in self.costs.items():
            if day == "total":
                continue
            for key in aggregated.keys():
                aggregated[key] += costs.get(key, 0.0)
        aggregated["total_cost"] = sum(aggregated.values())
        self.costs["total"] = aggregated
        return aggregated

    def __str__(self):
        weight_util, volume_util = self.get_capacity_utilization()
        
        # Format route as string of destinations
        route_str = ' -> '.join(leg.destination for leg in self.route) if self.route else 'Not planned'
        
        return (f"Vehicle(ID: {self.vehicle_id}, Type: {self.vehicle_type}\n"
                f"  Current Load: {self.current_load_weight:.2f}/{self.max_weight:.2f} kg ({weight_util:.1f}%)\n"
                f"  Current Volume: {self.current_load_length:.2f}/{self.max_cbm:.2f} m³ ({volume_util:.1f}%)\n"
                f"  Shipments: {len(self.shipments)}\n"
                f"  Route: {route_str}")
    
    def __repr__(self):
        return self.__str__()
