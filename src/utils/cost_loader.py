def assign_costs_to_vehicles(costs_data, vehicles):
    """
    Assign cost data to each vehicle.

    Parameters:
    - costs_data (dict): JSON data containing costs.
    - vehicles (list): List of Vehicle objects.

    Returns:
    - list: Updated list of Vehicle objects with costs assigned.
    """
    vehicle_dict = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    for vehicle_id, cost_info in costs_data.items():
        vehicle = vehicle_dict.get(vehicle_id)
        if not vehicle:
            print(f"Warning: Vehicle ID {vehicle_id} not found in vehicles list.")
            continue
        for date, costs in cost_info.items():
            if date == "total":
                vehicle.total_costs = costs
            else:
                vehicle.add_daily_cost(
                    date=date,
                    per_diem=costs.get("per_diem", 0.0),
                    driver_salary=costs.get("driver_salary", 0.0),
                    fuel_cost=costs.get("fuel_cost", 0.0),
                    custom_fee=costs.get("custom_fee", 0.0),
                    tax_on_goods=costs.get("tax_on_goods", 0.0),
                    overhead=costs.get("overhead", 0.0),
                    emergency=costs.get("emergency", 0.0)
                )
    return vehicles
