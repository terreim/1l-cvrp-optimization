# utils/route_and_cost_parser.py

from src.models.route_leg import RouteLeg

def parse_completed_route(completed_route_data, vehicles):
    """
    Parse completed_route.json and assign routes to vehicles.

    Parameters:
    - completed_route_data (dict): JSON data containing completed routes.
    - vehicles (list): List of Vehicle objects.

    Returns:
    - list: Updated list of Vehicle objects with routes assigned.
    """
    vehicle_dict = {vehicle.vehicle_id: vehicle for vehicle in vehicles}

    for vehicle_id, route_info in completed_route_data.items():
        vehicle = vehicle_dict.get(vehicle_id)
        if not vehicle:
            print(f"Warning: Vehicle ID {vehicle_id} not found in vehicles list.")
            continue

        starting_point = route_info.get("starting_point")
        starting_time = route_info.get("starting_time")
        route_legs = route_info.get("route", [])
        totals = route_info.get("totals", {})

        # Optionally, store starting point and time if needed
        # For now, we'll focus on route legs

        for leg in route_legs:
            route_leg = RouteLeg(
                destination=leg.get("destination"),
                arrival=leg.get("arrival"),
                time_stayed=leg.get("time_stayed"),
                departure=leg.get("departure"),
                time_travelled=leg.get("time_travelled"),
                time_rested=leg.get("time_rested"),
                distance_travelled=leg.get("distance_travelled"),
                refuel_count=leg.get("refuel_count")
            )
            vehicle.add_route_leg(route_leg)

    return vehicles
