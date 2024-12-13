# utils/time_calculations.py

from datetime import datetime, timedelta

def plan_route(destinations, nodes):
    """
    Plan a route based on delivery destinations.
    For simplicity, use a random order or a nearest neighbor heuristic.
    
    Parameters:
    - destinations (set): Set of destination node IDs.
    - nodes (dict): Dictionary of Node objects.
    
    Returns:
    - list: Ordered list of node IDs representing the route.
    """
    # Placeholder for actual route planning logic
    # For demonstration, return destinations in random order
    return list(destinations)

def calculate_travel_time(edge):
    """
    Calculate fuzzy travel time based on edge's fuzzy_travel_time.
    
    Parameters:
    - edge (Edge): The edge object.
    
    Returns:
    - TriangularFuzzyNumber: Fuzzy travel time.
    """
    return edge.fuzzy_travel_time

def parse_time(time_str):
    """
    Parse a time string into a datetime object.
    
    Parameters:
    - time_str (str): Time string in 'YYYY/MM/DD - HH:MM' format.
    
    Returns:
    - datetime: Parsed datetime object.
    """
    return datetime.strptime(time_str, "%Y/%m/%d - %H:%M")
