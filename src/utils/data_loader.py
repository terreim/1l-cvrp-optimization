# utils/data_loader.py

import json
from src.models.node import Node
from src.models.edge import Edge
from src.models.vehicle import Vehicle
from src.models.shipment import Shipment

def load_json(filepath):
    """
    Load JSON data from a file.

    Parameters:
    - filepath (str): Path to the JSON file.

    Returns:
    - dict: Parsed JSON data.
    """
    with open(filepath, 'r') as file:
        return json.load(file)

def parse_nodes(nodes_data):
    """
    Parse nodes from JSON data.

    Parameters:
    - nodes_data (dict): JSON data containing nodes.

    Returns:
    - dict: Dictionary of Node objects keyed by node_id.
    """
    nodes = {}
    
    # Parse depot nodes
    for depot in nodes_data['locations']['depots']:
        node = Node(
            node_id=depot['id'],
            name=depot['name'],
            country=depot['country'],
            node_type='depot',
            operating_hours=depot['operating_hours']
        )
        nodes[depot['id']] = node
    
    # Parse border crossing nodes
    for border in nodes_data['locations']['border_crossings']:
        # For border crossings, use the first country as the primary country
        node = Node(
            node_id=border['id'],
            name=border['name'],
            country=border['countries'][0],  # Use first country
            node_type='border_crossing',
            operating_hours=border['operating_hours']
        )
        nodes[border['id']] = node
    
    return nodes

def parse_vehicles(vehicles_data, packing_data=None):
    """
    Parse vehicles from JSON data and optionally load their shipments.

    Parameters:
    - vehicles_data (dict): JSON data containing vehicles.
    - packing_data (dict): Optional JSON data containing packing plans.

    Returns:
    - list: List of Vehicle objects.
    """
    vehicles = []
    # Create vehicle objects
    for veh in vehicles_data['fleet']:
        vehicle = Vehicle(
            vehicle_id=veh['id'],
            vehicle_type=veh['type'],
            dimensions=veh['dimensions'],
            max_weight=veh['max_weight'],
            fuel_capacity=veh['fuel_capacity'],
            fuel_efficiency=veh['fuel_efficiency']
        )
        vehicles.append(vehicle)
    
    # If packing data is provided, load the shipments
    if packing_data:
        vehicle_map = {v.vehicle_id: v for v in vehicles}
        for veh_data in packing_data['vehicles']:
            if veh_data['id'] in vehicle_map:
                vehicle = vehicle_map[veh_data['id']]
                for shipment_data in veh_data['shipments']:
                    shipment = Shipment(
                        shipment_id=shipment_data['id'],
                        order_id=shipment_data['order_id'],
                        total_cbm=shipment_data['total_cbm'],
                        weight=shipment_data['weight'],
                        origin=shipment_data['origin'],
                        delivery_location_id=shipment_data['delivery']['location_id'],
                        price=shipment_data['price']
                    )
                    vehicle.add_shipment(shipment)
    
    return vehicles

def parse_edges(edges_data):
    """
    Parse edges from JSON data.

    Parameters:
    - edges_data (dict): JSON data containing edges.

    Returns:
    - dict: Dictionary of Edge objects keyed by standardized edge ID.
    """
    edges = {}
    for country, data in edges_data['countries'].items():
        for route_name, route_info in data['routes'].items():
            node1, node2 = route_name.split('-')
            edge = Edge(
                from_node=node1,
                to_node=node2,
                distance=route_info['distance'],
                base_time=route_info['base_time'],
                road_type=route_info['road_type'],
                country_time_windows=data['time_windows']
            )
            edges[edge.edge_id] = edge
    return edges

def parse_packing_plan(packing_data):
    """
    Parse packing plan from JSON data into simplified 1D shipments.

    Parameters:
    - packing_data (dict): JSON data containing packing plans.

    Returns:
    - list: List of simplified Shipment objects.
    """
    shipments = []
    for vehicle in packing_data['vehicles']:
        for shipment in vehicle['shipments']:
            shipment_obj = Shipment(
                shipment_id=shipment['id'],
                order_id=shipment['order_id'],
                total_cbm=shipment['total_cbm'],  # Already calculated in the JSON
                weight=shipment['weight'],
                origin=shipment['origin'],
                delivery_location_id=shipment['delivery']['location_id'],
                price=shipment['price']
            )
            shipments.append(shipment_obj)
    return shipments
