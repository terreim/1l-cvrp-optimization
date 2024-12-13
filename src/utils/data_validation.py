# utils/data_validation.py

def validate_data(nodes, edges, vehicles, shipments):
    """
    Validate the integrity and consistency of loaded data.

    Parameters:
    - nodes (dict): Dictionary of Node objects.
    - edges (list): List of Edge objects.
    - vehicles (list): List of Vehicle objects.
    - shipments (list): List of Shipment objects.

    Returns:
    - bool: True if all validations pass, False otherwise.
    """
    # Validate that all edges connect existing nodes
    for edge in edges:
        if edge.node_a not in nodes or edge.node_b not in nodes:
            print(f"Error: Edge connects to non-existent node(s): {edge.node_a}, {edge.node_b}")
            return False

    # Validate that all shipments have valid delivery locations
    for shipment in shipments:
        if shipment.delivery_location_id not in nodes:
            print(f"Error: Shipment {shipment.shipment_id} has invalid delivery location ID: {shipment.delivery_location_id}")
            return False

    # Validate that all vehicles have unique IDs
    vehicle_ids = [vehicle.vehicle_id for vehicle in vehicles]
    if len(vehicle_ids) != len(set(vehicle_ids)):
        print("Error: Duplicate vehicle IDs found.")
        return False

    # Additional validations as needed...

    print("All data validations passed.")
    return True
