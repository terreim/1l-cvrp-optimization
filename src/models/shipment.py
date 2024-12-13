class Shipment:
    def __init__(self, shipment_id, order_id, total_cbm, weight, origin, delivery_location_id, price):
        """
        Initialize a Shipment.

        Parameters:
        - shipment_id (str): Unique identifier for the shipment.
        - order_id (str): Associated order ID.
        - total_cbm (float): Total cubic meters (used as length in 1D packing).
        - weight (float): Weight of the shipment in kilograms.
        - origin (str): Origin location ID.
        - delivery_location_id (str): Destination location ID.
        - price (float): Value of the shipment.
        """
        self.shipment_id = shipment_id
        self.order_id = order_id
        self.total_cbm = total_cbm
        self.weight = weight
        self.origin = origin
        self.delivery_location_id = delivery_location_id
        self.price = price

    def __str__(self):
        return (f"Shipment(ID: {self.shipment_id}, Order: {self.order_id}, "
                f"Weight: {self.weight} kg, CBM: {self.total_cbm} m³, "
                f"Destination: {self.delivery_location_id})")