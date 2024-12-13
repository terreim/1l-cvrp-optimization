# models/node.py

class Node:
    def __init__(self, node_id, name, country, node_type, operating_hours):
        """
        Initialize a Node.

        Parameters:
        - node_id (str): Unique identifier for the node.
        - name (str): Name of the location.
        - country (str): Country where the node is located.
        - node_type (str): Type of the node (e.g., 'depot', 'border_crossing').
        - operating_hours (dict): Operating hours with 'start' and 'end' times.
                                   Example: {'start': '06:00', 'end': '22:00'}
        """
        self.node_id = node_id
        self.name = name
        self.country = country
        self.node_type = node_type
        self.operating_hours = operating_hours  # {'start': 'HH:MM', 'end': 'HH:MM'}

    def __str__(self):
        return f"Node(ID: {self.node_id}, Name: {self.name}, Country: {self.country}, Type: {self.node_type})"

    def __repr__(self):
        return self.__str__()
