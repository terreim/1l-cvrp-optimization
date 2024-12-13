# models/edge.py

from src.fuzzy.fuzzy_number import TriangularFuzzyNumber

class Edge:
    def __init__(self, from_node, to_node, distance, base_time, road_type, country_time_windows):
        """
        Initialize an Edge.

        Parameters:
        - from_node (str): ID of the starting node.
        - to_node (str): ID of the ending node.
        - distance (float): Distance between nodes in kilometers.
        - base_time (float): Base travel time in minutes.
        - road_type (str): Type of road (e.g., 'highway').
        - country_time_windows (list): List of time windows with delay factors.
                                        Example:
                                        [
                                            {
                                                "start_time": "06:00",
                                                "end_time": "09:00",
                                                "delay_factor": 1.3
                                            },
                                            ...
                                        ]
        """
        self.nodes = tuple(sorted([from_node, to_node]))
        self.distance = distance  # in km
        self.base_time = base_time  # in minutes
        self.road_type = road_type
        self.country_time_windows = country_time_windows
        self.fuzzy_travel_time = self.calculate_fuzzy_travel_time()

    @property
    def edge_id(self):
        """Generate a consistent edge identifier."""
        return f"{self.nodes[0]}-{self.nodes[1]}"

    def connects(self, node1, node2):
        """Check if this edge connects the given nodes (in either direction)."""
        return set([node1, node2]) == set(self.nodes)

    def __str__(self):
        return (f"Edge(Between: {self.nodes[0]} <-> {self.nodes[1]}, "
                f"Distance: {self.distance} km, Base Time: {self.base_time} mins, "
                f"Road Type: {self.road_type})")

    def calculate_fuzzy_travel_time(self):
        """
        Calculate fuzzy travel time based on time windows and delay factors.

        Returns:
        - TriangularFuzzyNumber: The fuzzy travel time.
        """
        min_time = self.base_time
        peak_time = self.base_time
        max_time = self.base_time

        for window in self.country_time_windows:
            delay_factor = window['delay_factor']
            delayed_time = self.base_time * delay_factor
            if delayed_time > max_time:
                max_time = delayed_time
            if delayed_time < min_time:
                min_time = delayed_time
            # For peak time, assuming typical delay factor
            if delay_factor == 1.3 or delay_factor == 1.4 or delay_factor == 1.2 or delay_factor == 1.5:
                peak_time = delayed_time

        return TriangularFuzzyNumber(left=min_time, peak=peak_time, right=max_time)
