from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

class RouteLeg:
    DEFAULT_RULES = {
        'max_driving_time': 8.5,    # Maximum continuous driving hours
        'rest_time': 45,            # Minutes of rest required
        'max_daily_driving': 10,     # Maximum daily driving hours
        'daily_rest': 11,           # Daily rest hours required
        'border_crossing_time': 240, # Minutes for border crossing
        'loading_time': 60,         # Minutes for loading/unloading
        'avg_speed_city': 40,       # km/h in city
        'avg_speed_highway': 70,    # km/h on highway
        'night_driving_allowed': False,  # Whether night driving is permitted
        'working_hours': {
            'start': '06:00',
            'end': '20:00'
        }
    }

    def __init__(self, 
                 destination: str,
                 arrival: Optional[str],
                 time_stayed: Optional[int],
                 departure: Optional[str],
                 time_travelled: int,
                 time_rested: int,
                 distance_travelled: float,
                 refuel_count: int,
                 is_border_crossing: bool = False,
                 from_country: str = None,
                 to_country: str = None):
        """
        Initialize a RouteLeg.

        Parameters:
        - destination (str): Destination node name.
        - arrival (str or None): Arrival time in 'YYYY/M/D - HH:MM' format or None.
        - time_stayed (int or None): Time stayed in minutes or None.
        - departure (str or None): Departure time in 'YYYY/M/D - HH:MM' format or None.
        - time_travelled (int): Time traveled in minutes.
        - time_rested (int): Time rested in minutes.
        - distance_travelled (float): Distance traveled in kilometers.
        - refuel_count (int): Number of refuels during this leg.
        """
        self.destination = destination
        self.arrival = self.parse_time(arrival) if arrival else None
        self.time_stayed = time_stayed if time_stayed is not None else 0
        self.departure = self.parse_time(departure) if departure else None
        self.time_travelled = time_travelled
        self.time_rested = time_rested
        self.distance_travelled = distance_travelled
        self.refuel_count = refuel_count
        self.is_border_crossing = is_border_crossing
        self.from_country = from_country
        self.to_country = to_country

    @staticmethod
    def calculate_refuel_stops(distance: float, 
                             fuel_efficiency: float = 0.3,  # L/km
                             tank_capacity: float = 500.0   # L
                             ) -> Dict[str, float]:
        """
        Calculate number of refuel stops needed.
        
        Parameters:
        - distance: Distance in kilometers
        - fuel_efficiency: Kilometers per liter
        - tank_capacity: Fuel tank capacity in liters
        
        Returns:
        - Number of refuel stops needed
        """
        range_per_tank = tank_capacity / fuel_efficiency
        total_fuel_needed = distance * fuel_efficiency
        refuel_stops = max(0, int((distance - range_per_tank) / range_per_tank) + 1)
        
        return {
            'stops': refuel_stops,
            'total_fuel_needed': total_fuel_needed,
            'fuel_per_stop': tank_capacity if refuel_stops > 0 else total_fuel_needed
        }
    
    @staticmethod
    def calculate_working_periods(total_hours: float, rules: dict) -> Dict[str, int]:
        """
        Calculate number of working periods needed considering working hours restrictions.
        """
        work_start = datetime.strptime(rules['working_hours']['start'], '%H:%M')
        work_end = datetime.strptime(rules['working_hours']['end'], '%H:%M')
        working_hours_per_day = (work_end - work_start).seconds / 3600
        
        # Account for rest periods in daily working hours
        effective_working_hours = min(
            working_hours_per_day - 1,  # Account for breaks
            rules['max_daily_driving']
        )
        
        days_needed = max(1, int((total_hours / effective_working_hours) + 0.5))
        
        return {
            'days': days_needed,
            'effective_hours_per_day': effective_working_hours
        }

    def calculate_arrival_time(self, start_time: datetime) -> datetime:
        """
        Calculate expected arrival time based on travel and rest times.
        """
        if not start_time:
            return None
            
        travel_info = self.calculate_travel_time(
            self.distance_travelled,
            self.is_border_crossing
        )
        
        total_minutes = (
            travel_info['total_travel_time'] +
            travel_info['total_rest_time'] +
            travel_info['border_time'] +
            travel_info['refuel_time']
        )
        
        # Account for working hours restrictions
        working_periods = self.calculate_working_periods(
            total_minutes / 60,
            self.DEFAULT_RULES
        )
        
        return start_time + timedelta(days=working_periods['days'])

    @staticmethod
    def calculate_travel_time(
            distance: float,
            is_border_crossing: bool = False,
            highway_ratio: float = 0.8,  # Ratio of distance on highways
            rules: dict = None
            ) -> Dict[str, any]:
        """
        Calculate travel time including mandatory rest periods and refueling.
        
        Parameters:
        - distance: Distance in kilometers
        - avg_speed: Average speed in km/h
        - refuel_time: Time needed for refueling in minutes
        - rest_rules: Dictionary containing rest rules
            {
                'max_driving_time': 4,  # Maximum continuous driving hours
                'rest_time': 45,        # Minutes of rest required
                'max_daily_driving': 9, # Maximum daily driving hours
                'daily_rest': 11        # Daily rest hours required
            }
        
        Returns:
        - Tuple of (travel_time_minutes, rest_time_minutes)
        """
        rules = rules or RouteLeg.DEFAULT_RULES
        
        # Calculate base travel time considering road types
        highway_distance = distance * highway_ratio
        city_distance = distance * (1 - highway_ratio)
        
        highway_time = highway_distance / rules['avg_speed_highway']
        city_time = city_distance / rules['avg_speed_city']
        base_travel_time = highway_time + city_time
        
        # Calculate working periods needed
        working_hours = RouteLeg.calculate_working_periods(
            total_hours=base_travel_time,
            rules=rules
        )
        
        # Calculate rest periods
        rest_periods = int(base_travel_time / rules['max_driving_time'])
        rest_time = rest_periods * rules['rest_time']
        
        # Add border crossing time if applicable
        border_time = rules['border_crossing_time'] if is_border_crossing else 0
        
        # Calculate refueling needs
        refuel_info = RouteLeg.calculate_refuel_stops(distance)
        refuel_time = refuel_info['stops'] * 30  # 30 minutes per refuel stop
        
        return {
            'total_travel_time': int(base_travel_time * 60),  # minutes
            'total_rest_time': rest_time + (working_hours['days'] - 1) * rules['daily_rest'] * 60,
            'working_days': working_hours['days'],
            'border_time': border_time,
            'refuel_time': refuel_time,
            'breakdown': {
                'highway_time': highway_time * 60,
                'city_time': city_time * 60,
                'rest_periods': rest_periods,
                'refuel_stops': refuel_info['stops']
            }
        }

    def parse_time(self, time_str):
        """
        Parse a time string into a datetime object.

        Parameters:
        - time_str (str): Time string in 'YYYY/M/D - HH:MM' format.

        Returns:
        - datetime: Parsed datetime object.
        """
        return datetime.strptime(time_str, "%Y/%m/%d - %H:%M")

    def __str__(self):
        return (
            f"RouteLeg(\n"
            f"  Destination: {self.destination}\n"
            f"  Distance: {self.distance_travelled:.1f} km\n"
            f"  Travel Time: {self.time_travelled} mins\n"
            f"  Rest Time: {self.time_rested} mins\n"
            f"  Refuels: {self.refuel_count}\n"
            f"  Border Crossing: {self.is_border_crossing}\n"
            f"  Countries: {self.from_country} -> {self.to_country}\n"
            f"  Arrival: {self.arrival}\n"
            f"  Departure: {self.departure}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()
