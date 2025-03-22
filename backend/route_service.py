import requests
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import json

class RouteService:
    def __init__(self):
        self.api_key = os.getenv('AWS_LOCATION_SERVICE_API_KEY')
        if not self.api_key:
            raise ValueError("AWS_LOCATION_SERVICE_API_KEY environment variable is not set")
        
        print(f"API Key length: {len(self.api_key)}")  # Log API key length (not the key itself)
        self.base_url = "https://routes.geo.us-west-2.amazonaws.com"
        self.route_calculator_name = "roucalc"  # Name of your route calculator in AWS Location Service

    def calculate_route(
        self,
        departure_position: Tuple[float, float],
        destination_position: Tuple[float, float],
        avoid_areas: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate a route between two points using AWS Location Service.
        
        Args:
            departure_position: Tuple of (longitude, latitude) for departure point
            destination_position: Tuple of (longitude, latitude) for destination point
            avoid_areas: List of areas to avoid (e.g., areas with stairs)
            
        Returns:
            Dictionary containing route information including:
            - distance: Total distance in meters
            - duration: Estimated duration in seconds
            - geometry: Encoded polyline of the route
            - legs: List of route legs with detailed information
        """
        try:
            # Prepare the request parameters
            params = {
                'CalculatorName': self.route_calculator_name,
                'DeparturePosition': departure_position,
                'DestinationPosition': destination_position,
                'TravelMode': 'Walking',  # Can be changed to 'Car', 'Truck', etc.
                'IncludeLegGeometry': True
            }

            # Add avoid areas if provided
            if avoid_areas:
                params['AvoidAreas'] = avoid_areas

            # Make the API request with API key as URL parameter
            headers = {
                'Content-Type': 'application/json'
            }

            url = f"{self.base_url}/routes/v0/calculators/{self.route_calculator_name}/calculate/route?key={self.api_key}"
            
            print(f"Making request to: {url}")
            print(f"Params: {params}")

            response = requests.post(
                url,
                json=params,
                headers=headers
            )

            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")

            if response.status_code != 200:
                raise Exception(f"AWS Location Service API error: {response.text}")

            response_data = response.json()
            
            # Check if the response contains the expected data
            if 'Legs' not in response_data or 'Summary' not in response_data:
                raise Exception(f"Unexpected response format: {response_data}")

            # Extract the first leg (we only support single-leg routes for now)
            leg = response_data['Legs'][0]
            summary = response_data['Summary']

            # Create the route info with the required schema
            route_info = {
                'distance': float(summary['Distance']),
                'duration': int(summary['DurationSeconds']),
                'geometry': json.dumps(leg['Geometry']['LineString']),  # Convert to JSON string
                'legs': [{
                    'start_position': leg['StartPosition'],
                    'end_position': leg['EndPosition'],
                    'distance': float(leg['Distance']),
                    'duration': int(leg['DurationSeconds']),
                    'geometry': json.dumps(leg['Geometry']['LineString']),  # Convert to JSON string
                    'steps': leg['Steps']
                }],
                'summary': {
                    'distance': float(summary['Distance']),
                    'duration': int(summary['DurationSeconds']),
                    'departure_time': datetime.now().isoformat(),
                    'distance_unit': summary['DistanceUnit'],
                    'data_source': summary['DataSource']
                }
            }

            return route_info

        except Exception as e:
            print(f"Error calculating route: {str(e)}")
            raise

    def find_alternative_route(
        self,
        departure_position: Tuple[float, float],
        destination_position: Tuple[float, float],
        avoid_areas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find an alternative route that avoids specified areas (e.g., stairs).
        
        Args:
            departure_position: Tuple of (longitude, latitude) for departure point
            destination_position: Tuple of (longitude, latitude) for destination point
            avoid_areas: List of areas to avoid
            
        Returns:
            Dictionary containing alternative route information
        """
        try:
            # Calculate route with avoid areas
            route_info = self.calculate_route(
                departure_position=departure_position,
                destination_position=destination_position,
                avoid_areas=avoid_areas
            )

            return route_info

        except Exception as e:
            print(f"Error finding alternative route: {str(e)}")
            raise

    def decode_polyline(self, encoded_polyline: str) -> List[Tuple[float, float]]:
        """
        Decode an encoded polyline string into a list of coordinates.
        
        Args:
            encoded_polyline: Encoded polyline string from AWS Location Service
            
        Returns:
            List of (longitude, latitude) coordinate tuples
        """
        # Implementation of polyline decoding
        # This is a placeholder - you'll need to implement the actual decoding logic
        pass 