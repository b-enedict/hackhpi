from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from database import SessionLocal, Base, engine
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Tuple
import json
import requests
from statistics import mean
import asyncio
import math
from inference import acceleration_data

from models import DetectionEvent
from schemas import DetectionEventCreate, DetectionEvent as DetectionEventSchema, SensorDataPoint

from pydantic import BaseModel
import boto3

from models import DetectionEvent
from schemas import DetectionEventCreate, DetectionEvent as DetectionEventSchema
# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Geospatial API")

# Function to calculate Euclidean distance between two points (in meters)
def calculate_distance(lat1, lon1, lat2, lon2):
    # Earth radius in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula for distance
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_detection_record(
        db: Session, 
        detection_type: str,
        latitude: float,
        longitude: float
    ) -> DetectionEvent:
    db_detection = DetectionEvent(
        detection_type=detection_type,
        latitude=latitude,
        longitude=longitude
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    return db_detection

@app.get("/")
async def root():
    return {"message": "Welcome to the Geospatial API"}

@app.get("/debug/database", response_model=List[Dict[str, Any]])
async def debug_database(db: Session = Depends(get_db)):
    """Print all database contents for debugging purposes"""
    detections = db.query(DetectionEvent).all()
    result = []
    for detection in detections:
        result.append({
            "id": detection.id,
            "detection_type": detection.detection_type,
            "label": detection.label,
            "latitude": detection.latitude,
            "longitude": detection.longitude,
            "total_count": detection.total_count,
            "stairs_count": detection.stairs_count,
            "ratio": detection.stairs_count / detection.total_count if detection.total_count > 0 else 0,
            "timestamp": str(detection.timestamp)
        })
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def process_with_ml_model(sensor_data: List[SensorDataPoint], db: Session):
    """
    Process sensor data with ML model and store predictions in the database.
    This runs asynchronously after the API has already responded.
    """
    # TODO: Load your ML model (consider singleton pattern or global instance)
    # model = load_model()  # Your model loading code
    
    # Simulate ML model processing and getting a detection interval with label
    # In a real implementation, this would use your actual ML model

    acceleration_data(acceleration_data: List[Tuple[float, float, float, int]])
    
    
    
    # Find GPS coordinates within the detection time interval
    locations_in_interval = [
        point.location for point in sensor_data 
        if intervall_start_time <= point.timestamp <= intervall_end_time
    ]
    
    if locations_in_interval:
        # Calculate the middle GPS position
        avg_latitude = mean([loc.latitude for loc in locations_in_interval])
        avg_longitude = mean([loc.longitude for loc in locations_in_interval])
        
        # Check if there's a nearby detection within 20 meters
        existing_detections = db.query(DetectionEvent).all()
        nearest_detection = None
        min_distance = float('inf')
        
        for detection in existing_detections:
            distance = calculate_distance(
                detection.latitude, detection.longitude,
                avg_latitude, avg_longitude
            )
            
            if distance < 20 and distance < min_distance:  # Within 20 meters
                min_distance = distance
                nearest_detection = detection
        
        if nearest_detection:
            # Update existing detection
            nearest_detection.total_count += 1
            
            if label == "stairs":
                nearest_detection.stairs_count += 1
            
            # Update the GPS location as a weighted average based on total_count
            # This ensures the location is as close as possible to all detections
            weight_old = (nearest_detection.total_count - 1) / nearest_detection.total_count
            weight_new = 1 / nearest_detection.total_count
            
            nearest_detection.latitude = (nearest_detection.latitude * weight_old) + (avg_latitude * weight_new)
            nearest_detection.longitude = (nearest_detection.longitude * weight_old) + (avg_longitude * weight_new)
            
            db.commit()
            print(f"Updated detection ID {nearest_detection.id}: {label} at ({avg_latitude}, {avg_longitude})")
            print(f"New counts - Total: {nearest_detection.total_count}, Stairs: {nearest_detection.stairs_count}")
            
        else:
            # Create a new detection
            db_detection = DetectionEvent(
                detection_type=label,
                latitude=avg_latitude,
                longitude=avg_longitude,
                label=label,
                total_count=1,
                stairs_count=1 if label == "stairs" else 0
            )
            db.add(db_detection)
            db.commit()
            
            print(f"New detection stored: {label} at ({avg_latitude}, {avg_longitude})")
    else:
        print("No GPS coordinates found within the detection interval")


@app.post("/process-sensor-data/")
async def process_sensor_data(
    sensor_data: List[SensorDataPoint],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Receive sensor data, store it immediately, and process with ML model asynchronously.
    """
    # Acknowledge receipt of all data points
    received_count = len(sensor_data)
    
    # Schedule background processing with ML model
    background_tasks.add_task(process_with_ml_model, sensor_data, db)

    return {
        "message": f"Received {received_count} data points for processing",
        "status": "processing_scheduled"
    }

@app.post("/detections/", response_model=DetectionEventSchema)
async def create_detection(
    detection: DetectionEventCreate,
    db: Session = Depends(get_db)
):
    # Create database model with direct lat/lng values
    db_detection = DetectionEvent(
        detection_type=detection.detection_type,
        latitude=detection.latitude,
        longitude=detection.longitude,
        label=detection.label if hasattr(detection, 'label') else None,
        total_count=1,
        stairs_count=1 if detection.label == "stairs" else 0
    )
    
    # Add to database
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    
    return db_detection

@app.get("/detections/", response_model=List[DetectionEventSchema])
async def get_detections(
    skip: int = 0,
    limit: int = 100,
    min_stairs_ratio: float = 0.95,
    db: Session = Depends(get_db)
):
    # Get all detections first
    all_detections = db.query(DetectionEvent).offset(skip).limit(limit).all()
    
    # Filter for detections where stairs_count/total_count > min_stairs_ratio (default 0.95)
    filtered_detections = [
        detection for detection in all_detections
        if detection.total_count > 0 and 
           (detection.stairs_count / detection.total_count) > min_stairs_ratio
    ]
    
    return filtered_detections 

# ---------------------------
# AWS Location route calculator integration
# ---------------------------

class RouteRequest(BaseModel):
    departure: List[float]      # [longitude, latitude]
    destination: List[float]    # [longitude, latitude]

def calculate_route_avoiding_markers(departure: list, destination: list, avoid_markers: list, buffer_distance: float = 0.001):
    """
    Calculates a route between departure and destination using AWS Location Service.
    Avoids specified markers by placing offset waypoints near each marker.
    """
    # Prepare intermediate waypoints by offsetting the marker coordinates.
    waypoints = []
    for marker in avoid_markers:
        # Offset longitude and latitude to steer around the marker.
        waypoint = [marker[0] + buffer_distance, marker[1] + buffer_distance]
        waypoints.append(waypoint)
    
    client = boto3.client("location", region_name="us-west-2")
    try:
        response = client.calculate_route(
            CalculatorName="YourRouteCalculatorName",  # Replace with your route calculator name
            DeparturePosition=departure,
            DestinationPosition=destination,
            WaypointPositions=waypoints,
            DistanceUnit="Kilometers",
            TravelMode="Car"
        )
    except Exception as e:
        raise e
    
    return response

@app.post("/calculate-route")
async def calculate_route(req: RouteRequest, db: Session = Depends(get_db)):
    try:
        # Fetch all detection events from the database and prepare avoidance markers
        detection_events = db.query(DetectionEvent).all()
        # Convert each detection event into a marker [longitude, latitude]
        avoid_markers = [[d.longitude, d.latitude] for d in detection_events]
        
        route_response = calculate_route_avoiding_markers(
            departure=req.departure,
            destination=req.destination,
            avoid_markers=avoid_markers
        )
        return route_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))