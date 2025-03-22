from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from database import SessionLocal, Base, engine
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Tuple
import json
import requests
from statistics import mean
import asyncio
import math

# Try to import from inference, but handle the case when it's not available
from inference import acceleration_data

from models import DetectionEvent
from schemas import (
    DetectionEventCreate, 
    DetectionEvent as DetectionEventSchema, 
    SensorDataPoint,
    RouteRequest,
    RouteResponse
)
from route_service import RouteService

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Geospatial API")

# Initialize route service
route_service = RouteService()

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

    x_values = [point.linearAcceleration.x for point in sensor_data]
    y_values = [point.linearAcceleration.y for point in sensor_data]
    z_values = [point.linearAcceleration.z for point in sensor_data]
    
    # Get first timestamp from data
    start_time = sensor_data[0].timestamp if sensor_data else 0
    
    [labels, starts, ends] = acceleration_data(x_values, y_values, z_values, start_time)
    
    for i in range(len(labels)):
        label=labels[i]
        intervall_start_time = starts[i]
        intervall_end_time = ends[i]
    
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

@app.post("/calculate-route", response_model=RouteResponse)
async def calculate_route(request: RouteRequest):
    """
    Calculate a route between two points.
    
    Args:
        request: RouteRequest containing departure and destination positions
        
    Returns:
        RouteResponse containing route information
    """
    try:
        route_info = route_service.calculate_route(
            departure_position=request.departure_position,
            destination_position=request.destination_position,
            avoid_areas=request.avoid_areas
        )
        return route_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alternative-route", response_model=RouteResponse)
async def find_alternative_route(request: RouteRequest):
    """
    Find an alternative route that avoids specified areas.
    
    Args:
        request: RouteRequest containing departure, destination positions, and areas to avoid
        
    Returns:
        RouteResponse containing alternative route information
    """
    try:
        if not request.avoid_areas:
            raise HTTPException(
                status_code=400,
                detail="Avoid areas must be specified for alternative route calculation"
            )
            
        route_info = route_service.find_alternative_route(
            departure_position=request.departure_position,
            destination_position=request.destination_position,
            avoid_areas=request.avoid_areas
        )
        return route_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 