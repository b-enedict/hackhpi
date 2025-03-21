from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from database import SessionLocal, Base, engine
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Tuple
import json
import requests
from statistics import mean
import asyncio

from models import DetectionEvent
from schemas import DetectionEventCreate, DetectionEvent as DetectionEventSchema, SensorDataPoint

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Geospatial API")

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
            "latitude": detection.latitude,
            "longitude": detection.longitude,
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
    def run_ml_model(acceleration_data: List[Tuple[float, float, float, int]]) -> Dict:
        # This is a placeholder for your actual ML model logic
        # In reality, your model would analyze the acceleration data and return predictions
        
        # Simulating a detection - in your real implementation, this would be model output
        # Returns a dict with detected event time interval and label
        return {
            "label": "stairs",  # The type of event detected
            "intervall_start_time": acceleration_data[5][3],  # Start timestamp
            "intervall_end_time": acceleration_data[15][3],   # End timestamp
        }
    
    # Extract acceleration data from sensor data
    acceleration_data = [(
        point.linearAcceleration.x,
        point.linearAcceleration.y, 
        point.linearAcceleration.z,
        point.timestamp
    ) for point in sensor_data]
    
    # Process with ML model - in reality this would be your model prediction
    detection_result = run_ml_model(acceleration_data)
    

    # Extract detection time interval
    intervall_start_time = detection_result["intervall_start_time"]
    intervall_end_time = detection_result["intervall_end_time"]
    label = detection_result["label"]
    
    # Find GPS coordinates within the detection time interval
    locations_in_interval = [
        point.location for point in sensor_data 
        if intervall_start_time <= point.timestamp <= intervall_end_time
    ]
    
    if locations_in_interval:
        # Calculate the middle GPS position
        avg_latitude = mean([loc.latitude for loc in locations_in_interval])
        avg_longitude = mean([loc.longitude for loc in locations_in_interval])
        
        # Create a detection event
        detection_data = DetectionEventCreate(
            detection_type=label,
            latitude=avg_latitude,
            longitude=avg_longitude,
            label=label
        )
        
        # Store in database using the existing POST endpoint
        # In local mode, we can directly create it in the database
        db_detection = DetectionEvent(
            detection_type=detection_data.detection_type,
            latitude=detection_data.latitude,
            longitude=detection_data.longitude,
            label=detection_data.label
        )
        db.add(db_detection)
        db.commit()
        
        print(f"Detection stored: {label} at ({avg_latitude}, {avg_longitude})")
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
        longitude=detection.longitude
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
    db: Session = Depends(get_db)
):
    detections = db.query(DetectionEvent).offset(skip).limit(limit).all()
    return detections 