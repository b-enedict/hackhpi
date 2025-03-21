from fastapi import FastAPI, Depends, HTTPException
from database import SessionLocal, Base, engine
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json

from models import DetectionEvent
from schemas import DetectionEventCreate, DetectionEvent as DetectionEventSchema

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