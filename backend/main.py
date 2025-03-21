from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from typing import List

from models import DetectionEvent
from schemas import DetectionEventCreate, DetectionEvent as DetectionEventSchema

class Settings(BaseSettings):
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()

# Create SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

@app.get("/")
async def root():
    return {"message": "Welcome to the Geospatial API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detections/", response_model=DetectionEventSchema)
async def create_detection(
    detection: DetectionEventCreate,
    db: SessionLocal = Depends(get_db)
):
    # Create a Point geometry from latitude and longitude
    point = Point(detection.longitude, detection.latitude)
    geometry = from_shape(point, srid=4326)
    
    # Create database model
    db_detection = DetectionEvent(
        detection_type=detection.detection_type,
        location=geometry
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
    db: SessionLocal = Depends(get_db)
):
    detections = db.query(DetectionEvent).offset(skip).limit(limit).all()
    return detections 