from pydantic import BaseModel, RootModel   
from datetime import datetime
from typing import Optional, List

class DetectionEventBase(BaseModel):
    detection_type: str
    latitude: float
    longitude: float
    label: str

class DetectionEventCreate(DetectionEventBase):
    pass

class DetectionEvent(DetectionEventBase):
    id: int
    timestamp: datetime
    total_count: int = 1
    stairs_count: int = 0

    class Config:
        from_attributes = True

class Location(BaseModel):
    accuracy: float
    altitude: float
    latitude: float
    longitude: float
    speed: float

class LinearAcceleration(BaseModel):
    x: float
    y: float
    z: float

class SensorDataPoint(BaseModel):
    linearAcceleration: LinearAcceleration
    location: Location
    timestamp: int

class SensorData(RootModel):
    root: List[SensorDataPoint] 