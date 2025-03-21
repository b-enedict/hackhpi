from pydantic import BaseModel
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

class SensorData(BaseModel):
    __root__: List[SensorDataPoint] 