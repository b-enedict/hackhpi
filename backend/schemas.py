from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DetectionEventBase(BaseModel):
    detection_type: str
    latitude: float
    longitude: float

class DetectionEventCreate(DetectionEventBase):
    pass

class DetectionEvent(DetectionEventBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True 