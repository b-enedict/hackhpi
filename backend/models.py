from sqlalchemy import Column, Integer, String, DateTime, func
from geoalchemy2 import Geometry
from main import Base

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    detection_type = Column(String, nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False) 