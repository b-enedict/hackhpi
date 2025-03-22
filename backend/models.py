from sqlalchemy import Column, Integer, String, DateTime, func, Float
from database import Base

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    detection_type = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    label = Column(String, nullable=True)
    total_count = Column(Integer, default=1, nullable=False)
    stairs_count = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)