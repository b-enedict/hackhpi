from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Force using SQLite
    DATABASE_URL: str = "sqlite:///./geospatial.db"

    class Config:
        env_file = ".env"

settings = Settings()

# Force SQLite dialect explicitly
engine = create_engine(
    "sqlite:///./geospatial.db", 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() 