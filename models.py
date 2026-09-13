"""
Models & Schemas
FlyRank Capstone: AI Image Understanding & Content Matching Engine
Author: Deepak R
"""

import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./image_matching.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# -----------------------------------------------------------------------------
# SQLAlchemy Persistence Models
# -----------------------------------------------------------------------------
class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(String(50), primary_key=True)
    filename = Column(String(255), nullable=False)
    image_url = Column(String(500), nullable=False)
    subject = Column(String(100), nullable=True, index=True)
    category = Column(String(100), nullable=True, index=True)
    attributes = Column(JSON, nullable=True)
    caption = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    is_flagged_low_confidence = Column(Boolean, default=False)
    embedding = Column(JSON, nullable=True)  # Stored float vector
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PostRecord(Base):
    __tablename__ = "posts"

    id = Column(String(50), primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    target_subject = Column(String(100), nullable=False)
    target_category = Column(String(100), nullable=False)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ReviewRecord(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    post_id = Column(String(50), nullable=False, index=True)
    image_id = Column(String(50), nullable=False, index=True)
    decision = Column(String(20), nullable=False)  # "approved", "rejected"
    reviewer_notes = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=False)
    guard_passed = Column(Boolean, nullable=False)
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class CostRecord(Base):
    __tablename__ = "cost_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False)  # "vision_classification", "embedding_generation"
    target_id = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


# -----------------------------------------------------------------------------
# Pydantic Schemas (Boundary Validation & Structured Vision Output)
# -----------------------------------------------------------------------------
class ImageVisionMetadata(BaseModel):
    subject: str = Field(..., min_length=2, description="Primary subject detected")
    category: str = Field(..., min_length=2, description="High-level category, e.g. animal, vehicle")
    attributes: List[str] = Field(default_factory=list, description="Visual descriptors")
    caption: str = Field(..., min_length=5, description="Full natural language description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")


class ImageResponse(BaseModel):
    id: str
    filename: str
    image_url: str
    subject: Optional[str]
    category: Optional[str]
    attributes: Optional[List[str]]
    caption: Optional[str]
    confidence: float
    is_flagged_low_confidence: bool

    class Config:
        from_attributes = True


class PostCreate(BaseModel):
    id: str
    title: str
    content: str
    target_subject: str
    target_category: str


class MatchCandidate(BaseModel):
    image_id: str
    subject: str
    category: str
    caption: str
    similarity_score: float
    confidence: float
    guard_status: str  # "ACCEPTED", "REJECTED"
    rejection_reason: Optional[str] = None


class MatchResponse(BaseModel):
    post_id: str
    post_title: str
    status: str  # "MATCH_FOUND", "NO_CONFIDENT_MATCH"
    reason: Optional[str] = None
    suggested_image: Optional[MatchCandidate] = None
    all_candidates: List[MatchCandidate] = []


class ReviewCreate(BaseModel):
    post_id: str
    image_id: str
    decision: str = Field(..., pattern="^(approved|rejected)$")
    reviewer_notes: Optional[str] = None
