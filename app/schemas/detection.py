"""
KAVACH AI Pro - Detection Schemas
Pydantic models for detection request/response validation
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DetectionTypeEnum(str, Enum):
    DEEPFAKE = "deepfake"
    VOICE = "voice"
    PHISHING = "phishing"


class DetectionStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhishingRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    content: Optional[str] = Field(None, max_length=10000)


class DetectionSubmitResponse(BaseModel):
    success: bool = True
    detection_id: str
    status: str = "processing"
    message: str = "Analysis started"


class DeepfakeResultResponse(BaseModel):
    is_fake: bool
    confidence: float
    frames_analyzed: int = 0
    faces_detected: int = 0
    face_consistency_score: Optional[float] = None
    edge_artifact_score: Optional[float] = None
    compression_anomaly_score: Optional[float] = None
    explanation: str
    processing_time: float
    recommended_action: str


class VoiceResultResponse(BaseModel):
    is_fake: bool
    confidence: float
    spoof_type: Optional[str] = None
    spectral_centroid_mean: Optional[float] = None
    spectral_flatness: Optional[float] = None
    mfcc_variance: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    explanation: str
    processing_time: float
    recommended_action: str


class PhishingResultResponse(BaseModel):
    is_phishing: bool
    confidence: float
    url: str
    threats: List[str] = []
    url_entropy: Optional[float] = None
    domain_analysis: Optional[Dict[str, Any]] = None
    explanation: str
    recommended_action: str


class DetectionResponse(BaseModel):
    id: str
    detection_type: str
    status: str
    original_filename: Optional[str] = None
    is_fake: Optional[bool] = None
    confidence: Optional[float] = None
    spoof_type: Optional[str] = None
    explanation: Optional[str] = None
    recommended_action: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DetectionHistoryResponse(BaseModel):
    items: List[DetectionResponse]
    total: int
    page: int
    per_page: int
    has_next: bool


class SystemStatsResponse(BaseModel):
    total_users: int
    total_detections: int
    total_threats: int
    safe_content: int
    detection_rate: float
    detections_by_type: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


class AdminUserUpdate(BaseModel):
    role: str = Field(..., pattern=r'^(admin|user|premium)$')
