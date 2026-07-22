"""
NovaShield - Enterprise Configuration
Free & Open Source Stack
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, model_validator

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "NovaShield"
    VERSION: str = "3.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    PORT: int = 8005  # Backend server port (used in CSP headers)

    # Security
    SECRET_KEY: str = Field(default="change-this-in-production-min-32-chars-long")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database — PostgreSQL or SQLite fallback
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./novashield.db")

    # Redis (Free - for caching, rate limiting, queues)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # File Storage (Local - Free, or MinIO for S3-compatible)
    STORAGE_TYPE: str = "local"  # local, minio, s3
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # Enforce 50MB maximum upload limit for deepfake

    # AI Model Paths (Download free pretrained models)
    MODEL_CACHE_DIR: str = "models/pretrained"
    DEEPFAKE_MODEL: str = "deepfake_efficientnet.pth"
    DEEPFAKE_MODEL_SHA256: Optional[str] = None
    VOICE_MODEL: str = "voice_antispoofing.pkl"
    VOICE_MODEL_SHA256: Optional[str] = None
    PHISHING_MODEL: str = "phishing_xgb.pkl"
    PHISHING_MODEL_SHA256: Optional[str] = None

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 20  # Limit to 20 requests
    RATE_LIMIT_WINDOW: int = 3600  # Enforced hourly window (3600 seconds)

    # Email SMTP Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None

    # SendGrid Configuration
    SENDGRID_API_KEY: Optional[str] = None
    SENDGRID_FROM_EMAIL: Optional[str] = None

    # Resend Configuration
    RESEND_API_KEY: Optional[str] = None
    RESEND_FROM_EMAIL: Optional[str] = None

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json, text

    # Inference Protection
    INFERENCE_TIMEOUT: int = 300  # Maximum seconds allowed for single inference analysis (increased to allow model download)

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    @model_validator(mode="after")
    def validate_production_secrets(self) -> 'Settings':
        """Enforces that safe secrets are configured in production environment"""
        if self.ENVIRONMENT == "production":
            # Secret key checks
            if self.SECRET_KEY == "change-this-in-production-min-32-chars-long":
                raise ValueError("SECRET_KEY cannot be the default placeholder value in a production environment.")
            if len(self.SECRET_KEY) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters long in production.")

            # Database checks
            if "sqlite" in self.DATABASE_URL:
                raise ValueError("SQLite is not allowed in a production environment. Please configure PostgreSQL DATABASE_URL.")

            # Redis checks
            if "redis://" in self.REDIS_URL and "@" not in self.REDIS_URL:
                # Basic check to see if password credentials are missing in the URL scheme
                raise ValueError("REDIS_URL must use authentication credentials in production.")

        return self

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
