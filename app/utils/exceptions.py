"""
KAVACH AI Pro - Custom Exceptions
Proper error handling with fallback mechanisms
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any

class KavachException(Exception):
    """Base exception for KAVACH AI"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ModelLoadError(KavachException):
    """AI model failed to load"""
    def __init__(self, model_name: str, original_error: str):
        super().__init__(
            message=f"Failed to load model: {model_name}",
            code="MODEL_LOAD_ERROR",
            details={"model": model_name, "error": original_error}
        )

class ModelInferenceError(KavachException):
    """AI model inference failed"""
    def __init__(self, model_name: str, input_info: str, original_error: str):
        super().__init__(
            message=f"Model inference failed: {model_name}",
            code="MODEL_INFERENCE_ERROR",
            details={"model": model_name, "input": input_info, "error": original_error}
        )

class FileValidationError(KavachException):
    """File validation failed"""
    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"File validation failed: {filename}",
            code="FILE_VALIDATION_ERROR",
            details={"filename": filename, "reason": reason}
        )

class RateLimitExceeded(KavachException):
    """Rate limit exceeded"""
    def __init__(self, retry_after: int):
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after}
        )

class QuotaExceeded(KavachException):
    """User quota exceeded"""
    def __init__(self, limit: int, used: int):
        super().__init__(
            message="Daily quota exceeded. Upgrade to premium.",
            code="QUOTA_EXCEEDED",
            details={"limit": limit, "used": used}
        )

# Error response handler
def create_error_response(exc: KavachException) -> Dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }
    }
