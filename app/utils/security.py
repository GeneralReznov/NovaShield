"""
KAVACH AI Pro - Security Utilities
File validation, sanitization, encryption
"""

import os
import re
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates uploaded files for security"""

    ALLOWED_VIDEO = {'video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska'}
    ALLOWED_AUDIO = {'audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/x-m4a',
                     'audio/flac', 'audio/ogg', 'application/octet-stream'}
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    MAX_AUDIO_SIZE = 20 * 1024 * 1024

    @classmethod
    async def validate_file(cls, file: UploadFile, file_type: str) -> Tuple[bool, str]:
        if not file.filename:
            return False, "No file provided"

        safe_filename = cls._sanitize_filename(file.filename)
        if not safe_filename:
            return False, "Invalid filename"

        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_type == "video":
            max_size = cls.MAX_VIDEO_SIZE
        elif file_type == "audio":
            max_size = cls.MAX_AUDIO_SIZE
        else:
            return False, f"Unknown file type: {file_type}"

        if file_size > max_size:
            return False, f"File too large: {file_size / 1024 / 1024:.1f}MB (max {max_size / 1024 / 1024:.0f}MB)"

        if file_size == 0:
            return False, "Empty file"

        return True, ""
        
    @classmethod
    def verify_model_hash(cls, filepath: str, expected_hash: Optional[str]) -> bool:
        """Verify the SHA-256 hash of a model file against an expected value"""
        if not expected_hash:
            from app.core.config import settings
            if settings.ENVIRONMENT == "production":
                logger.error(f"Missing expected hash for {filepath} in production. Failing closed.")
                return False
            logger.info(f"No expected hash configured for {filepath}. Skipping cryptographic signature check.")
            return True

        if not os.path.exists(filepath):
            logger.error(f"Model file not found for verification: {filepath}")
            return False

        try:
            import hashlib
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest().lower()
            expected = expected_hash.strip().lower()
            if file_hash == expected:
                logger.info(f"Cryptographic validation passed for model: {filepath}")
                return True
            else:
                logger.error(f"Cryptographic validation FAILED for model: {filepath}. Expected {expected}, got {file_hash}")
                return False
        except Exception as e:
            logger.error(f"Failed to perform cryptographic validation for model {filepath}: {e}")
            return False

    @staticmethod
    def _sanitize_filename(filename: str) -> Optional[str]:
        filename = os.path.basename(filename)
        filename = filename.replace('\x00', '')
        if not re.match(r'^[\w\-. ]+$', filename):
            return None
        if not filename or filename in ('.', '..'):
            return None
        return filename


class EncryptionUtils:
    @staticmethod
    def hash_password(password: str) -> str:
        from passlib.hash import argon2
        return argon2.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        from passlib.hash import argon2
        return argon2.verify(password, hashed)

    @staticmethod
    def generate_api_key() -> str:
        return f"kav_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()


class AuditLogger:
    @staticmethod
    async def log_action(
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        details: Optional[dict] = None,
        db=None
    ):
        """Log security audit event to database (best-effort).
        
        If a db session is provided, the audit log is added to that session
        (no separate commit — the caller's transaction handles it).
        Otherwise a new session is opened and committed independently.
        """
        try:
            from app.models.database import AuditLog

            log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details or {}
            )

            if db is not None:
                # Use caller's session — no separate commit needed
                db.add(log)
            else:
                # Standalone: open a new session and commit
                from app.db.session import async_session
                async with async_session() as new_db:
                    new_db.add(log)
                    await new_db.commit()
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

