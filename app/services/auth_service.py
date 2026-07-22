from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

class AuthService:
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7)) # 7 days default
        
        to_encode.update({"exp": expire})
        
        # We need a secret key. Since we might not have it in settings yet, we fallback to a default (not safe for prod, but good for dev)
        secret_key = getattr(settings, "SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
        algorithm = getattr(settings, "ALGORITHM", "HS256")
        
        encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
        return encoded_jwt
        
    def decode_access_token(self, token: str) -> Optional[dict]:
        try:
            secret_key = getattr(settings, "SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
            algorithm = getattr(settings, "ALGORITHM", "HS256")
            
            payload = jwt.decode(token, secret_key, algorithms=[algorithm])
            return payload
        except JWTError as e:
            logger.error(f"JWT decode error: {e}")
            return None

auth_service = AuthService()
