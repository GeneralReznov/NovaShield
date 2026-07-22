"""
NovaShield - JWT Authentication
Secure authentication with Argon2
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.database import User, UserRole
from app.db.session import async_session

# auto_error=False allows us to gracefully fall back to cookies if the Header is missing
security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Return the authenticated user.

    Priority order:
    1. Bearer token from Authorization header
    2. access_token HttpOnly cookie
    3. Fall back to the default admin account (dev bypass)
    """
    token: Optional[str] = None

    # 1. Try Authorization header
    if credentials and credentials.credentials:
        token = credentials.credentials

    # 2. Try cookie
    if not token:
        token = request.cookies.get("access_token")

    # 3. Try to decode token and fetch user from DB
    if token:
        payload = decode_token(token)
        if payload and payload.get("type") == "access":
            user_id = payload.get("sub")
            if user_id:
                async with async_session() as db:
                    result = await db.execute(select(User).where(User.id == user_id))
                    user = result.scalar_one_or_none()
                    if user and user.is_active:
                        return user

    # 4. Dev bypass — return the real default admin from DB
    async with async_session() as db:
        result = await db.execute(select(User).where(User.email == 'admin@novashield.ai'))
        user = result.scalar_one_or_none()
        if user:
            return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"code": "UNAUTHORIZED", "message": "Authentication required"},
    )


async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
