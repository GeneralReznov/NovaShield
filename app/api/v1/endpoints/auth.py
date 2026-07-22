"""
NovaShield - Authentication Endpoints
Registration, login, token refresh, profile management
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.database import User, UserRole
from app.core.auth import (
    create_access_token, create_refresh_token, decode_token, get_current_user
)
from app.core.config import settings
from app.utils.security import EncryptionUtils, AuditLogger
from app.schemas.auth import (
    UserRegisterRequest, UserLoginRequest, TokenResponse,
    RefreshTokenRequest, UserResponse, MessageResponse
)

router = APIRouter()


@router.post("/register", response_model=MessageResponse, status_code=201)
async def register(
    request: Request,
    data: UserRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user account"""
    # Check existing email
    result = await db.execute(select(User).where(User.email == data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail={"code": "EMAIL_EXISTS", "message": "Email already registered"}
        )

    # Check existing username
    result = await db.execute(select(User).where(User.username == data.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail={"code": "USERNAME_EXISTS", "message": "Username already taken"}
        )

    # Create user
    user = User(
        email=data.email,
        username=data.username,
        hashed_password=EncryptionUtils.hash_password(data.password),
        full_name=data.full_name,
        role=UserRole.USER,
        is_active=True,
    )
    db.add(user)
    await db.commit()

    # Audit log
    await AuditLogger.log_action(
        user_id=user.id,
        action="user.register",
        resource_type="user",
        resource_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        db=db,
    )

    return MessageResponse(message="Account created successfully. Please login.")


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    response: Response,
    data: UserLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate and receive JWT tokens set in Secure HttpOnly cookies"""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    # Dummy hash to mitigate timing attacks (pre-computed argon2 hash for 'dummy')
    dummy_hash = "$argon2id$v=19$m=65536,t=3,p=4$2q1H1L7O8q0c9b0e2f5$8/Q2X5p4t2Z8e7v9c3q1H1L7O8q0c9b0e2f5L8+M="
    
    if not user:
        EncryptionUtils.verify_password(data.password, dummy_hash)
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_CREDENTIALS", "message": "Invalid email or password"}
        )
    elif not EncryptionUtils.verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_CREDENTIALS", "message": "Invalid email or password"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail={"code": "ACCOUNT_DISABLED", "message": "Account has been disabled"}
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    # Generate tokens
    token_data = {"sub": user.id, "email": user.email, "role": user.role.value}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Set secure HttpOnly cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="strict",
        secure=settings.ENVIRONMENT == "production",
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        expires=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        samesite="strict",
        secure=settings.ENVIRONMENT == "production",
    )

    # Audit log
    await AuditLogger.log_action(
        user_id=user.id,
        action="user.login",
        resource_type="user",
        resource_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        db=db,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(response: Response, data: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    payload = decode_token(data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN", "message": "Invalid or expired refresh token"}
        )

    token_data = {"sub": payload["sub"], "email": payload["email"], "role": payload["role"]}
    new_access = create_access_token(token_data)
    new_refresh = create_refresh_token(token_data)

    # Update cookies
    response.set_cookie(
        key="access_token",
        value=new_access,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="strict",
        secure=settings.ENVIRONMENT == "production",
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh,
        httponly=True,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        expires=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        samesite="strict",
        secure=settings.ENVIRONMENT == "production",
    )

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """Clear HttpOnly authentication cookies"""
    response.delete_cookie("access_token", httponly=True, samesite="strict", secure=settings.ENVIRONMENT == "production")
    response.delete_cookie("refresh_token", httponly=True, samesite="strict", secure=settings.ENVIRONMENT == "production")
    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse)
async def get_profile(user: User = Depends(get_current_user)):
    """Get current user profile"""
    return user
