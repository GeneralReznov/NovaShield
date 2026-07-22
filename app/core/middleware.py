"""
KAVACH AI Pro - Security Middleware
Sliding window rate limiting and security headers
"""

import time
import logging
import redis.asyncio as redis
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import secrets
from jose import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Adds security headers to all responses"""
    async def dispatch(self, request: Request, call_next):
        # Skip OPTIONS preflight — let CORSMiddleware handle it
        if request.method == "OPTIONS":
            return await call_next(request)

        # Generate request ID
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Request-ID"] = request_id
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # Enforce Content Security Policy (CSP)
        # connect-src must include the actual backend port to allow frontend API calls
        backend_port = getattr(settings, 'PORT', 8005)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            f"connect-src 'self' * blob:; "
            "frame-ancestors 'none';"
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter backed by Redis. Enforces 20 requests/hour/user."""

    EXEMPT_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/api/v1/auth/login", "/api/v1/auth/register"}

    def __init__(self, app, redis_url: str, max_requests: int = 20, window_seconds: int = 3600):
        super().__init__(app)
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def dispatch(self, request: Request, call_next):
        # Skip OPTIONS preflight — let CORSMiddleware handle it
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip rate limiting for health/docs/auth endpoints and non-production modes
        if request.url.path in self.EXEMPT_PATHS or request.url.path.startswith("/docs") or settings.ENVIRONMENT in ("testing", "development"):
            return await call_next(request)

        # 1. Identify user from cookie or authorization header
        user_id = "anonymous"
        token = request.cookies.get("access_token")
        
        if not token:
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]

        if token:
            try:
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                if payload.get("type") == "access":
                    user_id = payload.get("sub", "anonymous")
            except Exception:
                pass

        # If user is anonymous, rate limit by client IP, else rate limit by user_id
        if user_id == "anonymous":
            client_ip = request.client.host if request.client else "unknown"
            key = f"rl:ip:{client_ip}"
        else:
            key = f"rl:user:{user_id}"

        # 2. Query Redis rate limit window
        try:
            r = await self._get_redis()
            now = time.time()
            window_start = now - self.window_seconds

            pipe = r.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Add current request
            pipe.zadd(key, {f"{now}:{secrets.token_hex(4)}": now})
            # Count requests in window
            pipe.zcard(key)
            # Set TTL
            pipe.expire(key, self.window_seconds + 1)
            results = await pipe.execute()

            request_count = results[2]

            if request_count > self.max_requests:
                retry_after = int(self.window_seconds - (now - window_start))
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Try again in {retry_after}s.",
                        "retry_after": retry_after
                    }
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.max_requests)
            response.headers["X-RateLimit-Remaining"] = str(max(0, self.max_requests - request_count))
            response.headers["X-RateLimit-Reset"] = str(int(now + self.window_seconds))
            return response

        except HTTPException:
            raise
        except Exception as e:
            # Enforce FAIL-OPEN design: if Redis is down, allow request to prevent DoS via Redis downtime
            logger.error(f"Rate limiter Redis error (failing open): {e}")
            return await call_next(request)
