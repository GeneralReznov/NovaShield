"""
NovaShield - Main Application
FastAPI serving HTML/CSS/JS frontend + REST API
"""

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys
import os
import time
import asyncio

from app.core.config import settings
from app.core.middleware import SecurityMiddleware, RateLimitMiddleware
from app.api.v1.endpoints import detection, auth, admin, complaints
from app.api.v1.endpoints import groq_router, pages
from app.db.session import engine
from app.models.database import Base

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


def _cleanup_sync():
    now = time.time()
    if os.path.exists(settings.UPLOAD_DIR):
        for root, dirs, files in os.walk(settings.UPLOAD_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    if os.stat(filepath).st_mtime < now - 3600:
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error deleting file {filepath}: {e}")


async def cleanup_orphaned_files():
    while True:
        try:
            await asyncio.to_thread(_cleanup_sync)
        except Exception as e:
            logger.error(f"Error in cleanup job loop: {e}")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NovaShield...")
    cleanup_task = asyncio.create_task(cleanup_orphaned_files())

    try:
        app.state.redis = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await app.state.redis.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis not available: {e} — running without cache")
        app.state.redis = None

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")

        from app.db.session import async_session
        from app.models.database import User, UserRole
        from sqlalchemy import select
        async with async_session() as session:
            result = await session.execute(select(User).where(User.email == 'admin@novashield.ai'))
            user = result.scalar_one_or_none()
            if not user:
                from app.utils.security import EncryptionUtils
                user = User(
                    email='admin@novashield.ai',
                    username='admin',
                    hashed_password=EncryptionUtils.hash_password('Admin@123'),
                    role=UserRole.ADMIN,
                    is_active=True,
                    is_verified=True
                )
                session.add(user)
                try:
                    await session.commit()
                    logger.info("Default admin user created: admin@novashield.ai / Admin@123")
                except Exception:
                    await session.rollback()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    yield

    cleanup_task.cancel()
    if app.state.redis:
        await app.state.redis.close()
    logger.info("NovaShield shutdown complete")


app = FastAPI(
    title="NovaShield",
    version="3.0.0",
    description="AI-Powered Digital Public Safety Platform",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(SecurityMiddleware)

# Static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Uploads
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# ── Routers ────────────────────────────────────────────────────────────────────

# HTML pages (must come before /api routes)
app.include_router(pages.router, tags=["Pages"])

# API routes
app.include_router(auth.router,       prefix="/api/v1/auth",     tags=["Auth"])
app.include_router(detection.router,  prefix="/api/v1",          tags=["Detection"])
app.include_router(complaints.router, prefix="/api/v1",          tags=["Complaints"])
app.include_router(groq_router.router, prefix="/api/v1",         tags=["Groq AI"])

try:
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
except Exception as e:
    logger.warning(f"Admin router not loaded: {e}")


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    try:
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": str(e)})


@app.get("/metrics", tags=["Health"])
async def metrics():
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            user_count = (await conn.execute(text("SELECT COUNT(*) FROM users"))).scalar() or 0
            detection_count = (await conn.execute(text("SELECT COUNT(*) FROM detections"))).scalar() or 0
        return {"users_total": user_count, "detections_total": detection_count, "status": "healthy"}
    except Exception:
        return {"status": "healthy", "note": "database metrics unavailable"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=1)
