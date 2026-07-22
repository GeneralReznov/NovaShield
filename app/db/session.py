"""
Supports both PostgreSQL (production) and SQLite (development/demo)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Detect database type and configure accordingly
db_url = settings.DATABASE_URL

# Replit provides postgresql:// — convert to asyncpg driver URL
if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

# asyncpg doesn't support sslmode query param — strip it and pass ssl separately
_connect_args = {}
if "postgresql+asyncpg" in db_url and "sslmode" in db_url:
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    _parsed = urlparse(db_url)
    _qs = parse_qs(_parsed.query)
    _sslmode = _qs.pop("sslmode", ["disable"])[0]
    _new_query = urlencode({k: v[0] for k, v in _qs.items()})
    db_url = urlunparse(_parsed._replace(query=_new_query))
    if _sslmode in ("require", "verify-ca", "verify-full"):
        _connect_args["ssl"] = "require"

is_sqlite = "sqlite" in db_url

if is_sqlite:
    # SQLite — no connection pooling needed
    engine = create_async_engine(
        db_url,
        echo=settings.DEBUG,
        connect_args={"check_same_thread": False, "timeout": 30},  # Prevent lock errors
    )
    logger.info(f"Using SQLite database: {db_url}")
else:
    # PostgreSQL — with connection pooling
    engine = create_async_engine(
        db_url,
        echo=settings.DEBUG,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_timeout=30,
        connect_args=_connect_args,
    )
    logger.info(f"Using PostgreSQL database")

# Session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def get_db():
    """Dependency for database sessions"""
    async with async_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
