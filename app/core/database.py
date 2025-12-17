"""Database configuration and session management.

CURRENT STATUS (December 2024):
================================

Phase 1 (COMPLETED):
- ✅ Database models migrated to qontinui-web/backend
- ✅ Alembic migration created in backend (80fb6afdd21a_add_qontinui_api_tables.py)
- ✅ Models marked as deprecated in app/models/
- ✅ DATABASE_URL fixed to point to qontinui_db (shared with backend)
- ✅ Alembic configuration removed from qontinui-api
- ✅ qontinui-api now uses backend's database (qontinui_db)

Phase 2 (FUTURE):
- ⏳ Refactor routes to make API calls to backend instead of direct DB access
- ⏳ Remove SQLAlchemy models from app/models/ (keep Pydantic schemas)
- ⏳ Remove this database.py file entirely

Current Architecture:
- Backend database: qontinui_db (port 5432)
- Both qontinui-web/backend AND qontinui-api connect to the same database
- Schema is managed exclusively by backend Alembic migrations
- qontinui-api should NEVER run migrations or create tables

IMPORTANT:
- qontinui-api connects to backend's database (qontinui_db) directly
- All schema changes must be done via qontinui-web/backend migrations
- The models in app/models/ are deprecated but still functional
- DO NOT call init_db() - tables are managed by backend
"""

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Database connection to shared qontinui_db
# This connects to the same database as qontinui-web/backend
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_size=10,  # Connection pool size
    max_overflow=20,  # Max connections beyond pool_size
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base: Any = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that yields database sessions.

    This connects to the shared qontinui_db database.
    Schema is managed by qontinui-web/backend Alembic migrations.

    Usage in FastAPI endpoints:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables.

    DEPRECATED: This function should NEVER be called.

    All database tables are managed by qontinui-web/backend Alembic migrations.
    qontinui-api connects to the backend database (qontinui_db) but does NOT
    manage the schema.

    DO NOT USE THIS FUNCTION.
    """
    raise RuntimeError(
        "init_db() is deprecated. "
        "Database schema is managed by qontinui-web/backend Alembic migrations. "
        "qontinui-api should NOT create or modify tables."
    )
