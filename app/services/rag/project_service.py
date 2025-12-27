"""Project and RAG configuration management service."""

from datetime import datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Create a separate engine for the main backend database
BACKEND_DATABASE_URL = "postgresql://qontinui_user:qontinui_dev_password@localhost:5432/qontinui_db"
backend_engine = create_engine(BACKEND_DATABASE_URL, pool_pre_ping=True)
BackendSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=backend_engine)

# Type: ignore needed here because declarative_base() returns a dynamic class
# that mypy cannot properly infer at static analysis time
BackendBase: Any = declarative_base()


class Project(BackendBase):
    """Project model (mirror of main backend)."""

    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    configuration = Column(JSON, nullable=False, default={})
    version = Column(Integer, nullable=False, default=1)
    is_public = Column(Boolean, nullable=False, default=False)
    project_type = Column(String, nullable=False, default="traditional")
    rag_config = Column(JSON, nullable=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_backend_db():
    """Get database session for main backend database."""
    db = BackendSessionLocal()
    try:
        yield db
    finally:
        db.close()


class ProjectService:
    """Service for managing projects and RAG configurations."""

    @staticmethod
    def get_project(db: Session, project_id: str) -> Project:
        """Get project by ID or raise 404.

        Args:
            db: Database session
            project_id: Project UUID as string

        Returns:
            Project instance

        Raises:
            HTTPException: 404 if project not found
        """
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        return project

    @staticmethod
    def get_rag_config(project: Project) -> dict[str, Any]:
        """Get RAG config from project, initializing if needed.

        Args:
            project: Project instance

        Returns:
            RAG configuration dictionary with keys: elements, states, transitions
        """
        if project.rag_config is None:
            project.rag_config = {"elements": {}, "states": {}, "transitions": {}}
        # project.rag_config is a Column[JSON] type, but at runtime it contains the actual dict value
        # Cast to dict for type checking purposes
        return dict(project.rag_config)  # type: ignore[arg-type]

    @staticmethod
    def save_rag_config(db: Session, project: Project, rag_config: dict[str, Any]) -> None:
        """Save RAG config to project.

        Args:
            db: Database session
            project: Project instance
            rag_config: RAG configuration dictionary to save
        """
        # SQLAlchemy Column attributes accept the actual values at runtime
        project.rag_config = rag_config  # type: ignore[assignment]
        project.updated_at = datetime.utcnow()  # type: ignore[assignment]
        db.commit()
        db.refresh(project)

    @staticmethod
    def initialize_rag_config(project: Project) -> dict[str, Any]:
        """Initialize empty RAG config structure.

        Args:
            project: Project instance

        Returns:
            Empty RAG configuration dictionary
        """
        return {"elements": {}, "states": {}, "transitions": {}}
