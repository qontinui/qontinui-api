"""Service for managing RAG states."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.services.rag.project_service import ProjectService

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from app.routes.rag import RAGState, RAGStateFormData


class StateService:
    """Service for RAG state CRUD operations."""

    def __init__(self):
        """Initialize state service with dependencies."""
        self.project_service = ProjectService()

    def list_states(self, db: Session, project_id: str) -> list["RAGState"]:
        """List all states in a project.

        Args:
            db: Database session
            project_id: Project UUID

        Returns:
            List of RAG states
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGState

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        states = rag_config.get("states", {})
        return [RAGState(**state) for state in states.values()]

    def create_state(self, db: Session, project_id: str, data: "RAGStateFormData") -> "RAGState":
        """Create a new RAG state.

        Args:
            db: Database session
            project_id: Project UUID
            data: State form data

        Returns:
            Created RAG state
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGState

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        # Generate ID and timestamps
        state_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        state = RAGState(
            id=state_id,
            name=data.name or "",
            description=data.description or "",
            element_ids=data.element_ids or [],
            created_at=now,
            updated_at=now,
        )

        # Store in rag_config
        if "states" not in rag_config:
            rag_config["states"] = {}
        rag_config["states"][state_id] = state.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return state

    def get_state(self, db: Session, project_id: str, state_id: str) -> "RAGState":
        """Get a single RAG state by ID.

        Args:
            db: Database session
            project_id: Project UUID
            state_id: State UUID

        Returns:
            RAG state

        Raises:
            HTTPException: 404 if state not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGState

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        states = rag_config.get("states", {})
        if state_id not in states:
            raise HTTPException(status_code=404, detail=f"State {state_id} not found")

        return RAGState(**states[state_id])

    def update_state(
        self, db: Session, project_id: str, state_id: str, data: "RAGStateFormData"
    ) -> "RAGState":
        """Update an existing RAG state.

        Args:
            db: Database session
            project_id: Project UUID
            state_id: State UUID
            data: Updated state form data

        Returns:
            Updated RAG state

        Raises:
            HTTPException: 404 if state not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGState

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        states = rag_config.get("states", {})
        if state_id not in states:
            raise HTTPException(status_code=404, detail=f"State {state_id} not found")

        # Update state
        state_data = states[state_id]
        update_data = data.model_dump(exclude_none=True)
        state_data.update(update_data)
        state_data["updated_at"] = datetime.utcnow().isoformat()

        state = RAGState(**state_data)
        rag_config["states"][state_id] = state.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return state

    def delete_state(self, db: Session, project_id: str, state_id: str) -> None:
        """Delete a RAG state.

        Args:
            db: Database session
            project_id: Project UUID
            state_id: State UUID

        Raises:
            HTTPException: 404 if state not found
        """
        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        states = rag_config.get("states", {})
        if state_id not in states:
            raise HTTPException(status_code=404, detail=f"State {state_id} not found")

        del rag_config["states"][state_id]

        self.project_service.save_rag_config(db, project, rag_config)
