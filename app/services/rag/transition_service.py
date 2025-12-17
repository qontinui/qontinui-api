"""Service for managing RAG transitions."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.services.rag.project_service import ProjectService

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from app.routes.rag import RAGTransition, RAGTransitionFormData


class TransitionService:
    """Service for RAG transition CRUD operations."""

    def __init__(self):
        """Initialize transition service with dependencies."""
        self.project_service = ProjectService()

    def list_transitions(
        self, db: Session, project_id: str, from_state_id: str | None = None
    ) -> list["RAGTransition"]:
        """List all transitions in a project with optional filter.

        Args:
            db: Database session
            project_id: Project UUID
            from_state_id: Optional filter by source state

        Returns:
            List of RAG transitions
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGTransition

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        transitions = rag_config.get("transitions", {})

        # Apply filter if specified
        if from_state_id:
            return [
                RAGTransition(**trans)
                for trans in transitions.values()
                if trans.get("from_state_id") == from_state_id
            ]

        return [RAGTransition(**trans) for trans in transitions.values()]

    def create_transition(
        self, db: Session, project_id: str, data: "RAGTransitionFormData"
    ) -> "RAGTransition":
        """Create a new RAG transition.

        Args:
            db: Database session
            project_id: Project UUID
            data: Transition form data

        Returns:
            Created RAG transition
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGTransition

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        # Generate ID and timestamps
        transition_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        transition = RAGTransition(
            id=transition_id,
            from_state_id=data.from_state_id or "",
            to_state_id=data.to_state_id or "",
            action=data.action or "",
            description=data.description or "",
            created_at=now,
            updated_at=now,
        )

        # Store in rag_config
        if "transitions" not in rag_config:
            rag_config["transitions"] = {}
        rag_config["transitions"][transition_id] = transition.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return transition

    def get_transition(
        self, db: Session, project_id: str, transition_id: str
    ) -> "RAGTransition":
        """Get a single RAG transition by ID.

        Args:
            db: Database session
            project_id: Project UUID
            transition_id: Transition UUID

        Returns:
            RAG transition

        Raises:
            HTTPException: 404 if transition not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGTransition

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        transitions = rag_config.get("transitions", {})
        if transition_id not in transitions:
            raise HTTPException(
                status_code=404, detail=f"Transition {transition_id} not found"
            )

        return RAGTransition(**transitions[transition_id])

    def update_transition(
        self,
        db: Session,
        project_id: str,
        transition_id: str,
        data: "RAGTransitionFormData",
    ) -> "RAGTransition":
        """Update an existing RAG transition.

        Args:
            db: Database session
            project_id: Project UUID
            transition_id: Transition UUID
            data: Updated transition form data

        Returns:
            Updated RAG transition

        Raises:
            HTTPException: 404 if transition not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGTransition

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        transitions = rag_config.get("transitions", {})
        if transition_id not in transitions:
            raise HTTPException(
                status_code=404, detail=f"Transition {transition_id} not found"
            )

        # Update transition
        trans_data = transitions[transition_id]
        update_data = data.model_dump(exclude_none=True)
        trans_data.update(update_data)
        trans_data["updated_at"] = datetime.utcnow().isoformat()

        transition = RAGTransition(**trans_data)
        rag_config["transitions"][transition_id] = transition.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return transition

    def delete_transition(
        self, db: Session, project_id: str, transition_id: str
    ) -> None:
        """Delete a RAG transition.

        Args:
            db: Database session
            project_id: Project UUID
            transition_id: Transition UUID

        Raises:
            HTTPException: 404 if transition not found
        """
        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        transitions = rag_config.get("transitions", {})
        if transition_id not in transitions:
            raise HTTPException(
                status_code=404, detail=f"Transition {transition_id} not found"
            )

        del rag_config["transitions"][transition_id]

        self.project_service.save_rag_config(db, project, rag_config)
