"""Service for exporting and importing RAG configurations."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.services.rag.project_service import ProjectService

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from app.routes.rag import ImportResult, RAGExportData


class ExportService:
    """Service for RAG project export/import functionality."""

    def __init__(self):
        """Initialize export service with dependencies."""
        self.project_service = ProjectService()

    def export_project(self, db: Session, project_id: str) -> "RAGExportData":
        """Export RAG data for a project.

        Args:
            db: Database session
            project_id: Project UUID

        Returns:
            RAGExportData containing all elements, states, and transitions
        """
        # Import here to avoid circular dependency
        from app.routes.rag import (
            RAGElement,
            RAGExportData,
            RAGExportMetadata,
            RAGState,
            RAGTransition,
        )

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements = [RAGElement(**elem) for elem in rag_config.get("elements", {}).values()]
        states = [RAGState(**state) for state in rag_config.get("states", {}).values()]
        transitions = [
            RAGTransition(**trans) for trans in rag_config.get("transitions", {}).values()
        ]

        return RAGExportData(
            elements=elements,
            states=states,
            transitions=transitions,
            metadata=RAGExportMetadata(
                exported_at=datetime.utcnow().isoformat(),
                version="1.0.0",
                project_id=project_id,
            ),
        )

    def import_project(self, db: Session, project_id: str, data: "RAGExportData") -> "ImportResult":
        """Import RAG data into a project.

        Args:
            db: Database session
            project_id: Project UUID
            data: RAG export data to import

        Returns:
            ImportResult with counts of imported/skipped/failed items
        """
        # Import here to avoid circular dependency
        from app.routes.rag import ImportResult

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        imported = 0
        skipped = 0
        errors = []

        # Import elements
        if "elements" not in rag_config:
            rag_config["elements"] = {}

        for element in data.elements:
            try:
                if element.id in rag_config["elements"]:
                    skipped += 1
                else:
                    rag_config["elements"][element.id] = element.model_dump()
                    imported += 1
            except Exception as e:
                errors.append(f"Failed to import element {element.id}: {str(e)}")

        # Import states
        if "states" not in rag_config:
            rag_config["states"] = {}

        for state in data.states:
            try:
                if state.id in rag_config["states"]:
                    skipped += 1
                else:
                    rag_config["states"][state.id] = state.model_dump()
                    imported += 1
            except Exception as e:
                errors.append(f"Failed to import state {state.id}: {str(e)}")

        # Import transitions
        if "transitions" not in rag_config:
            rag_config["transitions"] = {}

        for transition in data.transitions:
            try:
                if transition.id in rag_config["transitions"]:
                    skipped += 1
                else:
                    rag_config["transitions"][transition.id] = transition.model_dump()
                    imported += 1
            except Exception as e:
                errors.append(f"Failed to import transition {transition.id}: {str(e)}")

        self.project_service.save_rag_config(db, project, rag_config)

        return ImportResult(imported=imported, skipped=skipped, errors=errors)

    def validate_import_data(self, data: "RAGExportData") -> tuple[bool, list[str]]:
        """Validate import data before processing.

        Args:
            data: RAG export data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check metadata version
        if data.metadata.version != "1.0.0":
            errors.append(f"Unsupported version: {data.metadata.version}")

        # Validate element IDs are unique
        element_ids = [e.id for e in data.elements]
        if len(element_ids) != len(set(element_ids)):
            errors.append("Duplicate element IDs found")

        # Validate state IDs are unique
        state_ids = [s.id for s in data.states]
        if len(state_ids) != len(set(state_ids)):
            errors.append("Duplicate state IDs found")

        # Validate transition IDs are unique
        transition_ids = [t.id for t in data.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            errors.append("Duplicate transition IDs found")

        # Validate transition references
        state_id_set = set(state_ids)
        for transition in data.transitions:
            if transition.from_state_id and transition.from_state_id not in state_id_set:
                errors.append(
                    f"Transition {transition.id} references missing state: {transition.from_state_id}"
                )
            if transition.to_state_id and transition.to_state_id not in state_id_set:
                errors.append(
                    f"Transition {transition.id} references missing state: {transition.to_state_id}"
                )

        return len(errors) == 0, errors
