"""Service for managing RAG elements."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.services.rag.description_service import DescriptionService
from app.services.rag.project_service import ProjectService

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    from app.routes.rag import RAGElement, RAGElementFormData


class ElementService:
    """Service for RAG element CRUD operations."""

    def __init__(self):
        """Initialize element service with dependencies."""
        self.project_service = ProjectService()
        self.description_service = DescriptionService()

    def list_elements(
        self,
        db: Session,
        project_id: str,
        state_name: str | None = None,
        is_defining: bool | None = None,
    ) -> list["RAGElement"]:
        """List all elements in a project with optional filters.

        Args:
            db: Database session
            project_id: Project UUID
            state_name: Optional filter by state name
            is_defining: Optional filter by is_defining_element

        Returns:
            List of RAG elements
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGElement

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements = rag_config.get("elements", {})

        # Apply filters
        filtered_elements = []
        for element_data in elements.values():
            # Filter by state_name
            if state_name and element_data.get("state_name") != state_name:
                continue

            # Filter by is_defining_element
            if is_defining is not None and element_data.get("is_defining_element") != is_defining:
                continue

            filtered_elements.append(RAGElement(**element_data))

        return filtered_elements

    def create_element(
        self, db: Session, project_id: str, data: "RAGElementFormData"
    ) -> "RAGElement":
        """Create a new RAG element.

        Args:
            db: Database session
            project_id: Project UUID
            data: Element form data

        Returns:
            Created RAG element
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGElement

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        # Generate ID and timestamps
        element_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Create element with defaults for required fields
        element_data = data.model_dump(exclude_none=True)
        element_data.update(
            {
                "id": element_id,
                "created_at": now,
                "updated_at": now,
                # Set defaults for required fields if not provided
                "source_app": element_data.get("source_app", ""),
                "extraction_method": element_data.get("extraction_method", "manual"),
                "width": element_data.get("width", 0),
                "height": element_data.get("height", 0),
                "aspect_ratio": element_data.get("aspect_ratio", 0.0),
                "area": element_data.get("area", 0),
                "position_quadrant": element_data.get("position_quadrant", ""),
                "dominant_colors": element_data.get("dominant_colors", []),
                "color_histogram": element_data.get("color_histogram", []),
                "average_brightness": element_data.get("average_brightness", 0.0),
                "contrast_ratio": element_data.get("contrast_ratio", 0.0),
                "edge_density": element_data.get("edge_density", 0.0),
                "has_text": element_data.get("has_text", False),
                "ocr_text": element_data.get("ocr_text", ""),
                "ocr_confidence": element_data.get("ocr_confidence", 0.0),
                "text_length": element_data.get("text_length", 0),
                "element_type": element_data.get("element_type", "unknown"),
                "element_subtype": element_data.get("element_subtype", ""),
                "is_interactive": element_data.get("is_interactive", False),
                "interaction_type": element_data.get("interaction_type", ""),
                "visual_state": element_data.get("visual_state", ""),
                "is_enabled": element_data.get("is_enabled", True),
                "is_selected": element_data.get("is_selected", False),
                "is_focused": element_data.get("is_focused", False),
                "depth_in_hierarchy": element_data.get("depth_in_hierarchy", 0),
                "sibling_count": element_data.get("sibling_count", 0),
                "platform": element_data.get("platform", ""),
                # Images with Masks
                "images": element_data.get("images", []),
                # Aggregated Embeddings
                "aggregated_image_embedding": element_data.get("aggregated_image_embedding"),
                "aggregated_text_embedding": element_data.get("aggregated_text_embedding"),
                # Legacy Embeddings
                "text_description": element_data.get("text_description", ""),
                # Matching Configuration
                "matching_strategy": element_data.get("matching_strategy"),
                "ocr_filter": element_data.get("ocr_filter"),
                "ocr_config": element_data.get("ocr_config"),
                "expected_text": element_data.get("expected_text"),
                "state_name": element_data.get("state_name", ""),
                "is_defining_element": element_data.get("is_defining_element", False),
                "is_optional_element": element_data.get("is_optional_element", False),
                "similarity_threshold": element_data.get("similarity_threshold", 0.8),
                "is_fixed_position": element_data.get("is_fixed_position", False),
                "is_shared": element_data.get("is_shared", False),
                "probability": element_data.get("probability", 0.0),
                "semantic_role": element_data.get("semantic_role", ""),
                "semantic_action": element_data.get("semantic_action", ""),
                "style_family": element_data.get("style_family", ""),
            }
        )

        element = RAGElement(**element_data)

        # Store in rag_config
        if "elements" not in rag_config:
            rag_config["elements"] = {}
        rag_config["elements"][element_id] = element.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return element

    def get_element(self, db: Session, project_id: str, element_id: str) -> "RAGElement":
        """Get a single RAG element by ID.

        Args:
            db: Database session
            project_id: Project UUID
            element_id: Element UUID

        Returns:
            RAG element

        Raises:
            HTTPException: 404 if element not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGElement

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements = rag_config.get("elements", {})
        if element_id not in elements:
            raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

        return RAGElement(**elements[element_id])

    def update_element(
        self, db: Session, project_id: str, element_id: str, data: "RAGElementFormData"
    ) -> "RAGElement":
        """Update an existing RAG element.

        Args:
            db: Database session
            project_id: Project UUID
            element_id: Element UUID
            data: Updated element form data

        Returns:
            Updated RAG element

        Raises:
            HTTPException: 404 if element not found
        """
        # Import here to avoid circular dependency
        from app.routes.rag import RAGElement

        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements = rag_config.get("elements", {})
        if element_id not in elements:
            raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

        # Update element
        element_data = elements[element_id]
        update_data = data.model_dump(exclude_none=True)
        element_data.update(update_data)
        element_data["updated_at"] = datetime.utcnow().isoformat()

        element = RAGElement(**element_data)
        rag_config["elements"][element_id] = element.model_dump()

        self.project_service.save_rag_config(db, project, rag_config)

        return element

    def delete_element(self, db: Session, project_id: str, element_id: str) -> None:
        """Delete a RAG element.

        Args:
            db: Database session
            project_id: Project UUID
            element_id: Element UUID

        Raises:
            HTTPException: 404 if element not found
        """
        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements = rag_config.get("elements", {})
        if element_id not in elements:
            raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

        del rag_config["elements"][element_id]
        self.project_service.save_rag_config(db, project, rag_config)

    def generate_description(self, db: Session, project_id: str, element_id: str) -> str:
        """Generate a description for an element.

        Args:
            db: Database session
            project_id: Project UUID
            element_id: Element UUID

        Returns:
            Generated description string

        Raises:
            HTTPException: 404 if element not found
        """
        element = self.get_element(db, project_id, element_id)
        description: str = self.description_service.generate_element_description(element)
        return description

    def batch_generate_descriptions(
        self, db: Session, project_id: str, element_ids: list[str] | None = None
    ) -> dict[str, str]:
        """Generate descriptions for multiple elements.

        Args:
            db: Database session
            project_id: Project UUID
            element_ids: Optional list of element IDs (None = all elements)

        Returns:
            Dictionary mapping element IDs to descriptions

        Raises:
            HTTPException: 404 if project not found
        """
        project = self.project_service.get_project(db, project_id)
        rag_config = self.project_service.get_rag_config(project)

        elements_dict = rag_config.get("elements", {})

        # Filter elements if specific IDs provided
        if element_ids:
            elements_dict = {
                eid: edata for eid, edata in elements_dict.items() if eid in element_ids
            }

        # Import here to avoid circular dependency
        from app.routes.rag import RAGElement

        elements = [RAGElement(**edata) for edata in elements_dict.values()]

        descriptions: dict[str, str] = self.description_service.generate_descriptions_batch(
            elements
        )
        return descriptions
