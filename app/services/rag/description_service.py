"""Service for generating natural language descriptions of RAG elements."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import for type checking only to avoid circular imports
    # The actual RAGElement will be imported from routes when needed
    from app.routes.rag import RAGElement


class DescriptionService:
    """Service for generating element descriptions for embeddings and search."""

    @staticmethod
    def generate_element_description(element: "RAGElement") -> str:
        """
        Generate a natural language description for an element based on its properties.

        This is used for text embeddings and semantic search. The description
        should capture the element's purpose, appearance, and context in a way
        that would be useful for both vector similarity search and human understanding.

        The description is generated using a rule-based approach that prioritizes:
        1. OCR text (most distinctive for matching)
        2. Semantic role and action (what the element does)
        3. Element type and interaction
        4. Visual state and context
        5. Positional information

        Args:
            element: RAG element to describe

        Returns:
            Natural language description string
        """
        # Start with the most important identifier - OCR text
        if element.ocr_text:
            base = f'"{element.ocr_text}"'

            # Add semantic role/action to explain what this text represents
            if element.semantic_role:
                base += f" {element.semantic_role}"
            elif element.element_type:
                # Fallback to element type if no semantic role
                type_str = (
                    element.element_subtype
                    if element.element_subtype
                    else element.element_type
                )
                base += f" {type_str}"
            else:
                base += " element"

            # Add action context if available
            if element.semantic_action:
                base += f" for {element.semantic_action}"
        else:
            # No OCR text - build description from type and role
            if element.semantic_role:
                base = f"{element.semantic_role}"
                if element.semantic_action:
                    base += f" for {element.semantic_action}"
            elif element.element_type:
                type_str = element.element_type
                if element.element_subtype:
                    type_str = f"{element.element_subtype} {type_str}"
                base = type_str
            else:
                base = "UI element"

        # Add contextual information
        context_parts = []

        # Interaction type adds important behavioral context
        if element.interaction_type and element.interaction_type != "none":
            context_parts.append(f"{element.interaction_type}")

        # Visual state is important for finding elements in specific states
        if element.visual_state and element.visual_state != "normal":
            context_parts.append(f"{element.visual_state}")

        # Position helps disambiguate similar elements
        if element.position_quadrant:
            context_parts.append(f"in {element.position_quadrant}")

        # Parent region provides hierarchical context
        if element.parent_region:
            context_parts.append(f"within {element.parent_region}")

        # State name is crucial for state-specific matching
        if element.state_name:
            context_parts.append(f"on '{element.state_name}' screen")

        # Combine base description with context
        if context_parts:
            return f"{base}, {', '.join(context_parts)}"

        return base

    @staticmethod
    def generate_descriptions_batch(elements: list["RAGElement"]) -> dict[str, str]:
        """
        Generate descriptions for multiple elements.

        Args:
            elements: List of RAG elements to describe

        Returns:
            Dictionary mapping element IDs to their descriptions
        """
        return {
            element.id: DescriptionService.generate_element_description(element)
            for element in elements
        }
