"""RAG (Retrieval-Augmented Generation) Services for qontinui-api.

This module provides RAG functionality for GUI element indexing and retrieval:
- Embedding generation for GUI elements
- Vector database indexing with Qdrant
- Semantic search over indexed elements
- Visual element matching using SAM3 segmentation
- CRUD operations for elements, states, and transitions
- Project configuration management
- Export/import functionality

The RAG system enables AI-powered element location and retrieval
in GUI automation workflows.
"""

from app.services.rag.description_service import DescriptionService
from app.services.rag.element_service import ElementService
from app.services.rag.export_service import ExportService
from app.services.rag.project_service import ProjectService
from app.services.rag.state_service import StateService
from app.services.rag.transition_service import TransitionService

__all__ = [
    # Legacy services (existing)
    "generate_embeddings",
    "find_rag",
    "search_rag",
    # New CRUD services
    "ProjectService",
    "ElementService",
    "StateService",
    "TransitionService",
    "DescriptionService",
    "ExportService",
]
