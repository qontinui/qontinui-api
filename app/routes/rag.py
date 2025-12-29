"""
RAG Builder API endpoints.

Manages RAG elements, states, and transitions stored in project rag_config.
These endpoints interact with the main backend's database (qontinui_db).
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError
from qontinui_schemas.common import utc_now
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Load .env file (for development - production uses actual env vars)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Create a separate engine for the main backend database
# Database URL must be set via environment variable for security
BACKEND_DATABASE_URL = os.environ.get("BACKEND_DATABASE_URL")
if not BACKEND_DATABASE_URL:
    raise RuntimeError(
        "BACKEND_DATABASE_URL environment variable is not set. "
        "This is required to connect to the main backend database."
    )

backend_engine = create_engine(BACKEND_DATABASE_URL, pool_pre_ping=True)
BackendSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=backend_engine)

# Type: ignore needed here because declarative_base() returns a dynamic class
# that mypy cannot properly infer at static analysis time
BackendBase: Any = declarative_base()

router = APIRouter(prefix="/rag", tags=["rag"])


# ============================================================================
# Database Models (Mirror of main backend's Project model)
# ============================================================================


class Project(BackendBase):
    """Project model (mirror of main backend)."""

    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    configuration = Column(JSON, nullable=False, default={})
    version = Column(Integer, nullable=False, default=1)
    is_public = Column(Boolean, nullable=False, default=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


# ============================================================================
# Pydantic Models
# ============================================================================


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x: int
    y: int
    width: int
    height: int


class RAGImage(BaseModel):
    """Image with mask for RAG element."""

    id: str
    pixel_data: str | None = None  # Base64 encoded image data (or S3 key)
    s3_key: str | None = None  # S3 object key for storage
    mask_data: str | None = None  # Base64 encoded mask (0/1 or 0.0-1.0)
    mask_density: float = 1.0  # Percentage of active pixels in mask
    width: int = 0
    height: int = 0
    image_embedding: list[float] | None = None  # CLIP 512-dim embedding


class OCRFilter(BaseModel):
    """OCR filtering configuration."""

    text: str | None = None
    match_mode: str = "contains"  # "exact", "contains", "regex"
    similarity: float = 0.8  # Fuzzy matching threshold


class OCRConfig(BaseModel):
    """OCR configuration for element matching."""

    enabled: bool = True
    weight: float = 0.3  # Weight in combined score (0.0-1.0)
    as_filter: bool = True  # True = must match, False = just affects score
    filter_threshold: float = 0.6  # Minimum OCR similarity to pass filter


class RAGElement(BaseModel):
    """RAG Element model matching frontend TypeScript interface."""

    # Identity
    id: str
    created_at: str
    updated_at: str

    # Source Information
    source_app: str
    source_state_id: str | None = None
    source_screenshot_id: str | None = None
    extraction_method: str

    # Geometry
    bounding_box: BoundingBox | None = None
    width: int
    height: int
    aspect_ratio: float
    area: int
    position_quadrant: str

    # Visual Features
    dominant_colors: list[tuple[int, int, int]]
    color_histogram: list[float]
    average_brightness: float
    contrast_ratio: float
    edge_density: float

    # Text Content
    has_text: bool
    ocr_text: str
    ocr_confidence: float
    text_length: int

    # Classification
    element_type: str
    element_subtype: str
    is_interactive: bool
    interaction_type: str

    # State Indicators
    visual_state: str
    is_enabled: bool
    is_selected: bool
    is_focused: bool

    # Context
    parent_region: str | None = None
    depth_in_hierarchy: int
    sibling_count: int

    # Platform
    platform: str

    # Images with Masks (REQUIRED - at least one image)
    images: list[RAGImage] = []

    # Aggregated Embeddings (computed from all images)
    aggregated_image_embedding: list[float] | None = None
    aggregated_text_embedding: list[float] | None = None

    # Legacy Embeddings (for backward compatibility)
    text_embedding: list[float] | None = None
    text_description: str
    image_embedding: list[float] | None = None

    # Matching Configuration
    matching_strategy: str | None = None  # "average" or "any_match", null = use project default
    ocr_filter: OCRFilter | None = None
    ocr_config: OCRConfig | None = None
    expected_text: str | None = None

    # State Machine Integration
    state_id: str | None = None
    state_name: str
    is_defining_element: bool
    is_optional_element: bool
    similarity_threshold: float | None = None  # null = use project default (0.7)
    is_fixed_position: bool
    is_shared: bool
    probability: float
    search_region_id: str | None = None

    # Cross-Application Semantics
    semantic_role: str
    semantic_action: str
    style_family: str


class RAGElementFormData(BaseModel):
    """Partial RAG element for create/update operations."""

    # All fields optional for partial updates
    source_app: str | None = None
    source_state_id: str | None = None
    source_screenshot_id: str | None = None
    extraction_method: str | None = None
    bounding_box: BoundingBox | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: float | None = None
    area: int | None = None
    position_quadrant: str | None = None
    dominant_colors: list[tuple[int, int, int]] | None = None
    color_histogram: list[float] | None = None
    average_brightness: float | None = None
    contrast_ratio: float | None = None
    edge_density: float | None = None
    has_text: bool | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    text_length: int | None = None
    element_type: str | None = None
    element_subtype: str | None = None
    is_interactive: bool | None = None
    interaction_type: str | None = None
    visual_state: str | None = None
    is_enabled: bool | None = None
    is_selected: bool | None = None
    is_focused: bool | None = None
    parent_region: str | None = None
    depth_in_hierarchy: int | None = None
    sibling_count: int | None = None
    platform: str | None = None
    # Images with Masks
    images: list[RAGImage] | None = None
    # Aggregated Embeddings
    aggregated_image_embedding: list[float] | None = None
    aggregated_text_embedding: list[float] | None = None
    # Legacy Embeddings
    text_embedding: list[float] | None = None
    text_description: str | None = None
    image_embedding: list[float] | None = None
    # Matching Configuration
    matching_strategy: str | None = None
    ocr_filter: OCRFilter | None = None
    ocr_config: OCRConfig | None = None
    expected_text: str | None = None
    state_id: str | None = None
    state_name: str | None = None
    is_defining_element: bool | None = None
    is_optional_element: bool | None = None
    similarity_threshold: float | None = None
    is_fixed_position: bool | None = None
    is_shared: bool | None = None
    probability: float | None = None
    search_region_id: str | None = None
    semantic_role: str | None = None
    semantic_action: str | None = None
    style_family: str | None = None


class RAGState(BaseModel):
    """RAG State model."""

    id: str
    name: str
    description: str
    element_ids: list[str]
    created_at: str
    updated_at: str


class RAGStateFormData(BaseModel):
    """Partial RAG state for create/update operations."""

    name: str | None = None
    description: str | None = None
    element_ids: list[str] | None = None


class RAGTransition(BaseModel):
    """RAG Transition model."""

    id: str
    from_state_id: str
    to_state_id: str
    action: str
    description: str
    created_at: str
    updated_at: str


class RAGTransitionFormData(BaseModel):
    """Partial RAG transition for create/update operations."""

    from_state_id: str | None = None
    to_state_id: str | None = None
    action: str | None = None
    description: str | None = None


class SearchQuery(BaseModel):
    """Search query parameters."""

    query: str
    element_types: list[str] | None = None
    states: list[str] | None = None
    limit: int = 50


class SearchMatch(BaseModel):
    """Search match scores."""

    text: float | None = None
    visual: float | None = None
    semantic: float | None = None


class SearchResult(BaseModel):
    """Search result with element and scores."""

    element: RAGElement
    score: float
    matches: SearchMatch


class RAGExportMetadata(BaseModel):
    """Export metadata."""

    exported_at: str
    version: str
    project_id: str | None = None


class RAGExportData(BaseModel):
    """Complete export data."""

    elements: list[RAGElement]
    states: list[RAGState]
    transitions: list[RAGTransition]
    metadata: RAGExportMetadata


class ImportResult(BaseModel):
    """Import operation result."""

    imported: int
    skipped: int
    errors: list[str]


class GenerateDescriptionResponse(BaseModel):
    """Generated description response."""

    description: str


class BatchGenerateDescriptionsResponse(BaseModel):
    """Batch description generation response."""

    updated: int
    total: int


# ============================================================================
# Database Dependency
# ============================================================================


def get_backend_db():
    """Get database session for main backend database."""
    db = BackendSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Helper Functions
# ============================================================================


def get_project(db: Session, project_id: str) -> Project:
    """Get project by ID or raise 404."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


def get_rag_config(project: Project) -> dict[str, Any]:
    """Get RAG config from project's configuration, initializing if needed."""
    # RAG config is stored inside configuration["rag_config"], not as a separate column
    raw_config = project.configuration
    config: dict[str, Any] = dict(raw_config) if raw_config else {}
    if isinstance(config, str):
        import json

        config = json.loads(config)
    rag_config = config.get("rag_config", {"elements": {}, "states": {}, "transitions": {}})
    return dict(rag_config)


def save_rag_config(db: Session, project: Project, rag_config: dict[str, Any]) -> None:
    """Save RAG config to project's configuration."""
    # RAG config is stored inside configuration["rag_config"]
    raw_config = project.configuration
    config: dict[str, Any] = dict(raw_config) if raw_config else {}
    if isinstance(config, str):
        import json

        config = json.loads(config)
    config["rag_config"] = rag_config
    project.configuration = config  # type: ignore[assignment]
    project.updated_at = utc_now()  # type: ignore[assignment]
    db.commit()
    db.refresh(project)


def generate_element_description(element: RAGElement) -> str:
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
    """
    # Start with the most important identifier - OCR text
    if element.ocr_text:
        base = f'"{element.ocr_text}"'

        # Add semantic role/action to explain what this text represents
        if element.semantic_role:
            base += f" {element.semantic_role}"
        elif element.element_type:
            # Fallback to element type if no semantic role
            type_str = element.element_subtype if element.element_subtype else element.element_type
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


# ============================================================================
# Element Endpoints
# ============================================================================


@router.get("/projects/{project_id}/elements", response_model=list[RAGElement])
async def get_elements(
    project_id: str,
    db: Session = Depends(get_backend_db),
) -> list[RAGElement]:
    """Get all RAG elements for a project."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    return [RAGElement(**elem) for elem in elements.values()]


@router.post("/projects/{project_id}/elements", response_model=RAGElement, status_code=201)
async def create_element(
    project_id: str,
    data: RAGElementFormData,
    db: Session = Depends(get_backend_db),
) -> RAGElement:
    """Create a new RAG element."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    # Generate ID and timestamps
    element_id = str(uuid.uuid4())
    now = utc_now().isoformat()

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

    save_rag_config(db, project, rag_config)

    return element


@router.get("/elements/{element_id}", response_model=RAGElement)
async def get_element(
    element_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGElement:
    """Get a single RAG element by ID."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    if element_id not in elements:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

    return RAGElement(**elements[element_id])


@router.put("/elements/{element_id}", response_model=RAGElement)
async def update_element(
    element_id: str,
    data: RAGElementFormData,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGElement:
    """Update an existing RAG element."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    if element_id not in elements:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

    # Update element
    element_data = elements[element_id]
    update_data = data.model_dump(exclude_none=True)
    element_data.update(update_data)
    element_data["updated_at"] = utc_now().isoformat()

    element = RAGElement(**element_data)
    rag_config["elements"][element_id] = element.model_dump()

    save_rag_config(db, project, rag_config)

    return element


@router.delete("/elements/{element_id}", status_code=204)
async def delete_element(
    element_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> None:
    """Delete a RAG element."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    if element_id not in elements:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

    del rag_config["elements"][element_id]

    save_rag_config(db, project, rag_config)


@router.post(
    "/elements/{element_id}/generate-description",
    response_model=GenerateDescriptionResponse,
)
async def generate_description(
    element_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> GenerateDescriptionResponse:
    """Generate AI description for an element."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    if element_id not in elements:
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")

    element = RAGElement(**elements[element_id])
    description = generate_element_description(element)

    return GenerateDescriptionResponse(description=description)


@router.post(
    "/projects/{project_id}/generate-descriptions",
    response_model=BatchGenerateDescriptionsResponse,
)
async def batch_generate_descriptions(
    project_id: str,
    db: Session = Depends(get_backend_db),
) -> BatchGenerateDescriptionsResponse:
    """
    Batch generate descriptions for all elements missing text_description.

    Finds all elements with empty or missing text_description field and
    generates descriptions based on element properties (type, OCR, semantic info).
    """
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    updated_count = 0
    total_missing = 0

    # Find and update elements with missing descriptions
    for _element_id, element_data in elements.items():
        element = RAGElement(**element_data)

        # Check if description is missing or empty
        if not element.text_description or element.text_description.strip() == "":
            total_missing += 1

            # Generate new description
            description = generate_element_description(element)

            # Update element data
            element_data["text_description"] = description
            element_data["updated_at"] = utc_now().isoformat()

            updated_count += 1

    # Save if any updates were made
    if updated_count > 0:
        save_rag_config(db, project, rag_config)

    return BatchGenerateDescriptionsResponse(updated=updated_count, total=total_missing)


# ============================================================================
# State Endpoints
# ============================================================================


@router.get("/projects/{project_id}/states", response_model=list[RAGState])
async def get_states(
    project_id: str,
    db: Session = Depends(get_backend_db),
) -> list[RAGState]:
    """Get all RAG states for a project."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    states = rag_config.get("states", {})
    return [RAGState(**state) for state in states.values()]


@router.post("/projects/{project_id}/states", response_model=RAGState, status_code=201)
async def create_state(
    project_id: str,
    data: RAGStateFormData,
    db: Session = Depends(get_backend_db),
) -> RAGState:
    """Create a new RAG state."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    # Generate ID and timestamps
    state_id = str(uuid.uuid4())
    now = utc_now().isoformat()

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

    save_rag_config(db, project, rag_config)

    return state


@router.get("/states/{state_id}", response_model=RAGState)
async def get_state(
    state_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGState:
    """Get a single RAG state by ID."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    states = rag_config.get("states", {})
    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State {state_id} not found")

    return RAGState(**states[state_id])


@router.put("/states/{state_id}", response_model=RAGState)
async def update_state(
    state_id: str,
    data: RAGStateFormData,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGState:
    """Update an existing RAG state."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    states = rag_config.get("states", {})
    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State {state_id} not found")

    # Update state
    state_data = states[state_id]
    update_data = data.model_dump(exclude_none=True)
    state_data.update(update_data)
    state_data["updated_at"] = utc_now().isoformat()

    state = RAGState(**state_data)
    rag_config["states"][state_id] = state.model_dump()

    save_rag_config(db, project, rag_config)

    return state


@router.delete("/states/{state_id}", status_code=204)
async def delete_state(
    state_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> None:
    """Delete a RAG state."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    states = rag_config.get("states", {})
    if state_id not in states:
        raise HTTPException(status_code=404, detail=f"State {state_id} not found")

    del rag_config["states"][state_id]

    save_rag_config(db, project, rag_config)


# ============================================================================
# Transition Endpoints
# ============================================================================


@router.get("/projects/{project_id}/transitions", response_model=list[RAGTransition])
async def get_transitions(
    project_id: str,
    db: Session = Depends(get_backend_db),
) -> list[RAGTransition]:
    """Get all RAG transitions for a project."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    transitions = rag_config.get("transitions", {})
    return [RAGTransition(**trans) for trans in transitions.values()]


@router.post("/projects/{project_id}/transitions", response_model=RAGTransition, status_code=201)
async def create_transition(
    project_id: str,
    data: RAGTransitionFormData,
    db: Session = Depends(get_backend_db),
) -> RAGTransition:
    """Create a new RAG transition."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    # Generate ID and timestamps
    transition_id = str(uuid.uuid4())
    now = utc_now().isoformat()

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

    save_rag_config(db, project, rag_config)

    return transition


@router.get("/transitions/{transition_id}", response_model=RAGTransition)
async def get_transition(
    transition_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGTransition:
    """Get a single RAG transition by ID."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    transitions = rag_config.get("transitions", {})
    if transition_id not in transitions:
        raise HTTPException(status_code=404, detail=f"Transition {transition_id} not found")

    return RAGTransition(**transitions[transition_id])


@router.put("/transitions/{transition_id}", response_model=RAGTransition)
async def update_transition(
    transition_id: str,
    data: RAGTransitionFormData,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> RAGTransition:
    """Update an existing RAG transition."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    transitions = rag_config.get("transitions", {})
    if transition_id not in transitions:
        raise HTTPException(status_code=404, detail=f"Transition {transition_id} not found")

    # Update transition
    trans_data = transitions[transition_id]
    update_data = data.model_dump(exclude_none=True)
    trans_data.update(update_data)
    trans_data["updated_at"] = utc_now().isoformat()

    transition = RAGTransition(**trans_data)
    rag_config["transitions"][transition_id] = transition.model_dump()

    save_rag_config(db, project, rag_config)

    return transition


@router.delete("/transitions/{transition_id}", status_code=204)
async def delete_transition(
    transition_id: str,
    project_id: str = Query(..., description="Project ID"),
    db: Session = Depends(get_backend_db),
) -> None:
    """Delete a RAG transition."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    transitions = rag_config.get("transitions", {})
    if transition_id not in transitions:
        raise HTTPException(status_code=404, detail=f"Transition {transition_id} not found")

    del rag_config["transitions"][transition_id]

    save_rag_config(db, project, rag_config)


# ============================================================================
# Search Endpoint
# ============================================================================


@router.post("/projects/{project_id}/search", response_model=list[SearchResult])
async def search_elements(
    project_id: str,
    query: SearchQuery,
    db: Session = Depends(get_backend_db),
) -> list[SearchResult]:
    """Search RAG elements using text and semantic similarity."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = rag_config.get("elements", {})
    results = []

    # Simple text-based search (can be enhanced with embeddings later)
    query_lower = query.query.lower()

    for element_data in elements.values():
        element = RAGElement(**element_data)

        # Skip if element type filter is specified and doesn't match
        if query.element_types and element.element_type not in query.element_types:
            continue

        # Skip if state filter is specified and doesn't match
        if query.states and element.state_id not in query.states:
            continue

        # Calculate text match score
        text_score = 0.0
        if query_lower in element.ocr_text.lower():
            text_score = 0.8
        if query_lower in element.text_description.lower():
            text_score = max(text_score, 0.6)
        if query_lower in element.semantic_role.lower():
            text_score = max(text_score, 0.5)

        if text_score > 0:
            results.append(
                SearchResult(
                    element=element,
                    score=text_score,
                    matches=SearchMatch(text=text_score, visual=None, semantic=None),
                )
            )

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Apply limit
    if query.limit:
        results = results[: query.limit]

    return results


# ============================================================================
# Export/Import Endpoints
# ============================================================================


@router.get("/projects/{project_id}/export", response_model=RAGExportData)
async def export_project(
    project_id: str,
    db: Session = Depends(get_backend_db),
) -> RAGExportData:
    """Export RAG data for a project."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

    elements = [RAGElement(**elem) for elem in rag_config.get("elements", {}).values()]
    states = [RAGState(**state) for state in rag_config.get("states", {}).values()]
    transitions = [RAGTransition(**trans) for trans in rag_config.get("transitions", {}).values()]

    return RAGExportData(
        elements=elements,
        states=states,
        transitions=transitions,
        metadata=RAGExportMetadata(
            exported_at=utc_now().isoformat(),
            version="1.0.0",
            project_id=project_id,
        ),
    )


@router.post("/projects/{project_id}/import", response_model=ImportResult)
async def import_project(
    project_id: str,
    data: RAGExportData,
    db: Session = Depends(get_backend_db),
) -> ImportResult:
    """Import RAG data into a project."""
    project = get_project(db, project_id)
    rag_config = get_rag_config(project)

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
        except (ValidationError, KeyError, AttributeError) as e:
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
        except (ValidationError, KeyError, AttributeError) as e:
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
        except (ValidationError, KeyError, AttributeError) as e:
            errors.append(f"Failed to import transition {transition.id}: {str(e)}")

    save_rag_config(db, project, rag_config)

    return ImportResult(imported=imported, skipped=skipped, errors=errors)


# ============================================================================
# RAG Find Endpoint - Vector Matching with SAM3 Segmentation
# Updated: Force reload
# ============================================================================


class RAGFindRequest(BaseModel):
    """Request for RAG element find operation."""

    screenshot_base64: str = Field(..., description="Base64 encoded screenshot image")
    element_id: str | None = Field(
        None, description="Specific element ID to find (optional, deprecated - use element_ids)"
    )
    element_ids: list[str] | None = Field(
        None, description="Specific element IDs to find (optional, filters search)"
    )
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    matching_strategy: str = Field(
        "average", description="Matching strategy: 'average' or 'any_match'"
    )
    use_ocr: bool = Field(False, description="Whether to enable OCR text extraction")
    return_segments: bool = Field(False, description="Return all segments (for visualization)")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results to return")


class RAGFindMatchBoundingBox(BaseModel):
    """Bounding box for a match result."""

    x: int
    y: int
    width: int
    height: int


class RAGFindMatch(BaseModel):
    """A single match from RAG find operation."""

    element_id: str
    element_name: str
    text_description: str | None = None
    visual_similarity: float
    text_similarity: float | None = None
    ocr_similarity: float | None = None
    ocr_text: str | None = None
    bounding_box: RAGFindMatchBoundingBox
    score: float  # Combined score


class RAGFindSegment(BaseModel):
    """A segment from SAM3 segmentation (for visualization)."""

    id: str
    bbox: RAGFindMatchBoundingBox
    mask_data: str | None = None  # Base64 encoded PNG mask (grayscale, 0=transparent, 255=segment)
    mask_density: float
    text_description: str | None = None


class RAGFindResponse(BaseModel):
    """Response from RAG find operation."""

    success: bool
    matches: list[RAGFindMatch]
    segments: list[RAGFindSegment] | None = None
    processing_time_ms: float
    segment_count: int
    error: str | None = None


# Global segment vectorizer (lazy loaded)
_segment_vectorizer: Any = None


def get_segment_vectorizer() -> Any:
    """Get or create the segment vectorizer."""
    global _segment_vectorizer
    if _segment_vectorizer is None:
        try:
            from qontinui.rag.segment_vectorizer import SegmentVectorizer

            _segment_vectorizer = SegmentVectorizer(enable_ocr=False)
        except ImportError:
            _segment_vectorizer = None
    return _segment_vectorizer


@router.post("/projects/{project_id}/find", response_model=RAGFindResponse)
async def find_elements(
    project_id: str,
    request: RAGFindRequest,
    db: Session = Depends(get_backend_db),
) -> RAGFindResponse:
    """
    Find RAG elements in a screenshot using SAM3 segmentation and vector matching.

    This endpoint:
    1. Segments the screenshot using SAM3 (or grid fallback)
    2. Vectorizes each segment using CLIP embeddings
    3. Matches segment vectors against indexed element embeddings
    4. Returns matches with bounding boxes and similarity scores

    The matching uses a priority cascade for similarity threshold:
    - Project default (lowest) < Element setting < Request override (highest)

    Two matching strategies are supported:
    - AVERAGE: Average all element pattern vectors into one query
    - ANY_MATCH: Match if ANY pattern exceeds threshold
    """
    import base64
    import io
    import time

    from PIL import Image as PILImage

    start_time = time.time()

    try:
        # Get project and elements
        project = get_project(db, project_id)
        rag_config = get_rag_config(project)
        elements = rag_config.get("elements", {})

        # Allow segmentation-only mode when return_segments=true and no elements
        segmentation_only = not elements and request.return_segments

        # Decode screenshot
        try:
            screenshot_data = request.screenshot_base64
            if "," in screenshot_data:
                screenshot_data = screenshot_data.split(",")[1]
            screenshot_bytes = base64.b64decode(screenshot_data)
            screenshot = PILImage.open(io.BytesIO(screenshot_bytes))
        except Exception as e:
            return RAGFindResponse(
                success=False,
                matches=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                segment_count=0,
                error=f"Failed to decode screenshot: {str(e)}",
            )

        # Get segment vectorizer
        vectorizer = get_segment_vectorizer()
        if vectorizer is None:
            return RAGFindResponse(
                success=False,
                matches=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                segment_count=0,
                error="Segment vectorizer not available (qontinui library not installed)",
            )

        # Configure OCR if requested
        if request.use_ocr:
            vectorizer.enable_ocr = True

        # Segment and vectorize the screenshot
        import numpy as np

        segment_vectors = vectorizer.vectorize_screenshot(
            screenshot,
            max_segments=100,
            min_confidence=0.5,
        )

        segment_count = len(segment_vectors)

        # Prepare response segments for visualization
        response_segments: list[RAGFindSegment] | None = None
        if request.return_segments:
            response_segments = []
            for i, seg in enumerate(segment_vectors):
                x, y, w, h = seg.bbox
                mask_area = np.sum(seg.mask > 0) if seg.mask is not None else w * h
                total_area = w * h
                mask_density = mask_area / total_area if total_area > 0 else 0.0

                # Encode mask as base64 PNG for frontend visualization
                mask_data_b64: str | None = None
                if seg.mask is not None:
                    try:
                        # Convert mask to uint8 (0-255)
                        mask_uint8 = (seg.mask * 255).astype(np.uint8)
                        mask_img = PILImage.fromarray(mask_uint8, mode="L")
                        mask_buffer = io.BytesIO()
                        mask_img.save(mask_buffer, format="PNG")
                        mask_buffer.seek(0)
                        mask_data_b64 = (
                            f"data:image/png;base64,{base64.b64encode(mask_buffer.read()).decode()}"
                        )
                    except Exception:
                        logger.warning("Failed to encode mask for segment {i}: {mask_err}")

                response_segments.append(
                    RAGFindSegment(
                        id=f"seg_{i}",
                        bbox=RAGFindMatchBoundingBox(x=x, y=y, width=w, height=h),
                        mask_data=mask_data_b64,
                        mask_density=float(mask_density),
                        text_description=seg.text_description,
                    )
                )

        # If segmentation-only mode, return early with just segments
        if segmentation_only:
            return RAGFindResponse(
                success=True,
                matches=[],
                segments=response_segments,
                processing_time_ms=(time.time() - start_time) * 1000,
                segment_count=segment_count,
                error=None,
            )

        # Filter elements if specific IDs requested
        # Support both element_ids (new) and element_id (deprecated, backward compat)
        filter_ids = request.element_ids or ([request.element_id] if request.element_id else None)

        if filter_ids:
            missing_ids = [eid for eid in filter_ids if eid not in elements]
            if missing_ids:
                return RAGFindResponse(
                    success=False,
                    matches=[],
                    segments=response_segments,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    segment_count=segment_count,
                    error=f"Elements not found: {', '.join(missing_ids)}",
                )
            elements_to_search = {eid: elements[eid] for eid in filter_ids}
        else:
            elements_to_search = elements

        # Match elements against segments
        from qontinui.rag.segment_vectorizer import MatchingStrategy

        strategy = (
            MatchingStrategy.ANY_MATCH
            if request.matching_strategy == "any_match"
            else MatchingStrategy.AVERAGE
        )

        matches: list[RAGFindMatch] = []

        for element_id, element_data in elements_to_search.items():
            element = RAGElement(**element_data)

            # Get element embedding (prefer image_embedding for visual matching)
            element_embedding = element.image_embedding
            if not element_embedding:
                # Fall back to text embedding
                element_embedding = element.text_embedding
            if not element_embedding:
                continue

            # Resolve threshold (element setting or request override)
            threshold = request.similarity_threshold
            if element.similarity_threshold and element.similarity_threshold > 0:
                threshold = element.similarity_threshold
            # Request threshold takes highest priority
            if request.similarity_threshold != 0.7:  # User specified custom threshold
                threshold = request.similarity_threshold

            # Get OCR filter text from element
            ocr_filter_text = element.ocr_text if request.use_ocr else None

            # Find matches based on strategy
            if strategy == MatchingStrategy.ANY_MATCH:
                match = vectorizer.find_any_match(
                    pattern_embeddings=[element_embedding],
                    segments=segment_vectors,
                    threshold=threshold,
                )
                element_matches = [match] if match else []
            else:
                element_matches = vectorizer.find_matches(
                    query_embedding=element_embedding,
                    segments=segment_vectors,
                    threshold=threshold,
                    strategy=strategy,
                    ocr_filter=ocr_filter_text,
                )

            # Add matches to results
            for match in element_matches[: request.max_results]:
                x, y, w, h = match.segment.bbox
                matches.append(
                    RAGFindMatch(
                        element_id=element_id,
                        element_name=(
                            element.ocr_text or element.text_description[:50]
                            if element.text_description
                            else element_id[:8]
                        ),
                        text_description=element.text_description,
                        visual_similarity=match.visual_similarity,
                        text_similarity=match.text_similarity,
                        ocr_similarity=match.ocr_similarity,
                        ocr_text=match.ocr_text,
                        bounding_box=RAGFindMatchBoundingBox(x=x, y=y, width=w, height=h),
                        score=match.combined_score,
                    )
                )

        # Sort by combined score and limit results
        matches.sort(key=lambda m: m.score, reverse=True)
        matches = matches[: request.max_results]

        processing_time_ms = (time.time() - start_time) * 1000

        return RAGFindResponse(
            success=True,
            matches=matches,
            segments=response_segments,
            processing_time_ms=processing_time_ms,
            segment_count=segment_count,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return RAGFindResponse(
            success=False,
            matches=[],
            processing_time_ms=(time.time() - start_time) * 1000,
            segment_count=0,
            error=f"Find operation failed: {str(e)}",
        )
