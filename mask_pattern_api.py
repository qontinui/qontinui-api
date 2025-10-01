"""
Mask and Pattern API endpoints for State Discovery and Pattern Optimization
"""

import base64
import io
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel, Field

# Import qontinui mask and pattern modules
from qontinui.masks import MaskGenerator, MaskType
from qontinui.patterns import Pattern

router = APIRouter(prefix="/masks", tags=["masks"])

# Storage for patterns (in production, use a proper database)
_pattern_storage: dict[str, Pattern] = {}
_mask_generator = MaskGenerator()


# Request/Response Models


class GenerateMaskRequest(BaseModel):
    """Request to generate a mask for an image."""

    image_data: str  # Base64 encoded image
    mask_type: str = "full"
    stability_threshold: float | None = 0.95
    edge_low_threshold: int | None = 50
    edge_high_threshold: int | None = 150
    saliency_threshold: float | None = 0.5
    alpha_threshold: float | None = 0.5


class GenerateMaskResponse(BaseModel):
    """Response containing generated mask."""

    mask_data: str  # Base64 encoded mask image
    metadata: dict[str, Any]
    density: float
    active_pixels: int
    total_pixels: int


class OptimizeMaskRequest(BaseModel):
    """Request to optimize a mask based on samples."""

    pattern_id: str
    positive_samples: list[str]  # Base64 encoded images
    negative_samples: list[str] | None = None
    method: str = "stability"  # "stability" or "discriminative"


class OptimizeMaskResponse(BaseModel):
    """Response containing optimized mask."""

    pattern_id: str
    mask_data: str  # Base64 encoded mask
    optimization_metrics: dict[str, Any]
    mask_density: float


class CreatePatternRequest(BaseModel):
    """Request to create a new Pattern."""

    name: str
    image_data: str  # Base64 encoded image
    mask_data: str | None = None  # Optional base64 encoded mask
    tags: list[str] = Field(default_factory=list)
    similarity_threshold: float = 0.95
    use_color: bool = True


class CreatePatternResponse(BaseModel):
    """Response containing created pattern details."""

    pattern_id: str
    name: str
    width: int
    height: int
    mask_density: float
    created_at: str


class PatternDetailsResponse(BaseModel):
    """Response containing pattern details."""

    id: str
    name: str
    width: int
    height: int
    mask_density: float
    mask_type: str
    tags: list[str]
    created_at: str
    updated_at: str
    similarity_threshold: float
    use_color: bool
    match_count: int
    success_rate: float
    avg_match_time: float
    active_pixels: int
    total_pixels: int
    variation_count: int
    optimization_count: int


class CalculateSimilarityRequest(BaseModel):
    """Request to calculate similarity between patterns."""

    pattern_id: str
    compare_image: str  # Base64 encoded image
    compare_mask: str | None = None  # Optional base64 encoded mask


class CalculateSimilarityResponse(BaseModel):
    """Response containing similarity score."""

    pattern_id: str
    similarity: float
    match_threshold: float
    is_match: bool


class AddVariationRequest(BaseModel):
    """Request to add a variation to a pattern."""

    pattern_id: str
    variation_image: str  # Base64 encoded image


class RefineMaskRequest(BaseModel):
    """Request to refine an existing mask."""

    mask_data: str  # Base64 encoded mask
    operation: str  # "erode", "dilate", "smooth", "threshold"
    strength: float = 1.0
    threshold: float | None = 0.5


class RefineMaskResponse(BaseModel):
    """Response containing refined mask."""

    mask_data: str  # Base64 encoded refined mask
    operation: str
    strength: float


class CombineMasksRequest(BaseModel):
    """Request to combine multiple masks."""

    mask_data_list: list[str]  # List of base64 encoded masks
    operation: str = "union"  # "union", "intersection", "weighted"
    weights: list[float] | None = None


class CombineMasksResponse(BaseModel):
    """Response containing combined mask."""

    mask_data: str  # Base64 encoded combined mask
    operation: str
    mask_count: int


# Helper Functions


def base64_to_numpy(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    image_bytes = base64.b64decode(base64_str)
    image = PILImage.open(io.BytesIO(image_bytes))
    return np.array(image)


def numpy_to_base64(array: np.ndarray, is_mask: bool = False) -> str:
    """Convert numpy array to base64 string."""
    if is_mask:
        # Convert mask (0.0-1.0) to grayscale image (0-255)
        array = (array * 255).astype(np.uint8)
        image = PILImage.fromarray(array, mode="L")
    else:
        if len(array.shape) == 2:
            image = PILImage.fromarray(array, mode="L")
        else:
            image = PILImage.fromarray(array, mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# API Endpoints


@router.post("/generate", response_model=GenerateMaskResponse)
async def generate_mask(request: GenerateMaskRequest):
    """Generate a mask for an image using various techniques."""
    try:
        # Convert base64 to numpy
        image = base64_to_numpy(request.image_data)

        # Parse mask type
        try:
            mask_type = MaskType(request.mask_type)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid mask type: {request.mask_type}"
            ) from e

        # Generate mask with appropriate parameters
        kwargs = {}
        if mask_type == MaskType.STABILITY:
            kwargs["stability_threshold"] = request.stability_threshold
        elif mask_type == MaskType.EDGE:
            kwargs["low_threshold"] = request.edge_low_threshold
            kwargs["high_threshold"] = request.edge_high_threshold
        elif mask_type == MaskType.SALIENCY:
            kwargs["threshold"] = request.saliency_threshold
        elif mask_type == MaskType.TRANSPARENCY:
            kwargs["alpha_threshold"] = request.alpha_threshold

        mask, metadata = _mask_generator.generate_mask(image, mask_type, **kwargs)

        # Convert mask to base64
        mask_base64 = numpy_to_base64(mask, is_mask=True)

        return GenerateMaskResponse(
            mask_data=mask_base64,
            metadata=metadata.generation_params,
            density=metadata.density,
            active_pixels=metadata.active_pixels,
            total_pixels=metadata.total_pixels,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/patterns/create", response_model=CreatePatternResponse)
async def create_pattern(request: CreatePatternRequest):
    """Create a new Pattern."""
    try:
        # Convert base64 to numpy
        image = base64_to_numpy(request.image_data)

        # Create or use provided mask
        if request.mask_data:
            mask = base64_to_numpy(request.mask_data)
            mask = mask.astype(np.float32) / 255.0  # Normalize to 0.0-1.0
        else:
            mask = np.ones(image.shape[:2], dtype=np.float32)

        # Generate unique ID
        pattern_id = f"pattern_{len(_pattern_storage)}_{int(datetime.now().timestamp())}"

        # Create Pattern
        pattern = Pattern(
            id=pattern_id,
            name=request.name,
            pixel_data=image,
            mask=mask,
            tags=request.tags,
            similarity_threshold=request.similarity_threshold,
            use_color=request.use_color,
        )

        # Store pattern
        _pattern_storage[pattern_id] = pattern

        return CreatePatternResponse(
            pattern_id=pattern_id,
            name=pattern.name,
            width=pattern.width,
            height=pattern.height,
            mask_density=pattern.mask_density,
            created_at=pattern.created_at.isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/patterns/{pattern_id}", response_model=PatternDetailsResponse)
async def get_pattern(pattern_id: str):
    """Get details of a specific pattern."""
    if pattern_id not in _pattern_storage:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    pattern = _pattern_storage[pattern_id]
    return PatternDetailsResponse(**pattern.to_dict())


@router.get("/patterns", response_model=list[PatternDetailsResponse])
async def list_patterns():
    """List all patterns."""
    return [PatternDetailsResponse(**pattern.to_dict()) for pattern in _pattern_storage.values()]


@router.post("/patterns/{pattern_id}/optimize", response_model=OptimizeMaskResponse)
async def optimize_pattern_mask(pattern_id: str, request: OptimizeMaskRequest):
    """Optimize a pattern's mask based on positive and negative samples."""
    try:
        if pattern_id not in _pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        pattern = _pattern_storage[pattern_id]

        # Convert samples from base64
        positive_samples = [base64_to_numpy(img) for img in request.positive_samples]
        negative_samples = None
        if request.negative_samples:
            negative_samples = [base64_to_numpy(img) for img in request.negative_samples]

        # Optimize mask
        optimized_mask, metrics = pattern.optimize_mask(
            positive_samples, negative_samples, method=request.method
        )

        # Update pattern mask
        pattern.update_mask(optimized_mask, record_history=True)

        # Convert mask to base64
        mask_base64 = numpy_to_base64(optimized_mask, is_mask=True)

        return OptimizeMaskResponse(
            pattern_id=pattern_id,
            mask_data=mask_base64,
            optimization_metrics=metrics,
            mask_density=pattern.mask_density,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/patterns/{pattern_id}/similarity", response_model=CalculateSimilarityResponse)
async def calculate_similarity(pattern_id: str, request: CalculateSimilarityRequest):
    """Calculate similarity between a pattern and an image."""
    try:
        if pattern_id not in _pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        pattern = _pattern_storage[pattern_id]

        # Convert image from base64
        compare_image = base64_to_numpy(request.compare_image)

        # Convert mask if provided
        compare_mask = None
        if request.compare_mask:
            compare_mask = base64_to_numpy(request.compare_mask)
            compare_mask = compare_mask.astype(np.float32) / 255.0

        # Calculate similarity
        similarity = pattern.calculate_similarity(compare_image, compare_mask)

        return CalculateSimilarityResponse(
            pattern_id=pattern_id,
            similarity=similarity,
            match_threshold=pattern.similarity_threshold,
            is_match=similarity >= pattern.similarity_threshold,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/patterns/{pattern_id}/variations", response_model=dict[str, Any])
async def add_variation(pattern_id: str, request: AddVariationRequest):
    """Add a variation image to a pattern for optimization."""
    try:
        if pattern_id not in _pattern_storage:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

        pattern = _pattern_storage[pattern_id]

        # Convert variation from base64
        variation_image = base64_to_numpy(request.variation_image)

        # Add variation
        pattern.add_variation(variation_image)

        return {
            "pattern_id": pattern_id,
            "variation_count": len(pattern.variations),
            "message": "Variation added successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/refine", response_model=RefineMaskResponse)
async def refine_mask(request: RefineMaskRequest):
    """Refine a mask using morphological operations."""
    try:
        # Convert mask from base64
        mask = base64_to_numpy(request.mask_data)
        mask = mask.astype(np.float32) / 255.0

        # Refine mask
        kwargs = {}
        if request.operation == "threshold":
            kwargs["threshold"] = request.threshold

        refined_mask = _mask_generator.refine_mask(
            mask, request.operation, request.strength, **kwargs
        )

        # Convert back to base64
        mask_base64 = numpy_to_base64(refined_mask, is_mask=True)

        return RefineMaskResponse(
            mask_data=mask_base64, operation=request.operation, strength=request.strength
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/combine", response_model=CombineMasksResponse)
async def combine_masks(request: CombineMasksRequest):
    """Combine multiple masks using various operations."""
    try:
        # Convert masks from base64
        masks = []
        for mask_data in request.mask_data_list:
            mask = base64_to_numpy(mask_data)
            mask = mask.astype(np.float32) / 255.0
            masks.append(mask)

        # Combine masks
        combined_mask = _mask_generator.combine_masks(
            masks, operation=request.operation, weights=request.weights
        )

        # Convert back to base64
        mask_base64 = numpy_to_base64(combined_mask, is_mask=True)

        return CombineMasksResponse(
            mask_data=mask_base64, operation=request.operation, mask_count=len(masks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/patterns/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete a pattern."""
    if pattern_id not in _pattern_storage:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    del _pattern_storage[pattern_id]
    return {"message": f"Pattern {pattern_id} deleted successfully"}


@router.get("/patterns/{pattern_id}/mask")
async def get_pattern_mask(pattern_id: str):
    """Get the mask of a pattern as base64 image."""
    if pattern_id not in _pattern_storage:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    pattern = _pattern_storage[pattern_id]
    mask_base64 = numpy_to_base64(pattern.mask, is_mask=True)

    return {
        "pattern_id": pattern_id,
        "mask_data": mask_base64,
        "mask_density": pattern.mask_density,
        "active_pixels": pattern.active_pixel_count,
        "total_pixels": pattern.total_pixel_count,
    }


@router.get("/patterns/{pattern_id}/image")
async def get_pattern_image(pattern_id: str):
    """Get the image of a pattern as base64."""
    if pattern_id not in _pattern_storage:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    pattern = _pattern_storage[pattern_id]
    image_base64 = numpy_to_base64(pattern.pixel_data)

    return {
        "pattern_id": pattern_id,
        "image_data": image_base64,
        "width": pattern.width,
        "height": pattern.height,
    }
