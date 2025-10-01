"""
API endpoints for masked pattern extraction and optimization.
This module handles pattern extraction with confidence-based masking and pixel averaging.
"""

import base64
import io
import logging
import uuid
from datetime import datetime
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/masked-patterns", tags=["Masked Patterns"])


# Request/Response Models
class MorphologicalOps(BaseModel):
    enabled: bool = True
    erosionSize: int = 1
    dilationSize: int = 2


class PatternExtractionConfig(BaseModel):
    similarityThreshold: float = 0.85  # Pixels below this confidence are masked out
    minActivePixels: int = 100  # Minimum active pixels required
    colorAveraging: Literal["mean", "median", "mode", "weighted"] = "weighted"
    useAlphaChannel: bool = False
    morphologicalOps: MorphologicalOps = MorphologicalOps()


class ExtractMaskedPatternRequest(BaseModel):
    state_image_id: str
    pattern_name: str
    config: PatternExtractionConfig
    screenshots: list[str] | None = None  # Base64 encoded screenshots
    regions: list[dict[str, float]] | None = None  # Regions for each screenshot


class UpdateThresholdRequest(BaseModel):
    similarity_threshold: float


class MaskedPatternResponse(BaseModel):
    id: str
    name: str
    width: int
    height: int
    similarityThreshold: float
    maskDensity: float
    activePixels: int
    totalPixels: int
    minConfidence: float
    maxConfidence: float
    avgConfidence: float
    stdDevConfidence: float
    createdAt: str
    updatedAt: str
    matchCount: int = 0
    successRate: float = 0.0
    avgMatchTime: float = 0.0


class MaskedPatternExtractor:
    """Handles extraction of masked patterns with pixel averaging."""

    def extract_masked_pattern(
        self,
        screenshots: list[np.ndarray],
        region: tuple,  # (x, y, width, height)
        config: PatternExtractionConfig,
    ) -> dict[str, Any]:
        """
        Extract a masked pattern from multiple screenshots.

        1. Calculate pixel-wise confidence across all screenshots
        2. Create binary mask based on similarity threshold
        3. Average pixel values for high-confidence pixels
        4. Apply morphological operations if enabled
        """
        x, y, w, h = region

        # Extract regions from all screenshots
        regions = []
        for screenshot in screenshots:
            region_img = screenshot[y : y + h, x : x + w]
            regions.append(region_img)

        if len(regions) == 0:
            raise ValueError("No regions extracted")

        # Calculate pixel-wise statistics
        regions_array = np.array(regions)

        # Calculate confidence map (inverse of variance)
        if len(regions) > 1:
            # Calculate standard deviation per pixel
            std_per_pixel = np.std(regions_array, axis=0)

            # Convert to confidence (0-1, where 1 is perfect match across all)
            # Use coefficient of variation for RGB channels
            mean_per_pixel = np.mean(regions_array, axis=0)

            # Avoid division by zero
            mean_per_pixel = np.where(mean_per_pixel == 0, 1, mean_per_pixel)

            # Calculate confidence as inverse of coefficient of variation
            coeff_var = std_per_pixel / mean_per_pixel

            # Average across color channels for overall confidence
            if len(coeff_var.shape) == 3:  # RGB image
                confidence_map = 1 - np.mean(coeff_var, axis=2)
            else:
                confidence_map = 1 - coeff_var

            # Normalize to 0-1 range
            confidence_map = np.clip(confidence_map, 0, 1)
        else:
            # Single screenshot - assume high confidence
            confidence_map = np.ones((h, w), dtype=np.float32)

        # Create binary mask based on threshold
        binary_mask = (confidence_map >= config.similarityThreshold).astype(np.uint8)

        # Apply morphological operations if enabled
        if config.morphologicalOps.enabled:
            # Remove noise (erosion)
            if config.morphologicalOps.erosionSize > 0:
                kernel = np.ones(
                    (config.morphologicalOps.erosionSize, config.morphologicalOps.erosionSize),
                    np.uint8,
                )
                binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

            # Fill gaps (dilation)
            if config.morphologicalOps.dilationSize > 0:
                kernel = np.ones(
                    (config.morphologicalOps.dilationSize, config.morphologicalOps.dilationSize),
                    np.uint8,
                )
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Calculate averaged pixels based on config
        if config.colorAveraging == "mean":
            averaged_pixels = np.mean(regions_array, axis=0)
        elif config.colorAveraging == "median":
            averaged_pixels = np.median(regions_array, axis=0)
        elif config.colorAveraging == "weighted":
            # Weight by confidence
            weights = (
                confidence_map[:, :, np.newaxis]
                if len(regions_array.shape) == 4
                else confidence_map
            )
            weighted_sum = np.zeros_like(regions_array[0], dtype=np.float64)
            weight_sum = np.zeros_like(confidence_map, dtype=np.float64)

            for _i, region in enumerate(regions):
                weighted_sum += region * weights
                weight_sum += weights[:, :, 0] if len(weights.shape) == 3 else weights

            weight_sum = np.where(weight_sum == 0, 1, weight_sum)  # Avoid division by zero
            if len(weighted_sum.shape) == 3:
                weight_sum = weight_sum[:, :, np.newaxis]
            averaged_pixels = weighted_sum / weight_sum
        else:  # mode
            # For mode, we need to discretize colors and find most common
            # This is computationally expensive, so we'll use a simplified approach
            averaged_pixels = np.mean(regions_array, axis=0)

        # Convert to uint8
        averaged_pixels = np.clip(averaged_pixels, 0, 255).astype(np.uint8)

        # Apply mask to averaged pixels (set masked pixels to transparent or zero)
        if len(averaged_pixels.shape) == 3:  # RGB
            # Add alpha channel if not present
            if averaged_pixels.shape[2] == 3:
                averaged_pixels = cv2.cvtColor(averaged_pixels, cv2.COLOR_RGB2RGBA)
            # Set alpha to 0 for masked pixels
            averaged_pixels[:, :, 3] = binary_mask * 255

        # Calculate statistics
        active_pixels = int(np.sum(binary_mask))
        total_pixels = int(w * h)
        mask_density = active_pixels / total_pixels if total_pixels > 0 else 0

        # Calculate confidence statistics
        masked_confidence = confidence_map[binary_mask == 1] if active_pixels > 0 else np.array([0])

        return {
            "mask": binary_mask,
            "averaged_pixels": averaged_pixels,
            "confidence_map": confidence_map,
            "statistics": {
                "width": w,
                "height": h,
                "activePixels": active_pixels,
                "totalPixels": total_pixels,
                "maskDensity": mask_density,
                "minConfidence": float(np.min(masked_confidence)),
                "maxConfidence": float(np.max(masked_confidence)),
                "avgConfidence": float(np.mean(masked_confidence)),
                "stdDevConfidence": float(np.std(masked_confidence)),
            },
        }


# Global extractor instance
extractor = MaskedPatternExtractor()

# In-memory storage for demo (replace with database in production)
patterns_storage = {}


@router.post("/extract-masked", response_model=MaskedPatternResponse)
async def extract_masked_pattern(request: ExtractMaskedPatternRequest):
    """Extract a masked pattern from screenshots with pixel averaging."""

    logger.info(f"Received extraction request for pattern: {request.pattern_name}")
    start_time = datetime.now()

    try:
        # Decode screenshots if provided
        screenshots = []
        if request.screenshots:
            logger.info(f"Processing {len(request.screenshots)} screenshots")
            for i, screenshot_b64 in enumerate(request.screenshots):
                img_data = base64.b64decode(screenshot_b64)
                img = PILImage.open(io.BytesIO(img_data))
                img_array = np.array(img)
                screenshots.append(img_array)
                logger.info(
                    f"Decoded screenshot {i+1}/{len(request.screenshots)}, shape: {img_array.shape}"
                )
        else:
            # Mock screenshots for demo
            screenshots = [
                np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8) for _ in range(3)
            ]

        # Use provided regions or mock for demo
        if request.regions and len(request.regions) > 0:
            # Use the first region (they should all be the same in our simplified approach)
            r = request.regions[0]
            region = (int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"]))
        else:
            # Mock region for demo
            region = (100, 100, 200, 150)  # x, y, width, height

        # Extract pattern
        logger.info(f"Starting pattern extraction for region {region}")
        result = extractor.extract_masked_pattern(screenshots, region, request.config)
        logger.info("Pattern extraction complete")

        # Create pattern object
        pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
        pattern = MaskedPatternResponse(
            id=pattern_id,
            name=request.pattern_name,
            width=result["statistics"]["width"],
            height=result["statistics"]["height"],
            similarityThreshold=request.config.similarityThreshold,
            maskDensity=result["statistics"]["maskDensity"],
            activePixels=result["statistics"]["activePixels"],
            totalPixels=result["statistics"]["totalPixels"],
            minConfidence=result["statistics"]["minConfidence"],
            maxConfidence=result["statistics"]["maxConfidence"],
            avgConfidence=result["statistics"]["avgConfidence"],
            stdDevConfidence=result["statistics"]["stdDevConfidence"],
            createdAt=datetime.now().isoformat(),
            updatedAt=datetime.now().isoformat(),
        )

        # Store pattern with additional data (convert to lists in background)
        # Don't block the response
        patterns_storage[pattern_id] = {
            **pattern.dict(),
            "mask": result["mask"],  # Keep as numpy array for now
            "averaged_pixels": result["averaged_pixels"],  # Keep as numpy array
            "confidence_map": result["confidence_map"],  # Keep as numpy array
        }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Extracted masked pattern: {pattern_id} in {elapsed:.2f} seconds")
        logger.info(
            f"Pattern statistics: active_pixels={pattern.activePixels}, mask_density={pattern.maskDensity:.2%}"
        )
        return pattern

    except Exception as e:
        logger.error(f"Failed to extract masked pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/masked", response_model=list[MaskedPatternResponse])
async def get_masked_patterns():
    """Get all masked patterns."""

    patterns = []
    for _pattern_id, pattern_data in patterns_storage.items():
        # Return pattern metadata without the actual mask data
        patterns.append(
            MaskedPatternResponse(
                id=pattern_data["id"],
                name=pattern_data["name"],
                width=pattern_data["width"],
                height=pattern_data["height"],
                similarityThreshold=pattern_data["similarityThreshold"],
                maskDensity=pattern_data["maskDensity"],
                activePixels=pattern_data["activePixels"],
                totalPixels=pattern_data["totalPixels"],
                minConfidence=pattern_data["minConfidence"],
                maxConfidence=pattern_data["maxConfidence"],
                avgConfidence=pattern_data["avgConfidence"],
                stdDevConfidence=pattern_data["stdDevConfidence"],
                createdAt=pattern_data["createdAt"],
                updatedAt=pattern_data["updatedAt"],
                matchCount=pattern_data.get("matchCount", 0),
                successRate=pattern_data.get("successRate", 0.0),
                avgMatchTime=pattern_data.get("avgMatchTime", 0.0),
            )
        )

    return patterns


@router.get("/{pattern_id}")
async def get_pattern_details(pattern_id: str):
    """Get detailed pattern data including mask and averaged pixels."""

    if pattern_id not in patterns_storage:
        raise HTTPException(status_code=404, detail="Pattern not found")

    pattern_data = patterns_storage[pattern_id].copy()

    # Convert arrays to base64 encoded images
    # Check if data is already numpy array or needs conversion
    mask_data = pattern_data["mask"]
    if isinstance(mask_data, np.ndarray):
        mask = mask_data.astype(np.uint8) * 255
    else:
        mask = np.array(mask_data, dtype=np.uint8) * 255

    averaged_pixels_data = pattern_data["averaged_pixels"]
    if isinstance(averaged_pixels_data, np.ndarray):
        averaged_pixels = averaged_pixels_data.astype(np.uint8)
    else:
        averaged_pixels = np.array(averaged_pixels_data, dtype=np.uint8)

    confidence_data = pattern_data["confidence_map"]
    if isinstance(confidence_data, np.ndarray):
        confidence_map = (confidence_data * 255).astype(np.uint8)
    else:
        confidence_map = (np.array(confidence_data) * 255).astype(np.uint8)

    # Convert to base64
    def array_to_base64(arr):
        img = PILImage.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Remove numpy arrays from response
    response_data = {
        k: v
        for k, v in pattern_data.items()
        if k not in ["mask", "averaged_pixels", "confidence_map"]
    }

    return {
        **response_data,
        "mask_image": f"data:image/png;base64,{array_to_base64(mask)}",
        "pattern_image": f"data:image/png;base64,{array_to_base64(averaged_pixels)}",
        "confidence_image": f"data:image/png;base64,{array_to_base64(confidence_map)}",
    }


@router.post("/{pattern_id}/update-threshold")
async def update_pattern_threshold(pattern_id: str, request: UpdateThresholdRequest):
    """Update the similarity threshold and regenerate the mask."""

    if pattern_id not in patterns_storage:
        raise HTTPException(status_code=404, detail="Pattern not found")

    pattern_data = patterns_storage[pattern_id]

    # Recalculate mask with new threshold
    confidence_map = np.array(pattern_data["confidence_map"])
    new_mask = (confidence_map >= request.similarity_threshold).astype(np.uint8)

    # Update statistics
    active_pixels = int(np.sum(new_mask))
    total_pixels = pattern_data["totalPixels"]
    mask_density = active_pixels / total_pixels if total_pixels > 0 else 0

    masked_confidence = confidence_map[new_mask == 1] if active_pixels > 0 else np.array([0])

    # Update pattern data
    pattern_data.update(
        {
            "similarityThreshold": request.similarity_threshold,
            "mask": new_mask.tolist(),
            "activePixels": active_pixels,
            "maskDensity": mask_density,
            "minConfidence": float(np.min(masked_confidence)),
            "maxConfidence": float(np.max(masked_confidence)),
            "avgConfidence": float(np.mean(masked_confidence)),
            "stdDevConfidence": float(np.std(masked_confidence)),
            "updatedAt": datetime.now().isoformat(),
        }
    )

    logger.info(f"Updated threshold for pattern {pattern_id} to {request.similarity_threshold}")

    return {
        "pattern_id": pattern_id,
        "new_threshold": request.similarity_threshold,
        "new_active_pixels": active_pixels,
        "new_mask_density": mask_density,
        "message": "Pattern threshold updated successfully",
    }


@router.delete("/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete a masked pattern."""

    if pattern_id not in patterns_storage:
        raise HTTPException(status_code=404, detail="Pattern not found")

    del patterns_storage[pattern_id]
    logger.info(f"Deleted pattern: {pattern_id}")

    return {"message": f"Pattern {pattern_id} deleted successfully"}
