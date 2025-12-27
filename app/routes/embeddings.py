"""API endpoints for computing embeddings.

This module provides HTTP endpoints for computing CLIP and text embeddings
for images and text descriptions. These are used by qontinui-web during
export to populate RAG fields for StateImages.
"""

import base64
import io
import logging
import time
from typing import Any

from fastapi import APIRouter
from PIL import Image as PILImage
# Import shared schemas from qontinui-schemas
from qontinui_schemas.api.rag import (BatchComputeEmbeddingRequest,
                                      BatchComputeEmbeddingResponse,
                                      BatchEmbeddingResult,
                                      ComputeEmbeddingRequest,
                                      ComputeEmbeddingResponse,
                                      ComputeTextEmbeddingRequest,
                                      ComputeTextEmbeddingResponse)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# =============================================================================
# Lazy-loaded models (to avoid loading at import time)
# =============================================================================

_clip_model: Any = None
_text_model: Any = None
_ocr_engine: Any = None


def get_clip_model() -> Any:
    """Get or initialize CLIP model."""
    global _clip_model
    if _clip_model is None:
        try:
            from qontinui.rag import CLIPEmbedder

            _clip_model = CLIPEmbedder(model_name="openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded successfully")
        except ImportError:
            logger.warning("CLIPEmbedder not available, using fallback")
            _clip_model = "fallback"
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            _clip_model = "fallback"
    return _clip_model


def get_text_model() -> Any:
    """Get or initialize text embedding model."""
    global _text_model
    if _text_model is None:
        try:
            from qontinui.rag import TextEmbedder

            _text_model = TextEmbedder(model_name="all-MiniLM-L6-v2")
            logger.info("Text embedding model loaded successfully")
        except ImportError:
            logger.warning("TextEmbedder not available, using fallback")
            _text_model = "fallback"
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            _text_model = "fallback"
    return _text_model


def get_ocr_engine() -> Any:
    """Get or initialize OCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            import easyocr

            _ocr_engine = easyocr.Reader(["en"], gpu=False)
            logger.info("OCR engine loaded successfully")
        except ImportError:
            logger.warning("easyocr not available, OCR disabled")
            _ocr_engine = "fallback"
        except Exception as e:
            logger.error(f"Failed to load OCR engine: {e}")
            _ocr_engine = "fallback"
    return _ocr_engine


# =============================================================================
# Helper Functions
# =============================================================================


def decode_base64_image(image_data: str) -> PILImage.Image:
    """Decode base64 image data to PIL Image."""
    # Handle data URL format
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    return PILImage.open(io.BytesIO(image_bytes))


def compute_clip_embedding(image: PILImage.Image) -> list[float] | None:
    """Compute CLIP embedding for an image."""
    model = get_clip_model()
    if model == "fallback":
        # Return a dummy embedding for testing
        return [0.0] * 512

    try:
        embedding = model.encode_image(image)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    except Exception as e:
        logger.error(f"Failed to compute CLIP embedding: {e}")
        return None


def compute_text_embedding(text: str) -> list[float] | None:
    """Compute text embedding using sentence transformer (384 dim)."""
    if not text or not text.strip():
        return None

    model = get_text_model()
    if model == "fallback":
        # Return a dummy embedding for testing
        return [0.0] * 384

    try:
        embedding = model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    except Exception as e:
        logger.error(f"Failed to compute text embedding: {e}")
        return None


def compute_clip_text_embedding(text: str) -> list[float] | None:
    """Compute CLIP text embedding (512 dim) for semantic search.

    Uses CLIP's text encoder to produce embeddings in the same space as
    CLIP image embeddings, enabling text-to-image similarity search.
    """
    if not text or not text.strip():
        return None

    model = get_clip_model()
    if model == "fallback":
        # Return a dummy embedding for testing
        return [0.0] * 512

    try:
        embedding = model.encode_text(text)
        return embedding if isinstance(embedding, list) else list(embedding)
    except Exception as e:
        logger.error(f"Failed to compute CLIP text embedding: {e}")
        return None


def generate_text_description(
    name: str | None = None,
    ocr_text: str | None = None,
) -> str | None:
    """Generate a natural language description for an element.

    Combines the element name and OCR text into a searchable description.

    Args:
        name: Optional element name (e.g., "login-button", "submit-form")
        ocr_text: Optional OCR-extracted text from the image

    Returns:
        Natural language description or None if no text available
    """
    parts = []

    # Add element name if available (convert kebab-case to natural language)
    if name:
        # Convert kebab-case/snake_case to readable text
        readable_name = name.replace("-", " ").replace("_", " ").strip()
        if readable_name:
            parts.append(readable_name)

    # Add OCR text if available and different from name
    if ocr_text:
        ocr_clean = ocr_text.strip()
        # Only add if it provides additional info
        if ocr_clean and (not name or ocr_clean.lower() not in name.lower()):
            parts.append(f'with text "{ocr_clean}"')

    if not parts:
        return None

    return " ".join(parts)


def extract_ocr_text(image: PILImage.Image) -> tuple[str | None, float | None]:
    """Extract text from image using OCR.

    Returns:
        Tuple of (extracted_text, confidence_score)
    """
    engine = get_ocr_engine()
    if engine == "fallback":
        return None, None

    try:
        import numpy as np

        # Convert PIL to numpy array
        image_np = np.array(image)

        # Run OCR
        results = engine.readtext(image_np)

        if not results:
            return None, None

        # Combine all detected text
        texts = []
        confidences = []
        for _bbox, text, confidence in results:
            texts.append(text)
            confidences.append(confidence)

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return None, None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/compute", response_model=ComputeEmbeddingResponse)
async def compute_embedding(request: ComputeEmbeddingRequest) -> ComputeEmbeddingResponse:
    """Compute embeddings for a single image.

    This endpoint:
    1. Decodes the base64 image
    2. Computes CLIP image embedding (512 dimensions)
    3. Optionally extracts OCR text
    4. Optionally computes text embedding (384 dimensions)

    The embeddings are returned and can be stored in StateImage fields
    for RAG-based semantic search.
    """
    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image_data)

        # Compute CLIP embedding
        image_embedding = compute_clip_embedding(image)

        # Extract OCR if requested
        ocr_text = None
        ocr_confidence = None
        if request.compute_text_embedding:
            ocr_text, ocr_confidence = extract_ocr_text(image)

        # Generate text description from name and OCR
        text_description = generate_text_description(
            name=request.text_description,  # text_description is used as name
            ocr_text=ocr_text,
        )

        # Use description for text embedding, fallback to provided description or OCR
        text_for_embedding = text_description or request.text_description or ocr_text

        # Compute text embedding
        text_embedding = None
        if text_for_embedding:
            text_embedding = compute_text_embedding(text_for_embedding)

        processing_time_ms = (time.time() - start_time) * 1000

        return ComputeEmbeddingResponse(
            success=True,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            text_description=text_description,
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Embedding computation failed: {e}", exc_info=True)
        return ComputeEmbeddingResponse(
            success=False,
            error=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.post("/compute-text", response_model=ComputeTextEmbeddingResponse)
async def compute_text_embedding_endpoint(
    request: ComputeTextEmbeddingRequest,
) -> ComputeTextEmbeddingResponse:
    """Compute CLIP text embedding for semantic search.

    This endpoint encodes text using CLIP's text encoder, producing a
    512-dimensional embedding in the same space as CLIP image embeddings.
    This enables semantic text-to-image search against indexed visual content.

    Use this for:
    - Searching visual content by text description
    - Finding patterns/UI elements that match a query
    """
    start_time = time.time()

    try:
        if not request.text or not request.text.strip():
            return ComputeTextEmbeddingResponse(
                success=False,
                error="Text cannot be empty",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        embedding = compute_clip_text_embedding(request.text)

        if embedding is None:
            return ComputeTextEmbeddingResponse(
                success=False,
                error="Failed to compute embedding",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        return ComputeTextEmbeddingResponse(
            success=True,
            embedding=embedding,
            embedding_dim=len(embedding),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Text embedding computation failed: {e}", exc_info=True)
        return ComputeTextEmbeddingResponse(
            success=False,
            error=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.post("/compute-batch", response_model=BatchComputeEmbeddingResponse)
async def compute_embeddings_batch(
    request: BatchComputeEmbeddingRequest,
) -> BatchComputeEmbeddingResponse:
    """Compute embeddings for multiple images in a single request.

    This is more efficient than calling /compute multiple times as it
    avoids repeated HTTP overhead and can potentially batch GPU operations.

    Each image in the request should have:
    - id: Unique identifier for the image
    - image_data: Base64 encoded image data
    - text_description: (Optional) Text description for text embedding
    """
    start_time = time.time()
    results: list[BatchEmbeddingResult] = []
    successful = 0
    failed = 0

    for image_info in request.images:
        image_id = image_info.get("id", "unknown")
        image_data = image_info.get("image_data")
        text_description = image_info.get("text_description")

        if not image_data:
            results.append(
                BatchEmbeddingResult(
                    id=image_id,
                    success=False,
                    error="Missing image_data",
                )
            )
            failed += 1
            continue

        try:
            # Decode image
            image = decode_base64_image(image_data)

            # Compute CLIP embedding
            image_embedding = compute_clip_embedding(image)

            # Extract OCR if requested
            ocr_text = None
            ocr_confidence = None
            if request.extract_ocr:
                ocr_text, ocr_confidence = extract_ocr_text(image)

            # Generate text description from name and OCR
            generated_description = generate_text_description(
                name=text_description,  # text_description from request is used as name
                ocr_text=ocr_text,
            )

            # Use generated description for text embedding, fallback to provided or OCR
            text_for_embedding = generated_description or text_description or ocr_text

            # Compute text embedding
            text_embedding = None
            if request.compute_text_embeddings and text_for_embedding:
                text_embedding = compute_text_embedding(text_for_embedding)

            results.append(
                BatchEmbeddingResult(
                    id=image_id,
                    success=True,
                    image_embedding=image_embedding,
                    text_embedding=text_embedding,
                    text_description=generated_description,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence,
                )
            )
            successful += 1

        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}")
            results.append(
                BatchEmbeddingResult(
                    id=image_id,
                    success=False,
                    error=str(e),
                )
            )
            failed += 1

    processing_time_ms = (time.time() - start_time) * 1000

    return BatchComputeEmbeddingResponse(
        success=failed == 0,
        results=results,
        total_processed=len(request.images),
        successful=successful,
        failed=failed,
        processing_time_ms=processing_time_ms,
    )


@router.get("/status")
async def embeddings_status() -> dict[str, Any]:
    """Get the status of embedding models.

    Returns information about which models are loaded and available.
    """
    clip_status = "not_loaded"
    text_status = "not_loaded"
    ocr_status = "not_loaded"

    # Check CLIP model
    if _clip_model is not None:
        clip_status = "fallback" if _clip_model == "fallback" else "loaded"

    # Check text model
    if _text_model is not None:
        text_status = "fallback" if _text_model == "fallback" else "loaded"

    # Check OCR engine
    if _ocr_engine is not None:
        ocr_status = "fallback" if _ocr_engine == "fallback" else "loaded"

    return {
        "clip_model": clip_status,
        "text_model": text_status,
        "ocr_engine": ocr_status,
        "embedding_dimensions": {
            "clip": 512,
            "text": 384,
        },
    }


@router.post("/warmup")
async def warmup_models() -> dict[str, Any]:
    """Warm up embedding models by loading them into memory.

    Call this endpoint before using /compute or /compute-batch to ensure
    models are loaded and ready, avoiding cold-start latency.
    """
    start_time = time.time()

    # Load all models
    clip_model = get_clip_model()
    text_model = get_text_model()
    ocr_engine = get_ocr_engine()

    processing_time_ms = (time.time() - start_time) * 1000

    return {
        "success": True,
        "clip_model": "loaded" if clip_model != "fallback" else "fallback",
        "text_model": "loaded" if text_model != "fallback" else "fallback",
        "ocr_engine": "loaded" if ocr_engine != "fallback" else "fallback",
        "warmup_time_ms": processing_time_ms,
    }
