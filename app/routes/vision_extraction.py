"""
Vision Extraction API Routes

Provides endpoints for running SAM3, Edge Detection, and OCR
on screenshots to extract UI element candidates.
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from PIL import Image as PILImage
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision-extraction", tags=["Vision Extraction"])


# Request/Response Models
class VisionExtractionRequest(BaseModel):
    """Request model for vision extraction."""

    screenshot: str  # base64 encoded image
    techniques: list[str] = ["edge", "sam3", "ocr"]  # Which techniques to run
    # Edge detection config
    canny_low: int = 50
    canny_high: int = 150
    min_contour_area: int = 100
    # SAM3 config
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    # OCR config
    ocr_engine: str = "easyocr"
    ocr_languages: list[str] = ["en"]
    ocr_confidence_threshold: float = 0.5
    # Fusion config
    iou_threshold: float = 0.5


class BoundingBox(BaseModel):
    """Bounding box for detected elements."""

    x: int
    y: int
    width: int
    height: int


class EdgeDetectionResult(BaseModel):
    """Result from edge detection."""

    id: str
    bbox: BoundingBox
    confidence: float
    contour_area: float
    contour_perimeter: float
    vertex_count: int
    aspect_ratio: float
    contour_points: list[tuple[int, int]] | None = None


class SAM3SegmentResult(BaseModel):
    """Result from SAM3 segmentation."""

    id: str
    bbox: BoundingBox
    stability_score: float
    predicted_iou: float
    mask_area: int


class OCRResult(BaseModel):
    """Result from OCR detection."""

    id: str
    bbox: BoundingBox
    text: str
    confidence: float
    language: str


class ExtractedCandidate(BaseModel):
    """Merged candidate from all techniques."""

    id: str
    bbox: BoundingBox
    confidence: float
    category: str | None = None
    text: str | None = None
    detection_technique: str
    is_clickable: bool = False


class VisionExtractionResponse(BaseModel):
    """Response model for vision extraction."""

    screenshot_id: str
    image_width: int
    image_height: int
    # Results from each technique
    edge_results: list[EdgeDetectionResult] = []
    sam3_results: list[SAM3SegmentResult] = []
    ocr_results: list[OCRResult] = []
    # Merged candidates
    merged_candidates: list[ExtractedCandidate] = []
    # Debug images (base64 encoded)
    edge_overlay: str | None = None
    sam3_overlay: str | None = None
    ocr_overlay: str | None = None
    # Processing info
    techniques_run: list[str] = []
    processing_time_ms: float = 0


def base64_to_pil(base64_string: str) -> PILImage.Image:
    """Convert base64 string to PIL Image."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    return PILImage.open(io.BytesIO(image_bytes))


def pil_to_base64(img: PILImage.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


@router.post("/extract", response_model=VisionExtractionResponse)
async def extract_vision(request: Request, extraction_request: VisionExtractionRequest):
    """
    Run vision extraction on a screenshot.

    Runs selected detection techniques (edge, sam3, ocr) and returns
    detected UI element candidates with bounding boxes.
    """
    import time

    import cv2
    import numpy as np

    start_time = time.time()
    screenshot_id = str(uuid.uuid4())

    try:
        # Decode screenshot
        pil_image = base64_to_pil(extraction_request.screenshot)
        img_array = np.array(pil_image)
        image_height, image_width = img_array.shape[:2]

        # Convert to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA -> BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3:
            # RGB -> BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Initialize results
        edge_results: list[EdgeDetectionResult] = []
        sam3_results: list[SAM3SegmentResult] = []
        ocr_results: list[OCRResult] = []
        edge_overlay: str | None = None
        sam3_overlay: str | None = None
        ocr_overlay: str | None = None
        techniques_run: list[str] = []

        # Run Edge Detection
        if "edge" in extraction_request.techniques:
            try:
                edge_results, edge_overlay = run_edge_detection(
                    img_bgr,
                    canny_low=extraction_request.canny_low,
                    canny_high=extraction_request.canny_high,
                    min_contour_area=extraction_request.min_contour_area,
                )
                techniques_run.append("edge")
                logger.info(f"Edge detection found {len(edge_results)} contours")
            except Exception as e:
                logger.error(f"Edge detection failed: {e}")

        # Run SAM3 Segmentation
        if "sam3" in extraction_request.techniques:
            try:
                sam3_results, sam3_overlay = run_sam3_segmentation(
                    img_array,
                    points_per_side=extraction_request.points_per_side,
                    pred_iou_thresh=extraction_request.pred_iou_thresh,
                    stability_score_thresh=extraction_request.stability_score_thresh,
                )
                techniques_run.append("sam3")
                logger.info(f"SAM3 found {len(sam3_results)} segments")
            except Exception as e:
                logger.error(f"SAM3 segmentation failed: {e}")

        # Run OCR
        if "ocr" in extraction_request.techniques:
            try:
                ocr_results, ocr_overlay = run_ocr_detection(
                    img_array,
                    engine=extraction_request.ocr_engine,
                    languages=extraction_request.ocr_languages,
                    confidence_threshold=extraction_request.ocr_confidence_threshold,
                )
                techniques_run.append("ocr")
                logger.info(f"OCR found {len(ocr_results)} text regions")
            except Exception as e:
                logger.error(f"OCR detection failed: {e}")

        # Merge results
        merged_candidates = merge_results(
            edge_results,
            sam3_results,
            ocr_results,
            iou_threshold=extraction_request.iou_threshold,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        return VisionExtractionResponse(
            screenshot_id=screenshot_id,
            image_width=image_width,
            image_height=image_height,
            edge_results=edge_results,
            sam3_results=sam3_results,
            ocr_results=ocr_results,
            merged_candidates=merged_candidates,
            edge_overlay=edge_overlay,
            sam3_overlay=sam3_overlay,
            ocr_overlay=ocr_overlay,
            techniques_run=techniques_run,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_edge_detection(
    img_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_contour_area: int = 100,
) -> tuple[list[EdgeDetectionResult], str | None]:
    """Run Canny edge detection and contour analysis."""
    import cv2

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[EdgeDetectionResult] = []

    # Create overlay image
    overlay = img_bgr.copy()

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate properties
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertex_count = len(approx)
        aspect_ratio = w / max(h, 1)

        # Confidence based on properties
        confidence = min(1.0, area / 10000)  # Scale by area

        # Get contour points
        contour_points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]

        results.append(
            EdgeDetectionResult(
                id=f"edge_{i}",
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                confidence=confidence,
                contour_area=area,
                contour_perimeter=perimeter,
                vertex_count=vertex_count,
                aspect_ratio=aspect_ratio,
                contour_points=contour_points[:50] if len(contour_points) > 50 else contour_points,
            )
        )

        # Draw on overlay
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Convert overlay to base64
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_pil = PILImage.fromarray(overlay_rgb)
    overlay_b64 = pil_to_base64(overlay_pil)

    return results, overlay_b64


def run_sam3_segmentation(
    img_array: np.ndarray,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> tuple[list[SAM3SegmentResult], str | None]:
    """Run SAM3 segmentation."""
    import numpy as np

    results: list[SAM3SegmentResult] = []

    try:
        # Try to import SAM
        # Check for model file
        import os

        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model_paths = [
            os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth"),
            os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth"),
            os.path.expanduser("~/.cache/sam/sam_vit_l_0b3195.pth"),
        ]

        model_path = None
        model_type = None
        for path, mtype in zip(model_paths, ["vit_h", "vit_b", "vit_l"], strict=False):
            if os.path.exists(path):
                model_path = path
                model_type = mtype
                break

        if model_path is None:
            logger.warning("SAM model not found, using fallback segmentation")
            return run_fallback_segmentation(img_array)

        # Load SAM model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)

        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=100,
        )

        # Generate masks
        masks = mask_generator.generate(img_array)

        # Create overlay
        overlay = img_array.copy()
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 128),
            (255, 128, 0),
        ]

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]  # x, y, w, h

            results.append(
                SAM3SegmentResult(
                    id=f"sam3_{i}",
                    bbox=BoundingBox(
                        x=int(bbox[0]),
                        y=int(bbox[1]),
                        width=int(bbox[2]),
                        height=int(bbox[3]),
                    ),
                    stability_score=float(mask_data["stability_score"]),
                    predicted_iou=float(mask_data["predicted_iou"]),
                    mask_area=int(mask_data["area"]),
                )
            )

            # Draw colored mask
            color = colors[i % len(colors)]
            overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

        overlay_pil = PILImage.fromarray(overlay.astype(np.uint8))
        overlay_b64 = pil_to_base64(overlay_pil)

        return results, overlay_b64

    except ImportError:
        logger.warning("SAM not available, using fallback segmentation")
        return run_fallback_segmentation(img_array)
    except Exception as e:
        logger.error(f"SAM3 error: {e}")
        return run_fallback_segmentation(img_array)


def run_fallback_segmentation(
    img_array: np.ndarray,
) -> tuple[list[SAM3SegmentResult], str | None]:
    """Fallback segmentation using connected components when SAM is unavailable."""
    import cv2
    import numpy as np

    results: list[SAM3SegmentResult] = []

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create overlay
    overlay = img_array.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 128, 0),
    ]

    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h, area = stats[i]

        if area < 100:
            continue

        results.append(
            SAM3SegmentResult(
                id=f"sam3_{i}",
                bbox=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                stability_score=0.8,  # Fixed score for fallback
                predicted_iou=0.75,
                mask_area=int(area),
            )
        )

        # Draw on overlay
        mask = labels == i
        color = colors[i % len(colors)]
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

    overlay_pil = PILImage.fromarray(overlay.astype(np.uint8))
    overlay_b64 = pil_to_base64(overlay_pil)

    return results, overlay_b64


def run_ocr_detection(
    img_array: np.ndarray,
    engine: str = "easyocr",
    languages: list[str] | None = None,
    confidence_threshold: float = 0.5,
) -> tuple[list[OCRResult], str | None]:
    """Run OCR detection."""
    import cv2
    import numpy as np

    if languages is None:
        languages = ["en"]

    results: list[OCRResult] = []
    overlay = img_array.copy()

    try:
        if engine == "easyocr":
            import easyocr

            reader = easyocr.Reader(languages, gpu=False)
            detections = reader.readtext(img_array)

            for i, (bbox_pts, text, confidence) in enumerate(detections):
                if confidence < confidence_threshold:
                    continue

                # Convert bbox points to x, y, w, h
                pts = np.array(bbox_pts)
                x = int(min(pts[:, 0]))
                y = int(min(pts[:, 1]))
                w = int(max(pts[:, 0]) - x)
                h = int(max(pts[:, 1]) - y)

                results.append(
                    OCRResult(
                        id=f"ocr_{i}",
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        text=text,
                        confidence=float(confidence),
                        language=languages[0] if languages else "en",
                    )
                )

                # Draw on overlay
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (128, 0, 255), 2)
                cv2.putText(
                    overlay, text[:20], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1
                )

        else:
            # Fallback to pytesseract
            import pytesseract

            data = pytesseract.image_to_data(img_array, output_type=pytesseract.Output.DICT)

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = float(data["conf"][i]) / 100.0

                if not text or conf < confidence_threshold:
                    continue

                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

                results.append(
                    OCRResult(
                        id=f"ocr_{i}",
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        text=text,
                        confidence=conf,
                        language="en",
                    )
                )

                cv2.rectangle(overlay, (x, y), (x + w, y + h), (128, 0, 255), 2)

    except ImportError as e:
        logger.warning(f"OCR engine not available: {e}")
    except Exception as e:
        logger.error(f"OCR error: {e}")

    overlay_pil = PILImage.fromarray(overlay)
    overlay_b64 = pil_to_base64(overlay_pil)

    return results, overlay_b64


def merge_results(
    edge_results: list[EdgeDetectionResult],
    sam3_results: list[SAM3SegmentResult],
    ocr_results: list[OCRResult],
    iou_threshold: float = 0.5,
) -> list[ExtractedCandidate]:
    """Merge results from all techniques using IoU-based deduplication."""

    candidates: list[ExtractedCandidate] = []

    # Convert edge results
    for r in edge_results:
        category = classify_edge_result(r)
        candidates.append(
            ExtractedCandidate(
                id=r.id,
                bbox=r.bbox,
                confidence=r.confidence,
                category=category,
                detection_technique="edge",
                is_clickable=category in ["button", "input", "link"],
            )
        )

    # Convert SAM3 results
    for r in sam3_results:
        category = classify_sam3_result(r)
        candidates.append(
            ExtractedCandidate(
                id=r.id,
                bbox=r.bbox,
                confidence=r.stability_score,
                category=category,
                detection_technique="sam3",
                is_clickable=category in ["button", "icon"],
            )
        )

    # Convert OCR results
    for r in ocr_results:
        category = classify_ocr_result(r)
        candidates.append(
            ExtractedCandidate(
                id=r.id,
                bbox=r.bbox,
                confidence=r.confidence,
                category=category,
                text=r.text,
                detection_technique="ocr",
                is_clickable=category in ["button", "link"],
            )
        )

    # Deduplicate based on IoU
    merged: list[ExtractedCandidate] = []
    used_indices: set[int] = set()

    for i, cand in enumerate(candidates):
        if i in used_indices:
            continue

        # Find overlapping candidates
        overlapping = [i]
        for j in range(i + 1, len(candidates)):
            if j in used_indices:
                continue
            if calculate_iou(cand.bbox, candidates[j].bbox) >= iou_threshold:
                overlapping.append(j)
                used_indices.add(j)

        # Merge overlapping candidates
        if len(overlapping) == 1:
            merged.append(cand)
        else:
            # Combine techniques
            techniques = set()
            best_confidence = 0.0
            best_text = None
            best_category = cand.category

            for idx in overlapping:
                c = candidates[idx]
                techniques.add(c.detection_technique)
                if c.confidence > best_confidence:
                    best_confidence = c.confidence
                    best_category = c.category
                if c.text:
                    best_text = c.text

            merged.append(
                ExtractedCandidate(
                    id=cand.id,
                    bbox=cand.bbox,
                    confidence=best_confidence,
                    category=best_category,
                    text=best_text,
                    detection_technique="+".join(sorted(techniques)),
                    is_clickable=cand.is_clickable,
                )
            )

        used_indices.add(i)

    return merged


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(bbox1.x, bbox2.x)
    y1 = max(bbox1.y, bbox2.y)
    x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
    y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox1.width * bbox1.height
    area2 = bbox2.width * bbox2.height
    union = area1 + area2 - intersection

    return intersection / max(union, 1)


def classify_edge_result(result: EdgeDetectionResult) -> str:
    """Classify edge detection result into element category."""
    bbox = result.bbox
    aspect_ratio = result.aspect_ratio
    vertex_count = result.vertex_count
    area = result.contour_area

    if vertex_count == 4:
        if 1.5 < aspect_ratio < 6.0 and 30 < bbox.width < 300 and 20 < bbox.height < 60:
            return "button"
        if aspect_ratio > 4.0 and bbox.height < 50:
            return "input"
        if bbox.width > 200 and bbox.height > 100:
            return "container"

    if 0.8 < aspect_ratio < 1.25 and bbox.width < 60 and bbox.height < 60:
        return "icon"

    if area > 10000:
        return "container"

    return "element"


def classify_sam3_result(result: SAM3SegmentResult) -> str:
    """Classify SAM3 segment into element category."""
    bbox = result.bbox
    aspect_ratio = bbox.width / max(bbox.height, 1)
    area = result.mask_area

    if area < 2500 and 0.7 < aspect_ratio < 1.4:
        return "icon"

    if 1.5 < aspect_ratio < 6.0 and 500 < area < 15000:
        return "button"

    if aspect_ratio > 4.0 and bbox.height < 50:
        return "input"

    if area > 50000:
        return "container"

    return "element"


def classify_ocr_result(result: OCRResult) -> str:
    """Classify OCR result into element category."""
    text = result.text.lower().strip()
    bbox = result.bbox

    button_keywords = [
        "submit",
        "cancel",
        "ok",
        "yes",
        "no",
        "save",
        "delete",
        "add",
        "remove",
        "edit",
        "update",
        "create",
        "close",
        "next",
        "back",
        "previous",
        "continue",
        "done",
        "finish",
        "login",
        "logout",
        "sign in",
        "sign out",
        "sign up",
        "search",
        "filter",
        "sort",
        "reset",
        "clear",
    ]

    if any(kw in text for kw in button_keywords):
        return "button"

    if text.startswith("http") or "click" in text or "learn more" in text:
        return "link"

    if len(text) < 15:
        aspect_ratio = bbox.width / max(bbox.height, 1)
        if 1.5 < aspect_ratio < 6:
            return "button"

    if len(text) > 50:
        return "paragraph"

    return "label"
