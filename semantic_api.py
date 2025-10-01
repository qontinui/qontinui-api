"""
Semantic Analysis API for screenshot processing.
This provides semantic segmentation and description of UI elements.
"""

import base64
import io
import logging
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

# Import real computer vision capabilities from qontinui
try:
    # Lazy import to avoid NumPy version conflicts
    from qontinui.perception.segmentation import ScreenSegmenter
    from qontinui.semantic.description.clip_generator import CLIPDescriptionGenerator

    QONTINUI_AVAILABLE = True
    EasyOCREngine = None  # Will be imported later if needed
except ImportError:
    QONTINUI_AVAILABLE = False
    logging.warning("Qontinui library not available - using fallback implementation")
    EasyOCREngine = None

# Try to import transformers for CLIP fallback
try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("Transformers/CLIP not available for semantic descriptions")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class ProcessingOptions(BaseModel):
    enable_ocr: bool = True
    min_confidence: float = 0.7
    description_model: str = "clip"  # clip or basic
    focus_regions: list[dict[str, int]] | None = None


class SemanticProcessRequest(BaseModel):
    image: str  # base64 encoded image
    strategy: str = "hybrid"  # sam2, ocr, or hybrid
    options: ProcessingOptions = ProcessingOptions()


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class SemanticObject(BaseModel):
    id: str
    description: str
    ocr_text: str | None
    type: str
    confidence: float
    bounding_box: BoundingBox
    pixel_mask: str | None = None  # base64 encoded mask
    attributes: dict[str, Any]


class SemanticScene(BaseModel):
    timestamp: str
    object_count: int
    objects: list[SemanticObject]


class SemanticProcessResponse(BaseModel):
    scene: SemanticScene
    processing_time_ms: float


class SemanticQueryRequest(BaseModel):
    scene_id: str | None = None
    filters: dict[str, Any]


class SemanticCompareRequest(BaseModel):
    image1: str
    image2: str
    comparison_type: str = "similarity"  # similarity or differences


class ComparisonDifference(BaseModel):
    added: list[SemanticObject]
    removed: list[SemanticObject]
    changed: list[Any]


class SemanticCompareResponse(BaseModel):
    similarity_score: float
    differences: ComparisonDifference


# Real semantic processor using computer vision
class RealSemanticProcessor:
    """Real semantic processor using OpenCV and OCR."""

    def __init__(self):
        """Initialize the processor with available CV models."""
        self.ocr_engine = None
        self.segmenter = None
        self.clip_generator = None
        self.clip_model = None
        self.clip_processor = None

        # Try to initialize real CV models if available
        if QONTINUI_AVAILABLE:
            try:
                # Lazy load EasyOCREngine to avoid NumPy conflicts
                try:
                    from qontinui.hal.implementations.easyocr_engine import EasyOCREngine

                    self.ocr_engine = EasyOCREngine()
                except ImportError:
                    self.ocr_engine = None
                    logger.warning("EasyOCR not available due to NumPy version conflict")

                self.segmenter = ScreenSegmenter()
                self.clip_generator = CLIPDescriptionGenerator()
                logger.info("Initialized with qontinui CV models including CLIP")
            except Exception as e:
                logger.warning(f"Failed to initialize qontinui models: {e}")

        # Try to initialize CLIP directly if not available through qontinui
        if not self.clip_generator and CLIP_AVAILABLE:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.clip_model = self.clip_model.to(self.device)
                logger.info("Initialized CLIP model for semantic descriptions")
            except Exception as e:
                logger.warning(f"Failed to initialize CLIP: {e}")

        # Fallback to OpenCV-based processing
        if not self.ocr_engine:
            logger.info("Using OpenCV-based fallback implementation")

    @staticmethod
    def decode_image(base64_str: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            # Remove data URL prefix if present
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            # Add padding if necessary
            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += "=" * (4 - missing_padding)

            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}") from e

    @staticmethod
    def encode_mask(mask: np.ndarray) -> str:
        """Encode mask as base64."""
        _, buffer = cv2.imencode(".png", mask)
        return base64.b64encode(buffer).decode("utf-8")

    def detect_ui_elements(
        self, image: np.ndarray, options: ProcessingOptions, strategy: str = "hybrid"
    ) -> list[SemanticObject]:
        """Detect UI elements using real computer vision."""
        height, width = image.shape[:2]
        objects = []

        # Choose strategy based on request
        if strategy == "sam2":
            # Use SAM2-style segmentation with masks
            segments = self._segment_with_masks(image, options)
            objects.extend(segments)
        elif strategy == "ocr":
            # Focus on text extraction
            ocr_objects = self._extract_text(image, options)
            objects.extend(ocr_objects)
        else:  # hybrid or default
            # Use real segmentation if available
            if self.segmenter and hasattr(self.segmenter, "segment_screen"):
                try:
                    segments = self._segment_with_qontinui(image)
                    objects.extend(segments)
                except Exception as e:
                    logger.warning(f"Qontinui segmentation failed: {e}, using OpenCV")
                    segments = self._segment_with_opencv(image, options)
                    objects.extend(segments)
            else:
                # Use OpenCV-based segmentation
                segments = self._segment_with_opencv(image, options)
                objects.extend(segments)

            # Apply OCR if enabled in hybrid mode
            if options.enable_ocr:
                ocr_objects = self._extract_text(image, options)
                objects.extend(ocr_objects)

        # Filter by confidence
        objects = [obj for obj in objects if obj.confidence >= options.min_confidence]

        # Remove duplicates based on overlapping bounding boxes
        objects = self._remove_duplicates(objects)

        return objects

    def _segment_with_opencv(
        self, image: np.ndarray, options: ProcessingOptions
    ) -> list[SemanticObject]:
        """Segment image using OpenCV techniques."""
        objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Edge detection for UI elements
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process significant contours
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Skip elements that are too small or too large
            if w < 20 or h < 20 or w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:
                continue

            # Extract region properties
            region = image[y : y + h, x : x + w]
            avg_color = (
                region.mean(axis=(0, 1))[:3].astype(int).tolist()
                if region.size > 0
                else [128, 128, 128]
            )

            # Classify element type based on properties
            element_type = self._classify_element(w, h, region)

            # Generate semantic description
            position = self._get_position_description(x, y, image.shape[1], image.shape[0])
            if options.description_model == "clip":
                clip_description = self._generate_clip_description(region, element_type)
                description = f"{clip_description} {position}"
            else:
                # OpenCV-based content analysis
                content_hint = self._analyze_content_opencv(region, element_type)
                description = f"{content_hint} {position}"

            obj = SemanticObject(
                id=f"opencv_{idx}_{x}_{y}",
                description=description,
                ocr_text=None,
                type=element_type,
                confidence=0.75,  # OpenCV detection confidence
                bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                pixel_mask=None,
                attributes={
                    "color": avg_color,
                    "interactable": element_type in ["button", "input", "link"],
                    "area": int(area),
                },
            )
            objects.append(obj)

        return objects

    def _segment_with_qontinui(self, image: np.ndarray) -> list[SemanticObject]:
        """Segment using qontinui's screen segmenter."""
        objects = []

        # Convert to PIL Image for qontinui
        pil_image = Image.fromarray(image)

        # Get segments from qontinui
        segments = self.segmenter.segment_screen(pil_image)

        for idx, segment in enumerate(segments):
            # Convert segment to SemanticObject
            bbox = segment.get("bbox", {"x": 0, "y": 0, "width": 50, "height": 50})

            obj = SemanticObject(
                id=f"qontinui_{idx}",
                description=segment.get("description", "UI element"),
                ocr_text=segment.get("text"),
                type=segment.get("type", "unknown"),
                confidence=segment.get("confidence", 0.8),
                bounding_box=BoundingBox(
                    x=bbox["x"], y=bbox["y"], width=bbox["width"], height=bbox["height"]
                ),
                pixel_mask=segment.get("mask"),
                attributes=segment.get("attributes", {}),
            )
            objects.append(obj)

        return objects

    def _extract_text(self, image: np.ndarray, options: ProcessingOptions) -> list[SemanticObject]:
        """Extract text regions using OCR."""
        objects = []

        if self.ocr_engine:
            # Use EasyOCR
            try:
                results = self.ocr_engine.readtext(image)
                for idx, (bbox, text, confidence) in enumerate(results):
                    # Convert bbox points to x, y, width, height
                    points = np.array(bbox)
                    x = int(points[:, 0].min())
                    y = int(points[:, 1].min())
                    w = int(points[:, 0].max() - x)
                    h = int(points[:, 1].max() - y)

                    obj = SemanticObject(
                        id=f"text_{idx}_{x}_{y}",
                        description=(
                            f"text region containing '{text[:30]}...'"
                            if len(text) > 30
                            else f"text region containing '{text}'"
                        ),
                        ocr_text=text,
                        type="text",
                        confidence=float(confidence),
                        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                        pixel_mask=None,
                        attributes={"text_confidence": float(confidence)},
                    )
                    objects.append(obj)
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        else:
            # Fallback to simple text detection with OpenCV
            objects.extend(self._detect_text_regions_opencv(image, options))

        return objects

    def _detect_text_regions_opencv(
        self, image: np.ndarray, options: ProcessingOptions
    ) -> list[SemanticObject]:
        """Detect text regions using OpenCV (fallback method)."""
        objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on aspect ratio (text is usually horizontal)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 1.5 or aspect_ratio > 15 or w < 30 or h < 10:
                continue

            obj = SemanticObject(
                id=f"text_region_{idx}_{x}_{y}",
                description="potential text region",
                ocr_text=None,
                type="text",
                confidence=0.6,  # Lower confidence for OpenCV detection
                bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                pixel_mask=None,
                attributes={"detection_method": "opencv_morphology"},
            )
            objects.append(obj)

        return objects

    def _generate_clip_description(self, image_region: np.ndarray, element_type: str) -> str:
        """Generate semantic description using CLIP."""
        if self.clip_generator:
            # Use qontinui's CLIP generator
            try:
                pil_image = Image.fromarray(image_region)
                description = self.clip_generator.generate(pil_image)
                return description
            except Exception as e:
                logger.warning(f"CLIP generation failed: {e}")

        elif self.clip_model and self.clip_processor:
            # Use direct CLIP model with content-focused candidates
            try:
                # Define content-focused candidates instead of just UI elements
                content_candidates = [
                    # Content descriptions
                    "text content",
                    "an image or photo",
                    "a logo or brand",
                    "a graph or chart",
                    "a table or grid",
                    "a video player",
                    "a map view",
                    "an advertisement",
                    "a product image",
                    "a user avatar or profile picture",
                    "a thumbnail image",
                    "an illustration or diagram",
                    "code or terminal output",
                    "a document or article",
                    "a calendar view",
                    "a dashboard widget",
                    "social media content",
                    "a news article",
                    "a search result",
                    "a notification or alert",
                    # Functional descriptions
                    "navigation controls",
                    "login or authentication form",
                    "search interface",
                    "settings or preferences",
                    "a shopping cart",
                    "payment information",
                    "user comments or reviews",
                    "a file browser",
                    "media controls",
                    "a toolbar with actions",
                    # Generic UI (as fallback)
                    f"a {element_type}",
                    "user interface element",
                    "interactive control",
                    "clickable element",
                ]

                # Process image and text
                pil_image = Image.fromarray(image_region)
                inputs = self.clip_processor(
                    text=content_candidates, images=pil_image, return_tensors="pt", padding=True
                ).to(self.device)

                # Get predictions
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Get top 3 matches for better context
                top_k = 3
                top_probs, top_indices = probs[0].topk(top_k)

                # Use the best match, but if it's too generic, combine with element type
                best_idx = top_indices[0].item()
                best_description = content_candidates[best_idx]
                confidence = top_probs[0].item()

                # If confidence is low or description is generic, add element type
                if confidence < 0.3 or "element" in best_description.lower():
                    best_description = f"{best_description} ({element_type})"

                return best_description
            except Exception as e:
                logger.warning(f"CLIP inference failed: {e}")

        # Fallback to simple description
        return element_type

    def _classify_element(self, width: int, height: int, region: np.ndarray) -> str:
        """Classify UI element type based on properties."""
        aspect_ratio = width / height if height > 0 else 1

        # Button detection (rectangular, moderate size)
        if 0.5 < aspect_ratio < 5 and 20 < height < 100 and 40 < width < 400:
            # Check for uniform color (common in buttons)
            if region.size > 0:
                std_dev = np.std(region)
                if std_dev < 50:
                    return "button"
            else:
                return "button"

        # Input field detection (wide, thin)
        if aspect_ratio > 3 and height < 60 and width > 100:
            return "input"

        # Image detection (square-ish, larger)
        if 0.5 < aspect_ratio < 2 and width > 50 and height > 50:
            # Check for high color variance (images have more variation)
            std_dev = np.std(region) if region.size > 0 else 0
            if std_dev > 30:
                return "image"

        # Text/label detection (horizontal)
        if aspect_ratio > 2 and height < 50:
            return "label"

        # Container/panel detection (large)
        if width > 200 and height > 100:
            return "container"

        return "element"

    def _analyze_content_opencv(self, region: np.ndarray, element_type: str) -> str:
        """Analyze content using OpenCV to provide meaningful descriptions."""
        if region.size == 0:
            return element_type

        # Calculate color statistics
        mean_color = region.mean(axis=(0, 1))[:3] if len(region.shape) == 3 else [region.mean()]
        std_dev = np.std(region)

        # Analyze dominant colors for content hints
        is_blue = mean_color[2] > mean_color[1] and mean_color[2] > mean_color[0]  # BGR format
        is_green = mean_color[1] > mean_color[2] and mean_color[1] > mean_color[0]
        is_red = mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]
        is_grayscale = np.std(mean_color) < 10

        # Check for specific patterns
        height, width = region.shape[:2]
        aspect_ratio = width / height if height > 0 else 1

        # Content-based descriptions based on visual properties
        if std_dev > 60:  # High variance suggests image content
            if aspect_ratio > 2:
                return "banner or header image"
            elif 0.8 < aspect_ratio < 1.2:
                return "square image or icon"
            else:
                return "image content"

        elif std_dev < 20:  # Low variance suggests solid color or simple element
            if is_blue:
                if element_type == "button":
                    return "primary action button"
                return "highlighted content area"
            elif is_green:
                return "success or positive action element"
            elif is_red:
                return "alert or important action"
            elif is_grayscale:
                if element_type == "input":
                    return "text input field"
                elif element_type == "container":
                    return "content panel or card"
                return "interface element"

        # Check for text-like patterns (horizontal lines, consistent spacing)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        horizontal_projection = np.sum(edges, axis=1)

        # Count peaks in horizontal projection (text lines)
        peaks = np.where(horizontal_projection > width * 0.1)[0]
        if len(peaks) > 3:
            return "text content or article"
        elif len(peaks) > 1:
            return "heading or label text"

        # Size-based content hints
        area = width * height
        total_area = region.shape[0] * region.shape[1] if len(region.shape) >= 2 else 0

        if area > total_area * 0.3:
            return "main content area"
        elif area < total_area * 0.05:
            return "small interface element"

        # Fallback to element type with context
        if element_type == "button":
            return "interactive button"
        elif element_type == "input":
            return "data entry field"
        elif element_type == "image":
            return "visual content"
        elif element_type == "container":
            return "content container"

        return f"{element_type} element"

    def _get_position_description(self, x: int, y: int, width: int, height: int) -> str:
        """Get position description for an element."""
        horizontal = "left" if x < width / 3 else ("right" if x > 2 * width / 3 else "center")
        vertical = "top" if y < height / 3 else ("bottom" if y > 2 * height / 3 else "middle")
        return f"in {vertical} {horizontal}"

    def _remove_duplicates(self, objects: list[SemanticObject]) -> list[SemanticObject]:
        """Remove duplicate objects based on overlapping bounding boxes."""
        if not objects:
            return objects

        # Sort by confidence
        objects.sort(key=lambda x: x.confidence, reverse=True)

        filtered = []
        for obj in objects:
            # Check if this object significantly overlaps with any already selected
            is_duplicate = False
            for selected in filtered:
                if self._calculate_iou(obj.bounding_box, selected.bounding_box) > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(obj)

        return filtered

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _segment_with_masks(
        self, image: np.ndarray, options: ProcessingOptions
    ) -> list[SemanticObject]:
        """Segment image and generate masks (SAM2-style)."""
        objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Use adaptive thresholding for better segmentation
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Skip elements that are too small or too large
            if w < 20 or h < 20 or w > image.shape[1] * 0.95 or h > image.shape[0] * 0.95:
                continue

            # Create mask for this contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Crop mask to bounding box
            mask_crop = mask[y : y + h, x : x + w]

            # Encode mask as base64
            encoded_mask = self.encode_mask(mask_crop)

            # Extract region properties
            region = image[y : y + h, x : x + w]
            avg_color = (
                region.mean(axis=(0, 1))[:3].astype(int).tolist()
                if region.size > 0
                else [128, 128, 128]
            )

            # Classify element type
            element_type = self._classify_element(w, h, region)

            # Generate semantic description based on content
            position = self._get_position_description(x, y, image.shape[1], image.shape[0])

            # Always try to get content-aware description
            if CLIP_AVAILABLE and self.clip_model:
                content_description = self._generate_clip_description(region, element_type)
            else:
                content_description = self._analyze_content_opencv(region, element_type)

            description = f"{content_description} {position}"

            obj = SemanticObject(
                id=f"sam2_{idx}_{x}_{y}",
                description=description,
                ocr_text=None,
                type=element_type,
                confidence=0.85,  # Higher confidence for SAM2-style
                bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                pixel_mask=encoded_mask,  # Include the actual mask
                attributes={
                    "color": avg_color,
                    "interactable": element_type in ["button", "input", "link"],
                    "area": int(area),
                    "has_mask": True,
                },
            )
            objects.append(obj)

        return objects


# Initialize the processor
processor = RealSemanticProcessor()


@router.post("/semantic/process", response_model=SemanticProcessResponse)
async def process_screenshot(request: SemanticProcessRequest):
    """Process a screenshot for semantic analysis."""
    try:
        start_time = datetime.now()

        # Decode image
        image = processor.decode_image(request.image)

        # Apply focus regions if provided
        if request.options.focus_regions:
            # In real implementation, this would limit detection to specific regions
            pass

        # Detect UI elements with strategy
        objects = processor.detect_ui_elements(image, request.options, request.strategy)

        # Filter by confidence
        objects = [obj for obj in objects if obj.confidence >= request.options.min_confidence]

        # Create scene
        scene = SemanticScene(
            timestamp=datetime.now().isoformat(), object_count=len(objects), objects=objects
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return SemanticProcessResponse(scene=scene, processing_time_ms=processing_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/semantic/query")
async def query_scene(request: SemanticQueryRequest):
    """Query/filter objects in a scene."""
    # This would query cached scenes in a real implementation
    return {
        "message": "Query endpoint - would filter cached scene objects",
        "filters": request.filters,
    }


@router.post("/semantic/compare", response_model=SemanticCompareResponse)
async def compare_screenshots(request: SemanticCompareRequest):
    """Compare two screenshots semantically."""
    try:
        # Process both images
        image1 = processor.decode_image(request.image1)
        image2 = processor.decode_image(request.image2)

        options = ProcessingOptions()
        objects1 = processor.detect_ui_elements(image1, options)
        objects2 = processor.detect_ui_elements(image2, options)

        # Simple similarity calculation based on object types
        types1 = {obj.type for obj in objects1}
        types2 = {obj.type for obj in objects2}

        if len(types1.union(types2)) > 0:
            similarity = len(types1.intersection(types2)) / len(types1.union(types2))
        else:
            similarity = 1.0 if len(types1) == len(types2) == 0 else 0.0

        # Detect differences
        added = [
            obj
            for obj in objects2
            if not any(
                o.type == obj.type and abs(o.bounding_box.x - obj.bounding_box.x) < 50
                for o in objects1
            )
        ]

        removed = [
            obj
            for obj in objects1
            if not any(
                o.type == obj.type and abs(o.bounding_box.x - obj.bounding_box.x) < 50
                for o in objects2
            )
        ]

        changed = []
        for obj1 in objects1:
            for obj2 in objects2:
                if (
                    obj1.type == obj2.type
                    and abs(obj1.bounding_box.x - obj2.bounding_box.x) < 50
                    and obj1.ocr_text != obj2.ocr_text
                ):
                    changed.append(
                        {
                            "id": obj1.id,
                            "old_text": obj1.ocr_text,
                            "new_text": obj2.ocr_text,
                            "type": obj1.type,
                        }
                    )

        return SemanticCompareResponse(
            similarity_score=similarity,
            differences=ComparisonDifference(added=added, removed=removed, changed=changed),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Health check
@router.get("/semantic/health")
async def health_check():
    """Check if semantic API is running."""
    return {"status": "healthy", "service": "semantic-api"}
