#!/usr/bin/env python3
"""
Inference testing script for object detection models.

This script handles:
- Loading exported models (ONNX, TensorRT, PyTorch)
- Running inference on test images/videos
- Benchmarking inference speed
- Visualizing detections
- Batch processing
- Exporting results
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Base class for inference engines."""

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: list[str] | None = None,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            class_names: List of class names
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Initialized inference engine with model: {self.model_path}")

    def preprocess(self, image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR format)
            target_size: Target size for model input

        Returns:
            Preprocessed image
        """
        # Resize maintaining aspect ratio
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # CHW format
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def postprocess(
        self,
        outputs: np.ndarray,
        original_shape: tuple[int, int],
        target_size: int = 640,
    ) -> list[dict]:
        """
        Postprocess model outputs.

        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (h, w)
            target_size: Model input size

        Returns:
            List of detections
        """
        raise NotImplementedError("Subclasses must implement postprocess")

    def predict(self, image: np.ndarray | Path) -> list[dict]:
        """
        Run inference on image.

        Args:
            image: Input image (numpy array or path)

        Returns:
            List of detections
        """
        raise NotImplementedError("Subclasses must implement predict")

    def visualize(
        self,
        image: np.ndarray,
        detections: list[dict],
        output_path: Path | None = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Visualize detections on image.

        Args:
            image: Input image
            detections: List of detections
            output_path: Optional path to save visualization
            show: Whether to display image

        Returns:
            Image with visualizations
        """
        viz_image = image.copy()

        # Generate colors for each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names) + 1, 3), dtype=np.uint8)

        for det in detections:
            bbox = det["bbox"]
            cls = det["class"]
            conf = det["confidence"]

            x1, y1, x2, y2 = map(int, bbox)

            # Get color for class
            color = tuple(map(int, colors[cls]))

            # Draw bounding box
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
            label = f"{class_name} {conf:.2f}"

            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw label background
            cv2.rectangle(
                viz_image,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                viz_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        if output_path:
            cv2.imwrite(str(output_path), viz_image)
            logger.info(f"Visualization saved to {output_path}")

        if show:
            cv2.imshow("Detections", viz_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return viz_image


class ONNXInferenceEngine(InferenceEngine):
    """Inference engine for ONNX models."""

    def __init__(self, *args, **kwargs):
        """Initialize ONNX inference engine."""
        super().__init__(*args, **kwargs)

        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime package not installed. " "Install with: pip install onnxruntime"
            ) from err

        logger.info("Loading ONNX model")

        # Create session with GPU support if available
        providers = ["CPUExecutionProvider"]
        if ort.get_device() == "GPU":
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info("ONNX model loaded successfully")
        logger.info(f"  Input: {self.input_name}, shape: {self.input_shape}")
        logger.info(f"  Outputs: {self.output_names}")
        logger.info(f"  Providers: {self.session.get_providers()}")

    def predict(self, image: np.ndarray | Path) -> list[dict]:
        """
        Run inference on image.

        Args:
            image: Input image (numpy array or path)

        Returns:
            List of detections
        """
        # Load image if path provided
        if isinstance(image, str | Path):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Failed to load image: {image}")

        original_shape_tuple = image.shape[:2]
        original_shape: tuple[int, int] = (
            int(original_shape_tuple[0]),
            int(original_shape_tuple[1]),
        )

        # Preprocess
        input_data = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Postprocess (assuming YOLO format output)
        detections = self.postprocess_yolo(outputs[0], original_shape)

        return detections

    def postprocess_yolo(self, outputs: np.ndarray, original_shape: tuple[int, int]) -> list[dict]:
        """
        Postprocess YOLO model outputs.

        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (h, w)

        Returns:
            List of detections
        """
        detections = []

        # outputs shape: (1, num_predictions, 4 + num_classes)
        # Format: [x_center, y_center, width, height, class_scores...]

        if len(outputs.shape) == 3:
            predictions = outputs[0]  # Remove batch dimension
        else:
            predictions = outputs

        # Transpose if needed (sometimes output is [num_classes + 4, num_predictions])
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        orig_h, orig_w = original_shape
        input_size = 640

        for pred in predictions:
            # Extract box coordinates and scores
            if len(pred) < 5:
                continue

            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]

            # Get class with highest score
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # Convert from normalized to pixel coordinates
            scale = max(orig_h, orig_w) / input_size

            x1 = (x_center - width / 2) * scale
            y1 = (y_center - height / 2) * scale
            x2 = (x_center + width / 2) * scale
            y2 = (y_center + height / 2) * scale

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": class_id,
                    "confidence": confidence,
                }
            )

        # Apply NMS
        detections = self.apply_nms(detections)

        return detections

    def apply_nms(self, detections: list[dict]) -> list[dict]:
        """
        Apply Non-Maximum Suppression.

        Args:
            detections: List of detections

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        # Group by class
        class_detections: dict[int, list[dict]] = {}
        for det in detections:
            cls = det["class"]
            if cls not in class_detections:
                class_detections[cls] = []
            class_detections[cls].append(det)

        # Apply NMS per class
        final_detections = []
        for _cls, dets in class_detections.items():
            # Sort by confidence
            dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)

            keep = []
            while dets:
                # Keep highest confidence detection
                best = dets.pop(0)
                keep.append(best)

                # Remove overlapping detections
                dets = [
                    det
                    for det in dets
                    if self.calculate_iou(best["bbox"], det["bbox"]) < self.iou_threshold
                ]

            final_detections.extend(keep)

        return final_detections

    def calculate_iou(self, box1: list[float], box2: list[float]) -> float:
        """
        Calculate IoU between two boxes.

        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


class YOLOv8InferenceEngine(InferenceEngine):
    """Inference engine for YOLOv8 models (PyTorch)."""

    def __init__(self, *args, **kwargs):
        """Initialize YOLOv8 inference engine."""
        super().__init__(*args, **kwargs)

        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "ultralytics package not installed. " "Install with: pip install ultralytics"
            ) from err

        logger.info("Loading YOLOv8 model")
        self.model = YOLO(str(self.model_path))

        # Get class names from model if not provided
        if not self.class_names:
            self.class_names = list(self.model.names.values())

        logger.info("YOLOv8 model loaded successfully")
        logger.info(f"  Classes: {len(self.class_names)}")

    def predict(self, image: np.ndarray | Path) -> list[dict]:
        """
        Run inference on image.

        Args:
            image: Input image (numpy array or path)

        Returns:
            List of detections
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # Parse results
        detections = []
        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confidences, strict=False):
                    detections.append(
                        {
                            "bbox": box.tolist(),
                            "class": int(cls),
                            "confidence": float(conf),
                        }
                    )

        return detections


class InferenceBenchmark:
    """Benchmark inference performance."""

    def __init__(self, engine: InferenceEngine):
        """
        Initialize benchmark.

        Args:
            engine: Inference engine to benchmark
        """
        self.engine = engine

    def benchmark_image(
        self,
        image_path: Path,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark inference on single image.

        Args:
            image_path: Path to test image
            num_iterations: Number of iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking on {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Warmup
        logger.info(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            _ = self.engine.predict(image)

        # Benchmark
        logger.info(f"Benchmarking ({num_iterations} iterations)...")
        timings = []

        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start = time.perf_counter()
            _ = self.engine.predict(image)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms

        results = {
            "mean_ms": float(np.mean(timings)),
            "std_ms": float(np.std(timings)),
            "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "fps": float(1000 / np.mean(timings)),
        }

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark Results")
        logger.info("=" * 60)
        logger.info(f"Mean:   {results['mean_ms']:.2f} ms")
        logger.info(f"Std:    {results['std_ms']:.2f} ms")
        logger.info(f"Median: {results['median_ms']:.2f} ms")
        logger.info(f"Min:    {results['min_ms']:.2f} ms")
        logger.info(f"Max:    {results['max_ms']:.2f} ms")
        logger.info(f"P95:    {results['p95_ms']:.2f} ms")
        logger.info(f"P99:    {results['p99_ms']:.2f} ms")
        logger.info(f"FPS:    {results['fps']:.2f}")
        logger.info("=" * 60 + "\n")

        return results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference on images with trained models")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_results"),
        help="Output directory for results (default: inference_results)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        help="Path to classes.txt file",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "onnx", "yolov8"],
        help="Model type (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detections to JSON",
    )

    args = parser.parse_args()

    # Load class names if provided
    class_names = []
    if args.classes:
        with open(args.classes) as f:
            class_names = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(class_names)} class names")

    # Auto-detect model type
    model_type = args.model_type
    if model_type == "auto":
        if args.model.suffix == ".onnx":
            model_type = "onnx"
        elif args.model.suffix in [".pt", ".pth"]:
            model_type = "yolov8"
        else:
            raise ValueError(f"Cannot auto-detect model type for {args.model.suffix}")

    # Create inference engine
    if model_type == "onnx":
        engine = ONNXInferenceEngine(
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            class_names=class_names,
        )
    elif model_type == "yolov8":
        engine = YOLOv8InferenceEngine(
            model_path=args.model,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            class_names=class_names,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Get input images
    if args.input.is_file():
        image_files = [args.input]
    elif args.input.is_dir():
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(args.input.glob(ext))
        image_files = sorted(image_files)
    else:
        raise ValueError(f"Input not found: {args.input}")

    logger.info(f"Found {len(image_files)} images")

    # Run benchmark if requested
    if args.benchmark and len(image_files) > 0:
        benchmark = InferenceBenchmark(engine)
        results = benchmark.benchmark_image(
            image_files[0],
            num_iterations=args.benchmark_iterations,
        )

        # Save benchmark results
        benchmark_file = args.output / "benchmark_results.json"
        with open(benchmark_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {benchmark_file}")

    # Process images
    all_results = {}

    for img_path in tqdm(image_files, desc="Processing images"):
        # Run inference
        detections = engine.predict(img_path)

        logger.info(f"{img_path.name}: {len(detections)} detections")

        # Save results
        all_results[str(img_path)] = detections

        # Visualize if requested
        if not args.no_visualize:
            image = cv2.imread(str(img_path))
            output_path = args.output / f"{img_path.stem}_viz{img_path.suffix}"
            engine.visualize(
                image,
                detections,
                output_path=output_path,
                show=args.show,
            )

    # Save JSON if requested
    if args.save_json:
        json_file = args.output / "detections.json"
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Detections saved to {json_file}")

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
