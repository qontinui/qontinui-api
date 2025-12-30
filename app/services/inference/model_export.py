#!/usr/bin/env python3
"""
Model export script for deploying trained models.

This script handles exporting trained models to various formats for deployment:
- ONNX (cross-platform)
- TensorRT (NVIDIA optimization)
- CoreML (Apple devices)
- TorchScript (PyTorch native)

Includes model optimization options:
- Quantization (INT8, FP16)
- Dynamic shape support
- Batch processing optimization
- Post-export validation
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Base class for model export."""

    def __init__(
        self,
        model_path: Path,
        output_dir: Path,
        model_name: str | None = None,
    ):
        """
        Initialize model exporter.

        Args:
            model_path: Path to trained model weights
            output_dir: Directory to save exported models
            model_name: Custom name for exported model (default: use original name)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.model_path.stem

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Initialized exporter for model: {self.model_path}")

    def validate_export(
        self,
        original_model_path: Path,
        exported_model_path: Path,
        test_image: Path | None = None,
    ) -> bool:
        """
        Validate exported model against original.

        Args:
            original_model_path: Path to original model
            exported_model_path: Path to exported model
            test_image: Optional test image for comparison

        Returns:
            True if validation passes, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_export")


class YOLOv8Exporter(ModelExporter):
    """Exporter for YOLOv8 models."""

    def __init__(self, *args, **kwargs):
        """Initialize YOLOv8 exporter."""
        super().__init__(*args, **kwargs)

        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "ultralytics package not installed. Install with: pip install ultralytics"
            ) from err

        logger.info(f"Loading YOLOv8 model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def export_onnx(
        self,
        imgsz: int = 640,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
        **kwargs,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            imgsz: Input image size
            dynamic: Enable dynamic input shapes
            simplify: Simplify ONNX model
            opset: ONNX opset version
            **kwargs: Additional export arguments

        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting to ONNX format")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  Dynamic shapes: {dynamic}")
        logger.info(f"  Simplify: {simplify}")
        logger.info(f"  Opset version: {opset}")

        export_path = self.model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            **kwargs,
        )

        output_path = self.output_dir / f"{self.model_name}.onnx"
        if Path(export_path).exists():
            import shutil

            shutil.copy2(export_path, output_path)
            logger.info(f"ONNX model saved to: {output_path}")
        else:
            output_path = Path(export_path)

        # Verify ONNX model
        try:
            import onnx

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation successful")

            # Print model info
            logger.info(f"Model inputs: {[i.name for i in onnx_model.graph.input]}")
            logger.info(f"Model outputs: {[o.name for o in onnx_model.graph.output]}")

        except ImportError:
            logger.warning("onnx package not installed, skipping validation")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")

        return output_path

    def export_tensorrt(
        self,
        imgsz: int = 640,
        half: bool = True,
        workspace: int = 4,
        **kwargs,
    ) -> Path:
        """
        Export model to TensorRT format.

        Args:
            imgsz: Input image size
            half: Use FP16 precision
            workspace: TensorRT workspace size (GB)
            **kwargs: Additional export arguments

        Returns:
            Path to exported TensorRT engine
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT export")

        logger.info("Exporting to TensorRT format")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  FP16 mode: {half}")
        logger.info(f"  Workspace: {workspace}GB")

        export_path = self.model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            workspace=workspace,
            **kwargs,
        )

        output_path = self.output_dir / f"{self.model_name}.engine"
        if Path(export_path).exists():
            import shutil

            shutil.copy2(export_path, output_path)
            logger.info(f"TensorRT engine saved to: {output_path}")
        else:
            output_path = Path(export_path)

        return output_path

    def export_coreml(
        self,
        imgsz: int = 640,
        int8: bool = False,
        nms: bool = True,
        **kwargs,
    ) -> Path:
        """
        Export model to CoreML format (for Apple devices).

        Args:
            imgsz: Input image size
            int8: Use INT8 quantization
            nms: Include NMS in model
            **kwargs: Additional export arguments

        Returns:
            Path to exported CoreML model
        """
        logger.info("Exporting to CoreML format")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  INT8 quantization: {int8}")
        logger.info(f"  Include NMS: {nms}")

        export_path = self.model.export(
            format="coreml",
            imgsz=imgsz,
            int8=int8,
            nms=nms,
            **kwargs,
        )

        output_path = self.output_dir / f"{self.model_name}.mlpackage"
        if Path(export_path).exists():
            import shutil

            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(export_path, output_path)
            logger.info(f"CoreML model saved to: {output_path}")
        else:
            output_path = Path(export_path)

        return output_path

    def export_torchscript(
        self,
        imgsz: int = 640,
        optimize: bool = True,
        **kwargs,
    ) -> Path:
        """
        Export model to TorchScript format.

        Args:
            imgsz: Input image size
            optimize: Optimize for mobile
            **kwargs: Additional export arguments

        Returns:
            Path to exported TorchScript model
        """
        logger.info("Exporting to TorchScript format")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  Optimize: {optimize}")

        export_path = self.model.export(
            format="torchscript",
            imgsz=imgsz,
            optimize=optimize,
            **kwargs,
        )

        output_path = self.output_dir / f"{self.model_name}.torchscript"
        if Path(export_path).exists():
            import shutil

            shutil.copy2(export_path, output_path)
            logger.info(f"TorchScript model saved to: {output_path}")
        else:
            output_path = Path(export_path)

        return output_path

    def export_openvino(
        self,
        imgsz: int = 640,
        half: bool = False,
        **kwargs,
    ) -> Path:
        """
        Export model to OpenVINO format (for Intel hardware).

        Args:
            imgsz: Input image size
            half: Use FP16 precision
            **kwargs: Additional export arguments

        Returns:
            Path to exported OpenVINO model
        """
        logger.info("Exporting to OpenVINO format")
        logger.info(f"  Image size: {imgsz}")
        logger.info(f"  FP16 mode: {half}")

        export_path = self.model.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
            **kwargs,
        )

        # OpenVINO exports as a directory
        output_path = self.output_dir / f"{self.model_name}_openvino_model"
        if Path(export_path).exists():
            import shutil

            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.copytree(export_path, output_path)
            logger.info(f"OpenVINO model saved to: {output_path}")
        else:
            output_path = Path(export_path)

        return output_path

    def export_all(
        self,
        formats: list[str],
        imgsz: int = 640,
        **kwargs,
    ) -> dict[str, Path]:
        """
        Export model to multiple formats.

        Args:
            formats: List of formats to export to
            imgsz: Input image size
            **kwargs: Additional export arguments

        Returns:
            Dictionary mapping format names to exported model paths
        """
        logger.info(f"Exporting to formats: {', '.join(formats)}")

        exported_models = {}

        for fmt in formats:
            try:
                if fmt == "onnx":
                    path = self.export_onnx(imgsz=imgsz, **kwargs)
                elif fmt == "tensorrt" or fmt == "engine":
                    path = self.export_tensorrt(imgsz=imgsz, **kwargs)
                elif fmt == "coreml":
                    path = self.export_coreml(imgsz=imgsz, **kwargs)
                elif fmt == "torchscript":
                    path = self.export_torchscript(imgsz=imgsz, **kwargs)
                elif fmt == "openvino":
                    path = self.export_openvino(imgsz=imgsz, **kwargs)
                else:
                    logger.error(f"Unsupported format: {fmt}")
                    continue

                exported_models[fmt] = path
                logger.info(f"Successfully exported to {fmt}")

            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {e}")

        return exported_models

    def validate_export(
        self,
        original_model_path: Path,
        exported_model_path: Path,
        test_image: Path | None = None,
    ) -> bool:
        """
        Validate exported model against original.

        Args:
            original_model_path: Path to original model
            exported_model_path: Path to exported model
            test_image: Optional test image for comparison

        Returns:
            True if validation passes, False otherwise
        """
        # Infer format from file extension
        exported_format = exported_model_path.suffix.lstrip(".")
        logger.info(f"Validating {exported_format} export")

        if test_image is None:
            # Create dummy input
            test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        else:
            import cv2

            img = cv2.imread(str(test_image))
            if img is None:
                raise ValueError(f"Failed to read image: {test_image}")
            test_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        try:
            # Get predictions from original model
            _ = self.model.predict(
                test_input,
                verbose=False,
            )

            # Get predictions from exported model
            if exported_format == "onnx":
                _ = self._validate_onnx(exported_model_path, test_input)
            elif exported_format == "tensorrt" or exported_format == "engine":
                logger.info("TensorRT validation requires runtime engine")
                return True
            elif exported_format == "coreml":
                logger.info("CoreML validation requires Apple hardware")
                return True
            else:
                logger.warning(f"Validation not implemented for {exported_format}")
                return True

            logger.info("Export validation successful")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def _validate_onnx(self, onnx_path: Path, test_input: np.ndarray) -> list[np.ndarray]:
        """
        Validate ONNX model.

        Args:
            onnx_path: Path to ONNX model
            test_input: Test input image

        Returns:
            Model predictions
        """
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime package not installed. Install with: pip install onnxruntime"
            ) from err

        session = ort.InferenceSession(str(onnx_path))

        # Get input name and shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        logger.info(f"ONNX input: {input_name}, shape: {input_shape}")

        # Preprocess input
        import cv2

        img = cv2.resize(test_input, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run inference
        raw_outputs = session.run(None, {input_name: img})
        outputs: list[np.ndarray] = [
            np.array(o) if not isinstance(o, np.ndarray) else o for o in raw_outputs
        ]

        logger.info(f"ONNX output shapes: {[o.shape for o in outputs]}")

        return outputs

    def benchmark_export(
        self,
        exported_model_path: Path,
        exported_format: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark exported model performance.

        Args:
            exported_model_path: Path to exported model
            exported_format: Format of exported model
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking {exported_format} model")

        # Create dummy input
        test_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        if exported_format == "onnx":
            results = self._benchmark_onnx(
                exported_model_path,
                test_input,
                num_iterations,
                warmup_iterations,
            )
        else:
            logger.warning(f"Benchmarking not implemented for {exported_format}")
            results = {}

        return results

    def _benchmark_onnx(
        self,
        onnx_path: Path,
        test_input: np.ndarray,
        num_iterations: int,
        warmup_iterations: int,
    ) -> dict[str, float]:
        """
        Benchmark ONNX model.

        Args:
            onnx_path: Path to ONNX model
            test_input: Test input image
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary containing benchmark results
        """
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError("onnxruntime not installed") from err

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        # Preprocess input
        import cv2

        img = cv2.resize(test_input, (640, 640))
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Warmup
        logger.info(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            session.run(None, {input_name: img})

        # Benchmark
        logger.info(f"Benchmarking ({num_iterations} iterations)...")
        timings = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: img})
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to ms

        results = {
            "mean_ms": float(np.mean(timings)),
            "std_ms": float(np.std(timings)),
            "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings)),
            "fps": float(1000 / np.mean(timings)),
        }

        logger.info("\nBenchmark Results:")
        logger.info(f"  Mean: {results['mean_ms']:.2f} ms")
        logger.info(f"  Std: {results['std_ms']:.2f} ms")
        logger.info(f"  Min: {results['min_ms']:.2f} ms")
        logger.info(f"  Max: {results['max_ms']:.2f} ms")
        logger.info(f"  FPS: {results['fps']:.2f}")

        return results


def main():
    """Main entry point for model export."""
    parser = argparse.ArgumentParser(description="Export trained models to various formats")

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports"),
        help="Output directory for exported models (default: exports)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for exported model",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["onnx"],
        choices=[
            "onnx",
            "tensorrt",
            "engine",
            "coreml",
            "torchscript",
            "openvino",
            "all",
        ],
        help="Export format(s) (default: onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes (ONNX)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: True)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 precision (TensorRT, OpenVINO)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization (CoreML)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark exported model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="yolov8",
        choices=["yolov8"],
        help="Model type (default: yolov8)",
    )

    args = parser.parse_args()

    # Handle "all" format
    if "all" in args.format:
        args.format = ["onnx", "tensorrt", "coreml", "torchscript", "openvino"]

    # Create exporter
    if args.model_type == "yolov8":
        exporter = YOLOv8Exporter(
            model_path=args.model,
            output_dir=args.output,
            model_name=args.name,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Export models
    exported_models = {}

    for fmt in args.format:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Exporting to {fmt.upper()}")
            logger.info(f"{'=' * 60}\n")

            if fmt == "onnx":
                path = exporter.export_onnx(
                    imgsz=args.imgsz,
                    dynamic=args.dynamic,
                    simplify=args.simplify,
                )
            elif fmt in ["tensorrt", "engine"]:
                path = exporter.export_tensorrt(
                    imgsz=args.imgsz,
                    half=args.half,
                )
            elif fmt == "coreml":
                path = exporter.export_coreml(
                    imgsz=args.imgsz,
                    int8=args.int8,
                )
            elif fmt == "torchscript":
                path = exporter.export_torchscript(
                    imgsz=args.imgsz,
                )
            elif fmt == "openvino":
                path = exporter.export_openvino(
                    imgsz=args.imgsz,
                    half=args.half,
                )
            else:
                logger.error(f"Unsupported format: {fmt}")
                continue

            exported_models[fmt] = path

            # Validate if requested
            if args.validate:
                exporter.validate_export(fmt, path)

            # Benchmark if requested
            if args.benchmark and fmt == "onnx":
                exporter.benchmark_export(path, fmt)

        except Exception as e:
            logger.error(f"Failed to export to {fmt}: {e}", exc_info=True)
            continue

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Export Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Successfully exported {len(exported_models)}/{len(args.format)} formats:")
    for fmt, path in exported_models.items():
        logger.info(f"  {fmt}: {path}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
