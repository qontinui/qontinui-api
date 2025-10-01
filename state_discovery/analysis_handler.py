"""Analysis task handler for state discovery.

This module handles the actual analysis execution.
Single Responsibility: Execute state discovery analysis.
"""

import asyncio
import logging
import time

import numpy as np

from qontinui.discovery import PixelStabilityMatrixAnalyzer
from qontinui.discovery.models import AnalysisConfig

from .upload_storage import Upload
from .websocket_manager import manager

logger = logging.getLogger(__name__)


class AnalysisHandler:
    """Handles state discovery analysis execution.

    Single Responsibility: Execute analysis and report progress.
    """

    def __init__(self):
        logger.info("AnalysisHandler initialized")
        print("[ANALYSIS] Handler initialized")

    async def execute_analysis(
        self,
        analysis_id: str,
        upload: Upload,
        config: AnalysisConfig,
        region: tuple[int, int, int, int] | None = None,
    ):
        """Execute the state discovery analysis.

        This is the main analysis execution method that:
        1. Prepares screenshots
        2. Runs the analyzer
        3. Reports progress via WebSocket
        4. Returns results
        """
        print(f"[ANALYSIS] Starting analysis {analysis_id}")
        logger.info(f"Starting analysis {analysis_id} with {len(upload.screenshots)} screenshots")

        try:
            # Send initial progress
            await manager.send_progress(
                analysis_id,
                "initialization",
                0,
                f"Starting analysis with {len(upload.screenshots)} screenshots",
            )

            # Prepare screenshots (crop if needed)
            screenshots = self._prepare_screenshots(upload.screenshots, region)

            # Send preparation complete
            await manager.send_progress(analysis_id, "preparation", 10, "Screenshots prepared")

            # Create and run analyzer
            print(f"[ANALYSIS] Creating analyzer for {analysis_id}")
            analyzer = PixelStabilityMatrixAnalyzer(config)

            await manager.send_progress(
                analysis_id, "analyzing", 20, "Running pixel stability analysis..."
            )

            # Run analysis
            start_time = time.time()
            print(f"[ANALYSIS] Running analyzer for {analysis_id}")
            result = analyzer.analyze_screenshots(screenshots, region)
            elapsed = time.time() - start_time

            print(f"[ANALYSIS] Analysis complete for {analysis_id} in {elapsed:.2f}s")
            logger.info(
                f"Analysis {analysis_id} completed in {elapsed:.2f}s: "
                f"{len(result.state_images)} state images, {len(result.states)} states"
            )

            # Convert result to dict for serialization
            result_dict = {
                "state_images": [si.to_dict() for si in result.state_images],
                "states": [s.to_dict() for s in result.states],
                "statistics": result.statistics,
            }

            # Send completion
            await manager.send_complete(analysis_id, result_dict)

            return result_dict

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"[ANALYSIS] Error in {analysis_id}: {error_msg}")
            logger.error(f"Analysis {analysis_id} failed: {e}", exc_info=True)

            # Send error to client
            await manager.send_error(analysis_id, error_msg)
            raise

    def _prepare_screenshots(
        self, screenshots: list[np.ndarray], region: tuple[int, int, int, int] | None = None
    ) -> list[np.ndarray]:
        """Prepare screenshots for analysis (crop if needed)."""
        if not region:
            return screenshots

        x, y, width, height = region
        cropped = []

        for i, screenshot in enumerate(screenshots):
            h, w = screenshot.shape[:2]
            # Ensure bounds are valid
            x1 = max(0, min(x, w))
            y1 = max(0, min(y, h))
            x2 = max(0, min(x + width, w))
            y2 = max(0, min(y + height, h))

            cropped_img = screenshot[y1:y2, x1:x2]
            cropped.append(cropped_img)

            logger.debug(f"Screenshot {i}: cropped from {screenshot.shape} to {cropped_img.shape}")

        return cropped


class BackgroundTaskRunner:
    """Runs analysis tasks in the background.

    Single Responsibility: Manage background task execution.
    """

    def __init__(self):
        self.handler = AnalysisHandler()
        logger.info("BackgroundTaskRunner initialized")
        print("[RUNNER] Initialized")

    def run_analysis_sync(
        self,
        analysis_id: str,
        upload: Upload,
        config: AnalysisConfig,
        region: tuple[int, int, int, int] | None = None,
    ):
        """Synchronous wrapper for running analysis in background thread."""
        print(f"[RUNNER] Starting sync analysis for {analysis_id}")
        logger.info(f"Starting background analysis for {analysis_id}")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async analysis
            print(f"[RUNNER] Running async analysis for {analysis_id}")
            loop.run_until_complete(
                self.handler.execute_analysis(analysis_id, upload, config, region)
            )
            print(f"[RUNNER] Completed analysis for {analysis_id}")
            logger.info(f"Background analysis completed for {analysis_id}")

        except Exception as e:
            print(f"[RUNNER] Error in analysis {analysis_id}: {e}")
            logger.error(f"Background analysis failed for {analysis_id}: {e}", exc_info=True)

        finally:
            loop.close()
            print(f"[RUNNER] Closed event loop for {analysis_id}")


# Global instance
task_runner = BackgroundTaskRunner()
