"""Frame extraction service for video files.

Uses FFmpeg to extract frames from video files at specific timestamps
or frame numbers. Supports caching for frequently accessed frames.
"""

import asyncio
import tempfile
from pathlib import Path

from app.services.storage.base import StorageBackendInterface


class FrameExtractor:
    """Service for extracting frames from video files.

    Uses FFmpeg for efficient frame extraction with support for:
    - Extraction by timestamp (milliseconds)
    - Extraction by frame number
    - Keyframe-based seeking for faster extraction
    - Frame caching in storage backend
    """

    def __init__(
        self,
        storage: StorageBackendInterface,
        ffmpeg_path: str = "ffmpeg",
        cache_frames: bool = True,
    ):
        """Initialize frame extractor.

        Args:
            storage: Storage backend for video files and cached frames
            ffmpeg_path: Path to ffmpeg executable
            cache_frames: Whether to cache extracted frames
        """
        self.storage = storage
        self.ffmpeg_path = ffmpeg_path
        self.cache_frames = cache_frames

    async def extract_frame_by_timestamp(
        self, session_id: str, video_filename: str, timestamp_ms: int, quality: int = 90
    ) -> bytes:
        """Extract a frame at a specific timestamp.

        Args:
            session_id: Capture session identifier
            video_filename: Video filename
            timestamp_ms: Timestamp in milliseconds from video start
            quality: JPEG quality (1-100)

        Returns:
            Frame image data as bytes (JPEG format)
        """
        # Calculate approximate frame number for caching
        # This is a heuristic; actual frame number depends on FPS
        frame_number = int(timestamp_ms / 33.33)  # Assuming ~30fps

        # Check cache first
        if self.cache_frames:
            cached_path = await self.storage.get_frame_path(session_id, frame_number)
            if cached_path:
                return await self.storage.get_frame_data(cached_path)

        # Get video path
        video_path = await self.storage.get_video_path(session_id, video_filename)

        # Extract frame using FFmpeg
        frame_data = await self._extract_with_ffmpeg(
            video_path, timestamp_ms=timestamp_ms, quality=quality
        )

        # Cache the frame
        if self.cache_frames:
            await self.storage.store_frame(
                session_id, frame_number, frame_data, content_type="image/jpeg"
            )

        return frame_data

    async def extract_frame_by_number(
        self,
        session_id: str,
        video_filename: str,
        frame_number: int,
        fps: float = 30.0,
        quality: int = 90,
    ) -> bytes:
        """Extract a specific frame by frame number.

        Args:
            session_id: Capture session identifier
            video_filename: Video filename
            frame_number: Frame number (0-based)
            fps: Video frames per second
            quality: JPEG quality (1-100)

        Returns:
            Frame image data as bytes (JPEG format)
        """
        # Check cache first
        if self.cache_frames:
            cached_path = await self.storage.get_frame_path(session_id, frame_number)
            if cached_path:
                return await self.storage.get_frame_data(cached_path)

        # Calculate timestamp from frame number
        timestamp_ms = int((frame_number / fps) * 1000)

        # Get video path
        video_path = await self.storage.get_video_path(session_id, video_filename)

        # Extract frame using FFmpeg
        frame_data = await self._extract_with_ffmpeg(
            video_path, timestamp_ms=timestamp_ms, quality=quality
        )

        # Cache the frame
        if self.cache_frames:
            await self.storage.store_frame(
                session_id, frame_number, frame_data, content_type="image/jpeg"
            )

        return frame_data

    async def extract_frames_range(
        self,
        session_id: str,
        video_filename: str,
        start_frame: int,
        end_frame: int,
        fps: float = 30.0,
        quality: int = 85,
    ) -> list[tuple[int, bytes]]:
        """Extract a range of frames.

        Args:
            session_id: Capture session identifier
            video_filename: Video filename
            start_frame: Starting frame number
            end_frame: Ending frame number (inclusive)
            fps: Video frames per second
            quality: JPEG quality (1-100)

        Returns:
            List of (frame_number, frame_data) tuples
        """
        frames = []
        for frame_num in range(start_frame, end_frame + 1):
            frame_data = await self.extract_frame_by_number(
                session_id, video_filename, frame_num, fps, quality
            )
            frames.append((frame_num, frame_data))
        return frames

    async def _extract_with_ffmpeg(
        self, video_path: str, timestamp_ms: int, quality: int = 90
    ) -> bytes:
        """Extract a single frame using FFmpeg.

        Args:
            video_path: Path to video file
            timestamp_ms: Timestamp in milliseconds
            quality: JPEG quality (1-100)

        Returns:
            Frame image data as bytes
        """
        # Convert milliseconds to seconds for FFmpeg
        timestamp_sec = timestamp_ms / 1000.0

        # Use a temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            output_path = tmp.name

        try:
            # FFmpeg command for frame extraction
            # -ss before -i enables fast seeking
            # -frames:v 1 extracts only one frame
            cmd = [
                self.ffmpeg_path,
                "-ss",
                str(timestamp_sec),
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                str(int((100 - quality) / 3)),  # FFmpeg quality scale is inverted
                "-y",  # Overwrite output
                output_path,
            ]

            # Run FFmpeg asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")

            # Read the extracted frame
            with open(output_path, "rb") as f:
                return f.read()

        finally:
            # Clean up temporary file
            Path(output_path).unlink(missing_ok=True)

    async def build_frame_index(
        self, session_id: str, video_filename: str, keyframes_only: bool = True
    ) -> list[dict]:
        """Build a frame index for a video.

        Extracts timestamp and position information for frames,
        optionally only for keyframes (I-frames) for faster seeking.

        Args:
            session_id: Capture session identifier
            video_filename: Video filename
            keyframes_only: If True, only index keyframes

        Returns:
            List of frame index entries
        """
        video_path = await self.storage.get_video_path(session_id, video_filename)

        # Use ffprobe to get frame information
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pkt_pts_time,pkt_pos,key_frame,pict_type",
            "-of",
            "json",
            video_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffprobe error: {stderr.decode()}")

        import json

        probe_data = json.loads(stdout.decode())

        frames = []
        frame_number = 0
        for frame in probe_data.get("frames", []):
            is_keyframe = frame.get("key_frame") == 1

            if keyframes_only and not is_keyframe:
                frame_number += 1
                continue

            timestamp_sec = float(frame.get("pkt_pts_time", 0))
            byte_offset = int(frame.get("pkt_pos", 0)) if frame.get("pkt_pos") else None

            frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp_ms": int(timestamp_sec * 1000),
                    "byte_offset": byte_offset,
                    "is_keyframe": is_keyframe,
                }
            )

            frame_number += 1

        return frames

    async def get_video_info(self, session_id: str, video_filename: str) -> dict:
        """Get video metadata.

        Args:
            session_id: Capture session identifier
            video_filename: Video filename

        Returns:
            Dictionary with video metadata (duration, fps, dimensions, etc.)
        """
        video_path = await self.storage.get_video_path(session_id, video_filename)

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            video_path,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"ffprobe error: {stderr.decode()}")

        import json

        probe_data = json.loads(stdout.decode())

        # Extract video stream info
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("No video stream found")

        # Parse frame rate
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = (
            float(fps_parts[0]) / float(fps_parts[1])
            if len(fps_parts) == 2
            else float(fps_parts[0])
        )

        return {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": fps,
            "codec": video_stream.get("codec_name"),
            "duration_ms": int(float(probe_data.get("format", {}).get("duration", 0)) * 1000),
            "total_frames": int(video_stream.get("nb_frames", 0)) or None,
            "bit_rate": int(video_stream.get("bit_rate", 0)) or None,
        }
