"""Local filesystem storage backend."""

import shutil
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aiofiles.os

from app.services.storage.base import StorageBackendInterface, StoredFile


class LocalStorageBackend(StorageBackendInterface):
    """Storage backend using local filesystem.

    Suitable for:
    - Local development
    - Single-server deployments
    - qontinui-train local processing
    """

    def __init__(self, base_path: str):
        """Initialize local storage backend.

        Args:
            base_path: Base directory for storing files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _full_path(self, relative_path: str) -> Path:
        """Get full path from relative storage path."""
        return self.base_path / relative_path

    async def store_video(
        self, session_id: str, video_data: BinaryIO, filename: str, content_type: str = "video/mp4"
    ) -> StoredFile:
        """Store a video file on local filesystem."""
        relative_path = self._generate_video_path(session_id, filename)
        full_path = self._full_path(relative_path)

        # Create directory structure
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write video data
        async with aiofiles.open(full_path, "wb") as f:
            # Read and write in chunks to handle large files
            while True:
                chunk = video_data.read(8192)
                if not chunk:
                    break
                await f.write(chunk)

        # Get file size
        size_bytes = await aiofiles.os.path.getsize(full_path)

        return StoredFile(
            path=relative_path, size_bytes=size_bytes, content_type=content_type, url=str(full_path)
        )

    async def store_frame(
        self,
        session_id: str,
        frame_number: int,
        frame_data: bytes,
        content_type: str = "image/jpeg",
    ) -> StoredFile:
        """Store an extracted video frame on local filesystem."""
        extension = "jpg" if "jpeg" in content_type else "png"
        relative_path = self._generate_frame_path(session_id, frame_number, extension)
        full_path = self._full_path(relative_path)

        # Create directory structure
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write frame data
        async with aiofiles.open(full_path, "wb") as f:
            await f.write(frame_data)

        return StoredFile(
            path=relative_path,
            size_bytes=len(frame_data),
            content_type=content_type,
            url=str(full_path),
        )

    async def get_video_path(self, session_id: str, filename: str) -> str:
        """Get the full path to a video file."""
        relative_path = self._generate_video_path(session_id, filename)
        return str(self._full_path(relative_path))

    async def get_frame_path(self, session_id: str, frame_number: int) -> str | None:
        """Get the path to a cached frame if it exists."""
        # Try both jpg and png extensions
        for ext in ["jpg", "png"]:
            relative_path = self._generate_frame_path(session_id, frame_number, ext)
            full_path = self._full_path(relative_path)
            if full_path.exists():
                return str(full_path)
        return None

    async def get_video_stream(self, path: str) -> BinaryIO:
        """Get a stream for reading video data."""
        full_path = self._full_path(path)
        return open(full_path, "rb")

    async def get_frame_data(self, path: str) -> bytes:
        """Get frame image data."""
        full_path = self._full_path(path)
        async with aiofiles.open(full_path, "rb") as f:
            data: bytes = await f.read()
            return data

    async def delete_session_data(self, session_id: str) -> bool:
        """Delete all data for a capture session."""
        session_path = self.base_path / "captures" / session_id
        if session_path.exists():
            shutil.rmtree(session_path)
            return True
        return False

    async def get_video_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a URL for accessing the video (local file path)."""
        full_path = self._full_path(path)
        return f"file://{full_path}"

    async def get_frame_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a URL for accessing a frame image (local file path)."""
        full_path = self._full_path(path)
        return f"file://{full_path}"

    async def video_exists(self, path: str) -> bool:
        """Check if a video file exists."""
        full_path = self._full_path(path)
        return full_path.exists()

    async def frame_exists(self, path: str) -> bool:
        """Check if a frame file exists."""
        full_path = self._full_path(path)
        return full_path.exists()

    async def list_sessions(self) -> list[str]:
        """List all capture session IDs."""
        captures_path = self.base_path / "captures"
        if not captures_path.exists():
            return []
        return [d.name for d in captures_path.iterdir() if d.is_dir()]

    async def get_session_size(self, session_id: str) -> int:
        """Get total size of a session's data in bytes."""
        session_path = self.base_path / "captures" / session_id
        if not session_path.exists():
            return 0

        total_size = 0
        for file_path in session_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
