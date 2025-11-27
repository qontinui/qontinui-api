"""Base interface for storage backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO


@dataclass
class StoredFile:
    """Information about a stored file."""

    path: str  # Storage path (local path or S3 key)
    size_bytes: int
    content_type: str
    etag: str | None = None  # For cache validation
    url: str | None = None  # Public URL if available


class StorageBackendInterface(ABC):
    """Abstract base class for storage backends.

    Implementations should handle:
    - Video file storage and retrieval
    - Frame image storage and retrieval
    - Path generation for organized storage
    """

    @abstractmethod
    async def store_video(
        self, session_id: str, video_data: BinaryIO, filename: str, content_type: str = "video/mp4"
    ) -> StoredFile:
        """Store a video file.

        Args:
            session_id: Capture session identifier
            video_data: Video file data stream
            filename: Original filename
            content_type: MIME type of the video

        Returns:
            StoredFile with path and metadata
        """
        pass

    @abstractmethod
    async def store_frame(
        self,
        session_id: str,
        frame_number: int,
        frame_data: bytes,
        content_type: str = "image/jpeg",
    ) -> StoredFile:
        """Store an extracted video frame.

        Args:
            session_id: Capture session identifier
            frame_number: Frame number in the video
            frame_data: Frame image data
            content_type: MIME type of the image

        Returns:
            StoredFile with path and metadata
        """
        pass

    @abstractmethod
    async def get_video_path(self, session_id: str, filename: str) -> str:
        """Get the full path to a video file.

        Args:
            session_id: Capture session identifier
            filename: Video filename

        Returns:
            Full path (local) or URL (S3) to the video
        """
        pass

    @abstractmethod
    async def get_frame_path(self, session_id: str, frame_number: int) -> str | None:
        """Get the path to a cached frame if it exists.

        Args:
            session_id: Capture session identifier
            frame_number: Frame number

        Returns:
            Path to frame if cached, None otherwise
        """
        pass

    @abstractmethod
    async def get_video_stream(self, path: str) -> BinaryIO:
        """Get a stream for reading video data.

        Args:
            path: Storage path to the video

        Returns:
            Binary stream for reading video data
        """
        pass

    @abstractmethod
    async def get_frame_data(self, path: str) -> bytes:
        """Get frame image data.

        Args:
            path: Storage path to the frame

        Returns:
            Frame image data as bytes
        """
        pass

    @abstractmethod
    async def delete_session_data(self, session_id: str) -> bool:
        """Delete all data for a capture session.

        Args:
            session_id: Capture session identifier

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def get_video_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a URL for accessing the video.

        For local storage, returns a file:// URL or server path.
        For S3, returns a pre-signed URL.

        Args:
            path: Storage path to the video
            expires_in: URL expiration time in seconds (for S3)

        Returns:
            URL for accessing the video
        """
        pass

    @abstractmethod
    async def get_frame_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a URL for accessing a frame image.

        Args:
            path: Storage path to the frame
            expires_in: URL expiration time in seconds (for S3)

        Returns:
            URL for accessing the frame
        """
        pass

    @abstractmethod
    async def video_exists(self, path: str) -> bool:
        """Check if a video file exists.

        Args:
            path: Storage path to check

        Returns:
            True if the video exists
        """
        pass

    @abstractmethod
    async def frame_exists(self, path: str) -> bool:
        """Check if a frame file exists.

        Args:
            path: Storage path to check

        Returns:
            True if the frame exists
        """
        pass

    def _generate_video_path(self, session_id: str, filename: str) -> str:
        """Generate organized path for video storage.

        Structure: captures/{session_id}/video/{filename}
        """
        return f"captures/{session_id}/video/{filename}"

    def _generate_frame_path(
        self, session_id: str, frame_number: int, extension: str = "jpg"
    ) -> str:
        """Generate organized path for frame storage.

        Structure: captures/{session_id}/frames/{frame_number:08d}.{ext}
        """
        return f"captures/{session_id}/frames/{frame_number:08d}.{extension}"
