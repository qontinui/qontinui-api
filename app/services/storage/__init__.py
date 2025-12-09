"""Storage abstraction layer for video and media files.

This module provides a unified interface for storing and retrieving
video files and extracted frames, supporting both local filesystem
and AWS S3 backends.
"""

from app.services.storage.base import StorageBackendInterface
from app.services.storage.factory import (StorageConfig, get_default_storage,
                                          get_storage_backend)
from app.services.storage.frame_extractor import FrameExtractor
from app.services.storage.local import LocalStorageBackend
from app.services.storage.s3 import S3StorageBackend

__all__ = [
    "StorageBackendInterface",
    "LocalStorageBackend",
    "S3StorageBackend",
    "get_storage_backend",
    "get_default_storage",
    "StorageConfig",
    "FrameExtractor",
]
