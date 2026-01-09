"""Upload storage manager for state discovery.

This module handles screenshot upload storage and retrieval.
Single Responsibility: Manage temporary storage of uploaded screenshots.
"""

import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UploadMetadata:
    """Metadata for an uploaded screenshot."""

    id: str
    name: str
    size: int
    width: int
    height: int
    pixel_hash: str


@dataclass
class Upload:
    """Represents a collection of uploaded screenshots."""

    id: str
    screenshots: list[np.ndarray]
    metadata: list[UploadMetadata]
    timestamp: float


class UploadStorage:
    """Manages temporary storage of uploaded screenshots.

    Single Responsibility: Store and retrieve uploaded screenshots.
    """

    def __init__(self):
        self.uploads: dict[str, Upload] = {}
        logger.info("UploadStorage initialized")
        print("[STORAGE] Initialized")

    def store_upload(self, screenshots: list[np.ndarray], metadata: list[dict]) -> str:
        """Store uploaded screenshots and return upload ID."""
        upload_id = f"upload_{uuid.uuid4().hex[:12]}"

        # Convert metadata dicts to UploadMetadata objects
        meta_objects = [
            UploadMetadata(
                id=m["id"],
                name=m["name"],
                size=m["size"],
                width=m["width"],
                height=m["height"],
                pixel_hash=m["pixel_hash"],
            )
            for m in metadata
        ]

        upload = Upload(
            id=upload_id,
            screenshots=screenshots,
            metadata=meta_objects,
            timestamp=0,  # Will be set by caller
        )

        self.uploads[upload_id] = upload

        logger.info(f"Stored upload {upload_id} with {len(screenshots)} screenshots")
        logger.debug("[STORAGE] Stored {upload_id}: {len(screenshots)} screenshots")

        return upload_id

    def get_upload(self, upload_id: str) -> Upload | None:
        """Retrieve an upload by ID."""
        upload = self.uploads.get(upload_id)
        if upload:
            logger.debug(f"Retrieved upload {upload_id}")
            logger.debug("[STORAGE] Retrieved {upload_id}")
        else:
            logger.warning(f"Upload {upload_id} not found")
            logger.debug("[STORAGE] Not found: {upload_id}")
        return upload

    def delete_upload(self, upload_id: str) -> bool:
        """Delete an upload from storage."""
        if upload_id in self.uploads:
            del self.uploads[upload_id]
            logger.info(f"Deleted upload {upload_id}")
            logger.debug("[STORAGE] Deleted {upload_id}")
            return True
        return False

    def list_uploads(self) -> list[str]:
        """List all upload IDs."""
        return list(self.uploads.keys())

    def clear_old_uploads(self, max_age_seconds: float = 3600) -> int:
        """Clear uploads older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds. Uploads older than this will be removed.
                           Defaults to 3600 (1 hour).

        Returns:
            Number of uploads removed.
        """
        current_time = time.time()
        expired_ids: list[str] = []

        for upload_id, upload in self.uploads.items():
            age = current_time - upload.timestamp
            if age > max_age_seconds:
                expired_ids.append(upload_id)

        for upload_id in expired_ids:
            del self.uploads[upload_id]
            logger.debug(f"[STORAGE] Cleared expired upload {upload_id}")

        if expired_ids:
            logger.info(f"Cleared {len(expired_ids)} expired uploads")

        return len(expired_ids)


# Global instance
storage = UploadStorage()
