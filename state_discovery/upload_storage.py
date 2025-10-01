"""Upload storage manager for state discovery.

This module handles screenshot upload storage and retrieval.
Single Responsibility: Manage temporary storage of uploaded screenshots.
"""

import logging
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
        print(f"[STORAGE] Stored {upload_id}: {len(screenshots)} screenshots")

        return upload_id

    def get_upload(self, upload_id: str) -> Upload | None:
        """Retrieve an upload by ID."""
        upload = self.uploads.get(upload_id)
        if upload:
            logger.debug(f"Retrieved upload {upload_id}")
            print(f"[STORAGE] Retrieved {upload_id}")
        else:
            logger.warning(f"Upload {upload_id} not found")
            print(f"[STORAGE] Not found: {upload_id}")
        return upload

    def delete_upload(self, upload_id: str) -> bool:
        """Delete an upload from storage."""
        if upload_id in self.uploads:
            del self.uploads[upload_id]
            logger.info(f"Deleted upload {upload_id}")
            print(f"[STORAGE] Deleted {upload_id}")
            return True
        return False

    def list_uploads(self) -> list[str]:
        """List all upload IDs."""
        return list(self.uploads.keys())

    def clear_old_uploads(self, max_age_seconds: float = 3600):
        """Clear uploads older than specified age."""
        # Implementation would check timestamps and remove old uploads
        pass


# Global instance
storage = UploadStorage()
