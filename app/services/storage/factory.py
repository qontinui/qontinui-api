"""Factory for creating storage backends based on configuration."""

from dataclasses import dataclass

from app.models.capture import StorageBackend
from app.services.storage.base import StorageBackendInterface
from app.services.storage.local import LocalStorageBackend
from app.services.storage.s3 import S3StorageBackend


@dataclass
class StorageConfig:
    """Configuration for storage backend."""

    backend: StorageBackend = StorageBackend.LOCAL

    # Local storage settings
    local_base_path: str = "./capture_data"

    # S3 settings
    s3_bucket_name: str | None = None
    s3_region_name: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_endpoint_url: str | None = None
    s3_prefix: str = ""

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create configuration from environment variables."""
        import os

        backend_str = os.getenv("STORAGE_BACKEND", "local").lower()
        backend = StorageBackend.S3 if backend_str == "s3" else StorageBackend.LOCAL

        return cls(
            backend=backend,
            local_base_path=os.getenv("STORAGE_LOCAL_PATH", "./capture_data"),
            s3_bucket_name=os.getenv("STORAGE_S3_BUCKET"),
            s3_region_name=os.getenv("STORAGE_S3_REGION", "us-east-1"),
            s3_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            s3_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            s3_endpoint_url=os.getenv("STORAGE_S3_ENDPOINT"),
            s3_prefix=os.getenv("STORAGE_S3_PREFIX", ""),
        )


def get_storage_backend(config: StorageConfig | None = None) -> StorageBackendInterface:
    """Create a storage backend based on configuration.

    Args:
        config: Storage configuration. If None, uses environment variables.

    Returns:
        Configured storage backend instance.
    """
    if config is None:
        config = StorageConfig.from_env()

    if config.backend == StorageBackend.S3:
        if not config.s3_bucket_name:
            raise ValueError("S3 bucket name is required for S3 backend")

        return S3StorageBackend(
            bucket_name=config.s3_bucket_name,
            region_name=config.s3_region_name,
            aws_access_key_id=config.s3_access_key_id,
            aws_secret_access_key=config.s3_secret_access_key,
            endpoint_url=config.s3_endpoint_url,
            prefix=config.s3_prefix,
        )
    else:
        return LocalStorageBackend(base_path=config.local_base_path)


# Global storage backend instance (lazily initialized)
_storage_backend: StorageBackendInterface | None = None


def get_default_storage() -> StorageBackendInterface:
    """Get the default storage backend (singleton).

    Uses environment variables for configuration.
    """
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = get_storage_backend()
    return _storage_backend
