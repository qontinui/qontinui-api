"""AWS S3 storage backend."""

import io
from typing import BinaryIO

import aioboto3
from botocore.exceptions import ClientError

from app.services.storage.base import StorageBackendInterface, StoredFile


class S3StorageBackend(StorageBackendInterface):
    """Storage backend using AWS S3.

    Suitable for:
    - Production deployments
    - Multi-server setups
    - Cloud-based qontinui-web
    """

    def __init__(
        self,
        bucket_name: str,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,  # For S3-compatible services
        prefix: str = "",  # Optional prefix for all keys
    ):
        """Initialize S3 storage backend.

        Args:
            bucket_name: S3 bucket name
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, uses env/IAM if not provided)
            aws_secret_access_key: AWS secret key
            endpoint_url: Custom endpoint URL for S3-compatible services
            prefix: Optional prefix for all keys (e.g., "production/")
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        # Configure boto3 session
        self.session_kwargs = {
            "region_name": region_name,
        }
        if aws_access_key_id and aws_secret_access_key:
            self.session_kwargs["aws_access_key_id"] = aws_access_key_id
            self.session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client_kwargs = {}
        if endpoint_url:
            self.client_kwargs["endpoint_url"] = endpoint_url

    def _s3_key(self, relative_path: str) -> str:
        """Get full S3 key from relative storage path."""
        return f"{self.prefix}{relative_path}"

    async def _get_client(self):
        """Get an S3 client from the session."""
        session = aioboto3.Session(**self.session_kwargs)
        return session.client("s3", **self.client_kwargs)

    async def store_video(
        self,
        session_id: str,
        video_data: BinaryIO,
        filename: str,
        content_type: str = "video/mp4",
    ) -> StoredFile:
        """Store a video file in S3."""
        relative_path = self._generate_video_path(session_id, filename)
        s3_key = self._s3_key(relative_path)

        # Read video data into memory for upload
        # For very large files, consider multipart upload
        video_bytes = video_data.read()

        async with await self._get_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=video_bytes,
                ContentType=content_type,
            )

            # Get the object info
            response = await client.head_object(Bucket=self.bucket_name, Key=s3_key)

        return StoredFile(
            path=relative_path,
            size_bytes=response["ContentLength"],
            content_type=content_type,
            etag=response.get("ETag", "").strip('"'),
        )

    async def store_frame(
        self,
        session_id: str,
        frame_number: int,
        frame_data: bytes,
        content_type: str = "image/jpeg",
    ) -> StoredFile:
        """Store an extracted video frame in S3."""
        extension = "jpg" if "jpeg" in content_type else "png"
        relative_path = self._generate_frame_path(session_id, frame_number, extension)
        s3_key = self._s3_key(relative_path)

        async with await self._get_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=frame_data,
                ContentType=content_type,
            )

        return StoredFile(
            path=relative_path, size_bytes=len(frame_data), content_type=content_type
        )

    async def get_video_path(self, session_id: str, filename: str) -> str:
        """Get the S3 key for a video file."""
        relative_path = self._generate_video_path(session_id, filename)
        return relative_path

    async def get_frame_path(self, session_id: str, frame_number: int) -> str | None:
        """Get the path to a cached frame if it exists in S3."""
        for ext in ["jpg", "png"]:
            relative_path = self._generate_frame_path(session_id, frame_number, ext)
            s3_key = self._s3_key(relative_path)

            async with await self._get_client() as client:
                try:
                    await client.head_object(Bucket=self.bucket_name, Key=s3_key)
                    return relative_path
                except ClientError:
                    continue

        return None

    async def get_video_stream(self, path: str) -> BinaryIO:
        """Get a stream for reading video data from S3."""
        s3_key = self._s3_key(path)

        async with await self._get_client() as client:
            response = await client.get_object(Bucket=self.bucket_name, Key=s3_key)
            body = await response["Body"].read()

        return io.BytesIO(body)

    async def get_frame_data(self, path: str) -> bytes:
        """Get frame image data from S3."""
        s3_key = self._s3_key(path)

        async with await self._get_client() as client:
            response = await client.get_object(Bucket=self.bucket_name, Key=s3_key)
            body_data: bytes = await response["Body"].read()
            return body_data

    async def delete_session_data(self, session_id: str) -> bool:
        """Delete all data for a capture session from S3."""
        prefix = self._s3_key(f"captures/{session_id}/")

        async with await self._get_client() as client:
            # List all objects with the session prefix
            paginator = client.get_paginator("list_objects_v2")
            objects_to_delete = []

            async for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix
            ):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})

            if not objects_to_delete:
                return False

            # Delete all objects
            await client.delete_objects(
                Bucket=self.bucket_name, Delete={"Objects": objects_to_delete}
            )

        return True

    async def get_video_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a pre-signed URL for accessing the video."""
        s3_key = self._s3_key(path)

        async with await self._get_client() as client:
            url: str = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expires_in,
            )

        return url

    async def get_frame_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a pre-signed URL for accessing a frame image."""
        s3_key = self._s3_key(path)

        async with await self._get_client() as client:
            url: str = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expires_in,
            )

        return url

    async def video_exists(self, path: str) -> bool:
        """Check if a video file exists in S3."""
        s3_key = self._s3_key(path)

        async with await self._get_client() as client:
            try:
                await client.head_object(Bucket=self.bucket_name, Key=s3_key)
                return True
            except ClientError:
                return False

    async def frame_exists(self, path: str) -> bool:
        """Check if a frame file exists in S3."""
        return await self.video_exists(path)  # Same logic

    async def list_sessions(self) -> list[str]:
        """List all capture session IDs in S3."""
        prefix = self._s3_key("captures/")
        sessions = set()

        async with await self._get_client() as client:
            paginator = client.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"
            ):
                if "CommonPrefixes" in page:
                    for common_prefix in page["CommonPrefixes"]:
                        # Extract session ID from prefix
                        session_id = (
                            common_prefix["Prefix"].replace(prefix, "").rstrip("/")
                        )
                        sessions.add(session_id)

        return list(sessions)
