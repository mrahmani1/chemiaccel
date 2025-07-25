"""
Cloud storage streaming capabilities for ChemiAccel.

Provides unified interface for streaming molecular data from S3, GCS, and Azure Blob Storage
with efficient chunked processing and automatic format detection.
"""

import gzip
import io
import logging
from abc import ABC, abstractmethod
from typing import Iterator
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


class CloudStorageProvider(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    def stream_file(self, uri: str, chunk_size: int = 8192) -> Iterator[bytes]:
        """Stream file content in chunks."""
        pass

    @abstractmethod
    def upload_file(self, local_path: str, uri: str) -> bool:
        """Upload file to cloud storage."""
        pass

    @abstractmethod
    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from cloud storage."""
        pass

    @abstractmethod
    def list_files(self, uri_prefix: str) -> Iterator[str]:
        """List files with given prefix."""
        pass


class S3StorageProvider(CloudStorageProvider):
    """AWS S3 storage provider."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize S3 client."""
        try:
            import boto3

            self.s3_client = boto3.client("s3", region_name=region)
            self.region = region
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 integration. Install with: pip install boto3"
            )

    def stream_file(self, uri: str, chunk_size: int = 8192) -> Iterator[bytes]:
        """Stream S3 file content in chunks."""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)

            # Stream the body content
            for chunk in response["Body"].iter_chunks(chunk_size=chunk_size):
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream S3 file {uri}: {e}")
            raise

    def upload_file(self, local_path: str, uri: str) -> bool:
        """Upload file to S3."""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded {local_path} to {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3 {uri}: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from S3."""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {uri} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3 {uri}: {e}")
            return False

    def list_files(self, uri_prefix: str) -> Iterator[str]:
        """List S3 files with given prefix."""
        parsed = urlparse(uri_prefix)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield f"s3://{bucket}/{obj['Key']}"

        except Exception as e:
            logger.error(f"Failed to list S3 files with prefix {uri_prefix}: {e}")
            raise


class GCSStorageProvider(CloudStorageProvider):
    """Google Cloud Storage provider."""

    def __init__(self, project_id: str = None):
        """Initialize GCS client."""
        try:
            from google.cloud import storage

            self.client = storage.Client(project=project_id)
            self.project_id = project_id
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS integration. "
                "Install with: pip install google-cloud-storage"
            )

    def stream_file(self, uri: str, chunk_size: int = 8192) -> Iterator[bytes]:
        """Stream GCS file content in chunks."""
        parsed = urlparse(uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download in chunks
            with blob.open("rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        except Exception as e:
            logger.error(f"Failed to stream GCS file {uri}: {e}")
            raise

    def upload_file(self, local_path: str, uri: str) -> bool:
        """Upload file to GCS."""
        parsed = urlparse(uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)

            logger.info(f"Uploaded {local_path} to {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCS {uri}: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from GCS."""
        parsed = urlparse(uri)
        bucket_name = parsed.netloc
        blob_name = parsed.path.lstrip("/")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            logger.info(f"Downloaded {uri} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from GCS {uri}: {e}")
            return False

    def list_files(self, uri_prefix: str) -> Iterator[str]:
        """List GCS files with given prefix."""
        parsed = urlparse(uri_prefix)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")

        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                yield f"gs://{bucket_name}/{blob.name}"

        except Exception as e:
            logger.error(f"Failed to list GCS files with prefix {uri_prefix}: {e}")
            raise


class AzureBlobStorageProvider(CloudStorageProvider):
    """Azure Blob Storage provider."""

    def __init__(self, account_name: str, account_key: str):
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient

            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url, credential=account_key
            )
            self.account_name = account_name
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure Blob Storage integration. "
                "Install with: pip install azure-storage-blob"
            )

    def stream_file(self, uri: str, chunk_size: int = 8192) -> Iterator[bytes]:
        """Stream Azure Blob file content in chunks."""
        # Parse Azure blob URI: https://account.blob.core.windows.net/container/blob
        parsed = urlparse(uri)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            # Stream download
            stream = blob_client.download_blob()
            for chunk in stream.chunks():
                yield chunk

        except Exception as e:
            logger.error(f"Failed to stream Azure Blob file {uri}: {e}")
            raise

    def upload_file(self, local_path: str, uri: str) -> bool:
        """Upload file to Azure Blob Storage."""
        parsed = urlparse(uri)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logger.info(f"Uploaded {local_path} to {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Azure Blob {uri}: {e}")
            return False

    def download_file(self, uri: str, local_path: str) -> bool:
        """Download file from Azure Blob Storage."""
        parsed = urlparse(uri)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            logger.info(f"Downloaded {uri} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from Azure Blob {uri}: {e}")
            return False

    def list_files(self, uri_prefix: str) -> Iterator[str]:
        """List Azure Blob files with given prefix."""
        parsed = urlparse(uri_prefix)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container_name = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ""

        try:
            container_client = self.blob_service_client.get_container_client(
                container_name
            )
            blobs = container_client.list_blobs(name_starts_with=prefix)

            for blob in blobs:
                yield f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob.name}"

        except Exception as e:
            logger.error(
                f"Failed to list Azure Blob files with prefix {uri_prefix}: {e}"
            )
            raise


class CloudStorageStreamer:
    """Unified cloud storage streaming interface."""

    def __init__(self, **provider_configs):
        """
        Initialize cloud storage streamer.

        Args:
            **provider_configs: Provider-specific configurations
                aws_region: AWS region
                gcp_project_id: GCP project ID
                azure_account_name: Azure storage account name
                azure_account_key: Azure storage account key
        """
        self.providers = {}
        self.provider_configs = provider_configs

    def _get_provider(self, uri: str) -> CloudStorageProvider:
        """Get appropriate storage provider for URI."""
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()

        if scheme == "s3":
            if "s3" not in self.providers:
                self.providers["s3"] = S3StorageProvider(
                    region=self.provider_configs.get("aws_region", "us-east-1")
                )
            return self.providers["s3"]

        elif scheme == "gs":
            if "gcs" not in self.providers:
                self.providers["gcs"] = GCSStorageProvider(
                    project_id=self.provider_configs.get("gcp_project_id")
                )
            return self.providers["gcs"]

        elif scheme == "https" and "blob.core.windows.net" in parsed.netloc:
            if "azure" not in self.providers:
                self.providers["azure"] = AzureBlobStorageProvider(
                    account_name=self.provider_configs["azure_account_name"],
                    account_key=self.provider_configs["azure_account_key"],
                )
            return self.providers["azure"]

        else:
            raise ValueError(f"Unsupported storage URI scheme: {scheme}")

    def stream_dataframe(
        self, uri: str, chunk_size: int = 10000, file_format: str = "auto"
    ) -> Iterator[pd.DataFrame]:
        """
        Stream molecular data as DataFrame chunks.

        Args:
            uri: Cloud storage URI
            chunk_size: Number of rows per chunk
            file_format: File format ('csv', 'parquet', 'smi', 'auto')

        Yields:
            DataFrame chunks containing molecular data
        """
        provider = self._get_provider(uri)

        # Detect file format if auto
        if file_format == "auto":
            file_format = self._detect_format(uri)

        # Stream file content
        content_buffer = io.BytesIO()

        # Handle compressed files
        is_compressed = uri.endswith(".gz")

        for chunk in provider.stream_file(uri):
            content_buffer.write(chunk)

        content_buffer.seek(0)

        # Decompress if needed
        if is_compressed:
            content_buffer = io.BytesIO(gzip.decompress(content_buffer.read()))

        # Parse based on format
        if file_format == "csv":
            yield from self._stream_csv(content_buffer, chunk_size)
        elif file_format == "parquet":
            yield from self._stream_parquet(content_buffer, chunk_size)
        elif file_format == "smi":
            yield from self._stream_smiles(content_buffer, chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def upload_dataframe(
        self, df: pd.DataFrame, uri: str, file_format: str = "parquet"
    ) -> bool:
        """
        Upload DataFrame to cloud storage.

        Args:
            df: DataFrame to upload
            uri: Destination URI
            file_format: Output format ('csv', 'parquet')

        Returns:
            Success status
        """
        provider = self._get_provider(uri)

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_format}"
        ) as tmp_file:
            if file_format == "csv":
                df.to_csv(tmp_file.name, index=False)
            elif file_format == "parquet":
                df.to_parquet(tmp_file.name, index=False)
            else:
                raise ValueError(f"Unsupported output format: {file_format}")

            # Upload to cloud storage
            success = provider.upload_file(tmp_file.name, uri)

            # Clean up
            import os

            os.unlink(tmp_file.name)

            return success

    def list_files(self, uri_prefix: str) -> Iterator[str]:
        """List files with given prefix."""
        provider = self._get_provider(uri_prefix)
        return provider.list_files(uri_prefix)

    def _detect_format(self, uri: str) -> str:
        """Detect file format from URI."""
        uri_lower = uri.lower()

        if uri_lower.endswith(".csv") or uri_lower.endswith(".csv.gz"):
            return "csv"
        elif uri_lower.endswith(".parquet") or uri_lower.endswith(".parquet.gz"):
            return "parquet"
        elif (
            uri_lower.endswith(".smi")
            or uri_lower.endswith(".smiles")
            or uri_lower.endswith(".smi.gz")
        ):
            return "smi"
        else:
            # Default to CSV
            return "csv"

    def _stream_csv(
        self, content_buffer: io.BytesIO, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Stream CSV content as DataFrame chunks."""
        content_buffer.seek(0)

        # Read in chunks
        for chunk in pd.read_csv(content_buffer, chunksize=chunk_size):
            yield chunk

    def _stream_parquet(
        self, content_buffer: io.BytesIO, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Stream Parquet content as DataFrame chunks."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        content_buffer.seek(0)

        # Read parquet file
        table = pq.read_table(pa.BufferReader(content_buffer.read()))
        df = table.to_pandas()

        # Yield in chunks
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size]

    def _stream_smiles(
        self, content_buffer: io.BytesIO, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Stream SMILES file content as DataFrame chunks."""
        content_buffer.seek(0)

        smiles_list = []
        for line in content_buffer:
            line = line.decode("utf-8").strip()
            if line and not line.startswith("#"):
                # Handle different SMILES formats
                parts = line.split("\t") if "\t" in line else line.split()
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{len(smiles_list)}"

                smiles_list.append({"smiles": smiles, "id": mol_id})

                # Yield chunk when ready
                if len(smiles_list) >= chunk_size:
                    yield pd.DataFrame(smiles_list)
                    smiles_list = []

        # Yield remaining molecules
        if smiles_list:
            yield pd.DataFrame(smiles_list)


# Utility functions for common operations
def stream_molecules_from_cloud(
    uri: str, chunk_size: int = 10000, **provider_configs
) -> Iterator[pd.DataFrame]:
    """
    Convenience function to stream molecular data from cloud storage.

    Args:
        uri: Cloud storage URI
        chunk_size: Number of molecules per chunk
        **provider_configs: Provider-specific configurations

    Yields:
        DataFrame chunks with molecular data
    """
    streamer = CloudStorageStreamer(**provider_configs)
    return streamer.stream_dataframe(uri, chunk_size)


def upload_results_to_cloud(df: pd.DataFrame, uri: str, **provider_configs) -> bool:
    """
    Convenience function to upload results to cloud storage.

    Args:
        df: Results DataFrame
        uri: Destination URI
        **provider_configs: Provider-specific configurations

    Returns:
        Success status
    """
    streamer = CloudStorageStreamer(**provider_configs)
    return streamer.upload_dataframe(df, uri)
