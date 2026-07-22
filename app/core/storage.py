import os
import aiofiles
import logging
from typing import Optional
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception
from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self):
        self.use_s3 = False
        
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.bucket_name = os.environ.get('AWS_S3_BUCKET', 'kavach-uploads')

        if aws_access_key_id and aws_secret_access_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=os.environ.get('AWS_REGION', 'us-east-1')
                )
                self.use_s3 = True
                logger.info("StorageManager initialized with AWS S3 backend.")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}. Falling back to local storage.")
        else:
            logger.info("AWS credentials not found. StorageManager using Local storage fallback.")

    async def save_file(self, file_content: bytes, destination_path: str) -> str:
        """
        Save a file either to local disk or S3 depending on configuration.
        """
        if self.use_s3:
            try:
                # S3 keys shouldn't have leading slashes
                s3_key = destination_path.lstrip('/')
                
                # We need to run boto3 upload in a threadpool since it's synchronous
                import asyncio
                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file_content
                )
                return f"s3://{self.bucket_name}/{s3_key}"
            except Exception as e:
                logger.error(f"S3 upload failed for {destination_path}: {e}")
                # Fallback to local if S3 fails
                return await self._save_local(file_content, destination_path)
        else:
            return await self._save_local(file_content, destination_path)

    async def save_from_path(self, local_path: str, destination_path: str) -> str:
        """
        Uploads a local file to S3 if configured, otherwise returns the local path.
        """
        if self.use_s3:
            try:
                s3_key = destination_path.lstrip('/')
                import asyncio
                await asyncio.to_thread(
                    self.s3_client.upload_file,
                    local_path,
                    self.bucket_name,
                    s3_key
                )
                return f"s3://{self.bucket_name}/{s3_key}"
            except Exception as e:
                logger.error(f"S3 upload_file failed for {local_path}: {e}")
                return local_path
        else:
            return local_path

    async def _save_local(self, file_content: bytes, destination_path: str) -> str:
        # Ensure directories exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        async with aiofiles.open(destination_path, 'wb') as f:
            await f.write(file_content)
            
        return destination_path

    async def get_file_path(self, uri: str) -> str:
        """
        Returns a local filepath. If the file is on S3, it downloads it to a temp location first.
        """
        if uri.startswith("s3://"):
            parts = uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            local_path = os.path.join(settings.UPLOAD_DIR, "tmp_" + os.path.basename(key))
            if not os.path.exists(local_path):
                import asyncio
                try:
                    await asyncio.to_thread(
                        self.s3_client.download_file,
                        bucket,
                        key,
                        local_path
                    )
                except Exception as e:
                    logger.error(f"Failed to download {uri} from S3: {e}")
                    raise FileNotFoundError(f"File {uri} could not be retrieved from S3")
            return local_path
        
        return uri

storage_manager = StorageManager()
