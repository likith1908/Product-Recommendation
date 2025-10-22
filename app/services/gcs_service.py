"""
Google Cloud Storage service for handling file uploads.
"""

import uuid
from pathlib import Path
from typing import Optional
from datetime import timedelta

from google.cloud import storage
from fastapi import UploadFile, HTTPException

from app.core.config import settings


class GCSService:
    """Service for managing file uploads to Google Cloud Storage"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(GCSService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize GCS client"""
        if not self._initialized:
            print("🔧 Initializing Google Cloud Storage client...")
            self.client = storage.Client(project=settings.GCS_PROJECT_NAME)
            self.bucket = self.client.bucket(settings.GCS_BUCKET_NAME)
            self._initialized = True
            print(f"✅ GCS initialized - Bucket: {settings.GCS_BUCKET_NAME}")
    
    def _validate_image_file(self, file: UploadFile) -> None:
        """Validate uploaded image file"""
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_EXTENSIONS)}"
            )
        
        # Check content type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
    
    def upload_user_image(
        self, 
        file: UploadFile, 
        user_id: Optional[str] = None
    ) -> str:
        """
        Upload user image to GCS and return public URL.
        
        Args:
            file: FastAPI UploadFile object
            user_id: Optional user ID for organizing files
        
        Returns:
            Public URL of uploaded file
        """
        # Validate file
        self._validate_image_file(file)
        
        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        
        # Organize by user if provided
        if user_id:
            blob_path = f"{settings.GCS_UPLOAD_FOLDER}/{user_id}/{unique_id}{file_ext}"
        else:
            blob_path = f"{settings.GCS_UPLOAD_FOLDER}/{unique_id}{file_ext}"
        
        try:
            # Create blob
            blob = self.bucket.blob(blob_path)
            
            # Set content type and cache control
            blob.content_type = file.content_type
            blob.cache_control = "public, max-age=3600"
            
            # Upload file
            file.file.seek(0)  # Reset file pointer
            blob.upload_from_file(file.file, content_type=file.content_type)
            
            # For uniform bucket-level access, the public URL works automatically
            # if the bucket has "allUsers" with "Storage Object Viewer" role
            public_url = f"https://storage.googleapis.com/{settings.GCS_BUCKET_NAME}/{blob_path}"
            
            print(f"✅ Uploaded: {public_url}")
            
            return public_url
        
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file: {str(e)}"
            )
    
    def delete_file(self, file_url: str) -> bool:
        """
        Delete a file from GCS given its public URL.
        
        Args:
            file_url: Public URL of the file
        
        Returns:
            True if deleted successfully
        """
        try:
            # Extract blob path from URL
            # URL format: https://storage.googleapis.com/bucket-name/path/to/file
            url_parts = file_url.split(f"{settings.GCS_BUCKET_NAME}/")
            if len(url_parts) != 2:
                return False
            
            blob_path = url_parts[1]
            blob = self.bucket.blob(blob_path)
            
            if blob.exists():
                blob.delete()
                print(f"🗑️  Deleted: {file_url}")
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ Delete failed: {str(e)}")
            return False
    
    def list_user_files(self, user_id: str, max_files: int = 100) -> list[str]:
        """
        List all files uploaded by a specific user.
        
        Args:
            user_id: User ID
            max_files: Maximum number of files to return
        
        Returns:
            List of public URLs
        """
        try:
            prefix = f"{settings.GCS_UPLOAD_FOLDER}/{user_id}/"
            blobs = self.client.list_blobs(
                settings.GCS_BUCKET_NAME, 
                prefix=prefix,
                max_results=max_files
            )
            
            urls = []
            for blob in blobs:
                public_url = f"https://storage.googleapis.com/{settings.GCS_BUCKET_NAME}/{blob.name}"
                urls.append(public_url)
            
            return urls
        
        except Exception as e:
            print(f"❌ List failed: {str(e)}")
            return []
    
    def check_bucket_access(self) -> dict:
        """
        Check bucket configuration and access settings.
        Useful for debugging.
        
        Returns:
            Dictionary with bucket information
        """
        try:
            bucket = self.client.get_bucket(settings.GCS_BUCKET_NAME)
            
            return {
                "bucket_name": bucket.name,
                "location": bucket.location,
                "storage_class": bucket.storage_class,
                "uniform_bucket_level_access": bucket.iam_configuration.uniform_bucket_level_access_enabled,
                "public_access_prevention": bucket.iam_configuration.public_access_prevention,
                "created": bucket.time_created.isoformat() if bucket.time_created else None
            }
        except Exception as e:
            return {
                "error": str(e)
            }


# Singleton instance
_gcs_service_instance = None

def get_gcs_service() -> GCSService:
    """Get or create the singleton GCS service instance"""
    global _gcs_service_instance
    if _gcs_service_instance is None:
        _gcs_service_instance = GCSService()
    return _gcs_service_instance