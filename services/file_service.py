import shutil
from pathlib import Path

from fastapi import UploadFile

from conf import settings


class FileService:
    """Handles saving files to disk and loading their content."""

    @staticmethod
    def upload_file(file: UploadFile) -> Path:
        """Save an uploaded file to disk."""
        if file.filename == None:
            raise ValueError("Uploaded file has no name")

        file_path = settings.UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path

    @staticmethod
    def load_file_content(file_path: Path, encoding: str = "utf-8") -> str:
        """Load file content into memory."""
        return file_path.read_text(encoding=encoding)
